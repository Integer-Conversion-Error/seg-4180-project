#!/usr/bin/env python3
"""
Pre-label bounding boxes using pretrained YOLO.
For synthetic images, uses image differencing to identify inserted vehicle.
"""

import argparse
import csv
import importlib
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


CLASS_NAME_TO_ID = {
    "vehicle": 0,
    "enforcement_vehicle": 1,
    "police_old": 2,
    "police_new": 3,
    "lookalike_negative": 4,
}


def detect_vehicles(model, image_path: Path, conf: float = 0.25) -> list:
    results = model(image_path, conf=conf, verbose=False)
    vehicle_classes = [2, 3, 5, 7]

    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": conf_score,
                        "coco_class": cls_id,
                    }
                )
    return boxes


def compute_change_region(
    original_path: Path, edited_path: Path
) -> tuple[int, int, int, int] | None:
    try:
        orig = cv2.imread(str(original_path))
        edit = cv2.imread(str(edited_path))

        if orig is None or edit is None:
            return None

        orig = cv2.resize(orig, (edit.shape[1], edit.shape[0]))

        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_edit = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_orig, gray_edit)

        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        return (x, y, x + w, y + h)
    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        return None


def analyze_diff_mask(
    original_path: Path,
    edited_path: Path,
    threshold: int = 30,
    area_ratio_threshold: float = 0.15,
) -> dict:
    """
    Analyze the diff mask between original and edited images for quality control.

    Args:
        original_path: Path to original image
        edited_path: Path to edited/synthetic image
        threshold: Pixel difference threshold (0-255)
        area_ratio_threshold: Max allowed ratio of changed area to image area

    Returns:
        dict with keys:
            - valid: bool, passed all QA checks
            - mask_area_ratio: float, ratio of changed pixels to total
            - touches_border: bool, if change region touches image border
            - num_large_components: int, number of connected components > 100px
            - bbox: tuple (x1, y1, x2, y2) of change region, or None
            - failure_reason: str, reason for failure (if any)
    """
    result = {
        "valid": True,
        "mask_area_ratio": 0.0,
        "touches_border": False,
        "num_large_components": 0,
        "bbox": None,
        "failure_reason": "",
    }

    try:
        orig = cv2.imread(str(original_path))
        edit = cv2.imread(str(edited_path))

        if orig is None or edit is None:
            result["valid"] = False
            result["failure_reason"] = "image_load_failed"
            return result

        # Ensure same dimensions
        if orig.shape != edit.shape:
            orig = cv2.resize(orig, (edit.shape[1], edit.shape[0]))

        # Compute absolute difference
        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_edit = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_orig, gray_edit)

        # Threshold diff
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Morphological closing to connect nearby changes
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Calculate area ratio
        mask_pixels = np.count_nonzero(closed)
        total_pixels = closed.shape[0] * closed.shape[1]
        result["mask_area_ratio"] = (
            mask_pixels / total_pixels if total_pixels > 0 else 0.0
        )

        # Check if touches border (dilate mask and check edges)
        border_check = closed.copy()
        border_kernel = np.ones((3, 3), np.uint8)
        border_check = cv2.dilate(border_check, border_kernel, iterations=5)
        h, w = border_check.shape
        touches_top = np.any(border_check[0, :] > 0)
        touches_bottom = np.any(border_check[-1, :] > 0)
        touches_left = np.any(border_check[:, 0] > 0)
        touches_right = np.any(border_check[:, -1] > 0)
        result["touches_border"] = (
            touches_top or touches_bottom or touches_left or touches_right
        )

        # Find connected components
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            closed, connectivity=8
        )
        large_components = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area > 100:
                large_components += 1
        result["num_large_components"] = large_components

        # Find bounding box of change region
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(largest)
            result["bbox"] = (x, y, x + width, y + height)

        # QA checks
        if result["mask_area_ratio"] > area_ratio_threshold:
            result["valid"] = False
            result["failure_reason"] = "edit_too_large"
        elif result["touches_border"]:
            result["valid"] = False
            result["failure_reason"] = "vehicle_cut_off"
        elif result["num_large_components"] > 1:
            result["valid"] = False
            result["failure_reason"] = "multiple_edits"

        return result

    except Exception as e:
        logger.error(f"QA analysis failed: {e}")
        result["valid"] = False
        result["failure_reason"] = f"exception: {str(e)[:50]}"
        return result


def box_iou(box1: tuple, box2: tuple) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def validate_yolo_label(
    x_center: float, y_center: float, width: float, height: float
) -> tuple:
    """
    Validate YOLO format label bounds.

    Args:
        x_center: Normalized x center (should be in [0, 1])
        y_center: Normalized y center (should be in [0, 1])
        width: Normalized width (should be in [0, 1])
        height: Normalized height (should be in [0, 1])

    Returns:
        (valid: bool, error: str or None)
    """
    errors = []

    # Check value ranges
    if not (0 <= x_center <= 1):
        errors.append(f"x_center {x_center:.4f} out of [0, 1]")
    if not (0 <= y_center <= 1):
        errors.append(f"y_center {y_center:.4f} out of [0, 1]")
    if not (0 <= width <= 1):
        errors.append(f"width {width:.4f} out of [0, 1]")
    if not (0 <= height <= 1):
        errors.append(f"height {height:.4f} out of [0, 1]")

    # Check dimensions are positive
    if width <= 0:
        errors.append(f"width {width:.4f} must be > 0")
    if height <= 0:
        errors.append(f"height {height:.4f} must be > 0")

    if errors:
        return False, "; ".join(errors)
    return True, None


def convert_to_yolo(boxes: list, img_width: int, img_height: int) -> list:
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height

        yolo_boxes.append(
            {
                "class": box["class"],
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
                "conf": box.get("conf", 1.0),
            }
        )
    return yolo_boxes


def write_yolo_labels(labels: list, output_path: Path, validate: bool = True) -> tuple:
    """
    Write YOLO format labels, optionally validating each before writing.

    Args:
        labels: List of label dicts with class, x_center, y_center, width, height
        output_path: Path to write labels to
        validate: Whether to validate labels before writing

    Returns:
        (num_written: int, num_skipped: int, errors: dict)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    num_skipped = 0
    errors = {}

    with open(output_path, "w") as f:
        for idx, label in enumerate(labels):
            if validate:
                valid, error = validate_yolo_label(
                    label["x_center"],
                    label["y_center"],
                    label["width"],
                    label["height"],
                )
                if not valid:
                    num_skipped += 1
                    errors[idx] = error
                    logger.warning(f"Skipping invalid label {idx}: {error}")
                    continue

            f.write(
                f"{label['class']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n"
            )
            num_written += 1

    return num_written, num_skipped, errors


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in manifest:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def resolve_target_class_id(row: dict) -> int | None:
    expected = (row.get("expected_inserted_class") or "").strip()
    if expected in CLASS_NAME_TO_ID:
        return CLASS_NAME_TO_ID[expected]

    edit_type = (row.get("edit_type") or "").strip()
    if edit_type in CLASS_NAME_TO_ID:
        return CLASS_NAME_TO_ID[edit_type]

    file_path = Path(row.get("file_path", ""))
    parent_name = file_path.parent.name
    if parent_name in CLASS_NAME_TO_ID:
        return CLASS_NAME_TO_ID[parent_name]

    return None


def is_rejected(row: dict) -> bool:
    return (row.get("review_status") or "").strip().lower() == "rejected"


def main():
    parser = argparse.ArgumentParser(description="Pre-label bounding boxes")
    parser.add_argument(
        "--manifest", "-m", default="manifests/images.csv", help="Input manifest"
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/labels_autogen", help="Output labels dir"
    )
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--qa-threshold-area",
        type=float,
        default=0.15,
        help="QA threshold for mask area ratio",
    )
    parser.add_argument("--qa-skip", action="store_true", help="Skip QA checks")
    parser.add_argument(
        "--validate-labels",
        action="store_true",
        default=True,
        help="Validate YOLO label bounds (default: True)",
    )
    parser.add_argument(
        "--skip-label-validation",
        action="store_true",
        help="Disable YOLO label validation",
    )
    parser.add_argument(
        "--histogram-out",
        default="reports/prelabel_vehicle_count_hist.png",
        help="Output PNG path for prelabel vehicle-count histogram",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include images with review_status=rejected (default: exclude)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    lists_dir = Path("lists")
    lists_dir.mkdir(exist_ok=True)
    bad_synthetics_path = lists_dir / "bad_synthetics.txt"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    manifest = load_manifest(manifest_path)

    logger.info(f"Loading YOLO model: {args.model}")
    ultralytics_module = importlib.import_module("ultralytics")
    model = ultralytics_module.YOLO(args.model)

    bad_synthetic_ids = []
    histogram_counts = []
    validation_enabled = not args.skip_label_validation
    total_boxes_validated = 0
    invalid_boxes_found = 0
    error_counts = {}

    for row in tqdm(manifest, desc="Pre-labeling"):
        image_id = row.get("image_id", "")
        file_path = Path(row.get("file_path", ""))

        if not file_path.exists() or row.get("status") != "ok":
            continue
        if not args.include_rejected and is_rejected(row):
            continue

        is_synthetic = row.get("is_synthetic") == "1"
        expected_class = row.get("expected_inserted_class", "none")
        target_class_id = resolve_target_class_id(row)

        boxes = detect_vehicles(model, file_path, args.conf)

        # Run QA checks for synthetic images
        qa_passed = True
        qa_failure_reason = ""
        if is_synthetic and target_class_id is not None and not args.qa_skip:
            parent_id = row.get("parent_image_id", "")
            parent_row = next(
                (r for r in manifest if r.get("image_id") == parent_id), None
            )

            if parent_row:
                parent_path = Path(parent_row.get("file_path", ""))
                if parent_path.exists():
                    qa_result = analyze_diff_mask(
                        parent_path,
                        file_path,
                        area_ratio_threshold=args.qa_threshold_area,
                    )
                    qa_passed = qa_result["valid"]
                    qa_failure_reason = qa_result["failure_reason"]

                    if not qa_passed:
                        bad_synthetic_ids.append(image_id)
                        row["needs_review"] = "1"
                else:
                    qa_passed = False
                    qa_failure_reason = "parent_not_found"
            else:
                qa_passed = False
                qa_failure_reason = "parent_not_found"

        # Add QA columns to manifest
        row["qa_passed"] = "1" if qa_passed else "0"
        if qa_failure_reason:
            row["qa_failure_reason"] = qa_failure_reason

        for box in boxes:
            box["class"] = 0

        if is_synthetic and target_class_id is not None:
            parent_id = row.get("parent_image_id", "")
            parent_row = next(
                (r for r in manifest if r.get("image_id") == parent_id), None
            )

            assigned_inserted_class = False

            if parent_row:
                parent_path = Path(parent_row.get("file_path", ""))
                if parent_path.exists():
                    change_bbox = compute_change_region(parent_path, file_path)

                    if change_bbox:
                        best_iou = 0
                        best_box_idx = -1

                        for idx, box in enumerate(boxes):
                            iou = box_iou(
                                (box["x1"], box["y1"], box["x2"], box["y2"]),
                                change_bbox,
                            )
                            if iou > best_iou:
                                best_iou = iou
                                best_box_idx = idx

                        if (
                            best_iou >= 0.1
                            and best_box_idx >= 0
                            and target_class_id is not None
                        ):
                            boxes[best_box_idx]["class"] = target_class_id
                            assigned_inserted_class = True
                        else:
                            row["needs_review"] = "1"
                            if target_class_id is not None and change_bbox is not None:
                                x1, y1, x2, y2 = change_bbox
                                boxes.append(
                                    {
                                        "x1": float(x1),
                                        "y1": float(y1),
                                        "x2": float(x2),
                                        "y2": float(y2),
                                        "class": target_class_id,
                                        "conf": 1.0,
                                    }
                                )
                                assigned_inserted_class = True
                    else:
                        row["needs_review"] = "1"
                else:
                    row["needs_review"] = "1"
            else:
                row["needs_review"] = "1"

            if not assigned_inserted_class and target_class_id is not None:
                if len(boxes) == 1:
                    boxes[0]["class"] = target_class_id
                    assigned_inserted_class = True
                elif len(boxes) > 1:
                    best_idx = max(
                        range(len(boxes)), key=lambda i: boxes[i].get("conf", 0.0)
                    )
                    boxes[best_idx]["class"] = target_class_id
                    row["needs_review"] = "1"

        img = cv2.imread(str(file_path))
        if img is not None:
            h, w = img.shape[:2]
            yolo_labels = convert_to_yolo(boxes, w, h)

            output_path = out_dir / f"{image_id}.txt"
            num_written, num_skipped, label_errors = write_yolo_labels(
                yolo_labels, output_path, validate=validation_enabled
            )

            # Track validation statistics
            total_boxes_validated += len(yolo_labels)
            invalid_boxes_found += num_skipped
            for error in label_errors.values():
                error_counts[error] = error_counts.get(error, 0) + 1

            # Update manifest with validation info
            row["num_boxes_autogen"] = str(len(boxes))
            row["label_valid"] = "1" if num_skipped == 0 else "0"
            if label_errors:
                # Store first error as sample
                first_error = next(iter(label_errors.values()))
                row["label_error"] = first_error
                if num_skipped > 0:
                    row["needs_review"] = "1"
            else:
                row["label_error"] = ""

            route_category = (row.get("route_category") or "").strip()
            excluded_reason = (row.get("excluded_reason") or "").strip()
            if (
                len(boxes) > 0
                and route_category
                not in {"non_road", "non_street", "existing_vehicle_only"}
                and not excluded_reason
            ):
                histogram_counts.append(len(boxes))

    save_manifest(manifest, manifest_path)
    logger.info(f"Labels saved to {out_dir}")

    histogram_out = Path(args.histogram_out)
    histogram_out.parent.mkdir(parents=True, exist_ok=True)
    if histogram_counts:
        plt.figure(figsize=(10, 5))
        bins = range(1, max(histogram_counts) + 2)
        plt.hist(histogram_counts, bins=bins, edgecolor="black", alpha=0.8)
        plt.title("Prelabel Vehicle Box Count Distribution")
        plt.xlabel("Vehicle boxes per image")
        plt.ylabel("Image count")
        plt.tight_layout()
        plt.savefig(histogram_out, dpi=150)
        plt.close()
        logger.info("Saved histogram PNG: %s", histogram_out)
    else:
        logger.warning(
            "Histogram skipped: no eligible images after exclusions (0 boxes/non-road/non-street/excluded)."
        )

    # Write bad synthetics list
    with open(bad_synthetics_path, "w") as f:
        for img_id in bad_synthetic_ids:
            f.write(f"{img_id}\n")

    logger.info(f"QA failed synthetics: {len(bad_synthetic_ids)}")
    if bad_synthetic_ids:
        logger.info(f"Bad synthetics list: {bad_synthetics_path}")

    # Log validation summary
    if validation_enabled:
        logger.info("=" * 60)
        logger.info("YOLO LABEL VALIDATION SUMMARY")
        logger.info(f"Total boxes validated: {total_boxes_validated}")
        logger.info(f"Invalid boxes found: {invalid_boxes_found}")
        if invalid_boxes_found > 0:
            logger.info("Error types:")
            for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  - {error_type}: {count}")
        else:
            logger.info("All labels passed validation!")
        logger.info("=" * 60)
    else:
        logger.info("Label validation disabled (--skip-label-validation)")


if __name__ == "__main__":
    main()
