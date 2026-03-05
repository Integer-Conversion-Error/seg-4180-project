#!/usr/bin/env python3
"""
Visualize bounding box sizes and properties.
Analyzes box area, aspect ratio, and position for all boxes in dataset.
Identifies and visualizes outliers (top-N biggest/smallest boxes).
"""

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Class mapping
CLASS_NAMES = {0: "vehicle", 1: "enforcement_vehicle", 2: "police_old", 3: "police_new", 4: "lookalike_negative"}

CLASS_COLORS = {
    0: (0, 255, 0),      # vehicle: green
    1: (0, 0, 255),      # enforcement_vehicle: red
    2: (255, 0, 0),      # police_old: blue
    3: (255, 255, 0),    # police_new: cyan
    4: (128, 0, 128),    # lookalike_negative: purple
}







def load_manifest(manifest_path: Path) -> list:
    """Load image manifest CSV."""
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def parse_yolo_label(line: str) -> tuple:
    """
    Parse YOLO label line.
    Format: class_id x_center y_center width height [conf]
    Returns: (class_id, x_center, y_center, width, height)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    return (class_id, x_center, y_center, width, height)


def denormalize_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple:
    """
    Denormalize YOLO coordinates to pixel coordinates.
    Returns: (x1, y1, x2, y2) in pixels
    """
    x1 = max(0, int((x_center - width / 2) * img_width))
    y1 = max(0, int((y_center - height / 2) * img_height))
    x2 = min(img_width, int((x_center + width / 2) * img_width))
    y2 = min(img_height, int((y_center + height / 2) * img_height))
    return (x1, y1, x2, y2)


def compute_box_metrics(x1: int, y1: int, x2: int, y2: int) -> dict:
    """
    Compute area, aspect ratio, and position metrics for a box.
    Returns: dict with area, aspect_ratio, width, height, center_x, center_y
    """
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = width / height if height > 0 else 0
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return {
        "area": area,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "center_x": center_x,
        "center_y": center_y,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def load_image_with_boxes(image_path: Path, label_path: Path) -> tuple:
    """
    Load image and parse its label file.
    Returns: (image, boxes) where boxes is list of dicts with box and class info
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []

    boxes = []
    if label_path.exists():
        img_height, img_width = img.shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                parsed = parse_yolo_label(line)
                if parsed:
                    class_id, x_center, y_center, width, height = parsed
                    x1, y1, x2, y2 = denormalize_box(
                        x_center, y_center, width, height, img_width, img_height
                    )
                    metrics = compute_box_metrics(x1, y1, x2, y2)
                    metrics["class_id"] = class_id
                    metrics["image_id"] = image_path.stem
                    metrics["image_path"] = str(image_path)
                    metrics["image_width"] = img_width
                    metrics["image_height"] = img_height
                    boxes.append(metrics)

    return img, boxes


def analyze_dataset(manifest: list, labels_dir: Path) -> tuple:
    """
    Analyze all boxes in dataset.
    Returns: (all_boxes, stats_by_class)
    """
    all_boxes = []
    stats_by_class = defaultdict(
        lambda: {"areas": [], "aspect_ratios": [], "boxes": []}
    )

    logger.info("Analyzing bounding boxes...")
    for row in tqdm(manifest):
        image_id = row.get("image_id", "")
        file_path = Path(row.get("file_path", ""))

        if not file_path.exists():
            continue

        # Try both label directories
        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.exists() and labels_dir.name == "labels_final":
            # Fallback to autogen
            alt_dir = labels_dir.parent / "labels_autogen"
            label_path = alt_dir / f"{image_id}.txt"

        if not label_path.exists():
            continue

        img, boxes = load_image_with_boxes(file_path, label_path)
        if img is None or not boxes:
            continue

        for box in boxes:
            class_id = box["class_id"]
            all_boxes.append(box)
            stats_by_class[class_id]["areas"].append(box["area"])
            stats_by_class[class_id]["aspect_ratios"].append(box["aspect_ratio"])
            stats_by_class[class_id]["boxes"].append(box)

    return all_boxes, stats_by_class


def find_outliers(all_boxes: list, top_n: int = 10) -> tuple:
    """
    Find top-N biggest and smallest boxes by area.
    Returns: (biggest_boxes, smallest_boxes)
    """
    sorted_by_area = sorted(all_boxes, key=lambda b: b["area"], reverse=True)
    biggest = sorted_by_area[:top_n]
    smallest = sorted_by_area[-top_n:][::-1]  # Reverse to ascending

    return biggest, smallest


def create_visualizations(all_boxes: list, stats_by_class: dict, output_dir: Path):
    """Create matplotlib visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histogram of box areas (log scale)
    fig, ax = plt.subplots(figsize=(12, 6))
    for class_id, stats in sorted(stats_by_class.items()):
        areas = stats["areas"]
        if areas:
            ax.hist(
                areas,
                bins=50,
                alpha=0.6,
                label=CLASS_NAMES.get(class_id, f"Class {class_id}"),
            )
    ax.set_xlabel("Box Area (pixels²)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Bounding Box Areas")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_areas.png", dpi=150)
    plt.close()
    logger.info(f"Saved: histogram_areas.png")

    # 2. Histogram of aspect ratios
    fig, ax = plt.subplots(figsize=(12, 6))
    for class_id, stats in sorted(stats_by_class.items()):
        aspect_ratios = [ar for ar in stats["aspect_ratios"] if ar > 0]
        if aspect_ratios:
            ax.hist(
                aspect_ratios,
                bins=50,
                alpha=0.6,
                label=CLASS_NAMES.get(class_id, f"Class {class_id}"),
            )
    ax.set_xlabel("Aspect Ratio (width/height)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Box Aspect Ratios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_aspect_ratios.png", dpi=150)
    plt.close()
    logger.info(f"Saved: histogram_aspect_ratios.png")

    # 3. Scatter plot: area vs aspect ratio, colored by class
    fig, ax = plt.subplots(figsize=(12, 8))
    for class_id, stats in sorted(stats_by_class.items()):
        boxes = stats["boxes"]
        if boxes:
            areas = [b["area"] for b in boxes]
            aspect_ratios = [b["aspect_ratio"] for b in boxes]
            color = CLASS_COLORS.get(class_id, (128, 128, 128))
            # Normalize BGR to RGB for matplotlib
            color = tuple(c / 255 for c in color[::-1])
            ax.scatter(
                aspect_ratios,
                areas,
                alpha=0.6,
                s=30,
                label=CLASS_NAMES.get(class_id, f"Class {class_id}"),
                color=color,
            )
    ax.set_xlabel("Aspect Ratio (width/height)")
    ax.set_ylabel("Box Area (pixels²)")
    ax.set_title("Box Area vs Aspect Ratio by Class")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_area_vs_aspect.png", dpi=150)
    plt.close()
    logger.info(f"Saved: scatter_area_vs_aspect.png")


def draw_box_on_image(
    img: np.ndarray,
    box: dict,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label: str = "",
) -> np.ndarray:
    """Draw a bounding box on image."""
    img = img.copy()
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness_text = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness_text
        )
        cv2.rectangle(img, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
        cv2.putText(
            img, label, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness_text
        )

    return img


def visualize_sample_images(
    all_boxes: list, manifest: list, top_n: int, sample_size: int, output_dir: Path
):
    """
    Visualize sample images with boxes drawn.
    Creates three sets: top-N biggest, top-N smallest, random sample.
    """
    viz_dir = output_dir / "sample_images"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Find outliers
    biggest, smallest = find_outliers(all_boxes, top_n)

    # Create manifest lookup for quick access
    manifest_lookup = {row.get("image_id"): row for row in manifest}

    # Visualize biggest boxes
    logger.info("Visualizing top-N biggest boxes...")
    for idx, box in enumerate(tqdm(biggest)):
        image_id = box["image_id"]
        image_path = Path(box["image_path"])

        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Get all boxes from this image
        image_boxes = [b for b in all_boxes if b["image_id"] == image_id]

        # Draw all boxes, highlight current one
        for b in image_boxes:
            color = CLASS_COLORS.get(b["class_id"], (128, 128, 128))
            label = CLASS_NAMES.get(b["class_id"], f"Class {b['class_id']}")
            if b == box:
                label = f"{label} (BIGGEST)"
            img = draw_box_on_image(img, b, color, 2, label)

        output_path = viz_dir / f"01_biggest_{idx:02d}_{image_id}.jpg"
        cv2.imwrite(str(output_path), img)
    logger.info(f"Saved {len(biggest)} biggest box images")

    # Visualize smallest boxes
    logger.info("Visualizing top-N smallest boxes...")
    for idx, box in enumerate(tqdm(smallest)):
        image_id = box["image_id"]
        image_path = Path(box["image_path"])

        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        image_boxes = [b for b in all_boxes if b["image_id"] == image_id]
        for b in image_boxes:
            color = CLASS_COLORS.get(b["class_id"], (128, 128, 128))
            label = CLASS_NAMES.get(b["class_id"], f"Class {b['class_id']}")
            if b == box:
                label = f"{label} (SMALLEST)"
            img = draw_box_on_image(img, b, color, 2, label)

        output_path = viz_dir / f"02_smallest_{idx:02d}_{image_id}.jpg"
        cv2.imwrite(str(output_path), img)
    logger.info(f"Saved {len(smallest)} smallest box images")

    # Random sample
    logger.info("Visualizing random sample...")
    sample_size = min(sample_size, len(set(b["image_id"] for b in all_boxes)))
    sampled_images = random.sample(
        list(set(b["image_id"] for b in all_boxes)), sample_size
    )

    for idx, image_id in enumerate(tqdm(sampled_images)):
        # Find image path from manifest or first box
        image_path = None
        for b in all_boxes:
            if b["image_id"] == image_id:
                image_path = Path(b["image_path"])
                break

        if image_path is None or not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        image_boxes = [b for b in all_boxes if b["image_id"] == image_id]
        for b in image_boxes:
            color = CLASS_COLORS.get(b["class_id"], (128, 128, 128))
            label = CLASS_NAMES.get(b["class_id"], f"Class {b['class_id']}")
            img = draw_box_on_image(img, b, color, 2, label)

        output_path = viz_dir / f"03_sample_{idx:03d}_{image_id}.jpg"
        cv2.imwrite(str(output_path), img)
    logger.info(f"Saved {len(sampled_images)} random sample images")


def generate_json_report(all_boxes: list, stats_by_class: dict, output_path: Path):
    """Generate JSON report with box statistics."""
    biggest, smallest = find_outliers(all_boxes, 10)

    # Summary stats
    summary = {}
    for class_id in sorted(stats_by_class.keys()):
        stats = stats_by_class[class_id]
        areas = stats["areas"]
        aspect_ratios = [ar for ar in stats["aspect_ratios"] if ar > 0]

        if areas:
            summary[CLASS_NAMES.get(class_id, f"class_{class_id}")] = {
                "total_boxes": len(areas),
                "area_min": float(min(areas)),
                "area_max": float(max(areas)),
                "area_mean": float(np.mean(areas)),
                "area_median": float(np.median(areas)),
                "area_std": float(np.std(areas)),
                "aspect_ratio_min": float(min(aspect_ratios)) if aspect_ratios else 0,
                "aspect_ratio_max": float(max(aspect_ratios)) if aspect_ratios else 0,
                "aspect_ratio_mean": float(np.mean(aspect_ratios))
                if aspect_ratios
                else 0,
            }

    # Top boxes
    report = {
        "summary": summary,
        "biggest_boxes": [
            {
                "rank": i + 1,
                "area": float(b["area"]),
                "aspect_ratio": float(b["aspect_ratio"]),
                "width": b["width"],
                "height": b["height"],
                "class": CLASS_NAMES.get(b["class_id"], f"class_{b['class_id']}"),
                "image_id": b["image_id"],
                "image_path": b["image_path"],
            }
            for i, b in enumerate(biggest)
        ],
        "smallest_boxes": [
            {
                "rank": i + 1,
                "area": float(b["area"]),
                "aspect_ratio": float(b["aspect_ratio"]),
                "width": b["width"],
                "height": b["height"],
                "class": CLASS_NAMES.get(b["class_id"], f"class_{b['class_id']}"),
                "image_id": b["image_id"],
                "image_path": b["image_path"],
            }
            for i, b in enumerate(smallest)
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bounding box sizes and properties"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/images.csv",
        help="Path to image manifest CSV",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels_final",
        help="Path to labels directory (will try labels_autogen as fallback)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of biggest/smallest boxes to visualize",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random images to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/box_viz",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    # Try fallback label directory
    if not labels_dir.exists():
        fallback = labels_dir.parent / "labels_autogen"
        if fallback.exists():
            logger.info(f"Labels dir not found, using fallback: {fallback}")
            labels_dir = fallback
        else:
            logger.warning(f"No label directory found: {labels_dir} or {fallback}")
            logger.warning("Continuing anyway, will skip images without labels")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze
    logger.info(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded {len(manifest)} images")

    all_boxes, stats_by_class = analyze_dataset(manifest, labels_dir)
    logger.info(f"Found {len(all_boxes)} total boxes")

    if not all_boxes:
        logger.warning("No boxes found! Skipping visualization.")
        return

    # Print summary stats
    logger.info("\n=== Summary Statistics ===")
    for class_id in sorted(stats_by_class.keys()):
        stats = stats_by_class[class_id]
        areas = stats["areas"]
        if areas:
            logger.info(f"\n{CLASS_NAMES.get(class_id, f'Class {class_id}')}:")
            logger.info(f"  Total boxes: {len(areas)}")
            logger.info(
                f"  Area: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.0f}"
            )

    # Create visualizations
    create_visualizations(all_boxes, stats_by_class, output_dir)
    visualize_sample_images(
        all_boxes, manifest, args.top_n, args.sample_size, output_dir
    )

    # Generate JSON report
    json_output = output_dir / "box_visualization.json"
    generate_json_report(all_boxes, stats_by_class, json_output)

    logger.info(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
