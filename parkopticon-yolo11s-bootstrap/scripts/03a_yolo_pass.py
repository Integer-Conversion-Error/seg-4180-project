#!/usr/bin/env python3

import argparse
import csv
import importlib
import json
import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def detect_vehicles(
    model, image_path: Path, conf: float = 0.25, device: str = "0"
) -> list[dict]:
    results = model(image_path, conf=conf, verbose=False, device=device)
    boxes: list[dict] = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in COCO_VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": float(box.conf[0]),
                        "coco_class": cls_id,
                        "coco_class_name": COCO_VEHICLE_CLASSES[cls_id],
                    }
                )
    return boxes


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(run_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else run_dir / candidate


def _resolve_image_path(run_dir: Path, raw_value: str) -> Path:
    if not raw_value:
        return run_dir / ""
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else run_dir / candidate


def nuke_run_state(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    data_dir = (run_dir / "data").resolve()
    manifests_dir = (run_dir / "manifests").resolve()
    keep_images_original = (data_dir / "images_original").resolve()
    keep_images_csv = (manifests_dir / "images.csv").resolve()
    keep_points_csv = (manifests_dir / "points.csv").resolve()

    if not run_dir.exists():
        return

    for child in list(run_dir.iterdir()):
        resolved = child.resolve()
        if resolved == data_dir:
            for dchild in list(data_dir.iterdir()):
                dres = dchild.resolve()
                if dres == keep_images_original:
                    continue
                if dchild.is_dir():
                    shutil.rmtree(dchild)
                elif dchild.exists():
                    dchild.unlink()
            continue
        if resolved == manifests_dir:
            for mchild in list(manifests_dir.iterdir()):
                mres = mchild.resolve()
                if mres in {keep_images_csv, keep_points_csv}:
                    continue
                if mchild.is_dir():
                    shutil.rmtree(mchild)
                elif mchild.exists():
                    mchild.unlink()
            continue
        if child.is_dir():
            shutil.rmtree(child)
        elif child.exists():
            child.unlink()


def reset_detect_stage_state(
    manifest: list[dict],
    boxes_out: Path,
    empty_out: Path,
    valid_road_out: Path,
    non_road_out: Path,
    non_street_out: Path,
    yolo_empty_out: Path,
    stash_non_road_dir: Path,
    stash_non_street_dir: Path,
) -> None:
    for output in [
        boxes_out,
        empty_out,
        valid_road_out,
        non_road_out,
        non_street_out,
        yolo_empty_out,
    ]:
        if output.exists():
            output.unlink()

    if stash_non_road_dir.exists():
        shutil.rmtree(stash_non_road_dir)
    if stash_non_street_dir.exists():
        shutil.rmtree(stash_non_street_dir)

    reset_fields = [
        "street_scene_valid",
        "street_scene_confidence_low",
        "street_scene_confidence_high",
        "street_scene_explanation",
        "road_scene_valid",
        "road_scene_confidence_low",
        "road_scene_confidence_high",
        "road_scene_explanation",
        "gemini_gate_model",
        "gemini_gate_json",
        "route_category",
        "excluded_reason",
        "num_boxes_autogen",
    ]

    for row in manifest:
        for field in reset_fields:
            if field in row:
                row[field] = ""


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Stage 03a: YOLO-only empty detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv")
    parser.add_argument("--out-manifest", "-o", default="manifests/images.csv")
    parser.add_argument("--boxes-out", "-b", default="manifests/boxes_autogen.jsonl")
    parser.add_argument("--empty-out", "-e", default="lists/empty_candidates.txt")
    parser.add_argument("--valid-road-out", default="lists/valid_road_candidates.txt")
    parser.add_argument("--non-road-out", default="lists/non_road_candidates.txt")
    parser.add_argument("--non-street-out", default="lists/non_street_candidates.txt")
    parser.add_argument("--yolo-empty-out", default="lists/yolo_empty_all.txt")
    parser.add_argument("--stash-non-road-dir", default="data/images_excluded/non_road")
    parser.add_argument(
        "--stash-non-street-dir", default="data/images_excluded/non_street"
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-fresh-reset", action="store_true")
    parser.add_argument("--nuke", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.nuke:
        nuke_run_state(run_dir)

    manifest_path = _resolve_path(run_dir, args.manifest)
    out_manifest_path = _resolve_path(run_dir, args.out_manifest)
    boxes_out = _resolve_path(run_dir, args.boxes_out)
    yolo_empty_out = _resolve_path(run_dir, args.yolo_empty_out)
    empty_out = _resolve_path(run_dir, args.empty_out)
    valid_road_out = _resolve_path(run_dir, args.valid_road_out)
    non_road_out = _resolve_path(run_dir, args.non_road_out)
    non_street_out = _resolve_path(run_dir, args.non_street_out)
    stash_non_road_dir = _resolve_path(run_dir, args.stash_non_road_dir)
    stash_non_street_dir = _resolve_path(run_dir, args.stash_non_street_dir)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1

    manifest = load_manifest(manifest_path)
    if not args.no_fresh_reset:
        reset_detect_stage_state(
            manifest,
            boxes_out,
            empty_out,
            valid_road_out,
            non_road_out,
            non_street_out,
            yolo_empty_out,
            stash_non_road_dir,
            stash_non_street_dir,
        )

    ultralytics_module = importlib.import_module("ultralytics")
    model = ultralytics_module.YOLO(args.model)

    boxes_out.parent.mkdir(parents=True, exist_ok=True)
    yolo_empty_out.parent.mkdir(parents=True, exist_ok=True)

    yolo_empty_ids: list[str] = []
    eligible = 0

    with open(boxes_out, "w", encoding="utf-8") as boxes_handle:
        for row in tqdm(manifest, desc="Stage 03a: YOLO pass"):
            image_id = row.get("image_id", "")
            file_path = _resolve_image_path(run_dir, row.get("file_path", ""))
            if not file_path.exists() or row.get("status") != "ok":
                continue
            eligible += 1
            boxes = detect_vehicles(
                model, file_path, conf=args.conf, device=args.device
            )
            boxes_handle.write(
                json.dumps({"image_id": image_id, "boxes": boxes}) + "\n"
            )
            row["num_boxes_autogen"] = str(len(boxes))
            if len(boxes) == 0:
                yolo_empty_ids.append(image_id)

    with open(yolo_empty_out, "w", encoding="utf-8") as handle:
        for image_id in yolo_empty_ids:
            handle.write(f"{image_id}\n")

    save_manifest(manifest, out_manifest_path)
    logger.info("Processed manifest rows: %s", len(manifest))
    logger.info("YOLO-eligible rows (status ok + file exists): %s", eligible)
    logger.info("YOLO 0-vehicle images (raw empty): %s", len(yolo_empty_ids))
    logger.info("YOLO empty list saved: %s", yolo_empty_out)
    logger.info("Boxes saved: %s", boxes_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
