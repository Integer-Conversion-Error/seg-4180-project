#!/usr/bin/env python3
"""
Detect vehicles in Street View images using pretrained YOLO.
Identifies images with no vehicle detections (empty_candidates).
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle", 
    5: "bus",
    7: "truck"
}


def detect_vehicles(
    model: YOLO,
    image_path: Path,
    conf: float = 0.25,
    vehicle_classes: list = None
) -> list:
    if vehicle_classes is None:
        vehicle_classes = [2, 3, 5, 7]
    
    results = model(image_path, conf=conf, verbose=False)
    
    boxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf_score,
                    "coco_class": cls_id,
                    "coco_class_name": COCO_VEHICLE_CLASSES.get(cls_id, "unknown")
                })
    
    return boxes


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, 'r') as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in images")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Input manifest")
    parser.add_argument("--out-manifest", "-o", default="manifests/images.csv", help="Updated manifest")
    parser.add_argument("--boxes-out", "-b", default="manifests/boxes_autogen.jsonl", help="Boxes JSONL output")
    parser.add_argument("--empty-out", "-e", default="lists/empty_candidates.txt", help="Empty candidates list")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model (auto-downloads)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device (cpu or 0)")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    boxes_out = Path(args.bboxes_out) if hasattr(args, 'bboxes_out') else Path(args.boxes_out.replace('.jsonl', '_temp.jsonl'))
    empty_out = Path(args.empty_out)
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    manifest = load_manifest(manifest_path)
    
    logger.info(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    boxes_out.parent.mkdir(parents=True, exist_ok=True)
    empty_out.parent.mkdir(parents=True, exist_ok=True)
    
    empty_candidates = []
    
    with open(boxes_out, 'w') as boxes_f:
        for row in tqdm(manifest, desc="Detecting vehicles"):
            image_id = row.get('image_id', '')
            file_path = Path(row.get('file_path', ''))
            
            if not file_path.exists() or row.get('status') != 'ok':
                continue
            
            boxes = detect_vehicles(model, file_path, args.conf)
            
            boxes_f.write(json.dumps({
                "image_id": image_id,
                "boxes": boxes
            }) + "\n")
            
            if len(boxes) == 0:
                empty_candidates.append(image_id)
                row['num_boxes_autogen'] = '0'
            else:
                row['num_boxes_autogen'] = str(len(boxes))
            
            if image_id in empty_candidates:
                row['needs_review'] = '1'
    
    with open(empty_out, 'w') as f:
        for img_id in empty_candidates:
            f.write(img_id + "\n")
    
    save_manifest(manifest, Path(args.out_manifest))
    
    logger.info(f"Processed {len(manifest)} images")
    logger.info(f"Found {len(empty_candidates)} empty candidates")
    logger.info(f"Boxes saved to {boxes_out}")
    logger.info(f"Empty candidates saved to {empty_out}")


if __name__ == "__main__":
    main()
