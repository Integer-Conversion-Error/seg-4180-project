#!/usr/bin/env python3
"""
Pre-label bounding boxes using pretrained YOLO.
For synthetic images, uses image differencing to identify inserted vehicle.
"""

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure
from tqdm import tqdm
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": conf_score, "coco_class": cls_id
                })
    return boxes


def compute_change_region(original_path: Path, edited_path: Path) -> tuple:
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
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        return (x, y, x + w, y + h)
    except Exception as e:
        logger.error(f"Change detection failed: {e}")
        return None


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


def convert_to_yolo(boxes: list, img_width: int, img_height: int) -> list:
    yolo_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_boxes.append({
            'class': box['class'],
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'conf': box.get('conf', 1.0)
        })
    return yolo_boxes


def write_yolo_labels(labels: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f"{label['class']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n")


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
    parser = argparse.ArgumentParser(description="Pre-label bounding boxes")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Input manifest")
    parser.add_argument("--out-dir", "-o", default="data/labels_autogen", help="Output labels dir")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    manifest = load_manifest(manifest_path)
    
    logger.info(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    for row in tqdm(manifest, desc="Pre-labeling"):
        image_id = row.get('image_id', '')
        file_path = Path(row.get('file_path', ''))
        
        if not file_path.exists() or row.get('status') != 'ok':
            continue
        
        is_synthetic = row.get('is_synthetic') == '1'
        expected_class = row.get('expected_inserted_class', 'none')
        
        boxes = detect_vehicles(model, file_path, args.conf)
        
        if is_synthetic and expected_class != 'none':
            parent_id = row.get('parent_image_id', '')
            parent_row = next((r for r in manifest if r.get('image_id') == parent_id), None)
            
            if parent_row:
                parent_path = Path(parent_row.get('file_path', ''))
                if parent_path.exists():
                    change_bbox = compute_change_region(parent_path, file_path)
                    
                    if change_bbox:
                        best_iou = 0
                        best_box_idx = -1
                        
                        for idx, box in enumerate(boxes):
                            iou = box_iou(
                                (box['x1'], box['y1'], box['x2'], box['y2']),
                                change_bbox
                            )
                            if iou > best_iou:
                                best_iou = iou
                                best_box_idx = idx
                        
                        if best_iou >= 0.1 and best_box_idx >= 0:
                            class_id = 1 if expected_class == 'enforcement_vehicle' else 0
                            boxes[best_box_idx]['class'] = class_id
                        else:
                            row['needs_review'] = '1'
                            for box in boxes:
                                box['class'] = 0
                    else:
                        for box in boxes:
                            box['class'] = 0
                else:
                    for box in boxes:
                        box['class'] = 0
            else:
                for box in boxes:
                    box['class'] = 0
        else:
            for box in boxes:
                box['class'] = 0
        
        img = cv2.imread(str(file_path))
        if img is not None:
            h, w = img.shape[:2]
            yolo_labels = convert_to_yolo(boxes, w, h)
            
            output_path = out_dir / f"{image_id}.txt"
            write_yolo_labels(yolo_labels, output_path)
            
            row['num_boxes_autogen'] = str(len(boxes))
    
    save_manifest(manifest, manifest_path)
    logger.info(f"Labels saved to {out_dir}")


if __name__ == "__main__":
    main()
