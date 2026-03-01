#!/usr/bin/env python3
"""
Labeling UI - FastAPI backend.
Serves images and labels, allows editing via REST API.
"""

import argparse
import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional


app = FastAPI(title="ParkOpticon Labeler")

MANIFEST_PATH = Path("manifests/images.csv")
LABELS_AUTOGEN_DIR = Path("data/labels_autogen")
LABELS_FINAL_DIR = Path("data/labels_final")
IMAGES_ORIGINAL_DIR = Path("data/images_original")
IMAGES_SYNTH_DIR = Path("data/images_synth")

templates = Jinja2Templates(directory="labeler/static")


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int


class LabelUpdate(BaseModel):
    boxes: List[Box]


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, 'r') as f:
        return list(csv.DictReader(f))


def save_manifest(manifest):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(MANIFEST_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def get_image_path(row):
    file_path = row.get('file_path', '')
    if file_path:
        return Path(file_path)
    
    image_id = row.get('image_id', '')
    for ext in ['.jpg', '.png']:
        path = IMAGES_ORIGINAL_DIR / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]
        
        path = IMAGES_SYNTH_DIR / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]
    
    return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("labeler/static/index.html", 'r') as f:
        return f.read()


@app.get("/api/images")
async def get_images(needs_review: bool = True, status: str = "todo"):
    manifest = load_manifest()
    
    images = []
    for row in manifest:
        if row.get('status') != 'ok':
            continue
        
        if needs_review and row.get('needs_review') != '1':
            continue
        
        if status and row.get('review_status') not in ['', status]:
            continue
        
        if row.get('review_status') == 'done':
            continue
        
        image_path = get_image_path(row)
        if image_path and image_path.exists():
            images.append({
                'image_id': row.get('image_id'),
                'image_path': str(image_path),
                'is_synthetic': row.get('is_synthetic') == '1',
                'edit_type': row.get('edit_type', ''),
                'expected_class': row.get('expected_inserted_class', ''),
                'review_status': row.get('review_status', 'todo')
            })
    
    return images


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    manifest = load_manifest()
    
    row = None
    for r in manifest:
        if r.get('image_id') == image_id:
            row = r
            break
    
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    
    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    label_path = LABELS_FINAL_DIR / f"{image_id}.txt"
    boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    x1 = (x_center - width / 2)
                    y1 = (y_center - height / 2)
                    x2 = (x_center + width / 2)
                    y2 = (y_center + height / 2)
                    
                    boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cls': cls
                    })
    else:
        label_path = LABELS_AUTOGEN_DIR / f"{image_id}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        x1 = (x_center - width / 2)
                        y1 = (y_center - height / 2)
                        x2 = (x_center + width / 2)
                        y2 = (y_center + height / 2)
                        
                        boxes.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cls': cls
                        })
    
    return {
        'image_id': image_id,
        'image_path': str(image_path),
        'boxes': boxes,
        'is_synthetic': row.get('is_synthetic') == '1',
        'edit_type': row.get('edit_type', ''),
        'expected_class': row.get('expected_inserted_class', '')
    }


@app.post("/api/labels/{image_id}")
async def save_labels(image_id: str, data: LabelUpdate):
    LABELS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    
    label_path = LABELS_FINAL_DIR / f"{image_id}.txt"
    with open(label_path, 'w') as f:
        for box in data.boxes:
            x_center = (box.x1 + box.x2) / 2
            y_center = (box.y1 + box.y2) / 2
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            
            f.write(f"{box.cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    manifest = load_manifest()
    for row in manifest:
        if row.get('image_id') == image_id:
            row['review_status'] = 'done'
            row['needs_review'] = '0'
            break
    
    save_manifest(manifest)
    
    return {'status': 'saved', 'boxes': len(data.boxes)}


@app.post("/api/review/{image_id}")
async def update_review_status(image_id: str, status: str):
    manifest = load_manifest()
    
    for row in manifest:
        if row.get('image_id') == image_id:
            row['review_status'] = status
            break
    
    save_manifest(manifest)
    
    return {'status': status}


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="Run labeler server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
