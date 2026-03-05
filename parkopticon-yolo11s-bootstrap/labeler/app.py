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
EXCLUDED_FROM_SYNTH_PATH = Path("lists/excluded_from_synth.txt")

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
    with open(MANIFEST_PATH, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def get_image_path(row):
    file_path = row.get("file_path", "")
    if file_path:
        return Path(file_path)

    image_id = row.get("image_id", "")
    for ext in [".jpg", ".png"]:
        path = IMAGES_ORIGINAL_DIR / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]

        path = IMAGES_SYNTH_DIR / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]

    return None


def get_row_by_image_id(manifest, image_id: str):
    for row in manifest:
        if row.get("image_id") == image_id:
            return row
    return None


def load_excluded_ids() -> set:
    if not EXCLUDED_FROM_SYNTH_PATH.exists():
        return set()
    with open(EXCLUDED_FROM_SYNTH_PATH, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("labeler/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/app.js")
async def app_js():
    return FileResponse("labeler/static/app.js", media_type="application/javascript")


@app.get("/api/images")
async def get_images(
    queue: str = "needs_review", status: str = "todo", include_excluded: bool = False
):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()

    images = []
    for row in manifest:
        if row.get("status") != "ok":
            continue

        image_id = row.get("image_id")
        if not include_excluded and image_id in excluded_ids:
            continue

        needs_review_value = row.get("needs_review", "")
        if queue == "needs_review" and needs_review_value != "1":
            continue
        if queue == "autolabel_ok" and needs_review_value != "0":
            continue

        if status and row.get("review_status") not in ["", status]:
            continue

        if row.get("review_status") == "done":
            continue

        image_path = get_image_path(row)
        if image_path and image_path.exists():
            images.append(
                {
                    "image_id": row.get("image_id"),
                    "image_path": f"/api/image_file/{row.get('image_id')}",
                    "is_synthetic": row.get("is_synthetic") == "1",
                    "is_excluded": row.get("image_id") in excluded_ids,
                    "edit_type": row.get("edit_type", ""),
                    "expected_class": row.get("expected_inserted_class", ""),
                    "review_status": row.get("review_status", "todo"),
                }
            )

    return images


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()
    row = get_row_by_image_id(manifest, image_id)

    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    from PIL import Image

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    # Try labels_final first, fall back to labels_autogen.
    # Only use labels_final if it has actual content (not an empty file).
    label_path_final = LABELS_FINAL_DIR / f"{image_id}.txt"
    label_path_autogen = LABELS_AUTOGEN_DIR / f"{image_id}.txt"

    label_path = None
    if label_path_final.exists() and label_path_final.read_text().strip():
        label_path = label_path_final
    elif label_path_autogen.exists():
        label_path = label_path_autogen

    boxes = []
    if label_path:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    x1 = (x_center - width / 2) * img_w
                    y1 = (y_center - height / 2) * img_h
                    x2 = (x_center + width / 2) * img_w
                    y2 = (y_center + height / 2) * img_h

                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})

    return {
        "image_id": image_id,
        "image_path": f"/api/image_file/{image_id}",
        "boxes": boxes,
        "is_synthetic": row.get("is_synthetic") == "1",
        "is_excluded": image_id in excluded_ids,
        "edit_type": row.get("edit_type", ""),
        "expected_class": row.get("expected_inserted_class", ""),
    }


@app.get("/api/image_file/{image_id}")
async def get_image_file(image_id: str):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(str(image_path))


@app.post("/api/labels/{image_id}")
async def save_labels(image_id: str, data: LabelUpdate):
    """
    Save labels in YOLO format.

    The frontend sends boxes as normalized corner coordinates:
        x1, y1, x2, y2  ∈ [0, 1]  (left/top/right/bottom divided by image dims)

    This endpoint converts to YOLO center format:
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width    = x2 - x1
        height   = y2 - y1
    All values remain normalized in [0, 1].
    """
    LABELS_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    label_path = LABELS_FINAL_DIR / f"{image_id}.txt"
    skipped = 0
    with open(label_path, "w") as f:
        for box in data.boxes:
            # Clamp corners to [0, 1] — canvas drag can occasionally produce
            # coords just outside this range.
            x1 = max(0.0, min(1.0, box.x1))
            y1 = max(0.0, min(1.0, box.y1))
            x2 = max(0.0, min(1.0, box.x2))
            y2 = max(0.0, min(1.0, box.y2))

            # Ensure x2 > x1 and y2 > y1 after clamping
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            f.write(
                f"{box.cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

    manifest = load_manifest()
    for row in manifest:
        if row.get("image_id") == image_id:
            row["review_status"] = "done"
            row["needs_review"] = "0"
            break

    save_manifest(manifest)

    saved = len(data.boxes) - skipped
    return {"status": "saved", "boxes": saved, "skipped": skipped}


@app.post("/api/review/{image_id}")
async def update_review_status(image_id: str, status: str):
    manifest = load_manifest()

    for row in manifest:
        if row.get("image_id") == image_id:
            row["review_status"] = status
            row["needs_review"] = "0" if status == "done" else "1"
            break

    save_manifest(manifest)

    return {"status": status}


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Run labeler server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
