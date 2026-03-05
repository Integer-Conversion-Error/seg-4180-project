"""
Render synthetic images with YOLO bounding boxes overlaid.

Picks a few examples from each class subdirectory under data/images_synth/,
reads the corresponding autogen label, and saves annotated images to
data/viz_synth_examples/.

Usage:
    python scripts/visualize_synth_boxes.py [--per-class 3] [--out data/viz_synth_examples]
"""

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = {
    0: "vehicle",
    1: "enforcement_vehicle",
    2: "police_old",
    3: "police_new",
    4: "lookalike_negative",
}

CLASS_COLORS = {
    0: (0, 200, 0),  # green
    1: (255, 165, 0),  # orange
    2: (0, 120, 255),  # blue
    3: (220, 20, 60),  # crimson
    4: (128, 128, 128),  # gray
}

SYNTH_DIR = Path("data/images_synth")
LABELS_DIR = Path("data/labels_autogen")
SYNTH_SUBDIRS = ["enforcement_vehicle", "police_new", "police_old", "random_vehicle"]


def parse_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into list of (cls, xc, yc, w, h)."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
        boxes.append((cls, xc, yc, w, h))
    return boxes


def draw_boxes(
    img: Image.Image, boxes: list[tuple[int, float, float, float, float]]
) -> Image.Image:
    """Draw YOLO boxes on an image. Returns annotated copy."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    iw, ih = img.size

    # Try to get a decent font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for cls, xc, yc, w, h in boxes:
        # Convert normalized center coords to pixel corners
        x1 = (xc - w / 2) * iw
        y1 = (yc - h / 2) * ih
        x2 = (xc + w / 2) * iw
        y2 = (yc + h / 2) * ih

        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = CLASS_NAMES.get(cls, f"cls_{cls}")

        # Draw box (3px wide)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Label background
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        # Place label above box if room, else inside top
        label_y = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
        draw.rectangle([x1, label_y, x1 + tw + 6, label_y + th + 4], fill=color)
        draw.text((x1 + 3, label_y + 2), label, fill=(255, 255, 255), font=font)

    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class", type=int, default=3, help="Examples per class subdir"
    )
    parser.add_argument(
        "--out", type=str, default="data/viz_synth_examples", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for subdir_name in SYNTH_SUBDIRS:
        subdir = SYNTH_DIR / subdir_name
        if not subdir.is_dir():
            print(f"SKIP {subdir} (not found)")
            continue

        images = sorted(subdir.glob("*.jpg"))
        if not images:
            print(f"SKIP {subdir} (no .jpg files)")
            continue

        # Pick random subset
        samples = random.sample(images, min(args.per_class, len(images)))

        for img_path in samples:
            image_id = img_path.stem  # e.g. 0134411e07b7_enforcement_vehicle
            label_path = LABELS_DIR / f"{image_id}.txt"
            boxes = parse_yolo_label(label_path)

            img = Image.open(img_path).convert("RGB")
            annotated = draw_boxes(img, boxes)

            out_name = f"{subdir_name}__{image_id}.jpg"
            out_path = out_dir / out_name
            annotated.save(out_path, quality=92)
            print(f"  {out_path}  ({len(boxes)} boxes)")
            total += 1

    print(f"\nDone. {total} annotated images saved to {out_dir}/")


if __name__ == "__main__":
    main()
