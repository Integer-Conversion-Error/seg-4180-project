#!/usr/bin/env python3
"""
Rebuild images.csv from existing images on disk, matching against points.csv.

Useful when images.csv was lost but images_original/ still exists.
Avoids re-downloading images and wasting API quota.
"""

import argparse
import csv
import hashlib
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_image_id(location: str, heading: int, pitch: int, fov: int) -> str:
    """Generate image ID matching 02_fetch_streetview.py logic."""
    content = f"{location}_{heading}_{pitch}_{fov}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def find_existing_images(images_dir: Path) -> dict[str, Path]:
    """
    Scan images_original directory for existing images.
    Returns dict mapping image_id -> file_path.
    """
    existing: dict[str, Path] = {}
    if not images_dir.exists():
        return existing

    for street_dir in images_dir.iterdir():
        if not street_dir.is_dir():
            continue
        for img_file in street_dir.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_id = img_file.stem
                existing[image_id] = img_file

    return existing


def rebuild_manifest(
    points_path: Path,
    images_dir: Path,
    output_path: Path,
    missing_status: str = "missing",
) -> tuple[int, int]:
    """
    Rebuild images.csv from points.csv and existing images.

    Returns (found_count, missing_count).
    """
    if not points_path.exists():
        logger.error("Points file not found: %s", points_path)
        return 0, 0

    existing_images = find_existing_images(images_dir)
    logger.info("Found %d existing images in %s", len(existing_images), images_dir)

    with open(points_path, "r", encoding="utf-8") as f:
        points = list(csv.DictReader(f))

    logger.info("Processing %d points from %s", len(points), points_path)

    manifest: list[dict] = []
    found_count = 0
    missing_count = 0

    for point in points:
        street = point.get("label") or point.get("street", "unknown")
        location = point.get("location", "")
        heading = int(point.get("heading", 0) or 0)
        pitch = int(point.get("pitch", 0) or 0)
        fov = int(point.get("fov", 80) or 80)

        image_id = generate_image_id(location, heading, pitch, fov)

        # Check if image exists on disk
        if image_id in existing_images:
            file_path = existing_images[image_id]
            status = "ok"
            found_count += 1
        else:
            file_path = (
                images_dir
                / street.replace("/", "_").replace(" ", "_")
                / f"{image_id}.jpg"
            )
            status = missing_status
            missing_count += 1

        row = {
            "image_id": image_id,
            "file_path": str(file_path) if status == "ok" else "",
            "split": "unset",
            "parent_image_id": "",
            "is_synthetic": "0",
            "edit_type": "none",
            "expected_inserted_class": "none",
            "street": street,
            "input_location": location,
            "heading": str(heading),
            "pitch": str(pitch),
            "fov": str(fov),
            "pano_id": point.get("pano_id", ""),
            "pano_lat": point.get("pano_lat", ""),
            "pano_lng": point.get("pano_lng", ""),
            "status": status,
            "num_boxes_autogen": "0",
            "needs_review": "0",
            "review_status": "todo",
            "created_at": point.get("created_at", datetime.now().isoformat()),
            "source_file_path": point.get("source_file_path", ""),
            "crop_bottom_px": point.get("bottom_crop", point.get("crop_bottom_px", "")),
            "crop_status": "pending" if status == "ok" else "",
            "cropped_at": "",
            "qa_passed": "",
            "label_valid": "",
            "label_error": "",
        }
        manifest.append(row)

    # Write manifest
    if manifest:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(manifest[0].keys())
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)
        logger.info("Wrote manifest to %s (%d entries)", output_path, len(manifest))

    return found_count, missing_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild images.csv from existing images on disk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--points",
        "-p",
        default="manifests/points.csv",
        help="Input points CSV",
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        default="data/images_original",
        help="Directory containing existing images",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="manifests/images.csv",
        help="Output manifest path",
    )
    parser.add_argument(
        "--missing-status",
        default="missing",
        help="Status to assign to points without images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing",
    )
    args = parser.parse_args()

    points_path = Path(args.points).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_path = Path(args.output).resolve()

    if not points_path.exists():
        logger.error("Points file not found: %s", points_path)
        return 1

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        return 1

    found, missing = rebuild_manifest(
        points_path,
        images_dir,
        output_path if not args.dry_run else Path("/dev/null"),
        args.missing_status,
    )

    logger.info("Summary: %d images found, %d missing", found, missing)

    if args.dry_run:
        logger.info("Dry run - no manifest written")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
