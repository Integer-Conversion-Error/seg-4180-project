#!/usr/bin/env python3
"""
Count bounding boxes by class per split.

Reads YOLO format labels from splits/{split}/labels/ and counts boxes by class.
Outputs results to console (table) and JSON file.

Classes:
  0 - vehicle
  1 - enforcement_vehicle
  2 - police_old
  3 - police_new
  4 - lookalike_negative
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "vehicle",
    1: "enforcement_vehicle",
    2: "police_old",
    3: "police_new",
    4: "lookalike_negative",
}

def load_manifest(manifest_path: Path) -> list:
    """Load manifest CSV and return list of dicts."""
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def get_split_images(manifest: list, split: str) -> set:
    """Get image IDs for a given split from manifest."""
    image_ids = set()
    for row in manifest:
        if row.get("split") == split:
            image_ids.add(row.get("image_id", ""))
    return image_ids


def count_boxes_in_file(label_path: Path) -> dict:
    """Count boxes by class in a single YOLO label file.

    Format per line: class_id x_center y_center width height
    Empty files return empty dict.
    """
    counts = defaultdict(int)

    if not label_path.exists():
        return counts

    try:
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                parts = line.split()
                if len(parts) < 5:  # Invalid line
                    logger.warning(f"Invalid line in {label_path}: {line}")
                    continue

                try:
                    class_id = int(parts[0])
                    counts[class_id] += 1
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse class ID in {label_path}: {line}")
                    continue
    except Exception as e:
        logger.error(f"Failed to read {label_path}: {e}")

    return counts


def count_boxes_per_split(splits_dir: Path, manifest: list) -> dict:
    """Count boxes by class for each split.

    Returns:
        {
            'train': {0: count, 1: count, ...},
            'val': {...},
            'test': {...}
        }
    """
    results = {}

    for split in ["train", "val", "test"]:
        split_counts = defaultdict(int)
        labels_dir = splits_dir / split / "labels"

        if not labels_dir.exists():
            logger.warning(f"Labels directory not found: {labels_dir}")
            results[split] = {}
            continue

        # Get image IDs for this split from manifest
        split_images = get_split_images(manifest, split)

        # Count boxes in each label file
        for label_file in labels_dir.glob("*.txt"):
            image_id = label_file.stem

            # Only count if image is in manifest for this split
            if image_id not in split_images:
                continue

            counts = count_boxes_in_file(label_file)
            for class_id, count in counts.items():
                split_counts[class_id] += count

        results[split] = dict(split_counts)

    return results


def compute_totals(split_counts: dict) -> dict:
    """Compute total counts across all splits.

    Returns:
        {0: count, 1: count, ...}
    """
    totals = defaultdict(int)
    for split_data in split_counts.values():
        for class_id, count in split_data.items():
            totals[class_id] += count
    return dict(totals)


def print_table(split_counts: dict, totals: dict):
    """Print results as a formatted table to console."""
    print("\n" + "=" * 80)
    print("BOUNDING BOX COUNTS BY CLASS PER SPLIT")
    print("=" * 80)

    # Determine all class IDs present
    all_classes = set()
    for split_data in split_counts.values():
        all_classes.update(split_data.keys())
    all_classes.update(totals.keys())
    all_classes = sorted(all_classes)

    # Header
    header = "Split".ljust(12)
    for class_id in all_classes:
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        header += f" {class_name.ljust(18)}"
    header += " TOTAL"
    print(header)
    print("-" * (len(header) + 10))

    # Data rows
    for split in ["train", "val", "test"]:
        row = split.ljust(12)
        split_data = split_counts.get(split, {})
        split_total = 0

        for class_id in all_classes:
            count = split_data.get(class_id, 0)
            split_total += count
            row += f" {str(count).ljust(18)}"

        row += f" {split_total}"
        print(row)

    # Totals row
    print("-" * (len(header) + 10))
    row = "TOTAL".ljust(12)
    grand_total = 0
    for class_id in all_classes:
        count = totals.get(class_id, 0)
        grand_total += count
        row += f" {str(count).ljust(18)}"
    row += f" {grand_total}"
    print(row)

    print("=" * 80 + "\n")


def save_json_report(split_counts: dict, totals: dict, output_path: Path):
    """Save results to JSON file."""
    # Convert integer keys to strings for JSON compatibility
    splits_json = {}
    for split, counts in split_counts.items():
        splits_json[split] = {}
        for class_id, count in counts.items():
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            splits_json[split][class_name] = count

    totals_json = {}
    for class_id, count in totals.items():
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        totals_json[class_name] = count

    report = {"splits": splits_json, "totals": totals_json}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Count bounding boxes by class per split"
    )
    parser.add_argument(
        "--manifest",
        default="manifests/images.csv",
        help="Path to image manifest CSV (default: manifests/images.csv)",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Path to splits directory (default: data/splits)",
    )
    parser.add_argument(
        "--output",
        default="reports/box_counts.json",
        help="Path to output JSON file (default: reports/box_counts.json)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    splits_dir = Path(args.splits_dir)
    output_path = Path(args.output)

    # Validate inputs
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    if not splits_dir.exists():
        logger.error(f"Splits directory not found: {splits_dir}")
        return

    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded {len(manifest)} images from manifest")

    # Count boxes per split
    logger.info(f"Counting boxes from {splits_dir}")
    split_counts = count_boxes_per_split(splits_dir, manifest)

    # Compute totals
    totals = compute_totals(split_counts)

    # Print table
    print_table(split_counts, totals)

    # Save JSON report
    save_json_report(split_counts, totals, output_path)


if __name__ == "__main__":
    main()
