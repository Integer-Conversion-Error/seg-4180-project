#!/usr/bin/env python3
"""
Detect duplicate and near-duplicate images using perceptual hashing.

Compares images within each split (train/val/test) and reports duplicates
with their perceptual hash distance. Supports both exact (MD5) and perceptual
hashing (pHash) for similarity detection.

Output: JSON report with duplicate pairs per split.
"""

import argparse
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_md5(file_path):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_phash(file_path):
    """Compute perceptual hash of an image."""
    if not HAS_IMAGEHASH:
        raise ImportError(
            "imagehash library not available. Install with: pip install imagehash"
        )

    try:
        img = Image.open(file_path)
        return imagehash.phash(img)
    except Exception as e:
        logger.warning(f"Failed to compute pHash for {file_path}: {e}")
        return None


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two imagehash objects."""
    if hash1 is None or hash2 is None:
        return float("inf")
    return hash1 - hash2


def find_duplicates_in_split(image_rows, splits_dir, threshold=8, hash_method="phash"):
    """
    Find duplicates within a single split.

    Args:
        image_rows: List of dicts with 'image_id' and 'file_path'
        splits_dir: Directory containing split images
        threshold: Maximum Hamming distance for perceptual hash
        hash_method: 'md5' for exact matching, 'phash' for perceptual

    Returns:
        List of dicts with {img1, img2, distance}
    """
    duplicates = []

    # Build hash map
    hash_map = {}
    valid_files = []

    for row in image_rows:
        image_id = row.get("image_id")
        file_path = row.get("file_path")

        if not file_path:
            logger.warning(f"No file_path for image_id {image_id}")
            continue

        # Resolve absolute path
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = Path(splits_dir).parent / full_path

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            continue

        try:
            if hash_method == "md5":
                file_hash = compute_md5(full_path)
            elif hash_method == "phash":
                file_hash = compute_phash(full_path)
            else:
                raise ValueError(f"Unknown hash method: {hash_method}")

            if file_hash is None:
                continue

            valid_files.append((image_id, file_path, file_hash))

            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(image_id)

        except Exception as e:
            logger.warning(f"Error hashing {image_id}: {e}")
            continue

    logger.info(f"Hashed {len(valid_files)} valid files in split")

    # Find exact hash collisions
    if hash_method == "md5":
        for file_hash, image_ids in hash_map.items():
            if len(image_ids) > 1:
                # All pairs with this hash are duplicates
                for i in range(len(image_ids)):
                    for j in range(i + 1, len(image_ids)):
                        duplicates.append(
                            {
                                "img1": image_ids[i],
                                "img2": image_ids[j],
                                "distance": 0,  # Exact match
                            }
                        )

    # Find perceptual hash near-duplicates
    elif hash_method == "phash":
        for i in range(len(valid_files)):
            for j in range(i + 1, len(valid_files)):
                img1_id, _, hash1 = valid_files[i]
                img2_id, _, hash2 = valid_files[j]

                distance = hamming_distance(hash1, hash2)

                if distance <= threshold:
                    duplicates.append(
                        {"img1": img1_id, "img2": img2_id, "distance": int(distance)}
                    )

    return duplicates


def load_manifest(manifest_path):
    """Load image manifest CSV."""
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(df)} rows")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate images using perceptual hashing"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/images.csv",
        help="Path to image manifest CSV",
    )
    parser.add_argument(
        "--splits-dir", type=str, default="data/splits", help="Path to splits directory"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Hamming distance threshold for perceptual hash similarity",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/duplicates.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--hash-method",
        type=str,
        choices=["md5", "phash"],
        default="phash",
        help="Hash method: md5 for exact matching, phash for perceptual",
    )

    args = parser.parse_args()

    # Check imagehash availability for phash
    if args.hash_method == "phash" and not HAS_IMAGEHASH:
        logger.error("imagehash library not found. Install with: pip install imagehash")
        raise ImportError("imagehash is required for perceptual hashing")

    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = load_manifest(manifest_path)

    # Group by split
    splits = df.get("split", None)
    if splits is None or (splits == "unset").all():
        logger.warning("No split assignment in manifest. Using all images as 'train'.")
        split_groups = {"train": df.to_dict("records")}
    else:
        split_groups = {}
        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]
            if len(split_df) > 0:
                split_groups[split_name] = split_df.to_dict("records")

    logger.info(f"Found {len(split_groups)} splits: {list(split_groups.keys())}")

    # Detect duplicates per split
    report = {}
    total_duplicates = 0

    for split_name, image_rows in split_groups.items():
        logger.info(f"\nProcessing split: {split_name} ({len(image_rows)} images)")

        duplicates = find_duplicates_in_split(
            image_rows,
            args.splits_dir,
            threshold=args.threshold,
            hash_method=args.hash_method,
        )

        report[split_name] = duplicates
        total_duplicates += len(duplicates)

        logger.info(f"  Found {len(duplicates)} duplicate pairs")

        # Show first few
        if duplicates:
            logger.info("  Sample duplicates:")
            for dup in duplicates[:3]:
                logger.info(
                    f"    {dup['img1']} <-> {dup['img2']} (distance: {dup['distance']})"
                )

    # Write report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "hash_method": args.hash_method,
        "threshold": args.threshold,
        "duplicates": report,
        "summary": {split_name: len(dups) for split_name, dups in report.items()},
        "total_duplicates": total_duplicates,
    }

    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Duplicate Detection Report")
    logger.info(f"{'=' * 60}")
    logger.info(f"Hash Method: {args.hash_method}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Total Duplicates: {total_duplicates}")
    logger.info(f"\nPer-split Summary:")
    for split_name, count in sorted(report_data["summary"].items()):
        logger.info(f"  {split_name:8s}: {count:4d} duplicate pairs")
    logger.info(f"\nReport written to: {output_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
