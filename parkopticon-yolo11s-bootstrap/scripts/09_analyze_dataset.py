#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze dataset statistics and detect leakage.

Computes:
- Unique pano_id counts per split
- Heading distribution per pano
- Group statistics (images per group, synthetic ratios)
- Leakage detection (panos in multiple splits)
- Enforcement class distribution

Outputs JSON report and console summary.
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> list:
    """Load CSV manifest into list of dicts."""
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def analyze_pano_splits(manifest: list) -> dict:
    """
    Compute unique pano_id counts per split.

    Returns:
        dict: {split_name: set of pano_ids}
    """
    splits = defaultdict(set)

    for row in manifest:
        split = row.get("split", "unset")
        pano_id = row.get("pano_id", "")

        if pano_id and split != "unset":
            splits[split].add(pano_id)

    return {k: v for k, v in splits.items()}


def analyze_heading_distribution(manifest: list) -> dict:
    """
    Compute heading counts per pano.

    Returns:
        dict: {pano_id: {heading: count}}
    """
    pano_headings = defaultdict(lambda: defaultdict(int))

    for row in manifest:
        pano_id = row.get("pano_id", "")
        heading = row.get("heading", "")

        if pano_id and heading:
            try:
                heading_int = int(heading)
                pano_headings[pano_id][heading_int] += 1
            except (ValueError, TypeError):
                pass

    return {k: dict(v) for k, v in pano_headings.items()}


def analyze_group_stats(manifest: list) -> dict:
    """
    Compute images per group (pano_id), synthetic ratios, split assignment.

    Returns:
        dict: {pano_id: {
            'split': split_name,
            'total_images': count,
            'synthetic_images': count,
            'original_images': count,
            'synthetic_ratio': float,
            'edit_types': {edit_type: count},
            'expected_classes': {class: count}
        }}
    """
    group_stats = defaultdict(
        lambda: {
            "splits": set(),
            "total_images": 0,
            "synthetic_images": 0,
            "original_images": 0,
            "edit_types": defaultdict(int),
            "expected_classes": defaultdict(int),
        }
    )

    for row in manifest:
        pano_id = row.get("pano_id", "")
        split = row.get("split", "unset")
        is_synthetic = int(row.get("is_synthetic", 0))
        edit_type = row.get("edit_type", "none")
        expected_class = row.get("expected_inserted_class", "none")

        if pano_id:
            if split != "unset":
                group_stats[pano_id]["splits"].add(split)

            group_stats[pano_id]["total_images"] += 1

            if is_synthetic:
                group_stats[pano_id]["synthetic_images"] += 1
                if edit_type and edit_type != "none":
                    group_stats[pano_id]["edit_types"][edit_type] += 1
            else:
                group_stats[pano_id]["original_images"] += 1

            if expected_class and expected_class != "none":
                group_stats[pano_id]["expected_classes"][expected_class] += 1

    # Convert to JSON-serializable format
    result = {}
    for pano_id, stats in group_stats.items():
        total = stats["total_images"]
        synthetic = stats["synthetic_images"]
        synthetic_ratio = (synthetic / total) if total > 0 else 0.0

        result[pano_id] = {
            "splits": list(stats["splits"]),
            "total_images": total,
            "synthetic_images": synthetic,
            "original_images": stats["original_images"],
            "synthetic_ratio": round(synthetic_ratio, 3),
            "edit_types": dict(stats["edit_types"]),
            "expected_classes": dict(stats["expected_classes"]),
        }

    return result


def detect_leakage(group_stats: dict) -> list:
    """
    Detect groups (pano_ids) appearing in multiple splits.

    Returns:
        list of {pano_id, splits} that appear in multiple splits
    """
    leakage = []

    for pano_id, stats in group_stats.items():
        splits = stats["splits"]
        if len(splits) > 1:
            leakage.append(
                {
                    "pano_id": pano_id,
                    "splits": sorted(splits),
                    "total_images": stats["total_images"],
                }
            )

    return sorted(leakage, key=lambda x: x["pano_id"])


def compute_enforcement_distribution(manifest: list) -> dict:
    """
    Compute enforcement class distribution per split.

    Returns:
        dict: {split_name: {
            'total_images': count,
            'enforcement_instances': count,
            'by_class': {class_name: count}
        }}
    """
    distributions = defaultdict(
        lambda: {
            "total_images": 0,
            "enforcement_instances": 0,
            "by_class": defaultdict(int),
        }
    )

    for row in manifest:
        split = row.get("split", "unset")
        expected_class = row.get("expected_inserted_class", "none")

        distributions[split]["total_images"] += 1

        if expected_class in ["enforcement_vehicle", "police_old", "police_new"]:
            distributions[split]["enforcement_instances"] += 1
            distributions[split]["by_class"][expected_class] += 1

    # Convert defaultdicts to regular dicts
    result = {}
    for split, stats in distributions.items():
        result[split] = {
            "total_images": stats["total_images"],
            "enforcement_instances": stats["enforcement_instances"],
            "enforcement_ratio": round(
                stats["enforcement_instances"] / stats["total_images"]
                if stats["total_images"] > 0
                else 0.0,
                3,
            ),
            "by_class": dict(stats["by_class"]),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset for statistics and leakage"
    )
    parser.add_argument(
        "--manifest", default="manifests/images.csv", help="Input manifest CSV"
    )
    parser.add_argument(
        "--output", default="reports/dataset_analysis.json", help="Output JSON report"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    logger.info(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)

    # Filter valid images
    valid_manifest = [row for row in manifest if row.get("status") == "ok"]

    logger.info(f"Total images: {len(manifest)}")
    logger.info(f"Valid images (status='ok'): {len(valid_manifest)}")

    # Analyze pano splits
    logger.info("Analyzing pano_id distributions per split...")
    pano_splits = analyze_pano_splits(valid_manifest)

    # Analyze heading distribution
    logger.info("Analyzing heading distributions...")
    heading_distribution = analyze_heading_distribution(valid_manifest)

    # Analyze group statistics
    logger.info("Analyzing group statistics...")
    group_stats = analyze_group_stats(valid_manifest)

    # Detect leakage
    logger.info("Detecting leakage...")
    leakage = detect_leakage(group_stats)

    # Compute enforcement distribution
    logger.info("Computing enforcement distribution...")
    enforcement_dist = compute_enforcement_distribution(valid_manifest)

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "manifest_path": str(manifest_path),
        "summary": {
            "total_images": len(manifest),
            "valid_images": len(valid_manifest),
            "total_unique_panos": len(group_stats),
            "has_leakage": len(leakage) > 0,
            "leakage_count": len(leakage),
        },
        "panos_per_split": {
            split: len(pano_ids) for split, pano_ids in pano_splits.items()
        },
        "images_per_split": {
            split: sum(1 for row in valid_manifest if row.get("split") == split)
            for split in set(
                row.get("split") for row in valid_manifest if row.get("split")
            )
        },
        "enforcement_distribution": enforcement_dist,
        "leakage": leakage,
        "group_stats": group_stats,
        "heading_distribution": heading_distribution,
    }

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    logger.info(f"Writing report: {output_path}")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal Images: {report['summary']['total_images']}")
    print(f"Valid Images: {report['summary']['valid_images']}")
    print(f"Unique Panoramas: {report['summary']['total_unique_panos']}")

    print("\n--- Panos per Split ---")
    for split, count in sorted(report["panos_per_split"].items()):
        print(f"  {split}: {count}")

    print("\n--- Images per Split ---")
    for split, count in sorted(report["images_per_split"].items()):
        print(f"  {split}: {count}")

    print("\n--- Enforcement Distribution ---")
    for split, stats in sorted(enforcement_dist.items()):
        if split == "unset":
            continue
        instances = stats["enforcement_instances"]
        total = stats["total_images"]
        ratio = stats["enforcement_ratio"]
        print(f"  {split}:")
        print(f"    Total images: {total}")
        print(f"    Enforcement instances: {instances} ({ratio * 100:.1f}%)")
        if stats["by_class"]:
            for cls, count in sorted(stats["by_class"].items()):
                print(f"      {cls}: {count}")

    # Leakage warnings
    if leakage:
        print("\n" + "!" * 70)
        print("WARNING: LEAKAGE DETECTED!")
        print("!" * 70)
        print(f"Found {len(leakage)} panoramas in multiple splits:\n")
        for item in leakage[:10]:  # Show first 10
            pano_id = item["pano_id"]
            splits = item["splits"]
            total = item["total_images"]
            print(f"  {pano_id}:")
            print(f"    Splits: {', '.join(splits)}")
            print(f"    Total images: {total}")

        if len(leakage) > 10:
            print(f"\n  ... and {len(leakage) - 10} more")

        print("\nFIX: Re-run 06_split_dataset.py with updated grouping logic")
    else:
        print("\nOK: No leakage detected - all panos are in single splits only")

    print("\n" + "=" * 70)
    logger.info(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
