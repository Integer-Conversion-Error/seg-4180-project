#!/usr/bin/env python3
"""
Split dataset into train/val/test with group-aware splitting.
Groups by pano_id or location+heading to prevent data leakage.

Includes threshold validation and rebalancing for enforcement classes.
"""

import argparse
import csv
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from utils.dataset_exclusion import load_dataset_excluded_ids


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# Threshold Configuration
# ============================================================================
MIN_INSTANCES = {
    "enforcement_vehicle": {"val": 50, "test": 30},
    "police_old": {"val": 10, "test": 5},
    "police_new": {"val": 10, "test": 5},
}

MIN_PANO_GROUPS = {"val": 15, "test": 8}


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def get_group_key(row: dict) -> str:
    pano_id = row.get("pano_id", "")
    if pano_id:
        return f"pano:{pano_id}"

    lat = row.get("pano_lat", "")
    lng = row.get("pano_lng", "")
    heading = row.get("heading", "")

    try:
        lat_rounded = round(float(lat), 4) if lat else 0
        lng_rounded = round(float(lng), 4) if lng else 0
        heading_bucket = (int(heading) // 90) * 90 if heading else 0
        return f"loc:{lat_rounded},{lng_rounded},h{heading_bucket}"
    except (ValueError, TypeError):
        return f"loc:{lat},{lng}"


def is_rejected(row: dict) -> bool:
    return (row.get("review_status") or "").strip().lower() == "rejected"


def count_class_instances(manifest: list, split_name: str, class_name: str) -> int:
    """Count instances of a class in a given split."""
    count = 0
    for row in manifest:
        if row.get("split") != split_name:
            continue
        expected_class = row.get("expected_inserted_class", "")
        if expected_class == class_name:
            count += 1
    return count


def count_pano_groups_with_class(
    manifest: list, split_name: str, class_name: str
) -> int:
    """Count unique pano groups containing a class in a given split."""
    pano_groups = set()
    for row in manifest:
        if row.get("split") != split_name:
            continue
        expected_class = row.get("expected_inserted_class", "")
        if expected_class == class_name:
            group_key = get_group_key(row)
            pano_groups.add(group_key)
    return len(pano_groups)


def validate_split_thresholds(manifest: list, split_name: str) -> list:
    """Validate that a split meets minimum instance and pano-group thresholds.

    Returns:
        List of violations as strings, empty if all thresholds met.
    """
    violations = []

    if split_name not in ["val", "test"]:
        return violations  # Only validate val and test

    min_instances = MIN_INSTANCES.get(split_name, {})
    min_pano_groups = MIN_PANO_GROUPS.get(split_name, 0)

    logger.info(f"Validating {split_name} split thresholds...")

    # Check per-class instance thresholds
    for class_name in ["enforcement_vehicle", "police_old", "police_new"]:
        instances = count_class_instances(manifest, split_name, class_name)
        required = min_instances.get(class_name, 0)
        if instances < required:
            violation = f"{class_name}: {instances} instances < {required} required"
            violations.append(violation)
            logger.warning(f"  VIOLATION: {violation}")
        else:
            logger.info(f"  ✓ {class_name}: {instances} instances >= {required}")

    # Check per-class pano-group thresholds
    for class_name in ["enforcement_vehicle", "police_old", "police_new"]:
        groups = count_pano_groups_with_class(manifest, split_name, class_name)
        if groups < min_pano_groups:
            violation = f"{class_name}_pano_groups: {groups} groups < {min_pano_groups} required"
            violations.append(violation)
            logger.warning(f"  VIOLATION: {violation}")
        else:
            logger.info(
                f"  ✓ {class_name}_pano_groups: {groups} groups >= {min_pano_groups}"
            )

    return violations


def get_enforcement_groups(manifest: list, split_name: str) -> dict:
    """Get all groups in a split that contain enforcement classes.

    Returns:
        Dict mapping group_key -> list of rows in that group with enforcement.
    """
    enforcement_groups = defaultdict(list)
    enforcement_classes = {"enforcement_vehicle", "police_old", "police_new"}

    for row in manifest:
        if row.get("split") != split_name:
            continue
        if row.get("expected_inserted_class") in enforcement_classes:
            group_key = get_group_key(row)
            enforcement_groups[group_key].append(row)

    return enforcement_groups


def rebalance_splits(manifest: list, violations: list) -> int:
    """Attempt to rebalance splits by moving enforcement groups from train.

    Moves enforcement-positive groups from train to val/test to meet thresholds.
    Maintains approximate 70/20/10 ratio where possible.

    Args:
        manifest: List of image records with split assignments
        violations: List of violation strings from validation

    Returns:
        Number of groups moved.
    """
    if not violations:
        logger.info("No threshold violations detected.")
        return 0

    logger.warning(f"\n{'=' * 70}")
    logger.warning(f"REBALANCING: {len(violations)} threshold violations detected")
    logger.warning(f"{'=' * 70}")

    groups_moved = 0

    # Get enforcement groups from train split
    train_enforcement = get_enforcement_groups(manifest, "train")
    logger.info(f"Found {len(train_enforcement)} enforcement groups in train split")

    # Try to fix val and test violations
    for split_target in ["val", "test"]:
        current_violations = validate_split_thresholds(manifest, split_target)
        if not current_violations:
            logger.info(f"{split_target} split meets all thresholds, skipping")
            continue

        logger.info(
            f"\nAttempting to fix {len(current_violations)} violations in {split_target}..."
        )

        # Extract which classes are missing in this split
        for violation in current_violations:
            if "pano_groups" in violation:
                continue  # Skip pano group violations for now, focus on instances

            # Parse violation like "enforcement_vehicle: 5 instances < 30 required"
            parts = violation.split(":")
            if len(parts) < 2:
                continue
            class_name = parts[0].strip()

            # Try moving enforcement groups from train to this split
            for group_key, rows in sorted(train_enforcement.items()):
                # Skip if group already partially in another split
                splits_in_group = set(r.get("split") for r in rows)
                if len(splits_in_group) > 1:
                    logger.debug(f"Skipping group {group_key}: spans multiple splits")
                    continue

                # Move entire group to target split
                group_class = rows[0].get("expected_inserted_class")
                moved_count = 0
                for row in rows:
                    if row.get("split") == "train":
                        row["split"] = split_target
                        moved_count += 1

                if moved_count > 0:
                    groups_moved += 1
                    logger.info(
                        f"  Moved group {group_key} ({group_class}) with {moved_count} "
                        f"images from train → {split_target}"
                    )
                    # Re-check violations after move
                    current_violations = validate_split_thresholds(
                        manifest, split_target
                    )
                    if not any(
                        "pano_groups" not in v and class_name in v
                        for v in current_violations
                    ):
                        logger.info(
                            f"  {class_name} threshold now met in {split_target}"
                        )
                        break

    logger.warning(f"\nRebalancing complete: Moved {groups_moved} groups")
    logger.warning(f"{'=' * 70}\n")

    return groups_moved


def ensure_synthetic_pano_inheritance(manifest: list) -> list:
    """For synthetic images, inherit pano_id from parent_image_id.

    If a synthetic image has parent_image_id set and no pano_id,
    copy pano_id from the parent image.

    Args:
        manifest: List of image records

    Returns:
        Updated manifest with pano_id inheritance applied
    """
    # Build lookup map: image_id -> row
    image_lookup = {row["image_id"]: row for row in manifest}

    updated_count = 0
    inherited_count_by_class = defaultdict(int)

    for row in manifest:
        parent_id = row.get("parent_image_id", "")
        if not parent_id:
            continue

        # This is a synthetic image
        current_pano = row.get("pano_id", "")
        if current_pano:
            continue  # Already has pano_id

        # Get parent's pano_id
        parent_row = image_lookup.get(parent_id)
        if not parent_row:
            logger.debug(
                f"Parent image {parent_id} not found for synthetic {row['image_id']}"
            )
            continue

        parent_pano = parent_row.get("pano_id", "")
        if parent_pano:
            row["pano_id"] = parent_pano
            updated_count += 1
            # Track by class for detailed logging
            class_name = row.get("expected_inserted_class", "unknown")
            inherited_count_by_class[class_name] += 1

    if updated_count > 0:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"PANO_ID INHERITANCE SUMMARY")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total synthetic images with inherited pano_id: {updated_count}")
        logger.info(f"")
        logger.info(f"Breakdown by class:")
        for class_name in sorted(inherited_count_by_class.keys()):
            count = inherited_count_by_class[class_name]
            logger.info(f"  {class_name}: {count}")
        logger.info(f"{'=' * 70}")
    else:
        logger.info("No synthetic images found or all already had pano_id assigned")

    return manifest


def _is_dataset_excluded(row: dict, excluded_ids: set[str]) -> bool:
    return (row.get("image_id") or "").strip() in excluded_ids


def validate_no_leakage(manifest: list) -> list:
    """Validate that no pano_id group appears in multiple splits.

    Groups by pano_id (or location+heading fallback). Checks that
    each group appears in only one split (train, val, or test).

    Args:
        manifest: List of image records with 'split' assigned

    Returns:
        List of leaking group_keys (empty if no leakage detected)
    """
    # Map group_key -> set of splits
    group_splits = defaultdict(set)

    for row in manifest:
        split = row.get("split", "")
        if not split:
            continue

        group_key = get_group_key(row)
        group_splits[group_key].add(split)

    # Find groups that appear in multiple splits
    leaking_groups = []
    for group_key, splits in group_splits.items():
        if len(splits) > 1:
            leaking_groups.append(group_key)
            logger.error(
                f"Leakage detected: {group_key} appears in splits: {', '.join(sorted(splits))}"
            )

    if leaking_groups:
        logger.warning(f"Total leaking groups: {len(leaking_groups)}")
    else:
        logger.info("✓ No leakage detected: all groups in single splits")

    return leaking_groups


def main():
    parser = argparse.ArgumentParser(description="Split dataset with group awareness")
    parser.add_argument(
        "--manifest", "-m", default="manifests/images.csv", help="Input manifest"
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/splits", help="Output directory"
    )
    parser.add_argument(
        "--labels-dir", "-l", default="data/labels_autogen", help="Labels directory"
    )
    parser.add_argument(
        "--labels-final-dir",
        default="data/labels_final",
        help="Labels override directory (human-reviewed, preferred over labels-dir)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Val ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-threshold-check",
        action="store_true",
        help="Skip validation of enforcement class thresholds",
    )
    parser.add_argument(
        "--min-enforcement-val",
        type=int,
        default=50,
        help="Minimum enforcement_vehicle instances required in val set (default: 50)",
    )
    parser.add_argument(
        "--min-enforcement-test",
        type=int,
        default=30,
        help="Minimum enforcement_vehicle instances required in test set (default: 30)",
    )
    parser.add_argument(
        "--validate-leakage",
        action="store_true",
        default=True,
        help="Validate that no pano groups appear in multiple splits (default: True)",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include images with review_status=rejected (default: exclude)",
    )
    args = parser.parse_args()

    # Parse custom threshold overrides
    if args.min_enforcement_val:
        MIN_INSTANCES["enforcement_vehicle"]["val"] = args.min_enforcement_val
    if args.min_enforcement_test:
        MIN_INSTANCES["enforcement_vehicle"]["test"] = args.min_enforcement_test

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    labels_dir = Path(args.labels_dir)
    labels_final_dir = Path(args.labels_final_dir)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        logger.error("Ratios must sum to 1.0")
        return

    manifest = load_manifest(manifest_path)
    excluded_ids = load_dataset_excluded_ids(manifest_path)

    # ========================================================================
    # Apply Synthetic Pano Inheritance
    # ========================================================================
    manifest = ensure_synthetic_pano_inheritance(manifest)

    valid_images = []
    for row in manifest:
        if not args.include_rejected and is_rejected(row):
            row["split"] = ""
            continue
        if row.get("status") != "ok":
            continue
        if _is_dataset_excluded(row, excluded_ids):
            row["split"] = ""
            continue
        if not Path(row.get("file_path", "")).exists():
            continue
        valid_images.append(row)

    groups = defaultdict(list)
    for row in valid_images:
        group_key = get_group_key(row)
        groups[group_key].append(row)

    logger.info(f"Found {len(valid_images)} valid images in {len(groups)} groups")

    random.seed(args.seed)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = int(n_groups * args.train_ratio)
    n_val = int(n_groups * args.val_ratio)

    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train : n_train + n_val])
    test_groups = set(group_keys[n_train + n_val :])

    split_map = {}
    for g in train_groups:
        for row in groups[g]:
            split_map[row["image_id"]] = "train"
    for g in val_groups:
        for row in groups[g]:
            split_map[row["image_id"]] = "val"
    for g in test_groups:
        for row in groups[g]:
            split_map[row["image_id"]] = "test"

    split_counts = defaultdict(int)
    enforcement_in_test = 0

    for row in valid_images:
        split = split_map.get(row["image_id"], "train")
        row["split"] = split
        split_counts[split] += 1

        if (
            split == "test"
            and row.get("expected_inserted_class") == "enforcement_vehicle"
        ):
            enforcement_in_test += 1

    save_manifest(manifest, manifest_path)

    # ========================================================================
    # Validate and Rebalance Thresholds
    # ========================================================================
    if not args.skip_threshold_check:
        logger.info("\n" + "=" * 70)
        logger.info("THRESHOLD VALIDATION")
        logger.info("=" * 70)

        all_violations = []
        for split in ["val", "test"]:
            violations = validate_split_thresholds(valid_images, split)
            all_violations.extend([(split, v) for v in violations])

        if all_violations:
            logger.warning(f"\nTotal violations detected: {len(all_violations)}")
            groups_moved = rebalance_splits(valid_images, all_violations)
            if groups_moved > 0:
                logger.info(f"Rechecking violations after rebalancing...")
                # Recount splits after rebalancing
                split_counts = defaultdict(int)
                for row in valid_images:
                    split = row.get("split", "train")
                    split_counts[split] += 1
                logger.info("Updated split counts after rebalancing:")
                for split, count in sorted(split_counts.items()):
                    logger.info(f"  {split}: {count}")
                # Save updated manifest
                save_manifest(manifest, manifest_path)
        else:
            logger.info("\nAll threshold validation checks passed!")
    else:
        logger.info("Threshold validation skipped (--skip-threshold-check flag set)")

    # ========================================================================
    # Validate Leakage
    # ========================================================================
    if args.validate_leakage:
        logger.info("\n" + "=" * 70)
        logger.info("LEAKAGE VALIDATION")
        logger.info("=" * 70)
        leaking_groups = validate_no_leakage(valid_images)
        if leaking_groups:
            logger.error(
                f"\nCritical: {len(leaking_groups)} groups appear in multiple splits!"
            )
            logger.error("This will cause data leakage. Please review the splits.")
        else:
            logger.info("\n✓ Leakage validation passed!")
    else:
        logger.info("Leakage validation skipped (--validate-leakage flag set to False)")

    for split in ["train", "val", "test"]:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        if img_dir.parent.exists():
            shutil.rmtree(img_dir.parent)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(valid_images, desc="Copying files"):
        split = row["split"]
        image_id = row["image_id"]

        src_img = Path(row["file_path"])
        dst_img = out_dir / split / "images" / f"{image_id}.jpg"

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Prefer labels_final (human-reviewed) over labels_autogen
        src_lbl_final = labels_final_dir / f"{image_id}.txt"
        src_lbl_auto = labels_dir / f"{image_id}.txt"
        if src_lbl_final.exists() and src_lbl_final.read_text().strip():
            src_lbl = src_lbl_final
        else:
            src_lbl = src_lbl_auto
        dst_lbl = out_dir / split / "labels" / f"{image_id}.txt"

        if src_lbl.exists() and not dst_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

    logger.info(f"Split complete:")
    for split, count in sorted(split_counts.items()):
        logger.info(f"  {split}: {count}")

    if enforcement_in_test == 0:
        logger.warning("=" * 60)
        logger.warning("WARNING: No enforcement_vehicle examples in test set!")
        logger.warning("Consider rebalancing or increasing enforcement rate.")
        logger.warning("=" * 60)
    else:
        logger.info(f"Test set has {enforcement_in_test} enforcement_vehicle examples")

    logger.info("\nDataset split to:")
    logger.info(f"  {out_dir}")
    logger.info("Run 'python scripts/07_train_yolo.py' to train the model.")


if __name__ == "__main__":
    main()
