#!/usr/bin/env python3
"""
Unified cleanup of all auto-generated YOLO bounding box labels.

Executes 6 steps in order:
  Step 1: Register 2,573 missing synth images in manifests/images.csv
  Step 2: Remove sentinel boxes (full-image fallback) from synth label files
  Step 3: Fix sentinel-only-target files (reassign target class to best class-0 box)
  Step 4: Resolve cross-class overlaps (IoU >= 0.8, same vehicle detected twice)
  Step 5: Deduplicate same-class overlaps (NMS artifacts)
  Step 6: Merge 12 non-empty labels_final overrides into labels_autogen

Usage:
    python scripts/cleanup_autogen_labels.py [--dry-run]
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.dataset_exclusion import load_dataset_excluded_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABELS_AUTOGEN = Path("data/labels_autogen")
LABELS_FINAL = Path("data/labels_final")
IMAGES_SYNTH = Path("data/images_synth")
MANIFEST_PATH = Path("manifests/images.csv")

EDIT_TYPE_TO_CLASS = {
    "enforcement_vehicle": 1,
    "police_old": 2,
    "police_new": 3,
    "random_vehicle": 0,
}

# Class name as it appears in the expected_inserted_class manifest column
# random_vehicle maps to "vehicle" class name but edit_type stays random_vehicle
EDIT_TYPE_TO_CLASS_NAME = {
    "enforcement_vehicle": "enforcement_vehicle",
    "police_old": "police_old",
    "police_new": "police_new",
    "random_vehicle": "random_vehicle",
}

# Sentinel detection thresholds
SENTINEL_XC_TOL = 0.05  # xc within this of 0.5
SENTINEL_MIN_WIDTH = 0.90  # w >= this
SENTINEL_MIN_AREA = 0.50  # w*h >= this

# Overlap thresholds
OVERLAP_IOU_THRESH = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_label(path: Path) -> list[list[str]]:
    """Read a YOLO label file. Returns list of raw line strings (stripped)."""
    text = path.read_text().strip()
    if not text:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_box(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse a single YOLO line into (cls, xc, yc, w, h)."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        return (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )
    except (ValueError, IndexError):
        return None


def format_box(cls: int, xc: float, yc: float, w: float, h: float) -> str:
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def write_label(path: Path, lines: list[str], dry_run: bool = False):
    """Write lines back to a label file."""
    if dry_run:
        return
    path.write_text("\n".join(lines) + "\n" if lines else "")


def iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two parsed boxes (cls, xc, yc, w, h)."""
    _, axc, ayc, aw, ah = a
    _, bxc, byc, bw, bh = b

    ax1, ay1 = axc - aw / 2, ayc - ah / 2
    ax2, ay2 = axc + aw / 2, ayc + ah / 2
    bx1, by1 = bxc - bw / 2, byc - bh / 2
    bx2, by2 = bxc + bw / 2, byc + bh / 2

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_sentinel(box: tuple) -> bool:
    """Check if a parsed box matches sentinel/full-image pattern."""
    _, xc, _yc, w, h = box
    area = w * h
    return (
        abs(xc - 0.5) <= SENTINEL_XC_TOL
        and w >= SENTINEL_MIN_WIDTH
        and area >= SENTINEL_MIN_AREA
    )


def extract_edit_type(filename: str) -> str | None:
    """Extract edit type from synth filename like '0134411e07b7_enforcement_vehicle.txt'.

    Returns the edit type string (e.g. 'enforcement_vehicle') or None if not a synth file.
    """
    stem = filename.replace(".txt", "")
    for edit_type in EDIT_TYPE_TO_CLASS:
        if stem.endswith(f"_{edit_type}"):
            return edit_type
    return None


def extract_parent_id(filename: str) -> str | None:
    """Extract parent image ID from synth filename."""
    stem = filename.replace(".txt", "").replace(".jpg", "")
    for edit_type in EDIT_TYPE_TO_CLASS:
        suffix = f"_{edit_type}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


# ---------------------------------------------------------------------------
# Step 1: Register missing synth images in manifest
# ---------------------------------------------------------------------------
def step1_register_synth_images(manifest_rows: list[dict], dry_run: bool) -> list[dict]:
    """Add rows for synth images missing from the manifest."""
    log.info("=" * 70)
    log.info("STEP 1: Register missing synth images in manifest")
    log.info("=" * 70)

    # Build set of existing image_ids
    existing_ids = {row["image_id"] for row in manifest_rows}
    excluded_ids = load_dataset_excluded_ids(MANIFEST_PATH)

    # Build parent lookup: image_id -> row
    parent_lookup = {row["image_id"]: row for row in manifest_rows}

    # Columns to inherit from parent
    GEO_COLS = [
        "street",
        "input_location",
        "heading",
        "pitch",
        "fov",
        "pano_id",
        "pano_lat",
        "pano_lng",
    ]

    # Get all manifest column names from first row
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []

    new_rows = []
    missing_parents = 0
    already_registered = 0

    # Walk synth image directories
    for edit_type in sorted(EDIT_TYPE_TO_CLASS.keys()):
        synth_dir = IMAGES_SYNTH / edit_type
        if not synth_dir.is_dir():
            log.warning(f"  Synth directory not found: {synth_dir}")
            continue

        for img_path in sorted(synth_dir.glob("*.jpg")):
            image_id = img_path.stem  # e.g. 0134411e07b7_enforcement_vehicle
            if image_id in excluded_ids:
                continue
            if image_id in existing_ids:
                already_registered += 1
                continue

            parent_id = extract_parent_id(img_path.name)
            if not parent_id or parent_id not in parent_lookup:
                missing_parents += 1
                log.debug(f"  Parent not found for {image_id} (parent_id={parent_id})")
                continue

            parent_row = parent_lookup[parent_id]

            # Build new manifest row with all columns
            new_row = {col: "" for col in fieldnames}
            new_row["image_id"] = image_id
            new_row["file_path"] = str(img_path).replace("/", "\\")
            new_row["split"] = "unset"
            new_row["parent_image_id"] = parent_id
            new_row["is_synthetic"] = "1"
            new_row["edit_type"] = edit_type
            new_row["expected_inserted_class"] = edit_type
            new_row["status"] = "ok"
            new_row["created_at"] = datetime.now(timezone.utc).isoformat()

            # Inherit geo columns from parent
            for col in GEO_COLS:
                new_row[col] = parent_row.get(col, "")

            new_rows.append(new_row)
            existing_ids.add(image_id)

    log.info(f"  Already registered: {already_registered}")
    log.info(f"  New rows to add:    {len(new_rows)}")
    log.info(f"  Missing parents:    {missing_parents}")

    if new_rows and not dry_run:
        manifest_rows.extend(new_rows)
        log.info(
            f"  Added {len(new_rows)} rows to manifest (total: {len(manifest_rows)})"
        )
    elif new_rows:
        log.info(f"  [DRY RUN] Would add {len(new_rows)} rows")

    return manifest_rows


# ---------------------------------------------------------------------------
# Step 2: Remove sentinel boxes from synth label files
# ---------------------------------------------------------------------------
def step2_remove_sentinels(dry_run: bool) -> dict:
    """Remove sentinel/full-image boxes from synth label files.

    Returns dict with stats and the set of files that had their target class
    ONLY on the sentinel (needed for Step 3).
    """
    log.info("")
    log.info("=" * 70)
    log.info("STEP 2: Remove sentinel boxes from synth label files")
    log.info("=" * 70)

    stats = {
        "files_scanned": 0,
        "sentinels_removed": 0,
        "files_modified": 0,
        "sentinel_only_target": [],  # files where target class was ONLY on sentinel
    }

    for label_path in sorted(LABELS_AUTOGEN.glob("*.txt")):
        edit_type = extract_edit_type(label_path.name)
        if edit_type is None:
            continue  # Not a synth file

        stats["files_scanned"] += 1
        lines = parse_label(label_path)
        if not lines:
            continue

        # Parse all boxes
        boxes = [parse_box(line) for line in lines]
        if not boxes or boxes[-1] is None:
            continue

        # Check if last line is a sentinel
        last_box = boxes[-1]
        if not is_sentinel(last_box):
            continue

        # Sentinel found. Check if target class exists on non-sentinel boxes
        target_cls = EDIT_TYPE_TO_CLASS[edit_type]
        non_sentinel_lines = lines[:-1]
        non_sentinel_boxes = boxes[:-1]

        has_target_on_real_box = any(
            b is not None and b[0] == target_cls for b in non_sentinel_boxes
        )

        if not has_target_on_real_box:
            stats["sentinel_only_target"].append(label_path.name)

        # Remove sentinel (last line)
        write_label(label_path, non_sentinel_lines, dry_run)
        stats["sentinels_removed"] += 1
        stats["files_modified"] += 1

    log.info(f"  Synth files scanned:    {stats['files_scanned']}")
    log.info(f"  Sentinels removed:      {stats['sentinels_removed']}")
    log.info(f"  Files modified:         {stats['files_modified']}")
    log.info(f"  Sentinel-only-target:   {len(stats['sentinel_only_target'])}")

    return stats


# ---------------------------------------------------------------------------
# Step 3: Fix sentinel-only-target files
# ---------------------------------------------------------------------------
def step3_fix_sentinel_only_target(
    sentinel_only_files: list[str], dry_run: bool
) -> int:
    """For files where target class was only on sentinel, reassign target class
    to the best class-0 box (largest area, most central).
    """
    log.info("")
    log.info("=" * 70)
    log.info("STEP 3: Fix sentinel-only-target files (reassign target class)")
    log.info("=" * 70)

    fixed = 0

    for fname in sentinel_only_files:
        label_path = LABELS_AUTOGEN / fname
        edit_type = extract_edit_type(fname)
        if edit_type is None:
            continue

        target_cls = EDIT_TYPE_TO_CLASS[edit_type]
        lines = parse_label(label_path)
        if not lines:
            log.warning(f"  {fname}: no lines after sentinel removal, skipping")
            continue

        boxes = [parse_box(line) for line in lines]

        # Find class-0 boxes and pick best one (largest area, then most central)
        candidates = []
        for i, box in enumerate(boxes):
            if box is None:
                continue
            cls, xc, yc, w, h = box
            if cls == 0:
                area = w * h
                # Centrality: distance from image center (0.5, 0.5)
                centrality = ((xc - 0.5) ** 2 + (yc - 0.5) ** 2) ** 0.5
                candidates.append((i, area, centrality, box))

        if not candidates:
            log.warning(f"  {fname}: no class-0 boxes to reassign, skipping")
            continue

        # Sort by area descending, then centrality ascending (closer to center = better)
        candidates.sort(key=lambda c: (-c[1], c[2]))
        best_idx = candidates[0][0]
        best_box = candidates[0][3]

        # Reassign the best box to target class
        _, xc, yc, w, h = best_box
        lines[best_idx] = format_box(target_cls, xc, yc, w, h)

        write_label(label_path, lines, dry_run)
        fixed += 1

    log.info(f"  Files fixed: {fixed} / {len(sentinel_only_files)}")
    return fixed


# ---------------------------------------------------------------------------
# Step 4: Resolve cross-class overlaps
# ---------------------------------------------------------------------------
def step4_resolve_cross_class_overlaps(dry_run: bool) -> int:
    """For pairs where the same vehicle is detected as both class 0 and target class
    (IoU >= threshold), remove the class-0 duplicate.
    """
    log.info("")
    log.info("=" * 70)
    log.info("STEP 4: Resolve cross-class overlaps")
    log.info("=" * 70)

    total_removed = 0
    files_modified = 0

    for label_path in sorted(LABELS_AUTOGEN.glob("*.txt")):
        lines = parse_label(label_path)
        if len(lines) < 2:
            continue

        boxes = [parse_box(line) for line in lines]

        # Find cross-class overlapping pairs
        indices_to_remove = set()
        for i in range(len(boxes)):
            if boxes[i] is None:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j] is None:
                    continue

                cls_i, cls_j = boxes[i][0], boxes[j][0]

                # Only care about cross-class: one is 0, the other is non-zero
                if cls_i == cls_j:
                    continue
                if cls_i != 0 and cls_j != 0:
                    continue

                iou_val = iou(boxes[i], boxes[j])
                if iou_val >= OVERLAP_IOU_THRESH:
                    # Remove the class-0 box
                    if cls_i == 0:
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)

        if indices_to_remove:
            new_lines = [
                line for idx, line in enumerate(lines) if idx not in indices_to_remove
            ]
            write_label(label_path, new_lines, dry_run)
            total_removed += len(indices_to_remove)
            files_modified += 1

    log.info(f"  Class-0 duplicates removed: {total_removed}")
    log.info(f"  Files modified:             {files_modified}")
    return total_removed


# ---------------------------------------------------------------------------
# Step 5: Deduplicate same-class overlaps
# ---------------------------------------------------------------------------
def step5_deduplicate_same_class(dry_run: bool) -> int:
    """For same-class box pairs with IoU >= threshold, keep the larger box."""
    log.info("")
    log.info("=" * 70)
    log.info("STEP 5: Deduplicate same-class overlaps (NMS artifacts)")
    log.info("=" * 70)

    total_removed = 0
    files_modified = 0

    for label_path in sorted(LABELS_AUTOGEN.glob("*.txt")):
        lines = parse_label(label_path)
        if len(lines) < 2:
            continue

        boxes = [parse_box(line) for line in lines]

        # Greedy NMS-style dedup: mark smaller box for removal
        indices_to_remove = set()
        for i in range(len(boxes)):
            if boxes[i] is None or i in indices_to_remove:
                continue
            for j in range(i + 1, len(boxes)):
                if boxes[j] is None or j in indices_to_remove:
                    continue

                # Same class only
                if boxes[i][0] != boxes[j][0]:
                    continue

                iou_val = iou(boxes[i], boxes[j])
                if iou_val >= OVERLAP_IOU_THRESH:
                    # Keep larger area
                    area_i = boxes[i][3] * boxes[i][4]
                    area_j = boxes[j][3] * boxes[j][4]
                    if area_j > area_i:
                        indices_to_remove.add(i)
                    else:
                        indices_to_remove.add(j)

        if indices_to_remove:
            new_lines = [
                line for idx, line in enumerate(lines) if idx not in indices_to_remove
            ]
            write_label(label_path, new_lines, dry_run)
            total_removed += len(indices_to_remove)
            files_modified += 1

    log.info(f"  Duplicate boxes removed: {total_removed}")
    log.info(f"  Files modified:          {files_modified}")
    return total_removed


# ---------------------------------------------------------------------------
# Step 6: Merge labels_final overrides into labels_autogen
# ---------------------------------------------------------------------------
def step6_merge_labels_final(dry_run: bool) -> int:
    """Copy non-empty labels_final files into labels_autogen (human review overrides autogen)."""
    log.info("")
    log.info("=" * 70)
    log.info("STEP 6: Merge labels_final overrides into labels_autogen")
    log.info("=" * 70)

    if not LABELS_FINAL.is_dir():
        log.warning(f"  labels_final directory not found: {LABELS_FINAL}")
        return 0

    excluded_ids = load_dataset_excluded_ids(MANIFEST_PATH)

    merged = 0
    for final_path in sorted(LABELS_FINAL.glob("*.txt")):
        if final_path.stem in excluded_ids:
            continue
        content = final_path.read_text().strip()
        if not content:
            continue

        autogen_path = LABELS_AUTOGEN / final_path.name
        if not dry_run:
            autogen_path.write_text(content + "\n")
        merged += 1
        log.info(f"  Merged: {final_path.name} ({len(content.splitlines())} boxes)")

    log.info(f"  Total overrides merged: {merged}")
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified cleanup of auto-generated YOLO labels"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files",
    )
    args = parser.parse_args()

    if args.dry_run:
        log.info("*** DRY RUN MODE — no files will be modified ***\n")

    # Validate paths
    if not LABELS_AUTOGEN.is_dir():
        log.error(f"Labels directory not found: {LABELS_AUTOGEN}")
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        log.error(f"Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    # Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    log.info(f"Loaded manifest: {len(manifest_rows)} rows\n")

    # ---- Step 1 ----
    manifest_rows = step1_register_synth_images(manifest_rows, args.dry_run)

    # Save manifest
    if not args.dry_run and manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        log.info(f"  Manifest saved: {len(manifest_rows)} rows\n")

    # ---- Step 2 ----
    step2_stats = step2_remove_sentinels(args.dry_run)

    # ---- Step 3 ----
    step3_fix_sentinel_only_target(step2_stats["sentinel_only_target"], args.dry_run)

    # ---- Step 4 ----
    step4_resolve_cross_class_overlaps(args.dry_run)

    # ---- Step 5 ----
    step5_deduplicate_same_class(args.dry_run)

    # ---- Step 6 ----
    step6_merge_labels_final(args.dry_run)

    # ---- Summary ----
    log.info("")
    log.info("=" * 70)
    log.info("CLEANUP COMPLETE")
    log.info("=" * 70)
    if args.dry_run:
        log.info("This was a dry run. Re-run without --dry-run to apply changes.")
    else:
        log.info("All 6 steps completed. Run the audit script to verify:")
        log.info("  python scripts/audit_autogen_labels.py")


if __name__ == "__main__":
    main()
