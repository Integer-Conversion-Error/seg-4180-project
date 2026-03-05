#!/usr/bin/env python3
"""
Crop fetched Street View images from the bottom by a fixed pixel amount.

Behavior:
- Reads `manifests/images.csv`
- For eligible rows (`status=ok`), creates a sibling cropped file with suffix `_bc{N}`
- Updates `file_path` to point at the cropped file
- Keeps originals intact
- Skips images already cropped to prevent re-cropping

Default crop: 30px from bottom.
"""

import argparse
import csv
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from utils.preprocessing import crop_image_bottom


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

CROP_SUFFIX_RE = re.compile(r"_bc(?P<px>\d+)$")
EXTRA_COLUMNS = ["source_file_path", "crop_bottom_px", "crop_status", "cropped_at"]


def load_manifest(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return rows, columns


def save_manifest_atomic(rows: list[dict[str, str]], columns: list[str], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=str(manifest_path.parent))
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, manifest_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def build_cropped_path(source_path: Path, crop_px: int) -> Path:
    return source_path.with_name(f"{source_path.stem}_bc{crop_px}{source_path.suffix}")


def parse_existing_crop_px(path: Path) -> int | None:
    match = CROP_SUFFIX_RE.search(path.stem)
    if not match:
        return None
    return int(match.group("px"))


def is_ok_status(row: dict[str, str]) -> bool:
    return str(row.get("status") or "").strip().lower() == "ok"


def is_synthetic(row: dict[str, str]) -> bool:
    return str(row.get("is_synthetic") or "0").strip() == "1"



def main() -> int:
    parser = argparse.ArgumentParser(description="Crop Street View images from bottom")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Input/output image manifest")
    parser.add_argument("--crop-px", type=int, default=30, help="Bottom crop in pixels (default: 30)")
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also crop rows where is_synthetic=1 (default: originals only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.crop_px <= 0:
        parser.error("--crop-px must be > 0")

    manifest_path = Path(args.manifest)
    rows, columns = load_manifest(manifest_path)

    for column in EXTRA_COLUMNS:
        if column not in columns:
            columns.append(column)

    stats = {
        "rows_total": len(rows),
        "rows_eligible": 0,
        "rows_cropped": 0,
        "rows_skipped_already": 0,
        "rows_skipped_missing": 0,
        "rows_skipped_status": 0,
        "rows_skipped_synth": 0,
        "rows_failed": 0,
    }

    changed = False

    for row in rows:
        if not is_ok_status(row):
            stats["rows_skipped_status"] += 1
            continue

        if (not args.include_synthetic) and is_synthetic(row):
            stats["rows_skipped_synth"] += 1
            continue

        source_text = str(row.get("file_path") or "").strip()
        if not source_text:
            stats["rows_skipped_missing"] += 1
            continue

        source_path = Path(source_text)
        if not source_path.exists():
            stats["rows_skipped_missing"] += 1
            LOGGER.warning("Missing source image: %s", source_path)
            continue

        stats["rows_eligible"] += 1

        existing_crop_px = parse_existing_crop_px(source_path)
        if existing_crop_px is not None:
            stats["rows_skipped_already"] += 1
            row["crop_bottom_px"] = str(existing_crop_px)
            row["crop_status"] = "ok"
            continue

        target_path = build_cropped_path(source_path, args.crop_px)

        if target_path.exists():
            stats["rows_skipped_already"] += 1
            row["source_file_path"] = str(row.get("source_file_path") or source_path)
            row["file_path"] = str(target_path)
            row["crop_bottom_px"] = str(args.crop_px)
            row["crop_status"] = "ok"
            row["cropped_at"] = str(row.get("cropped_at") or "")
            changed = True
            continue

        if args.dry_run:
            stats["rows_cropped"] += 1
            continue

        ok, error = crop_image_bottom(source_path, target_path, args.crop_px)
        if not ok:
            stats["rows_failed"] += 1
            row["crop_status"] = f"failed:{error}"
            changed = True
            LOGGER.warning("Crop failed for %s: %s", source_path, error)
            continue

        stats["rows_cropped"] += 1
        row["source_file_path"] = str(row.get("source_file_path") or source_path)
        row["file_path"] = str(target_path)
        row["crop_bottom_px"] = str(args.crop_px)
        row["crop_status"] = "ok"
        row["cropped_at"] = datetime.now().isoformat()
        changed = True

    LOGGER.info("rows_total=%s", stats["rows_total"])
    LOGGER.info("rows_eligible=%s", stats["rows_eligible"])
    LOGGER.info("rows_cropped=%s", stats["rows_cropped"])
    LOGGER.info("rows_skipped_already=%s", stats["rows_skipped_already"])
    LOGGER.info("rows_skipped_missing=%s", stats["rows_skipped_missing"])
    LOGGER.info("rows_skipped_status=%s", stats["rows_skipped_status"])
    LOGGER.info("rows_skipped_synth=%s", stats["rows_skipped_synth"])
    LOGGER.info("rows_failed=%s", stats["rows_failed"])

    if args.dry_run:
        LOGGER.info("dry_run=true, manifest unchanged")
        return 0

    if changed:
        save_manifest_atomic(rows, columns, manifest_path)
        LOGGER.info("Updated manifest: %s", manifest_path)
    else:
        LOGGER.info("No manifest changes required")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
