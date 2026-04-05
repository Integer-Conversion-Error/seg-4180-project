#!/usr/bin/env python3
"""Remove oversized YOLO boxes from synthetic-image labels.

A box is considered oversized when its normalized area (w * h) is greater
than the configured threshold (default: 0.40, i.e., 40% of image area).

This script only edits label text files. It never deletes image files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
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


def _is_rejected(row: dict) -> bool:
    return (row.get("review_status") or "").strip().lower() == "rejected"


def prune_massive_synth_boxes(
    manifest_path: Path,
    labels_dir: Path,
    area_threshold: float = 0.40,
    dry_run: bool = False,
    include_rejected: bool = False,
) -> dict:
    if area_threshold <= 0 or area_threshold >= 1:
        raise ValueError("area_threshold must be between 0 and 1")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    synth_ids: set[str] = set()
    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("status") or "").strip() != "ok":
                continue
            if (row.get("is_synthetic") or "0").strip() != "1":
                continue
            if not include_rejected and _is_rejected(row):
                continue
            image_id = (row.get("image_id") or "").strip()
            if image_id:
                synth_ids.add(image_id)

    stats = {
        "manifest_path": str(manifest_path),
        "labels_dir": str(labels_dir),
        "area_threshold": area_threshold,
        "dry_run": dry_run,
        "include_rejected": include_rejected,
        "synthetic_images_total": len(synth_ids),
        "label_files_present": 0,
        "label_files_missing": 0,
        "label_files_empty": 0,
        "boxes_total": 0,
        "massive_boxes_found": 0,
        "massive_boxes_removed": 0,
        "images_with_massive_boxes": 0,
        "images_modified": 0,
        "invalid_label_lines": 0,
    }

    if not labels_dir.exists():
        return stats

    for image_id in sorted(synth_ids):
        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            stats["label_files_missing"] += 1
            continue

        stats["label_files_present"] += 1
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            stats["label_files_empty"] += 1
            continue

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        keep_lines: list[str] = []
        removed_count = 0

        for line in lines:
            parsed = _parse_yolo_line(line)
            if parsed is None:
                stats["invalid_label_lines"] += 1
                keep_lines.append(line)
                continue

            _cls, _xc, _yc, w, h = parsed
            stats["boxes_total"] += 1
            if (w * h) > area_threshold:
                removed_count += 1
                continue

            keep_lines.append(line)

        if removed_count == 0:
            continue

        stats["images_with_massive_boxes"] += 1
        stats["massive_boxes_found"] += removed_count
        stats["massive_boxes_removed"] += removed_count

        if not dry_run:
            label_path.write_text(
                ("\n".join(keep_lines) + "\n") if keep_lines else "",
                encoding="utf-8",
            )
            stats["images_modified"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune oversized synth bounding boxes")
    parser.add_argument(
        "--manifest",
        default="manifests/images.csv",
        help="Manifest CSV path",
    )
    parser.add_argument(
        "--labels-dir",
        default="data/labels_autogen",
        help="Directory containing YOLO .txt labels",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Optional run dir; overrides default manifest/labels paths when set",
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.40,
        help="Remove boxes with area ratio above this threshold (default: 0.40)",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected synthetic rows (default: exclude)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed without writing files",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    labels_dir = Path(args.labels_dir)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        manifest = run_dir / "manifests" / "images.csv"
        labels_dir = run_dir / "data" / "labels_autogen"

    stats = prune_massive_synth_boxes(
        manifest_path=manifest,
        labels_dir=labels_dir,
        area_threshold=args.area_threshold,
        dry_run=args.dry_run,
        include_rejected=args.include_rejected,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
