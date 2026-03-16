#!/usr/bin/env python3
"""
Sweep confidence values for prelabeling and summarize tuning metrics.

This script executes scripts/05_prelabel_boxes.py multiple times with isolated,
non-overwriting output paths per confidence value.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    epsilon = step / 1000.0
    while current <= stop + epsilon:
        values.append(round(current, 3))
        current += step
    return values


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _absolutize_path(value: str, run_dir: Path) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return cleaned
    candidate = Path(cleaned)
    return str(
        candidate if candidate.is_absolute() else (run_dir / candidate).resolve()
    )


def sanitize_manifest_for_prelabel(
    rows: list[dict[str, str]], run_dir: Path
) -> list[dict[str, str]]:
    reset_fields = {
        "needs_review",
        "qa_passed",
        "qa_failure_reason",
        "label_valid",
        "label_error",
        "num_boxes_autogen",
    }
    cleaned: list[dict[str, str]] = []
    for row in rows:
        row_copy = dict(row)
        if "file_path" in row_copy:
            row_copy["file_path"] = _absolutize_path(
                row_copy.get("file_path", ""), run_dir
            )
        if "source_file_path" in row_copy:
            row_copy["source_file_path"] = _absolutize_path(
                row_copy.get("source_file_path", ""), run_dir
            )
        for field in reset_fields:
            if field in row_copy:
                row_copy[field] = ""
        cleaned.append(row_copy)
    return cleaned


def write_manifest_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(
    manifest_rows: list[dict[str, str]], bad_synthetics_count: int
) -> dict:
    valid_rows = [r for r in manifest_rows if (r.get("status") or "") == "ok"]
    synthetic_rows = [r for r in valid_rows if (r.get("is_synthetic") or "0") == "1"]

    def as_int(value: str, default: int = 0) -> int:
        try:
            return int(float((value or "").strip()))
        except Exception:
            return default

    total_boxes = sum(as_int(r.get("num_boxes_autogen", "0"), 0) for r in valid_rows)
    nonzero_box_images = sum(
        1 for r in valid_rows if as_int(r.get("num_boxes_autogen", "0"), 0) > 0
    )
    yolo_empty_images = len(valid_rows) - nonzero_box_images
    needs_review_new_count = sum(
        1 for r in valid_rows if (r.get("needs_review") or "0") == "1"
    )
    qa_failed_count = sum(
        1 for r in synthetic_rows if (r.get("qa_passed") or "") == "0"
    )
    label_invalid_count = sum(
        1 for r in valid_rows if (r.get("label_valid") or "") == "0"
    )
    label_error_count = sum(
        1 for r in valid_rows if (r.get("label_error") or "").strip()
    )

    return {
        "images_ok": len(valid_rows),
        "images_synthetic_ok": len(synthetic_rows),
        "images_nonzero_boxes": nonzero_box_images,
        "yolo_empty_images": yolo_empty_images,
        "total_boxes_autogen": total_boxes,
        "needs_review_new_count": needs_review_new_count,
        "qa_failed_count": qa_failed_count,
        "bad_synthetics_count": bad_synthetics_count,
        "label_invalid_count": label_invalid_count,
        "label_error_count": label_error_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep prelabel confidence values")
    parser.add_argument(
        "--run-dir",
        default="runs/Benchmark_Run_002",
        help="Run directory (e.g. runs/Benchmark_Run_002)",
    )
    parser.add_argument("--start", type=float, default=0.10, help="Start confidence")
    parser.add_argument("--end", type=float, default=0.50, help="End confidence")
    parser.add_argument("--step", type=float, default=0.025, help="Confidence step")
    parser.add_argument(
        "--model", default="yolo11s.pt", help="YOLO model for prelabel script"
    )
    parser.add_argument(
        "--qa-threshold-area",
        type=float,
        default=0.15,
        help="QA threshold area passed through to prelabel script",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional custom output root. Defaults to <run-dir>/sweeps/prelabel_confidence",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifests" / "images.csv"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else run_dir / "sweeps" / "prelabel_confidence"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    confidence_values = frange(args.start, args.end, args.step)
    if not confidence_values:
        print("ERROR: no confidence values generated", file=sys.stderr)
        return 1

    results: list[dict] = []
    base_rows = load_manifest_rows(manifest_path)
    if not base_rows:
        print(f"ERROR: manifest has no rows: {manifest_path}", file=sys.stderr)
        return 1

    for conf in confidence_values:
        conf_tag = f"conf_{conf:.3f}".replace(".", "p")
        sweep_dir = output_root / conf_tag
        labels_dir = sweep_dir / "labels_autogen"
        lists_dir = sweep_dir / "lists"
        reports_dir = sweep_dir / "reports"
        labels_dir.mkdir(parents=True, exist_ok=True)
        lists_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        manifest_input = sweep_dir / "images_input.csv"
        manifest_copy = sweep_dir / "images_after_prelabel.csv"
        bad_synthetics = lists_dir / "bad_synthetics.txt"
        histogram_png = reports_dir / "prelabel_vehicle_count_hist.png"

        run_rows = sanitize_manifest_for_prelabel(base_rows, run_dir)
        write_manifest_rows(manifest_input, run_rows)

        command = [
            sys.executable,
            str((Path(__file__).resolve().parent / "05_prelabel_boxes.py").resolve()),
            "--manifest",
            str(manifest_input),
            "--out-dir",
            str(labels_dir),
            "--model",
            args.model,
            "--conf",
            f"{conf:.3f}",
            "--qa-threshold-area",
            f"{args.qa_threshold_area:.3f}",
            "--histogram-out",
            str(histogram_png),
        ]

        completed = subprocess.run(
            command,
            cwd=str(sweep_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if completed.returncode != 0:
            error_log = sweep_dir / "run_error.log"
            with open(error_log, "w", encoding="utf-8") as handle:
                handle.write(completed.stdout)
                handle.write("\n--- STDERR ---\n")
                handle.write(completed.stderr)
            results.append(
                {
                    "confidence": f"{conf:.3f}",
                    "status": "failed",
                    "returncode": completed.returncode,
                    "error_log": str(error_log),
                }
            )
            continue

        updated_rows = load_manifest_rows(manifest_input)
        if updated_rows:
            with open(manifest_copy, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(updated_rows[0].keys()))
                writer.writeheader()
                writer.writerows(updated_rows)

        metrics = summarize_metrics(updated_rows, count_lines(bad_synthetics))
        metrics.update(
            {
                "confidence": f"{conf:.3f}",
                "status": "ok",
                "labels_dir": str(labels_dir),
                "bad_synthetics_file": str(bad_synthetics),
                "histogram_png": str(histogram_png),
                "manifest_snapshot": str(manifest_copy),
            }
        )
        results.append(metrics)

    summary_csv = output_root / "summary.csv"
    if results:
        fieldnames = []
        seen = set()
        for row in results:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(summary_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("\n=== Prelabel Confidence Sweep Summary ===")
    print(f"Run dir: {run_dir}")
    print(f"Output root: {output_root}")
    print(f"Summary CSV: {summary_csv}")
    print(
        "\nconfidence | status | images_ok | nonzero_images | yolo_empty | total_boxes | needs_review_new | qa_failed | bad_synthetics | label_invalid"
    )
    for row in results:
        print(
            f"{row.get('confidence', '?'):>9} | "
            f"{row.get('status', '?'):>6} | "
            f"{str(row.get('images_ok', '-')):>9} | "
            f"{str(row.get('images_nonzero_boxes', '-')):>14} | "
            f"{str(row.get('yolo_empty_images', '-')):>10} | "
            f"{str(row.get('total_boxes_autogen', '-')):>11} | "
            f"{str(row.get('needs_review_new_count', '-')):>16} | "
            f"{str(row.get('qa_failed_count', '-')):>9} | "
            f"{str(row.get('bad_synthetics_count', '-')):>14} | "
            f"{str(row.get('label_invalid_count', '-')):>13}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
