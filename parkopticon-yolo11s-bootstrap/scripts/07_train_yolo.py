#!/usr/bin/env python3
"""
Train YOLO11s detector on the dataset.
"""

import argparse
import csv
import importlib
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.metadata import append_training_run_record, save_run_metadata
from utils.dataset_exclusion import load_dataset_excluded_ids


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LOOKALIKE_CLASS_ID = 4
VEHICLE_CLASS_ID = 0


def _is_rejected(row: dict) -> bool:
    return (row.get("review_status") or "").strip().lower() == "rejected"


def _find_rejected_assigned_to_splits(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return []

    rejected_ids: list[str] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if row.get("split") not in {"train", "val", "test"}:
                continue
            if _is_rejected(row):
                rejected_ids.append(row.get("image_id", ""))
    return rejected_ids


def _parse_dataset_split_dirs(data_yaml: Path) -> list[Path]:
    dataset_root = data_yaml.parent
    base_path = ""
    relative_dirs: dict[str, str] = {}

    with open(data_yaml, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == "path":
                base_path = value
            elif key in {"train", "val", "test"}:
                relative_dirs[key] = value

    if not all(k in relative_dirs for k in ("train", "val", "test")):
        return []

    base = (dataset_root / base_path).resolve() if base_path else dataset_root.resolve()
    return [(base / relative_dirs[k]).resolve() for k in ("train", "val", "test")]


def _find_rejected_present_in_split_dirs(
    manifest_path: Path, data_yaml: Path
) -> list[str]:
    if not manifest_path.exists() or not data_yaml.exists():
        return []

    rejected_ids: set[str] = set()
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if _is_rejected(row):
                image_id = (row.get("image_id") or "").strip()
                if image_id:
                    rejected_ids.add(image_id)

    if not rejected_ids:
        return []

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    leaked_ids: set[str] = set()
    for split_dir in _parse_dataset_split_dirs(data_yaml):
        if not split_dir.exists():
            continue
        for image_path in split_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
                continue
            image_id = image_path.stem
            if image_id in rejected_ids:
                leaked_ids.add(image_id)

    return sorted(leaked_ids)


def _find_dataset_excluded_assigned_to_splits(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return []
    excluded_ids = load_dataset_excluded_ids(manifest_path)
    leaked_ids: list[str] = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            if row.get("split") not in {"train", "val", "test"}:
                continue
            image_id = (row.get("image_id") or "").strip()
            if image_id and image_id in excluded_ids:
                leaked_ids.append(image_id)
    return leaked_ids


def _find_dataset_excluded_present_in_split_dirs(
    manifest_path: Path, data_yaml: Path
) -> list[str]:
    if not manifest_path.exists() or not data_yaml.exists():
        return []
    excluded_ids = load_dataset_excluded_ids(manifest_path)
    if not excluded_ids:
        return []
    leaked_ids: set[str] = set()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for split_dir in _parse_dataset_split_dirs(data_yaml):
        if not split_dir.exists():
            continue
        for image_path in split_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in image_exts:
                continue
            if image_path.stem in excluded_ids:
                leaked_ids.add(image_path.stem)
    return sorted(leaked_ids)


def _label_dirs_from_data_yaml(data_yaml: Path) -> list[Path]:
    split_image_dirs = _parse_dataset_split_dirs(data_yaml)
    if not split_image_dirs:
        return []

    label_dirs: list[Path] = []
    for image_dir in split_image_dirs:
        label_dirs.append(image_dir.parent / "labels")
    return label_dirs


def _remap_lookalike_labels_to_vehicle(
    label_dirs: list[Path],
) -> tuple[dict[Path, str], int, int]:
    backups: dict[Path, str] = {}
    remapped_boxes = 0
    remapped_files = 0

    for label_dir in label_dirs:
        if not label_dir.exists() or not label_dir.is_dir():
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            original_text = label_path.read_text(encoding="utf-8")
            if not original_text.strip():
                continue

            trailing_newline = original_text.endswith("\n")
            changed = False
            updated_lines: list[str] = []

            for line in original_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    updated_lines.append(line)
                    continue

                parts = stripped.split()
                cls_token = parts[0]
                try:
                    cls_id = int(cls_token)
                except ValueError:
                    updated_lines.append(line)
                    continue

                if cls_id == LOOKALIKE_CLASS_ID:
                    parts[0] = str(VEHICLE_CLASS_ID)
                    updated_lines.append(" ".join(parts))
                    remapped_boxes += 1
                    changed = True
                else:
                    updated_lines.append(line)

            if not changed:
                continue

            remapped_files += 1
            backups[label_path] = original_text
            remapped_text = "\n".join(updated_lines)
            if trailing_newline:
                remapped_text += "\n"
            label_path.write_text(remapped_text, encoding="utf-8")

    return backups, remapped_boxes, remapped_files


def _restore_label_files(backups: dict[Path, str]) -> None:
    for path, content in backups.items():
        path.write_text(content, encoding="utf-8")


def _write_collapsed_lookalike_data_yaml(data_yaml: Path) -> Path:
    source_lines = data_yaml.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    skipping_names_block = False

    for line in source_lines:
        stripped = line.strip()

        if stripped.startswith("names:"):
            skipping_names_block = True
            continue

        if skipping_names_block:
            if stripped == "" or line.startswith((" ", "\t")):
                continue
            skipping_names_block = False

        if stripped.startswith("nc:"):
            continue

        rewritten.append(line)

    while rewritten and rewritten[-1].strip() == "":
        rewritten.pop()

    rewritten.extend(
        [
            "",
            "nc: 4",
            "names:",
            "  0: vehicle",
            "  1: enforcement_vehicle",
            "  2: police_old",
            "  3: police_new",
        ]
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="train_data_lookalike_as_vehicle_",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(rewritten) + "\n")
        return Path(handle.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11s detector")
    parser.add_argument("--data", "-d", default="dataset.yaml", help="Dataset YAML")
    parser.add_argument("--model", "-m", default="yolo11s.pt", help="Base model")
    parser.add_argument(
        "--project", "-p", default="runs/detect", help="Output project directory"
    )
    parser.add_argument("--epochs", "-e", type=int, default=300, help="Epochs")
    parser.add_argument("--imgsz", "-s", type=int, default=640, help="Image size")
    parser.add_argument(
        "--batch", "-b", type=int, default=None, help="Batch size (auto if None)"
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (-1 to disable periodic saves)",
    )
    parser.add_argument("--device", default="0", help="Device (cpu, 0, etc)")
    parser.add_argument(
        "--name", "-n", default="parkopticon_vehicle_enforcement", help="Run name"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--metadata", action="store_true", default=True, help="Save run metadata"
    )
    parser.add_argument(
        "--metadata-path", default=None, help="Custom metadata path (auto if None)"
    )
    parser.add_argument(
        "--manifest",
        default="manifests/images.csv",
        help="Manifest CSV used for split assignment checks",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Allow training when rejected images are assigned to splits (default: block)",
    )
    parser.add_argument(
        "--count-lookalike-as-vehicle",
        action="store_true",
        help=(
            "During this training run, treat class 4 (lookalike_negative) labels as "
            "class 0 (vehicle) and train as a 4-class model"
        ),
    )
    parser.add_argument(
        "--timestamp-name",
        action="store_true",
        help="Append a timestamp suffix to --name to avoid collisions",
    )
    parser.add_argument(
        "--timestamp-format",
        default="%Y%m%d_%H%M%S",
        help="strftime format used when --timestamp-name is enabled",
    )
    parser.add_argument(
        "--registry-path",
        default="reports/training_run_registry.jsonl",
        help="Path to JSONL registry where run records are appended",
    )
    parser.add_argument(
        "--summary-path",
        default="reports/train_summary.md",
        help="Path to markdown summary output (supports {run_name} placeholder)",
    )
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.info("Run 06_split_dataset.py first to create the dataset split")
        return 1

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        logger.error(
            "Training blocked: manifest not found for safety checks: %s",
            manifest_path,
        )
        return 1

    excluded_ids = load_dataset_excluded_ids(manifest_path)
    excluded_assigned = _find_dataset_excluded_assigned_to_splits(manifest_path)
    if excluded_assigned:
        sample = ", ".join(excluded_assigned[:5])
        logger.error(
            "Training blocked: %d dataset-excluded images are assigned to train/val/test splits.",
            len(excluded_assigned),
        )
        logger.error("Sample image_ids: %s", sample)
        return 1

    excluded_leaked = _find_dataset_excluded_present_in_split_dirs(
        manifest_path, data_yaml
    )
    if excluded_leaked:
        sample = ", ".join(excluded_leaked[:5])
        logger.error(
            "Training blocked: %d dataset-excluded images are present in dataset split directories.",
            len(excluded_leaked),
        )
        logger.error("Sample image_ids: %s", sample)
        return 1

    if not args.include_rejected:
        rejected_ids = _find_rejected_assigned_to_splits(manifest_path)
        if rejected_ids:
            sample = ", ".join(rejected_ids[:5])
            logger.error(
                "Training blocked: %d rejected images are still assigned to train/val/test splits.",
                len(rejected_ids),
            )
            logger.error("Sample image_ids: %s", sample)
            logger.error(
                "Re-run split without rejected items (default): python scripts/06_split_dataset.py"
            )
            logger.error("Or override explicitly: --include-rejected")
            return 1

        leaked_ids = _find_rejected_present_in_split_dirs(manifest_path, data_yaml)
        if leaked_ids:
            sample = ", ".join(leaked_ids[:5])
            logger.error(
                "Training blocked: %d rejected images are present in dataset split directories.",
                len(leaked_ids),
            )
            logger.error("Sample image_ids: %s", sample)
            logger.error(
                "Re-run split without rejected items (default): python scripts/06_split_dataset.py"
            )
            logger.error("Or override explicitly: --include-rejected")
            return 1

    effective_data_yaml = data_yaml
    label_backups: dict[Path, str] = {}
    temp_data_yaml: Path | None = None

    if args.count_lookalike_as_vehicle:
        label_dirs = _label_dirs_from_data_yaml(data_yaml)
        if not label_dirs:
            logger.error(
                "Unable to parse split image directories from dataset YAML for lookalike remap: %s",
                data_yaml,
            )
            return 1

        label_backups, remapped_boxes, remapped_files = (
            _remap_lookalike_labels_to_vehicle(label_dirs)
        )
        logger.info(
            "Lookalike remap enabled: remapped %d boxes across %d label files",
            remapped_boxes,
            remapped_files,
        )

        temp_data_yaml = _write_collapsed_lookalike_data_yaml(data_yaml)
        effective_data_yaml = temp_data_yaml
        logger.info(
            "Using temporary 4-class dataset YAML for training: %s", effective_data_yaml
        )

    try:
        logger.info(f"Loading model: {args.model}")
        ultralytics_module = importlib.import_module("ultralytics")
        model = ultralytics_module.YOLO(args.model)

        project = args.project
        name = args.name
        if args.timestamp_name:
            timestamp = datetime.now().strftime(args.timestamp_format)
            name = f"{name}_{timestamp}"

        logger.info(f"Starting training with {args.epochs} epochs")
        results = model.train(
            data=str(effective_data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=project,
            name=name,
            exist_ok=True,
            resume=args.resume,
            verbose=True,
            plots=True,
            save=True,
            save_period=args.save_period,
        )

        logger.info("Training complete!")

        best_weights = Path(project) / name / "weights" / "best.pt"
        if best_weights.exists():
            logger.info(f"Best weights: {best_weights}")

        last_weights = Path(project) / name / "weights" / "last.pt"
        if last_weights.exists():
            logger.info(f"Last weights: {last_weights}")

        results_path = Path(project) / name / "results.png"
        if results_path.exists():
            logger.info(f"Results plot: {results_path}")

        if args.metadata_path:
            metadata_path = args.metadata_path.replace("{run_name}", name)
        else:
            metadata_path = str(Path(project) / name / "metadata.json")

        if args.metadata:
            additional_info = {
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "save_period": args.save_period,
                "device": args.device,
                "run_name": name,
                "resume": args.resume,
                "count_lookalike_as_vehicle": args.count_lookalike_as_vehicle,
            }
            if results is not None and hasattr(results, "results_dict"):
                additional_info["training_metrics"] = results.results_dict
            save_run_metadata(metadata_path, additional_metadata=additional_info)

        append_training_run_record(
            registry_path=args.registry_path,
            run_name=name,
            data_yaml=str(data_yaml),
            best_weights_path=str(best_weights),
            last_weights_path=str(last_weights),
            metadata_path=metadata_path if args.metadata else None,
            manifest_path=str(manifest_path),
            extra={
                "model": args.model,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "save_period": args.save_period,
                "device": args.device,
                "resume": args.resume,
                "count_lookalike_as_vehicle": args.count_lookalike_as_vehicle,
            },
        )
        summary_path = Path(args.summary_path.replace("{run_name}", name))
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Training Summary\n\n")
            f.write(f"**Date**: {datetime.now().isoformat()}\n\n")
            f.write(f"**Model**: {args.model}\n\n")
            f.write(f"**Epochs**: {args.epochs}\n\n")
            f.write(f"**Image Size**: {args.imgsz}\n\n")
            f.write(f"**Save Period**: {args.save_period}\n\n")
            f.write(f"**Device**: {args.device}\n\n")
            f.write(f"**Run Name**: {name}\n\n")
            f.write(
                f"**Count Lookalike As Vehicle**: {args.count_lookalike_as_vehicle}\n\n"
            )

            if results is not None and hasattr(results, "results_dict"):
                f.write("## Metrics\n\n")
                metrics = results.results_dict
                for key, value in metrics.items():
                    f.write(f"- {key}: {value}\n")

            f.write(f"\n## Weights\n\n")
            f.write(f"- Best: `{best_weights}`\n")
            f.write(f"- Last: `{last_weights}`\n")

        logger.info(f"Summary saved to {summary_path}")
        return 0
    finally:
        if label_backups:
            _restore_label_files(label_backups)
            logger.info("Restored original label files after temporary lookalike remap")
        if temp_data_yaml and temp_data_yaml.exists():
            temp_data_yaml.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
