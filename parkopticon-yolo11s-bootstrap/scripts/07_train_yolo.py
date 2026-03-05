#!/usr/bin/env python3
"""
Train YOLO11s detector on the dataset.
"""

import argparse
import importlib
import logging
import os
from datetime import datetime
from pathlib import Path

from utils.metadata import append_training_run_record, save_run_metadata


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11s detector")
    parser.add_argument("--data", "-d", default="dataset.yaml", help="Dataset YAML")
    parser.add_argument("--model", "-m", default="yolo11s.pt", help="Base model")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Epochs")
    parser.add_argument("--imgsz", "-s", type=int, default=640, help="Image size")
    parser.add_argument(
        "--batch", "-b", type=int, default=None, help="Batch size (auto if None)"
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu, 0, etc)")
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
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        logger.info("Run 06_split_dataset.py first to create the dataset split")
        return

    logger.info(f"Loading model: {args.model}")
    ultralytics_module = importlib.import_module("ultralytics")
    model = ultralytics_module.YOLO(args.model)

    project = "runs/detect"
    name = args.name

    logger.info(f"Starting training with {args.epochs} epochs")
    results = model.train(
        data=str(data_yaml),
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
        save_period=10,
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
    logger.info(f"Results plot: {results_path}")

    metadata_path = args.metadata_path or str(Path(project) / name / "metadata.json")

    if args.metadata:
        additional_info = {
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "run_name": name,
            "resume": args.resume,
        }
        if results is not None and hasattr(results, "results_dict"):
            additional_info["training_metrics"] = results.results_dict
        save_run_metadata(metadata_path, additional_metadata=additional_info)

    append_training_run_record(
        registry_path="reports/training_run_registry.jsonl",
        run_name=name,
        data_yaml=str(data_yaml),
        best_weights_path=str(best_weights),
        last_weights_path=str(last_weights),
        metadata_path=metadata_path if args.metadata else None,
        extra={
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "resume": args.resume,
        },
    )
    summary_path = Path("reports/train_summary.md")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("# Training Summary\n\n")
        f.write(f"**Date**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Model**: {args.model}\n\n")
        f.write(f"**Epochs**: {args.epochs}\n\n")
        f.write(f"**Image Size**: {args.imgsz}\n\n")
        f.write(f"**Device**: {args.device}\n\n")
        f.write(f"**Run Name**: {name}\n\n")

        if results is not None and hasattr(results, "results_dict"):
            f.write("## Metrics\n\n")
            metrics = results.results_dict
            for key, value in metrics.items():
                f.write(f"- {key}: {value}\n")

        f.write(f"\n## Weights\n\n")
        f.write(f"- Best: `{best_weights}`\n")
        f.write(f"- Last: `{last_weights}`\n")

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
