#!/usr/bin/env python3
"""
Evaluate trained YOLO model on test set.
Produces metrics and confusion-like summary.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--weights", "-w", required=True, help="Model weights path")
    parser.add_argument("--data", "-d", default="dataset.yaml", help="Dataset YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out-dir", "-o", default="reports", help="Output directory")
    args = parser.parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights not found: {weights_path}")
        logger.info("Run 07_train_yolo.py first to train the model")
        return
    
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        logger.error(f"Dataset YAML not found: {data_yaml}")
        return
    
    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))
    
    logger.info("Evaluating on test set...")
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        conf=args.conf,
        verbose=True,
        plots=True,
        save=True
    )
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "eval_date": datetime.now().isoformat(),
        "weights": str(weights_path),
        "confidence_threshold": args.conf,
        "metrics": {}
    }
    
    if hasattr(metrics, 'box'):
        box_metrics = metrics.box
        report["metrics"]["mAP50"] = float(box_metrics.map50) if box_metrics.map50 is not None else 0.0
        report["metrics"]["mAP50-95"] = float(box_metrics.map) if box_metrics.map is not None else 0.0
        report["metrics"]["precision"] = float(box_metrics.mp) if box_metrics.mp is not None else 0.0
        report["metrics"]["recall"] = float(box_metrics.mr) if box_metrics.mr is not None else 0.0
        
        if hasattr(box_metrics, 'ap50'):
            per_class = {}
            for i, ap in enumerate(box_metrics.ap50):
                class_name = "vehicle" if i == 0 else "enforcement_vehicle"
                per_class[class_name] = float(ap) if ap is not None else 0.0
            report["metrics"]["per_class_ap50"] = per_class
    
    logger.info("Evaluation metrics:")
    for key, value in report["metrics"].items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v:.4f}")
        else:
            logger.info(f"  {key}: {value:.4f}")
    
    json_path = out_dir / "eval_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report: {json_path}")
    
    md_content = f"""# Evaluation Report

**Date**: {datetime.now().isoformat()}

**Model**: {weights_path}

**Confidence Threshold**: {args.conf}

## Metrics

| Metric | Value |
|--------|-------|
"""
    
    for key, value in report["metrics"].items():
        if isinstance(value, dict):
            continue
        md_content += f"| {key} | {value:.4f} |\n"
    
    if "per_class_ap50" in report["metrics"]:
        md_content += "\n### Per-Class mAP50\n\n"
        md_content += "| Class | mAP50 |\n|-------|-------|\n"
        for class_name, ap in report["metrics"]["per_class_ap50"].items():
            md_content += f"| {class_name} | {ap:.4f} |\n"
    
    md_content += """
## Interpretation

- **mAP50**: Mean Average Precision at IoU=0.5 (loose matching)
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)

## Next Steps

1. Review confusion matrix in Val graphs
2. If enforcement_vehicle recall is low, add more synthetic examples
3. Consider test-time augmentation for improved robustness
"""
    
    md_path = out_dir / "eval_report.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
