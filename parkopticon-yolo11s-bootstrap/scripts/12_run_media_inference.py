#!/usr/bin/env python3
"""
Run trained YOLO weights on a media source and save outputs.
"""

import argparse
import importlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on media")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", required=True, help="Path to media file")
    parser.add_argument(
        "--project",
        default="runs/inference",
        help="Output project directory for inference results",
    )
    parser.add_argument(
        "--name",
        default=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Run name inside project directory",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="Device (cpu, 0, etc)")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO txt outputs",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Save confidences in txt outputs",
    )
    parser.add_argument(
        "--preview-dir",
        default="",
        help="Optional directory to write live preview assets",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help="Write preview every N frames/items",
    )
    parser.add_argument(
        "--preview-quality",
        type=int,
        default=80,
        help="JPEG quality for preview images (1-95)",
    )
    return parser.parse_args()


def _write_preview(
    preview_dir: Path,
    frame_number: int,
    image_bgr,
    status: str,
    quality: int,
) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    latest_image = preview_dir / "latest.jpg"
    latest_image_tmp = preview_dir / "latest.jpg.tmp"

    image_rgb = image_bgr[:, :, ::-1]
    Image.fromarray(image_rgb).save(latest_image_tmp, format="JPEG", quality=quality)
    os.replace(latest_image_tmp, latest_image)

    _write_preview_status(
        preview_dir=preview_dir,
        frame_number=frame_number,
        status=status,
    )


def _write_preview_status(preview_dir: Path, frame_number: int, status: str) -> None:
    latest_meta = preview_dir / "latest.json"
    latest_meta_tmp = preview_dir / "latest.json.tmp"

    payload = {
        "updated_at": datetime.now().isoformat(),
        "frame": frame_number,
        "status": status,
    }
    with open(latest_meta_tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    os.replace(latest_meta_tmp, latest_meta)


def main() -> int:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists() or not weights_path.is_file():
        logger.error("Weights not found: %s", weights_path)
        return 1

    source_path = Path(args.source)
    if not source_path.exists() or not source_path.is_file():
        logger.error("Media source not found: %s", source_path)
        return 1

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    preview_every = max(1, int(args.preview_every))
    preview_quality = max(1, min(95, int(args.preview_quality)))
    preview_dir = Path(args.preview_dir).resolve() if args.preview_dir else None
    preview_warning_emitted = False

    ultralytics_module = importlib.import_module("ultralytics")
    model = ultralytics_module.YOLO(str(weights_path))

    logger.info("Running inference")
    logger.info("Weights: %s", weights_path)
    logger.info("Source: %s", source_path)
    logger.info("Output project: %s", project_dir)
    logger.info("Run name: %s", args.name)

    result_count = 0
    result_stream = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        exist_ok=False,
        stream=True,
        verbose=True,
    )

    for result in result_stream:
        result_count += 1
        if (
            preview_dir
            and (result_count == 1 or result_count % preview_every == 0)
            and hasattr(result, "plot")
        ):
            try:
                image_bgr = result.plot()
                _write_preview(
                    preview_dir=preview_dir,
                    frame_number=result_count,
                    image_bgr=image_bgr,
                    status="running",
                    quality=preview_quality,
                )
            except Exception as exc:
                if not preview_warning_emitted:
                    logger.warning("Preview write failed: %s", exc)
                    preview_warning_emitted = True

    output_dir = project_dir / args.name
    output_files = []
    if output_dir.exists() and output_dir.is_dir():
        output_files = [
            str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()
        ]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(weights_path.resolve()),
        "source": str(source_path.resolve()),
        "project": str(project_dir.resolve()),
        "name": args.name,
        "output_dir": str(output_dir.resolve()),
        "frames_or_items_processed": result_count,
        "output_files": output_files,
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "save_txt": args.save_txt,
            "save_conf": args.save_conf,
        },
    }

    summary_path = output_dir / "inference_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    if preview_dir and preview_dir.exists():
        _write_preview_status(
            preview_dir=preview_dir,
            frame_number=result_count,
            status="completed",
        )

    logger.info("Inference complete. Processed %d frames/items", result_count)
    logger.info("Output directory: %s", output_dir)
    logger.info("Summary written to: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
