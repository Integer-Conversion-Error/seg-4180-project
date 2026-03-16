#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import shutil
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DropGeminiNonTextWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Warning: there are non-text parts in the response" not in message


def suppress_noisy_gemini_logs() -> None:
    noisy_loggers = [
        "google",
        "google.genai",
        "google.generativeai",
        "httpx",
        "httpcore",
        "urllib3",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    suppress_filter = DropGeminiNonTextWarningFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(suppress_filter)
    for handler in root_logger.handlers:
        handler.addFilter(suppress_filter)

    warnings.filterwarnings(
        "ignore",
        message=r"Warning: there are non-text parts in the response:.*",
    )


STREET_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_street_scene": {"type": "boolean"},
        "street_confidence_low": {"type": "number", "minimum": 0, "maximum": 1},
        "street_confidence_high": {"type": "number", "minimum": 0, "maximum": 1},
        "street_explanation": {"type": "string"},
    },
    "required": [
        "is_street_scene",
        "street_confidence_low",
        "street_confidence_high",
        "street_explanation",
    ],
}

ROAD_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "has_visible_roadway": {"type": "boolean"},
        "road_confidence_low": {"type": "number", "minimum": 0, "maximum": 1},
        "road_confidence_high": {"type": "number", "minimum": 0, "maximum": 1},
        "road_explanation": {"type": "string"},
    },
    "required": [
        "has_visible_roadway",
        "road_confidence_low",
        "road_confidence_high",
        "road_explanation",
    ],
}


def _safe_conf_interval(low: float, high: float) -> tuple[float, float]:
    low_v = max(0.0, min(1.0, float(low)))
    high_v = max(0.0, min(1.0, float(high)))
    if high_v < low_v:
        low_v, high_v = high_v, low_v
    return low_v, high_v


def extract_text_from_response(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(str(text))
    return "".join(text_parts).strip()


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_yolo_empty_ids_from_manifest(manifest: list[dict], run_dir: Path) -> set[str]:
    ids: set[str] = set()
    for row in manifest:
        if (row.get("status") or "") != "ok":
            continue
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            continue
        file_path = _resolve_image_path(run_dir, row.get("file_path", ""))
        if not file_path.exists():
            continue
        try:
            box_count = int(float((row.get("num_boxes_autogen") or "").strip()))
        except Exception:
            continue
        if box_count == 0:
            ids.add(image_id)
    return ids


def save_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    # Collect all unique fieldnames across all rows
    all_keys: set[str] = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(run_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else run_dir / candidate


def _resolve_image_path(run_dir: Path, raw_value: str) -> Path:
    if not raw_value:
        return run_dir / ""
    candidate = Path(raw_value)
    return candidate if candidate.is_absolute() else run_dir / candidate


class RequestRateLimiter:
    def __init__(self, max_requests: int, period_seconds: float):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            wait_for = 0.0
            with self._lock:
                now = time.monotonic()
                while (
                    self._timestamps
                    and now - self._timestamps[0] >= self.period_seconds
                ):
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                wait_for = self.period_seconds - (now - self._timestamps[0])

            if wait_for > 0:
                time.sleep(wait_for)


def classify_street_with_gemini(
    client: genai.Client,
    model_id: str,
    image_path: Path,
    max_retries: int,
    min_street_confidence: float,
    rate_limiter: RequestRateLimiter | None = None,
) -> dict:
    prompt = (
        "You are a strict street-scene auditor for dataset gating. "
        "Return ONLY valid JSON matching the provided schema. Do not return markdown.\n"
        "Decision goal:\n"
        "Classify whether this image is a true outdoor street-level roadway scene suitable for vehicle-detection training.\n"
        "Positive evidence (favor true):\n"
        "- Outdoor viewpoint at or near road-user eye level\n"
        "- Visible drivable roadway (asphalt/concrete lane or road surface)\n"
        "- Context consistent with a street environment (lane markings, curbs, intersections, roadside buildings)\n"
        "Negative evidence (favor false):\n"
        "- Indoor/interior environments (garage, lobby, parking structure interior)\n"
        "- Beach, trail, field, or non-road natural scene\n"
        "- High-floor/aerial/rooftop/far-overlook perspective not representative of street-level driving view\n"
        "- Scene too occluded or ambiguous to confidently identify as street-level roadway\n"
        "Uncertainty policy:\n"
        "- If uncertain, set is_street_scene=false with lower confidence.\n"
        "Output requirements:\n"
        "- Populate is_street_scene, street_confidence_low, street_confidence_high, street_explanation.\n"
        "- Confidence must be within [0,1], with low <= high.\n"
        "- Explanation must cite concrete visual cues in one concise sentence."
    )
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            with image_path.open("rb") as handle:
                image_bytes = handle.read()
            if rate_limiter is not None:
                rate_limiter.acquire()
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=STREET_GATE_SCHEMA,
                    temperature=0.1,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            raw_text = extract_text_from_response(response)
            data = json.loads(raw_text)
            low, high = _safe_conf_interval(
                data["street_confidence_low"], data["street_confidence_high"]
            )
            return {
                "is_street_scene": bool(data["is_street_scene"])
                and high >= min_street_confidence,
                "street_confidence_low": low,
                "street_confidence_high": high,
                "street_explanation": str(data.get("street_explanation", "")).strip(),
                "raw_json": raw_text,
            }
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Street gate attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                image_path.name,
                exc,
            )
            time.sleep(min(2**attempt, 5))
    return {
        "is_street_scene": False,
        "street_confidence_low": 0.0,
        "street_confidence_high": 0.0,
        "street_explanation": f"gate_failed: {last_error}",
        "raw_json": "",
    }


def classify_road_with_gemini(
    client: genai.Client,
    model_id: str,
    image_path: Path,
    max_retries: int,
    min_road_confidence: float,
    rate_limiter: RequestRateLimiter | None = None,
) -> dict:
    prompt = (
        "You are a strict roadway-placement auditor for synthetic insertion gating. "
        "Return ONLY valid JSON matching the provided schema. Do not return markdown.\n"
        "Decision goal:\n"
        "Determine whether there is visible drivable roadway with plausible free space to insert one vehicle realistically.\n"
        "Use-case constraints:\n"
        "- Typical target vehicles are in approximately 2m-15m range from camera\n"
        "- Perspective should support realistic traffic direction and scale\n"
        "- Prefer placement opportunities in mid/far roadway when plausible\n"
        "Positive evidence (favor true):\n"
        "- Clear contiguous road surface with available lane/shoulder space\n"
        "- Geometry/perspective supports natural vehicle orientation\n"
        "- Lighting and shadows appear suitable for realistic compositing\n"
        "Negative evidence (favor false):\n"
        "- No visible drivable road surface\n"
        "- Road fully blocked/occupied, unsafe or implausible insertion area\n"
        "- Severe occlusion/cropping or perspective mismatch preventing realistic insertion\n"
        "Uncertainty policy:\n"
        "- If uncertain, set has_visible_roadway=false with lower confidence.\n"
        "Output requirements:\n"
        "- Populate has_visible_roadway, road_confidence_low, road_confidence_high, road_explanation.\n"
        "- Confidence must be within [0,1], with low <= high.\n"
        "- Explanation must cite concrete visual cues in one concise sentence."
    )
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            with image_path.open("rb") as handle:
                image_bytes = handle.read()
            if rate_limiter is not None:
                rate_limiter.acquire()
            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ROAD_GATE_SCHEMA,
                    temperature=0.1,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            raw_text = extract_text_from_response(response)
            data = json.loads(raw_text)
            low, high = _safe_conf_interval(
                data["road_confidence_low"], data["road_confidence_high"]
            )
            return {
                "has_visible_roadway": bool(data["has_visible_roadway"])
                and high >= min_road_confidence,
                "road_confidence_low": low,
                "road_confidence_high": high,
                "road_explanation": str(data.get("road_explanation", "")).strip(),
                "raw_json": raw_text,
            }
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Road gate attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                image_path.name,
                exc,
            )
            time.sleep(min(2**attempt, 5))
    return {
        "has_visible_roadway": False,
        "road_confidence_low": 0.0,
        "road_confidence_high": 0.0,
        "road_explanation": f"gate_failed: {last_error}",
        "raw_json": "",
    }


def stash_image(image_path: Path, stash_root: Path) -> Path:
    stash_root.mkdir(parents=True, exist_ok=True)
    destination = stash_root / image_path.name
    if destination.exists():
        destination = (
            stash_root / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"
        )
    shutil.move(str(image_path), str(destination))
    return destination


def copy_for_soft_gate(image_path: Path, soft_root: Path, category: str) -> Path:
    target_dir = soft_root / category
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / image_path.name
    if destination.exists():
        destination = (
            target_dir / f"{image_path.stem}_{int(time.time())}{image_path.suffix}"
        )
    shutil.copy2(str(image_path), str(destination))
    return destination


def evaluate_single_image(
    image_id: str,
    file_path: Path,
    gate_client: genai.Client | None,
    gemini_gate_model: str,
    gemini_retries: int,
    min_street_confidence: float,
    min_road_confidence: float,
    rate_limiter: RequestRateLimiter | None = None,
) -> tuple[str, dict, dict]:
    """Evaluate a single image through street and road gates (sequential per image).

    Returns (image_id, street_result, road_result).
    """
    street = {
        "is_street_scene": True,
        "street_confidence_low": 1.0,
        "street_confidence_high": 1.0,
        "street_explanation": "gate_skipped",
        "raw_json": "",
    }
    road = {
        "has_visible_roadway": True,
        "road_confidence_low": 1.0,
        "road_confidence_high": 1.0,
        "road_explanation": "gate_skipped",
        "raw_json": "",
    }

    if gate_client is not None:
        street = classify_street_with_gemini(
            gate_client,
            gemini_gate_model,
            file_path,
            max_retries=max(1, gemini_retries),
            min_street_confidence=min_street_confidence,
            rate_limiter=rate_limiter,
        )
        if street["is_street_scene"]:
            road = classify_road_with_gemini(
                gate_client,
                gemini_gate_model,
                file_path,
                max_retries=max(1, gemini_retries),
                min_road_confidence=min_road_confidence,
                rate_limiter=rate_limiter,
            )
        else:
            road = {
                "has_visible_roadway": False,
                "road_confidence_low": 0.0,
                "road_confidence_high": 0.0,
                "road_explanation": "not_street_scene",
                "raw_json": "",
            }

    return image_id, street, road


def main() -> int:
    load_dotenv()
    suppress_noisy_gemini_logs()
    parser = argparse.ArgumentParser(
        description="Stage 03b: Parallel Gemini gating for YOLO-empty images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", default=".")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv")
    parser.add_argument("--out-manifest", "-o", default="manifests/images.csv")
    parser.add_argument("--empty-out", "-e", default="lists/empty_candidates.txt")
    parser.add_argument("--valid-road-out", default="lists/valid_road_candidates.txt")
    parser.add_argument("--non-road-out", default="lists/non_road_candidates.txt")
    parser.add_argument("--non-street-out", default="lists/non_street_candidates.txt")
    parser.add_argument("--yolo-empty-in", default="lists/yolo_empty_all.txt")
    parser.add_argument("--stash-non-road-dir", default="data/images_excluded/non_road")
    parser.add_argument(
        "--stash-non-street-dir", default="data/images_excluded/non_street"
    )
    parser.add_argument(
        "--soft-gate",
        action="store_true",
        help="Copy gated images to timestamped snapshot folders instead of moving originals",
    )
    parser.add_argument(
        "--soft-gate-dir",
        default="data/gating_snapshots",
        help="Soft-gate snapshot root directory",
    )
    parser.add_argument("--gemini-gate-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--gemini-api-key", default=None)
    parser.add_argument("--gemini-retries", type=int, default=3)
    parser.add_argument("--min-street-confidence", type=float, default=0.75)
    parser.add_argument("--min-road-confidence", type=float, default=0.70)
    parser.add_argument("--skip-gemini-gate", action="store_true")
    parser.add_argument(
        "--gemini-workers",
        type=int,
        default=60,
        help="Number of parallel workers for Gemini API calls",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = _resolve_path(run_dir, args.manifest)
    out_manifest_path = _resolve_path(run_dir, args.out_manifest)
    empty_out = _resolve_path(run_dir, args.empty_out)
    valid_road_out = _resolve_path(run_dir, args.valid_road_out)
    non_road_out = _resolve_path(run_dir, args.non_road_out)
    non_street_out = _resolve_path(run_dir, args.non_street_out)
    yolo_empty_in = _resolve_path(run_dir, args.yolo_empty_in)
    stash_non_road_dir = _resolve_path(run_dir, args.stash_non_road_dir)
    stash_non_street_dir = _resolve_path(run_dir, args.stash_non_street_dir)
    soft_gate_dir = _resolve_path(run_dir, args.soft_gate_dir)

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        return 1
    manifest = load_manifest(manifest_path)
    if yolo_empty_in.exists():
        with open(yolo_empty_in, "r", encoding="utf-8") as handle:
            yolo_empty_ids = {line.strip() for line in handle if line.strip()}
    else:
        yolo_empty_ids = build_yolo_empty_ids_from_manifest(manifest, run_dir)
        if not yolo_empty_ids:
            logger.error(
                "YOLO empty list not found (%s) and could not infer empty IDs from manifest num_boxes_autogen",
                yolo_empty_in,
            )
            return 1
        yolo_empty_in.parent.mkdir(parents=True, exist_ok=True)
        with open(yolo_empty_in, "w", encoding="utf-8") as handle:
            for image_id in sorted(yolo_empty_ids):
                handle.write(f"{image_id}\n")
        logger.warning(
            "YOLO empty list missing; bootstrapped %s IDs from manifest into %s",
            len(yolo_empty_ids),
            yolo_empty_in,
        )

    snapshot_root = None
    if args.soft_gate:
        snapshot_root = soft_gate_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_root.mkdir(parents=True, exist_ok=True)

    gate_client = None
    if not args.skip_gemini_gate:
        api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Gemini gate enabled but GEMINI_API_KEY is missing")
            return 1
        gate_client = genai.Client(api_key=api_key)

    empty_out.parent.mkdir(parents=True, exist_ok=True)
    valid_road_out.parent.mkdir(parents=True, exist_ok=True)
    non_road_out.parent.mkdir(parents=True, exist_ok=True)
    non_street_out.parent.mkdir(parents=True, exist_ok=True)

    empty_candidates: list[str] = []
    valid_road_candidates: list[str] = []
    non_road_candidates: list[str] = []
    non_street_candidates: list[str] = []
    pass2_items: list[tuple[str, dict, Path]] = []

    for row in manifest:
        image_id = row.get("image_id", "")
        file_path = _resolve_image_path(run_dir, row.get("file_path", ""))

        if image_id not in yolo_empty_ids:
            row["route_category"] = "valid_road_existing_vehicle"
            row["excluded_reason"] = ""
            row["street_scene_valid"] = ""
            row["road_scene_valid"] = ""
            row["street_scene_explanation"] = "skipped_non_empty_yolo"
            row["road_scene_explanation"] = "skipped_non_empty_yolo"
            row["gemini_gate_model"] = "none"
            row["gemini_gate_json"] = ""
            valid_road_candidates.append(image_id)
            continue

        if not file_path.exists() or row.get("status") != "ok":
            continue

        pass2_items.append((image_id, row, file_path))

    logger.info(
        "Stage 03b pass-2 request count (YOLO-empty after filters): %s",
        len(pass2_items),
    )

    # Build lookup for rows by image_id
    row_lookup: dict[str, tuple[dict, Path]] = {
        image_id: (row, file_path) for image_id, row, file_path in pass2_items
    }

    logger.info(
        "Stage 03b pass-2 request count (YOLO-empty after filters): %s",
        len(pass2_items),
    )
    logger.info("Using %d parallel workers for Gemini API calls", args.gemini_workers)

    rate_limiter = RequestRateLimiter(max_requests=280, period_seconds=60.0)

    # Parallel execution with ThreadPoolExecutor
    results: list[tuple[str, dict, dict]] = []
    with ThreadPoolExecutor(max_workers=args.gemini_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_image,
                image_id,
                file_path,
                gate_client,
                args.gemini_gate_model,
                args.gemini_retries,
                args.min_street_confidence,
                args.min_road_confidence,
                rate_limiter,
            ): image_id
            for image_id, row, file_path in pass2_items
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Stage 03b: Gemini gating parallel",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                image_id = futures[future]
                logger.error("Worker failed for %s: %s", image_id, exc)
                # Add a failed result
                results.append(
                    (
                        image_id,
                        {
                            "is_street_scene": False,
                            "street_confidence_low": 0.0,
                            "street_confidence_high": 0.0,
                            "street_explanation": f"worker_exception: {exc}",
                            "raw_json": "",
                        },
                        {
                            "has_visible_roadway": False,
                            "road_confidence_low": 0.0,
                            "road_confidence_high": 0.0,
                            "road_explanation": "worker_exception",
                            "raw_json": "",
                        },
                    )
                )

    # Process results
    for image_id, street, road in results:
        row, file_path = row_lookup[image_id]

        row["street_scene_valid"] = "1" if street["is_street_scene"] else "0"
        row["street_scene_confidence_low"] = f"{street['street_confidence_low']:.3f}"
        row["street_scene_confidence_high"] = f"{street['street_confidence_high']:.3f}"
        row["street_scene_explanation"] = street["street_explanation"]
        row["road_scene_valid"] = "1" if road["has_visible_roadway"] else "0"
        row["road_scene_confidence_low"] = f"{road['road_confidence_low']:.3f}"
        row["road_scene_confidence_high"] = f"{road['road_confidence_high']:.3f}"
        row["road_scene_explanation"] = road["road_explanation"]
        row["gemini_gate_model"] = args.gemini_gate_model if gate_client else "none"
        row["gemini_gate_json"] = json.dumps(
            {
                "street_gate": street.get("raw_json", ""),
                "road_gate": road.get("raw_json", ""),
            }
        )

        if not street["is_street_scene"]:
            non_street_candidates.append(image_id)
            row["route_category"] = "non_street"
            row["excluded_reason"] = "not_street_scene"
            row["needs_review"] = "1"
            if file_path.exists():
                if args.soft_gate and snapshot_root is not None:
                    copied = copy_for_soft_gate(file_path, snapshot_root, "non_street")
                    row["soft_gate_snapshot_path"] = str(copied)
                else:
                    row["file_path"] = str(stash_image(file_path, stash_non_street_dir))
            continue

        if road["has_visible_roadway"]:
            valid_road_candidates.append(image_id)
            empty_candidates.append(image_id)
            row["route_category"] = "valid_road_empty"
            row["needs_review"] = "1"
            row["excluded_reason"] = ""
        else:
            non_road_candidates.append(image_id)
            row["route_category"] = "non_road"
            row["excluded_reason"] = "no_visible_roadway"
            row["needs_review"] = "1"
            if file_path.exists():
                if args.soft_gate and snapshot_root is not None:
                    copied = copy_for_soft_gate(file_path, snapshot_root, "non_road")
                    row["soft_gate_snapshot_path"] = str(copied)
                else:
                    row["file_path"] = str(stash_image(file_path, stash_non_road_dir))

    with open(empty_out, "w", encoding="utf-8") as handle:
        for image_id in empty_candidates:
            handle.write(f"{image_id}\n")
    with open(valid_road_out, "w", encoding="utf-8") as handle:
        for image_id in valid_road_candidates:
            handle.write(f"{image_id}\n")
    with open(non_road_out, "w", encoding="utf-8") as handle:
        for image_id in non_road_candidates:
            handle.write(f"{image_id}\n")
    with open(non_street_out, "w", encoding="utf-8") as handle:
        for image_id in non_street_candidates:
            handle.write(f"{image_id}\n")

    save_manifest(manifest, out_manifest_path)
    if args.soft_gate and snapshot_root is not None:
        logger.info("Soft-gate snapshot root: %s", snapshot_root)
    logger.info("Empty candidates (legacy): %s", len(empty_candidates))
    logger.info("Valid-road candidates: %s", len(valid_road_candidates))
    logger.info("Non-road candidates: %s", len(non_road_candidates))
    logger.info("Non-street candidates: %s", len(non_street_candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
