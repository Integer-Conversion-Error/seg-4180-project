#!/usr/bin/env python3
"""
Synthesize vehicle edits using Gemini batch mode by default, with optional Grok support.
Creates random_vehicle and enforcement_vehicle variants of empty frames.
Uses reference images for enforcement classes.
"""

import argparse
import csv
import base64
import json
import logging
import os
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import resize_cover_center_crop
from utils.dataset_exclusion import load_dataset_excluded_ids
from utils.grok_image_api import (
    GrokImageAPIError,
    encode_image_to_base64,
    generate_image,
)
from utils.gemini_image_api import (
    GeminiImageAPIError,
    generate_image as generate_gemini_image,
)


load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reference image directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENFORCEMENT_DATASET_DIR = PROJECT_ROOT / "enforcement-dataset"
POLICE_OLD_DATASET_DIR = PROJECT_ROOT / "police-dataset-old"
POLICE_NEW_DATASET_DIR = PROJECT_ROOT / "police-dataset-new"

# BK|# Enforcement subtype models
# BK|ENFORCEMENT_MODELS = [
# BK|    "hyundai_kona_electric_1st_gen_2019",
# BK|    "toyota_yaris_xp_150_facelift_2020",
# BK|    "ford_fusion_hybrid_2015",
# BK|    "hyundai_ioniq_1st_gen_2018",
# BK|    "hyundai_ioniq_1st_gen_facelift_2021",
# BK|    "toyota_rav4_hybrid_4th_gen_facelift_2017",
# BK|    "ford_escape_4th_gen_2021",
# BK|    "ford_explorer_5th_gen_2017",
# BK|    "ford_explorer_6th_gen_2021",
# BK|]


# Constraints (shared with synth_ui)
STRICT_SIZE_CONSTRAINT = """Critical output constraint: return an image with exactly the same width and height as the input street scene. Do not crop, resize, pad, rotate, or change aspect ratio."""

TRAFFIC_DIRECTION_CONSTRAINT = """Orientation constraint: the inserted vehicle must be aligned with the correct lane direction for this road scene, matching traffic flow and lane markings. Do not place a wrong-way vehicle."""

SCENE_LOCK_CONSTRAINT = """Scene integrity constraint: preserve the original scene geometry and composition exactly. Do not move or alter camera pose, horizon, vanishing points, road/curb alignment, lane markings, buildings, trees, poles, signs, parked objects, or any static landmark. Do not shift, warp, redraw, or restyle the background."""

LOCAL_EDIT_ONLY_CONSTRAINT = """Local-edit-only constraint: modify only the minimal pixel region required to insert the new vehicle, plus immediate physically consistent contact shadow/occlusion near that vehicle. All other pixels should remain visually unchanged. If constraints cannot be satisfied, return the original scene unchanged."""

SHADOW_CONSISTENCY_CONSTRAINT = """Shadow consistency constraint: shadow intensity and sharpness must match the scene's apparent lighting conditions. Bright sunny scenes should cast sharp, dark shadows. Overcast, cloudy, or diffuse lighting scenes should have soft, faint, or no visible shadows. Do not add hard, distinct shadows when the scene appears overcast or has soft ambient lighting."""

PLAUSIBLE_PLACEMENT_CONSTRAINT = """Plausible placement constraint: the vehicle must be placed on a valid, drivable asphalt or concrete roadway surface. Do not place the vehicle on sidewalks, in buildings, on grass, or floating in mid-air. It must be positioned naturally within the lane markers as if actively driving in traffic. If the target street scene does not contain any visible, drivable roadway (e.g., only sidewalks, grass, or water are visible), you MUST refuse the request by returning the exact text "REFUSE: NO ROAD VISIBLE" and no image."""

# New constraint: Google Street View context
STREET_VIEW_CONTEXT = """CRITICAL CONTEXT: This is a Google Street View image. The camera that captured this photo has specific characteristics: fish-eye distortion at edges, particular color grading, specific perspective vanishing point, and metadata embedded in the image style. The vehicle you insert MUST look exactly like it was captured by the same Street View camera - as if it were actually there when the panorama was taken. Match the Street View look precisely."""

# New constraint: Natural placement (avoid "advertisement" center placement)
NATURAL_PLACEMENT_CONSTRAINT = """Natural placement constraint: place the vehicle at a natural driving position within the lane, NOT centered like an advertisement or product shot. The vehicle should be positioned where a real car would naturally appear - typically offset to one side following lane position, at a realistic depth within the scene. Avoid placing vehicles dead-center of the frame unless the lane geometry truly warrants it."""

# New constraint: Prevent perspective tilt issues (the ~30 degree tilt problem)
VEHICLE_ORIENTATION_CONSTRAINT = """Vehicle orientation precision: the vehicle's orientation must match the road's perspective exactly. If the road appears straight to the horizon with no curve, the vehicle must be perfectly straight with NO tilt angle - do not tilt it ~30 degrees. The vehicle should align with the road's vanishing point and lane markings. When viewing from the front or rear (not from the side), the vehicle should appear rectangular, not angled. The vehicle must follow the exact perspective lines of the road and lane markings."""

# New constraint: Seamless blending
SEAMLESS_BLEND_CONSTRAINT = """Seamless blending constraint: the inserted vehicle must be seamlessly integrated into the scene with no visible edges, seams, or "placed in" appearance. Match the exact color temperature, white balance, contrast, grain, and post-processing of the original Street View image. The vehicle should have the same fish-eye distortion characteristics at the edges if applicable. No harsh outlines, color discontinuities, or obvious insertion artifacts."""

# New constraint: Shadow and reflection accuracy
SHADOW_ACCURACY_CONSTRAINT = """Shadow and reflection accuracy: ensure the vehicle casts shadows that are consistent with existing shadows in the scene. If trees or buildings cast long shadows, the vehicle must too. If the scene has no visible shadows (overcast), the vehicle should have minimal or no shadow. The vehicle's shadow should fall on the road surface naturally, matching the direction of existing shadows. Ground reflections on wet pavement must be consistent with the vehicle and lighting."""

REALISTIC_VEHICLE_COLORS = [
    "white",
    "black",
    "silver",
    "gray",
    "dark gray",
    "blue",
    "navy blue",
    "red",
    "burgundy",
    "green",
    "beige",
    "brown",
]

REALISTIC_VEHICLE_BODY_TYPES = [
    "sedan",
    "compact sedan",
    "hatchback",
    "coupe",
    "SUV",
    "compact SUV",
    "pickup truck",
    "minivan",
    "crossover",
    "wagon",
]

# Distance/Scale definitions
DISTANCE_OPTIONS = [
    "very close (foreground, large size)",
    "medium distance (midground, average size)",
    "far away (background, small size)",
]


def build_random_vehicle_prompt(
    image_id: str,
    seed: int,
    distance: str = "medium distance (midground, average size)",
) -> tuple[str, str, str]:
    rng = random.Random(f"{seed}:{image_id}")
    color = rng.choice(REALISTIC_VEHICLE_COLORS)
    body_type = rng.choice(REALISTIC_VEHICLE_BODY_TYPES)

    prompt = (
        "Change only what is necessary. Keep camera perspective, lighting, shadows, color grading identical. "
        f"Add exactly ONE {color} {body_type} on the roadway (not parked), {distance}, clearly visible. "
        "It must look like a normal civilian vehicle with no emergency lightbar, no police markings, and no municipal enforcement markings. "
        "Avoid changing signs, buildings, or sky. No global style shifts. "
        + SCENE_LOCK_CONSTRAINT
        + " "
        + PLAUSIBLE_PLACEMENT_CONSTRAINT
        + " "
        + LOCAL_EDIT_ONLY_CONSTRAINT
        + " "
        + TRAFFIC_DIRECTION_CONSTRAINT
        + " "
        + SHADOW_CONSISTENCY_CONSTRAINT
        + " "
        + STRICT_SIZE_CONSTRAINT
        + " "
        + STREET_VIEW_CONTEXT
        + " "
        + NATURAL_PLACEMENT_CONSTRAINT
        + " "
        + VEHICLE_ORIENTATION_CONSTRAINT
        + " "
        + SEAMLESS_BLEND_CONSTRAINT
        + " "
        + SHADOW_ACCURACY_CONSTRAINT
    )
    return prompt, color, body_type


# Prompts for enforcement_vehicle WITH reference image
ENFORCEMENT_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference enforcement vehicle. "
    "Insert one enforcement vehicle into the target scene, matching the reference style while preserving scene realism. "
    "Place the vehicle at {distance}. "
    "{spatial_instruction}"
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + PLAUSIBLE_PLACEMENT_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
    + " "
    + STREET_VIEW_CONTEXT
    + " "
    + NATURAL_PLACEMENT_CONSTRAINT
    + " "
    + VEHICLE_ORIENTATION_CONSTRAINT
    + " "
    + SEAMLESS_BLEND_CONSTRAINT
    + " "
    + SHADOW_ACCURACY_CONSTRAINT
)

# Prompts for police_old WITH reference image
POLICE_OLD_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with OLD livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Place the vehicle at {distance}. "
    "{spatial_instruction}"
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + PLAUSIBLE_PLACEMENT_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
    + " "
    + STREET_VIEW_CONTEXT
    + " "
    + NATURAL_PLACEMENT_CONSTRAINT
    + " "
    + VEHICLE_ORIENTATION_CONSTRAINT
    + " "
    + SEAMLESS_BLEND_CONSTRAINT
    + " "
    + SHADOW_ACCURACY_CONSTRAINT
)

# Prompts for police_new WITH reference image
POLICE_NEW_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with NEW livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Place the vehicle at {distance}. "
    "{spatial_instruction}"
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + PLAUSIBLE_PLACEMENT_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
    + " "
    + STREET_VIEW_CONTEXT
    + " "
    + NATURAL_PLACEMENT_CONSTRAINT
    + " "
    + VEHICLE_ORIENTATION_CONSTRAINT
    + " "
    + SEAMLESS_BLEND_CONSTRAINT
    + " "
    + SHADOW_ACCURACY_CONSTRAINT
)


def generate_image_id(parent_id: str, edit_type: str) -> str:
    return f"{parent_id}_{edit_type}"


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    # Union of all keys across all rows so new columns added by synthetic rows
    # (e.g. variant_index, review_bucket) don't cause DictWriter to raise.
    seen: dict = {}
    for row in manifest:
        for k in row.keys():
            if k not in seen:
                seen[k] = None
    fieldnames = list(seen.keys())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)


CARDINAL_ANGLES = ["N", "W", "S", "E"]  # Front, Left, Rear, Right
CARDINAL_NAME_TO_ANGLE = {
    "front": "N",
    "rear": "S",
    "back": "S",
    "left": "W",
    "right": "E",
}


def _extract_cardinal_angle(stem: str) -> Optional[str]:
    for angle in CARDINAL_ANGLES:
        if stem.endswith(f"_{angle}"):
            return angle

    tokens = [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]
    keyword_angles = [
        CARDINAL_NAME_TO_ANGLE[token]
        for token in tokens
        if token in CARDINAL_NAME_TO_ANGLE
    ]
    if not keyword_angles:
        return None

    if len(set(keyword_angles)) != 1:
        return None
    return keyword_angles[0]


def _reference_parent_key(dataset_dir: Path, image: Path) -> str:
    try:
        rel = image.parent.relative_to(dataset_dir).as_posix()
        return rel if rel != "." else dataset_dir.name
    except ValueError:
        return image.parent.as_posix()


def _keyword_group_key(stem: str, angle: str) -> str:
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem.lower()) if token]
    base_tokens = []
    matched = 0
    for token in tokens:
        mapped = CARDINAL_NAME_TO_ANGLE.get(token)
        if mapped is None:
            base_tokens.append(token)
            continue
        if mapped != angle:
            return ""
        matched += 1

    if matched == 0:
        return ""
    return "_".join(base_tokens) if base_tokens else "__cardinal_keywords__"


def load_reference_images(dataset_dir: Path) -> dict[str, list[Path]]:
    if not dataset_dir.exists():
        logger.warning("Reference directory not found: %s", dataset_dir)
        return {}

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    grouped: dict[str, dict[str, Path]] = {}

    for image in sorted(dataset_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in extensions:
            continue

        stem = image.stem
        angle = _extract_cardinal_angle(stem)
        if angle is None:
            continue

        parent_key = _reference_parent_key(dataset_dir, image)
        suffix = f"_{angle}"
        if stem.endswith(suffix):
            base_stem = stem[: -len(suffix)]
            group_id = f"{parent_key}/{base_stem}" if base_stem else parent_key
        else:
            keyword_key = _keyword_group_key(stem, angle)
            if not keyword_key:
                continue
            group_id = f"{parent_key}/{keyword_key}"

        grouped.setdefault(group_id, {})
        grouped[group_id].setdefault(angle, image)

    complete_groups: dict[str, list[Path]] = {}
    for group_id, angle_map in grouped.items():
        if all(angle in angle_map for angle in CARDINAL_ANGLES):
            complete_groups[group_id] = [angle_map[angle] for angle in CARDINAL_ANGLES]

    if not complete_groups:
        logger.warning("No complete cardinal reference sets found in %s", dataset_dir)

    return complete_groups


def synthesize_image_grok(
    grok_api_key: str,
    image_path: Path,
    prompt: str,
    model: str = "grok-imagine-image",
    reference_path: Optional[
        Union[Path, List[Path]]
    ] = None,  # Can be a single Path or a list of Paths
    rate_limiter: Optional["RequestRateLimiter"] = None,
) -> Optional[bytes]:
    try:
        with image_path.open("rb") as handle:
            image_inputs: List[str] = [encode_image_to_base64(handle.read())]

        if reference_path:
            if isinstance(reference_path, list):
                # Multiple reference images (cardinal views)
                for ref in reference_path:
                    if ref.exists():
                        with ref.open("rb") as handle:
                            image_inputs.append(encode_image_to_base64(handle.read()))
            elif reference_path.exists():
                # Single reference image
                with reference_path.open("rb") as handle:
                    image_inputs.append(encode_image_to_base64(handle.read()))

        with Image.open(image_path) as source_image:
            expected_size = source_image.size

        if rate_limiter is not None:
            rate_limiter.acquire()

        data = generate_image(
            api_key=grok_api_key,
            model=model,
            prompt=prompt,
            input_images=image_inputs,
        )
        with Image.open(BytesIO(data)) as generated_image:
            generated_size = generated_image.size
        if generated_size != expected_size:
            try:
                data = resize_cover_center_crop(data, expected_size)
                logger.warning(
                    "Adjusted generated image size from %s to %s via cover-resize + center-crop",
                    generated_size,
                    expected_size,
                )
            except Exception as exc:
                logger.error(
                    "Failed to adjust generated image size: expected=%s got=%s error=%s",
                    expected_size,
                    generated_size,
                    exc,
                )
                return None
        return data

    except GrokImageAPIError as e:
        logger.error(f"Grok synthesis failed: {e}")
        return None

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None


def synthesize_image_gemini(
    gemini_api_key: str,
    image_path: Path,
    prompt: str,
    model: str = "gemini-3.1-flash-image-preview",
    reference_path: Optional[Union[Path, List[Path]]] = None,
    rate_limiter: Optional["RequestRateLimiter"] = None,
) -> Optional[bytes]:
    try:
        with image_path.open("rb") as handle:
            image_inputs: List[bytes] = [handle.read()]

        if reference_path:
            if isinstance(reference_path, list):
                for ref in reference_path:
                    if ref.exists():
                        with ref.open("rb") as handle:
                            image_inputs.append(handle.read())
            elif reference_path.exists():
                with reference_path.open("rb") as handle:
                    image_inputs.append(handle.read())

        with Image.open(image_path) as source_image:
            expected_size = source_image.size

        if rate_limiter is not None:
            rate_limiter.acquire()

        data = generate_gemini_image(
            api_key=gemini_api_key,
            model=model,
            prompt=prompt,
            input_images=image_inputs,
        )
        with Image.open(BytesIO(data)) as generated_image:
            generated_size = generated_image.size
        if generated_size != expected_size:
            try:
                data = resize_cover_center_crop(data, expected_size)
                logger.warning(
                    "Adjusted generated image size from %s to %s via cover-resize + center-crop",
                    generated_size,
                    expected_size,
                )
            except Exception as exc:
                logger.error(
                    "Failed to adjust generated image size: expected=%s got=%s error=%s",
                    expected_size,
                    generated_size,
                    exc,
                )
                return None
        return data
    except GeminiImageAPIError as e:
        logger.error(f"Gemini synthesis failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None


def _guess_image_mime_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".webp"}:
        return "image/webp"
    return "image/jpeg"


def _decode_inline_data_value(raw_data: object) -> bytes:
    if isinstance(raw_data, bytes):
        return raw_data
    if isinstance(raw_data, str):
        return base64.b64decode(raw_data)
    raise ValueError("Unsupported inline data type")


def decode_gemini_batch_result_line(line: str) -> Optional[bytes]:
    payload = json.loads(line)
    response = payload.get("response") or {}
    candidates = response.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            data = inline_data.get("data")
            if data:
                return _decode_inline_data_value(data)
    return None


def build_gemini_batch_request(
    *,
    request_id: str,
    source_file_uri: str,
    source_mime_type: str,
    prompt: str,
    reference_file_uris: Optional[List[str]] = None,
    reference_mime_types: Optional[List[str]] = None,
) -> dict:
    parts = [
        {
            "file_data": {
                "file_uri": source_file_uri,
                "mime_type": source_mime_type,
            }
        }
    ]
    reference_mime_types = reference_mime_types or []
    for idx, ref_uri in enumerate(reference_file_uris or []):
        ref_mime_type = (
            reference_mime_types[idx]
            if idx < len(reference_mime_types)
            else source_mime_type
        )
        parts.append(
            {
                "file_data": {
                    "file_uri": ref_uri,
                    "mime_type": ref_mime_type,
                }
            }
        )
    parts.append({"text": prompt})
    return {
        "key": request_id,
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": {"responseModalities": ["TEXT", "IMAGE"]},
        },
    }


def _upload_gemini_batch_file(client: genai.Client, image_path: Path):
    return client.files.upload(
        file=str(image_path),
        config=types.UploadFileConfig(
            display_name=image_path.name,
            mime_type=_guess_image_mime_type(image_path),
        ),
    )


def _poll_gemini_batch_job(client: genai.Client, batch_job, poll_interval: float = 5.0):
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "SUCCEEDED", "FAILED"}
    while True:
        state_obj = getattr(batch_job, "state", getattr(batch_job, "status", None))
        state = getattr(state_obj, "name", str(state_obj))
        if state in terminal_states:
            return batch_job
        time.sleep(poll_interval)
        getter = getattr(client.batches, "get", None) or getattr(
            client.batches, "retrieve", None
        )
        if getter is None:
            return batch_job
        batch_job = getter(
            name=getattr(batch_job, "name", getattr(batch_job, "id", None))
        )


def _download_gemini_batch_results(client: genai.Client, batch_job) -> str:
    dest = getattr(batch_job, "dest", None)
    file_name = getattr(dest, "file_name", None) or getattr(
        dest, "responses_file", None
    )
    if not file_name:
        raise RuntimeError("Batch job did not provide a destination file")
    downloader = getattr(client.files, "download", None)
    if downloader is None:
        raise RuntimeError("Gemini client does not support file downloads")
    try:
        downloaded = downloader(file=file_name)
    except TypeError:
        downloaded = downloader(file_name=file_name)
    if isinstance(downloaded, bytes):
        return downloaded.decode("utf-8")
    if isinstance(downloaded, str):
        return downloaded
    text = getattr(downloaded, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(downloaded, "decode"):
        return downloaded.decode("utf-8")
    return str(downloaded)


def _uploaded_file_uri(uploaded_file) -> str:
    return str(
        getattr(uploaded_file, "uri", None) or getattr(uploaded_file, "name", "")
    )


def _uploaded_file_mime_type(uploaded_file, fallback: str) -> str:
    return str(getattr(uploaded_file, "mime_type", None) or fallback)


PLACEMENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "can_insert": {"type": "boolean"},
        "confidence_low": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence_high": {"type": "number", "minimum": 0, "maximum": 1},
        "distance_preference": {
            "type": "string",
            "enum": [
                "very close (foreground, large size)",
                "medium distance (midground, average size)",
                "far away (background, small size)",
            ],
        },
        "guidance": {"type": "string"},
    },
    "required": [
        "can_insert",
        "confidence_low",
        "confidence_high",
        "distance_preference",
        "guidance",
    ],
}


def _safe_conf_interval(low: float, high: float) -> tuple[float, float]:
    low_v = max(0.0, min(1.0, float(low)))
    high_v = max(0.0, min(1.0, float(high)))
    if high_v < low_v:
        low_v, high_v = high_v, low_v
    return low_v, high_v


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


def plan_vehicle_placement(
    client: genai.Client,
    model_id: str,
    image_path: Path,
    max_retries: int,
    rate_limiter: RequestRateLimiter | None = None,
) -> dict:
    prompt = (
        "You are planning a synthetic vehicle insertion for a Google Street View image. "
        "This is a Street View panorama capture - the vehicle you add must look exactly as if it was captured by the Street View camera at that moment. "
        "Return ONLY JSON schema-compliant output. "
        "Assess if a realistic insertion is possible, prefer farther placement when plausible, "
        "and provide concise guidance about lighting, shadows, perspective, vehicle direction, and placement position. "
        "CRITICAL: Consider that vehicle orientation must match road perspective exactly - avoid awkward angles. "
        "Consider natural lane position - avoid centering the vehicle like an advertisement."
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
                    response_schema=PLACEMENT_PLAN_SCHEMA,
                    temperature=0.1,
                ),
            )
            payload = json.loads((response.text or "").strip())
            low, high = _safe_conf_interval(
                payload["confidence_low"], payload["confidence_high"]
            )
            return {
                "ok": True,
                "can_insert": bool(payload["can_insert"]),
                "confidence_low": low,
                "confidence_high": high,
                "distance_preference": payload["distance_preference"],
                "guidance": str(payload.get("guidance", "")).strip(),
            }
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "Placement planner attempt %s/%s failed for %s: %s",
                attempt,
                max_retries,
                image_path.name,
                exc,
            )
    return {
        "ok": False,
        "can_insert": False,
        "confidence_low": 0.0,
        "confidence_high": 0.0,
        "distance_preference": "far away (background, small size)",
        "guidance": f"planner_failed: {last_error}",
    }


# Thread-safe lock for manifest updates
manifest_lock = threading.Lock()


def process_image(
    client: genai.Client,
    image_provider: str,
    image_api_key: str,
    img_id: str,
    parent_row: dict,
    out_dir: Path,
    existing_ids: set,
    enforcement_candidates: set,
    reference_images: dict,
    model: str,
    seed: int,
    hard_negative: bool = False,
    variant_index: int = 0,
    planner_model: str = "gemini-3.1-pro-preview",
    planner_retries: int = 3,
    planner_min_confidence: float = 0.7,
    run_dir: Optional[Path] = None,
    rate_limiter: RequestRateLimiter | None = None,
) -> Tuple[int, list]:
    """Process a single image and return (count, new_rows)."""
    new_rows = []
    count = 0

    rng = random.Random(f"{seed}:{img_id}")
    raw_path = parent_row.get("file_path", "")
    if raw_path:
        parent_path = Path(raw_path)
        if not parent_path.is_absolute() and run_dir is not None:
            parent_path = run_dir / parent_path
    else:
        return (0, [])
    if not parent_path.exists():
        return (0, [])
    parent_box_count = int(parent_row.get("num_boxes_autogen", "0") or "0")
    has_existing_vehicle = parent_box_count > 0

    plan = plan_vehicle_placement(
        client,
        planner_model,
        parent_path,
        max_retries=max(1, planner_retries),
        rate_limiter=rate_limiter,
    )
    if not plan["can_insert"] or plan["confidence_high"] < planner_min_confidence:
        return (0, [])

    # Determine distance/scale for this batch
    distance = plan.get("distance_preference") or rng.choice(DISTANCE_OPTIONS)

    spatial_instruction = ""
    if hard_negative:
        spatial_instruction = "Crucially, place this vehicle in close proximity to the existing vehicle(s) in the scene, as if they are driving or parked near each other. "

    # Generate random_vehicle (no reference needed)
    random_edit_suffix = (
        "random_vehicle" if variant_index == 0 else f"random_vehicle_v{variant_index}"
    )
    synth_id_random = generate_image_id(img_id, random_edit_suffix)
    if synth_id_random not in existing_ids:
        random_vehicle_prompt, vehicle_color, vehicle_body_type = (
            build_random_vehicle_prompt(img_id, seed, distance=distance)
        )
        if plan.get("guidance"):
            random_vehicle_prompt = (
                f"{random_vehicle_prompt} Additional guidance: {plan['guidance']}"
            )
        logger.info(
            "Creating random_vehicle for %s with style: %s %s at %s",
            img_id,
            vehicle_color,
            vehicle_body_type,
            distance,
        )
        synthesize_fn = (
            synthesize_image_gemini
            if image_provider == "gemini"
            else synthesize_image_grok
        )
        img_data = synthesize_fn(
            image_api_key,
            parent_path,
            random_vehicle_prompt,
            model,
            rate_limiter=rate_limiter,
        )

        if img_data:
            out_path = out_dir / "random_vehicle" / f"{synth_id_random}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(img_data)

            new_rows.append(
                {
                    "image_id": synth_id_random,
                    "file_path": str(out_path),
                    "split": "unset",
                    "parent_image_id": img_id,
                    "is_synthetic": "1",
                    "edit_type": "random_vehicle",
                    "expected_inserted_class": "vehicle",
                    "street": parent_row.get("street", ""),
                    "input_location": parent_row.get("input_location", ""),
                    "heading": parent_row.get("heading", ""),
                    "pitch": parent_row.get("pitch", ""),
                    "fov": parent_row.get("fov", ""),
                    "pano_id": parent_row.get("pano_id", ""),
                    "pano_lat": parent_row.get("pano_lat", ""),
                    "pano_lng": parent_row.get("pano_lng", ""),
                    "status": "ok",
                    "num_boxes_autogen": "0",
                    "needs_review": "1",
                    "review_status": "todo",
                    "review_bucket": "EV-SNV" if has_existing_vehicle else "SVO-NV",
                    "variant_index": str(variant_index),
                    "created_at": datetime.now().isoformat(),
                }
            )
            count += 1

    # Generate enforcement classes if this is an enforcement candidate
    if img_id in enforcement_candidates:
        for edit_type, prompt, reference_key in [
            ("enforcement_vehicle", ENFORCEMENT_PROMPT_WITH_REF, "enforcement"),
            ("police_old", POLICE_OLD_PROMPT_WITH_REF, "police_old"),
            ("police_new", POLICE_NEW_PROMPT_WITH_REF, "police_new"),
        ]:
            enf_suffix = (
                edit_type if variant_index == 0 else f"{edit_type}_v{variant_index}"
            )
            synth_id_enf = generate_image_id(img_id, enf_suffix)
            if synth_id_enf in existing_ids:
                continue

            # Get reference images group for this class (grouped by model)
            ref_groups = reference_images.get(reference_key, {})
            if not ref_groups:
                continue

            # Pick a random model group
            model_id = rng.choice(list(ref_groups.keys()))
            group_refs = ref_groups[model_id]

            # Format enforcement prompt with distance and spatial instructions
            formatted_prompt = prompt.format(
                distance=distance, spatial_instruction=spatial_instruction
            )
            if plan.get("guidance"):
                formatted_prompt = (
                    f"{formatted_prompt} Additional guidance: {plan['guidance']}"
                )

            logger.info(
                f"Creating {edit_type} for {img_id} at {distance} using model: {model_id} ({len(group_refs)} refs)"
                + (" (HARD NEGATIVE)" if hard_negative else "")
            )

            # Call synthesize_image with the list of references for this model
            synthesize_fn = (
                synthesize_image_gemini
                if image_provider == "gemini"
                else synthesize_image_grok
            )
            img_data = synthesize_fn(
                image_api_key,
                parent_path,
                formatted_prompt,
                model,
                reference_path=group_refs,
                rate_limiter=rate_limiter,
            )

            if img_data:
                out_path = out_dir / edit_type / f"{synth_id_enf}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(img_data)

                new_rows.append(
                    {
                        "image_id": synth_id_enf,
                        "file_path": str(out_path),
                        "split": "unset",
                        "parent_image_id": img_id,
                        "is_synthetic": "1",
                        "edit_type": edit_type,
                        "expected_inserted_class": edit_type,
                        "street": parent_row.get("street", ""),
                        "input_location": parent_row.get("input_location", ""),
                        "heading": parent_row.get("heading", ""),
                        "pitch": parent_row.get("pitch", ""),
                        "fov": parent_row.get("fov", ""),
                        "pano_id": parent_row.get("pano_id", ""),
                        "pano_lat": parent_row.get("pano_lat", ""),
                        "pano_lng": parent_row.get("pano_lng", ""),
                        "status": "ok",
                        "num_boxes_autogen": "0",
                        "needs_review": "1",
                        "review_status": "todo",
                        "review_bucket": "EV-SE" if has_existing_vehicle else "SVO-E",
                        "variant_index": str(variant_index),
                        "created_at": datetime.now().isoformat(),
                    }
                )
                count += 1

    return (count, new_rows)


def build_synthesis_tasks(
    *,
    client: genai.Client,
    image_provider: str,
    image_api_key: str,
    manifest_lookup: dict,
    empty_candidates: list[str],
    enforcement_candidates: set,
    reference_images: dict,
    seed: int,
    hard_negative_candidates: set,
    planner_model: str,
    planner_retries: int,
    planner_min_confidence: float,
    run_dir: Path,
    max_random_variants_per_image: int,
    max_enforcement_variants_per_image: int,
) -> list[dict]:
    tasks: list[dict] = []
    for img_id in empty_candidates:
        parent_row = manifest_lookup.get(img_id)
        if not parent_row:
            continue
        random_variants = max(1, max_random_variants_per_image)
        enforcement_variants = max(1, max_enforcement_variants_per_image)
        total_variants = max(random_variants, enforcement_variants)
        for variant_idx in range(total_variants):
            variant_seed = seed + variant_idx * 100_003
            parent_raw = parent_row.get("file_path", "")
            parent_path = Path(parent_raw)
            if not parent_path.is_absolute():
                parent_path = run_dir / parent_path
            if not parent_path.exists():
                continue
            plan = plan_vehicle_placement(
                client,
                planner_model,
                parent_path,
                max_retries=max(1, planner_retries),
            )
            if (
                not plan["can_insert"]
                or plan["confidence_high"] < planner_min_confidence
            ):
                continue
            distance = plan.get("distance_preference") or random.choice(
                DISTANCE_OPTIONS
            )
            spatial_instruction = (
                "Crucially, place this vehicle in close proximity to the existing vehicle(s) in the scene, as if they are driving or parked near each other. "
                if img_id in hard_negative_candidates
                else ""
            )
            task_base = {
                "img_id": img_id,
                "parent_row": parent_row,
                "parent_path": parent_path,
                "distance": distance,
                "guidance": plan.get("guidance", ""),
                "variant_index": variant_idx,
                "variant_seed": variant_seed,
                "spatial_instruction": spatial_instruction,
            }
            task_base["random_vehicle"] = {
                "enabled": True,
                "synth_id": generate_image_id(
                    img_id,
                    "random_vehicle"
                    if variant_idx == 0
                    else f"random_vehicle_v{variant_idx}",
                ),
            }
            if img_id in enforcement_candidates:
                task_base["enforcement"] = []
                for edit_type, reference_key in [
                    ("enforcement_vehicle", "enforcement"),
                    ("police_old", "police_old"),
                    ("police_new", "police_new"),
                ]:
                    ref_groups = reference_images.get(reference_key, {})
                    if not ref_groups:
                        continue
                    model_id = random.choice(list(ref_groups.keys()))
                    task_base["enforcement"].append(
                        {
                            "edit_type": edit_type,
                            "reference_key": reference_key,
                            "reference_model_id": model_id,
                            "reference_paths": ref_groups[model_id],
                            "synth_id": generate_image_id(
                                img_id,
                                edit_type
                                if variant_idx == 0
                                else f"{edit_type}_v{variant_idx}",
                            ),
                        }
                    )
            tasks.append(task_base)
    return tasks


def run_gemini_batch_generation(
    *,
    client: genai.Client,
    model: str,
    tasks: list[dict],
    reference_images: dict,
    out_dir: Path,
    manifest: list,
    manifest_path: Path,
    existing_ids: Optional[set] = None,
) -> int:
    existing_ids = existing_ids or set()
    unique_paths: dict[Path, object] = {}
    for task in tasks:
        unique_paths[task["parent_path"]] = None
        for group in task.get("enforcement") or []:
            for ref_path in group["reference_paths"]:
                unique_paths[ref_path] = None
    uploaded = {path: _upload_gemini_batch_file(client, path) for path in unique_paths}
    requests = []
    output_plan = []
    for task in tasks:
        source_upload = uploaded[task["parent_path"]]
        source_uri = _uploaded_file_uri(source_upload)
        source_mime_type = _uploaded_file_mime_type(
            source_upload, _guess_image_mime_type(task["parent_path"])
        )
        if task.get("random_vehicle"):
            prompt, _, _ = build_random_vehicle_prompt(
                task["img_id"], task["variant_seed"], distance=task["distance"]
            )
            if task.get("guidance"):
                prompt = f"{prompt} Additional guidance: {task['guidance']}"
            if task.get("spatial_instruction"):
                prompt = f"{prompt} {task['spatial_instruction']}"
            synth_id = task["random_vehicle"]["synth_id"]
            if synth_id not in existing_ids:
                requests.append(
                    build_gemini_batch_request(
                        request_id=synth_id,
                        source_file_uri=source_uri,
                        source_mime_type=source_mime_type,
                        prompt=prompt,
                    )
                )
                output_plan.append(
                    (
                        synth_id,
                        out_dir / "random_vehicle" / f"{synth_id}.jpg",
                        task["parent_row"],
                        task["img_id"],
                        "random_vehicle",
                        "vehicle",
                        task["variant_index"],
                    )
                )
        for enf in task.get("enforcement") or []:
            prompt_template = {
                "enforcement_vehicle": ENFORCEMENT_PROMPT_WITH_REF,
                "police_old": POLICE_OLD_PROMPT_WITH_REF,
                "police_new": POLICE_NEW_PROMPT_WITH_REF,
            }[enf["edit_type"]]
            prompt = prompt_template.format(
                distance=task["distance"],
                spatial_instruction=task["spatial_instruction"],
            )
            if task.get("guidance"):
                prompt = f"{prompt} Additional guidance: {task['guidance']}"
            synth_id = enf["synth_id"]
            if synth_id in existing_ids:
                continue
            reference_uploads = [uploaded[p] for p in enf["reference_paths"]]
            reference_uris = [
                _uploaded_file_uri(upload) for upload in reference_uploads
            ]
            reference_mime_types = [
                _uploaded_file_mime_type(upload, _guess_image_mime_type(path))
                for upload, path in zip(reference_uploads, enf["reference_paths"])
            ]
            request = build_gemini_batch_request(
                request_id=synth_id,
                source_file_uri=source_uri,
                source_mime_type=source_mime_type,
                prompt=prompt,
                reference_file_uris=reference_uris,
                reference_mime_types=reference_mime_types,
            )
            requests.append(request)
            output_plan.append(
                (
                    synth_id,
                    out_dir / enf["edit_type"] / f"{synth_id}.jpg",
                    task["parent_row"],
                    task["img_id"],
                    enf["edit_type"],
                    enf["edit_type"],
                    task["variant_index"],
                )
            )

    if not requests:
        logger.info("No Gemini batch requests to submit")
        return 0

    batch_input = out_dir / "gemini_batch_input.jsonl"
    batch_input.parent.mkdir(parents=True, exist_ok=True)
    with batch_input.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    uploaded_jsonl = client.files.upload(
        file=str(batch_input),
        config=types.UploadFileConfig(display_name=batch_input.name, mime_type="jsonl"),
    )
    batch_job = client.batches.create(
        model=model,
        src=uploaded_jsonl.name,
        config={"display_name": "synth_vehicle_edits"},
    )
    logger.info("Created Gemini batch job: %s", getattr(batch_job, "name", batch_job))
    batch_job = _poll_gemini_batch_job(client, batch_job)
    batch_state_obj = getattr(batch_job, "state", getattr(batch_job, "status", None))
    batch_state = getattr(batch_state_obj, "name", str(batch_state_obj))
    if batch_state not in {"JOB_STATE_SUCCEEDED", "SUCCEEDED"}:
        raise RuntimeError(
            f"Gemini batch job failed with state {batch_state}: {getattr(batch_job, 'error', '')}"
        )
    results_text = _download_gemini_batch_results(client, batch_job)
    results_by_id = {}
    for line in results_text.splitlines():
        line = line.strip()
        if not line:
            continue
        decoded = decode_gemini_batch_result_line(line)
        payload = json.loads(line)
        req_id = (
            payload.get("key")
            or payload.get("request_id")
            or payload.get("id")
            or (payload.get("metadata") or {}).get("key")
        )
        if not decoded and payload.get("error"):
            logger.warning(
                "Gemini batch request failed for %s: %s", req_id, payload["error"]
            )
        if decoded and req_id:
            results_by_id[req_id] = decoded

    count = 0
    for (
        synth_id,
        out_path,
        parent_row,
        parent_image_id,
        edit_type,
        expected_class,
        variant_index,
    ) in output_plan:
        img_data = results_by_id.get(synth_id)
        if not img_data:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(img_data)
        manifest.append(
            {
                "image_id": synth_id,
                "file_path": str(out_path),
                "split": "unset",
                "parent_image_id": parent_image_id,
                "is_synthetic": "1",
                "edit_type": edit_type,
                "expected_inserted_class": expected_class,
                "street": parent_row.get("street", ""),
                "input_location": parent_row.get("input_location", ""),
                "heading": parent_row.get("heading", ""),
                "pitch": parent_row.get("pitch", ""),
                "fov": parent_row.get("fov", ""),
                "pano_id": parent_row.get("pano_id", ""),
                "pano_lat": parent_row.get("pano_lat", ""),
                "pano_lng": parent_row.get("pano_lng", ""),
                "status": "ok",
                "num_boxes_autogen": "0",
                "needs_review": "1",
                "review_status": "todo",
                "review_bucket": "EV-SNV" if edit_type == "random_vehicle" else "EV-SE",
                "variant_index": str(variant_index),
                "created_at": datetime.now().isoformat(),
            }
        )
        count += 1
    save_manifest(manifest, manifest_path)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize vehicle edits with reference images"
    )
    parser.add_argument(
        "--run-dir",
        default=".",
        help="Run directory root (all relative paths resolve against this)",
    )
    parser.add_argument(
        "--manifest", "-m", default="manifests/images.csv", help="Input manifest"
    )
    parser.add_argument(
        "--empty-list",
        "-e",
        default="lists/valid_road_candidates.txt",
        help="Empty candidates file",
    )
    parser.add_argument(
        "--excluded-list",
        default="lists/excluded_from_synth.txt",
        help="Excluded from synthesis file",
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/images_synth", help="Output directory"
    )
    parser.add_argument(
        "--api-key",
        "-k",
        default=None,
        help="Gemini API key for planner model",
    )
    parser.add_argument(
        "--image-provider",
        choices=["gemini", "grok"],
        default="gemini",
        help="Image generation provider",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["batch", "realtime"],
        default="batch",
        help="Gemini generation mode (batch is the default)",
    )
    parser.add_argument(
        "--grok-api-key",
        "--flux-api-key",
        dest="grok_api_key",
        default=None,
        help="Grok API key (falls back to XAI_API_KEY / GROK_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Image model alias (defaults based on --image-provider)",
    )
    parser.add_argument(
        "--planner-model",
        default="gemini-3.1-pro-preview",
        help="Gemini text planner model",
    )
    parser.add_argument(
        "--planner-retries",
        type=int,
        default=3,
        help="Planner retry attempts",
    )
    parser.add_argument(
        "--planner-min-confidence",
        type=float,
        default=0.7,
        help="Min planner confidence upper bound",
    )
    parser.add_argument(
        "--enforcement-rate", type=float, default=0.2, help="Fraction for enforcement"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers", type=int, default=60, help="Number of parallel workers"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume from existing"
    )
    parser.add_argument(
        "--hard-negative-rate",
        type=float,
        default=0.3,
        help="Fraction of synth images to use hard negative strategy",
    )
    parser.add_argument(
        "--max-random-variants-per-image",
        type=int,
        default=1,
        help="How many normal-vehicle variants to generate per source image",
    )
    parser.add_argument(
        "--max-enforcement-variants-per-image",
        type=int,
        default=1,
        help="How many variants to generate for each enforcement edit type per source image",
    )
    parser.add_argument(
        "--max-non-road-hard-negatives",
        type=int,
        default=400,
        help="Max non-road images to retain as hard negatives",
    )
    parser.add_argument(
        "--non-road-list",
        default="lists/non_road_candidates.txt",
        help="Input list of non-road images",
    )
    parser.add_argument(
        "--non-road-keep-list",
        default="lists/non_road_keep.txt",
        help="Output list of retained non-road hard negatives",
    )
    parser.add_argument(
        "--non-road-drop-list",
        default="lists/non_road_drop.txt",
        help="Output list of non-road images not retained",
    )
    args = parser.parse_args()

    # Resolve run directory
    run_dir = Path(args.run_dir).resolve()

    def _resolve_path(raw_value: str) -> Path:
        candidate = Path(raw_value)
        return candidate if candidate.is_absolute() else run_dir / candidate

    def _resolve_image_path(raw_value: str) -> Path:
        if not raw_value:
            return run_dir / ""
        candidate = Path(raw_value)
        return candidate if candidate.is_absolute() else run_dir / candidate

    planner_api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not planner_api_key:
        logger.error("No planner API key. Set GEMINI_API_KEY or use --api-key")
        return

    image_provider = args.image_provider
    if image_provider == "grok":
        image_api_key = (
            args.grok_api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        )
        if not image_api_key:
            logger.error(
                "No Grok API key. Set XAI_API_KEY/GROK_API_KEY or --grok-api-key"
            )
            return
    else:
        image_api_key = os.getenv("GEMINI_API_KEY")
        if not image_api_key:
            logger.error("No Gemini API key. Set GEMINI_API_KEY or use --api-key")
            return

    if args.model:
        image_model = args.model
    else:
        image_model = (
            "gemini-3.1-flash-image-preview"
            if image_provider == "gemini"
            else "grok-imagine-image"
        )

    client = genai.Client(api_key=planner_api_key)

    manifest_path = _resolve_path(args.manifest)
    empty_list_path = _resolve_path(args.empty_list)
    excluded_list_path = _resolve_path(args.excluded_list)
    out_dir = _resolve_path(args.out_dir)
    non_road_list_path = _resolve_path(args.non_road_list)
    non_road_keep_list_path = _resolve_path(args.non_road_keep_list)
    non_road_drop_list_path = _resolve_path(args.non_road_drop_list)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    manifest = load_manifest(manifest_path)

    if empty_list_path.exists():
        with open(empty_list_path, "r") as f:
            empty_candidates = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"Empty list not found: {empty_list_path}")
        empty_candidates = []

    non_road_ids = []
    if non_road_list_path.exists():
        with open(non_road_list_path, "r") as f:
            non_road_ids = [line.strip() for line in f if line.strip()]

    random.shuffle(non_road_ids)
    keep_cap = max(0, args.max_non_road_hard_negatives)
    non_road_keep_ids = non_road_ids[:keep_cap]
    non_road_drop_ids = non_road_ids[keep_cap:]

    non_road_keep_list_path.parent.mkdir(parents=True, exist_ok=True)
    non_road_drop_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(non_road_keep_list_path, "w", encoding="utf-8") as f:
        for image_id in non_road_keep_ids:
            f.write(f"{image_id}\n")
    with open(non_road_drop_list_path, "w", encoding="utf-8") as f:
        for image_id in non_road_drop_ids:
            f.write(f"{image_id}\n")
    logger.info(
        "Non-road hard negatives retained=%s dropped=%s",
        len(non_road_keep_ids),
        len(non_road_drop_ids),
    )

    # Filter out excluded images
    excluded_ids = load_dataset_excluded_ids(manifest_path)
    if excluded_ids:
        original_count = len(empty_candidates)
        empty_candidates = [cid for cid in empty_candidates if cid not in excluded_ids]
        logger.info(
            f"Filtered {original_count - len(empty_candidates)} dataset-excluded images from synthesis"
        )

    # Build set of existing synthetic IDs for resume
    existing_ids = set()
    if args.resume:
        existing_ids = {
            row["image_id"] for row in manifest if row.get("is_synthetic") == "1"
        }
        logger.info(f"Resuming with {len(existing_ids)} existing synthetics")

    # Load reference images for each enforcement class
    reference_images = {
        "enforcement": load_reference_images(ENFORCEMENT_DATASET_DIR),
        "police_old": load_reference_images(POLICE_OLD_DATASET_DIR),
        "police_new": load_reference_images(POLICE_NEW_DATASET_DIR),
    }

    for ref_type, refs in reference_images.items():
        logger.info(f"Loaded {len(refs)} reference images for {ref_type}")

    # Verify we have reference images for enforcement classes
    for ref_type, refs in reference_images.items():
        if not refs:
            ref_dir = (
                ENFORCEMENT_DATASET_DIR
                if ref_type == "enforcement"
                else POLICE_OLD_DATASET_DIR
                if ref_type == "police_old"
                else POLICE_NEW_DATASET_DIR
            )
            logger.error(
                "No complete cardinal reference sets found for %s in %s. "
                "Expected 4-view sets (N/W/S/E suffixes or front/left/rear/right filenames).",
                ref_type,
                ref_dir,
            )
            return

    random.seed(args.seed)

    enforcement_candidates = set(
        random.sample(
            empty_candidates, k=int(len(empty_candidates) * args.enforcement_rate)
        )
        if empty_candidates
        else []
    )

    # Build lookup for manifest rows
    manifest_lookup = {row.get("image_id"): row for row in manifest}

    # Select hard negative candidates (images that already have vehicles)
    hard_negative_candidates = set()
    potential_hard_negs = [
        rid
        for rid, row in manifest_lookup.items()
        if (
            row.get("num_boxes_autogen")
            and str(row.get("num_boxes_autogen")).isdigit()
            and int(row.get("num_boxes_autogen")) > 0
        )
        and rid in empty_candidates
    ]

    if potential_hard_negs:
        num_hard_negs = int(len(potential_hard_negs) * args.hard_negative_rate)
        hard_negative_candidates = set(
            random.sample(potential_hard_negs, num_hard_negs)
        )
        logger.info(
            f"Selected {len(hard_negative_candidates)} hard negative candidates from {len(potential_hard_negs)} images with existing vehicles"
        )

    total_count = 0
    if image_provider == "gemini" and args.generation_mode == "batch":
        tasks = build_synthesis_tasks(
            client=client,
            image_provider=image_provider,
            image_api_key=image_api_key,
            manifest_lookup=manifest_lookup,
            empty_candidates=empty_candidates,
            enforcement_candidates=enforcement_candidates,
            reference_images=reference_images,
            seed=args.seed,
            hard_negative_candidates=hard_negative_candidates,
            planner_model=args.planner_model,
            planner_retries=args.planner_retries,
            planner_min_confidence=args.planner_min_confidence,
            run_dir=run_dir,
            max_random_variants_per_image=args.max_random_variants_per_image,
            max_enforcement_variants_per_image=args.max_enforcement_variants_per_image,
        )
        total_count = run_gemini_batch_generation(
            client=client,
            model=image_model,
            tasks=tasks,
            reference_images=reference_images,
            out_dir=out_dir,
            manifest=manifest,
            manifest_path=manifest_path,
            existing_ids=existing_ids,
        )
    else:
        if image_provider == "grok" and args.generation_mode == "batch":
            logger.warning(
                "Batch mode is only supported for Gemini; falling back to realtime for Grok"
            )
        rate_limiter = RequestRateLimiter(max_requests=280, period_seconds=60.0)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for img_id in empty_candidates:
                parent_row = manifest_lookup.get(img_id)
                if not parent_row:
                    continue

                random_variants = max(1, args.max_random_variants_per_image)
                enforcement_variants = max(1, args.max_enforcement_variants_per_image)
                total_variants = max(random_variants, enforcement_variants)

                for variant_idx in range(total_variants):
                    variant_seed = args.seed + variant_idx * 100_003
                    future = executor.submit(
                        process_image,
                        client,
                        image_provider,
                        image_api_key,
                        img_id,
                        parent_row,
                        out_dir,
                        existing_ids,
                        enforcement_candidates,
                        reference_images,
                        image_model,
                        variant_seed,
                        img_id in hard_negative_candidates,
                        variant_idx,
                        args.planner_model,
                        args.planner_retries,
                        args.planner_min_confidence,
                        run_dir,
                        rate_limiter,
                    )
                    futures[future] = img_id

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Synthesizing"
            ):
                count, new_rows = future.result()
                total_count += count
                if new_rows:
                    with manifest_lock:
                        manifest.extend(new_rows)
                        save_manifest(manifest, manifest_path)
                        for row in new_rows:
                            existing_ids.add(row["image_id"])

    logger.info(f"Created {total_count} synthetic images")
    logger.info(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
