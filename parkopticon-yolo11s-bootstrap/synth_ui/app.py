#!/usr/bin/env python3
"""Synthetic data harness for quick, manual generation checks.

Goal:
- Select daylight Street View background images
- Pick target class (random/enforcement/police)
- Optionally provide a reference vehicle image
- Generate and visually compare before/after results
"""

import csv
import os
import shutil
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file
from google import genai
from google.genai import types
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import resize_cover_center_crop

from utils.preprocessing import resize_cover_center_crop


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MANIFEST_PATH = PROJECT_ROOT / "manifests" / "images.csv"
EMPTY_CANDIDATES_PATH = PROJECT_ROOT / "lists" / "empty_candidates.txt"
EXCLUDED_FROM_SYNTH_PATH = PROJECT_ROOT / "lists" / "excluded_from_synth.txt"
SYNTH_OUTPUT_DIR = PROJECT_ROOT / "data" / "images_synth"
DELETED_IMAGES_DIR = PROJECT_ROOT / "deleted_images"
DELETED_IMAGES_LOG_PATH = DELETED_IMAGES_DIR / "deleted_images_log.csv"

ENFORCEMENT_DATASET_DIR = PROJECT_ROOT / "enforcement-dataset"
POLICE_OLD_DATASET_DIR = PROJECT_ROOT / "police-dataset-old"
POLICE_NEW_DATASET_DIR = PROJECT_ROOT / "police-dataset-new"

PRIMARY_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_MODEL", "models/gemini-3-pro-image-preview"
)
FALLBACK_IMAGE_MODEL = os.getenv(
    "GEMINI_IMAGE_FALLBACK", "models/gemini-2.5-flash-image"
)
EDIT_TYPES = {"random_vehicle", "enforcement_vehicle", "police_old", "police_new"}

STRICT_SIZE_CONSTRAINT = (
    "Critical output constraint: return an image with exactly the same width and height "
    "as the input street scene. Do not crop, resize, pad, rotate, or change aspect ratio."
)

SCENE_LOCK_CONSTRAINT = (
    "Scene integrity constraint: preserve the original scene geometry and composition exactly. "
    "Do not move or alter camera pose, horizon, vanishing points, road/curb alignment, lane markings, "
    "buildings, trees, poles, signs, parked objects, or any static landmark. "
    "Do not shift, warp, redraw, or restyle the background."
)

LOCAL_EDIT_ONLY_CONSTRAINT = (
    "Local-edit-only constraint: modify only the minimal pixel region required to insert the new vehicle, "
    "plus immediate physically consistent contact shadow/occlusion near that vehicle. "
    "All other pixels should remain visually unchanged. If constraints cannot be satisfied, return the original scene unchanged."
)

SHADOW_CONSISTENCY_CONSTRAINT = (
    "Shadow consistency constraint: shadow intensity and sharpness must match the scene's apparent lighting conditions. "
    "Bright sunny scenes should cast sharp, dark shadows. Overcast, cloudy, or diffuse lighting scenes should have soft, faint, or no visible shadows. "
    "Do not add hard, distinct shadows when the scene appears overcast or has soft ambient lighting."
)

TRAFFIC_DIRECTION_CONSTRAINT = (
    "Orientation constraint: the inserted vehicle must be aligned with the correct lane direction "
    "for this road scene, matching traffic flow and lane markings. Do not place a wrong-way vehicle."
)

RANDOM_VEHICLE_PROMPT = (
    "Edit this street image by adding exactly one normal passenger vehicle on the roadway. "
    "Keep camera perspective, scale, lighting, shadows, and color grading consistent. "
    "Do not alter buildings, signs, roads, or sky beyond what is required to insert the vehicle. "
f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

ENFORCEMENT_PROMPT_NO_REF = (
    "Edit this street image by adding exactly one parking enforcement vehicle on the roadway. "
    "Use a realistic bylaw/enforcement style (light bar, municipal markings, no real logos). "
    "Keep perspective, lighting, shadows, and scene geometry consistent. "
f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

ENFORCEMENT_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference enforcement vehicle. "
    "Insert one enforcement vehicle into the target scene, matching the reference style while preserving scene realism. "
    "Do not introduce unrelated scene changes. "
f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

POLICE_OLD_PROMPT_NO_REF = (
    "Edit this street image by adding exactly one Ottawa Police cruiser with OLD livery on the roadway. "
    "The old livery has a white base with blue and yellow stripes on the sides. "
    "Include a roof lightbar. Keep perspective, lighting, shadows, and scene geometry consistent. "
    f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

POLICE_OLD_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with OLD livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Do not introduce unrelated scene changes. "
    f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

POLICE_NEW_PROMPT_NO_REF = (
    "Edit this street image by adding exactly one Ottawa Police cruiser with NEW livery on the roadway. "
    "The new livery has a dark base with reflective yellow/gold chevron stripes and modern markings. "
    "Include a roof lightbar. Keep perspective, lighting, shadows, and scene geometry consistent. "
    f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)

POLICE_NEW_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with NEW livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Do not introduce unrelated scene changes. "
    f"{SCENE_LOCK_CONSTRAINT} "
    f"{LOCAL_EDIT_ONLY_CONSTRAINT} "
    f"{TRAFFIC_DIRECTION_CONSTRAINT} "
    f"{SHADOW_CONSISTENCY_CONSTRAINT} "
    f"{STRICT_SIZE_CONSTRAINT}"
)
app = Flask(__name__)


def _normalize_under_root(path: Path) -> Path | None:
    root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _resolve_data_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    normalized = _normalize_under_root(candidate)
    if normalized and normalized.exists() and normalized.is_file():
        return normalized
    return None


def _relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _file_url(path: Path) -> str:
    return f"/files/{quote(_relative_posix(path))}"


def _load_manifest_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_manifest_image_path(row: dict[str, str]) -> Path | None:
    direct = _resolve_data_path((row.get("file_path") or "").strip())
    if direct:
        return direct
    source = _resolve_data_path((row.get("source_file_path") or "").strip())
    if source:
        return source
    return None


def _load_empty_ids() -> set[str]:
    if not EMPTY_CANDIDATES_PATH.exists():
        return set()
    with open(EMPTY_CANDIDATES_PATH, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _save_empty_ids(empty_ids: set[str]) -> None:
    EMPTY_CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{EMPTY_CANDIDATES_PATH.name}.",
        suffix=".tmp",
        dir=str(EMPTY_CANDIDATES_PATH.parent),
    )
    os.close(fd)
    sorted_ids = sorted({value.strip() for value in empty_ids if value.strip()})
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            for image_id in sorted_ids:
                handle.write(f"{image_id}\n")
        os.replace(tmp_name, EMPTY_CANDIDATES_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
def _load_excluded_ids() -> set[str]:
    if not EXCLUDED_FROM_SYNTH_PATH.exists():
        return set()
    with open(EXCLUDED_FROM_SYNTH_PATH, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _save_excluded_ids(excluded_ids: set[str]) -> None:
    EXCLUDED_FROM_SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{EXCLUDED_FROM_SYNTH_PATH.name}.",
        suffix=".tmp",
        dir=str(EXCLUDED_FROM_SYNTH_PATH.parent),
    )
    os.close(fd)
    sorted_ids = sorted({value.strip() for value in excluded_ids if value.strip()})
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            for image_id in sorted_ids:
                handle.write(f"{image_id}\n")
        os.replace(tmp_name, EXCLUDED_FROM_SYNTH_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _append_excluded_log(row: dict[str, str]) -> None:
    """Log excluded image metadata for tracking purposes."""
    pass  # Optional: implement if tracking needed

def _append_deleted_log(row: dict[str, str]) -> None:
    DELETED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "deleted_at",
        "image_id",
        "pano_id",
        "heading",
        "street",
        "original_relative_path",
        "deleted_relative_path",
        "source_file_relative_path",
    ]
    write_header = not DELETED_IMAGES_LOG_PATH.exists()
    with open(DELETED_IMAGES_LOG_PATH, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def _find_manifest_row(image_id: str) -> dict[str, str] | None:
    for candidate in _load_manifest_rows():
        if (candidate.get("image_id") or "").strip() == image_id:
            return candidate
    return None


def _safe_move_to_deleted(image_path: Path, image_id: str) -> Path:
    DELETED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target = DELETED_IMAGES_DIR / image_path.name
    if target.exists():
        stem = image_path.stem
        suffix = image_path.suffix
        target = DELETED_IMAGES_DIR / f"{stem}_{image_id}{suffix}"
    if target.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = (
            DELETED_IMAGES_DIR
            / f"{image_path.stem}_{image_id}_{timestamp}{image_path.suffix}"
        )
    shutil.move(str(image_path), str(target))
    return target


def _build_image_record(row: dict[str, str], image_path: Path) -> dict[str, str]:
    image_id = (row.get("image_id") or "").strip()
    street = (row.get("street") or "").strip()
    heading = (row.get("heading") or "").strip()
    pano_id = (row.get("pano_id") or "").strip()

    label_parts = [street or image_id]
    if heading:
        label_parts.append(f"h{heading}")

    return {
        "id": image_id,
        "label": " | ".join(label_parts),
        "pano_id": pano_id,
        "heading": heading,
        "url": _file_url(image_path),
        "relative_path": _relative_posix(image_path),
    }


def _list_reference_images(dataset_dir: Path) -> list[dict[str, str]]:
    if not dataset_dir.exists():
        return []

    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(dataset_dir.glob(pattern))

    files = sorted({p.resolve() for p in files if p.is_file()})
    records: list[dict[str, str]] = []
    for path in files:
        records.append(
            {
                "name": path.name,
                "url": _file_url(path),
                "relative_path": _relative_posix(path),
            }
        )
    return records


def _pil_copy(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.copy()


def _generate_with_model(
    client: genai.Client, model_name: str, contents: list[Any]
) -> bytes:
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(response_modalities=["image"]),
    )
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("No candidates returned by model.")

    parts = getattr(candidates[0].content, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            return inline_data.data
    raise RuntimeError("No image bytes returned by model.")


def _build_contents(
    edit_type: str, background_path: Path, reference_path: Path | None
) -> list[Any]:
    background_image = _pil_copy(background_path)

    if edit_type == "random_vehicle":
        return [RANDOM_VEHICLE_PROMPT, background_image]

    if edit_type == "enforcement_vehicle":
        if reference_path:
            reference_image = _pil_copy(reference_path)
            return [ENFORCEMENT_PROMPT_WITH_REF, background_image, reference_image]
        return [ENFORCEMENT_PROMPT_NO_REF, background_image]

    if edit_type == "police_old":
        if reference_path:
            reference_image = _pil_copy(reference_path)
            return [POLICE_OLD_PROMPT_WITH_REF, background_image, reference_image]
        return [POLICE_OLD_PROMPT_NO_REF, background_image]

    if edit_type == "police_new":
        if reference_path:
            reference_image = _pil_copy(reference_path)
            return [POLICE_NEW_PROMPT_WITH_REF, background_image, reference_image]
        return [POLICE_NEW_PROMPT_NO_REF, background_image]

    raise ValueError(f"Unsupported edit_type: {edit_type}")


def _resolve_reference_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    cleaned = raw_value.strip()
    if cleaned.startswith("/files/"):
        cleaned = cleaned[len("/files/") :]

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    normalized = _normalize_under_root(candidate)
    if not normalized or not normalized.exists() or not normalized.is_file():
        return None
    return normalized


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment/.env")
    return genai.Client(api_key=api_key)



def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@app.route("/")
def index() -> Any:
    index_path = BASE_DIR / "index.html"
    if not index_path.exists():
        abort(500, description="index.html is missing from synth_ui/")
    response = send_file(index_path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/review-empty")
def review_empty_page() -> Any:
    review_path = BASE_DIR / "review_empty.html"
    if not review_path.exists():
        abort(500, description="review_empty.html is missing from synth_ui/")
    response = send_file(review_path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/files/<path:relative_path>")
def files(relative_path: str) -> Any:
    target = _normalize_under_root(PROJECT_ROOT / relative_path)
    if not target or not target.exists() or not target.is_file():
        abort(404)
    response = send_file(target)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/api/models")
def api_models() -> Any:
    return jsonify(
        {
            "primary": PRIMARY_IMAGE_MODEL,
            "fallback": FALLBACK_IMAGE_MODEL,
            "goal": "Create and inspect synthetic daylight edits before bulk generation.",
        }
    )


@app.route("/api/images")
def api_images() -> Any:
    source = (request.args.get("source") or "empty").strip().lower()
    try:
        limit = max(1, min(500, int(request.args.get("limit", "120"))))
    except ValueError:
        limit = 120

    empty_ids = _load_empty_ids()
    excluded_ids = _load_excluded_ids()
    rows = _load_manifest_rows()
    rows = _load_manifest_rows()
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in rows:
        image_id = (row.get("image_id") or "").strip()
        if not image_id or image_id in seen:
            continue
        if (row.get("status") or "").strip().lower() != "ok":
            continue
        if (row.get("is_synthetic") or "0").strip() == "1":
            continue
        if source == "empty" and image_id not in empty_ids:
            continue
        if image_id in excluded_ids:
            continue
            continue

        image_path = _resolve_manifest_image_path(row)
        if not image_path:
            continue

        records.append(_build_image_record(row, image_path))
        seen.add(image_id)
        if len(records) >= limit:
            break

    return jsonify(records)


    return jsonify(
        {
            "enforcement": _list_reference_images(ENFORCEMENT_DATASET_DIR),
            "police_old": _list_reference_images(POLICE_OLD_DATASET_DIR),
            "police_new": _list_reference_images(POLICE_NEW_DATASET_DIR),
        }
    )

@app.route("/api/empty-review")
def api_empty_review() -> Any:
    try:
        limit = max(1, min(2000, int(request.args.get("limit", "1000"))))
    except ValueError:
        limit = 1000

    empty_ids = _load_empty_ids()
    excluded_ids = _load_excluded_ids()
    records: list[dict[str, str]] = []
    
    # First pass: count all empty images (not limited)
    total_empty = 0
    total_excluded = 0
    seen_ids: set[str] = set()
    
    for row in _load_manifest_rows():
        image_id = (row.get("image_id") or "").strip()
        if not image_id or image_id not in empty_ids:
            continue
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        
        total_empty += 1
        if image_id in excluded_ids:
            total_excluded += 1
    
    # Second pass: get available records up to limit
    seen_ids = set()
    for row in _load_manifest_rows():
        image_id = (row.get("image_id") or "").strip()
        if not image_id or image_id not in empty_ids:
            continue
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        
        if image_id in excluded_ids:
            continue

        image_path = _resolve_manifest_image_path(row)
        if not image_path:
            continue

        records.append(
            {
                "image_id": image_id,
                "pano_id": (row.get("pano_id") or "").strip(),
                "heading": (row.get("heading") or "").strip(),
                "street": (row.get("street") or "").strip(),
                "url": _file_url(image_path),
                "relative_path": _relative_posix(image_path),
                "source_file_path": (row.get("source_file_path") or "").strip(),
            }
        )
        if len(records) >= limit:
            break

    return jsonify({
        "items": records,
        "counts": {
            "total_empty": total_empty,
            "excluded": total_excluded,
            "available": total_empty - total_excluded,
        }
    })


@app.route("/api/empty-review/exclude", methods=["POST"])
def api_empty_review_exclude() -> Any:
    payload = request.get_json(silent=True) or {}
    image_ids = payload.get("image_ids") or [payload.get("image_id")]
    image_ids = [str(iid).strip() for iid in image_ids if str(iid).strip()]
    
    if not image_ids:
        return jsonify({"error": "image_id or image_ids is required"}), 400

    excluded_ids = _load_excluded_ids()
    empty_ids = _load_empty_ids()
    
    added_count = 0
    for image_id in image_ids:
        if image_id in empty_ids and image_id not in excluded_ids:
            excluded_ids.add(image_id)
            added_count += 1
    
    _save_excluded_ids(excluded_ids)
    
    return jsonify(
        {
            "success": True,
            "excluded_count": added_count,
            "image_ids": image_ids,
        }
    )


@app.route("/api/empty-review/unexclude", methods=["POST"])
def api_empty_review_unexclude() -> Any:
    payload = request.get_json(silent=True) or {}
    image_ids = payload.get("image_ids") or [payload.get("image_id")]
    image_ids = [str(iid).strip() for iid in image_ids if str(iid).strip()]
    
    if not image_ids:
        return jsonify({"error": "image_id or image_ids is required"}), 400

    excluded_ids = _load_excluded_ids()
    
    removed_count = 0
    for image_id in image_ids:
        if image_id in excluded_ids:
            excluded_ids.discard(image_id)
            removed_count += 1
    
    _save_excluded_ids(excluded_ids)
    
    return jsonify(
        {
            "success": True,
            "unexcluded_count": removed_count,
            "image_ids": image_ids,
        }
    )


@app.route("/api/empty-review/excluded")
def api_empty_review_excluded() -> Any:
    """Return list of excluded image IDs."""
    excluded_ids = _load_excluded_ids()
    return jsonify(list(excluded_ids))

@app.route("/api/generate", methods=["POST"])
def api_generate() -> Any:
    payload = request.get_json(silent=True) or {}
    image_id = str(payload.get("image_id") or "").strip()
    edit_type = str(payload.get("edit_type") or "").strip()
    reference_raw = payload.get("reference_path") or payload.get("reference_image")
    auto_crop_to_original = _to_bool(payload.get("auto_crop_to_original"))

    if not image_id:
        return jsonify({"error": "image_id is required"}), 400
    if edit_type not in EDIT_TYPES:
        return jsonify(
            {"error": f"edit_type must be one of: {sorted(EDIT_TYPES)}"}
        ), 400

    row = None
    for candidate in _load_manifest_rows():
        if (candidate.get("image_id") or "").strip() == image_id:
            row = candidate
            break
    if row is None:
        return jsonify({"error": f"image_id not found in manifest: {image_id}"}), 404

    background_path = _resolve_manifest_image_path(row)
    if not background_path:
        return jsonify({"error": "Background image file not found on disk"}), 404

    reference_path = _resolve_reference_path(
        str(reference_raw) if reference_raw else None
    )

    try:
        client = _get_client()
        contents = _build_contents(edit_type, background_path, reference_path)

        models_to_try = [PRIMARY_IMAGE_MODEL]
        if FALLBACK_IMAGE_MODEL and FALLBACK_IMAGE_MODEL not in models_to_try:
            models_to_try.append(FALLBACK_IMAGE_MODEL)

        image_bytes: bytes | None = None
        used_model = ""
        errors: list[str] = []

        for model_name in models_to_try:
            try:
                image_bytes = _generate_with_model(client, model_name, contents)
                used_model = model_name
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{model_name}: {exc}")

        if image_bytes is None:
            return jsonify({"error": "Generation failed", "details": errors}), 500

        with Image.open(background_path) as background_image:
            expected_size = background_image.size
        with Image.open(BytesIO(image_bytes)) as generated_image:
            generated_size = generated_image.size

        size_mismatch = generated_size != expected_size
        auto_crop_applied = False
        warning: str | None = None
        output_bytes = image_bytes

        if size_mismatch:
            warning = f"Generated size {generated_size} differs from original {expected_size}."

            if auto_crop_to_original:
                try:
                    output_bytes = resize_cover_center_crop(image_bytes, expected_size)
                    auto_crop_applied = True
                    warning = (
                        f"Generated size {generated_size} differed from original {expected_size}; "
                        "resize-to-cover + center-crop was applied to match original size."
                    )
                except Exception as exc:  # noqa: BLE001
                    warning = (
                        f"Generated size {generated_size} differs from original {expected_size}; "
                        f"auto-crop failed ({exc}). Showing raw generated image."
                    )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = SYNTH_OUTPUT_DIR / edit_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_suffix = ""
        if size_mismatch and auto_crop_applied:
            output_suffix = "_autocrop"
        elif size_mismatch:
            output_suffix = "_size_mismatch"

        output_path = (
            output_dir / f"{image_id}_{edit_type}_{timestamp}{output_suffix}.jpg"
        )

        with open(output_path, "wb") as handle:
            handle.write(output_bytes)

        return jsonify(
            {
                "success": True,
                "model_used": used_model,
                "result_url": _file_url(output_path),
                "result_relative_path": _relative_posix(output_path),
                "expected_size": [expected_size[0], expected_size[1]],
                "generated_size": [generated_size[0], generated_size[1]],
                "size_mismatch": size_mismatch,
                "auto_crop_requested": auto_crop_to_original,
                "auto_crop_applied": auto_crop_applied,
                "warning": warning,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print(
        "Synthetic Harness Goal: generate and inspect synthetic edits before bulk runs"
    )
    print(f"Primary model: {PRIMARY_IMAGE_MODEL}")
    print(f"Fallback model: {FALLBACK_IMAGE_MODEL}")
    print("Open http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
