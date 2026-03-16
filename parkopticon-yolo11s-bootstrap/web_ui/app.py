import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Literal, cast
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.flux_api import (
    FluxAPIError,
    encode_image_to_base64 as encode_flux_image_to_base64,
    generate_image as generate_flux_image,
)
from utils.gemini_image_api import (
    GeminiImageAPIError,
    GeminiImageResult,
    generate_image_with_usage as generate_gemini_image_with_usage,
)
from utils.grok_image_api import (
    GrokImageAPIError,
    encode_image_to_base64 as encode_grok_image_to_base64,
    generate_image as generate_grok_image,
)

load_dotenv()

# Extract --run-dir before FastAPI initialization if it exists
_run_dir_str = "."
for i, arg in enumerate(sys.argv):
    if arg == "--run-dir" and i + 1 < len(sys.argv):
        _run_dir_str = sys.argv[i + 1]

# Mutable run directory state (can be changed via UI)
_current_run_dir = Path(_run_dir_str).resolve()


def _get_run_dir() -> Path:
    return _current_run_dir


def _set_run_dir(path: Path) -> None:
    global _current_run_dir
    _current_run_dir = path.resolve()


# Dynamic path helpers that use current run directory
def _run_path(*parts: str) -> Path:
    return _get_run_dir() / Path(*parts)


# Property-like functions for dynamic paths
def _manifest_path() -> Path:
    return _run_path("manifests", "images.csv")


def _labels_autogen_dir() -> Path:
    return _run_path("data", "labels_autogen")


def _labels_final_dir() -> Path:
    return _run_path("data", "labels_final")


def _presets_file() -> Path:
    return _run_path("manifests", "custom_presets.json")


def _cost_tracker_file() -> Path:
    return _run_path("manifests", "provider_costs.json")


def _load_presets() -> dict:
    """Load custom presets from JSON file."""
    presets_file = _presets_file()
    if presets_file.exists():
        try:
            with open(presets_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_presets(presets: dict) -> None:
    """Save custom presets to JSON file."""
    presets_file = _presets_file()
    presets_file.parent.mkdir(parents=True, exist_ok=True)
    with open(presets_file, "w") as f:
        json.dump(presets, f, indent=2)


def _prompts_file() -> Path:
    return PROJECT_ROOT / "prompts" / "touchup_prompts.json"


def _load_touchup_prompts() -> dict:
    """Load touchup prompts from JSON file."""
    prompts_file = _prompts_file()
    if prompts_file.exists():
        try:
            with open(prompts_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"prompts": []}
    return {"prompts": []}


def _save_touchup_prompts(data: dict) -> None:
    """Save touchup prompts to JSON file."""
    prompts_file = _prompts_file()
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_file, "w") as f:
        json.dump(data, f, indent=2)


def _images_original_dir() -> Path:
    return _run_path("data", "images_original")


def _images_synth_dir() -> Path:
    return _run_path("data", "images_synth")


def _excluded_from_synth_path() -> Path:
    return _run_path("lists", "excluded_from_synth.txt")


def _empty_candidates_path() -> Path:
    return _run_path("lists", "empty_candidates.txt")


def _deleted_images_dir() -> Path:
    return _run_path("deleted_images")


def _deleted_images_log_path() -> Path:
    return _deleted_images_dir() / "deleted_images_log.csv"


def _trash_root() -> Path:
    return _deleted_images_dir() / "synthetic_cleanup"


def _delete_log() -> Path:
    return _trash_root() / "deleted_synthetics.csv"


ENFORCEMENT_DATASET_DIR = PROJECT_ROOT / "enforcement-dataset"
POLICE_OLD_DATASET_DIR = PROJECT_ROOT / "police-dataset-old"
POLICE_NEW_DATASET_DIR = PROJECT_ROOT / "police-dataset-new"

SUPPORTED_IMAGE_PROVIDERS = {"grok", "gemini", "flux"}

PRIMARY_IMAGE_MODEL = os.getenv("GROK_IMAGE_MODEL", "grok-imagine-image")
FALLBACK_IMAGE_MODEL = os.getenv("GROK_IMAGE_FALLBACK", "")

GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
GEMINI_IMAGE_FALLBACK = os.getenv("GEMINI_IMAGE_FALLBACK", "")

FLUX_IMAGE_MODEL = os.getenv("FLUX_IMAGE_MODEL", "black-forest-labs/flux.2-pro")
FLUX_IMAGE_FALLBACK = os.getenv("FLUX_IMAGE_FALLBACK", "")

DEFAULT_SYNTH_PROVIDER = (os.getenv("SYNTH_IMAGE_PROVIDER") or "grok").strip().lower()
if DEFAULT_SYNTH_PROVIDER not in SUPPORTED_IMAGE_PROVIDERS:
    DEFAULT_SYNTH_PROVIDER = "grok"

DEFAULT_TOUCHUP_PROVIDER = (
    (os.getenv("TOUCHUP_IMAGE_PROVIDER") or "gemini").strip().lower()
)
if DEFAULT_TOUCHUP_PROVIDER not in SUPPORTED_IMAGE_PROVIDERS:
    DEFAULT_TOUCHUP_PROVIDER = "gemini"

ImageProvider = Literal["grok", "gemini", "flux"]
EDIT_TYPES = {"random_vehicle", "enforcement_vehicle", "police_old", "police_new"}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_cost_tracker_lock = threading.Lock()

app = FastAPI(title="ParkOpticon Web UI")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --- Directory Switching API ---


@app.get("/api/directories/list")
def list_available_directories():
    """List available run directories under PROJECT_ROOT."""
    candidates = []

    # Always include project root
    if PROJECT_ROOT.exists():
        candidates.append(
            {"name": ".", "path": str(PROJECT_ROOT), "label": "Project Root"}
        )

    runs_dir = PROJECT_ROOT / "runs"
    if runs_dir.exists():
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                candidates.append(
                    {
                        "name": f"runs/{d.name}",
                        "path": str(d),
                        "label": f"runs/{d.name}",
                    }
                )

    # Also check for Benchmark directories in PROJECT_ROOT
    for d in sorted(PROJECT_ROOT.iterdir()):
        if d.is_dir() and d.name.startswith("Benchmark"):
            candidates.append({"name": d.name, "path": str(d), "label": d.name})

    return candidates


@app.get("/api/directories/current")
def get_current_directory():
    """Get the current run directory."""
    return {"path": str(_get_run_dir()), "name": _get_run_dir().name}


class SetDirectoryRequest(BaseModel):
    path: str


@app.post("/api/directories/set")
def set_current_directory(req: SetDirectoryRequest):
    """Set the current run directory."""
    new_path = Path(req.path).resolve()
    if not new_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.path}")
    if not new_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.path}")

    _set_run_dir(new_path)
    return {"path": str(_get_run_dir()), "name": _get_run_dir().name}


# --- Helper functions (from labeler, synth_ui, synth_cleanup) ---


def _normalize_under_root(path: Path, root: Optional[Path] = None) -> Path | None:
    if root is None:
        root = PROJECT_ROOT
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def _resolve_data_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        candidate = _get_run_dir() / candidate
    normalized = _normalize_under_root(candidate)
    if normalized and normalized.exists() and normalized.is_file():
        return normalized
    return None


def _relative_posix(path: Path, root: Optional[Path] = None) -> str:
    if root is None:
        root = PROJECT_ROOT
    return path.resolve().relative_to(root.resolve()).as_posix()


def _file_url(path: Path) -> str:
    return f"/files/{quote(_relative_posix(path))}"


def load_manifest() -> List[dict]:
    if not _manifest_path().exists():
        return []
    with open(_manifest_path(), "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: List[dict]):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    with open(_manifest_path(), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def get_image_path(row: dict) -> Optional[Path]:
    file_path = row.get("file_path", "")
    if file_path:
        return Path(file_path)

    image_id = row.get("image_id", "")
    for ext in [".jpg", ".png"]:
        path = _images_original_dir() / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]

        path = _images_synth_dir() / "**" / f"{image_id}{ext}"
        matches = list(path.parent.glob(path.name))
        if matches:
            return matches[0]

    return None


def _resolve_manifest_image_path(row: dict) -> Path | None:
    direct = _resolve_data_path((row.get("file_path") or "").strip())
    if direct:
        return direct
    source = _resolve_data_path((row.get("source_file_path") or "").strip())
    if source:
        return source
    return None


def get_row_by_image_id(manifest, image_id: str):
    for row in manifest:
        if row.get("image_id") == image_id:
            return row
    return None


def load_excluded_ids() -> set:
    if not _excluded_from_synth_path().exists():
        return set()
    with open(_excluded_from_synth_path(), "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _save_excluded_ids(excluded_ids: set[str]) -> None:
    _excluded_from_synth_path().parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{_excluded_from_synth_path().name}.",
        suffix=".tmp",
        dir=str(_excluded_from_synth_path().parent),
    )
    os.close(fd)
    sorted_ids = sorted({value.strip() for value in excluded_ids if value.strip()})
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            for image_id in sorted_ids:
                handle.write(f"{image_id}\n")
        os.replace(tmp_name, _excluded_from_synth_path())
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_empty_ids() -> set[str]:
    if not _empty_candidates_path().exists():
        return set()
    with open(_empty_candidates_path(), "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


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
    return [
        {
            "name": path.name,
            "url": _file_url(path),
            "relative_path": _relative_posix(path),
        }
        for path in files
    ]


def resize_cover_center_crop(image_bytes: bytes, target_size: tuple[int, int]) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) / 2
        top = (new_h - target_h) / 2
        right = (new_w + target_w) / 2
        bottom = (new_h + target_h) / 2
        img_cropped = img_resized.crop((left, top, right, bottom))
        out = io.BytesIO()
        img_cropped.save(out, format="JPEG", quality=95)
        return out.getvalue()


# Synths prompts
STRICT_SIZE_CONSTRAINT = "Critical output constraint: return an image with exactly the same width and height as the input street scene. Do not crop, resize, pad, rotate, or change aspect ratio."
SCENE_LOCK_CONSTRAINT = "Scene integrity constraint: preserve the original scene geometry and composition exactly. Do not move or alter camera pose, horizon, vanishing points, road/curb alignment, lane markings, buildings, trees, poles, signs, parked objects, or any static landmark. Do not shift, warp, redraw, or restyle the background."
LOCAL_EDIT_ONLY_CONSTRAINT = "Local-edit-only constraint: modify only the minimal pixel region required to insert the new vehicle, plus immediate physically consistent contact shadow/occlusion near that vehicle. All other pixels should remain visually unchanged. If constraints cannot be satisfied, return the original scene unchanged."
SHADOW_CONSISTENCY_CONSTRAINT = "Shadow consistency constraint: shadow intensity and sharpness must match the scene's apparent lighting conditions. Bright sunny scenes should cast sharp, dark shadows. Overcast, cloudy, or diffuse lighting scenes should have soft, faint, or no visible shadows. Do not add hard, distinct shadows when the scene appears overcast or has soft ambient lighting."
TRAFFIC_DIRECTION_CONSTRAINT = "Orientation constraint: the inserted vehicle must be aligned with the correct lane direction for this road scene, matching traffic flow and lane markings. Do not place a wrong-way vehicle."
GROK_HAND_HOLDING_SCENE_CONSTRAINT = "Edit-in-place constraint: this is an edit task, not an image regeneration task. Keep the original scene, camera pose, geometry, objects, and background identity intact. Never replace or redesign the scenery, road layout, buildings, vegetation, vehicles, signs, sky, weather, or time of day."
GROK_HAND_HOLDING_PIXEL_CONSTRAINT = "Pixel stability constraint: outside the requested local edit region, pixels should remain as close as possible to the original image. Do not apply global style transfer, relighting, recoloring, denoising, sharpening, artistic restyling, or composition changes."
TOUCHUP_ONLY_CONSTRAINT = "Touch-up scope constraint: only refine the already inserted synthetic vehicle and its immediate contact shadow/occlusion. Do not create new vehicles or remove/replace existing scene elements."

RANDOM_VEHICLE_PROMPT = f"Edit this street image by adding exactly one normal passenger vehicle on the roadway. Keep camera perspective, scale, lighting, shadows, and color grading consistent. Do not alter buildings, signs, roads, or sky beyond what is required to insert the vehicle. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
ENFORCEMENT_PROMPT_NO_REF = f"Edit this street image by adding exactly one parking enforcement vehicle on the roadway. Use a realistic bylaw/enforcement style (light bar, municipal markings, no real logos). Keep perspective, lighting, shadows, and scene geometry consistent. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
ENFORCEMENT_PROMPT_WITH_REF = f"You are given two images: first is the target street scene, second is a reference enforcement vehicle. Insert one enforcement vehicle into the target scene, matching the reference style while preserving scene realism. Do not introduce unrelated scene changes. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
POLICE_OLD_PROMPT_NO_REF = f"Edit this street image by adding exactly one Ottawa Police cruiser with OLD livery on the roadway. The old livery has a white base with blue and yellow stripes on the sides. Include a roof lightbar. Keep perspective, lighting, shadows, and scene geometry consistent. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
POLICE_OLD_PROMPT_WITH_REF = f"You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with OLD livery. Insert one police cruiser matching the reference style into the target scene, preserving scene realism. Do not introduce unrelated scene changes. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
POLICE_NEW_PROMPT_NO_REF = f"Edit this street image by adding exactly one Ottawa Police cruiser with NEW livery on the roadway. The new livery has a dark base with reflective yellow/gold chevron stripes and modern markings. Include a roof lightbar. Keep perspective, lighting, shadows, and scene geometry consistent. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"
POLICE_NEW_PROMPT_WITH_REF = f"You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with NEW livery. Insert one police cruiser matching the reference style into the target scene, preserving scene realism. Do not introduce unrelated scene changes. {SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TRAFFIC_DIRECTION_CONSTRAINT} {SHADOW_CONSISTENCY_CONSTRAINT} {STRICT_SIZE_CONSTRAINT}"


def _apply_grok_hand_holding(prompt: str, provider: ImageProvider) -> str:
    if provider != "grok":
        return prompt
    return (
        f"{prompt} {GROK_HAND_HOLDING_SCENE_CONSTRAINT} {GROK_HAND_HOLDING_PIXEL_CONSTRAINT} "
        "If these constraints conflict with the request, keep the original scene unchanged."
    )


def _build_contents(
    edit_type: str,
    background_path: Path,
    reference_path: Path | None,
    provider: ImageProvider,
) -> tuple[str, list[Path]]:
    if edit_type == "random_vehicle":
        return _apply_grok_hand_holding(RANDOM_VEHICLE_PROMPT, provider), [
            background_path
        ]
    if edit_type == "enforcement_vehicle":
        return (
            (
                _apply_grok_hand_holding(ENFORCEMENT_PROMPT_WITH_REF, provider),
                [background_path, reference_path],
            )
            if reference_path
            else (
                _apply_grok_hand_holding(ENFORCEMENT_PROMPT_NO_REF, provider),
                [background_path],
            )
        )
    if edit_type == "police_old":
        return (
            (
                _apply_grok_hand_holding(POLICE_OLD_PROMPT_WITH_REF, provider),
                [background_path, reference_path],
            )
            if reference_path
            else (
                _apply_grok_hand_holding(POLICE_OLD_PROMPT_NO_REF, provider),
                [background_path],
            )
        )
    if edit_type == "police_new":
        return (
            (
                _apply_grok_hand_holding(POLICE_NEW_PROMPT_WITH_REF, provider),
                [background_path, reference_path],
            )
            if reference_path
            else (
                _apply_grok_hand_holding(POLICE_NEW_PROMPT_NO_REF, provider),
                [background_path],
            )
        )
    raise ValueError(f"Unsupported edit_type: {edit_type}")


def _normalize_provider(provider_value: str | None, fallback: str) -> ImageProvider:
    provider = (provider_value or "").strip().lower()
    fallback_value = (fallback or "grok").strip().lower()
    if fallback_value not in SUPPORTED_IMAGE_PROVIDERS:
        fallback_value = "grok"

    if provider == "grok":
        return "grok"
    if provider == "gemini":
        return "gemini"
    if provider == "flux":
        return "flux"
    return cast(ImageProvider, fallback_value)


def _validate_provider_or_400(provider_value: str | None) -> None:
    raw = (provider_value or "").strip().lower()
    if raw and raw not in SUPPORTED_IMAGE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider must be one of: {sorted(SUPPORTED_IMAGE_PROVIDERS)}",
        )


def _provider_models(provider: ImageProvider) -> list[str]:
    if provider == "grok":
        models = [PRIMARY_IMAGE_MODEL, FALLBACK_IMAGE_MODEL]
    elif provider == "gemini":
        models = [GEMINI_IMAGE_MODEL, GEMINI_IMAGE_FALLBACK]
    else:
        models = [FLUX_IMAGE_MODEL, FLUX_IMAGE_FALLBACK]
    deduped: list[str] = []
    for model in models:
        value = (model or "").strip()
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _get_provider_api_key(provider: ImageProvider) -> str:
    if provider == "grok":
        api_key = (os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY/GROK_API_KEY is missing in environment/.env"
            )
        return api_key
    if provider == "gemini":
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing in environment/.env")
        return api_key
    api_key = (os.getenv("FLUX_API_KEY") or os.getenv("BFL_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("FLUX_API_KEY/BFL_API_KEY is missing in environment/.env")
    return api_key


def _run_image_provider_subagent(
    provider: ImageProvider,
    api_key: str,
    model_name: str,
    prompt: str,
    image_paths: list[Path],
) -> tuple[bytes, dict[str, object]]:
    if not image_paths:
        raise RuntimeError("At least one image path is required")

    with image_paths[0].open("rb") as handle:
        base_image_bytes = handle.read()
    extra_image_bytes: list[bytes] = []
    for path in image_paths[1:8]:
        with path.open("rb") as handle:
            extra_image_bytes.append(handle.read())

    if provider == "grok":
        encoded_inputs = [encode_grok_image_to_base64(base_image_bytes)]
        for data in extra_image_bytes:
            encoded_inputs.append(encode_grok_image_to_base64(data))
        image_bytes = generate_grok_image(
            api_key=api_key,
            model=model_name,
            prompt=prompt,
            input_images=encoded_inputs,
        )
        return image_bytes, {}

    if provider == "flux":
        with Image.open(io.BytesIO(base_image_bytes)) as base_image:
            width, height = base_image.size

        encoded_inputs = [encode_flux_image_to_base64(base_image_bytes)]
        for data in extra_image_bytes:
            encoded_inputs.append(encode_flux_image_to_base64(data))
        image_bytes = generate_flux_image(
            api_key=api_key,
            model=model_name,
            prompt=prompt,
            input_images=encoded_inputs,
            width=width,
            height=height,
        )
        return image_bytes, {}

    gemini_result: GeminiImageResult = generate_gemini_image_with_usage(
        api_key=api_key,
        model=model_name,
        prompt=prompt,
        input_images=[base_image_bytes, *extra_image_bytes],
    )
    return gemini_result.image_bytes, {
        "prompt_tokens": gemini_result.prompt_tokens,
        "candidate_tokens": gemini_result.candidate_tokens,
        "total_tokens": gemini_result.total_tokens,
    }


def _default_cost_pricing() -> dict[str, object]:
    return {
        "grok": {
            "source": "https://docs.x.ai/developers/rate-limits",
            "notes": "xAI prices can vary by team in console billing; defaults are configurable estimate values",
            "output_per_image_usd": float(
                os.getenv("COST_GROK_OUTPUT_PER_IMAGE_USD", "0.020")
            ),
            "input_image_per_image_usd": float(
                os.getenv("COST_GROK_INPUT_IMAGE_PER_IMAGE_USD", "0.002")
            ),
        },
        "gemini": {
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "model_source": "https://ai.google.dev/gemini-api/docs/models",
            "notes": "Gemini 3.1 Flash Image Preview defaults: input $0.50/1M, text output $3/1M, image output $60/1M; approx image costs: 0.5K=$0.045, 1K=$0.067, 2K=$0.101, 4K=$0.151",
            "input_per_million_tokens_usd": float(
                os.getenv("COST_GEMINI_INPUT_PER_MTOK_USD", "0.50")
            ),
            "text_output_per_million_tokens_usd": float(
                os.getenv("COST_GEMINI_TEXT_OUTPUT_PER_MTOK_USD", "3.00")
            ),
            "image_output_per_million_tokens_usd": float(
                os.getenv("COST_GEMINI_IMAGE_OUTPUT_PER_MTOK_USD", "60.00")
            ),
            "output_per_million_tokens_usd": float(
                os.getenv("COST_GEMINI_OUTPUT_PER_MTOK_USD", "60.00")
            ),
            "fallback_output_image_tokens": int(
                os.getenv("COST_GEMINI_FALLBACK_OUTPUT_TOKENS", "1290")
            ),
            "fallback_input_image_tokens": int(
                os.getenv("COST_GEMINI_FALLBACK_INPUT_IMAGE_TOKENS", "1290")
            ),
        },
        "flux": {
            "source": "https://docs.bfl.ml/quick_start/pricing",
            "notes": "FLUX.2 pro image editing is MP-based",
            "output_first_megapixel_usd": float(
                os.getenv("COST_FLUX_OUTPUT_FIRST_MP_USD", "0.03")
            ),
            "output_additional_megapixel_usd": float(
                os.getenv("COST_FLUX_OUTPUT_ADDITIONAL_MP_USD", "0.015")
            ),
            "input_megapixel_usd": float(os.getenv("COST_FLUX_INPUT_MP_USD", "0.015")),
        },
    }


def _load_cost_tracker() -> dict[str, object]:
    tracker_path = _cost_tracker_file()
    if tracker_path.exists():
        try:
            with tracker_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    return {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "pricing": _default_cost_pricing(),
        "totals": {
            "overall_usd": 0.0,
            "providers": {
                "grok": {"usd": 0.0, "count": 0},
                "gemini": {"usd": 0.0, "count": 0},
                "flux": {"usd": 0.0, "count": 0},
            },
        },
        "events": [],
    }


def _save_cost_tracker(data: dict[str, object]) -> None:
    tracker_path = _cost_tracker_file()
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="provider_costs_", suffix=".json", dir=str(tracker_path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, tracker_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _image_megapixels(path: Path) -> float:
    with Image.open(path) as image:
        w, h = image.size
    return (float(w) * float(h)) / 1_000_000.0


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except Exception:
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(cast(Any, value))
    except Exception:
        return default


def _calculate_cost_event(
    *,
    operation: str,
    provider: ImageProvider,
    model_name: str,
    prompt: str,
    image_paths: list[Path],
    usage: dict[str, object] | None,
) -> dict[str, object]:
    pricing = _default_cost_pricing()
    usage_data = cast(dict[str, object], usage or {})
    output_usd = 0.0
    input_usd = 0.0
    cost_status = "ok"

    if provider == "grok":
        rate = cast(dict[str, object], pricing["grok"])
        output_usd = _as_float(rate.get("output_per_image_usd", 0.0))
        input_usd = _as_float(rate.get("input_image_per_image_usd", 0.0)) * float(
            max(1, len(image_paths))
        )
    elif provider == "gemini":
        rate = cast(dict[str, object], pricing["gemini"])
        prompt_tokens = _as_int(usage_data.get("prompt_tokens") or 0)
        candidate_tokens = _as_int(usage_data.get("candidate_tokens") or 0)
        if prompt_tokens <= 0 and candidate_tokens <= 0:
            cost_status = "estimated"
        if prompt_tokens <= 0:
            fallback_input_image_tokens = _as_int(
                rate.get("fallback_input_image_tokens", 1290), 1290
            )
            prompt_tokens = max(1, len(image_paths)) * fallback_input_image_tokens + (
                max(1, len(prompt)) // 4
            )
        if candidate_tokens <= 0:
            candidate_tokens = _as_int(
                rate.get("fallback_output_image_tokens", 1290), 1290
            )

        input_usd = (
            _as_float(rate.get("input_per_million_tokens_usd", 0.0))
            * float(prompt_tokens)
            / 1_000_000.0
        )
        output_usd = (
            _as_float(
                rate.get(
                    "image_output_per_million_tokens_usd",
                    rate.get("output_per_million_tokens_usd", 0.0),
                )
            )
            * float(candidate_tokens)
            / 1_000_000.0
        )
    else:
        rate = cast(dict[str, object], pricing["flux"])
        output_mp = _image_megapixels(image_paths[0]) if image_paths else 1.0
        output_usd = _as_float(rate.get("output_first_megapixel_usd", 0.0)) + (
            max(0.0, output_mp - 1.0)
            * _as_float(rate.get("output_additional_megapixel_usd", 0.0))
        )
        input_mp = 0.0
        for path in image_paths:
            input_mp += _image_megapixels(path)
        input_usd = input_mp * _as_float(rate.get("input_megapixel_usd", 0.0))

    total_usd = round(input_usd + output_usd, 6)
    return {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "provider": provider,
        "model": model_name,
        "input_images": len(image_paths),
        "prompt_chars": len(prompt),
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "total_usd": total_usd,
        "cost_status": cost_status,
        "usage": usage_data,
    }


def _append_cost_event(event: dict[str, object]) -> dict[str, object]:
    with _cost_tracker_lock:
        tracker = _load_cost_tracker()
        tracker["updated_at"] = datetime.now().isoformat()
        tracker["pricing"] = _default_cost_pricing()

        events = tracker.get("events")
        if not isinstance(events, list):
            events = []
        events.append(event)
        if len(events) > 5000:
            events = events[-5000:]
        tracker["events"] = events

        totals = tracker.get("totals")
        if not isinstance(totals, dict):
            totals = {"overall_usd": 0.0, "providers": {}}
        provider_totals = totals.get("providers")
        if not isinstance(provider_totals, dict):
            provider_totals = {}

        provider = str(event.get("provider") or "").strip().lower()
        if provider not in SUPPORTED_IMAGE_PROVIDERS:
            provider = "grok"
        event_total = _as_float(event.get("total_usd") or 0.0)

        provider_entry = provider_totals.get(provider)
        if not isinstance(provider_entry, dict):
            provider_entry = {"usd": 0.0, "count": 0}

        provider_entry["usd"] = round(
            float(provider_entry.get("usd") or 0.0) + event_total, 6
        )
        provider_entry["count"] = int(provider_entry.get("count") or 0) + 1
        provider_totals[provider] = provider_entry

        for known_provider in sorted(SUPPORTED_IMAGE_PROVIDERS):
            if known_provider not in provider_totals:
                provider_totals[known_provider] = {"usd": 0.0, "count": 0}

        totals["providers"] = provider_totals
        totals["overall_usd"] = round(
            float(totals.get("overall_usd") or 0.0) + event_total, 6
        )
        tracker["totals"] = totals

        _save_cost_tracker(tracker)
        return tracker


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


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int


class LabelUpdate(BaseModel):
    boxes: List[Box]


# --- HTML Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/labeler", response_class=HTMLResponse)
async def labeler(request: Request):
    return templates.TemplateResponse("labeler.html", {"request": request})


@app.get("/synth-gen", response_class=HTMLResponse)
async def synth_gen(request: Request):
    return templates.TemplateResponse("synth_gen.html", {"request": request})


@app.get("/synth-review-empty", response_class=HTMLResponse)
async def synth_review_empty(request: Request):
    return templates.TemplateResponse("synth_review_empty.html", {"request": request})


@app.get("/synth-cleanup", response_class=HTMLResponse)
async def synth_cleanup(request: Request):
    return templates.TemplateResponse("synth_cleanup.html", {"request": request})


@app.get("/synth-review-buckets", response_class=HTMLResponse)
async def synth_review_buckets(request: Request):
    return templates.TemplateResponse("synth_review_buckets.html", {"request": request})


@app.get("/training-viewer", response_class=HTMLResponse)
async def training_viewer(request: Request):
    return templates.TemplateResponse("training_viewer.html", {"request": request})


@app.get("/map-selector", response_class=HTMLResponse)
async def map_selector(request: Request):
    return templates.TemplateResponse("map_selector.html", {"request": request})


@app.get("/point-manager", response_class=HTMLResponse)
async def point_manager(request: Request):
    return templates.TemplateResponse("point_manager.html", {"request": request})


@app.get("/streetview-tester", response_class=HTMLResponse)
async def streetview_tester(request: Request):
    return templates.TemplateResponse("streetview_tester.html", {"request": request})


@app.get("/touchup-playground", response_class=HTMLResponse)
async def touchup_playground(request: Request):
    return templates.TemplateResponse("touchup_playground.html", {"request": request})


@app.get("/files/{relative_path:path}")
async def serve_file(relative_path: str):
    target = _normalize_under_root(PROJECT_ROOT / relative_path)
    if not target or not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(target),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


# --- Labeler API ---


@app.get("/api/labeler/images")
async def get_labeler_images(
    queue: str = "needs_review", status: str = "todo", include_excluded: bool = False
):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()
    images = []
    for row in manifest:
        if row.get("status") != "ok":
            continue
        image_id = row.get("image_id")
        if not include_excluded and image_id in excluded_ids:
            continue
        needs_review_value = row.get("needs_review", "")
        if queue == "needs_review" and needs_review_value != "1":
            continue
        if queue == "autolabel_ok" and needs_review_value != "0":
            continue
        if status and row.get("review_status") not in ["", status]:
            continue
        if row.get("review_status") == "done":
            continue
        image_path = get_image_path(row)
        if image_path and image_path.exists():
            images.append(
                {
                    "image_id": row.get("image_id"),
                    "image_path": f"/api/labeler/image_file/{row.get('image_id')}",
                    "is_synthetic": row.get("is_synthetic") == "1",
                    "is_excluded": row.get("image_id") in excluded_ids,
                    "edit_type": row.get("edit_type", ""),
                    "expected_class": row.get("expected_inserted_class", ""),
                    "review_status": row.get("review_status", "todo"),
                }
            )
    return images


@app.get("/api/labeler/image/{image_id}")
async def get_labeler_image(image_id: str):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    from PIL import Image

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    label_path_final = _labels_final_dir() / f"{image_id}.txt"
    label_path_autogen = _labels_autogen_dir() / f"{image_id}.txt"
    label_path = None
    if label_path_final.exists() and label_path_final.read_text().strip():
        label_path = label_path_final
    elif label_path_autogen.exists():
        label_path = label_path_autogen

    boxes = []
    if label_path:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    x1 = (x_center - width / 2) * img_w
                    y1 = (y_center - height / 2) * img_h
                    x2 = (x_center + width / 2) * img_w
                    y2 = (y_center + height / 2) * img_h
                    boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})

    return {
        "image_id": image_id,
        "image_path": f"/api/labeler/image_file/{image_id}",
        "boxes": boxes,
        "is_synthetic": row.get("is_synthetic") == "1",
        "is_excluded": image_id in excluded_ids,
        "edit_type": row.get("edit_type", ""),
        "expected_class": row.get("expected_inserted_class", ""),
    }


@app.get("/api/labeler/image_file/{image_id}")
async def get_labeler_image_file(image_id: str):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(image_path))


@app.post("/api/labeler/labels/{image_id}")
async def save_labeler_labels(image_id: str, data: LabelUpdate):
    _labels_final_dir().mkdir(parents=True, exist_ok=True)
    label_path = _labels_final_dir() / f"{image_id}.txt"
    skipped = 0
    with open(label_path, "w") as f:
        for box in data.boxes:
            x1 = max(0.0, min(1.0, box.x1))
            y1 = max(0.0, min(1.0, box.y1))
            x2 = max(0.0, min(1.0, box.x2))
            y2 = max(0.0, min(1.0, box.y2))
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            f.write(
                f"{box.cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )

    manifest = load_manifest()
    for row in manifest:
        if row.get("image_id") == image_id:
            row["review_status"] = "done"
            row["needs_review"] = "0"
            break
    save_manifest(manifest)
    saved = len(data.boxes) - skipped
    return {"status": "saved", "boxes": saved, "skipped": skipped}


@app.post("/api/labeler/review/{image_id}")
async def update_labeler_review_status(image_id: str, status: str):
    manifest = load_manifest()
    for row in manifest:
        if row.get("image_id") == image_id:
            row["review_status"] = status
            row["needs_review"] = "0" if status == "done" else "1"
            break
    save_manifest(manifest)
    return {"status": status}


# --- Synth Gen API ---


@app.get("/api/synth/models")
def get_synth_models():
    return {
        "providers": sorted(SUPPORTED_IMAGE_PROVIDERS),
        "default_synth_provider": DEFAULT_SYNTH_PROVIDER,
        "default_touchup_provider": DEFAULT_TOUCHUP_PROVIDER,
        "models": {
            "grok": {
                "primary": PRIMARY_IMAGE_MODEL,
                "fallback": FALLBACK_IMAGE_MODEL,
            },
            "gemini": {
                "primary": GEMINI_IMAGE_MODEL,
                "fallback": GEMINI_IMAGE_FALLBACK,
            },
            "flux": {
                "primary": FLUX_IMAGE_MODEL,
                "fallback": FLUX_IMAGE_FALLBACK,
            },
        },
        "goal": "Create and inspect synthetic daylight edits before bulk generation.",
    }


@app.get("/api/synth/costs")
def get_synth_costs():
    tracker = _load_cost_tracker()
    tracker["pricing"] = _default_cost_pricing()
    totals = tracker.get("totals")
    if not isinstance(totals, dict):
        totals = {"overall_usd": 0.0, "providers": {}}
    providers = totals.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    for provider in sorted(SUPPORTED_IMAGE_PROVIDERS):
        if provider not in providers or not isinstance(providers.get(provider), dict):
            providers[provider] = {"usd": 0.0, "count": 0}
    totals["providers"] = providers
    totals["overall_usd"] = _as_float(totals.get("overall_usd") or 0.0)
    tracker["totals"] = totals
    return tracker


@app.get("/api/synth/images")
def get_synth_images(source: str = "empty", limit: int = 120):
    limit = max(1, min(500, limit))
    empty_ids = load_empty_ids()
    excluded_ids = load_excluded_ids()
    rows = load_manifest()
    records = []
    seen = set()

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

        image_path = _resolve_manifest_image_path(row)
        if not image_path:
            continue

        records.append(_build_image_record(row, image_path))
        seen.add(image_id)
        if len(records) >= limit:
            break

    return records


@app.get("/api/synth/references")
def get_synth_references():
    return {
        "enforcement": _list_reference_images(ENFORCEMENT_DATASET_DIR),
        "police_old": _list_reference_images(POLICE_OLD_DATASET_DIR),
        "police_new": _list_reference_images(POLICE_NEW_DATASET_DIR),
    }


@app.get("/api/synth/empty-review")
def get_synth_empty_review(limit: int = 1000):
    limit = max(1, min(2000, limit))
    empty_ids = load_empty_ids()
    excluded_ids = load_excluded_ids()
    records = []

    total_empty = 0
    total_excluded = 0
    seen_ids = set()

    for row in load_manifest():
        image_id = (row.get("image_id") or "").strip()
        if not image_id or image_id not in empty_ids:
            continue
        if image_id in seen_ids:
            continue
        seen_ids.add(image_id)
        total_empty += 1
        if image_id in excluded_ids:
            total_excluded += 1

    seen_ids = set()
    for row in load_manifest():
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

    return {
        "items": records,
        "counts": {
            "total_empty": total_empty,
            "excluded": total_excluded,
            "available": total_empty - total_excluded,
        },
    }


class ExcludeRequest(BaseModel):
    image_ids: Optional[list[str]] = None
    image_id: Optional[str] = None


@app.post("/api/synth/empty-review/exclude")
def synth_empty_review_exclude(req: ExcludeRequest):
    image_ids = req.image_ids or ([req.image_id] if req.image_id else [])
    image_ids = [str(iid).strip() for iid in image_ids if str(iid).strip()]
    if not image_ids:
        raise HTTPException(status_code=400, detail="image_id or image_ids is required")

    excluded_ids = load_excluded_ids()
    empty_ids = load_empty_ids()
    added_count = 0
    for image_id in image_ids:
        if image_id in empty_ids and image_id not in excluded_ids:
            excluded_ids.add(image_id)
            added_count += 1

    _save_excluded_ids(excluded_ids)
    return {"success": True, "excluded_count": added_count, "image_ids": image_ids}


@app.post("/api/synth/empty-review/unexclude")
def synth_empty_review_unexclude(req: ExcludeRequest):
    image_ids = req.image_ids or ([req.image_id] if req.image_id else [])
    image_ids = [str(iid).strip() for iid in image_ids if str(iid).strip()]
    if not image_ids:
        raise HTTPException(status_code=400, detail="image_id or image_ids is required")

    excluded_ids = load_excluded_ids()
    removed_count = 0
    for image_id in image_ids:
        if image_id in excluded_ids:
            excluded_ids.discard(image_id)
            removed_count += 1

    _save_excluded_ids(excluded_ids)
    return {"success": True, "unexcluded_count": removed_count, "image_ids": image_ids}


@app.get("/api/synth/empty-review/excluded")
def synth_empty_review_excluded():
    return list(load_excluded_ids())


class GenerateRequest(BaseModel):
    image_id: str
    edit_type: str
    provider: Optional[str] = None
    reference_path: Optional[str] = None
    reference_image: Optional[str] = None
    auto_crop_to_original: Optional[bool] = False


@app.post("/api/synth/generate")
def synth_generate(req: GenerateRequest):
    image_id = req.image_id.strip()
    edit_type = req.edit_type.strip()
    reference_raw = req.reference_path or req.reference_image
    auto_crop = req.auto_crop_to_original

    if not image_id:
        raise HTTPException(status_code=400, detail="image_id is required")
    if edit_type not in EDIT_TYPES:
        raise HTTPException(
            status_code=400, detail=f"edit_type must be one of: {sorted(EDIT_TYPES)}"
        )

    row = None
    for candidate in load_manifest():
        if (candidate.get("image_id") or "").strip() == image_id:
            row = candidate
            break
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"image_id not found in manifest: {image_id}"
        )

    background_path = _resolve_manifest_image_path(row)
    if not background_path:
        raise HTTPException(
            status_code=404, detail="Background image file not found on disk"
        )

    reference_path = _resolve_reference_path(
        str(reference_raw) if reference_raw else None
    )

    _validate_provider_or_400(req.provider)
    provider = _normalize_provider(req.provider, DEFAULT_SYNTH_PROVIDER)

    try:
        api_key = _get_provider_api_key(provider)
        prompt, image_paths = _build_contents(
            edit_type, background_path, reference_path, provider
        )

        if provider == "flux" and len(image_paths) > 1:
            raise HTTPException(
                status_code=400,
                detail="Flux provider currently supports single-image edit mode in this UI. Use Grok or Gemini for reference-guided edits.",
            )

        models_to_try = _provider_models(provider)

        image_bytes: bytes | None = None
        usage_meta: dict[str, object] = {}
        successful_cost_event: dict[str, object] | None = None
        used_model = ""
        errors = []

        for model_name in models_to_try:
            attempt_usage: dict[str, object] | None = None
            attempt_status = "error"
            attempt_error = ""
            try:
                image_bytes, usage_meta = _run_image_provider_subagent(
                    provider, api_key, model_name, prompt, image_paths
                )
                attempt_usage = usage_meta
                attempt_status = "success"
                used_model = model_name
                break
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
                attempt_error = str(exc)
            finally:
                attempt_cost_event = _calculate_cost_event(
                    operation="generate",
                    provider=provider,
                    model_name=model_name,
                    prompt=prompt,
                    image_paths=image_paths,
                    usage=attempt_usage,
                )
                attempt_cost_event["attempt_status"] = attempt_status
                if attempt_error:
                    attempt_cost_event["error"] = attempt_error
                _append_cost_event(attempt_cost_event)
                if attempt_status == "success":
                    successful_cost_event = attempt_cost_event

        if image_bytes is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Generation failed", "details": errors},
            )

        with Image.open(background_path) as background_image:
            expected_size = background_image.size
        with Image.open(io.BytesIO(image_bytes)) as generated_image:
            generated_size = generated_image.size

        size_mismatch = generated_size != expected_size
        auto_crop_applied = False
        warning = None
        output_bytes = image_bytes

        if size_mismatch:
            warning = f"Generated size {generated_size} differs from original {expected_size}."
            if auto_crop:
                try:
                    output_bytes = resize_cover_center_crop(image_bytes, expected_size)
                    auto_crop_applied = True
                    warning = f"Generated size {generated_size} differed from original {expected_size}; resize-to-cover + center-crop was applied."
                except Exception as exc:
                    warning = f"Generated size {generated_size} differs from original {expected_size}; auto-crop failed ({exc}). Showing raw generated image."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _images_synth_dir() / edit_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_suffix = (
            "_autocrop"
            if (size_mismatch and auto_crop_applied)
            else "_size_mismatch"
            if size_mismatch
            else ""
        )
        output_path = (
            output_dir / f"{image_id}_{edit_type}_{timestamp}{output_suffix}.jpg"
        )

        with open(output_path, "wb") as handle:
            handle.write(output_bytes)

        return {
            "success": True,
            "provider_used": provider,
            "model_used": used_model,
            "result_url": _file_url(output_path),
            "result_relative_path": _relative_posix(output_path),
            "expected_size": [expected_size[0], expected_size[1]],
            "generated_size": [generated_size[0], generated_size[1]],
            "size_mismatch": size_mismatch,
            "auto_crop_requested": auto_crop,
            "auto_crop_applied": auto_crop_applied,
            "warning": warning,
            "cost_event": successful_cost_event,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Synth Cleanup API ---


def _collect_synth_images(class_filter: str | None) -> list[Path]:
    if not _images_synth_dir().exists():
        return []
    images: list[Path] = []
    for class_dir in sorted(_images_synth_dir().iterdir()):
        if not class_dir.is_dir():
            continue
        if class_filter and class_dir.name != class_filter:
            continue
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                images.append(p)
    return images


def _move_to_trash(src: Path) -> Path:
    class_name = src.parent.name
    dest_dir = _trash_root() / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"{src.stem}_{ts}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


def _ensure_delete_log_header() -> None:
    trash_dir = _trash_root()
    trash_dir.mkdir(parents=True, exist_ok=True)
    log_file = _delete_log()
    if log_file.exists():
        return
    with open(log_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["deleted_at", "class_name", "source_rel", "trash_rel"]
        )
        writer.writeheader()


@app.get("/api/cleanup/classes")
def cleanup_classes():
    synth_dir = _images_synth_dir()
    if not synth_dir.exists():
        return []
    return [d.name for d in sorted(synth_dir.iterdir()) if d.is_dir()]

    @app.get("/api/cleanup/images")
    def cleanup_images(
        page: int = 1, page_size: int = 500, class_filter: Optional[str] = None
    ):
        page = max(1, page)
        page_size = max(1, min(500, page_size))

        all_images = _collect_synth_images(class_filter)
        total = len(all_images)
        total_pages = max(1, math.ceil(total / page_size)) if total else 1
        page = min(page, total_pages)

        start = (page - 1) * page_size
        end = start + page_size
        batch = all_images[start:end]

        items = []
        for p in batch:
            rel = _relative_posix(p, PROJECT_ROOT)
            items.append(
                {
                    "id": rel,
                    "url": f"/files/{quote(rel)}",
                    "class_name": p.parent.name,
                    "name": p.name,
                }
            )

        return {
            "items": items,
            "page": page,
            "total_pages": total_pages,
            "total_items": total,
        }


@app.get("/api/synth/custom-presets")
def list_custom_presets():
    """List all custom presets."""
    presets = _load_presets()
    preset_list = [{"name": name, "prompt": prompt} for name, prompt in presets.items()]
    return {"presets": preset_list}


@app.post("/api/synth/custom-presets")
def save_custom_preset(req: dict):
    """Save a custom preset."""
    name = req.get("name", "").strip()
    prompt = req.get("prompt", "").strip()

    if not name:
        raise HTTPException(status_code=400, detail="Preset name is required")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt text is required")

    presets = _load_presets()
    presets[name] = prompt
    _save_presets(presets)

    return {"status": "ok", "message": f"Preset '{name}' saved"}


@app.get("/api/synth/custom-presets/{preset_name}")
def get_custom_preset(preset_name: str):
    """Get a specific custom preset."""
    presets = _load_presets()
    if preset_name not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    return {"name": preset_name, "prompt": presets[preset_name]}


@app.delete("/api/synth/custom-presets/{preset_name}")
def delete_custom_preset(preset_name: str):
    """Delete a custom preset."""
    presets = _load_presets()
    if preset_name not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

    del presets[preset_name]
    _save_presets(presets)

    return {"status": "ok", "message": f"Preset '{preset_name}' deleted"}


@app.get("/api/touchup-prompts")
def list_touchup_prompts():
    """List all touchup prompts."""
    data = _load_touchup_prompts()
    return {"prompts": data.get("prompts", [])}


@app.post("/api/touchup-prompts")
def add_touchup_prompt(req: dict):
    """Add a new touchup prompt."""
    name = req.get("name", "").strip()
    prompt_text = req.get("prompt_text", "").strip()
    description = req.get("description", "").strip()

    if not name or not prompt_text:
        raise HTTPException(status_code=400, detail="name and prompt_text are required")

    data = _load_touchup_prompts()
    prompts = data.get("prompts", [])

    new_id = name.lower().replace(" ", "_")
    for p in prompts:
        if p.get("id") == new_id:
            raise HTTPException(
                status_code=400, detail=f"Prompt with id '{new_id}' already exists"
            )

    prompts.append(
        {
            "id": new_id,
            "name": name,
            "description": description,
            "prompt_text": prompt_text,
            "is_default": False,
        }
    )

    _save_touchup_prompts({"prompts": prompts})
    return {"status": "ok", "id": new_id, "message": f"Prompt '{name}' added"}


@app.put("/api/touchup-prompts/{prompt_id}")
def update_touchup_prompt(prompt_id: str, req: dict):
    """Update an existing touchup prompt."""
    name = req.get("name", "").strip()
    prompt_text = req.get("prompt_text", "").strip()
    description = req.get("description", "").strip()

    if not name or not prompt_text:
        raise HTTPException(status_code=400, detail="name and prompt_text are required")

    data = _load_touchup_prompts()
    prompts = data.get("prompts", [])

    found = False
    for p in prompts:
        if p.get("id") == prompt_id:
            p["name"] = name
            p["description"] = description
            p["prompt_text"] = prompt_text
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    _save_touchup_prompts({"prompts": prompts})
    return {"status": "ok", "message": f"Prompt '{name}' updated"}


@app.delete("/api/touchup-prompts/{prompt_id}")
def delete_touchup_prompt(prompt_id: str):
    """Delete a touchup prompt."""
    data = _load_touchup_prompts()
    prompts = data.get("prompts", [])

    found = False
    for i, p in enumerate(prompts):
        if p.get("id") == prompt_id:
            prompts.pop(i)
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail=f"Prompt '{prompt_id}' not found")

    _save_touchup_prompts({"prompts": prompts})
    return {"status": "ok", "message": f"Prompt '{prompt_id}' deleted"}


@app.post("/api/synth/apply-touchup")
def apply_touchup(req: dict):
    original_path = req.get("original_image_path", "")
    touchup_path = req.get("touchup_image_path", "")

    if not original_path or not touchup_path:
        raise HTTPException(
            status_code=400,
            detail="Both original_image_path and touchup_image_path are required",
        )

    original_target = _normalize_under_root(PROJECT_ROOT / original_path)
    touchup_target = _normalize_under_root(PROJECT_ROOT / touchup_path)

    if not original_target or not touchup_target:
        raise HTTPException(status_code=400, detail="Invalid image path")

    if not original_target.exists():
        raise HTTPException(
            status_code=404, detail=f"Original image not found: {original_path}"
        )

    if not touchup_target.exists():
        raise HTTPException(
            status_code=404, detail=f"Touchup image not found: {touchup_path}"
        )

    with Image.open(original_target) as original_img:
        original_size = original_img.size
    with Image.open(touchup_target) as touchup_img:
        touchup_size = touchup_img.size

    if touchup_size != original_size:
        raise HTTPException(
            status_code=400,
            detail=f"Touchup size mismatch: touchup={touchup_size} original={original_size}",
        )

    shutil.copy2(touchup_target, original_target)

    try:
        touchup_target.unlink()
        temp_dir = touchup_target.parent
        if (
            temp_dir.exists()
            and temp_dir.name == "temp_touchups"
            and not any(temp_dir.iterdir())
        ):
            temp_dir.rmdir()
    except Exception:
        pass

    return {"status": "ok", "message": "Touchup applied successfully"}


class DeleteRequest(BaseModel):
    relative_paths: list[str]


@app.post("/api/cleanup/delete")
def cleanup_delete(req: DeleteRequest):
    _ensure_delete_log_header()
    manifest = load_manifest()

    deleted_count = 0
    with open(_delete_log(), "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["deleted_at", "class_name", "source_rel", "trash_rel"]
        )

        for rel_path in req.relative_paths:
            source = _resolve_data_path(rel_path)
            if not source or not source.exists():
                continue

            class_name = source.parent.name
            target = _trash_root() / class_name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(source), str(target))
            deleted_count += 1

            for row in manifest:
                if row.get("image_id") == source.stem:
                    row["status"] = "deleted"
                    row["review_status"] = "rejected"
                    break

            writer.writerow(
                {
                    "deleted_at": datetime.now().isoformat(),
                    "class_name": class_name,
                    "source_rel": rel_path,
                    "trash_rel": _relative_posix(target),
                }
            )

    save_manifest(manifest)
    return {"deleted": deleted_count}


@app.get("/api/synth/review-buckets/items")
def get_review_bucket_items(bucket: str = "", status: str = "todo", limit: int = 1000):
    """Get synthetic images filtered by review_bucket and review_status."""
    limit = max(1, min(2000, limit))
    manifest = load_manifest()
    items = []
    seen_ids = set()

    for row in manifest:
        image_id = (row.get("image_id") or "").strip()
        if not image_id or image_id in seen_ids:
            continue
        seen_ids.add(image_id)

        is_synthetic = (row.get("is_synthetic") or "0").strip() == "1"
        row_bucket = (row.get("review_bucket") or "").strip()
        row_status = (row.get("review_status") or "").strip()

        if not is_synthetic:
            continue
        if bucket and row_bucket != bucket:
            continue
        if status and row_status != status:
            continue

        file_path = row.get("file_path", "")
        if not file_path:
            continue

        resolved_path = _resolve_data_path(file_path)
        if not resolved_path:
            continue

        # Build URL for the image
        rel_path = _relative_posix(resolved_path, PROJECT_ROOT)
        url = f"/files/{rel_path}"

        items.append(
            {
                "image_id": image_id,
                "review_bucket": row_bucket,
                "review_status": row_status,
                "parent_image_id": row.get("parent_image_id", ""),
                "edit_type": row.get("edit_type", ""),
                "relative_path": rel_path,
                "url": url,
            }
        )

        if len(items) >= limit:
            break

    return {"items": items}


class UpdateReviewStatusRequest(BaseModel):
    image_ids: list[str]
    review_status: str


@app.post("/api/synth/review-buckets/status")
def update_review_bucket_status(req: UpdateReviewStatusRequest):
    """Bulk update review_status for synthetic images."""
    image_ids = req.image_ids
    new_status = req.review_status

    if not image_ids or not new_status:
        raise HTTPException(
            status_code=400, detail="image_ids and review_status are required"
        )

    manifest = load_manifest()
    updated = 0

    for row in manifest:
        img_id = (row.get("image_id") or "").strip()
        is_synthetic = (row.get("is_synthetic") or "0").strip() == "1"
        if img_id in image_ids and is_synthetic:
            row["review_status"] = new_status
            updated += 1

    if updated > 0:
        save_manifest(manifest)

    return {"updated": updated, "status": new_status}


class TouchupRequest(BaseModel):
    image_relative_path: str
    provider: Optional[str] = None
    prompt_mode: str = "preset"
    custom_prompt: Optional[str] = None


def _compose_touchup_prompt(change_request: str, provider: ImageProvider) -> str:
    if provider == "grok":
        # Grok needs explicit hand-holding: it tends to regenerate the whole scene or
        # apply global style changes unless tightly constrained at the wrapper level.
        return (
            "You are performing a precise image touch-up on an existing synthetic edit. "
            "Do not regenerate or redesign the scene. "
            f"Requested local change: {change_request} "
            "Keep the inserted vehicle identity stable: preserve class/livery/markings and keep the same vehicle count. "
            f"{SCENE_LOCK_CONSTRAINT} {LOCAL_EDIT_ONLY_CONSTRAINT} {TOUCHUP_ONLY_CONSTRAINT} "
            f"{STRICT_SIZE_CONSTRAINT} "
            f"{GROK_HAND_HOLDING_SCENE_CONSTRAINT} {GROK_HAND_HOLDING_PIXEL_CONSTRAINT} "
            "If the requested change cannot be satisfied while preserving the original scene, return the original image unchanged."
        )

    # Gemini: expressive wrapper that sets physical context without overriding the
    # change_request's own shadow/lighting semantics.
    return (
        "You are a photorealistic image compositor performing a targeted touch-up on a synthetic street scene. "
        "The scene already contains a synthetically inserted enforcement or police vehicle. "
        "Your task is described below — execute it with physical accuracy and photographic realism. "
        f"Task: {change_request} "
        "Preserve the following invariants: "
        "(1) Vehicle identity — the vehicle's class, livery, markings, and position must remain unchanged. "
        "(2) Scene integrity — do not alter camera pose, geometry, vanishing points, background elements, or any pixel outside the scope defined by the task. "
        "(3) Output dimensions — return an image with exactly the same width and height as the input. "
        "Apply your understanding of real-world optics and scene geometry to produce a result that is physically plausible and visually seamless."
    )


@app.post("/api/synth/touchup")
def synth_touchup(req: TouchupRequest):
    """Touch-up a synthetic image with a new prompt."""
    from utils.preprocessing import resize_cover_center_crop
    import logging
    import uuid
    from pathlib import Path

    image_relative_path = req.image_relative_path
    prompt_mode = req.prompt_mode
    custom_prompt = req.custom_prompt

    _validate_provider_or_400(req.provider)
    provider = _normalize_provider(req.provider, DEFAULT_TOUCHUP_PROVIDER)

    # Resolve the image path
    target = PROJECT_ROOT / image_relative_path
    if not target.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    # Load the image
    with target.open("rb") as f:
        image_bytes = f.read()
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        expected_size = source_image.size

    # Build the prompt based on mode
    if prompt_mode == "custom":
        if not custom_prompt:
            raise HTTPException(
                status_code=400, detail="custom_prompt required for custom mode"
            )
        user_prompt = custom_prompt
    elif prompt_mode == "fix_direction":
        # Make enforcement/police vehicle point the correct direction based on the road
        user_prompt = (
            "Adjust the orientation of the enforcement or police vehicle in this image so it points in the correct direction "
            "based on the road orientation and traffic flow. The vehicle should be positioned naturally as if it was "
            "originally captured driving along the road. Maintain the vehicle's position but adjust its angle to align "
            "with the road direction. Keep all other aspects of the image unchanged."
        )
    elif prompt_mode == "fix_lighting":
        user_prompt = (
            "This image contains a synthetically inserted enforcement vehicle (parking enforcement, bylaw, or police cruiser). "
            "Your task: correct the lighting, exposure, and color-temperature on that enforcement vehicle so it matches the lighting conditions at its exact position in the scene. "
            "Critical: the vehicle may be sitting inside a cast shadow from a tree, building, or other object. "
            "If the road surface and surroundings immediately around the vehicle are dark or shaded, the vehicle must also appear dark and shaded to that same degree — do not brighten it to match the sunlit parts of the scene. "
            "Infer local illumination from the surroundings — road surface tone, nearby ground, immediate area — not from the brightest or most representative parts of the image. "
            "The background is ground truth and must not change. Only the vehicle's surface shading, exposure, and color temperature should be adjusted to be consistent with the local lighting where it sits."
        )
    else:
        # Preset: improve blending (default)
        user_prompt = (
            "Improve the blending of the inserted vehicle in this image. "
            "Fix any visible edges, seams, color mismatches, or unnatural shadows. "
            "Make the vehicle look seamlessly integrated into the scene as if it was originally captured with it. "
            "Match the lighting, color temperature, and perspective exactly. "
            "Do not change the vehicle's position or orientation."
        )

    final_prompt = _compose_touchup_prompt(user_prompt, provider)

    prompt_preview = f"Touchup prompt for {image_relative_path} (mode: {prompt_mode}, provider: {provider}): {final_prompt}"
    print(prompt_preview)
    logging.info(prompt_preview)

    models_to_try = _provider_models(provider)

    generated_bytes: bytes | None = None
    usage_meta: dict[str, object] = {}
    successful_cost_event: dict[str, object] | None = None
    errors: list[str] = []
    used_model = ""
    provider_error_types: tuple[type[Exception], ...]
    if provider == "grok":
        provider_error_types = (GrokImageAPIError,)
    elif provider == "flux":
        provider_error_types = (FluxAPIError,)
    else:
        provider_error_types = (GeminiImageAPIError,)

    try:
        api_key = _get_provider_api_key(provider)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    for model_name in models_to_try:
        attempt_usage: dict[str, object] | None = None
        attempt_status = "error"
        attempt_error = ""
        try:
            generated_bytes, usage_meta = _run_image_provider_subagent(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                prompt=final_prompt,
                image_paths=[target],
            )
            attempt_usage = usage_meta
            attempt_status = "success"
            used_model = model_name
            break
        except provider_error_types as exc:
            errors.append(f"{model_name}: {exc}")
            attempt_error = str(exc)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            attempt_error = str(exc)
        finally:
            attempt_cost_event = _calculate_cost_event(
                operation="touchup",
                provider=provider,
                model_name=model_name,
                prompt=final_prompt,
                image_paths=[target],
                usage=attempt_usage,
            )
            attempt_cost_event["attempt_status"] = attempt_status
            if attempt_error:
                attempt_cost_event["error"] = attempt_error
            _append_cost_event(attempt_cost_event)
            if attempt_status == "success":
                successful_cost_event = attempt_cost_event

    if generated_bytes is None:
        raise HTTPException(
            status_code=500,
            detail=f"{provider} touchup failed: {' | '.join(errors)}",
        )

    temp_dir = PROJECT_ROOT / "temp_touchups"
    temp_dir.mkdir(exist_ok=True)

    touchup_filename = f"{uuid.uuid4().hex}_{Path(image_relative_path).name}"
    temp_target = temp_dir / touchup_filename

    output_bytes = generated_bytes
    with Image.open(io.BytesIO(generated_bytes)) as generated_image:
        generated_size = generated_image.size
    if generated_size != expected_size:
        output_bytes = resize_cover_center_crop(generated_bytes, expected_size)

    with temp_target.open("wb") as f:
        f.write(output_bytes)

    logging.info(
        "Touchup image saved to: %s (provider=%s model=%s)",
        temp_target,
        provider,
        used_model,
    )

    temp_relative_path = f"temp_touchups/{touchup_filename}"
    return {
        "status": "ok",
        "image": temp_relative_path,
        "is_temp": True,
        "provider_used": provider,
        "model_used": used_model,
        "cost_event": successful_cost_event,
        "prompt_used": final_prompt,
    }
