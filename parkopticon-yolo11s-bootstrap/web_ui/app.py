import argparse
from collections import Counter
import csv
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
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

_labeler_include_rejected = "--labeler-include-rejected" in sys.argv

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


def _lookalike_metadata_dir() -> Path:
    return _run_path("data", "lookalike_metadata")


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


def _get_touchup_prompt_by_id(prompt_id: str) -> Optional[dict[str, Any]]:
    normalized_id = prompt_id.strip()
    if not normalized_id:
        return None

    data = _load_touchup_prompts()
    prompts = data.get("prompts", [])
    for prompt in prompts:
        if str(prompt.get("id", "")).strip() == normalized_id:
            return cast(dict[str, Any], prompt)
    return None


def _get_touchup_prompt_text(prompt_id: str) -> Optional[str]:
    prompt = _get_touchup_prompt_by_id(prompt_id)
    if not prompt:
        return None

    prompt_text = str(prompt.get("prompt_text", "")).strip()
    return prompt_text or None


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

LOOKALIKE_CLASS_ID = 4
LOOKALIKE_SIMILARITY_OPTIONS = {
    "",
    "enforcement_vehicle",
    "police_old",
    "police_new",
    "unsure",
}

_cost_tracker_lock = threading.Lock()

BATCH_MANAGER_SCRIPT = PROJECT_ROOT / "scripts" / "07b_batch_run_manager.py"
MEDIA_INFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "12_run_media_inference.py"

_batch_jobs_lock = threading.Lock()
_batch_jobs: dict[str, dict[str, Any]] = {}

_inference_jobs_lock = threading.Lock()
_inference_jobs: dict[str, dict[str, Any]] = {}
_inference_processes: dict[str, subprocess.Popen[Any]] = {}

INFERENCE_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
INFERENCE_ACTIVE_STATUSES = {"queued", "running", "cancelling"}

INFERENCE_MEDIA_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".webm",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def _batch_plans_dir() -> Path:
    return _run_path("manifests", "batch_run_plans")


def _batch_logs_dir() -> Path:
    return _run_path("logs", "batch_runs")


def _inference_logs_dir() -> Path:
    return _run_path("logs", "inference_runs")


def _inference_outputs_dir() -> Path:
    return _run_path("runs", "inference")


def _inference_previews_dir() -> Path:
    return _run_path("logs", "inference_previews")


def _resolve_existing_project_file(
    raw_value: str, *, allow_exts: set[str] | None = None
) -> Path:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Path cannot be empty")

    candidate = Path(cleaned)
    attempts: list[Path] = []
    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.append(_get_run_dir() / candidate)
        attempts.append(PROJECT_ROOT / candidate)

    resolved: Path | None = None
    for attempt in attempts:
        normalized = _normalize_under_root(attempt)
        if normalized and normalized.exists() and normalized.is_file():
            resolved = normalized
            break

    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=f"File not found under project root: {raw_value}",
        )

    if allow_exts and resolved.suffix.lower() not in allow_exts:
        allowed = ", ".join(sorted(allow_exts))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{resolved.suffix}'. Expected one of: {allowed}",
        )

    return resolved


def _collect_inference_weight_files(limit: int = 300) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    roots = [(_get_run_dir() / "runs" / "detect"), (PROJECT_ROOT / "runs")]

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*.pt"):
            normalized = _normalize_under_root(path)
            if not normalized or not normalized.is_file():
                continue
            key = str(normalized)
            if key in seen:
                continue
            seen.add(key)
            files.append(normalized)
            if len(files) >= limit:
                break
        if len(files) >= limit:
            break

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _collect_inference_media_files(limit: int = 120) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    roots = [
        PROJECT_ROOT / "test-videos",
        PROJECT_ROOT,
        _get_run_dir() / "test-videos",
        _get_run_dir(),
    ]
    for root in roots:
        normalized_root = _normalize_under_root(root)
        if (
            not normalized_root
            or not normalized_root.exists()
            or not normalized_root.is_dir()
        ):
            continue
        for path in sorted(normalized_root.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in INFERENCE_MEDIA_EXTS:
                continue
            normalized = _normalize_under_root(path)
            if not normalized:
                continue
            key = str(normalized)
            if key in seen:
                continue
            seen.add(key)
            files.append(normalized)
            if len(files) >= limit:
                return files
    return files


def _default_inference_media() -> Path | None:
    search_dirs = [
        PROJECT_ROOT / "test-videos",
        PROJECT_ROOT,
        _get_run_dir() / "test-videos",
        _get_run_dir(),
    ]
    for directory in search_dirs:
        normalized_dir = _normalize_under_root(directory)
        if (
            not normalized_dir
            or not normalized_dir.exists()
            or not normalized_dir.is_dir()
        ):
            continue
        matches = sorted(
            [
                path
                for path in normalized_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".mp4"
            ]
        )
        if matches:
            return _normalize_under_root(matches[0])
    return None


def _safe_plan_name(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (name or "").strip())
    return slug.strip("_-")


def _default_batch_plan() -> dict[str, Any]:
    return {
        "run_dir": str(_get_run_dir()),
        "manifest": "manifests/images.csv",
        "labels_dir": "data/labels_autogen",
        "labels_final_dir": "data/labels_final",
        "split_root": "data/splits/batch",
        "train_project": "runs/detect",
        "registry_path": "reports/training_run_registry.jsonl",
        "batch_reports_root": "reports/batch_runs",
        "defaults": {
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.2,
                "test_ratio": 0.1,
                "seed": 42,
            },
            "train": {
                "epochs": 300,
                "save_period": 10,
                "imgsz": 640,
                "batch": -1,
                "device": "0",
                "count_lookalike_as_vehicle": False,
            },
        },
        "jobs": [
            {
                "name": "baseline_yolo11s",
                "split": {"seed": 42},
                "train": {"model_size": "s", "epochs": 300, "save_period": 10},
            },
            {
                "name": "smaller_yolo11n",
                "split": {
                    "train_ratio": 0.75,
                    "val_ratio": 0.15,
                    "test_ratio": 0.1,
                    "seed": 1337,
                },
                "train": {
                    "model_size": "n",
                    "epochs": 300,
                    "save_period": 10,
                    "imgsz": 512,
                },
            },
        ],
    }


def _coerce_cli_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise HTTPException(
        status_code=400,
        detail="extra_args must be a string or an array",
    )


def _assert_no_reserved_cli_flags(
    extra_args: list[str],
    reserved_flags: set[str],
    context: str,
) -> None:
    conflicts: set[str] = set()
    for token in extra_args:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            if flag in reserved_flags:
                conflicts.add(flag)
            continue
        if token.startswith("-") and len(token) >= 2:
            short_flag = token[:2]
            if short_flag in reserved_flags:
                conflicts.add(short_flag)

    if conflicts:
        joined = ", ".join(sorted(conflicts))
        raise HTTPException(
            status_code=400,
            detail=f"{context} extra_args cannot override reserved flags: {joined}",
        )


def _resolve_run_scoped_path(
    raw_value: str,
    *,
    must_exist: bool = False,
    expect_file: bool = False,
) -> Path:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Path cannot be empty")

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = _get_run_dir() / candidate

    normalized = _normalize_under_root(candidate)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"Path must remain under project root: {raw_value}",
        )

    if must_exist and not normalized.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {normalized}")

    if expect_file and normalized.exists() and not normalized.is_file():
        raise HTTPException(status_code=400, detail=f"Expected file path: {normalized}")

    return normalized


def _has_active_batch_job(run_dir_text: str) -> bool:
    with _batch_jobs_lock:
        for job in _batch_jobs.values():
            if job.get("run_dir") != run_dir_text:
                continue
            if job.get("status") in {"queued", "running"}:
                return True
    return False


def _has_active_inference_job(run_dir_text: str) -> bool:
    with _inference_jobs_lock:
        for job in _inference_jobs.values():
            if job.get("run_dir") != run_dir_text:
                continue
            if job.get("status") in {"queued", "running", "cancelling"}:
                return True
    return False


def _read_log_tail(path: Path, max_lines: int = 300) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""


def _run_batch_job_background(job_id: str, command: list[str], log_path: Path) -> None:
    def _worker() -> None:
        with _batch_jobs_lock:
            job = _batch_jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["started_at"] = datetime.now().isoformat()

        returncode = -1
        error_text = ""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(
                    f"Started: {datetime.now().isoformat()}\n"
                    f"Command: {' '.join(shlex.quote(part) for part in command)}\n\n"
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                if process.stdout is not None:
                    for line in process.stdout:
                        log_file.write(line)
                returncode = process.wait()
                log_file.write(
                    f"\nFinished: {datetime.now().isoformat()} (returncode={returncode})\n"
                )
        except Exception as exc:
            error_text = str(exc)
            try:
                with open(
                    log_path, "a", encoding="utf-8", errors="replace"
                ) as log_file:
                    log_file.write(f"\nInternal error: {error_text}\n")
            except Exception:
                pass

        with _batch_jobs_lock:
            job = _batch_jobs.get(job_id)
            if not job:
                return
            if error_text:
                job["status"] = "failed"
                job["error"] = error_text
                job["returncode"] = -1
            else:
                job["status"] = "completed" if returncode == 0 else "failed"
                job["returncode"] = returncode
            job["finished_at"] = datetime.now().isoformat()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()


def _run_inference_job_background(
    job_id: str, command: list[str], log_path: Path
) -> None:
    def _worker() -> None:
        with _inference_jobs_lock:
            job = _inference_jobs.get(job_id)
            if not job:
                return
            if job.get("status") == "cancelled":
                if not job.get("finished_at"):
                    job["finished_at"] = datetime.now().isoformat()
                return

        returncode = -1
        error_text = ""
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(
                    f"Started: {datetime.now().isoformat()}\n"
                    f"Command: {' '.join(shlex.quote(part) for part in command)}\n\n"
                )
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                with _inference_jobs_lock:
                    _inference_processes[job_id] = process
                    current_job = _inference_jobs.get(job_id)
                    if not current_job:
                        try:
                            process.terminate()
                        except Exception:
                            pass
                    else:
                        current_status = str(current_job.get("status") or "")
                        if current_status == "queued":
                            current_job["status"] = "running"
                            current_job["started_at"] = datetime.now().isoformat()
                        if current_status in {"cancelling", "cancelled"}:
                            try:
                                process.terminate()
                            except Exception:
                                pass

                if process.stdout is not None:
                    for line in process.stdout:
                        log_file.write(line)
                returncode = process.wait()
                log_file.write(
                    f"\nFinished: {datetime.now().isoformat()} (returncode={returncode})\n"
                )
        except Exception as exc:
            error_text = str(exc)
            try:
                with open(
                    log_path, "a", encoding="utf-8", errors="replace"
                ) as log_file:
                    log_file.write(f"\nInternal error: {error_text}\n")
            except Exception:
                pass
        finally:
            with _inference_jobs_lock:
                _inference_processes.pop(job_id, None)

        with _inference_jobs_lock:
            job = _inference_jobs.get(job_id)
            if not job:
                return
            current_status = str(job.get("status") or "")
            if current_status in {"cancelling", "cancelled"}:
                job["status"] = "cancelled"
                job["returncode"] = returncode if returncode != -1 else -15
            elif error_text:
                job["status"] = "failed"
                job["error"] = error_text
                job["returncode"] = -1
            else:
                job["status"] = "completed" if returncode == 0 else "failed"
                job["returncode"] = returncode
            job["finished_at"] = datetime.now().isoformat()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()


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
    normalized = _normalize_under_root(new_path)
    if not normalized:
        raise HTTPException(
            status_code=400,
            detail=f"Directory must remain under project root: {req.path}",
        )

    if not normalized.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {req.path}")
    if not normalized.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {req.path}")

    _set_run_dir(normalized)
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
    resolved = _resolve_manifest_image_path(row)
    if resolved:
        return resolved

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


class LookalikeMetadataEntry(BaseModel):
    box_key: str
    similar_to: str = ""


class LookalikeMetadataUpdate(BaseModel):
    entries: List[LookalikeMetadataEntry]


class BatchPlanSaveRequest(BaseModel):
    name: str
    config: dict[str, Any]


class BatchRunStartRequest(BaseModel):
    plan_name: Optional[str] = None
    config: Optional[dict[str, Any]] = None
    dry_run: bool = False
    continue_on_error: bool = False


class SplitRunStartRequest(BaseModel):
    name: Optional[str] = None
    config: dict[str, Any]


class InferenceRunRequest(BaseModel):
    weights_path: str
    media_path: Optional[str] = None
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "0"
    save_txt: bool = False
    save_conf: bool = False


# --- HTML Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/labeler", response_class=HTMLResponse)
async def labeler(request: Request):
    return templates.TemplateResponse("labeler.html", {"request": request})


@app.get("/oversized-box-audit", response_class=HTMLResponse)
async def oversized_box_audit(request: Request):
    return templates.TemplateResponse("oversized_box_audit.html", {"request": request})


@app.get("/lookalike-tracker", response_class=HTMLResponse)
async def lookalike_tracker(request: Request):
    return templates.TemplateResponse("lookalike_tracker.html", {"request": request})


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


@app.get("/inference-runner", response_class=HTMLResponse)
async def inference_runner(request: Request):
    return templates.TemplateResponse("inference_runner.html", {"request": request})


@app.get("/batch-run-manager", response_class=HTMLResponse)
async def batch_run_manager(request: Request):
    return templates.TemplateResponse("batch_run_manager.html", {"request": request})


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


def _plan_path_from_name(plan_name: str) -> Path:
    safe_name = _safe_plan_name(plan_name)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid plan name")
    return _batch_plans_dir() / f"{safe_name}.json"


@app.get("/api/batch-runs/template")
def get_batch_run_template():
    return _default_batch_plan()


@app.get("/api/batch-runs/plans")
def list_batch_run_plans():
    plans_dir = _batch_plans_dir()
    plans_dir.mkdir(parents=True, exist_ok=True)

    plans: list[dict[str, Any]] = []
    for path in sorted(plans_dir.glob("*.json")):
        plans.append(
            {
                "name": path.stem,
                "path": str(path),
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            }
        )
    return plans


@app.get("/api/batch-runs/plans/{plan_name}")
def get_batch_run_plan(plan_name: str):
    plan_path = _plan_path_from_name(plan_name)
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Plan not found")
    with open(plan_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"name": plan_path.stem, "path": str(plan_path), "config": payload}


@app.post("/api/batch-runs/plans")
def save_batch_run_plan(req: BatchPlanSaveRequest):
    if not isinstance(req.config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")

    plans_dir = _batch_plans_dir()
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_path = _plan_path_from_name(req.name)

    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(req.config, handle, indent=2)
        handle.write("\n")

    return {
        "status": "saved",
        "name": plan_path.stem,
        "path": str(plan_path),
        "updated_at": datetime.now().isoformat(),
    }


@app.delete("/api/batch-runs/plans/{plan_name}")
def delete_batch_run_plan(plan_name: str):
    plan_path = _plan_path_from_name(plan_name)
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Plan not found")
    plan_path.unlink()
    return {"status": "deleted", "name": plan_path.stem}


@app.post("/api/batch-runs/start")
def start_batch_run(req: BatchRunStartRequest):
    if not BATCH_MANAGER_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Batch manager script missing: {BATCH_MANAGER_SCRIPT}",
        )

    run_dir = _get_run_dir().resolve()
    run_dir_text = str(run_dir)

    if _has_active_batch_job(run_dir_text):
        raise HTTPException(
            status_code=409,
            detail="A batch run is already active for this run directory",
        )

    plans_dir = _batch_plans_dir()
    plans_dir.mkdir(parents=True, exist_ok=True)

    plan_path: Path
    if req.config is not None:
        plan_base_name = (
            req.plan_name or f"adhoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        plan_path = _plan_path_from_name(plan_base_name)
        plan_payload = dict(req.config)
        plan_payload.setdefault("run_dir", run_dir_text)
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(plan_payload, handle, indent=2)
            handle.write("\n")
    else:
        if not req.plan_name:
            raise HTTPException(
                status_code=400,
                detail="Provide plan_name or inline config",
            )
        plan_path = _plan_path_from_name(req.plan_name)
        if not plan_path.exists():
            raise HTTPException(status_code=404, detail="Plan not found")

    command = [
        sys.executable,
        "-u",
        str(BATCH_MANAGER_SCRIPT),
        "--config",
        str(plan_path),
        "--run-dir",
        run_dir_text,
    ]
    if req.dry_run:
        command.append("--dry-run")
    if req.continue_on_error:
        command.append("--continue-on-error")

    job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    log_path = _batch_logs_dir() / f"{job_id}.log"

    job_record = {
        "id": job_id,
        "kind": "batch",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "error": None,
        "run_dir": run_dir_text,
        "plan_name": plan_path.stem,
        "plan_path": str(plan_path),
        "log_path": str(log_path),
        "command": " ".join(shlex.quote(part) for part in command),
        "dry_run": req.dry_run,
        "continue_on_error": req.continue_on_error,
    }

    with _batch_jobs_lock:
        _batch_jobs[job_id] = job_record

    _run_batch_job_background(job_id=job_id, command=command, log_path=log_path)
    return job_record


@app.post("/api/batch-runs/run-split")
def run_split_only(req: SplitRunStartRequest):
    run_dir = _get_run_dir().resolve()
    run_dir_text = str(run_dir)

    if _has_active_batch_job(run_dir_text):
        raise HTTPException(
            status_code=409,
            detail="A batch or split job is already active for this run directory",
        )

    split_script = PROJECT_ROOT / "scripts" / "06_split_dataset.py"
    if not split_script.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Split script missing: {split_script}",
        )

    cfg = dict(req.config or {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_hint = _safe_plan_name(req.name or str(cfg.get("name") or ""))
    if not name_hint:
        name_hint = f"split_{timestamp}"

    manifest_path = _resolve_run_scoped_path(
        str(cfg.get("manifest") or "manifests/images.csv"),
        must_exist=True,
        expect_file=True,
    )
    labels_dir = _resolve_run_scoped_path(
        str(cfg.get("labels_dir") or "data/labels_autogen")
    )
    labels_final_dir = _resolve_run_scoped_path(
        str(cfg.get("labels_final_dir") or "data/labels_final")
    )

    out_dir_value = str(cfg.get("out_dir") or f"data/splits/manual/{name_hint}")
    out_dir = _resolve_run_scoped_path(out_dir_value)

    command = [
        sys.executable,
        "-u",
        str(split_script),
        "--manifest",
        str(manifest_path),
        "--out-dir",
        str(out_dir),
        "--labels-dir",
        str(labels_dir),
        "--labels-final-dir",
        str(labels_final_dir),
    ]

    number_flags = {
        "train_ratio": "--train-ratio",
        "val_ratio": "--val-ratio",
        "test_ratio": "--test-ratio",
        "seed": "--seed",
        "min_enforcement_val": "--min-enforcement-val",
        "min_enforcement_test": "--min-enforcement-test",
    }
    for key, flag in number_flags.items():
        value = cfg.get(key)
        if value is None or value == "":
            continue
        command.extend([flag, str(value)])

    if bool(cfg.get("skip_threshold_check")):
        command.append("--skip-threshold-check")
    if bool(cfg.get("include_rejected")):
        command.append("--include-rejected")

    extra_args = _coerce_cli_extra_args(cfg.get("extra_args"))
    reserved_flags = {
        "--manifest",
        "--out-dir",
        "--labels-dir",
        "--labels-final-dir",
        "--train-ratio",
        "--val-ratio",
        "--test-ratio",
        "--seed",
        "--min-enforcement-val",
        "--min-enforcement-test",
        "--skip-threshold-check",
        "--include-rejected",
        "-m",
        "-o",
        "-l",
    }
    _assert_no_reserved_cli_flags(extra_args, reserved_flags, "split")
    command.extend(extra_args)

    job_id = f"split_{timestamp}_{uuid.uuid4().hex[:8]}"
    log_path = _batch_logs_dir() / f"{job_id}.log"

    job_record = {
        "id": job_id,
        "kind": "split",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "error": None,
        "run_dir": run_dir_text,
        "plan_name": name_hint,
        "plan_path": None,
        "log_path": str(log_path),
        "command": " ".join(shlex.quote(part) for part in command),
        "dry_run": False,
        "continue_on_error": False,
    }

    with _batch_jobs_lock:
        _batch_jobs[job_id] = job_record

    _run_batch_job_background(job_id=job_id, command=command, log_path=log_path)
    return job_record


@app.get("/api/batch-runs/jobs")
def list_batch_run_jobs():
    run_dir_text = str(_get_run_dir().resolve())
    with _batch_jobs_lock:
        jobs = [
            dict(job)
            for job in _batch_jobs.values()
            if job.get("run_dir") == run_dir_text
        ]
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return jobs


@app.get("/api/batch-runs/jobs/{job_id}")
def get_batch_run_job(job_id: str):
    with _batch_jobs_lock:
        job = _batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    payload = dict(job)
    payload["log_tail"] = _read_log_tail(Path(payload["log_path"]), max_lines=250)
    return payload


@app.get("/api/batch-runs/jobs/{job_id}/log")
def get_batch_run_log(job_id: str, lines: int = 400):
    with _batch_jobs_lock:
        job = _batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    max_lines = max(20, min(int(lines), 2000))
    text = _read_log_tail(Path(job["log_path"]), max_lines=max_lines)
    return {"id": job_id, "lines": max_lines, "text": text}


@app.get("/api/batch-runs/jobs/{job_id}/stream")
def stream_batch_run_log(job_id: str):
    with _batch_jobs_lock:
        job = _batch_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = Path(job["log_path"])

    def _event_stream():
        last_position = 0
        keepalive_tick = 0

        while True:
            with _batch_jobs_lock:
                current_job = _batch_jobs.get(job_id)
            if not current_job:
                payload = json.dumps({"error": "Job not found"})
                yield f"event: error\ndata: {payload}\n\n"
                break

            status = str(current_job.get("status") or "")
            emitted = False
            if log_path.exists() and log_path.is_file():
                with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(last_position)
                    chunk = handle.read()
                    last_position = handle.tell()
                if chunk:
                    emitted = True
                    payload = json.dumps(
                        {
                            "chunk": chunk,
                            "status": status,
                            "returncode": current_job.get("returncode"),
                        }
                    )
                    yield f"data: {payload}\n\n"

            if status in {"completed", "failed"}:
                if log_path.exists() and log_path.is_file():
                    with open(
                        log_path, "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        handle.seek(last_position)
                        final_chunk = handle.read()
                        last_position = handle.tell()
                    if final_chunk:
                        payload = json.dumps(
                            {
                                "chunk": final_chunk,
                                "status": status,
                                "returncode": current_job.get("returncode"),
                            }
                        )
                        yield f"data: {payload}\n\n"

                done_payload = json.dumps(
                    {
                        "status": status,
                        "returncode": current_job.get("returncode"),
                    }
                )
                yield f"event: done\ndata: {done_payload}\n\n"
                break

            keepalive_tick += 1
            if not emitted and keepalive_tick % 5 == 0:
                yield ": keepalive\n\n"
            time.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _event_stream(), media_type="text/event-stream", headers=headers
    )


@app.get("/api/inference/options")
def get_inference_options():
    weights_paths = _collect_inference_weight_files()
    media_paths = _collect_inference_media_files()
    default_media = _default_inference_media()

    def _serialize(path: Path) -> dict[str, Any]:
        rel = _relative_posix(path)
        return {
            "path": str(path),
            "relative": rel,
            "name": path.name,
            "url": _file_url(path),
            "size": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }

    return {
        "weights": [_serialize(path) for path in weights_paths],
        "media": [_serialize(path) for path in media_paths],
        "default_media": str(default_media) if default_media else None,
        "default_device": "0",
        "default_imgsz": 640,
        "default_conf": 0.25,
        "default_iou": 0.45,
    }


@app.post("/api/inference/run")
def run_inference(req: InferenceRunRequest):
    if not MEDIA_INFERENCE_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Inference script missing: {MEDIA_INFERENCE_SCRIPT}",
        )

    run_dir = _get_run_dir().resolve()
    run_dir_text = str(run_dir)

    if req.conf < 0 or req.conf > 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1")
    if req.iou < 0 or req.iou > 1:
        raise HTTPException(status_code=400, detail="iou must be between 0 and 1")
    if req.imgsz < 32:
        raise HTTPException(status_code=400, detail="imgsz must be at least 32")
    device_value = str(req.device or "").strip() or "0"

    weights_path = _resolve_existing_project_file(req.weights_path, allow_exts={".pt"})
    if req.media_path and req.media_path.strip():
        media_path = _resolve_existing_project_file(req.media_path)
    else:
        default_media = _default_inference_media()
        if not default_media:
            raise HTTPException(
                status_code=400,
                detail="No media provided and no default MP4 found in test-videos, project root, or current run directory",
            )
        media_path = default_media

    if media_path.suffix.lower() not in INFERENCE_MEDIA_EXTS:
        allowed = ", ".join(sorted(INFERENCE_MEDIA_EXTS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media type '{media_path.suffix}'. Expected one of: {allowed}",
        )

    output_project = _inference_outputs_dir().resolve()
    output_project.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _safe_plan_name(
        f"{weights_path.stem}_{timestamp}_{uuid.uuid4().hex[:6]}"
    )
    if not run_name:
        run_name = f"inference_{timestamp}"

    command = [
        sys.executable,
        "-u",
        str(MEDIA_INFERENCE_SCRIPT),
        "--weights",
        str(weights_path),
        "--source",
        str(media_path),
        "--project",
        str(output_project),
        "--name",
        run_name,
        "--conf",
        str(req.conf),
        "--iou",
        str(req.iou),
        "--imgsz",
        str(req.imgsz),
        "--device",
        device_value,
    ]
    if req.save_txt:
        command.append("--save-txt")
    if req.save_conf:
        command.append("--save-conf")

    job_id = (
        f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    log_path = _inference_logs_dir() / f"{job_id}.log"
    preview_dir = _inference_previews_dir() / job_id
    output_dir = output_project / run_name

    command.extend(["--preview-dir", str(preview_dir), "--preview-every", "1"])

    job_record = {
        "id": job_id,
        "kind": "inference",
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "error": None,
        "run_dir": run_dir_text,
        "weights_path": str(weights_path),
        "media_path": str(media_path),
        "output_project": str(output_project),
        "output_dir": str(output_dir),
        "preview_dir": str(preview_dir),
        "log_path": str(log_path),
        "command": " ".join(shlex.quote(part) for part in command),
        "settings": {
            "conf": req.conf,
            "iou": req.iou,
            "imgsz": req.imgsz,
            "device": device_value,
            "save_txt": req.save_txt,
            "save_conf": req.save_conf,
        },
    }

    with _inference_jobs_lock:
        for existing in _inference_jobs.values():
            if existing.get("run_dir") != run_dir_text:
                continue
            if existing.get("status") in {"queued", "running", "cancelling"}:
                raise HTTPException(
                    status_code=409,
                    detail="An inference run is already active for this run directory",
                )
        _inference_jobs[job_id] = job_record

    _run_inference_job_background(job_id=job_id, command=command, log_path=log_path)
    return job_record


@app.get("/api/inference/jobs")
def list_inference_jobs():
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        jobs = [
            dict(job)
            for job in _inference_jobs.values()
            if job.get("run_dir") == run_dir_text
        ]
    jobs.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return jobs


@app.get("/api/inference/jobs/{job_id}")
def get_inference_job(job_id: str):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    payload = dict(job)
    payload["log_tail"] = _read_log_tail(Path(payload["log_path"]), max_lines=250)

    output_dir = Path(payload["output_dir"])
    normalized_output_dir = _normalize_under_root(output_dir)
    if (
        normalized_output_dir
        and normalized_output_dir.exists()
        and normalized_output_dir.is_dir()
    ):
        files = [
            _file_url(path)
            for path in sorted(normalized_output_dir.rglob("*"))
            if path.is_file()
        ]
        payload["output_files"] = files
    else:
        payload["output_files"] = []

    return payload


@app.get("/api/inference/jobs/{job_id}/preview")
def get_inference_preview(job_id: str):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    preview_dir_raw = job.get("preview_dir")
    if not preview_dir_raw:
        return {
            "id": job_id,
            "status": job.get("status"),
            "has_image": False,
            "image_url": None,
            "updated_at": None,
            "frame": None,
        }

    preview_dir = _normalize_under_root(Path(str(preview_dir_raw)))
    if not preview_dir:
        raise HTTPException(status_code=400, detail="Invalid preview path")

    latest_image = preview_dir / "latest.jpg"
    latest_meta = preview_dir / "latest.json"

    updated_at: str | None = None
    frame: int | None = None
    if latest_meta.exists() and latest_meta.is_file():
        try:
            with open(latest_meta, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            updated_at = str(payload.get("updated_at") or "") or None
            frame_value = payload.get("frame")
            if isinstance(frame_value, int):
                frame = frame_value
            elif isinstance(frame_value, str) and frame_value.isdigit():
                frame = int(frame_value)
        except Exception:
            pass

    has_image = latest_image.exists() and latest_image.is_file()
    image_url = (
        f"/api/inference/jobs/{quote(job_id)}/preview.jpg" if has_image else None
    )

    return {
        "id": job_id,
        "status": job.get("status"),
        "has_image": has_image,
        "image_url": image_url,
        "updated_at": updated_at,
        "frame": frame,
    }


@app.get("/api/inference/jobs/{job_id}/preview.jpg")
def get_inference_preview_image(job_id: str):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    preview_dir_raw = job.get("preview_dir")
    if not preview_dir_raw:
        raise HTTPException(status_code=404, detail="Preview not available")

    preview_dir = _normalize_under_root(Path(str(preview_dir_raw)))
    if not preview_dir:
        raise HTTPException(status_code=400, detail="Invalid preview path")

    latest_image = preview_dir / "latest.jpg"
    if not latest_image.exists() or not latest_image.is_file():
        raise HTTPException(status_code=404, detail="Preview not available")

    return FileResponse(
        str(latest_image),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/inference/jobs/{job_id}/preview-stream")
def stream_inference_preview(job_id: str):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    preview_dir_raw = job.get("preview_dir")
    if not preview_dir_raw:
        raise HTTPException(status_code=404, detail="Preview not available")

    preview_dir = _normalize_under_root(Path(str(preview_dir_raw)))
    if not preview_dir:
        raise HTTPException(status_code=400, detail="Invalid preview path")

    latest_image = preview_dir / "latest.jpg"

    def _mjpeg_stream():
        last_sig: tuple[int, int] | None = None
        while True:
            with _inference_jobs_lock:
                current_job = _inference_jobs.get(job_id)
            if not current_job:
                break

            status = str(current_job.get("status") or "")
            emitted = False

            if latest_image.exists() and latest_image.is_file():
                stat = latest_image.stat()
                sig = (int(stat.st_mtime_ns), int(stat.st_size))
                if sig != last_sig:
                    try:
                        frame_bytes = latest_image.read_bytes()
                    except Exception:
                        frame_bytes = b""
                    if frame_bytes:
                        last_sig = sig
                        emitted = True
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame_bytes)}\r\n\r\n".encode(
                                "ascii"
                            )
                        )
                        yield header + frame_bytes + b"\r\n"

            if status in INFERENCE_TERMINAL_STATUSES:
                if latest_image.exists() and latest_image.is_file():
                    final_stat = latest_image.stat()
                    final_sig = (int(final_stat.st_mtime_ns), int(final_stat.st_size))
                    if final_sig != last_sig:
                        try:
                            final_bytes = latest_image.read_bytes()
                        except Exception:
                            final_bytes = b""
                        if final_bytes:
                            final_header = (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(final_bytes)}\r\n\r\n".encode(
                                    "ascii"
                                )
                            )
                            yield final_header + final_bytes + b"\r\n"
                            last_sig = final_sig
                break

            if not emitted:
                time.sleep(0.03)

        yield b"--frame--\r\n"

    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@app.post("/api/inference/jobs/{job_id}/cancel")
def cancel_inference_job(job_id: str):
    run_dir_text = str(_get_run_dir().resolve())
    process: subprocess.Popen[Any] | None = None
    should_signal = False
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.get("run_dir") != run_dir_text:
            raise HTTPException(status_code=404, detail="Job not found")

        status = str(job.get("status") or "")
        if status in INFERENCE_TERMINAL_STATUSES:
            return dict(job)

        now = datetime.now().isoformat()
        if not job.get("cancel_requested_at"):
            job["cancel_requested_at"] = now

        if status == "queued":
            job["status"] = "cancelled"
            job["finished_at"] = now
            job["returncode"] = -15
            return dict(job)

        process = _inference_processes.get(job_id)
        if status == "running" and process is not None and process.poll() is None:
            job["status"] = "cancelling"
            should_signal = True
        elif status == "cancelling":
            should_signal = process is not None and process.poll() is None

    if should_signal and process is not None and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    with _inference_jobs_lock:
        final_job = _inference_jobs.get(job_id)
        if not final_job:
            raise HTTPException(status_code=404, detail="Job not found")
        return dict(final_job)


@app.get("/api/inference/jobs/{job_id}/log")
def get_inference_log(job_id: str, lines: int = 400):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    max_lines = max(20, min(int(lines), 2000))
    text = _read_log_tail(Path(job["log_path"]), max_lines=max_lines)
    return {"id": job_id, "lines": max_lines, "text": text}


@app.get("/api/inference/jobs/{job_id}/stream")
def stream_inference_log(job_id: str, from_end: bool = False):
    run_dir_text = str(_get_run_dir().resolve())
    with _inference_jobs_lock:
        job = _inference_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("run_dir") != run_dir_text:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = Path(job["log_path"])

    def _event_stream():
        if from_end and log_path.exists() and log_path.is_file():
            last_position = log_path.stat().st_size
        else:
            last_position = 0
        keepalive_tick = 0

        while True:
            with _inference_jobs_lock:
                current_job = _inference_jobs.get(job_id)
            if not current_job:
                payload = json.dumps({"error": "Job not found"})
                yield f"event: error\ndata: {payload}\n\n"
                break

            status = str(current_job.get("status") or "")
            emitted = False
            if log_path.exists() and log_path.is_file():
                with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(last_position)
                    chunk = handle.read()
                    last_position = handle.tell()
                if chunk:
                    emitted = True
                    payload = json.dumps(
                        {
                            "chunk": chunk,
                            "status": status,
                            "returncode": current_job.get("returncode"),
                        }
                    )
                    yield f"data: {payload}\n\n"

            if status in INFERENCE_TERMINAL_STATUSES:
                if log_path.exists() and log_path.is_file():
                    with open(
                        log_path, "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        handle.seek(last_position)
                        final_chunk = handle.read()
                        last_position = handle.tell()
                    if final_chunk:
                        payload = json.dumps(
                            {
                                "chunk": final_chunk,
                                "status": status,
                                "returncode": current_job.get("returncode"),
                            }
                        )
                        yield f"data: {payload}\n\n"

                done_payload = json.dumps(
                    {
                        "status": status,
                        "returncode": current_job.get("returncode"),
                    }
                )
                yield f"event: done\ndata: {done_payload}\n\n"
                break

            keepalive_tick += 1
            if not emitted and keepalive_tick % 5 == 0:
                yield ": keepalive\n\n"
            time.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _event_stream(), media_type="text/event-stream", headers=headers
    )


# --- Labeler API ---


def _is_rejected_row(row: dict) -> bool:
    return (row.get("review_status") or "").strip().lower() == "rejected"


def _allow_labeler_rejected(include_rejected: bool = False) -> bool:
    return include_rejected or _labeler_include_rejected


def _raise_if_labeler_hidden(row: dict, include_rejected: bool = False) -> None:
    if _is_rejected_row(row) and not _allow_labeler_rejected(include_rejected):
        raise HTTPException(status_code=404, detail="Image not found")


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


def _collect_synth_image_ids(include_rejected: bool = False) -> tuple[set[str], int]:
    manifest = load_manifest()
    synth_ids: set[str] = set()
    rejected_excluded = 0
    for row in manifest:
        if (row.get("status") or "").strip() != "ok":
            continue
        if (row.get("is_synthetic") or "0").strip() != "1":
            continue
        if _is_rejected_row(row) and not include_rejected:
            rejected_excluded += 1
            continue
        image_id = (row.get("image_id") or "").strip()
        if image_id:
            synth_ids.add(image_id)
    return synth_ids, rejected_excluded


def _validate_box_metric_inputs(
    metric: str, threshold: float
) -> tuple[Literal["width", "height", "area"], float]:
    normalized_threshold = threshold
    if normalized_threshold >= 1 and normalized_threshold <= 100:
        normalized_threshold = normalized_threshold / 100.0
    if normalized_threshold <= 0 or normalized_threshold >= 1:
        raise HTTPException(
            status_code=400,
            detail="threshold must be between 0 and 1, or between 0 and 100",
        )

    normalized_metric = (metric or "").strip().lower()
    if normalized_metric not in {"width", "height", "area"}:
        raise HTTPException(
            status_code=400,
            detail="metric must be one of: width, height, area",
        )
    return cast(
        Literal["width", "height", "area"], normalized_metric
    ), normalized_threshold


def _normalize_optional_bound(value: float | None, label: str) -> float | None:
    if value is None:
        return None

    normalized = value
    if normalized >= 1 and normalized <= 100:
        normalized = normalized / 100.0
    if normalized < 0 or normalized > 1:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be between 0 and 1, or between 0 and 100",
        )
    return normalized


def _resolve_oversized_filter_bounds(
    metric: str | None = None,
    threshold: float | None = None,
    width_min: float | None = None,
    width_max: float | None = None,
    height_min: float | None = None,
    height_max: float | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
) -> dict[str, Any]:
    legacy_metric: Literal["width", "height", "area"] | None = None
    legacy_threshold: float | None = None
    if (metric is None) != (threshold is None):
        raise HTTPException(
            status_code=400,
            detail="metric and threshold must be provided together",
        )
    if metric is not None and threshold is not None:
        legacy_metric, legacy_threshold = _validate_box_metric_inputs(metric, threshold)

    bounds: dict[str, float | None] = {
        "width_min": _normalize_optional_bound(width_min, "width_min"),
        "width_max": _normalize_optional_bound(width_max, "width_max"),
        "height_min": _normalize_optional_bound(height_min, "height_min"),
        "height_max": _normalize_optional_bound(height_max, "height_max"),
        "area_min": _normalize_optional_bound(area_min, "area_min"),
        "area_max": _normalize_optional_bound(area_max, "area_max"),
    }

    if legacy_metric is not None and any(
        value is not None for value in bounds.values()
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide either legacy metric+threshold or range bounds, not both",
        )

    if legacy_metric is not None and legacy_threshold is not None:
        min_key = f"{legacy_metric}_min"
        if bounds[min_key] is None:
            bounds[min_key] = legacy_threshold

    for metric_name in ("width", "height", "area"):
        min_key = f"{metric_name}_min"
        max_key = f"{metric_name}_max"
        min_v = bounds[min_key]
        max_v = bounds[max_key]
        if min_v is not None and max_v is not None and min_v > max_v:
            raise HTTPException(
                status_code=400,
                detail=f"{min_key} cannot be greater than {max_key}",
            )

    if all(value is None for value in bounds.values()):
        raise HTTPException(
            status_code=400,
            detail="At least one filter bound is required",
        )

    return {
        "bounds": bounds,
        "legacy_metric": legacy_metric,
        "legacy_threshold_input": threshold,
        "legacy_threshold": legacy_threshold,
    }


def _box_metric_value(
    metric: Literal["width", "height", "area"], w: float, h: float
) -> float:
    if metric == "width":
        return w
    if metric == "height":
        return h
    return w * h


def _box_matches_oversized_filters(
    w: float, h: float, bounds: dict[str, float | None]
) -> bool:
    area = w * h
    values = {
        "width": w,
        "height": h,
        "area": area,
    }
    for metric_name, value in values.items():
        min_v = bounds.get(f"{metric_name}_min")
        max_v = bounds.get(f"{metric_name}_max")
        if min_v is not None and value < min_v:
            return False
        if max_v is not None and value > max_v:
            return False
    return True


def _prune_massive_synth_boxes(
    area_threshold: float = 0.40,
    include_rejected: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if area_threshold <= 0 or area_threshold >= 1:
        raise HTTPException(
            status_code=400, detail="area_threshold must be between 0 and 1"
        )

    synth_ids, rejected_excluded = _collect_synth_image_ids(
        include_rejected=include_rejected
    )

    labels_dir = _labels_autogen_dir()
    stats: dict[str, Any] = {
        "run_dir": str(_get_run_dir()),
        "labels_dir": str(labels_dir),
        "area_threshold": area_threshold,
        "dry_run": dry_run,
        "include_rejected": include_rejected,
        "rejected_synthetic_excluded": rejected_excluded,
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


def _audit_oversized_synth_boxes(
    metric: str | None = None,
    threshold: float | None = None,
    width_min: float | None = None,
    width_max: float | None = None,
    height_min: float | None = None,
    height_max: float | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    include_rejected: bool = False,
    max_results: int = 300,
) -> dict[str, Any]:
    resolved = _resolve_oversized_filter_bounds(
        metric=metric,
        threshold=threshold,
        width_min=width_min,
        width_max=width_max,
        height_min=height_min,
        height_max=height_max,
        area_min=area_min,
        area_max=area_max,
    )
    bounds = cast(dict[str, float | None], resolved["bounds"])
    legacy_metric = cast(
        Literal["width", "height", "area"] | None, resolved["legacy_metric"]
    )
    legacy_threshold = cast(float | None, resolved["legacy_threshold"])

    max_results = max(1, min(int(max_results), 5000))

    synth_ids, rejected_excluded = _collect_synth_image_ids(
        include_rejected=include_rejected
    )
    labels_dir = _labels_autogen_dir()

    image_url_by_id: dict[str, str] = {}
    for row in load_manifest():
        if (row.get("status") or "").strip() != "ok":
            continue
        if (row.get("is_synthetic") or "0").strip() != "1":
            continue
        if _is_rejected_row(row) and not include_rejected:
            continue
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            continue
        image_path = _resolve_manifest_image_path(row)
        if image_path:
            image_url_by_id[image_id] = _file_url(image_path)

    stats: dict[str, Any] = {
        "run_dir": str(_get_run_dir()),
        "labels_dir": str(labels_dir),
        "metric": legacy_metric,
        "threshold_input": resolved["legacy_threshold_input"],
        "threshold": legacy_threshold,
        "filter_bounds": bounds,
        "include_rejected": include_rejected,
        "rejected_synthetic_excluded": rejected_excluded,
        "synthetic_images_total": len(synth_ids),
        "label_files_present": 0,
        "label_files_missing": 0,
        "label_files_empty": 0,
        "boxes_total": 0,
        "oversized_boxes_found": 0,
        "images_with_oversized_boxes": 0,
        "invalid_label_lines": 0,
        "max_results": max_results,
        "matches_returned": 0,
        "matches_truncated": False,
    }

    matches: list[dict[str, Any]] = []
    if not labels_dir.exists():
        return {"stats": stats, "matches": matches}

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
        file_oversized = 0
        for idx, line in enumerate(lines):
            parsed = _parse_yolo_line(line)
            if parsed is None:
                stats["invalid_label_lines"] += 1
                continue

            cls, xc, yc, w, h = parsed
            stats["boxes_total"] += 1
            area = w * h
            if not _box_matches_oversized_filters(w, h, bounds):
                continue

            if legacy_metric is not None:
                metric_name: Literal["width", "height", "area"] = legacy_metric
                metric_value = _box_metric_value(metric_name, w, h)
            else:
                metric_name = "area"
                metric_value = area

            file_oversized += 1
            stats["oversized_boxes_found"] += 1

            if len(matches) < max_results:
                matches.append(
                    {
                        "image_id": image_id,
                        "image_url": image_url_by_id.get(
                            image_id, f"/api/labeler/image_file/{image_id}"
                        ),
                        "label_path": _relative_posix(label_path),
                        "box_index": idx,
                        "cls": cls,
                        "x_center": xc,
                        "y_center": yc,
                        "width": w,
                        "height": h,
                        "area": area,
                        "box_key": _normalized_box_key(cls, xc, yc, w, h),
                        "metric": metric_name,
                        "metric_value": metric_value,
                    }
                )

        if file_oversized > 0:
            stats["images_with_oversized_boxes"] += 1

    stats["matches_returned"] = len(matches)
    stats["matches_truncated"] = stats["oversized_boxes_found"] > len(matches)
    return {"stats": stats, "matches": matches}


def _prune_oversized_synth_boxes(
    metric: str | None = None,
    threshold: float | None = None,
    width_min: float | None = None,
    width_max: float | None = None,
    height_min: float | None = None,
    height_max: float | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    include_rejected: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    resolved = _resolve_oversized_filter_bounds(
        metric=metric,
        threshold=threshold,
        width_min=width_min,
        width_max=width_max,
        height_min=height_min,
        height_max=height_max,
        area_min=area_min,
        area_max=area_max,
    )
    bounds = cast(dict[str, float | None], resolved["bounds"])
    legacy_metric = cast(
        Literal["width", "height", "area"] | None, resolved["legacy_metric"]
    )
    legacy_threshold = cast(float | None, resolved["legacy_threshold"])

    synth_ids, rejected_excluded = _collect_synth_image_ids(
        include_rejected=include_rejected
    )
    labels_dir = _labels_autogen_dir()

    stats: dict[str, Any] = {
        "run_dir": str(_get_run_dir()),
        "labels_dir": str(labels_dir),
        "metric": legacy_metric,
        "threshold_input": resolved["legacy_threshold_input"],
        "threshold": legacy_threshold,
        "filter_bounds": bounds,
        "dry_run": dry_run,
        "include_rejected": include_rejected,
        "rejected_synthetic_excluded": rejected_excluded,
        "synthetic_images_total": len(synth_ids),
        "label_files_present": 0,
        "label_files_missing": 0,
        "label_files_empty": 0,
        "boxes_total": 0,
        "oversized_boxes_found": 0,
        "oversized_boxes_removed": 0,
        "images_with_oversized_boxes": 0,
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
            if _box_matches_oversized_filters(w, h, bounds):
                removed_count += 1
                continue
            keep_lines.append(line)

        if removed_count == 0:
            continue

        stats["images_with_oversized_boxes"] += 1
        stats["oversized_boxes_found"] += removed_count
        stats["oversized_boxes_removed"] += removed_count

        if not dry_run:
            label_path.write_text(
                ("\n".join(keep_lines) + "\n") if keep_lines else "",
                encoding="utf-8",
            )
            stats["images_modified"] += 1

    return stats


def _prune_selected_synth_box_keys(
    targets: list[dict[str, Any]],
    metric: str | None = None,
    threshold: float | None = None,
    width_min: float | None = None,
    width_max: float | None = None,
    height_min: float | None = None,
    height_max: float | None = None,
    area_min: float | None = None,
    area_max: float | None = None,
    include_rejected: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    resolved = _resolve_oversized_filter_bounds(
        metric=metric,
        threshold=threshold,
        width_min=width_min,
        width_max=width_max,
        height_min=height_min,
        height_max=height_max,
        area_min=area_min,
        area_max=area_max,
    )
    bounds = cast(dict[str, float | None], resolved["bounds"])
    legacy_metric = cast(
        Literal["width", "height", "area"] | None, resolved["legacy_metric"]
    )
    legacy_threshold = cast(float | None, resolved["legacy_threshold"])

    labels_dir = _labels_autogen_dir()
    manifest = load_manifest()
    row_by_image_id = {
        (row.get("image_id") or "").strip(): row
        for row in manifest
        if (row.get("image_id") or "").strip()
    }

    target_counters: dict[str, Counter[str]] = {}
    for target in targets:
        image_id = str(target.get("image_id") or "").strip()
        if not image_id:
            continue
        box_keys = [
            str(value).strip()
            for value in (target.get("box_keys") or [])
            if str(value).strip()
        ]
        if not box_keys:
            continue
        if image_id not in target_counters:
            target_counters[image_id] = Counter()
        target_counters[image_id].update(box_keys)

    stats: dict[str, Any] = {
        "run_dir": str(_get_run_dir()),
        "labels_dir": str(labels_dir),
        "metric": legacy_metric,
        "threshold_input": resolved["legacy_threshold_input"],
        "threshold": legacy_threshold,
        "filter_bounds": bounds,
        "dry_run": dry_run,
        "include_rejected": include_rejected,
        "images_targeted": len(target_counters),
        "boxes_targeted": sum(
            sum(counter.values()) for counter in target_counters.values()
        ),
        "valid_synth_targets": 0,
        "invalid_target_images": 0,
        "rejected_targets_excluded": 0,
        "label_files_missing": 0,
        "label_files_empty": 0,
        "invalid_label_lines": 0,
        "boxes_removed": 0,
        "boxes_requested_not_found": 0,
        "images_with_removed_boxes": 0,
        "images_modified": 0,
    }

    if not labels_dir.exists():
        stats["boxes_requested_not_found"] = stats["boxes_targeted"]
        return stats

    for image_id, requested_counts in target_counters.items():
        row = row_by_image_id.get(image_id)
        if not row:
            stats["invalid_target_images"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue
        if (row.get("status") or "").strip() != "ok":
            stats["invalid_target_images"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue
        if (row.get("is_synthetic") or "0").strip() != "1":
            stats["invalid_target_images"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue
        if _is_rejected_row(row) and not include_rejected:
            stats["rejected_targets_excluded"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue

        stats["valid_synth_targets"] += 1

        label_path = labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            stats["label_files_missing"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue

        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            stats["label_files_empty"] += 1
            stats["boxes_requested_not_found"] += sum(requested_counts.values())
            continue

        remaining = Counter(requested_counts)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        keep_lines: list[str] = []
        removed_in_image = 0

        for line in lines:
            parsed = _parse_yolo_line(line)
            if parsed is None:
                stats["invalid_label_lines"] += 1
                keep_lines.append(line)
                continue

            cls, xc, yc, w, h = parsed
            box_key = _normalized_box_key(cls, xc, yc, w, h)
            if remaining.get(box_key, 0) > 0 and _box_matches_oversized_filters(
                w, h, bounds
            ):
                remaining[box_key] -= 1
                removed_in_image += 1
                continue

            keep_lines.append(line)

        not_found = sum(count for count in remaining.values() if count > 0)
        stats["boxes_requested_not_found"] += not_found

        if removed_in_image == 0:
            continue

        stats["boxes_removed"] += removed_in_image
        stats["images_with_removed_boxes"] += 1
        if not dry_run:
            label_path.write_text(
                ("\n".join(keep_lines) + "\n") if keep_lines else "",
                encoding="utf-8",
            )
            stats["images_modified"] += 1

    return stats


def _lookalike_metadata_path(image_id: str) -> Path:
    safe_image_id = (image_id or "").strip()
    return _lookalike_metadata_dir() / f"{safe_image_id}.json"


def _normalized_box_key(
    cls: int, x_center: float, y_center: float, width: float, height: float
) -> str:
    return f"{cls}:{x_center:.6f}:{y_center:.6f}:{width:.6f}:{height:.6f}"


def _resolve_label_path(image_id: str) -> Path | None:
    label_path_final = _labels_final_dir() / f"{image_id}.txt"
    label_path_autogen = _labels_autogen_dir() / f"{image_id}.txt"

    if label_path_final.exists():
        return label_path_final
    if label_path_autogen.exists():
        return label_path_autogen
    return None


def _load_image_boxes(image_id: str, img_w: int, img_h: int) -> list[dict]:
    label_path = _resolve_label_path(image_id)
    if not label_path:
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                continue

            x1 = (x_center - width / 2) * img_w
            y1 = (y_center - height / 2) * img_h
            x2 = (x_center + width / 2) * img_w
            y2 = (y_center + height / 2) * img_h
            boxes.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cls": cls,
                    "box_key": _normalized_box_key(
                        cls, x_center, y_center, width, height
                    ),
                }
            )
    return boxes


def _count_lookalike_boxes(image_id: str) -> int:
    label_path = _resolve_label_path(image_id)
    if not label_path:
        return 0

    count = 0
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                if int(parts[0]) == LOOKALIKE_CLASS_ID:
                    count += 1
            except ValueError:
                continue
    return count


def _load_lookalike_metadata(image_id: str) -> dict[str, str]:
    metadata_path = _lookalike_metadata_path(image_id)
    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(payload, dict):
        return {}

    raw_mapping = payload.get("box_similarity")
    if not isinstance(raw_mapping, dict):
        raw_mapping = payload

    normalized: dict[str, str] = {}
    for key, value in raw_mapping.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        clean_value = value.strip()
        if clean_value in LOOKALIKE_SIMILARITY_OPTIONS and clean_value:
            normalized[key.strip()] = clean_value
    return normalized


def _save_lookalike_metadata(image_id: str, mapping: dict[str, str]) -> None:
    metadata_path = _lookalike_metadata_path(image_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "image_id": image_id,
        "updated_at": datetime.now().isoformat(),
        "box_similarity": {
            key: mapping[key]
            for key in sorted(mapping.keys())
            if mapping[key] in LOOKALIKE_SIMILARITY_OPTIONS and mapping[key]
        },
    }

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=str(metadata_path.parent),
    )
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_name, metadata_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


@app.get("/api/labeler/images")
async def get_labeler_images(
    queue: str = "needs_review",
    status: str = "",
    include_excluded: bool = False,
    include_rejected: bool = False,
):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()
    images = []
    allow_rejected = _allow_labeler_rejected(include_rejected)
    for row in manifest:
        if row.get("status") != "ok":
            continue
        if _is_rejected_row(row) and not allow_rejected:
            continue
        image_id = row.get("image_id")
        if not include_excluded and image_id in excluded_ids:
            continue
        needs_review_value = row.get("needs_review", "")
        is_synthetic = row.get("is_synthetic") == "1"

        if queue == "needs_review" and needs_review_value != "1":
            continue
        if queue == "autolabel_ok" and needs_review_value != "0":
            continue
        if queue == "synthetic_only" and not is_synthetic:
            continue

        row_status = (row.get("review_status") or "").strip()
        status_filter = (status or "").strip()
        if status_filter.lower() in {"all", "*"}:
            status_filter = ""
        if status_filter and row_status not in ["", status_filter]:
            continue
        if row_status == "done":
            continue
        image_path = get_image_path(row)
        if image_path and image_path.exists():
            images.append(
                {
                    "image_id": row.get("image_id"),
                    "image_path": f"/api/labeler/image_file/{row.get('image_id')}",
                    "is_synthetic": is_synthetic,
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
    _raise_if_labeler_hidden(row)
    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    from PIL import Image

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    boxes_with_keys = _load_image_boxes(image_id, img_w, img_h)
    boxes = [
        {"x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"], "cls": b["cls"]}
        for b in boxes_with_keys
    ]

    return {
        "image_id": image_id,
        "image_path": f"/api/labeler/image_file/{image_id}",
        "boxes": boxes,
        "is_synthetic": row.get("is_synthetic") == "1",
        "is_excluded": image_id in excluded_ids,
        "edit_type": row.get("edit_type", ""),
        "expected_class": row.get("expected_inserted_class", ""),
        "lookalike_similarity": _load_lookalike_metadata(image_id),
    }


@app.get("/api/lookalike/images")
async def get_lookalike_images(include_rejected: bool = False):
    manifest = load_manifest()
    excluded_ids = load_excluded_ids()
    images = []
    allow_rejected = _allow_labeler_rejected(include_rejected)

    for row in manifest:
        if row.get("status") != "ok":
            continue
        if _is_rejected_row(row) and not allow_rejected:
            continue
        image_id = (row.get("image_id") or "").strip()
        if not image_id:
            continue
        image_path = get_image_path(row)
        if not image_path or not image_path.exists():
            continue

        lookalike_count = _count_lookalike_boxes(image_id)
        if lookalike_count <= 0:
            continue

        images.append(
            {
                "image_id": image_id,
                "image_path": f"/api/labeler/image_file/{image_id}",
                "is_synthetic": row.get("is_synthetic") == "1",
                "is_excluded": image_id in excluded_ids,
                "edit_type": row.get("edit_type", ""),
                "expected_class": row.get("expected_inserted_class", ""),
                "review_status": row.get("review_status", "todo"),
                "lookalike_count": lookalike_count,
            }
        )

    return images


@app.get("/api/lookalike/image/{image_id}")
async def get_lookalike_image(image_id: str):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    _raise_if_labeler_hidden(row)

    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    boxes = _load_image_boxes(image_id, img_w, img_h)
    lookalike_boxes = [box for box in boxes if box["cls"] == LOOKALIKE_CLASS_ID]
    if not lookalike_boxes:
        raise HTTPException(status_code=404, detail="No lookalike boxes for this image")

    mapping = _load_lookalike_metadata(image_id)
    lookalike_entries = []
    for idx, box in enumerate(lookalike_boxes):
        lookalike_entries.append(
            {
                "index": idx,
                "box_key": box["box_key"],
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "similar_to": mapping.get(box["box_key"], ""),
            }
        )

    return {
        "image_id": image_id,
        "image_path": f"/api/labeler/image_file/{image_id}",
        "boxes": boxes,
        "lookalike_boxes": lookalike_entries,
        "similarity_options": [
            "enforcement_vehicle",
            "police_old",
            "police_new",
            "unsure",
        ],
        "is_synthetic": row.get("is_synthetic") == "1",
        "edit_type": row.get("edit_type", ""),
        "expected_class": row.get("expected_inserted_class", ""),
    }


@app.post("/api/lookalike/metadata/{image_id}")
async def save_lookalike_metadata(image_id: str, data: LookalikeMetadataUpdate):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    _raise_if_labeler_hidden(row)

    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    boxes = _load_image_boxes(image_id, img_w, img_h)
    valid_keys = {
        box["box_key"] for box in boxes if int(box.get("cls", -1)) == LOOKALIKE_CLASS_ID
    }
    if not valid_keys:
        raise HTTPException(status_code=400, detail="Image has no lookalike boxes")

    mapping: dict[str, str] = {}
    unknown_keys: list[str] = []
    for entry in data.entries:
        box_key = entry.box_key.strip()
        similar_to = entry.similar_to.strip()
        if box_key not in valid_keys:
            unknown_keys.append(box_key)
            continue
        if similar_to not in LOOKALIKE_SIMILARITY_OPTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid similar_to value: {similar_to}",
            )
        if similar_to:
            mapping[box_key] = similar_to

    if unknown_keys:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Lookalike boxes changed. Refresh and retry.",
                "unknown_box_keys": sorted(set(unknown_keys)),
                "valid_box_keys": sorted(valid_keys),
            },
        )

    _save_lookalike_metadata(image_id, mapping)
    return {
        "status": "saved",
        "image_id": image_id,
        "saved_entries": len(mapping),
        "total_lookalike_boxes": len(valid_keys),
    }


@app.get("/api/labeler/image_file/{image_id}")
async def get_labeler_image_file(image_id: str):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    _raise_if_labeler_hidden(row)
    image_path = get_image_path(row)
    if not image_path or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(str(image_path))


@app.post("/api/labeler/labels/{image_id}")
async def save_labeler_labels(image_id: str, data: LabelUpdate):
    manifest = load_manifest()
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    _raise_if_labeler_hidden(row)

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
    row = get_row_by_image_id(manifest, image_id)
    if not row:
        raise HTTPException(status_code=404, detail="Image not found")
    _raise_if_labeler_hidden(row)

    for row in manifest:
        if row.get("image_id") == image_id:
            row["review_status"] = status
            row["needs_review"] = "0" if status == "done" else "1"
            break
    save_manifest(manifest)
    return {"status": status}


class MassiveBoxCleanupRequest(BaseModel):
    area_threshold: float = 0.40
    include_rejected: bool = False
    dry_run: bool = False


@app.post("/api/labeler/cleanup-massive-boxes")
def cleanup_massive_label_boxes(req: MassiveBoxCleanupRequest):
    stats = _prune_massive_synth_boxes(
        area_threshold=req.area_threshold,
        include_rejected=req.include_rejected,
        dry_run=req.dry_run,
    )
    return {"status": "ok", "stats": stats}


class OversizedBoxAuditRequest(BaseModel):
    metric: Literal["width", "height", "area"] | None = None
    threshold: float | None = None
    width_min: float | None = None
    width_max: float | None = None
    height_min: float | None = None
    height_max: float | None = None
    area_min: float | None = None
    area_max: float | None = None
    include_rejected: bool = False
    max_results: int = 300


@app.post("/api/labeler/oversized-boxes/audit")
def audit_oversized_label_boxes(req: OversizedBoxAuditRequest):
    result = _audit_oversized_synth_boxes(
        metric=req.metric,
        threshold=req.threshold,
        width_min=req.width_min,
        width_max=req.width_max,
        height_min=req.height_min,
        height_max=req.height_max,
        area_min=req.area_min,
        area_max=req.area_max,
        include_rejected=req.include_rejected,
        max_results=req.max_results,
    )
    return {"status": "ok", **result}


class OversizedBoxPruneRequest(BaseModel):
    metric: Literal["width", "height", "area"] | None = None
    threshold: float | None = None
    width_min: float | None = None
    width_max: float | None = None
    height_min: float | None = None
    height_max: float | None = None
    area_min: float | None = None
    area_max: float | None = None
    include_rejected: bool = False
    dry_run: bool = False


@app.post("/api/labeler/oversized-boxes/prune")
def prune_oversized_label_boxes(req: OversizedBoxPruneRequest):
    stats = _prune_oversized_synth_boxes(
        metric=req.metric,
        threshold=req.threshold,
        width_min=req.width_min,
        width_max=req.width_max,
        height_min=req.height_min,
        height_max=req.height_max,
        area_min=req.area_min,
        area_max=req.area_max,
        include_rejected=req.include_rejected,
        dry_run=req.dry_run,
    )
    return {"status": "ok", "stats": stats}


class OversizedBoxTargetRequest(BaseModel):
    image_id: str
    box_keys: list[str]


class OversizedBoxPruneSelectedRequest(BaseModel):
    targets: list[OversizedBoxTargetRequest]
    metric: Literal["width", "height", "area"] | None = None
    threshold: float | None = None
    width_min: float | None = None
    width_max: float | None = None
    height_min: float | None = None
    height_max: float | None = None
    area_min: float | None = None
    area_max: float | None = None
    include_rejected: bool = False
    dry_run: bool = False


@app.post("/api/labeler/oversized-boxes/prune-selected")
def prune_selected_oversized_label_boxes(req: OversizedBoxPruneSelectedRequest):
    targets_payload = [
        {"image_id": target.image_id, "box_keys": target.box_keys}
        for target in req.targets
    ]
    stats = _prune_selected_synth_box_keys(
        targets=targets_payload,
        metric=req.metric,
        threshold=req.threshold,
        width_min=req.width_min,
        width_max=req.width_max,
        height_min=req.height_min,
        height_max=req.height_max,
        area_min=req.area_min,
        area_max=req.area_max,
        include_rejected=req.include_rejected,
        dry_run=req.dry_run,
    )
    return {"status": "ok", "stats": stats}


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
    prompt_id: Optional[str] = None
    prompt_mode: str = "preset"
    custom_prompt: Optional[str] = None


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
    prompt_id = (req.prompt_id or "").strip()

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

    selected_prompt_id: Optional[str] = None

    if prompt_id:
        selected_prompt_id = prompt_id
        prompt_text = _get_touchup_prompt_text(prompt_id)
        if not prompt_text:
            raise HTTPException(
                status_code=404,
                detail=f"Touchup prompt '{prompt_id}' not found in prompts/touchup_prompts.json",
            )
        user_prompt = prompt_text
    elif prompt_mode == "custom":
        if not custom_prompt:
            raise HTTPException(
                status_code=400, detail="custom_prompt required for custom mode"
            )
        user_prompt = custom_prompt
    else:
        mode_to_prompt_id = {
            "preset": "improve_blending",
            "improve_blending": "improve_blending",
            "fix_direction": "fix_direction",
            "fix_lighting": "fix_lighting",
        }
        selected_prompt_id = mode_to_prompt_id.get(prompt_mode, prompt_mode)
        prompt_text = _get_touchup_prompt_text(selected_prompt_id)
        if not prompt_text:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unknown prompt selection. Provide prompt_id from /api/touchup-prompts "
                    "or prompt_mode='custom' with custom_prompt."
                ),
            )
        user_prompt = prompt_text

    final_prompt = user_prompt

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
        "prompt_id_used": selected_prompt_id,
        "prompt_used": final_prompt,
    }
