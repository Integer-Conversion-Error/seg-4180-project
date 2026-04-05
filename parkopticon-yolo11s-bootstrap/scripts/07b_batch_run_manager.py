#!/usr/bin/env python3
"""
Run multiple split+train jobs sequentially from a JSON plan.

Each job gets:
- its own split directory
- its own dataset YAML pointing at that split
- a timestamped training run name to avoid collisions
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_MODEL_SIZES = {"n", "s", "m", "l", "x"}

RESERVED_SPLIT_FLAGS = {
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

RESERVED_TRAIN_FLAGS = {
    "--data",
    "--model",
    "--project",
    "--epochs",
    "--save-period",
    "--imgsz",
    "--batch",
    "--device",
    "--name",
    "--manifest",
    "--registry-path",
    "--summary-path",
    "--resume",
    "--include-rejected",
    "--count-lookalike-as-vehicle",
    "--metadata-path",
    "-d",
    "-m",
    "-p",
    "-e",
    "-s",
    "-b",
    "-n",
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    slug = slug.strip("_-")
    return slug or "job"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("extra_args must be a list of strings or a string")


def _resolve_path(
    value: str,
    run_dir: Path,
    repo_root: Path,
    *,
    prefer_run_dir: bool,
    must_exist: bool,
) -> Path:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Path value cannot be empty")

    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        base = run_dir if prefer_run_dir else repo_root
        resolved = (base / candidate).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required path not found: {resolved}")
    return resolved


def _assert_no_reserved_flags(
    extra_args: list[str], reserved_flags: set[str], section_name: str
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
        conflict_list = ", ".join(sorted(conflicts))
        raise ValueError(
            f"{section_name} extra_args cannot override reserved flags: {conflict_list}"
        )


def _resolve_model_value(model_value: str, run_dir: Path, repo_root: Path) -> str:
    model_text = (model_value or "").strip()
    if not model_text:
        return model_text

    model_path = Path(model_text)
    if model_path.is_absolute() and model_path.exists():
        return str(model_path.resolve())

    if not model_path.is_absolute():
        run_candidate = (run_dir / model_path).resolve()
        if run_candidate.exists():
            return str(run_candidate)
        repo_candidate = (repo_root / model_path).resolve()
        if repo_candidate.exists():
            return str(repo_candidate)

    return model_text


def _select_model(train_cfg: dict[str, Any], run_dir: Path, repo_root: Path) -> str:
    explicit_model = str(train_cfg.get("model") or "").strip()
    if explicit_model:
        return _resolve_model_value(explicit_model, run_dir, repo_root)

    size = str(train_cfg.get("model_size") or "").strip().lower()
    if size:
        if size not in SUPPORTED_MODEL_SIZES:
            valid = ", ".join(sorted(SUPPORTED_MODEL_SIZES))
            raise ValueError(
                f"Unsupported model_size '{size}'. Expected one of: {valid}"
            )
        return f"yolo11{size}.pt"

    return "yolo11s.pt"


def _write_dataset_yaml_from_template(template_path: Path, output_path: Path) -> None:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    rewritten: list[str] = []
    seen = {"path": False, "train": False, "val": False, "test": False}
    split_root = output_path.parent.resolve()
    split_root_yaml = split_root.as_posix()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("path:"):
            rewritten.append(f'path: "{split_root_yaml}"')
            seen["path"] = True
            continue
        if stripped.startswith("train:"):
            rewritten.append("train: train/images")
            seen["train"] = True
            continue
        if stripped.startswith("val:"):
            rewritten.append("val: val/images")
            seen["val"] = True
            continue
        if stripped.startswith("test:"):
            rewritten.append("test: test/images")
            seen["test"] = True
            continue
        rewritten.append(line)

    if not seen["path"]:
        rewritten.append(f'path: "{split_root_yaml}"')
    if not seen["train"]:
        rewritten.append("train: train/images")
    if not seen["val"]:
        rewritten.append("val: val/images")
    if not seen["test"]:
        rewritten.append("test: test/images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rewritten).rstrip() + "\n", encoding="utf-8")


def _render_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run_command(command: list[str], cwd: Path, dry_run: bool) -> int:
    logger.info("Command: %s", _render_command(command))
    if dry_run:
        return 0
    env = os.environ.copy()
    cwd_path = str(cwd.resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [
        entry for entry in existing_pythonpath.split(os.pathsep) if entry
    ]
    if cwd_path not in pythonpath_entries:
        pythonpath_entries.insert(0, cwd_path)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    completed = subprocess.run(command, cwd=str(cwd), env=env)
    return completed.returncode


def _load_plan(plan_path: Path) -> dict[str, Any]:
    with open(plan_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Plan root must be a JSON object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sequential batch manager for split+train jobs"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to batch JSON config file",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override run directory from config",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining jobs if one job fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not execute split/train",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    config_input = Path(args.config)
    if not config_input.is_absolute():
        config_input = (repo_root / config_input).resolve()
    if not config_input.exists():
        logger.error("Config not found: %s", config_input)
        return 1

    try:
        config = _load_plan(config_input)
    except Exception as exc:
        logger.error("Failed to parse config %s: %s", config_input, exc)
        return 1

    run_dir_text = args.run_dir or str(config.get("run_dir") or "").strip()
    if not run_dir_text:
        logger.error("Missing run_dir. Provide --run-dir or set run_dir in config.")
        return 1

    run_dir = Path(run_dir_text)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        logger.error("Run directory does not exist: %s", run_dir)
        return 1

    defaults = config.get("defaults") or {}
    if not isinstance(defaults, dict):
        logger.error("defaults must be a JSON object when provided")
        return 1

    default_split = defaults.get("split") or {}
    default_train = defaults.get("train") or {}
    if not isinstance(default_split, dict) or not isinstance(default_train, dict):
        logger.error("defaults.split and defaults.train must be JSON objects")
        return 1

    jobs = config.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        logger.error("Config must include a non-empty jobs array")
        return 1

    manifest_default = str(config.get("manifest") or "manifests/images.csv")
    labels_dir_default = str(config.get("labels_dir") or "data/labels_autogen")
    labels_final_default = str(config.get("labels_final_dir") or "data/labels_final")
    split_root_default = str(config.get("split_root") or "data/splits/batch")
    train_project_default = str(config.get("train_project") or "runs/detect")
    try:
        registry_path = _resolve_path(
            str(config.get("registry_path") or "reports/training_run_registry.jsonl"),
            run_dir,
            repo_root,
            prefer_run_dir=True,
            must_exist=False,
        )

        reports_root = _resolve_path(
            str(config.get("batch_reports_root") or "reports/batch_runs"),
            run_dir,
            repo_root,
            prefer_run_dir=True,
            must_exist=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Invalid report path configuration: %s", exc)
        return 1
    batch_started = datetime.now()
    batch_id = batch_started.strftime("%Y%m%d_%H%M%S")
    batch_report_dir = reports_root / f"batch_{batch_id}"
    if not args.dry_run:
        batch_report_dir.mkdir(parents=True, exist_ok=True)

    configured_template = str(config.get("dataset_template") or "").strip()
    if configured_template:
        try:
            dataset_template = _resolve_path(
                configured_template,
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Invalid dataset_template path: %s", exc)
            return 1
    else:
        run_template = run_dir / "data" / "splits" / "data.yaml"
        repo_template = repo_root / "dataset.yaml"
        if run_template.exists():
            dataset_template = run_template
        elif repo_template.exists():
            dataset_template = repo_template
        else:
            logger.error(
                "No dataset template found. Set dataset_template in config explicitly."
            )
            return 1

    continue_on_error = args.continue_on_error or _truthy(
        config.get("continue_on_error")
    )

    split_script = (repo_root / "scripts" / "06_split_dataset.py").resolve()
    train_script = (repo_root / "scripts" / "07_train_yolo.py").resolve()
    if not split_script.exists() or not train_script.exists():
        logger.error("Expected split/train scripts are missing under scripts/")
        return 1

    logger.info("Batch run manager started")
    logger.info("Config: %s", config_input)
    logger.info("Run dir: %s", run_dir)
    logger.info("Jobs: %d", len(jobs))
    logger.info("Dataset template: %s", dataset_template)

    results: list[dict[str, Any]] = []
    all_success = True

    for index, raw_job in enumerate(jobs, start=1):
        if not isinstance(raw_job, dict):
            logger.error("Job %d is not a JSON object", index)
            all_success = False
            if not continue_on_error:
                break
            continue

        job_name = str(raw_job.get("name") or f"job_{index:02d}")
        job_slug = _slugify(job_name)
        job_split_cfg = dict(default_split)
        job_split_cfg.update(raw_job.get("split") or {})
        job_train_cfg = dict(default_train)
        job_train_cfg.update(raw_job.get("train") or {})

        job_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_id = f"{index:02d}_{job_slug}_{job_stamp}"
        split_out_dir = (
            _resolve_path(
                str(job_split_cfg.get("split_root") or split_root_default),
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=False,
            )
            / job_id
        )
        data_yaml_path = split_out_dir / "data.yaml"
        report_dir = batch_report_dir / job_id
        summary_path = report_dir / "train_summary.md"

        try:
            manifest_path = _resolve_path(
                str(job_split_cfg.get("manifest") or manifest_default),
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=True,
            )
            labels_dir = _resolve_path(
                str(job_split_cfg.get("labels_dir") or labels_dir_default),
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=False,
            )
            labels_final_dir = _resolve_path(
                str(job_split_cfg.get("labels_final_dir") or labels_final_default),
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=False,
            )

            train_project = _resolve_path(
                str(job_train_cfg.get("project") or train_project_default),
                run_dir,
                repo_root,
                prefer_run_dir=True,
                must_exist=False,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Invalid path configuration for job '%s': %s", job_name, exc)
            all_success = False
            results.append(
                {
                    "job_name": job_name,
                    "job_id": job_id,
                    "status": "config_error",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break
            continue

        train_name_base = _slugify(str(job_train_cfg.get("name") or job_name))
        train_name = f"{train_name_base}_{job_stamp}"

        try:
            model_value = _select_model(job_train_cfg, run_dir, repo_root)
        except ValueError as exc:
            logger.error("%s", exc)
            all_success = False
            results.append(
                {
                    "job_name": job_name,
                    "job_id": job_id,
                    "status": "config_error",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break
            continue

        split_cmd = [
            sys.executable,
            "-u",
            str(split_script),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(split_out_dir),
            "--labels-dir",
            str(labels_dir),
            "--labels-final-dir",
            str(labels_final_dir),
        ]

        split_number_args = {
            "train_ratio": "--train-ratio",
            "val_ratio": "--val-ratio",
            "test_ratio": "--test-ratio",
            "seed": "--seed",
            "min_enforcement_val": "--min-enforcement-val",
            "min_enforcement_test": "--min-enforcement-test",
        }
        for key, flag in split_number_args.items():
            if key in job_split_cfg and job_split_cfg[key] is not None:
                split_cmd.extend([flag, str(job_split_cfg[key])])

        if _truthy(job_split_cfg.get("skip_threshold_check")):
            split_cmd.append("--skip-threshold-check")
        if _truthy(job_split_cfg.get("include_rejected")):
            split_cmd.append("--include-rejected")

        try:
            split_extra_args = _coerce_extra_args(job_split_cfg.get("extra_args"))
            _assert_no_reserved_flags(split_extra_args, RESERVED_SPLIT_FLAGS, "split")
            split_cmd.extend(split_extra_args)
        except ValueError as exc:
            logger.error("Invalid split extra_args for job %s: %s", job_name, exc)
            all_success = False
            if not continue_on_error:
                break
            continue

        logger.info("[%d/%d] Split job '%s'", index, len(jobs), job_name)
        split_rc = _run_command(split_cmd, repo_root, args.dry_run)
        if split_rc != 0:
            all_success = False
            logger.error("Split failed for job '%s' (exit code %d)", job_name, split_rc)
            results.append(
                {
                    "job_name": job_name,
                    "job_id": job_id,
                    "status": "split_failed",
                    "split_returncode": split_rc,
                    "split_out_dir": str(split_out_dir),
                }
            )
            if not continue_on_error:
                break
            continue

        if not args.dry_run:
            _write_dataset_yaml_from_template(dataset_template, data_yaml_path)

        epochs = int(job_train_cfg.get("epochs", 300))
        save_period = int(job_train_cfg.get("save_period", 10))
        imgsz = int(job_train_cfg.get("imgsz", 640))
        device = str(job_train_cfg.get("device", "0"))
        batch_value = job_train_cfg.get("batch", None)

        if not args.dry_run:
            report_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "-u",
            str(train_script),
            "--data",
            str(data_yaml_path),
            "--model",
            model_value,
            "--project",
            str(train_project),
            "--epochs",
            str(epochs),
            "--save-period",
            str(save_period),
            "--imgsz",
            str(imgsz),
            "--device",
            device,
            "--name",
            train_name,
            "--manifest",
            str(manifest_path),
            "--registry-path",
            str(registry_path),
            "--summary-path",
            str(summary_path),
        ]

        if batch_value is not None:
            train_cmd.extend(["--batch", str(batch_value)])
        if _truthy(job_train_cfg.get("resume")):
            train_cmd.append("--resume")
        if _truthy(job_train_cfg.get("include_rejected")):
            train_cmd.append("--include-rejected")
        if _truthy(job_train_cfg.get("count_lookalike_as_vehicle")):
            train_cmd.append("--count-lookalike-as-vehicle")

        metadata_path = str(job_train_cfg.get("metadata_path") or "").strip()
        if metadata_path:
            try:
                resolved_metadata = _resolve_path(
                    metadata_path,
                    run_dir,
                    repo_root,
                    prefer_run_dir=True,
                    must_exist=False,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Invalid metadata_path for job %s: %s", job_name, exc)
                all_success = False
                if not continue_on_error:
                    break
                continue
            train_cmd.extend(["--metadata-path", str(resolved_metadata)])

        try:
            train_extra_args = _coerce_extra_args(job_train_cfg.get("extra_args"))
            _assert_no_reserved_flags(train_extra_args, RESERVED_TRAIN_FLAGS, "train")
            train_cmd.extend(train_extra_args)
        except ValueError as exc:
            logger.error("Invalid train extra_args for job %s: %s", job_name, exc)
            all_success = False
            if not continue_on_error:
                break
            continue

        logger.info("[%d/%d] Train job '%s'", index, len(jobs), job_name)
        train_rc = _run_command(train_cmd, repo_root, args.dry_run)
        status = "ok" if train_rc == 0 else "train_failed"
        if train_rc != 0:
            all_success = False
            logger.error(
                "Training failed for job '%s' (exit code %d)", job_name, train_rc
            )

        result = {
            "job_name": job_name,
            "job_id": job_id,
            "status": status,
            "split_out_dir": str(split_out_dir),
            "data_yaml": str(data_yaml_path),
            "train_project": str(train_project),
            "train_name": train_name,
            "model": model_value,
            "epochs": epochs,
            "save_period": save_period,
            "imgsz": imgsz,
            "device": device,
            "summary_path": str(summary_path),
            "registry_path": str(registry_path),
            "split_command": split_cmd,
            "train_command": train_cmd,
            "split_returncode": split_rc,
            "train_returncode": train_rc,
        }
        if batch_value is not None:
            result["batch"] = batch_value
        results.append(result)

        if train_rc != 0 and not continue_on_error:
            break

    batch_summary = {
        "batch_id": batch_id,
        "started_at": batch_started.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "config": str(config_input),
        "dry_run": args.dry_run,
        "continue_on_error": continue_on_error,
        "all_success": all_success,
        "jobs": results,
    }

    if not args.dry_run:
        batch_summary_path = batch_report_dir / "batch_summary.json"
        with open(batch_summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2)
        logger.info("Batch summary saved to %s", batch_summary_path)

    success_count = sum(1 for item in results if item.get("status") == "ok")
    logger.info(
        "Batch finished: %d/%d successful jobs",
        success_count,
        len(results),
    )

    return 0 if all_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
