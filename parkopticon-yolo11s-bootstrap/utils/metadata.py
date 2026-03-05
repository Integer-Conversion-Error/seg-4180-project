"""
Metadata capture and saving utilities for reproducibility tracking.

Captures:
- Git commit hash and branch info
- Manifest hash (MD5 of images.csv)
- Python version
- Package versions (ultralytics, opencv, numpy, etc.)
- Random seeds
- Timestamp
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _get_git_info() -> Dict[str, Any]:
    """Get Git commit hash, branch, and status."""
    git_info = {
        "commit": None,
        "branch": None,
        "status": "unavailable",
        "available": False,
    }

    try:
        # Get commit hash
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=os.getcwd()
            )
            .decode()
            .strip()
        )
        git_info["commit"] = commit

        # Get branch name
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )
            .decode()
            .strip()
        )
        git_info["branch"] = branch

        # Check git status (clean/dirty)
        status_output = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )
            .decode()
            .strip()
        )

        git_info["status"] = "clean" if not status_output else "dirty"
        git_info["available"] = True

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Git info unavailable: {e}. Continuing without git metadata.")
        git_info["status"] = "unavailable"

    return git_info


def _get_manifest_hash(manifest_path: str = "manifests/images.csv") -> Optional[str]:
    """Compute MD5 hash of the manifest file."""
    try:
        path = Path(manifest_path)
        if not path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return None

        md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    except Exception as e:
        logger.warning(f"Could not compute manifest hash: {e}")
        return None


def _get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = {
        "python": platform.python_version(),
    }

    key_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "torch",
        "torchvision",
        "pillow",
        "pyyaml",
    ]

    for pkg in key_packages:
        try:
            module = __import__(pkg.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")
            packages[pkg] = version
        except ImportError:
            packages[pkg] = "not_installed"
        except Exception as e:
            packages[pkg] = f"error: {e}"

    return packages


def save_run_metadata(
    output_path: str,
    random_seeds: Optional[Dict[str, int]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    manifest_path: str = "manifests/images.csv",
) -> bool:
    """
    Save comprehensive run metadata to a JSON file.

    Args:
        output_path (str): Path where metadata.json will be saved
        random_seeds (dict, optional): Dict of random seed names -> values
            Example: {"numpy": 42, "torch": 42, "random": 42}
        additional_metadata (dict, optional): Any extra metadata to include
        manifest_path (str): Path to manifest file for hash computation

    Returns:
        bool: True if metadata saved successfully, False otherwise

    Example:
        >>> save_run_metadata(
        ...     "runs/detect/my_run/metadata.json",
        ...     random_seeds={"numpy": 42, "torch": 42},
        ...     additional_metadata={"epochs": 50, "batch_size": 16}
        ... )
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build metadata dictionary
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "git": _get_git_info(),
            "manifest_hash": _get_manifest_hash(manifest_path),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "working_directory": os.getcwd(),
            },
            "packages": _get_package_versions(),
        }

        # Add random seeds if provided
        if random_seeds:
            metadata["random_seeds"] = random_seeds

        # Add any additional metadata
        if additional_metadata:
            metadata["additional"] = additional_metadata

        # Write to JSON
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save metadata to {output_path}: {e}")
        return False


def load_run_metadata(metadata_path: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a JSON file.

    Args:
        metadata_path (str): Path to metadata.json file

    Returns:
        dict or None: Loaded metadata dict, or None if file not found/readable
    """
    try:
        path = Path(metadata_path)
        if not path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        with open(path, "r") as f:
            return json.load(f)

    except Exception as e:
        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return None


def append_training_run_record(
    registry_path: str,
    run_name: str,
    data_yaml: str,
    best_weights_path: str,
    last_weights_path: str,
    metadata_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    manifest_path: str = "manifests/images.csv",
) -> bool:
    try:
        record = {
            "run_id": uuid4().hex,
            "timestamp": datetime.now().isoformat(),
            "run_name": run_name,
            "data_yaml": data_yaml,
            "best_weights_path": best_weights_path,
            "last_weights_path": last_weights_path,
            "metadata_path": metadata_path,
            "git": _get_git_info(),
            "manifest_hash": _get_manifest_hash(manifest_path),
        }
        if extra:
            record["extra"] = extra

        path = Path(registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        logger.info(f"Appended training run record to {registry_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to append training run record: {e}")
        return False
