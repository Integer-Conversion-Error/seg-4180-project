#!/usr/bin/env python3
"""Synthetic image cleanup harness.

Browse synthetic images as tiles, select multiple, and remove poor outputs.
Pagination is fixed to 500 images per page by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flask import Flask, abort, jsonify, request, send_file


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SYNTH_ROOT = PROJECT_ROOT / "data" / "images_synth"
TRASH_ROOT = PROJECT_ROOT / "deleted_images" / "synthetic_cleanup"
DELETE_LOG = TRASH_ROOT / "deleted_synthetics.csv"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PAGE_SIZE = 500
MAX_PAGE_SIZE = 500

app = Flask(__name__)


def _normalize_under_root(path: Path, root: Path) -> Path | None:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _collect_synth_images(class_filter: str | None) -> list[Path]:
    if not SYNTH_ROOT.exists():
        return []

    images: list[Path] = []
    for class_dir in sorted(SYNTH_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_filter and class_dir.name != class_filter:
            continue
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                images.append(p)
    return images


def _ensure_delete_log_header() -> None:
    TRASH_ROOT.mkdir(parents=True, exist_ok=True)
    if DELETE_LOG.exists():
        return
    with open(DELETE_LOG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "deleted_at",
                "class_name",
                "source_rel",
                "trash_rel",
            ],
        )
        writer.writeheader()


def _move_to_trash(src: Path) -> Path:
    class_name = src.parent.name
    dest_dir = TRASH_ROOT / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = dest_dir / f"{src.stem}_{ts}{src.suffix}"

    shutil.move(str(src), str(dest))
    return dest


@app.route("/")
def index() -> Any:
    page_path = BASE_DIR / "synth_cleanup.html"
    if not page_path.exists():
        abort(500, description="synth_cleanup.html not found")
    response = send_file(page_path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/files/<path:relative_path>")
def files(relative_path: str) -> Any:
    target = _normalize_under_root(PROJECT_ROOT / relative_path, PROJECT_ROOT)
    if not target or not target.exists() or not target.is_file():
        abort(404)
    response = send_file(target)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


@app.route("/api/classes")
def classes() -> Any:
    if not SYNTH_ROOT.exists():
        return jsonify([])
    names = [d.name for d in sorted(SYNTH_ROOT.iterdir()) if d.is_dir()]
    return jsonify(names)


@app.route("/api/images")
def images() -> Any:
    class_filter = (request.args.get("class") or "").strip() or None
    try:
        page = max(1, int(request.args.get("page", "1")))
    except ValueError:
        page = 1

    try:
        page_size = int(request.args.get("page_size", str(DEFAULT_PAGE_SIZE)))
    except ValueError:
        page_size = DEFAULT_PAGE_SIZE
    page_size = max(1, min(MAX_PAGE_SIZE, page_size))

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
                "class_name": p.parent.name,
                "filename": p.name,
                "relative_path": rel,
                "url": f"/files/{quote(rel)}",
            }
        )

    return jsonify(
        {
            "items": items,
            "paging": {
                "page": page,
                "page_size": page_size,
                "total_items": total,
                "total_pages": total_pages,
            },
            "class_filter": class_filter,
        }
    )


@app.route("/api/delete", methods=["POST"])
def delete_images() -> Any:
    payload = request.get_json(silent=True) or {}
    paths = payload.get("relative_paths") or []
    if not isinstance(paths, list):
        return jsonify({"error": "relative_paths must be an array"}), 400

    cleaned = [str(p).strip() for p in paths if str(p).strip()]
    if not cleaned:
        return jsonify({"error": "No images selected"}), 400

    _ensure_delete_log_header()

    deleted = []
    skipped = []
    now = datetime.now().isoformat(timespec="seconds")

    with open(DELETE_LOG, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["deleted_at", "class_name", "source_rel", "trash_rel"],
        )

        for rel in cleaned:
            src = _normalize_under_root(PROJECT_ROOT / rel, PROJECT_ROOT)
            if not src or not src.exists() or not src.is_file():
                skipped.append({"relative_path": rel, "reason": "not_found"})
                continue
            if _normalize_under_root(src, SYNTH_ROOT) is None:
                skipped.append({"relative_path": rel, "reason": "outside_synth_root"})
                continue

            dest = _move_to_trash(src)
            src_rel = _relative_posix(Path(PROJECT_ROOT / rel), PROJECT_ROOT)
            dest_rel = _relative_posix(dest, PROJECT_ROOT)
            writer.writerow(
                {
                    "deleted_at": now,
                    "class_name": src.parent.name,
                    "source_rel": src_rel,
                    "trash_rel": dest_rel,
                }
            )
            deleted.append({"relative_path": src_rel, "trash_path": dest_rel})

    return jsonify({"deleted": deleted, "skipped": skipped, "count": len(deleted)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic cleanup harness server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5050, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("Synthetic cleanup harness")
    print(f"Serving: {SYNTH_ROOT}")
    print(f"Trash:   {TRASH_ROOT}")
    print(f"Open:    http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
