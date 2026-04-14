from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path


def resolve_exclusion_path(manifest_path: Path) -> Path:
    manifest_path = Path(manifest_path)
    for root in [manifest_path.parent, *manifest_path.parents]:
        candidate = root / "lists" / "excluded_from_synth.txt"
        if candidate.exists():
            return candidate

    # Fallback to the conventional run-root location so callers can still write there.
    run_root = manifest_path.parent.parent
    return run_root / "lists" / "excluded_from_synth.txt"


def load_explicit_excluded_ids(exclusion_path: Path) -> set[str]:
    exclusion_path = Path(exclusion_path)
    if not exclusion_path.exists():
        return set()
    return {
        line.strip()
        for line in exclusion_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _collect_children(manifest_path: Path) -> dict[str, set[str]]:
    children: dict[str, set[str]] = {}
    with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = (row.get("image_id") or "").strip()
            parent_id = (row.get("parent_image_id") or "").strip()
            if image_id and parent_id:
                children.setdefault(parent_id, set()).add(image_id)
    return children


@lru_cache(maxsize=32)
def _closure_cache(manifest_key: str, exclusion_key: str) -> frozenset[str]:
    manifest_path = Path(manifest_key)
    exclusion_path = Path(exclusion_key)
    excluded = set(load_explicit_excluded_ids(exclusion_path))
    if not excluded or not manifest_path.exists():
        return frozenset(excluded)

    children = _collect_children(manifest_path)
    queue = list(excluded)
    while queue:
        current = queue.pop()
        for child in children.get(current, set()):
            if child not in excluded:
                excluded.add(child)
                queue.append(child)
    return frozenset(excluded)


def load_dataset_excluded_ids(manifest_path: Path) -> set[str]:
    manifest_path = Path(manifest_path)
    exclusion_path = resolve_exclusion_path(manifest_path)
    return set(
        _closure_cache(str(manifest_path.resolve()), str(exclusion_path.resolve()))
    )


def clear_dataset_exclusion_cache() -> None:
    _closure_cache.cache_clear()


def is_dataset_excluded(image_id: str, excluded_ids: set[str]) -> bool:
    return (image_id or "").strip() in excluded_ids
