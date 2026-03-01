#!/usr/bin/env python3

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UTILS_DIR = PROJECT_ROOT / "utils"
UTILS_FILE = UTILS_DIR / "area_sampler_utils.py"
UTILS_SPEC = importlib.util.spec_from_file_location("area_sampler_utils", UTILS_FILE)
if UTILS_SPEC is None or UTILS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load utility module at {UTILS_FILE}")
UTILS_MODULE = importlib.util.module_from_spec(UTILS_SPEC)
UTILS_SPEC.loader.exec_module(UTILS_MODULE)

append_rows_safely = UTILS_MODULE.append_rows_safely
haversine_m = UTILS_MODULE.haversine_m
metadata_call = UTILS_MODULE.metadata_call
point_in_polygon = UTILS_MODULE.point_in_polygon
read_points_preserve_columns = UTILS_MODULE.read_points_preserve_columns

LOGGER = logging.getLogger(__name__)

BASE_COLUMNS = [
    "street",
    "label",
    "location",
    "heading",
    "pitch",
    "fov",
    "radius",
    "bottom_crop",
]


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bbox must be min_lat,min_lng,max_lat,max_lng")
    try:
        min_lat, min_lng, max_lat, max_lng = [float(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--bbox values must be numeric") from exc
    if min_lat >= max_lat or min_lng >= max_lng:
        raise argparse.ArgumentTypeError("--bbox min values must be lower than max values")
    return min_lat, min_lng, max_lat, max_lng


def parse_headings(value: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            heading = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--headings must be a comma-separated int list") from exc
        if heading < 0 or heading > 360:
            raise argparse.ArgumentTypeError("--headings values must be in range 0..360")
        if heading not in seen:
            seen.add(heading)
            result.append(heading)
    if not result:
        raise argparse.ArgumentTypeError("--headings must contain at least one heading")
    return result


def load_polygon(path: Path) -> list[list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
        if not features:
            raise ValueError("polygon GeoJSON has no features")
        payload = features[0]

    if payload.get("type") == "Feature":
        geometry = payload.get("geometry") or {}
    else:
        geometry = payload

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")

    if geometry_type == "Polygon":
        ring = (coordinates or [[]])[0]
    elif geometry_type == "MultiPolygon":
        ring = ((coordinates or [[[]]])[0] or [[]])[0]
    else:
        raise ValueError("polygon GeoJSON must be Polygon or MultiPolygon")

    if len(ring) < 3:
        raise ValueError("polygon ring must contain at least 3 points")

    first = ring[0]
    if not isinstance(first, list) or len(first) < 2:
        raise ValueError("polygon coordinates are malformed")

    lat_lng: list[list[float]] = []
    first_a = float(first[0])
    first_b = float(first[1])
    treat_as_lat_lng = abs(first_a) <= 90 and abs(first_b) > 90

    for vertex in ring:
        a = float(vertex[0])
        b = float(vertex[1])
        if treat_as_lat_lng:
            lat_lng.append([a, b])
        else:
            lat_lng.append([b, a])

    return lat_lng


def get_bottom_crop_default(rows: list[dict[str, str]]) -> str:
    values: list[str] = []
    for row in rows:
        raw = (row.get("bottom_crop") or "").strip()
        if not raw:
            continue
        try:
            float(raw)
        except ValueError:
            continue
        values.append(raw)
    if not values:
        return "0"
    return Counter(values).most_common(1)[0][0]


def format_number(value: float | int) -> str:
    text = f"{value:.8f}" if isinstance(value, float) else str(value)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def format_lat_lng(lat: float, lng: float) -> str:
    return f"{lat:.7f},{lng:.7f}"


def build_street_id_allocator(existing_rows: list[dict[str, str]]) -> Callable[[], str]:
    taken = {str(row.get("street") or "").strip() for row in existing_rows if row.get("street")}
    next_num = 1
    for value in taken:
        match = re.fullmatch(r"auto_(\d+)", value)
        if match:
            next_num = max(next_num, int(match.group(1)) + 1)

    def allocate() -> str:
        nonlocal next_num
        while True:
            candidate = f"auto_{next_num}"
            next_num += 1
            if candidate not in taken:
                taken.add(candidate)
                return candidate

    return allocate


def inside_bbox(lat: float, lng: float, bbox: tuple[float, float, float, float]) -> bool:
    min_lat, min_lng, max_lat, max_lng = bbox
    return min_lat <= lat <= max_lat and min_lng <= lng <= max_lng


def sample_point(
    rng: random.Random,
    bounds: tuple[float, float, float, float],
    polygon: list[list[float]] | None,
) -> tuple[float, float] | None:
    min_lat, min_lng, max_lat, max_lng = bounds
    if polygon is None:
        return rng.uniform(min_lat, max_lat), rng.uniform(min_lng, max_lng)

    for _ in range(200):
        lat = rng.uniform(min_lat, max_lat)
        lng = rng.uniform(min_lng, max_lng)
        if point_in_polygon(lat, lng, polygon):
            return lat, lng
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Append-only Street View area sampler")
    parser.add_argument("--points_csv", default="manifests/points.csv")
    parser.add_argument("--bbox", default=None)
    parser.add_argument("--polygon_geojson", default=None)
    parser.add_argument("--target_panos", type=int, default=200)
    parser.add_argument("--radius", type=int, default=50)
    parser.add_argument("--min_spacing_m", type=float, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_attempts", type=int, default=10000)
    parser.add_argument("--headings", default="0,45,90,135,180,225,270,315")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--fov", type=int, default=80)
    parser.add_argument("--bottom_crop", type=float, default=None)
    parser.add_argument("--label_prefix", default="auto")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--api_key_env_var", default="GSV_API_KEY")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if bool(args.bbox) == bool(args.polygon_geojson):
        parser.error("Exactly one of --bbox or --polygon_geojson is required")

    headings = parse_headings(args.headings)
    if args.pitch < -90 or args.pitch > 90:
        parser.error("--pitch must be in range -90..90")
    if args.fov < 30 or args.fov > 120:
        parser.error("--fov must be in range 30..120")
    if args.radius <= 0:
        parser.error("--radius must be > 0")
    if args.target_panos < 0:
        parser.error("--target_panos must be >= 0")
    if args.max_attempts < 0:
        parser.error("--max_attempts must be >= 0")
    if args.min_spacing_m is not None and args.min_spacing_m <= 0:
        parser.error("--min_spacing_m must be > 0 when provided")

    api_key = args.api_key or os.getenv(args.api_key_env_var)
    if not api_key:
        parser.error(f"Missing API key. Set {args.api_key_env_var} or pass --api_key")

    points_path = Path(args.points_csv)
    existing_rows, existing_columns = read_points_preserve_columns(points_path)
    if not existing_columns:
        existing_columns = list(BASE_COLUMNS)

    bottom_crop = (
        format_number(args.bottom_crop)
        if args.bottom_crop is not None
        else get_bottom_crop_default(existing_rows)
    )

    polygon: list[list[float]] | None = None
    if args.polygon_geojson:
        polygon = load_polygon(Path(args.polygon_geojson))
        lat_values = [p[0] for p in polygon]
        lng_values = [p[1] for p in polygon]
        bounds = (min(lat_values), min(lng_values), max(lat_values), max(lng_values))
    else:
        bounds = parse_bbox(args.bbox)

    existing_pano_ids = {
        str(row.get("pano_id") or "").strip()
        for row in existing_rows
        if str(row.get("pano_id") or "").strip()
    }

    min_spacing_reference: list[tuple[float, float]] = []
    if args.min_spacing_m is not None:
        seen_panos: set[str] = set()
        for row in existing_rows:
            pano_id = str(row.get("pano_id") or "").strip()
            if not pano_id or pano_id in seen_panos:
                continue
            try:
                pano_lat = float(str(row.get("pano_lat") or "").strip())
                pano_lng = float(str(row.get("pano_lng") or "").strip())
            except ValueError:
                continue
            min_spacing_reference.append((pano_lat, pano_lng))
            seen_panos.add(pano_id)

    alloc_street_id = build_street_id_allocator(existing_rows)
    rng = random.Random(args.seed)

    accepted_panos: set[str] = set()
    new_rows: list[dict[str, str]] = []

    stats = {
        "attempted": 0,
        "accepted_panos": 0,
        "duplicates": 0,
        "no_imagery": 0,
        "out_of_area": 0,
        "too_close": 0,
        "failed": 0,
    }

    while stats["accepted_panos"] < args.target_panos and stats["attempted"] < args.max_attempts:
        stats["attempted"] += 1
        sampled = sample_point(rng, bounds, polygon)
        if sampled is None:
            stats["failed"] += 1
            continue

        sample_lat, sample_lng = sampled
        sample_location = format_lat_lng(sample_lat, sample_lng)
        metadata = metadata_call(sample_location, api_key=api_key, radius=args.radius)

        if metadata["status"] == "failed":
            stats["failed"] += 1
            continue

        if metadata["status"] != "ok":
            stats["no_imagery"] += 1
            continue

        pano_id = str(metadata.get("pano_id") or "").strip()
        pano_lat = metadata.get("pano_lat")
        pano_lng = metadata.get("pano_lng")

        if not pano_id or pano_lat is None or pano_lng is None:
            stats["failed"] += 1
            continue

        if pano_id in existing_pano_ids or pano_id in accepted_panos:
            stats["duplicates"] += 1
            continue

        if polygon is not None:
            if not point_in_polygon(float(pano_lat), float(pano_lng), polygon):
                stats["out_of_area"] += 1
                continue
        else:
            if not inside_bbox(float(pano_lat), float(pano_lng), bounds):
                stats["out_of_area"] += 1
                continue

        if args.min_spacing_m is not None:
            too_close = False
            for ref_lat, ref_lng in min_spacing_reference:
                if haversine_m(float(pano_lat), float(pano_lng), ref_lat, ref_lng) < args.min_spacing_m:
                    too_close = True
                    break
            if too_close:
                stats["too_close"] += 1
                continue

        accepted_panos.add(pano_id)
        existing_pano_ids.add(pano_id)
        min_spacing_reference.append((float(pano_lat), float(pano_lng)))
        stats["accepted_panos"] += 1

        created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        safe_pano = re.sub(r"[^A-Za-z0-9_-]", "_", pano_id)
        snapped_location = format_lat_lng(float(pano_lat), float(pano_lng))
        pano_date = str(metadata.get("pano_date") or "")

        for heading in headings:
            street_id = alloc_street_id()
            new_rows.append(
                {
                    "street": street_id,
                    "label": f"{args.label_prefix}_{safe_pano}_h{heading}",
                    "location": snapped_location,
                    "heading": str(heading),
                    "pitch": str(args.pitch),
                    "fov": str(args.fov),
                    "radius": str(args.radius),
                    "bottom_crop": bottom_crop,
                    "source": "auto_sampler",
                    "sample_location": sample_location,
                    "pano_id": pano_id,
                    "pano_lat": format_number(float(pano_lat)),
                    "pano_lng": format_number(float(pano_lng)),
                    "pano_date": pano_date,
                    "status": "ok",
                    "group_id": pano_id,
                    "created_at": created_at,
                }
            )

    LOGGER.info("attempted=%s", stats["attempted"])
    LOGGER.info("accepted_panos=%s", stats["accepted_panos"])
    LOGGER.info("duplicates=%s", stats["duplicates"])
    LOGGER.info("no_imagery=%s", stats["no_imagery"])
    LOGGER.info("out_of_area=%s", stats["out_of_area"])
    LOGGER.info("too_close=%s", stats["too_close"])
    LOGGER.info("failed=%s", stats["failed"])
    LOGGER.info("new_rows=%s", len(new_rows))

    if args.dry_run:
        LOGGER.info("dry_run=true, skipping file write")
        return 0

    final_columns = append_rows_safely(
        points_path,
        new_rows,
        existing_rows=existing_rows,
        existing_columns=existing_columns,
    )
    if new_rows:
        LOGGER.info("appended_rows=%s", len(new_rows))
    else:
        LOGGER.info("no rows appended")
    if final_columns != existing_columns:
        LOGGER.info("ensured_enrichment_columns=true")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
