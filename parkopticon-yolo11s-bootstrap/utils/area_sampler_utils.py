import csv
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

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

ENRICHMENT_COLUMNS = [
    "source",
    "sample_location",
    "pano_id",
    "pano_lat",
    "pano_lng",
    "pano_date",
    "status",
    "group_id",
    "created_at",
]


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def metadata_call(
    location: str,
    api_key: str,
    radius: int = 50,
    retries: int = 4,
    backoff: float = 1.0,
    timeout: float = 10.0,
) -> dict[str, Any]:
    params = {"location": location, "radius": radius, "key": api_key}
    last_error = ""

    for attempt in range(retries):
        try:
            response = requests.get(GSV_METADATA_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            api_status = str(payload.get("status", "UNKNOWN"))

            if api_status == "OK":
                location_obj = payload.get("location") or {}
                return {
                    "status": "ok",
                    "api_status": api_status,
                    "pano_id": str(payload.get("pano_id") or ""),
                    "pano_lat": _to_float(location_obj.get("lat")),
                    "pano_lng": _to_float(location_obj.get("lng")),
                    "pano_date": str(payload.get("date") or ""),
                    "error": "",
                }

            return {
                "status": "no_imagery",
                "api_status": api_status,
                "pano_id": "",
                "pano_lat": None,
                "pano_lng": None,
                "pano_date": "",
                "error": str(payload.get("error_message") or ""),
            }
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))

    return {
        "status": "failed",
        "api_status": "REQUEST_FAILED",
        "pano_id": "",
        "pano_lat": None,
        "pano_lng": None,
        "pano_date": "",
        "error": last_error or "metadata request failed",
    }


def point_in_polygon(lat: float, lng: float, polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        if ((lng_i > lng) != (lng_j > lng)) and (lng_j != lng_i):
            boundary_lat = (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i) + lat_i
            if lat < boundary_lat:
                inside = not inside
        j = i
    return inside


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_m = 6371000.0
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)

    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(delta_lng / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


def read_points_preserve_columns(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not csv_path.exists():
        return [], []

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return rows, columns


def append_rows_safely(
    csv_path: Path,
    new_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]] | None = None,
    existing_columns: list[str] | None = None,
) -> list[str]:
    if existing_rows is None or existing_columns is None:
        existing_rows, existing_columns = read_points_preserve_columns(csv_path)

    original_columns = list(existing_columns or [])
    columns = list(original_columns)
    if not columns:
        columns = list(BASE_COLUMNS)
    else:
        for col in BASE_COLUMNS:
            if col not in columns:
                columns.append(col)

    for col in ENRICHMENT_COLUMNS:
        if col not in columns:
            columns.append(col)

    if not new_rows and columns == original_columns:
        return columns

    merged_rows = list(existing_rows) + list(new_rows)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{csv_path.name}.", suffix=".tmp", dir=str(csv_path.parent))
    os.close(fd)

    try:
        with open(tmp_name, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged_rows)
        os.replace(tmp_name, csv_path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    return columns
