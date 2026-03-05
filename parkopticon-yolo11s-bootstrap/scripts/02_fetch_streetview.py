#!/usr/bin/env python3
"""
Fetch Street View images and metadata for points in points.csv.
Downloads images to data/images_original/{street}/{image_id}.jpg
"""

import argparse
import csv
import hashlib
import json
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
GSV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"


def generate_image_id(location: str, heading: int, pitch: int, fov: int) -> str:
    content = f"{location}_{heading}_{pitch}_{fov}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def get_street_view_metadata(
    location: str,
    heading: int,
    pitch: int,
    fov: int,
    radius: int,
    api_key: str,
    retries: int = 3,
    backoff: float = 1.0
) -> Optional[dict]:
    params = {
        "location": location,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "radius": radius,
        "key": api_key
    }
    
    for attempt in range(retries):
        try:
            resp = requests.get(GSV_METADATA_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("status") == "OK":
                return {
                    "pano_id": data.get("pano_id"),
                    "pano_lat": data.get("location", {}).get("lat"),
                    "pano_lng": data.get("location", {}).get("lng"),
                    "date": data.get("date"),
                    "status": "ok"
                }
            else:
                return {"status": data.get("status", "unknown"), "error": data.get("error_message", "")}
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    
    return {"status": "failed", "error": "All retries exhausted"}


def download_street_view_image(
    location: str,
    heading: int,
    pitch: int,
    fov: int,
    radius: int,
    api_key: str,
    output_path: Path,
    retries: int = 3,
    backoff: float = 1.0
) -> bool:
    params = {
        "location": location,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "radius": radius,
        "key": api_key,
        "return_error_code": "true",
        "size": "640x640",
        "format": "jpg"
    }
    
    for attempt in range(retries):
        try:
            resp = requests.get(GSV_IMAGE_URL, params=params, timeout=30)
            resp.raise_for_status()
            
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type:
                error_msg = resp.text[:500]
                logger.warning(f"Non-image response: {content_type} - {error_msg}")
                return False
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(resp.content)
            return True
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff * (2 ** attempt))
    
    return False


def load_or_create_manifest(manifest_path: Path) -> list:
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    return []


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    
    fieldnames = list(manifest[0].keys())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def main():
    parser = argparse.ArgumentParser(description="Fetch Street View images")
    parser.add_argument("--points", "-p", default="manifests/points.csv", help="Input points CSV")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Output manifest")
    parser.add_argument("--out-dir", "-o", default="data/images_original", help="Output directory")
    parser.add_argument("--api-key", "-k", default=None, help="GSV API key (or env GSV_API_KEY)")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("GSV_API_KEY")
    if not api_key:
        logger.error("No API key provided. Set GSV_API_KEY or use --api-key")
        return
    
    points_path = Path(args.points)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    
    if not points_path.exists():
        logger.error(f"Points file not found: {points_path}")
        return
    
    with open(points_path, 'r') as f:
        points = list(csv.DictReader(f))
    
    manifest = load_or_create_manifest(manifest_path) if args.resume else []
    existing_ids = {row['image_id'] for row in manifest if row.get('image_id')}
    
    for row in tqdm(points, desc="Fetching Street View"):
        street = row['label'] or row['street']
        location = row['location']
        heading = int(row.get('heading', 0))
        pitch = int(row.get('pitch', 0))
        fov = int(row.get('fov', 80))
        radius = int(row.get('radius', 50))
        
        image_id = generate_image_id(location, heading, pitch, fov)
        
        if image_id in existing_ids:
            logger.debug(f"Skipping {image_id} (already in manifest)")
            continue
        
        metadata = get_street_view_metadata(
            location, heading, pitch, fov, radius, api_key
        )
        
        if metadata.get('status') != 'ok':
            manifest.append({
                'image_id': image_id,
                'file_path': '',
                'split': 'unset',
                'parent_image_id': '',
                'is_synthetic': '0',
                'edit_type': 'none',
                'expected_inserted_class': 'none',
                'street': street,
                'input_location': location,
                'heading': str(heading),
                'pitch': str(pitch),
                'fov': str(fov),
                'pano_id': metadata.get('pano_id', ''),
                'pano_lat': str(metadata.get('pano_lat', '')),
                'pano_lng': str(metadata.get('pano_lng', '')),
                'status': metadata.get('status'),
                'num_boxes_autogen': '0',
                'needs_review': '0',
                'review_status': 'todo',
                'created_at': datetime.now().isoformat()
            })
            save_manifest(manifest, manifest_path)
            continue
        
        safe_street = street.replace('/', '_').replace(' ', '_')
        output_path = out_dir / safe_street / f"{image_id}.jpg"
        
        success = download_street_view_image(
            location, heading, pitch, fov, radius, api_key, output_path
        )
        
        manifest.append({
            'image_id': image_id,
            'file_path': str(output_path),
            'split': 'unset',
            'parent_image_id': '',
            'is_synthetic': '0',
            'edit_type': 'none',
            'expected_inserted_class': 'none',
            'street': street,
            'input_location': location,
            'heading': str(heading),
            'pitch': str(pitch),
            'fov': str(fov),
            'pano_id': metadata.get('pano_id', ''),
            'pano_lat': str(metadata.get('pano_lat', '')),
            'pano_lng': str(metadata.get('pano_lng', '')),
            'status': 'ok' if success else 'failed',
            'num_boxes_autogen': '0',
            'needs_review': '0',
            'review_status': 'todo',
            'created_at': datetime.now().isoformat()
        })
        
        save_manifest(manifest, manifest_path)
        
        if success:
            logger.info(f"Downloaded: {output_path}")
        else:
            logger.warning(f"Failed to download: {location} h={heading}")
    
    logger.info(f"Manifest saved to {manifest_path} ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
