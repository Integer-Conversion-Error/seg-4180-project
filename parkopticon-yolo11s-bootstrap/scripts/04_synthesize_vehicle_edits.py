#!/usr/bin/env python3
"""
Synthesize vehicle edits using Gemini image editing API.
Creates random_vehicle and enforcement_vehicle variants of empty frames.
"""

import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


RANDOM_VEHICLE_PROMPT = """Change only what is necessary. Keep camera perspective, lighting, shadows, color grading identical. Add exactly ONE vehicle on the roadway (not parked), medium size, clearly visible. Avoid changing signs, buildings, or sky. No global style shifts. The vehicle should be a normal passenger car."""

ENFORCEMENT_VEHICLE_PROMPT = """Change only what is necessary. Keep camera perspective, lighting, shadows, color grading identical. Add exactly ONE parking enforcement vehicle on the roadway (not parked), medium size, clearly visible. The vehicle should have a roof light bar and clear high-contrast 'PARKING ENFORCEMENT' text; generic, fictional markings; no real-world logos. Avoid changing signs, buildings, or sky. No global style shifts."""


def generate_image_id(parent_id: str, edit_type: str) -> str:
    return f"{parent_id}_{edit_type}"


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, 'r') as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def synthesize_image(
    client: genai.Client,
    image_path: Path,
    prompt: str,
    model: str = "gemini-2.0-flash-exp-image-generation",
    output_path: Optional[Path] = None
) -> Optional[bytes]:
    try:
        img = Image.open(image_path)
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_modalities=["image"],
            )
        )
        
        if hasattr(response, 'candidates') and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    return part.inline_data.data
        
        return None
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Synthesize vehicle edits")
    parser.add_argument("--manifest", "-m", default="manifests/images.csv", help="Input manifest")
    parser.add_argument("--empty-list", "-e", default="lists/empty_candidates.txt", help="Empty candidates file")
    parser.add_argument("--out-dir", "-o", default="data/images_synth", help="Output directory")
    parser.add_argument("--api-key", "-k", default=None, help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash-exp-image-generation", help="Gemini model")
    parser.add_argument("--enforcement-rate", type=float, default=0.2, help="Fraction for enforcement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume/--no-resume", default=True, help="Resume from existing")
    args = parser.parse_args()
    
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key. Set GEMINI_API_KEY or use --api-key")
        return
    
    client = genai.Client(api_key=api_key)
    
    manifest_path = Path(args.manifest)
    empty_list_path = Path(args.empty_list)
    out_dir = Path(args.out_dir)
    
    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    if empty_list_path.exists():
        with open(empty_list_path, 'r') as f:
            empty_candidates = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"Empty list not found: {empty_list_path}")
        empty_candidates = []
    
    manifest = load_manifest(manifest_path)
    
    if args.resume:
        existing = {row['image_id']: row for row in manifest if row.get('is_synthetic') == '1'}
        logger.info(f"Resuming with {len(existing)} existing synthetics")
    
    random.seed(args.seed)
    
    enforcement_candidates = random.sample(
        empty_candidates, 
        k=int(len(empty_candidates) * args.enforcement_rate)
    ) if empty_candidates else []
    
    synth_count = 0
    
    for img_id in tqdm(empty_candidates, desc="Synthesizing"):
        parent_row = None
        for row in manifest:
            if row.get('image_id') == img_id:
                parent_row = row
                break
        
        if not parent_row:
            continue
        
        parent_path = Path(parent_row.get('file_path', ''))
        if not parent_path.exists():
            continue
        
        synth_id_random = generate_image_id(img_id, "random_vehicle")
        if args.resume and any(r.get('image_id') == synth_id_random for r in manifest):
            continue
        
        logger.info(f"Creating random_vehicle for {img_id}")
        img_data = synthesize_image(client, parent_path, RANDOM_VEHICLE_PROMPT, args.model)
        
        if img_data:
            out_path = out_dir / "random_vehicle" / f"{synth_id_random}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(img_data)
            
            manifest.append({
                'image_id': synth_id_random,
                'file_path': str(out_path),
                'split': 'unset',
                'parent_image_id': img_id,
                'is_synthetic': '1',
                'edit_type': 'random_vehicle',
                'expected_inserted_class': 'vehicle',
                'street': parent_row.get('street', ''),
                'input_location': parent_row.get('input_location', ''),
                'heading': parent_row.get('heading', ''),
                'pitch': parent_row.get('pitch', ''),
                'fov': parent_row.get('fov', ''),
                'pano_id': parent_row.get('pano_id', ''),
                'pano_lat': parent_row.get('pano_lat', ''),
                'pano_lng': parent_row.get('pano_lng', ''),
                'status': 'ok',
                'num_boxes_autogen': '0',
                'needs_review': '1',
                'review_status': 'todo',
                'created_at': datetime.now().isoformat()
            })
            synth_count += 1
            save_manifest(manifest, manifest_path)
        
        if img_id in enforcement_candidates:
            synth_id_enf = generate_image_id(img_id, "enforcement_vehicle")
            if args.resume and any(r.get('image_id') == synth_id_enf for r in manifest):
                continue
            
            logger.info(f"Creating enforcement_vehicle for {img_id}")
            img_data = synthesize_image(client, parent_path, ENFORCEMENT_VEHICLE_PROMPT, args.model)
            
            if img_data:
                out_path = out_dir / "enforcement_vehicle" / f"{synth_id_enf}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, 'wb') as f:
                    f.write(img_data)
                
                manifest.append({
                    'image_id': synth_id_enf,
                    'file_path': str(out_path),
                    'split': 'unset',
                    'parent_image_id': img_id,
                    'is_synthetic': '1',
                    'edit_type': 'enforcement_vehicle',
                    'expected_inserted_class': 'enforcement_vehicle',
                    'street': parent_row.get('street', ''),
                    'input_location': parent_row.get('input_location', ''),
                    'heading': parent_row.get('heading', ''),
                    'pitch': parent_row.get('pitch', ''),
                    'fov': parent_row.get('fov', ''),
                    'pano_id': parent_row.get('pano_id', ''),
                    'pano_lat': parent_row.get('pano_lat', ''),
                    'pano_lng': parent_row.get('pano_lng', ''),
                    'status': 'ok',
                    'num_boxes_autogen': '0',
                    'needs_review': '1',
                    'review_status': 'todo',
                    'created_at': datetime.now().isoformat()
                })
                synth_count += 1
                save_manifest(manifest, manifest_path)
    
    logger.info(f"Created {synth_count} synthetic images")
    logger.info(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
