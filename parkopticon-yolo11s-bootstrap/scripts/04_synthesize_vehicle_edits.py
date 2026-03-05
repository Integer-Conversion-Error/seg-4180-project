#!/usr/bin/env python3
"""
Synthesize vehicle edits using Gemini image editing API.
Creates random_vehicle and enforcement_vehicle variants of empty frames.
Uses reference images for enforcement classes.

Future enhancement: Will add lookalike_negative class (Class 4) for hard negative examples
that look like enforcement but are NOT enforcement vehicles. See ONBOARDING.md for details.
"""

import argparse
import csv
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import resize_cover_center_crop


load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Reference image directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENFORCEMENT_DATASET_DIR = PROJECT_ROOT / "enforcement-dataset"
POLICE_OLD_DATASET_DIR = PROJECT_ROOT / "police-dataset-old"
POLICE_NEW_DATASET_DIR = PROJECT_ROOT / "police-dataset-new"

# Constraints (shared with synth_ui)
STRICT_SIZE_CONSTRAINT = """Critical output constraint: return an image with exactly the same width and height as the input street scene. Do not crop, resize, pad, rotate, or change aspect ratio."""

TRAFFIC_DIRECTION_CONSTRAINT = """Orientation constraint: the inserted vehicle must be aligned with the correct lane direction for this road scene, matching traffic flow and lane markings. Do not place a wrong-way vehicle."""

SCENE_LOCK_CONSTRAINT = """Scene integrity constraint: preserve the original scene geometry and composition exactly. Do not move or alter camera pose, horizon, vanishing points, road/curb alignment, lane markings, buildings, trees, poles, signs, parked objects, or any static landmark. Do not shift, warp, redraw, or restyle the background."""

LOCAL_EDIT_ONLY_CONSTRAINT = """Local-edit-only constraint: modify only the minimal pixel region required to insert the new vehicle, plus immediate physically consistent contact shadow/occlusion near that vehicle. All other pixels should remain visually unchanged. If constraints cannot be satisfied, return the original scene unchanged."""

SHADOW_CONSISTENCY_CONSTRAINT = """Shadow consistency constraint: shadow intensity and sharpness must match the scene's apparent lighting conditions. Bright sunny scenes should cast sharp, dark shadows. Overcast, cloudy, or diffuse lighting scenes should have soft, faint, or no visible shadows. Do not add hard, distinct shadows when the scene appears overcast or has soft ambient lighting."""

REALISTIC_VEHICLE_COLORS = [
    "white",
    "black",
    "silver",
    "gray",
    "dark gray",
    "blue",
    "navy blue",
    "red",
    "burgundy",
    "green",
    "beige",
    "brown",
]

REALISTIC_VEHICLE_BODY_TYPES = [
    "sedan",
    "compact sedan",
    "hatchback",
    "coupe",
    "SUV",
    "compact SUV",
    "pickup truck",
    "minivan",
    "crossover",
    "wagon",
]


def build_random_vehicle_prompt(image_id: str, seed: int) -> tuple[str, str, str]:
    rng = random.Random(f"{seed}:{image_id}")
    color = rng.choice(REALISTIC_VEHICLE_COLORS)
    body_type = rng.choice(REALISTIC_VEHICLE_BODY_TYPES)

    prompt = (
        "Change only what is necessary. Keep camera perspective, lighting, shadows, color grading identical. "
        f"Add exactly ONE {color} {body_type} on the roadway (not parked), medium size, clearly visible. "
        "It must look like a normal civilian vehicle with no emergency lightbar, no police markings, and no municipal enforcement markings. "
        "Avoid changing signs, buildings, or sky. No global style shifts. "
        + SCENE_LOCK_CONSTRAINT
        + " "
        + LOCAL_EDIT_ONLY_CONSTRAINT
        + " "
        + TRAFFIC_DIRECTION_CONSTRAINT
        + " "
        + SHADOW_CONSISTENCY_CONSTRAINT
        + " "
        + STRICT_SIZE_CONSTRAINT
    )
    return prompt, color, body_type


# Prompts for enforcement_vehicle WITH reference image
ENFORCEMENT_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference enforcement vehicle. "
    "Insert one enforcement vehicle into the target scene, matching the reference style while preserving scene realism. "
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
)

# Prompts for police_old WITH reference image
POLICE_OLD_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with OLD livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
)

# Prompts for police_new WITH reference image
POLICE_NEW_PROMPT_WITH_REF = (
    "You are given two images: first is the target street scene, second is a reference Ottawa Police cruiser with NEW livery. "
    "Insert one police cruiser matching the reference style into the target scene, preserving scene realism. "
    "Do not introduce unrelated scene changes. "
    + SCENE_LOCK_CONSTRAINT
    + " "
    + LOCAL_EDIT_ONLY_CONSTRAINT
    + " "
    + TRAFFIC_DIRECTION_CONSTRAINT
    + " "
    + SHADOW_CONSISTENCY_CONSTRAINT
    + " "
    + STRICT_SIZE_CONSTRAINT
)


def generate_image_id(parent_id: str, edit_type: str) -> str:
    return f"{parent_id}_{edit_type}"


def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def save_manifest(manifest: list, manifest_path: Path):
    if not manifest:
        return
    fieldnames = list(manifest[0].keys())
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def load_reference_images(dataset_dir: Path) -> List[Path]:
    """Load all reference images from a directory."""
    if not dataset_dir.exists():
        logger.warning(f"Reference directory not found: {dataset_dir}")
        return []

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for ext in extensions:
        images.extend(dataset_dir.glob(f"*{ext}"))
        images.extend(dataset_dir.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def synthesize_image(
    client: genai.Client,
    image_path: Path,
    prompt: str,
    model: str = "models/gemini-3-pro-image-preview",
    reference_path: Optional[Path] = None,
) -> Optional[bytes]:
    try:
        # Load background image
        background_image = Image.open(image_path)

        # Build contents list
        if reference_path and reference_path.exists():
            # With reference image
            reference_image = Image.open(reference_path)
            contents = [prompt, background_image, reference_image]
        else:
            # No reference image
            contents = [prompt, background_image]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["image"],
            ),
        )

        if hasattr(response, "candidates") and response.candidates:
            first_candidate = response.candidates[0]
            content = getattr(first_candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    with Image.open(image_path) as source_image:
                        expected_size = source_image.size
                    with Image.open(BytesIO(data)) as generated_image:
                        generated_size = generated_image.size
                    if generated_size != expected_size:
                        try:
                            data = resize_cover_center_crop(data, expected_size)
                            logger.warning(
                                "Adjusted generated image size from %s to %s via cover-resize + center-crop",
                                generated_size,
                                expected_size,
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed to adjust generated image size: expected=%s got=%s error=%s",
                                expected_size,
                                generated_size,
                                exc,
                            )
                            return None
                    return data

        return None

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return None


# Thread-safe lock for manifest updates
manifest_lock = threading.Lock()


def process_image(
    client: genai.Client,
    img_id: str,
    parent_row: dict,
    out_dir: Path,
    existing_ids: set,
    enforcement_candidates: set,
    reference_images: dict,
    model: str,
    seed: int,
) -> Tuple[int, list]:
    """Process a single image and return (count, new_rows)."""
    new_rows = []
    count = 0

    parent_path = Path(parent_row.get("file_path", ""))
    if not parent_path.exists():
        return (0, [])

    # Generate random_vehicle (no reference needed)
    synth_id_random = generate_image_id(img_id, "random_vehicle")
    if synth_id_random not in existing_ids:
        random_vehicle_prompt, vehicle_color, vehicle_body_type = (
            build_random_vehicle_prompt(img_id, seed)
        )
        logger.info(
            "Creating random_vehicle for %s with style: %s %s",
            img_id,
            vehicle_color,
            vehicle_body_type,
        )
        img_data = synthesize_image(client, parent_path, random_vehicle_prompt, model)

        if img_data:
            out_path = out_dir / "random_vehicle" / f"{synth_id_random}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(img_data)

            new_rows.append(
                {
                    "image_id": synth_id_random,
                    "file_path": str(out_path),
                    "split": "unset",
                    "parent_image_id": img_id,
                    "is_synthetic": "1",
                    "edit_type": "random_vehicle",
                    "expected_inserted_class": "vehicle",
                    "street": parent_row.get("street", ""),
                    "input_location": parent_row.get("input_location", ""),
                    "heading": parent_row.get("heading", ""),
                    "pitch": parent_row.get("pitch", ""),
                    "fov": parent_row.get("fov", ""),
                    "pano_id": parent_row.get("pano_id", ""),
                    "pano_lat": parent_row.get("pano_lat", ""),
                    "pano_lng": parent_row.get("pano_lng", ""),
                    "status": "ok",
                    "num_boxes_autogen": "0",
                    "needs_review": "1",
                    "review_status": "todo",
                    "created_at": datetime.now().isoformat(),
                }
            )
            count += 1

    # Generate enforcement classes if this is an enforcement candidate
    if img_id in enforcement_candidates:
        for edit_type, prompt, reference_key in [
            ("enforcement_vehicle", ENFORCEMENT_PROMPT_WITH_REF, "enforcement"),
            ("police_old", POLICE_OLD_PROMPT_WITH_REF, "police_old"),
            ("police_new", POLICE_NEW_PROMPT_WITH_REF, "police_new"),
        ]:
            synth_id_enf = generate_image_id(img_id, edit_type)
            if synth_id_enf in existing_ids:
                continue

            # Get a random reference image for this class
            ref_images = reference_images.get(reference_key, [])
            ref_path = random.choice(ref_images) if ref_images else None

            logger.info(
                f"Creating {edit_type} for {img_id}"
                + (f" with reference: {ref_path.name}" if ref_path else "")
            )
            img_data = synthesize_image(
                client, parent_path, prompt, model, reference_path=ref_path
            )

            if img_data:
                out_path = out_dir / edit_type / f"{synth_id_enf}.jpg"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    f.write(img_data)

                new_rows.append(
                    {
                        "image_id": synth_id_enf,
                        "file_path": str(out_path),
                        "split": "unset",
                        "parent_image_id": img_id,
                        "is_synthetic": "1",
                        "edit_type": edit_type,
                        "expected_inserted_class": edit_type,
                        "street": parent_row.get("street", ""),
                        "input_location": parent_row.get("input_location", ""),
                        "heading": parent_row.get("heading", ""),
                        "pitch": parent_row.get("pitch", ""),
                        "fov": parent_row.get("fov", ""),
                        "pano_id": parent_row.get("pano_id", ""),
                        "pano_lat": parent_row.get("pano_lat", ""),
                        "pano_lng": parent_row.get("pano_lng", ""),
                        "status": "ok",
                        "num_boxes_autogen": "0",
                        "needs_review": "1",
                        "review_status": "todo",
                        "created_at": datetime.now().isoformat(),
                    }
                )
                count += 1

        # TODO: Future enhancement - Generate lookalike_negative class
        # Class 4: lookalike_negative includes vehicles that visually resemble enforcement
        # but are NOT enforcement vehicles. These hard negatives improve model discriminative power.

        # Example future prompts for lookalike_negative synthesis:
        # - 'Add one black vehicle with white door panels (resembling two-tone paint, not police livery)'
        # - 'Add one white vehicle with blue accent stripes (like a taxi, not police markings)'
        # - 'Add one dark vehicle with custom paint but no police insignia or emergency equipment'

        # Implementation should:
        # 1. Add lookalike_negative reference images directory (if using styled references)
        # 2. Define LOOKALIKE_NEGATIVE_PROMPT_WITH_REF prompt with safety constraints
        # 3. Add ('lookalike_negative', LOOKALIKE_NEGATIVE_PROMPT_WITH_REF, 'lookalike_negative') to edit_type loop
        # 4. Create output directory: out_dir / 'lookalike_negative'
        # 5. Update expected_inserted_class to 'lookalike_negative' in manifest row

    return (count, new_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize vehicle edits with reference images"
    )
    parser.add_argument(
        "--manifest", "-m", default="manifests/images.csv", help="Input manifest"
    )
    parser.add_argument(
        "--empty-list",
        "-e",
        default="lists/empty_candidates.txt",
        help="Empty candidates file",
    )
    parser.add_argument(
        "--excluded-list",
        default="lists/excluded_from_synth.txt",
        help="Excluded from synthesis file",
    )
    parser.add_argument(
        "--out-dir", "-o", default="data/images_synth", help="Output directory"
    )
    parser.add_argument("--api-key", "-k", default=None, help="Gemini API key")
    parser.add_argument(
        "--model", default="models/gemini-3-pro-image-preview", help="Gemini model"
    )
    parser.add_argument(
        "--enforcement-rate", type=float, default=0.2, help="Fraction for enforcement"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of parallel workers"
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume from existing"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key. Set GEMINI_API_KEY or use --api-key")
        return

    client = genai.Client(api_key=api_key)

    manifest_path = Path(args.manifest)
    empty_list_path = Path(args.empty_list)
    excluded_list_path = Path(args.excluded_list)
    out_dir = Path(args.out_dir)

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    manifest = load_manifest(manifest_path)

    if empty_list_path.exists():
        with open(empty_list_path, "r") as f:
            empty_candidates = [line.strip() for line in f if line.strip()]
    else:
        logger.warning(f"Empty list not found: {empty_list_path}")
        empty_candidates = []

    # Filter out excluded images
    excluded_ids = set()
    if excluded_list_path.exists():
        with open(excluded_list_path, "r") as f:
            excluded_ids = {line.strip() for line in f if line.strip()}
        if excluded_ids:
            original_count = len(empty_candidates)
            empty_candidates = [
                cid for cid in empty_candidates if cid not in excluded_ids
            ]
            logger.info(
                f"Filtered {original_count - len(empty_candidates)} excluded images from synthesis"
            )

    # Build set of existing synthetic IDs for resume
    existing_ids = set()
    if args.resume:
        existing_ids = {
            row["image_id"] for row in manifest if row.get("is_synthetic") == "1"
        }
        logger.info(f"Resuming with {len(existing_ids)} existing synthetics")

    # Load reference images for each enforcement class
    reference_images = {
        "enforcement": load_reference_images(ENFORCEMENT_DATASET_DIR),
        "police_old": load_reference_images(POLICE_OLD_DATASET_DIR),
        "police_new": load_reference_images(POLICE_NEW_DATASET_DIR),
    }

    for ref_type, refs in reference_images.items():
        logger.info(f"Loaded {len(refs)} reference images for {ref_type}")

    # Verify we have reference images for enforcement classes
    for ref_type, refs in reference_images.items():
        if not refs:
            logger.error(
                f"No reference images found for {ref_type} in {ENFORCEMENT_DATASET_DIR if ref_type == 'enforcement' else POLICE_OLD_DATASET_DIR if ref_type == 'police_old' else POLICE_NEW_DATASET_DIR}"
            )
            return

    random.seed(args.seed)

    enforcement_candidates = set(
        random.sample(
            empty_candidates, k=int(len(empty_candidates) * args.enforcement_rate)
        )
        if empty_candidates
        else []
    )

    # Build lookup for manifest rows
    manifest_lookup = {row.get("image_id"): row for row in manifest}

    # Process images in parallel
    total_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for img_id in empty_candidates:
            parent_row = manifest_lookup.get(img_id)
            if not parent_row:
                continue

            future = executor.submit(
                process_image,
                client,
                img_id,
                parent_row,
                out_dir,
                existing_ids,
                enforcement_candidates,
                reference_images,
                args.model,
                args.seed,
            )
            futures[future] = img_id

        # Collect results with progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Synthesizing"
        ):
            count, new_rows = future.result()
            total_count += count

            # Update manifest (thread-safe)
            if new_rows:
                with manifest_lock:
                    manifest.extend(new_rows)
                    save_manifest(manifest, manifest_path)
                    # Add new IDs to existing_ids to avoid duplicates
                    for row in new_rows:
                        existing_ids.add(row["image_id"])

    logger.info(f"Created {total_count} synthetic images")
    logger.info(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
