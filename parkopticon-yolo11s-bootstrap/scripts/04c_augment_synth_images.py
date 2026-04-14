#!/usr/bin/env python3
"""
Augment synthetic images with noise, blur, compression, and color jitter.
Creates copies instead of modifying originals.
"""

import argparse
import csv
import io
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from utils.dataset_exclusion import load_dataset_excluded_ids

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_gaussian_noise(img: Image.Image, sigma: float = 2.0) -> Image.Image:
    """Add Gaussian noise to image."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_film_grain(
    img: Image.Image, intensity: float = 0.3, grain_size: int = 1
) -> Image.Image:
    """Add realistic film grain effect."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create grain at lower resolution for more realistic look
    grain_h, grain_w = h // grain_size, w // grain_size
    grain = np.random.normal(0, intensity * 127, (grain_h, grain_w))

    # Scale up grain to original size (creates clumping effect)
    from PIL import Image as PILImage

    grain_img = PILImage.fromarray(
        np.clip(grain + 127, 0, 255).astype(np.uint8), mode="L"
    )
    grain_img = grain_img.resize((w, h), PILImage.BILINEAR)
    grain = np.array(grain_img, dtype=np.float32) - 127

    # Apply grain to all channels
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + grain * (intensity * 0.5), 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def add_iso_noise(img: Image.Image, strength: float = 0.1) -> Image.Image:
    """Add high-ISO camera noise (color noise in shadows)."""
    arr = np.array(img, dtype=np.float32) / 255.0

    # Color noise (more visible in shadows)
    luminance = np.mean(arr, axis=2)
    shadow_mask = 1.0 - luminance  # More noise in shadows
    shadow_mask = np.clip(shadow_mask * 1.5, 0, 1)

    for c in range(3):
        noise = np.random.normal(0, strength, arr.shape[:2])
        arr[:, :, c] += noise * shadow_mask

    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_jpeg_compression(img: Image.Image, quality: int = 85) -> Image.Image:
    """Apply JPEG compression artifacts."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_chromatic_aberration(img: Image.Image, shift: int = 1) -> Image.Image:
    """Add slight chromatic aberration (color channel shift)."""
    arr = np.array(img, dtype=np.uint8)
    # Shift red channel right, blue channel left
    result = arr.copy()
    if shift > 0:
        result[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)  # Red right
        result[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)  # Blue left
    return Image.fromarray(result)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """Adjust saturation."""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def add_blur(img: Image.Image, radius: float) -> Image.Image:
    """Add Gaussian blur."""
    if radius > 0:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def add_sharpen(img: Image.Image, factor: float) -> Image.Image:
    """Add sharpening."""
    if factor > 0:
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(factor)
    return img


def apply_random_augmentation(
    img: Image.Image,
    seed: int,
    noise_range: tuple = (0.5, 3.0),
    jpeg_quality_range: tuple = (75, 95),
    chromatic_shift_range: tuple = (0, 2),
    brightness_range: tuple = (0.9, 1.1),
    contrast_range: tuple = (0.9, 1.1),
    saturation_range: tuple = (0.85, 1.15),
    blur_range: tuple = (0.0, 0.8),
    sharpen_range: tuple = (0.8, 1.3),
    film_grain_range: tuple = (0.0, 0.0),
    iso_noise_range: tuple = (0.0, 0.0),
    noise_passes: int = 1,
) -> Image.Image:
    """Apply a random combination of augmentations."""
    rng = random.Random(seed)
    result = img.copy()

    # Noise - apply multiple passes for heavier grain
    for _ in range(noise_passes):
        sigma = rng.uniform(*noise_range)
        if rng.random() > 0.2:  # 80% chance
            result = add_gaussian_noise(result, sigma=sigma)

    # Film grain (more realistic noise)
    if film_grain_range[0] > 0 or film_grain_range[1] > 0:
        grain_intensity = rng.uniform(*film_grain_range)
        if grain_intensity > 0 and rng.random() > 0.4:
            result = add_film_grain(result, intensity=grain_intensity, grain_size=2)

    # ISO noise (color noise in shadows)
    if iso_noise_range[0] > 0 or iso_noise_range[1] > 0:
        iso_strength = rng.uniform(*iso_noise_range)
        if iso_strength > 0 and rng.random() > 0.5:
            result = add_iso_noise(result, strength=iso_strength)

    # JPEG compression
    quality = rng.randint(*jpeg_quality_range)
    if rng.random() > 0.3:  # 70% chance
        result = add_jpeg_compression(result, quality=quality)

    # Chromatic aberration
    if rng.random() > 0.5:  # 50% chance
        shift = rng.randint(*chromatic_shift_range)
        result = add_chromatic_aberration(result, shift=shift)

    # Brightness
    brightness = rng.uniform(*brightness_range)
    result = adjust_brightness(result, brightness)

    # Contrast
    contrast = rng.uniform(*contrast_range)
    result = adjust_contrast(result, contrast)

    # Saturation
    saturation = rng.uniform(*saturation_range)
    result = adjust_saturation(result, saturation)

    # Blur or Sharpen (not both)
    if rng.random() > 0.7:  # 30% chance of blur
        radius = rng.uniform(*blur_range)
        result = add_blur(result, radius)
    elif rng.random() > 0.5:  # 35% chance of sharpen
        factor = rng.uniform(*sharpen_range)
        result = add_sharpen(result, factor)

    return result


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


def main():
    parser = argparse.ArgumentParser(
        description="Augment synthetic images with noise, blur, compression, etc."
    )
    parser.add_argument(
        "--run-dir",
        default=".",
        help="Run directory root (all relative paths resolve against this)",
    )
    parser.add_argument(
        "--manifest", "-m", default="manifests/images.csv", help="Input manifest"
    )
    parser.add_argument(
        "--out-manifest",
        "-o",
        default=None,
        help="Output manifest (default: same as input)",
    )
    parser.add_argument(
        "--synth-dir",
        "-s",
        default="data/images_synth",
        help="Synthetic images directory",
    )
    parser.add_argument(
        "--out-dir",
        "-d",
        default=None,
        help="Output directory (default: same as synth-dir)",
    )
    parser.add_argument(
        "--copies-per-image",
        "-n",
        type=int,
        default=2,
        help="Number of augmented copies per image",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Skip already augmented images"
    )
    parser.add_argument(
        "--edit-types",
        default="enforcement_vehicle,police_old,police_new,random_vehicle",
        help="Comma-separated edit types to augment",
    )

    # Augmentation intensity presets
    parser.add_argument(
        "--intensity",
        choices=["light", "medium", "heavy", "extreme"],
        default="medium",
        help="Augmentation intensity preset",
    )

    # Direct control overrides (override intensity preset)
    parser.add_argument(
        "--noise-sigma",
        type=float,
        nargs=2,
        default=None,
        help="Noise sigma range (min max), e.g. --noise-sigma 2.0 8.0",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        nargs=2,
        default=None,
        help="JPEG quality range (min max), e.g. --jpeg-quality 40 75",
    )
    parser.add_argument(
        "--film-grain",
        type=float,
        nargs=2,
        default=None,
        help="Film grain intensity range (min max), e.g. --film-grain 0.2 0.5",
    )
    parser.add_argument(
        "--iso-noise",
        type=float,
        nargs=2,
        default=None,
        help="ISO noise strength range (min max), e.g. --iso-noise 0.05 0.15",
    )
    parser.add_argument(
        "--noise-passes",
        type=int,
        default=None,
        help="Number of noise application passes (1-5), more = heavier grain",
    )
    args = parser.parse_args()

    # Resolve run directory
    run_dir = Path(args.run_dir).resolve()

    def _resolve_path(raw_value: str) -> Path:
        candidate = Path(raw_value)
        return candidate if candidate.is_absolute() else run_dir / candidate

    def _resolve_image_path(raw_value: str) -> Path:
        if not raw_value:
            return run_dir / ""
        candidate = Path(raw_value)
        return candidate if candidate.is_absolute() else run_dir / candidate

    manifest_path = _resolve_path(args.manifest)
    out_manifest_path = (
        _resolve_path(args.out_manifest) if args.out_manifest else manifest_path
    )
    synth_dir = _resolve_path(args.synth_dir)
    out_dir = _resolve_path(args.out_dir) if args.out_dir else synth_dir

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return

    manifest = load_manifest(manifest_path)
    excluded_ids = load_dataset_excluded_ids(manifest_path)

    # Filter synthetic images by edit type
    edit_types = [et.strip() for et in args.edit_types.split(",")]
    synthetic_images = [
        row
        for row in manifest
        if row.get("is_synthetic") == "1"
        and row.get("edit_type") in edit_types
        and (row.get("image_id") or "").strip() not in excluded_ids
    ]

    logger.info(
        f"Found {len(synthetic_images)} synthetic images to potentially augment"
    )

    # Set augmentation parameters based on intensity
    if args.intensity == "light":
        noise_range = (0.3, 1.5)
        jpeg_quality_range = (85, 95)
        brightness_range = (0.95, 1.05)
        contrast_range = (0.95, 1.05)
        saturation_range = (0.9, 1.1)
        film_grain_range = (0.0, 0.0)
        iso_noise_range = (0.0, 0.0)
        noise_passes = 1
    elif args.intensity == "heavy":
        noise_range = (3.0, 8.0)
        jpeg_quality_range = (45, 70)
        brightness_range = (0.85, 1.15)
        contrast_range = (0.85, 1.15)
        saturation_range = (0.75, 1.25)
        film_grain_range = (0.15, 0.35)
        iso_noise_range = (0.03, 0.10)
        noise_passes = 2
    elif args.intensity == "extreme":
        noise_range = (5.0, 15.0)
        jpeg_quality_range = (25, 50)
        brightness_range = (0.75, 1.25)
        contrast_range = (0.75, 1.25)
        saturation_range = (0.65, 1.35)
        film_grain_range = (0.25, 0.50)
        iso_noise_range = (0.08, 0.18)
        noise_passes = 3
    else:  # medium
        noise_range = (0.5, 3.0)
        jpeg_quality_range = (75, 95)
        brightness_range = (0.9, 1.1)
        contrast_range = (0.9, 1.1)
        saturation_range = (0.85, 1.15)
        film_grain_range = (0.0, 0.15)
        iso_noise_range = (0.0, 0.05)
        noise_passes = 1

    # Apply direct control overrides
    if args.noise_sigma:
        noise_range = tuple(args.noise_sigma)
        logger.info(f"Noise sigma override: {noise_range}")
    if args.jpeg_quality:
        jpeg_quality_range = tuple(args.jpeg_quality)
        logger.info(f"JPEG quality override: {jpeg_quality_range}")
    if args.film_grain:
        film_grain_range = tuple(args.film_grain)
        logger.info(f"Film grain override: {film_grain_range}")
    if args.iso_noise:
        iso_noise_range = tuple(args.iso_noise)
        logger.info(f"ISO noise override: {iso_noise_range}")
    if args.noise_passes:
        noise_passes = args.noise_passes
        logger.info(f"Noise passes override: {noise_passes}")

    # Build set of existing augmented IDs for resume
    existing_ids = set()
    if args.resume:
        existing_ids = {
            row["image_id"] for row in manifest if "_aug" in row.get("image_id", "")
        }
        logger.info(f"Resuming with {len(existing_ids)} existing augmented images")

    random.seed(args.seed)
    new_rows = []
    augmented_count = 0

    for row in tqdm(synthetic_images, desc="Augmenting images"):
        image_id = row.get("image_id", "")
        raw_file_path = row.get("file_path", "")
        file_path = _resolve_image_path(raw_file_path)

        if not file_path.exists():
            continue

        # Load original image
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue

        # Create multiple augmented copies
        for copy_num in range(args.copies_per_image):
            aug_id = f"{image_id}_aug{copy_num + 1}"

            if aug_id in existing_ids:
                continue

            # Apply random augmentation
            aug_seed = hash(f"{args.seed}:{image_id}:{copy_num}")
            aug_img = apply_random_augmentation(
                img,
                seed=aug_seed,
                noise_range=noise_range,
                jpeg_quality_range=jpeg_quality_range,
                brightness_range=brightness_range,
                contrast_range=contrast_range,
                saturation_range=saturation_range,
                film_grain_range=film_grain_range,
                iso_noise_range=iso_noise_range,
                noise_passes=noise_passes,
            )

            # Determine output path
            rel_path = (
                file_path.relative_to(synth_dir)
                if file_path.is_relative_to(synth_dir)
                else Path(file_path.name)
            )
            aug_path = out_dir / rel_path.parent / f"{aug_id}.jpg"
            aug_path.parent.mkdir(parents=True, exist_ok=True)

            # Save augmented image
            aug_img.save(aug_path, "JPEG", quality=92)

            # Create manifest entry
            new_row = row.copy()
            new_row["image_id"] = aug_id
            new_row["file_path"] = str(aug_path)
            new_row["parent_image_id"] = image_id
            new_row["is_synthetic"] = "1"
            new_row["edit_type"] = f"{row.get('edit_type', '')}_augmented"
            new_row["needs_review"] = "1"
            new_row["review_status"] = "todo"
            new_row["created_at"] = datetime.now().isoformat()

            new_rows.append(new_row)
            augmented_count += 1

    # Update manifest
    if new_rows:
        manifest.extend(new_rows)
        save_manifest(manifest, out_manifest_path)
        logger.info(f"Created {augmented_count} augmented images")
        logger.info(f"Manifest updated: {out_manifest_path}")
    else:
        logger.info("No new images augmented")


if __name__ == "__main__":
    main()
