#!/usr/bin/env python3
"""
Generate 4 cardinal views (Front, Left, Rear, Right) for each enforcement vehicle model.
Uses Grok to synthesize these specific angles from a provided source image.
The output images are saved with cardinal suffixes (e.g., model_N.jpg, model_W.jpg, model_S.jpg, model_E.jpg).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.grok_image_api import (
    GrokImageAPIError,
    encode_image_to_base64,
    generate_image,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# cardinal mapping for easier prompting with strict side-specific checks
CARDINAL_MAP = {
    "N": {
        "desc": "Front view (0 degrees)",
        "check": "The vehicle MUST be facing DIRECTLY TOWARDS the camera. Both headlights and the front grille must be centered and symmetrical. No side profile should be visible.",
    },
    "W": {
        "desc": "Left side view (90 degrees counter-clockwise from front)",
        "check": "The vehicle MUST be pointing DIRECTLY TO THE LEFT. This is a strict 90-degree profile view. The front of the car must be on the left side of the frame, and the rear on the right. No front grille or rear bumper should be visible.",
    },
    "S": {
        "desc": "Rear view (180 degrees)",
        "check": "The vehicle MUST be facing DIRECTLY AWAY from the camera. Both taillights and the rear license plate area must be centered and symmetrical. No side profile should be visible.",
    },
    "E": {
        "desc": "Right side view (270 degrees counter-clockwise from front)",
        "check": "The vehicle MUST be pointing DIRECTLY TO THE RIGHT. This is a strict 90-degree profile view. The front of the car must be on the right side of the frame, and the rear on the left. No front grille or rear bumper should be visible.",
    },
}

CARDINAL_SUFFIXES = list(CARDINAL_MAP.keys())

# System prompt for lighting and scene agnostic reference generation
REFERENCE_GEN_PROMPT = """
You are an expert vehicle reference generator. 
Given the provided image of a specific vehicle model, generate a high-fidelity reference image of the EXACT same vehicle model and livery from the requested cardinal angle.

Requirements:
1. BACKGROUND: Pure, solid white studio background. No road, no trees, no sky.
2. LIGHTING: Neutral, even studio lighting. No harsh shadows or reflections.
3. ANGLE: Exactly the requested cardinal angle ({angle_desc}). {strict_check}
4. MODEL FIDELITY: Maintain every detail of the vehicle model, including lightbars, markings, and livery colors.
5. CROPPING: The vehicle should be centered and fill most of the frame.
6. FORMAT: Return only the generated image. Do not add text or borders.

Requested Angle: {angle_desc}
"""


def synthesize_angle(
    grok_api_key: str,
    source_image_path: Path,
    angle_suffix: str,
    model: str = "grok-imagine-image",
) -> bytes:
    angle_info = CARDINAL_MAP[angle_suffix]
    angle_desc = angle_info["desc"]
    strict_check = angle_info["check"]
    prompt = REFERENCE_GEN_PROMPT.format(
        angle_desc=angle_desc, strict_check=strict_check
    )

    try:
        with source_image_path.open("rb") as handle:
            source_b64 = encode_image_to_base64(handle.read())

        return generate_image(
            api_key=grok_api_key,
            model=model,
            prompt=prompt,
            input_images=[source_b64],
        )
    except GrokImageAPIError as e:
        logger.error(
            f"Grok generation failed for {angle_suffix} / {source_image_path.name}: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Failed to synthesize {angle_suffix} for {source_image_path.name}: {e}"
        )
        raise


def process_model_directory(grok_api_key: str, model_dir: Path, model_name: str, args):
    """Process all source images in a model directory to generate cardinal references."""
    source_images = []
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        source_images.extend(model_dir.glob(f"*{ext}"))
        source_images.extend(model_dir.glob(f"*{ext.upper()}"))

    # Filter out already generated cardinal images (those ending with _N, _S, _E, _W)
    sources = []
    for img in source_images:
        is_cardinal = False
        for suffix in CARDINAL_SUFFIXES:
            if img.stem.endswith(f"_{suffix}"):
                is_cardinal = True
                break
        if not is_cardinal:
            sources.append(img)

    if not sources:
        logger.warning(f"No source images found in {model_dir}")
        return

    logger.info(f"Processing model '{model_name}' with {len(sources)} source images")

    for source_img in sources:
        for suffix in CARDINAL_SUFFIXES:
            out_name = f"{source_img.stem}_{suffix}.jpg"
            out_path = model_dir / out_name

            if out_path.exists() and not args.force:
                logger.info(f"Skipping existing reference: {out_path.name}")
                continue

            logger.info(f"Generating {suffix} view for {source_img.name} -> {out_name}")
            try:
                img_data = synthesize_angle(
                    grok_api_key, source_img, suffix, model=args.model
                )
                with open(out_path, "wb") as f:
                    f.write(img_data)
                logger.info(f"Saved: {out_path.name}")
            except Exception:
                continue

        if (
            args.limit_per_model
            and sources.index(source_img) + 1 >= args.limit_per_model
        ):
            break


def main():
    parser = argparse.ArgumentParser(description="Generate cardinal vehicle references")
    parser.add_argument(
        "--dir",
        default="enforcement-dataset",
        help="Root directory for enforcement models",
    )
    parser.add_argument(
        "--model",
        default="grok-imagine-image",
        help="Grok image model alias",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing references"
    )
    parser.add_argument(
        "--limit-per-model",
        type=int,
        default=1,
        help="Max source images to process per model",
    )
    parser.add_argument("--api-key", help="Grok API key")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        logger.error("No API key. Set XAI_API_KEY/GROK_API_KEY or use --api-key")
        return

    root_dir = Path(args.dir)

    if not root_dir.exists():
        logger.error(f"Directory not found: {root_dir}")
        return

    # Iterate through each model subdirectory
    for model_dir in root_dir.iterdir():
        if model_dir.is_dir():
            process_model_directory(api_key, model_dir, model_dir.name, args)


if __name__ == "__main__":
    main()
