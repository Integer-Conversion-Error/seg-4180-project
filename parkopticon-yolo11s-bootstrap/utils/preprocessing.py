"""Shared preprocessing utilities for image operations.

Provides functions for:
- Bottom cropping (removing overlays/watermarks from street view images)
- Resizing with cover-to-fit and center crop (for handling size mismatches)
- Composable preprocessing pipelines
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def crop_image_bottom(
    source_path: Path, target_path: Path, crop_px: int, quality: int = 95
) -> Tuple[bool, str]:
    """Crop pixels from the bottom of an image.

    Args:
        source_path (Path): Path to source image file
        target_path (Path): Path where cropped image will be saved
        crop_px (int): Number of pixels to remove from bottom
        quality (int): JPEG quality for saved image (1-100, default 95)

    Returns:
        Tuple[bool, str]: (success, error_message) where success is True if crop
            succeeded, and error_message is empty string on success or error
            description on failure.

    Example:
        >>> success, error = crop_image_bottom(
        ...     Path("image.jpg"), Path("image_bc30.jpg"), 30
        ... )
        >>> if not success:
        ...     print(f"Crop failed: {error}")
    """
    try:
        with Image.open(source_path) as image:
            width, height = image.size
            if height <= crop_px:
                return False, f"image height ({height}) <= crop ({crop_px})"

            cropped = image.crop((0, 0, width, height - crop_px))
            target_path.parent.mkdir(parents=True, exist_ok=True)

            image_format = (image.format or "").upper()
            if image_format in {"JPEG", "JPG"}:
                cropped.save(target_path, quality=quality, optimize=True)
            else:
                cropped.save(target_path)

            logger.info(
                "Cropped %s: removed %dpx from bottom -> %s",
                source_path.name,
                crop_px,
                target_path.name,
            )
            return True, ""
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        logger.error("Crop failed for %s: %s", source_path, error_msg)
        return False, error_msg


def resize_cover_center_crop(image_bytes: bytes, target_size: Tuple[int, int]) -> bytes:
    """Resize image to cover target size, then center crop to exact dimensions.

    This is useful for handling generated images that don't match expected
    dimensions. The algorithm:
    1. Scale image up so it covers the target dimensions (no black bars)
    2. Center crop to exact target size
    3. Return as JPEG bytes

    Args:
        image_bytes (bytes): Image data (any PIL-supported format)
        target_size (Tuple[int, int]): Target (width, height) in pixels

    Returns:
        bytes: JPEG-encoded image with exact target dimensions

    Raises:
        ValueError: If image_bytes is invalid or target_size invalid

    Example:
        >>> from pathlib import Path
        >>> original = Path("640x480.jpg").read_bytes()
        >>> resized = resize_cover_center_crop(original, (640, 480))
        >>> with Image.open(BytesIO(resized)) as img:
        ...     assert img.size == (640, 480)
    """
    target_w, target_h = target_size
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    with Image.open(BytesIO(image_bytes)) as generated_image:
        src_w, src_h = generated_image.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"Invalid generated image size: {(src_w, src_h)}")

        scale = max(target_w / src_w, target_h / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        resized = generated_image.resize(
            (resized_w, resized_h), Image.Resampling.LANCZOS
        )

        left = max(0, (resized_w - target_w) // 2)
        top = max(0, (resized_h - target_h) // 2)
        right = left + target_w
        bottom = top + target_h
        cropped = resized.crop((left, top, right, bottom))

        if cropped.mode not in {"RGB", "L"}:
            cropped = cropped.convert("RGB")
        elif cropped.mode == "L":
            cropped = cropped.convert("RGB")

        output = BytesIO()
        cropped.save(output, format="JPEG", quality=95, optimize=True)
        result = output.getvalue()

        logger.info(
            "Resized image from %s to %s (scale=%.2f)",
            (src_w, src_h),
            target_size,
            scale,
        )
        return result


def apply_consistent_preprocessing(
    image: Image.Image,
    crop_bottom_px: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """Apply a preprocessing pipeline to an image.

    Applies (in order):
    1. Bottom crop (if crop_bottom_px provided)
    2. Resize to cover + center crop (if target_size provided)

    Args:
        image (Image.Image): PIL Image object
        crop_bottom_px (Optional[int]): Pixels to crop from bottom, or None to skip
        target_size (Optional[Tuple[int, int]]): Target (width, height) or None to skip

    Returns:
        Image.Image: Processed image

    Example:
        >>> from PIL import Image
        >>> img = Image.open("path/to/image.jpg")
        >>> # Crop 30px from bottom, then resize to 640x480
        >>> processed = apply_consistent_preprocessing(
        ...     img, crop_bottom_px=30, target_size=(640, 480)
        ... )
    """
    result = image
    step = "input"

    try:
        # Step 1: Bottom crop
        if crop_bottom_px and crop_bottom_px > 0:
            width, height = result.size
            if height > crop_bottom_px:
                result = result.crop((0, 0, width, height - crop_bottom_px))
                step = f"cropped_bottom_{crop_bottom_px}px"
                logger.debug("Applied bottom crop: %dpx", crop_bottom_px)

        # Step 2: Resize to cover + center crop
        if target_size:
            target_w, target_h = target_size
            src_w, src_h = result.size

            # Skip if already exact size
            if (src_w, src_h) != target_size:
                scale = max(target_w / src_w, target_h / src_h)
                resized_w = max(1, int(round(src_w * scale)))
                resized_h = max(1, int(round(src_h * scale)))
                resized = result.resize(
                    (resized_w, resized_h), Image.Resampling.LANCZOS
                )

                left = max(0, (resized_w - target_w) // 2)
                top = max(0, (resized_h - target_h) // 2)
                right = left + target_w
                bottom = top + target_h
                result = resized.crop((left, top, right, bottom))
                step = f"{step}->resize_cover_center_crop"
                logger.debug(
                    "Applied resize-cover-center-crop: %s to %s",
                    (src_w, src_h),
                    target_size,
                )

        logger.info("Preprocessing complete: %s", step)
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("Preprocessing failed at step %s: %s", step, exc)
        raise
