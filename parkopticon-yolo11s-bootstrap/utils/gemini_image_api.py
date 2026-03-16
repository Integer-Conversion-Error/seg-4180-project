import base64
import io
from dataclasses import dataclass

from google import genai
from google.genai import types
from PIL import Image


DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


class GeminiImageAPIError(RuntimeError):
    pass


@dataclass
class GeminiImageResult:
    image_bytes: bytes
    prompt_tokens: int
    candidate_tokens: int
    total_tokens: int


def _decode_inline_data(raw_data: object) -> bytes:
    if isinstance(raw_data, bytes):
        return raw_data
    if isinstance(raw_data, str):
        try:
            return base64.b64decode(raw_data)
        except Exception as exc:
            raise GeminiImageAPIError(
                "Failed to decode Gemini inline image data"
            ) from exc
    raise GeminiImageAPIError("Gemini inline image data has unsupported type")


def _guess_mime_type(image_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            fmt = (image.format or "").upper()
    except Exception:
        return "image/jpeg"

    if fmt == "PNG":
        return "image/png"
    if fmt == "WEBP":
        return "image/webp"
    return "image/jpeg"


def generate_image_with_usage(
    *,
    api_key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    input_images: list[bytes] | None = None,
) -> GeminiImageResult:
    if not api_key:
        raise GeminiImageAPIError("Gemini API key is required")
    if not prompt:
        raise GeminiImageAPIError("Prompt is required")

    images = [value for value in (input_images or []) if value]
    if not images:
        raise GeminiImageAPIError(
            "Gemini image editing requires at least one input image"
        )

    parts: list[types.Part] = []
    for image_bytes in images[:8]:
        mime_type = _guess_mime_type(image_bytes)
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt))

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=types.Content(role="user", parts=parts),
        )
    except Exception as exc:
        raise GeminiImageAPIError(f"Gemini image API request failed: {exc}") from exc

    prompt_tokens = 0
    candidate_tokens = 0
    total_tokens = 0
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is not None:
        prompt_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
        candidate_tokens = int(
            getattr(usage_metadata, "candidates_token_count", 0) or 0
        )
        total_tokens = int(getattr(usage_metadata, "total_token_count", 0) or 0)

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if not inline_data:
                continue
            raw_data = getattr(inline_data, "data", None)
            if not raw_data:
                continue
            return GeminiImageResult(
                image_bytes=_decode_inline_data(raw_data),
                prompt_tokens=prompt_tokens,
                candidate_tokens=candidate_tokens,
                total_tokens=total_tokens,
            )

    raise GeminiImageAPIError("No image payload returned by Gemini image API")


def generate_image(
    *,
    api_key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    input_images: list[bytes] | None = None,
) -> bytes:
    return generate_image_with_usage(
        api_key=api_key,
        prompt=prompt,
        model=model,
        input_images=input_images,
    ).image_bytes
