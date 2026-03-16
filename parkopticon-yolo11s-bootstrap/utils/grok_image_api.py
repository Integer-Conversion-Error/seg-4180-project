import base64
import io
import json
from urllib import error, request

from PIL import Image


DEFAULT_BASE_URL = "https://api.x.ai"
DEFAULT_MODEL = "grok-imagine-image"


class GrokImageAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _guess_mime_from_base64(image_b64: str) -> str:
    try:
        decoded = base64.b64decode(image_b64, validate=False)
    except Exception:
        return "image/jpeg"

    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if decoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if decoded.startswith(b"RIFF") and len(decoded) > 12 and decoded[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _to_data_uri(image_value: str) -> str:
    value = image_value.strip()
    if value.startswith("data:"):
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    mime = _guess_mime_from_base64(value)
    return f"data:{mime};base64,{value}"


def _maybe_reduce_data_uri(
    image_uri: str,
    *,
    max_bytes: int = 2_500_000,
    max_side_px: int = 1536,
    jpeg_quality: int = 88,
) -> str:
    if not image_uri.startswith("data:"):
        return image_uri

    if "," not in image_uri:
        return image_uri

    header, encoded = image_uri.split(",", 1)
    if ";base64" not in header:
        return image_uri

    try:
        raw = base64.b64decode(encoded, validate=False)
    except Exception:
        return image_uri

    if len(raw) <= max_bytes:
        return image_uri

    try:
        with Image.open(io.BytesIO(raw)) as image:
            converted = image.convert("RGB")
            converted.thumbnail((max_side_px, max_side_px), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            converted.save(
                out,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )
            compressed = out.getvalue()
    except Exception:
        return image_uri

    new_b64 = base64.b64encode(compressed).decode("ascii")
    return f"data:image/jpeg;base64,{new_b64}"


def _post_json(
    url: str, api_key: str, payload: dict[str, object], timeout: int
) -> dict[str, object]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "User-Agent": "ParkOpticon/1.0 (+python-urllib)",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        hint = ""
        if exc.code == 403 and "1010" in details:
            hint = (
                " (xAI returned 403/1010 access denied; verify XAI_API_KEY scope and that the "
                "network/IP is allowed to access api.x.ai images endpoints)"
            )
        raise GrokImageAPIError(
            f"Grok image API HTTP {exc.code} for {url}: {details[:500]}{hint}",
            status_code=exc.code,
        ) from exc
    except error.URLError as exc:
        raise GrokImageAPIError(
            f"Grok image API request failed for {url}: {exc}"
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GrokImageAPIError(
            f"Grok image API returned non-JSON response: {raw[:300]}"
        ) from exc


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = request.Request(
        url,
        headers={"accept": "*/*", "User-Agent": "ParkOpticon/1.0 (+python-urllib)"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        raise GrokImageAPIError(f"Failed to download generated image: {exc}") from exc


def generate_image(
    *,
    api_key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    input_images: list[str] | None = None,
    output_format: str = "jpeg",
    base_url: str = DEFAULT_BASE_URL,
    timeout_seconds: int = 120,
) -> bytes:
    _ = output_format

    if not api_key:
        raise GrokImageAPIError("Grok API key is required")
    if not prompt:
        raise GrokImageAPIError("Prompt is required")

    images = [value for value in (input_images or []) if value]

    base_payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "response_format": "b64_json",
    }

    if images:
        endpoint = f"{base_url.rstrip('/')}/v1/images/edits"

        images_uris = [
            _maybe_reduce_data_uri(_to_data_uri(value)) for value in images[:3]
        ]
        payload_candidates: list[dict[str, object]] = []

        if len(images_uris) == 1:
            image_uri = images_uris[0]
            payload_candidates.append({**base_payload, "image_url": image_uri})
            payload_candidates.append(
                {
                    **base_payload,
                    "image": {"url": image_uri, "type": "image_url"},
                }
            )
            payload_candidates.append(
                {
                    **base_payload,
                    "images": [{"url": image_uri, "type": "image_url"}],
                }
            )
        else:
            image_objects = [
                {"url": value, "type": "image_url"} for value in images_uris
            ]
            payload_candidates.append({**base_payload, "image_urls": images_uris})
            payload_candidates.append({**base_payload, "images": image_objects})

        last_error: GrokImageAPIError | None = None
        response: dict[str, object] | None = None
        for payload in payload_candidates:
            try:
                response = _post_json(
                    endpoint, api_key, payload, timeout=timeout_seconds
                )
                break
            except GrokImageAPIError as exc:
                last_error = exc
                if exc.status_code not in {400, 403, 422}:
                    raise

        if response is None:
            if last_error is not None:
                raise last_error
            raise GrokImageAPIError("All Grok image edit payload attempts failed")
    else:
        endpoint = f"{base_url.rstrip('/')}/v1/images/generations"
        response = _post_json(
            endpoint,
            api_key,
            base_payload,
            timeout=timeout_seconds,
        )

    data_list = response.get("data")
    if not isinstance(data_list, list) or not data_list:
        raise GrokImageAPIError(f"No image data returned by Grok API: {response}")

    first = data_list[0]
    if not isinstance(first, dict):
        raise GrokImageAPIError(f"Unexpected Grok image payload shape: {response}")

    b64_value = first.get("b64_json")
    if isinstance(b64_value, str) and b64_value:
        try:
            return base64.b64decode(b64_value)
        except Exception as exc:
            raise GrokImageAPIError("Failed to decode b64 image from Grok API") from exc

    url_value = first.get("url")
    if isinstance(url_value, str) and url_value:
        return _download_bytes(url_value, timeout=timeout_seconds)

    raise GrokImageAPIError(f"No usable image payload in Grok response: {response}")
