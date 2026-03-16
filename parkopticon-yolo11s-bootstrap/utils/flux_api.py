import base64
import json
import time
from typing import Optional, cast
from urllib import error, request


DEFAULT_BASE_URL = "https://api.bfl.ai"
DEFAULT_MODEL = "black-forest-labs/flux.2-pro"


class FluxAPIError(RuntimeError):
    pass


def model_to_endpoint(model: str) -> str:
    value = (model or "").strip().lower()
    if not value:
        return "flux-2-pro"

    aliases = {
        "black-forest-labs/flux.2-pro": "flux-2-pro",
        "flux.2-pro": "flux-2-pro",
        "flux-2-pro": "flux-2-pro",
        "flux2-pro": "flux-2-pro",
    }
    if value in aliases:
        return aliases[value]

    if value.startswith("black-forest-labs/"):
        value = value.split("/", 1)[1]

    value = value.replace(".", "-").replace("_", "-")
    if value.startswith("v1/"):
        value = value[3:]
    if value.startswith("/v1/"):
        value = value[4:]
    if value.startswith("/"):
        value = value[1:]
    return value


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def _http_json(
    method: str,
    url: str,
    api_key: str,
    payload: Optional[dict[str, object]] = None,
    timeout: int = 120,
) -> dict[str, object]:
    headers = {
        "accept": "application/json",
        "x-key": api_key,
    }
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise FluxAPIError(
            f"Flux API HTTP {exc.code} for {url}: {details[:500]}"
        ) from exc
    except error.URLError as exc:
        raise FluxAPIError(f"Flux API request failed for {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FluxAPIError(f"Flux API returned non-JSON response: {raw[:300]}") from exc


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    req = request.Request(url, headers={"accept": "*/*"}, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        raise FluxAPIError(f"Failed to download generated image: {exc}") from exc


def generate_image(
    *,
    api_key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    input_images: Optional[list[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_format: str = "jpeg",
    seed: Optional[int] = None,
    safety_tolerance: int = 2,
    disable_pup: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    poll_interval_seconds: float = 0.5,
    max_wait_seconds: float = 120.0,
    timeout_seconds: int = 120,
) -> bytes:
    if not api_key:
        raise FluxAPIError("Flux API key is required")
    if not prompt:
        raise FluxAPIError("Prompt is required")

    endpoint = model_to_endpoint(model)
    submit_url = f"{base_url.rstrip('/')}/v1/{endpoint}"

    payload: dict[str, object] = {
        "prompt": prompt,
        "output_format": output_format,
        "safety_tolerance": safety_tolerance,
        "disable_pup": bool(disable_pup),
    }
    if width and width > 0:
        payload["width"] = int(width)
    if height and height > 0:
        payload["height"] = int(height)
    if seed is not None:
        payload["seed"] = int(seed)

    image_inputs = input_images or []
    if not image_inputs:
        raise FluxAPIError("FLUX.2 image editing requires at least one input image")
    for index, image_data in enumerate(image_inputs[:8], start=1):
        key = "input_image" if index == 1 else f"input_image_{index}"
        payload[key] = image_data

    submit_response = _http_json(
        "POST",
        submit_url,
        api_key,
        payload=payload,
        timeout=timeout_seconds,
    )

    polling_value = submit_response.get("polling_url")
    polling_url: str = polling_value if isinstance(polling_value, str) else ""
    if not polling_url:
        raise FluxAPIError(
            f"Flux submit response missing polling_url: {submit_response}"
        )

    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        status_response = _http_json(
            "GET",
            polling_url,
            api_key,
            timeout=timeout_seconds,
        )
        status = str(status_response.get("status", "")).strip()
        if status == "Ready":
            result_value = status_response.get("result")
            result = (
                cast(dict[str, object], result_value)
                if isinstance(result_value, dict)
                else {}
            )
            sample_value = result.get("sample")
            sample_url = sample_value if isinstance(sample_value, str) else ""
            if not sample_url:
                raise FluxAPIError(f"Flux result missing sample URL: {status_response}")
            return _download_bytes(sample_url, timeout=timeout_seconds)

        if status in {
            "Error",
            "Failed",
            "Task not found",
            "Request Moderated",
            "Content Moderated",
        }:
            raise FluxAPIError(
                f"Flux generation failed with status {status}: {status_response}"
            )

        time.sleep(poll_interval_seconds)

    raise FluxAPIError("Timed out waiting for Flux generation result")
