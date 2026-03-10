"""Together AI image generation adapter."""

from __future__ import annotations

import base64
import requests
from pathlib import Path

from .errors import ConfigurationError, ImageGenerationError

_TOGETHER_IMPORT_ERROR = None
try:
    from together import Together
except ImportError as exc:  # pragma: no cover - exercised at runtime if dependency missing.
    Together = None
    _TOGETHER_IMPORT_ERROR = exc


class TogetherImageGenerator:
    def __init__(self, *, api_key: str | None, model: str) -> None:
        if _TOGETHER_IMPORT_ERROR is not None:
            raise ConfigurationError(
                "Missing dependency 'together'. Install with: pip install together"
            ) from _TOGETHER_IMPORT_ERROR
        if not api_key:
            raise ConfigurationError("Missing TOGETHER_API_KEY.")
        if not model:
            raise ConfigurationError("TOGETHER_IMAGE_MODEL cannot be empty.")

        self._client = Together(api_key=api_key)
        self._model = model

    def generate_image(self, *, prompt: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self._client.images.generate(
                prompt=prompt,
                model=self._model,
            )
        except Exception as exc:
            raise ImageGenerationError(f"Together AI image generation failed: {exc}") from exc

        data = getattr(response, "data", None) or []
        if not data:
            raise ImageGenerationError("Together AI did not return any images.")

        item = data[0]
        b64_json = getattr(item, "b64_json", None) or getattr(item, "b64", None)
        if b64_json:
            try:
                output_path.write_bytes(base64.b64decode(b64_json))
                return output_path
            except Exception as exc:
                raise ImageGenerationError(f"Failed to write Together AI image: {exc}") from exc

        url = getattr(item, "url", None) or getattr(item, "image_url", None)
        if url:
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                output_path.write_bytes(resp.content)
                return output_path
            except Exception as exc:
                raise ImageGenerationError(f"Failed to download Together AI image: {exc}") from exc

        raise ImageGenerationError(
            "Together AI response missing image data (expected b64_json/b64 or url/image_url)."
        )

        return output_path
