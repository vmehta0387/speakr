"""Google GenAI image generation adapter."""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigurationError, ImageGenerationError, ModelAccessError, QuotaExceededError

_GENAI_IMPORT_ERROR = None
try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - exercised at runtime if dependency missing.
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = exc


class GoogleImageGenerator:
    def __init__(self, *, api_key: str | None, model: str) -> None:
        if _GENAI_IMPORT_ERROR is not None:
            raise ConfigurationError(
                "Missing dependency 'google-genai'. Install with: pip install google-genai"
            ) from _GENAI_IMPORT_ERROR

        if not api_key:
            raise ConfigurationError("Missing API key. Set GOOGLE_API_KEY (or GEMINI_API_KEY).")
        if not model:
            raise ConfigurationError("GOOGLE_IMAGE_MODEL cannot be empty.")

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_image(self, *, prompt: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if _is_gemini_model(self._model):
                self._generate_image_with_gemini(prompt=prompt, output_path=output_path)
            else:
                self._generate_image_with_imagen(prompt=prompt, output_path=output_path)
        except (ImageGenerationError, ModelAccessError, QuotaExceededError):
            raise
        except Exception as exc:
            self._raise_generation_error(exc)

        return output_path

    def _generate_image_with_imagen(self, *, prompt: str, output_path: Path) -> None:
        result = self._client.models.generate_images(
            model=self._model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )

        generated_images = getattr(result, "generated_images", None)
        if not generated_images:
            raise ImageGenerationError("Image generation failed: no images were returned.")

        generated_image = generated_images[0].image
        if generated_image is None:
            raise ImageGenerationError("Image generation failed: empty image payload.")

        try:
            if hasattr(generated_image, "save"):
                generated_image.save(str(output_path))
            elif getattr(generated_image, "image_bytes", None):
                output_path.write_bytes(generated_image.image_bytes)
            else:
                raise ImageGenerationError("Image generation failed: unsupported image payload type.")
        except ImageGenerationError:
            raise
        except Exception as exc:
            raise ImageGenerationError(f"Failed to write generated image file: {exc}") from exc

    def _generate_image_with_gemini(self, *, prompt: str, output_path: Path) -> None:
        result = self._client.models.generate_content(
            model=self._model,
            contents=[prompt],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )

        parts = getattr(result, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            image_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
            if image_bytes:
                output_path.write_bytes(image_bytes)
                return

        raise ImageGenerationError("Image generation failed: Gemini did not return image bytes.")

    def _raise_generation_error(self, exc: Exception) -> None:
        message = str(exc)
        if _is_quota_error(message):
            raise QuotaExceededError(
                "Image generation quota/rate limit exceeded for this API key. "
                "Check usage limits/billing in Google AI Studio and retry later."
            ) from exc
        if _is_model_access_error(message):
            raise ModelAccessError(
                "Configured Google image model is not available for this account plan. "
                "Set GOOGLE_IMAGE_MODEL to a model your account can access or enable billing."
            ) from exc
        if _is_model_not_found_or_unsupported_error(message):
            raise ImageGenerationError(
                f"Configured image model '{self._model}' is not found or does not support this image API path."
            ) from exc
        raise ImageGenerationError(f"Image generation request failed: {exc}") from exc


def _is_model_access_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "only available on paid plans" in lowered
        or "please upgrade your account" in lowered
        or "permission denied" in lowered
    )


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "resource_exhausted" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "too many requests" in lowered
    )


def _is_model_not_found_or_unsupported_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "not found for api version" in lowered
        or "is not found for api version" in lowered
        or "not supported for predict" in lowered
        or "does not support the requested response modalities" in lowered
    )


def _is_gemini_model(model_name: str) -> bool:
    normalized = model_name.lower().strip()
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    return normalized.startswith("gemini-")

