"""Core sticker generation orchestration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from .ai import GoogleImageGenerator
from .config import Settings
from .errors import AudioTranscriptionError, ConfigurationError
from .image_processing import convert_png_to_escpos_raster_bytes, convert_to_coloring_sticker
from .prompting import optimize_kid_prompt, sanitize_prompt
from .speech import WhisperAudioTranscriber
from .together_ai import TogetherImageGenerator

logger = logging.getLogger(__name__)


class ImageGenerator(Protocol):
    def generate_image(self, *, prompt: str, output_path: Path) -> Path: ...


class StickerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._generator: ImageGenerator | None = None
        self._transcriber: WhisperAudioTranscriber | None = None

    def generate_sticker_from_prompt(self, prompt: str) -> Path:
        clean_prompt = sanitize_prompt(
            prompt,
            max_words=self._settings.max_prompt_words,
            max_chars=self._settings.max_prompt_chars,
        )
        cache_path = self._cache_path(clean_prompt)
        if cache_path.exists():
            logger.info("Cache hit for prompt hash=%s", cache_path.stem)
            return cache_path

        logger.info("Cache miss for prompt hash=%s; generating sticker.", cache_path.stem)
        optimized_prompt = optimize_kid_prompt(clean_prompt)

        source_path = self._settings.output_dir / f"ai_{uuid4().hex}.png"
        processed_path = self._settings.output_dir / f"sticker_{uuid4().hex}.png"

        try:
            self._get_generator().generate_image(prompt=optimized_prompt, output_path=source_path)
            convert_to_coloring_sticker(
                source_path,
                processed_path,
                target_width=self._settings.target_width,
                thermal_white_target=self._settings.thermal_white_target,
            )
            processed_path.replace(cache_path)
            logger.info("Generated new sticker at %s", cache_path)
        finally:
            for temp_path in (source_path, processed_path):
                if temp_path.exists() and temp_path != cache_path:
                    temp_path.unlink(missing_ok=True)

        return cache_path

    def transcribe_prompt_from_audio(self, audio_path: Path) -> str:
        try:
            transcript = self._get_transcriber().transcribe_audio_wav(audio_path=audio_path)
        except AudioTranscriptionError as exc:
            # If speech was too quiet/short and Whisper returns nothing, continue with safe fallback.
            if "empty transcript" in str(exc).lower():
                fallback_prompt = sanitize_prompt(
                    "",
                    max_words=self._settings.max_prompt_words,
                    max_chars=self._settings.max_prompt_chars,
                )
                logger.warning("Whisper returned empty transcript. Falling back to prompt: %s", fallback_prompt)
                return fallback_prompt
            raise

        clean_prompt = sanitize_prompt(
            transcript,
            max_words=self._settings.max_prompt_words,
            max_chars=self._settings.max_prompt_chars,
        )
        logger.info("Transcribed prompt from audio: %s", clean_prompt)
        return clean_prompt

    def generate_sticker_from_audio(self, audio_path: Path) -> tuple[str, Path]:
        prompt = self.transcribe_prompt_from_audio(audio_path=audio_path)
        sticker_path = self.generate_sticker_from_prompt(prompt)
        return prompt, sticker_path

    def generate_sticker_thermal_bytes_from_audio(self, audio_path: Path) -> tuple[str, bytes]:
        prompt, sticker_path = self.generate_sticker_from_audio(audio_path=audio_path)
        raster_bytes = convert_png_to_escpos_raster_bytes(
            sticker_path,
            target_width=self._settings.target_width,
        )
        return prompt, raster_bytes

    def generate_sticker_thermal_bytes_from_prompt(self, prompt: str) -> bytes:
        sticker_path = self.generate_sticker_from_prompt(prompt)
        return convert_png_to_escpos_raster_bytes(
            sticker_path,
            target_width=self._settings.target_width,
        )

    def _get_generator(self) -> ImageGenerator:
        if self._generator is None:
            provider = self._settings.image_provider.strip().lower()
            if provider in {"google", "gemini"}:
                self._generator = GoogleImageGenerator(
                    api_key=self._settings.google_api_key,
                    model=self._settings.image_model,
                )
            elif provider in {"together", "together-ai", "together_ai"}:
                self._generator = TogetherImageGenerator(
                    api_key=self._settings.together_api_key,
                    model=self._settings.together_image_model,
                )
            else:
                raise ConfigurationError(
                    f"Unsupported IMAGE_PROVIDER '{self._settings.image_provider}'. "
                    "Use 'google' or 'together'."
                )
        return self._generator

    def _get_transcriber(self) -> WhisperAudioTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperAudioTranscriber(
                model_size=self._settings.whisper_model_size,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
        return self._transcriber

    def _cache_path(self, prompt: str) -> Path:
        digest = hashlib.md5(
            f"{self._settings.cache_version}:{prompt}".encode("utf-8")
        ).hexdigest()
        return self._settings.output_dir / f"{digest}.png"
