"""Environment-driven configuration for Speakr."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _read_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got '{raw}'.") from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {value}.")
    return value


def _read_ratio(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got '{raw}'.") from exc

    if value <= 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    return value


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    image_provider: str
    image_model: str
    together_api_key: str | None
    together_image_model: str
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    audio_sample_rate: int
    cache_version: str
    target_width: int
    google_api_key: str | None
    max_prompt_words: int
    max_prompt_chars: int
    thermal_white_target: float

    @classmethod
    def from_env(cls) -> "Settings":
        output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            output_dir=output_dir,
            image_provider=os.getenv("IMAGE_PROVIDER", "google"),
            image_model=os.getenv("GOOGLE_IMAGE_MODEL", "imagen-4.0-fast-generate-001"),
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            together_image_model=os.getenv("TOGETHER_IMAGE_MODEL", "Lykon/DreamShaper"),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            audio_sample_rate=_read_positive_int("AUDIO_SAMPLE_RATE", 16_000),
            cache_version=os.getenv("CACHE_VERSION", "v3"),
            target_width=_read_positive_int("TARGET_WIDTH", 384),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            max_prompt_words=_read_positive_int("MAX_PROMPT_WORDS", 12),
            max_prompt_chars=_read_positive_int("MAX_PROMPT_CHARS", 80),
            thermal_white_target=_read_ratio("THERMAL_WHITE_TARGET", 0.88),
        )
