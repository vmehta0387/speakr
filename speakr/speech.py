"""Local speech-to-text adapters."""

from __future__ import annotations

import logging
from pathlib import Path

from .errors import AudioTranscriptionError, ConfigurationError

_FASTER_WHISPER_IMPORT_ERROR = None
try:
    from faster_whisper import WhisperModel
except ImportError as exc:  # pragma: no cover - exercised at runtime if dependency missing.
    WhisperModel = None
    _FASTER_WHISPER_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)


class WhisperAudioTranscriber:
    def __init__(self, *, model_size: str, device: str, compute_type: str) -> None:
        if _FASTER_WHISPER_IMPORT_ERROR is not None:
            raise ConfigurationError(
                "Missing dependency 'faster-whisper'. Install with: pip install faster-whisper"
            ) from _FASTER_WHISPER_IMPORT_ERROR
        if not model_size:
            raise ConfigurationError("WHISPER_MODEL_SIZE cannot be empty.")
        if not device:
            raise ConfigurationError("WHISPER_DEVICE cannot be empty.")
        if not compute_type:
            raise ConfigurationError("WHISPER_COMPUTE_TYPE cannot be empty.")

        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def transcribe_audio_wav(self, *, audio_path: Path) -> str:
        if not audio_path.exists():
            raise AudioTranscriptionError(f"Audio file not found: {audio_path}")
        if audio_path.stat().st_size == 0:
            raise AudioTranscriptionError("Audio file is empty.")

        model = self._get_model()
        try:
            segments, _ = model.transcribe(
                str(audio_path),
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False,
                beam_size=1,
            )
        except Exception as exc:
            raise AudioTranscriptionError(f"Whisper transcription failed: {exc}") from exc

        transcript = " ".join(
            segment.text.strip() for segment in segments if getattr(segment, "text", "").strip()
        ).strip()
        if not transcript:
            raise AudioTranscriptionError("Whisper transcription failed: empty transcript returned.")

        return transcript

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' (device=%s, compute_type=%s).",
                self._model_size,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model
