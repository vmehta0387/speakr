"""FastAPI app wiring for Speakr."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
import shutil
from urllib.parse import quote
import wave

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings
from .errors import (
    AudioTranscriptionError,
    ConfigurationError,
    ImageGenerationError,
    ModelAccessError,
    QuotaExceededError,
    SpeakrError,
)
from .logging_utils import configure_logging
from .schemas import ErrorResponse, PromptRequest, StatusResponse, TranscriptResponse
from .service import StickerService

logger = logging.getLogger(__name__)

AUDIO_WAV_CHANNELS = 1
AUDIO_WAV_SAMPLE_WIDTH_BYTES = 2


def _convert_raw_pcm_to_wav(raw_path: Path, wav_path: Path, sample_rate: int) -> None:
    raw_audio = raw_path.read_bytes()
    if not raw_audio:
        raise ValueError("Empty audio payload.")

    if len(raw_audio) % AUDIO_WAV_SAMPLE_WIDTH_BYTES != 0:
        raise ValueError(
            "Invalid PCM payload length. Expected 16-bit PCM byte length to be divisible by 2."
        )

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(AUDIO_WAV_CHANNELS)
        wav_file.setsampwidth(AUDIO_WAV_SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_audio)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings.from_env()
    except ValueError as exc:
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_service() -> StickerService:
    return StickerService(settings=get_settings())


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="Speakr API",
        version="1.0.0",
        description="Generate thermal-printer-friendly kids stickers from prompts.",
    )
    audio_dir = Path("audio")
    audio_dir.mkdir(parents=True, exist_ok=True)

    @app.exception_handler(SpeakrError)
    async def speakr_error_handler(_, exc: SpeakrError) -> JSONResponse:
        logger.exception("Speakr request failed: %s", exc)
        status_code = 500
        if isinstance(exc, ModelAccessError):
            status_code = 402
        elif isinstance(exc, QuotaExceededError):
            status_code = 429
        elif isinstance(exc, AudioTranscriptionError):
            status_code = 422
        elif isinstance(exc, ImageGenerationError):
            status_code = 502
        elif isinstance(exc, ConfigurationError):
            status_code = 500
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.get("/", response_model=StatusResponse)
    def root() -> StatusResponse:
        return StatusResponse(status="Speakr AI Server Running")

    @app.get("/health", response_model=StatusResponse)
    def health() -> StatusResponse:
        return StatusResponse(status="ok")

    @app.post(
        "/generate-sticker",
        response_class=FileResponse,
        responses={500: {"model": ErrorResponse}},
    )
    def generate_sticker(data: PromptRequest) -> FileResponse:
        file_path = get_service().generate_sticker_from_prompt(data.prompt)
        return FileResponse(path=file_path, media_type="image/png", filename=file_path.name)

    @app.post(
        "/generate-sticker-thermal",
        responses={500: {"model": ErrorResponse}},
    )
    def generate_sticker_thermal(data: PromptRequest) -> Response:
        raster_bytes = get_service().generate_sticker_thermal_bytes_from_prompt(data.prompt)
        return Response(content=raster_bytes, media_type="application/octet-stream")

    @app.post("/upload-audio", response_model=StatusResponse)
    async def upload_audio(
        request: Request,
        file: UploadFile | None = File(default=None),
    ) -> StatusResponse:
        raw_path = audio_dir / "voice.raw"
        wav_path = audio_dir / "voice.wav"
        content_type = request.headers.get("content-type", "")

        if file is not None:
            try:
                with raw_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                await file.close()
        else:
            if content_type.startswith("multipart/form-data"):
                raise HTTPException(status_code=400, detail="Missing file field in form-data payload.")

            payload = await request.body()
            if not payload:
                raise HTTPException(status_code=400, detail="Empty audio payload.")

            with raw_path.open("wb") as buffer:
                buffer.write(payload)

        try:
            _convert_raw_pcm_to_wav(
                raw_path=raw_path,
                wav_path=wav_path,
                sample_rate=settings.audio_sample_rate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info(
            "Audio received and saved to %s, converted to %s at %d Hz (content-type=%s)",
            raw_path,
            wav_path,
            settings.audio_sample_rate,
            content_type,
        )
        return StatusResponse(status="audio_received")

    @app.post(
        "/transcribe-audio",
        response_model=TranscriptResponse,
        responses={500: {"model": ErrorResponse}},
    )
    def transcribe_audio() -> TranscriptResponse:
        wav_path = audio_dir / "voice.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="No audio file found. Upload audio first.")

        transcript = get_service().transcribe_prompt_from_audio(wav_path)
        return TranscriptResponse(status="transcribed", transcript=transcript)

    @app.post(
        "/generate-sticker-from-audio",
        response_class=FileResponse,
        responses={500: {"model": ErrorResponse}},
    )
    def generate_sticker_from_audio() -> FileResponse:
        wav_path = audio_dir / "voice.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="No audio file found. Upload audio first.")

        transcript, file_path = get_service().generate_sticker_from_audio(wav_path)
        return FileResponse(
            path=file_path,
            media_type="image/png",
            filename=file_path.name,
            headers={"X-Transcribed-Prompt": quote(transcript, safe="")},
        )

    @app.post(
        "/generate-sticker-from-audio-thermal",
        responses={500: {"model": ErrorResponse}},
    )
    def generate_sticker_from_audio_thermal() -> Response:
        wav_path = audio_dir / "voice.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="No audio file found. Upload audio first.")

        transcript, raster_bytes = get_service().generate_sticker_thermal_bytes_from_audio(wav_path)
        return Response(
            content=raster_bytes,
            media_type="application/octet-stream",
            headers={"X-Transcribed-Prompt": quote(transcript, safe="")},
        )

    return app
