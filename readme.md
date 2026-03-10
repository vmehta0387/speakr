## Speakr API

Speakr turns kid-friendly prompts into thermal-printer-ready black/white sticker images.

### Project layout

```text
speakr/
  api.py              # FastAPI routes and exception handling
  service.py          # Sticker generation orchestration + cache
  ai.py               # Google GenAI adapter
  together_ai.py      # Together AI adapter (optional backend)
  speech.py           # Whisper speech-to-text adapter
  image_processing.py # Thermal printer conversion pipeline
  prompting.py        # Prompt sanitization and style policy
  config.py           # Environment-driven settings
  errors.py           # Domain exception types
server.py             # ASGI entrypoint (uvicorn target)
```

### Installation

```bash
pip install -r requirements.txt
```

Optional for Together AI backend:

```bash
pip install together
```

### Environment setup

```bash
copy .env.example .env
```

Minimum required variable in `.env` when `IMAGE_PROVIDER=google`:

```env
GOOGLE_API_KEY=your_google_ai_api_key
```

Default model:

```env
IMAGE_PROVIDER=google
GOOGLE_IMAGE_MODEL=imagen-4.0-fast-generate-001

# Together AI backend (only used when IMAGE_PROVIDER=together)
TOGETHER_API_KEY=your_together_api_key
TOGETHER_IMAGE_MODEL=Lykon/DreamShaper

WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
AUDIO_SAMPLE_RATE=16000
```

### Run locally

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

### API endpoints

- `GET /` basic server status
- `GET /health` health check endpoint
- `POST /generate-sticker` generate/return PNG sticker from prompt
- `POST /generate-sticker-thermal` generate/return ESC/POS raster bytes (`application/octet-stream`)
- `POST /upload-audio` upload ESP32 audio payload; stores `audio/voice.raw` and auto-converts to `audio/voice.wav`
- `POST /transcribe-audio` transcribe `audio/voice.wav` into a safe sticker prompt
- `POST /generate-sticker-from-audio` transcribe `audio/voice.wav` and return sticker PNG
- `POST /generate-sticker-from-audio-thermal` transcribe `audio/voice.wav` and return ESC/POS raster bytes (`application/octet-stream`)

Request body:

```json
{
  "prompt": "cute baby tiger waving"
}
```

Audio upload request formats:

- `multipart/form-data` with field name `file`:
  `curl -X POST -F "file=@voice.raw" http://localhost:8000/upload-audio`
- raw bytes body (ESP32-friendly), usually `Content-Type: application/octet-stream`:
  `curl -X POST --data-binary "@voice.raw" -H "Content-Type: application/octet-stream" http://localhost:8000/upload-audio`

WAV conversion notes for `/upload-audio`:

- expects raw PCM16 little-endian input
- writes WAV as mono, 16-bit, sample rate from `AUDIO_SAMPLE_RATE` (`audio/voice.wav`)

Whisper notes:

- uses local open-source Whisper via `faster-whisper`
- first transcription can take longer because the model is downloaded once
- if Whisper returns an empty transcript (very quiet/short audio), the service falls back to a safe prompt

Audio-to-sticker flow:

1. `POST /upload-audio`
2. `POST /transcribe-audio` (optional, to inspect prompt text)
3. `POST /generate-sticker-from-audio` (returns PNG)
4. `POST /generate-sticker-from-audio-thermal` (returns ESC/POS raster bytes)

### Production notes

- Uses deterministic prompt-hash cache in `output/`
- Uses temporary unique filenames to avoid request collisions
- Includes centralized domain error handling
- Returns `402 Payment Required` when the configured image model is not accessible for the account plan
- Returns `429 Too Many Requests` when image-generation quota/rate limits are exceeded
