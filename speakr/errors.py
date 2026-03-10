"""Domain-specific exceptions for Speakr."""


class SpeakrError(Exception):
    """Base error for all handled service failures."""


class ConfigurationError(SpeakrError):
    """Raised when required configuration is missing or invalid."""


class ImageGenerationError(SpeakrError):
    """Raised when upstream AI image generation fails."""


class ModelAccessError(ImageGenerationError):
    """Raised when the configured image model is not accessible for the account."""


class QuotaExceededError(ImageGenerationError):
    """Raised when the API key/project has exhausted available image generation quota."""


class ImageProcessingError(SpeakrError):
    """Raised when image conversion for thermal printing fails."""


class AudioTranscriptionError(SpeakrError):
    """Raised when upstream AI speech-to-text transcription fails."""
