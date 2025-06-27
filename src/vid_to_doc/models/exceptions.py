"""Custom exceptions for the vid_to_doc package."""

from typing import Optional


class VidToDocError(Exception):
    """Base exception for all vid_to_doc errors."""
    
    def __init__(self, message: str, details: Optional[str] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(VidToDocError):
    """Raised when there's a configuration error."""
    pass


class AudioExtractionError(VidToDocError):
    """Raised when audio extraction fails."""
    pass


class TranscriptionError(VidToDocError):
    """Raised when transcription fails."""
    pass


class SummarizationError(VidToDocError):
    """Raised when summarization fails."""
    pass


class DocumentationError(VidToDocError):
    """Raised when documentation generation fails."""
    pass


class FileError(VidToDocError):
    """Raised when file operations fail."""
    pass


class APIError(VidToDocError):
    """Raised when API calls fail."""
    pass


class AudioError(VidToDocError):
    """Raised when audio processing fails."""
    pass


class TextProcessingError(VidToDocError):
    """Raised when text processing fails."""
    pass


class YouTubeDownloadError(VidToDocError):
    """Raised when YouTube video download fails."""
    pass 