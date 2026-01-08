"""Base provider interface for LLM transcription and translation."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """Abstract base class for transcription and translation providers."""

    def __init__(self, target_language: str = "English"):
        """Initialize the provider.

        Args:
            target_language: Target language for translation
        """
        self.target_language = target_language
        self._source_language: Optional[str] = None

    @property
    def source_language(self) -> Optional[str]:
        """Get the detected source language from last transcription."""
        return self._source_language

    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (16-bit PCM, mono, 16kHz)

        Returns:
            Transcribed text

        Raises:
            TranscriptionError: If transcription fails
        """
        pass

    @abstractmethod
    def translate(self, text: str) -> str:
        """Translate text to target language.

        If the text is already in the target language, return it unchanged.

        Args:
            text: Text to translate

        Returns:
            Translated text

        Raises:
            TranslationError: If translation fails
        """
        pass

    def transcribe_and_translate(self, audio_data: bytes) -> str:
        """Transcribe and translate audio in one operation.

        This default implementation calls transcribe() then translate().
        Providers may override this for more efficient combined operations.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Translated text
        """
        transcription = self.transcribe(audio_data)
        if not transcription or not transcription.strip():
            return ""
        return self.translate(transcription)


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


class TranslationError(Exception):
    """Raised when translation fails."""

    pass
