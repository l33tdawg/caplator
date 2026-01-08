"""Tests for the Translator provider."""

import pytest
from unittest.mock import MagicMock, patch

from src.providers.base import BaseProvider, TranscriptionError, TranslationError


class TestBaseProvider:
    """Test cases for BaseProvider abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()

    def test_transcribe_and_translate_default(self):
        """Test default transcribe_and_translate implementation."""
        class TestProvider(BaseProvider):
            def transcribe(self, audio_data):
                return "Hello"

            def translate(self, text):
                return "Hola"

        provider = TestProvider(target_language="Spanish")
        result = provider.transcribe_and_translate(b"audio")
        assert result == "Hola"

    def test_transcribe_and_translate_empty(self):
        """Test transcribe_and_translate with empty transcription."""
        class TestProvider(BaseProvider):
            def transcribe(self, audio_data):
                return ""

            def translate(self, text):
                return "Should not be called"

        provider = TestProvider()
        result = provider.transcribe_and_translate(b"audio")
        assert result == ""


class TestTranslator:
    """Test cases for the main Translator class."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_initialization(self, mock_ollama, mock_whisper):
        """Test Translator initialization."""
        from src.providers import Translator

        translator = Translator(
            target_language="Spanish",
            whisper_model="small",
            ollama_model="llama3.2",
        )

        assert translator.target_language == "Spanish"
        assert translator.whisper_model_name == "small"
        assert translator.ollama_model == "llama3.2"

    @patch('src.providers.translator.Translator._load_ollama')
    def test_transcribe(self, mock_ollama, sample_audio_data):
        """Test transcription with mocked Whisper."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper') as mock_load:
            # Mock Whisper model
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_segment.text = "Hello world"
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_model.transcribe.return_value = ([mock_segment], mock_info)
            mock_load.return_value = mock_model

            translator = Translator()
            translator._whisper = mock_model

            result = translator.transcribe(sample_audio_data)

            assert result == "Hello world"
            assert translator._detected_language == "en"

    @patch('src.providers.translator.Translator._load_whisper')
    def test_translate(self, mock_whisper):
        """Test translation with mocked Ollama."""
        from src.providers import Translator

        with patch('ollama.Client') as mock_client:
            mock_client.return_value.generate.return_value = {
                "response": "Hola mundo"
            }

            translator = Translator(target_language="Spanish")
            translator._ollama = mock_client.return_value
            translator._detected_language = "en"

            result = translator.translate("Hello world")

            assert result == "Hola mundo"

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_translate_same_language(self, mock_ollama, mock_whisper):
        """Test that same-language text is returned unchanged."""
        from src.providers import Translator

        translator = Translator(target_language="English")
        translator._detected_language = "en"

        result = translator.translate("Hello world")

        assert result == "Hello world"

    @patch('src.providers.translator.Translator._load_whisper')
    def test_translate_empty_text(self, mock_whisper):
        """Test that empty text returns empty string."""
        from src.providers import Translator

        translator = Translator()

        assert translator.translate("") == ""
        assert translator.translate("   ") == ""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_clean_translation(self, mock_ollama, mock_whisper):
        """Test translation output cleaning."""
        from src.providers import Translator

        translator = Translator()

        # Test prefix removal
        assert translator._clean_translation("Translation: Hola") == "Hola"
        assert translator._clean_translation("Here's the translation: Bonjour") == "Bonjour"

        # Test quote removal
        assert translator._clean_translation('"Hola mundo"') == "Hola mundo"
        assert translator._clean_translation("'Bonjour'") == "Bonjour"

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_language_code_mapping(self, mock_ollama, mock_whisper):
        """Test language name to code conversion."""
        from src.providers import Translator

        translator = Translator()

        assert translator._get_language_code("English") == "en"
        assert translator._get_language_code("Spanish") == "es"
        assert translator._get_language_code("Japanese") == "ja"
        assert translator._get_language_code("Chinese (Simplified)") == "zh"

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_create_wav(self, mock_ollama, mock_whisper, sample_audio_data):
        """Test WAV file creation."""
        from src.providers import Translator

        translator = Translator()
        wav_bytes = translator._create_wav(sample_audio_data)

        # Check WAV header
        assert wav_bytes[:4] == b'RIFF'
        assert wav_bytes[8:12] == b'WAVE'
        assert wav_bytes[12:16] == b'fmt '
        assert wav_bytes[36:40] == b'data'

    @patch('src.providers.translator.Translator._load_whisper')
    def test_check_ollama_success(self, mock_whisper):
        """Test Ollama health check when model is available."""
        from src.providers import Translator

        with patch('ollama.Client') as mock_client:
            mock_client.return_value.list.return_value = {
                "models": [{"name": "llama3.2"}]
            }

            translator = Translator(ollama_model="llama3.2")
            translator._ollama = mock_client.return_value

            ok, msg = translator.check_ollama()

            assert ok is True
            assert "llama3.2" in msg

    @patch('src.providers.translator.Translator._load_whisper')
    def test_check_ollama_model_missing(self, mock_whisper):
        """Test Ollama health check when model is missing."""
        from src.providers import Translator

        with patch('ollama.Client') as mock_client:
            mock_client.return_value.list.return_value = {
                "models": [{"name": "mistral"}]
            }

            translator = Translator(ollama_model="llama3.2")
            translator._ollama = mock_client.return_value

            ok, msg = translator.check_ollama()

            assert ok is False
            assert "not found" in msg

    @patch('src.providers.translator.Translator._load_whisper')
    def test_check_ollama_not_running(self, mock_whisper):
        """Test Ollama health check when Ollama is not running."""
        from src.providers import Translator

        with patch('ollama.Client') as mock_client:
            mock_client.return_value.list.side_effect = Exception("Connection refused")

            translator = Translator()
            translator._ollama = mock_client.return_value

            ok, msg = translator.check_ollama()

            assert ok is False
            assert "not running" in msg.lower() or "error" in msg.lower()
