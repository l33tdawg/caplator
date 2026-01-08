"""Integration tests for the translation pipeline."""

import json
import struct
import pytest
from unittest.mock import MagicMock, patch

from src.utils.config import Config


class TestTranslationPipeline:
    """Integration tests for the audio -> transcribe -> translate pipeline."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a test configuration."""
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "target_language": "Spanish",
                "whisper_model": "tiny",
                "ollama_model": "llama3.2",
            }, f)
        return Config(str(config_path))

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data."""
        return b'\x00\x01' * 16000  # 1 second

    def test_full_pipeline(self, mock_config, sample_audio):
        """Test complete transcription and translation pipeline."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper') as mock_whisper:
            with patch('ollama.Client') as mock_client:
                # Mock Whisper
                mock_model = MagicMock()
                mock_segment = MagicMock()
                mock_segment.text = "Hello world"
                mock_info = MagicMock()
                mock_info.language = "en"
                mock_model.transcribe.return_value = ([mock_segment], mock_info)
                mock_whisper.return_value = mock_model

                # Mock Ollama
                mock_client.return_value.generate.return_value = {
                    "response": "Hola mundo"
                }

                translator = Translator(target_language="Spanish")
                translator._whisper = mock_model
                translator._ollama = mock_client.return_value

                result = translator.transcribe_and_translate(sample_audio)

                assert result == "Hola mundo"

    def test_transcription_only(self, sample_audio):
        """Test transcription without translation (same language)."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper') as mock_whisper:
            with patch('src.providers.translator.Translator._load_ollama'):
                mock_model = MagicMock()
                mock_segment = MagicMock()
                mock_segment.text = "Hello world"
                mock_info = MagicMock()
                mock_info.language = "en"
                mock_model.transcribe.return_value = ([mock_segment], mock_info)
                mock_whisper.return_value = mock_model

                translator = Translator(target_language="English")
                translator._whisper = mock_model
                translator._detected_language = "en"

                # Transcribe
                transcription = translator.transcribe(sample_audio)
                assert transcription == "Hello world"

                # Translate (should return same text)
                translation = translator.translate(transcription)
                assert translation == "Hello world"

    def test_empty_audio_handling(self, sample_audio):
        """Test handling of audio with no speech."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper') as mock_whisper:
            with patch('src.providers.translator.Translator._load_ollama'):
                mock_model = MagicMock()
                mock_info = MagicMock()
                mock_info.language = "en"
                mock_model.transcribe.return_value = ([], mock_info)  # No segments
                mock_whisper.return_value = mock_model

                translator = Translator()
                translator._whisper = mock_model

                result = translator.transcribe(sample_audio)
                assert result == ""

    def test_error_recovery(self, sample_audio):
        """Test error handling and recovery."""
        from src.providers import Translator
        from src.providers.base import TranscriptionError

        with patch('src.providers.translator.Translator._load_whisper') as mock_whisper:
            mock_model = MagicMock()
            mock_model.transcribe.side_effect = Exception("Model error")
            mock_whisper.return_value = mock_model

            translator = Translator()
            translator._whisper = mock_model

            with pytest.raises(TranscriptionError):
                translator.transcribe(sample_audio)


class TestAudioProcessing:
    """Tests for audio processing utilities."""

    def test_wav_header_format(self, sample_audio_data):
        """Test WAV header is correctly formatted."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper'):
            with patch('src.providers.translator.Translator._load_ollama'):
                translator = Translator()

                wav_bytes = translator._create_wav(sample_audio_data)

                # Verify RIFF header
                assert wav_bytes[:4] == b'RIFF'
                assert wav_bytes[8:12] == b'WAVE'
                assert wav_bytes[12:16] == b'fmt '
                assert wav_bytes[36:40] == b'data'

    def test_wav_data_size(self, sample_audio_data):
        """Test WAV data size is correct."""
        from src.providers import Translator

        with patch('src.providers.translator.Translator._load_whisper'):
            with patch('src.providers.translator.Translator._load_ollama'):
                translator = Translator()

                wav_bytes = translator._create_wav(sample_audio_data)

                # Data size is at offset 40-44
                data_size = struct.unpack('<I', wav_bytes[40:44])[0]
                assert data_size == len(sample_audio_data)


class TestConfigIntegration:
    """Tests for configuration with Translator."""

    def test_translator_from_config(self, tmp_path):
        """Test creating Translator from config settings."""
        config_path = tmp_path / "config.json"

        with open(config_path, 'w') as f:
            json.dump({
                "target_language": "French",
                "whisper_model": "base",
                "ollama_model": "mistral",
            }, f)

        config = Config(str(config_path))

        with patch('src.providers.translator.Translator._load_whisper'):
            with patch('src.providers.translator.Translator._load_ollama'):
                from src.providers import Translator

                translator = Translator(
                    target_language=config.get("target_language"),
                    whisper_model=config.get("whisper_model"),
                    ollama_model=config.get("ollama_model"),
                )

                assert translator.target_language == "French"
                assert translator.whisper_model_name == "base"
                assert translator.ollama_model == "mistral"
