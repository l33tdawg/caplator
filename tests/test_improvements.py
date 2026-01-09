"""Tests for the v1.1 accuracy and UX improvements."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call


class TestBeamSizeConfiguration:
    """Test cases for configurable beam size."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_default_beam_size(self, mock_ollama, mock_whisper):
        """Test that default beam size is 3."""
        from src.providers import Translator
        
        translator = Translator()
        assert translator.beam_size == 3
        assert translator.best_of == 2

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_custom_beam_size(self, mock_ollama, mock_whisper):
        """Test custom beam size configuration."""
        from src.providers import Translator
        
        translator = Translator(beam_size=5, best_of=3)
        assert translator.beam_size == 5
        assert translator.best_of == 3

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_context_buffer_size(self, mock_ollama, mock_whisper):
        """Test that context buffer is now 7 (increased from 3)."""
        from src.providers import Translator
        
        translator = Translator()
        assert translator._max_context == 7


class TestHighQualityResampling:
    """Test cases for scipy-based audio resampling."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_resample_audio_basic(self, mock_ollama, mock_whisper):
        """Test basic audio resampling functionality."""
        from src.providers import Translator
        
        translator = Translator()
        
        # Create 1 second of audio at 44100 Hz
        sample_rate_in = 44100
        sample_rate_out = 16000
        duration = 0.1  # 100ms
        num_samples = int(sample_rate_in * duration)
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, num_samples)
        audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
        audio_bytes = audio.tobytes()
        
        # Resample
        resampled = translator._resample_audio(audio_bytes, sample_rate_in, sample_rate_out)
        
        # Check output length is correct
        expected_samples = int(num_samples * sample_rate_out / sample_rate_in)
        actual_samples = len(resampled) // 2  # 2 bytes per int16 sample
        
        # Allow small variance due to resampling
        assert abs(actual_samples - expected_samples) <= 2

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_resample_preserves_dtype(self, mock_ollama, mock_whisper):
        """Test that resampling preserves int16 data type."""
        from src.providers import Translator
        
        translator = Translator()
        
        # Create test audio
        audio = np.array([1000, -1000, 2000, -2000], dtype=np.int16)
        audio_bytes = audio.tobytes()
        
        resampled = translator._resample_audio(audio_bytes, 16000, 8000)
        
        # Convert back and check dtype
        resampled_array = np.frombuffer(resampled, dtype=np.int16)
        assert resampled_array.dtype == np.int16


class TestEnhancedOllamaPrompts:
    """Test cases for enhanced translation prompts."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_build_translation_prompt_contains_examples(self, mock_ollama, mock_whisper):
        """Test that translation prompt includes few-shot examples."""
        from src.providers import Translator
        
        translator = Translator(target_language="English")
        translator._detected_language = "es"
        
        prompt = translator._build_translation_prompt("Hola mundo")
        
        # Check for few-shot examples
        assert "Spanish:" in prompt or "Japanese:" in prompt
        assert "Examples" in prompt
        assert "RULES:" in prompt
        assert "Hola mundo" in prompt

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_build_translation_prompt_includes_languages(self, mock_ollama, mock_whisper):
        """Test that prompt includes source and target languages."""
        from src.providers import Translator
        
        translator = Translator(target_language="French")
        translator._detected_language = "en"
        
        prompt = translator._build_translation_prompt("Hello")
        
        assert "French" in prompt
        assert "en" in prompt or "auto-detected" in prompt


class TestRMSSilenceDetection:
    """Test cases for RMS-based silence detection."""

    def test_rms_silence_detection_silent(self):
        """Test that silent audio is detected as silent."""
        from src.audio.capture import AudioCapture
        
        capture = AudioCapture(silence_threshold=100)
        
        # Create silent audio (all zeros)
        silent_audio = np.zeros(1024, dtype=np.int16).tobytes()
        
        assert capture._is_silent(silent_audio) == True

    def test_rms_silence_detection_loud(self):
        """Test that loud audio is not detected as silent."""
        from src.audio.capture import AudioCapture
        
        capture = AudioCapture(silence_threshold=100)
        
        # Create loud audio
        loud_audio = (np.ones(1024) * 10000).astype(np.int16).tobytes()
        
        assert capture._is_silent(loud_audio) == False

    def test_rms_silence_detection_threshold(self):
        """Test silence detection respects threshold."""
        from src.audio.capture import AudioCapture
        
        # Create audio with known RMS
        audio = np.array([500, -500, 500, -500] * 256, dtype=np.int16)
        audio_bytes = audio.tobytes()
        
        # With low threshold, should not be silent
        capture_low = AudioCapture(silence_threshold=100)
        assert capture_low._is_silent(audio_bytes) == False
        
        # With high threshold, should be silent
        capture_high = AudioCapture(silence_threshold=1000)
        assert capture_high._is_silent(audio_bytes) == True


class TestAudioNormalization:
    """Test cases for audio normalization."""

    def test_normalize_audio_boosts_quiet(self):
        """Test that quiet audio is boosted."""
        from src.audio.capture import AudioCapture
        
        capture = AudioCapture()
        
        # Create quiet audio (low amplitude)
        quiet_audio = np.array([100, -100, 50, -50] * 256, dtype=np.int16)
        quiet_bytes = quiet_audio.tobytes()
        
        normalized = capture._normalize_audio(quiet_bytes, target_level=0.8)
        
        # Convert back and check amplitude increased
        normalized_array = np.frombuffer(normalized, dtype=np.int16).astype(np.float32)
        original_peak = np.max(np.abs(quiet_audio))
        normalized_peak = np.max(np.abs(normalized_array))
        
        assert normalized_peak > original_peak

    def test_normalize_audio_respects_max_gain(self):
        """Test that normalization doesn't exceed max gain of 10x."""
        from src.audio.capture import AudioCapture
        
        capture = AudioCapture()
        
        # Create very quiet audio
        very_quiet = np.array([1, -1, 1, -1] * 256, dtype=np.int16)
        quiet_bytes = very_quiet.tobytes()
        
        normalized = capture._normalize_audio(quiet_bytes, target_level=0.8)
        
        # Convert back and check gain is limited
        normalized_array = np.frombuffer(normalized, dtype=np.int16).astype(np.float32)
        original_peak = np.max(np.abs(very_quiet))
        normalized_peak = np.max(np.abs(normalized_array))
        
        # Gain should be at most 10x
        assert normalized_peak <= original_peak * 10 + 1  # +1 for rounding

    def test_normalize_audio_clips_properly(self):
        """Test that normalization clips to valid int16 range."""
        from src.audio.capture import AudioCapture
        
        capture = AudioCapture()
        
        # Create audio that when boosted would exceed int16 range
        loud_audio = np.array([20000, -20000, 15000, -15000] * 256, dtype=np.int16)
        loud_bytes = loud_audio.tobytes()
        
        normalized = capture._normalize_audio(loud_bytes, target_level=0.8)
        
        # Convert back and verify clipping
        normalized_array = np.frombuffer(normalized, dtype=np.int16)
        
        assert np.max(normalized_array) <= 32767
        assert np.min(normalized_array) >= -32768


class TestQualityProfiles:
    """Test cases for quality profile presets."""

    def test_quality_profiles_exist(self):
        """Test that all quality profiles are defined."""
        from src.utils.config import QUALITY_PROFILES
        
        assert "fast" in QUALITY_PROFILES
        assert "balanced" in QUALITY_PROFILES
        assert "accurate" in QUALITY_PROFILES

    def test_fast_profile_values(self):
        """Test fast profile has low beam_size for speed."""
        from src.utils.config import QUALITY_PROFILES
        
        fast = QUALITY_PROFILES["fast"]
        assert fast["whisper_beam_size"] == 1
        assert fast["whisper_best_of"] == 1
        assert fast["audio.chunk_duration"] == 2.0

    def test_balanced_profile_values(self):
        """Test balanced profile has medium settings."""
        from src.utils.config import QUALITY_PROFILES
        
        balanced = QUALITY_PROFILES["balanced"]
        assert balanced["whisper_beam_size"] == 3
        assert balanced["whisper_best_of"] == 2
        assert balanced["audio.chunk_duration"] == 3.0

    def test_accurate_profile_values(self):
        """Test accurate profile has high beam_size for accuracy."""
        from src.utils.config import QUALITY_PROFILES
        
        accurate = QUALITY_PROFILES["accurate"]
        assert accurate["whisper_beam_size"] == 5
        assert accurate["whisper_best_of"] == 3
        assert accurate["audio.chunk_duration"] == 4.0

    def test_apply_quality_profile(self):
        """Test applying a quality profile updates config."""
        from src.utils.config import Config
        import tempfile
        import os
        
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            config_path = f.name
        
        try:
            config = Config(config_path)
            
            # Apply fast profile
            config.apply_quality_profile("fast")
            
            assert config.get("whisper_beam_size") == 1
            assert config.get("whisper_best_of") == 1
            assert config.get("audio.chunk_duration") == 2.0
            assert config.get("quality_profile") == "fast"
        finally:
            os.unlink(config_path)

    def test_get_quality_profiles_static(self):
        """Test static method returns all profiles."""
        from src.utils.config import Config
        
        profiles = Config.get_quality_profiles()
        
        assert "fast" in profiles
        assert "balanced" in profiles
        assert "accurate" in profiles


class TestStreamingTranslation:
    """Test cases for streaming translation."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_translate_streaming_calls_callback(self, mock_ollama, mock_whisper):
        """Test that streaming translation calls the callback."""
        from src.providers import Translator
        
        # Create a mock Ollama client
        mock_client = MagicMock()
        # Mock streaming response
        mock_client.generate.return_value = [
            {"response": "Hola"},
            {"response": " mundo"},
            {"response": "!"},
        ]
        
        translator = Translator(target_language="Spanish")
        translator._ollama = mock_client
        translator._detected_language = "en"
        
        callback_results = []
        def on_token(text):
            callback_results.append(text)
        
        result = translator.translate_streaming("Hello world!", on_token)
        
        # Callback should have been called with accumulated text
        assert len(callback_results) >= 1

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_translate_streaming_same_language(self, mock_ollama, mock_whisper):
        """Test streaming returns text unchanged for same language."""
        from src.providers import Translator
        
        translator = Translator(target_language="English")
        translator._detected_language = "en"
        
        callback_results = []
        def on_token(text):
            callback_results.append(text)
        
        result = translator.translate_streaming("Hello world", on_token)
        
        assert result == "Hello world"
        assert "Hello world" in callback_results


class TestHallucinationDetection:
    """Test cases for hallucination detection."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_detects_common_hallucinations(self, mock_ollama, mock_whisper):
        """Test detection of common Whisper hallucination patterns."""
        from src.providers import Translator
        
        translator = Translator()
        
        # Test exact match hallucinations
        assert translator._is_hallucination("you") is True
        assert translator._is_hallucination("...") is True
        assert translator._is_hallucination("um") is True
        
        # Test phrase hallucinations
        assert translator._is_hallucination("Thank you for watching!") is True
        assert translator._is_hallucination("Please subscribe") is True
        
        # Test valid text
        assert translator._is_hallucination("Hello, how are you today?") is False

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_detects_music_notation(self, mock_ollama, mock_whisper):
        """Test detection of music notation hallucinations."""
        from src.providers import Translator
        
        translator = Translator()
        
        assert translator._is_hallucination("♪") is True
        assert translator._is_hallucination("♪♪") is True
        assert translator._is_hallucination("[music]") is True


class TestConfigDefaults:
    """Test cases for new config defaults."""

    def test_default_config_has_new_fields(self):
        """Test that default config includes new accuracy fields."""
        from src.utils.config import Config
        
        defaults = Config.DEFAULT_CONFIG
        
        assert "whisper_beam_size" in defaults
        assert "whisper_best_of" in defaults
        assert "quality_profile" in defaults
        assert defaults["whisper_beam_size"] == 3
        assert defaults["whisper_best_of"] == 2
        assert defaults["quality_profile"] == "balanced"


class TestTranslatorStats:
    """Test cases for translator statistics."""

    @patch('src.providers.translator.Translator._load_whisper')
    @patch('src.providers.translator.Translator._load_ollama')
    def test_get_stats_returns_all_fields(self, mock_ollama, mock_whisper):
        """Test that get_stats returns all expected fields."""
        from src.providers import Translator
        
        translator = Translator(whisper_model="small", ollama_model="llama3.2")
        translator._detected_language = "ja"
        translator._language_confidence = 0.95
        translator._last_mode_used = "ollama"
        
        stats = translator.get_stats()
        
        assert stats["language"] == "ja"
        assert stats["confidence"] == 0.95
        assert stats["mode"] == "ollama"
        assert stats["whisper_model"] == "small"
        assert stats["ollama_model"] == "llama3.2"
