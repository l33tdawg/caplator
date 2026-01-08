"""Tests for UI components."""

import json
import pytest


class TestSettingsDialog:
    """Test cases for SettingsDialog."""

    def test_languages_list(self):
        """Test that common languages are in the list."""
        from src.ui.settings import SettingsDialog

        expected = ["English", "Spanish", "French", "German", "Japanese", "Korean"]
        for lang in expected:
            assert lang in SettingsDialog.LANGUAGES

    def test_whisper_models_list(self):
        """Test that all Whisper models are defined."""
        from src.ui.settings import SettingsDialog

        model_ids = [m[0] for m in SettingsDialog.WHISPER_MODELS]
        assert "tiny" in model_ids
        assert "base" in model_ids
        assert "small" in model_ids
        assert "medium" in model_ids
        assert "large-v3" in model_ids


class TestUIIntegration:
    """Integration tests for UI with config."""

    def test_config_values_for_ui(self, tmp_path):
        """Test that config values match UI expectations."""
        config_path = tmp_path / "config.json"

        with open(config_path, 'w') as f:
            json.dump({
                "target_language": "Japanese",
                "whisper_model": "base",
                "ollama_model": "mistral",
            }, f)

        from src.utils.config import Config
        config = Config(str(config_path))

        assert config.get("target_language") == "Japanese"
        assert config.get("whisper_model") == "base"
        assert config.get("ollama_model") == "mistral"

    def test_overlay_config(self, tmp_path):
        """Test overlay configuration structure."""
        from src.utils.config import Config

        # Test that DEFAULT_CONFIG has overlay settings
        assert "overlay" in Config.DEFAULT_CONFIG
        assert "opacity" in Config.DEFAULT_CONFIG["overlay"]
        assert "font_size" in Config.DEFAULT_CONFIG["overlay"]
        assert "position" in Config.DEFAULT_CONFIG["overlay"]

        # Test that values are reasonable
        assert 0 < Config.DEFAULT_CONFIG["overlay"]["opacity"] <= 1.0
        assert 10 <= Config.DEFAULT_CONFIG["overlay"]["font_size"] <= 48
