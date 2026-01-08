"""Tests for configuration management."""

import json
import pytest

from src.utils.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_default_config_creation(self, tmp_path):
        """Test that default config is created when no file exists."""
        config_path = tmp_path / "config.json"
        config = Config(str(config_path))

        assert config_path.exists()
        assert config.get("target_language") == "English"
        assert config.get("whisper_model") == "small"
        assert config.get("ollama_model") == "llama3.2"

    def test_load_existing_config(self, tmp_path):
        """Test loading an existing config file."""
        config_path = tmp_path / "config.json"

        with open(config_path, 'w') as f:
            json.dump({
                "target_language": "Spanish",
                "whisper_model": "tiny",
            }, f)

        config = Config(str(config_path))

        assert config.get("target_language") == "Spanish"
        assert config.get("whisper_model") == "tiny"

    def test_get_nested_value(self, tmp_path):
        """Test getting nested values with dot notation."""
        config = Config(str(tmp_path / "config.json"))

        assert config.get("audio.sample_rate") == 16000
        assert config.get("overlay.opacity") == 0.85

    def test_get_default_value(self, tmp_path):
        """Test getting default value for missing keys."""
        config = Config(str(tmp_path / "config.json"))

        assert config.get("nonexistent", "default") == "default"
        assert config.get("nonexistent") is None

    def test_set_value(self, tmp_path):
        """Test setting configuration values."""
        config = Config(str(tmp_path / "config.json"))

        config.set("target_language", "French")
        config.set("overlay.opacity", 0.5)

        assert config.get("target_language") == "French"
        assert config.get("overlay.opacity") == 0.5

    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config_path = tmp_path / "config.json"
        config = Config(str(config_path))

        config.set("target_language", "German")
        config.save()

        # Reload and verify
        with open(config_path) as f:
            saved = json.load(f)

        assert saved["target_language"] == "German"

    def test_config_merge_with_defaults(self, tmp_path):
        """Test that loaded config merges with defaults."""
        config_path = tmp_path / "config.json"

        # Create partial config
        with open(config_path, 'w') as f:
            json.dump({"whisper_model": "base"}, f)

        config = Config(str(config_path))

        # Custom value preserved
        assert config.get("whisper_model") == "base"

        # Default values present
        assert config.get("target_language") == "English"
        assert config.get("ollama_model") == "llama3.2"

    def test_to_dict(self, tmp_path):
        """Test converting config to dictionary."""
        config = Config(str(tmp_path / "config.json"))
        config.set("custom_key", "custom_value")

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["custom_key"] == "custom_value"

    def test_invalid_json_fallback(self, tmp_path):
        """Test fallback to defaults when JSON is invalid."""
        config_path = tmp_path / "config.json"

        with open(config_path, 'w') as f:
            f.write("invalid json {{{")

        config = Config(str(config_path))

        # Should use defaults
        assert config.get("whisper_model") == "small"

    def test_ollama_settings(self, tmp_path):
        """Test Ollama-specific settings."""
        config = Config(str(tmp_path / "config.json"))

        assert config.get("ollama_host") == "http://localhost:11434"
        assert config.get("ollama_model") == "llama3.2"

        config.set("ollama_model", "mistral")
        assert config.get("ollama_model") == "mistral"

    def test_audio_settings(self, tmp_path):
        """Test audio settings."""
        config = Config(str(tmp_path / "config.json"))

        assert config.get("audio.sample_rate") == 16000
        assert config.get("audio.chunk_duration") == 2.0
        assert config.get("audio.silence_threshold") == 500
