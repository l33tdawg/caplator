"""Configuration management for the Real-Time Audio Translator."""

import json
import os
from pathlib import Path
from typing import Any, Optional


# Quality profile presets
QUALITY_PROFILES = {
    "fast": {
        "whisper_beam_size": 1,
        "whisper_best_of": 1,
        "audio.chunk_duration": 2.0,
        "description": "Fastest response, lower accuracy. Best for live conversations.",
    },
    "balanced": {
        "whisper_beam_size": 3,
        "whisper_best_of": 2,
        "audio.chunk_duration": 3.0,
        "description": "Good balance of speed and accuracy. Recommended for most use.",
    },
    "accurate": {
        "whisper_beam_size": 5,
        "whisper_best_of": 3,
        "audio.chunk_duration": 4.0,
        "description": "Highest accuracy, slower response. Best for important recordings.",
    },
}


class Config:
    """Manages application configuration with JSON storage."""

    DEFAULT_CONFIG = {
        # Translation settings
        "translation_mode": "auto",  # transcribe_only, ollama, auto
        "target_language": "English",
        "whisper_model": "small",  # tiny, base, small, medium, large-v3, large-v3-turbo
        "whisper_backend": "faster-whisper",  # faster-whisper, mlx
        "ollama_model": "llama3.2",
        "ollama_host": "http://localhost:11434",
        "device": "auto",  # auto, cpu, cuda
        
        # Whisper accuracy settings
        "whisper_beam_size": 3,  # 1-5, higher = more accurate but slower
        "whisper_best_of": 2,  # Number of candidates to consider
        
        # Quality profile: fast, balanced, accurate
        "quality_profile": "balanced",

        # Audio settings
        "audio_device": None,  # None = auto-detect BlackHole
        "audio": {
            "sample_rate": 44100,
            "chunk_duration": 3.0,
            "silence_threshold": 100,
        },

        # Overlay appearance
        "overlay": {
            "opacity": 0.85,
            "font_size": 18,
            "position": {"x": 100, "y": 100},
            "size": {"width": 600, "height": 150},
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        if config_path is None:
            self.config_path = Path(__file__).parent.parent.parent / "config.json"
        else:
            self.config_path = Path(config_path)

        self._config = self._load_config()

    def _load_config(self) -> dict:
        """Load config from file or create default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded = json.load(f)
                return self._merge(self.DEFAULT_CONFIG.copy(), loaded)
            except (json.JSONDecodeError, IOError):
                return self.DEFAULT_CONFIG.copy()
        else:
            # Return defaults and save them (save must come after _config is set)
            defaults = self.DEFAULT_CONFIG.copy()
            self._save_defaults(defaults)
            return defaults

    def _save_defaults(self, config: dict) -> None:
        """Save default config to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _merge(self, default: dict, loaded: dict) -> dict:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge(result[key], value)
            else:
                result[key] = value
        return result

    def save(self) -> None:
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot notation (e.g., 'overlay.opacity')."""
        keys = key.split(".")
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot notation."""
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> dict:
        """Return config as dictionary."""
        return self._config.copy()
    
    def apply_quality_profile(self, profile_name: str) -> None:
        """Apply a quality profile preset.
        
        Args:
            profile_name: One of 'fast', 'balanced', 'accurate'
        """
        if profile_name not in QUALITY_PROFILES:
            return
        
        profile = QUALITY_PROFILES[profile_name]
        for key, value in profile.items():
            if key != "description":  # Skip the description field
                self.set(key, value)
        
        self.set("quality_profile", profile_name)
    
    @staticmethod
    def get_quality_profiles() -> dict:
        """Return available quality profiles."""
        return QUALITY_PROFILES