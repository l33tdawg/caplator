"""
Local Real-Time Translator

Combines faster-whisper for transcription and Ollama for translation.
Runs entirely locally with zero API costs.
"""

import os
import struct
import tempfile
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

from .base import BaseProvider, TranscriptionError, TranslationError


class Translator(BaseProvider):
    """Real-time translator using faster-whisper + Ollama.
    
    This is the main (and only) provider for the app. Everything runs locally:
    - Transcription: faster-whisper (CTranslate2 optimized Whisper)
    - Translation: Ollama with your choice of model (llama3.2, mistral, etc.)
    
    Model size recommendations for real-time use:
    - tiny/base: Best for older hardware or when speed is critical
    - small: Recommended balance of speed and accuracy
    - medium: High accuracy, needs decent CPU or GPU
    - large-v3: Best accuracy, recommended only with GPU
    """

    # Default cache directory for Whisper models
    CACHE_DIR = Path.home() / ".cache" / "translator"

    def __init__(
        self,
        target_language: str = "English",
        whisper_model: str = "small",
        whisper_backend: str = "auto",  # auto, mlx, faster-whisper
        ollama_model: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
        device: str = "auto",
        compute_type: str = "auto",
        translation_mode: str = "auto",  # transcribe_only, ollama, auto
    ):
        """Initialize the translator.

        Args:
            target_language: Language to translate into
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
            whisper_backend: Which Whisper implementation (auto, mlx, faster-whisper)
                            - mlx: Fastest on Apple Silicon (M1/M2/M3)
                            - faster-whisper: Good on all platforms
                            - auto: Use mlx if available on Apple Silicon, else faster-whisper
            ollama_model: Ollama model for translation (llama3.2, mistral, etc.)
            ollama_host: Ollama server URL
            device: Compute device (auto, cpu, cuda)
            compute_type: Precision (auto, int8, float16, float32)
            translation_mode: How to handle translation
                            - transcribe_only: Just transcribe, no translation
                            - ollama: Always use Ollama for translation
                            - auto: Use Whisper for English, Ollama for others
        """
        super().__init__(target_language)

        self.whisper_model_name = whisper_model
        self.whisper_backend = self._detect_backend(whisper_backend)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.device = device
        self.compute_type = compute_type
        self.translation_mode = translation_mode

        # Lazy-loaded models
        self._whisper = None
        self._ollama = None
        self._detected_language = None
        self._language_confidence = 0.0
        self._last_mode_used = None  # Track which translation method was used
        
        # Context buffer for better accuracy
        self._context_buffer = []  # Recent transcriptions for context
        self._max_context = 3  # Keep last 3 transcriptions

        # Ensure cache directory exists
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _detect_backend(self, backend: str) -> str:
        """Detect which Whisper backend to use."""
        if backend != "auto":
            return backend
        
        # MLX-Whisper can be unstable, so default to faster-whisper
        # Users can manually set "mlx" in config if they want to try it
        return "faster-whisper"

    @property
    def whisper(self):
        """Lazy-load Whisper model."""
        if self._whisper is None:
            self._whisper = self._load_whisper()
        return self._whisper

    @property
    def ollama(self):
        """Lazy-load Ollama client."""
        if self._ollama is None:
            self._ollama = self._load_ollama()
        return self._ollama

    def _load_whisper(self):
        """Load the Whisper model based on selected backend."""
        if self.whisper_backend == "mlx":
            return self._load_mlx_whisper()
        else:
            return self._load_faster_whisper()
    
    def _load_mlx_whisper(self):
        """Load MLX-Whisper (optimized for Apple Silicon)."""
        try:
            import mlx_whisper
        except ImportError:
            raise TranscriptionError(
                "mlx-whisper not installed. Run: pip install mlx-whisper"
            )
        
        print(f"🍎 Loading MLX-Whisper '{self.whisper_model_name}'...")
        # MLX-Whisper uses a different model naming convention
        model_name = f"mlx-community/whisper-{self.whisper_model_name}-mlx"
        print(f"MLX-Whisper ready! (model: {model_name})")
        return {"backend": "mlx", "model_name": model_name}

    def _load_faster_whisper(self):
        """Load faster-whisper (CTranslate2 optimized)."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

        # Determine device
        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        print(f"Loading faster-whisper '{self.whisper_model_name}' on {device} ({compute_type})...")

        model = WhisperModel(
            self.whisper_model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.CACHE_DIR / "whisper"),
        )

        print("faster-whisper model loaded!")
        return {"backend": "faster-whisper", "model": model}

    def _load_ollama(self):
        """Load the Ollama client."""
        try:
            import ollama
        except ImportError:
            raise TranslationError(
                "ollama not installed. Run: pip install ollama"
            )

        return ollama.Client(host=self.ollama_host)

    def transcribe(self, audio_data: bytes, source_sample_rate: int = 44100) -> str:
        """Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes (16-bit PCM, mono)
            source_sample_rate: Sample rate of input audio (default 44100)

        Returns:
            Transcribed text
        """
        try:
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if source_sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, source_sample_rate, 16000)
            
            # Create temp WAV file
            wav_data = self._create_wav(audio_data)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_data)
                wav_path = f.name

            try:
                # Determine task based on translation mode
                if self.translation_mode == "transcribe_only":
                    # Always transcribe in transcribe-only mode
                    task = "transcribe"
                    use_whisper_translate = False
                elif self.translation_mode == "auto":
                    # Use Whisper's built-in translation if target is English
                    use_whisper_translate = self.target_language.lower() in ["en", "english"]
                    task = "translate" if use_whisper_translate else "transcribe"
                else:
                    # Ollama mode: always transcribe first, then translate with Ollama
                    task = "transcribe"
                    use_whisper_translate = False
                
                whisper_data = self.whisper
                
                if whisper_data["backend"] == "mlx":
                    # MLX-Whisper transcription
                    text, detected_lang = self._transcribe_mlx(wav_path, task)
                else:
                    # faster-whisper transcription
                    text, detected_lang = self._transcribe_faster_whisper(wav_path, task, whisper_data["model"])

                # Store detected language
                self._detected_language = detected_lang
                self._source_language = detected_lang
                
                # Mark if we already translated via Whisper
                self._whisper_translated = use_whisper_translate

                return text

            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _transcribe_mlx(self, wav_path: str, task: str) -> Tuple[str, str]:
        """Transcribe using MLX-Whisper (Apple Silicon optimized)."""
        import mlx_whisper
        
        model_name = self.whisper["model_name"]
        
        result = mlx_whisper.transcribe(
            wav_path,
            path_or_hf_repo=model_name,
            task=task,
        )
        
        text = result.get("text", "").strip()
        language = result.get("language", "en")
        
        return text, language
    
    def _transcribe_faster_whisper(self, wav_path: str, task: str, model) -> Tuple[str, str]:
        """Transcribe using faster-whisper (CTranslate2 optimized)."""
        # Build context from recent transcriptions
        initial_prompt = None
        if self._context_buffer:
            # Use last few transcriptions as context hint
            initial_prompt = " ".join(self._context_buffer[-self._max_context:])
        
        segments, info = model.transcribe(
            wav_path,
            beam_size=1,
            best_of=1,
            vad_filter=False,
            task=task,
            without_timestamps=True,
            initial_prompt=initial_prompt,  # Provide context!
            condition_on_previous_text=False,  # Avoid hallucinations
        )
        
        text = " ".join(seg.text for seg in segments).strip()
        
        # Store language confidence
        self._language_confidence = info.language_probability
        
        # Update context buffer
        if text:
            self._context_buffer.append(text)
            if len(self._context_buffer) > self._max_context * 2:
                self._context_buffer = self._context_buffer[-self._max_context:]
        
        return text, info.language

    def translate(self, text: str) -> str:
        """Translate text to target language using Ollama.

        Args:
            text: Text to translate

        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""

        # If source matches target, return as-is
        source = self._detected_language or "unknown"
        target_code = self._get_language_code(self.target_language)

        if source == target_code:
            return text

        try:
            # Build translation prompt
            prompt = f"""Translate the following text to {self.target_language}.

RULES:
- Output ONLY the translation, nothing else
- Keep it natural and conversational
- Preserve the tone and meaning

Text: {text}

Translation:"""

            response = self.ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 500,
                    "top_p": 0.9,
                },
            )

            translation = response.get("response", "").strip()
            return self._clean_translation(translation)

        except Exception as e:
            raise TranslationError(f"Translation failed: {e}") from e

    def transcribe_and_translate(self, audio_data: bytes, source_sample_rate: int = 44100) -> str:
        """Transcribe audio and translate to target language.

        Args:
            audio_data: Raw audio bytes
            source_sample_rate: Sample rate of input audio (default 44100)

        Returns:
            Translated text (or just transcription if in transcribe_only mode)
        """
        transcription = self.transcribe(audio_data, source_sample_rate)
        if not transcription:
            return ""
        
        # Transcription-only mode: no translation
        if self.translation_mode == "transcribe_only":
            self._last_mode_used = "transcribe_only"
            return transcription
        
        # Skip Ollama if Whisper already translated to English (much faster!)
        if getattr(self, '_whisper_translated', False):
            self._last_mode_used = "whisper"
            return transcription
        
        self._last_mode_used = "ollama"
        return self.translate(transcription)
    
    def get_stats(self) -> dict:
        """Get current translation statistics.
        
        Returns:
            Dict with language, confidence, and mode info
        """
        return {
            "language": self._detected_language,
            "confidence": self._language_confidence,
            "mode": self._last_mode_used or self.translation_mode,
        }
    
    def clear_context(self):
        """Clear the context buffer. Call when starting a new session."""
        self._context_buffer = []

    def _clean_translation(self, text: str) -> str:
        """Clean up LLM output artifacts."""
        result = text.strip()

        # Remove common prefixes
        prefixes = [
            "Translation:",
            "Here's the translation:",
            "The translation is:",
            f"In {self.target_language}:",
        ]
        for prefix in prefixes:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()

        # Remove surrounding quotes
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result.startswith("'") and result.endswith("'"):
            result = result[1:-1]

        return result.strip()

    def _get_language_code(self, language: str) -> str:
        """Convert language name to code."""
        codes = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Dutch": "nl",
            "Russian": "ru",
            "Chinese": "zh",
            "Chinese (Simplified)": "zh",
            "Chinese (Traditional)": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Arabic": "ar",
            "Hindi": "hi",
            "Vietnamese": "vi",
            "Thai": "th",
            "Indonesian": "id",
            "Turkish": "tr",
            "Polish": "pl",
            "Ukrainian": "uk",
            "Swedish": "sv",
            "Danish": "da",
            "Norwegian": "no",
            "Finnish": "fi",
            "Greek": "el",
            "Hebrew": "he",
            "Czech": "cs",
            "Romanian": "ro",
            "Hungarian": "hu",
        }
        return codes.get(language, language.lower()[:2])

    def _resample_audio(self, audio_data: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio from one sample rate to another.
        
        Uses simple linear interpolation for speed.
        """
        # Convert to numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Calculate new length
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        
        # Resample using linear interpolation
        indices = np.linspace(0, len(audio) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(audio)), audio)
        
        # Convert back to int16 bytes
        return resampled.astype(np.int16).tobytes()

    def _create_wav(self, pcm_data: bytes, sample_rate: int = 16000) -> bytes:
        """Create WAV file from raw PCM data."""
        channels = 1
        bits = 16
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(pcm_data),
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            b"data",
            len(pcm_data),
        )

        return header + pcm_data

    def check_ollama(self) -> Tuple[bool, str]:
        """Check if Ollama is running and model is available.
        
        Auto-selects an available model if configured one isn't found.

        Returns:
            Tuple of (success, message)
        """
        try:
            response = self.ollama.list()
            
            # Handle both dict and object responses
            if hasattr(response, 'models'):
                models_list = response.models
            else:
                models_list = response.get("models", [])
            
            # Extract model names
            model_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict):
                    model_names.append(m.get("name", ""))
            
            print(f"Available models: {model_names}")

            # Check if our model is available
            for name in model_names:
                if self.ollama_model in name:
                    self.ollama_model = name
                    return True, f"Using {name}"

            # Try to find an alternative model (prefer smaller/faster ones for translation)
            preferred = ["llama3.2", "llama3", "gemma", "mistral", "phi", "qwen", "deepseek"]
            
            for pref in preferred:
                for name in model_names:
                    if pref in name.lower():
                        self.ollama_model = name
                        print(f"Auto-selected model: {name}")
                        return True, f"Using {name}"

            # Use any available model
            if model_names:
                self.ollama_model = model_names[0]
                print(f"Auto-selected model: {model_names[0]}")
                return True, f"Using {model_names[0]}"

            return False, "No Ollama models found. Run: ollama pull llama3.2"

        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                return False, "Ollama not running. Start it: ollama serve"
            return False, f"Ollama error: {e}"

    @classmethod
    def download_whisper_model(cls, model: str = "small"):
        """Pre-download a Whisper model for offline use.

        Args:
            model: Model to download (tiny, base, small, medium, large-v3)
        """
        try:
            from faster_whisper import WhisperModel

            print(f"Downloading Whisper model '{model}'...")
            cache_dir = cls.CACHE_DIR / "whisper"
            cache_dir.mkdir(parents=True, exist_ok=True)

            WhisperModel(
                model,
                device="cpu",
                compute_type="int8",
                download_root=str(cache_dir),
            )

            print(f"✓ Whisper model '{model}' downloaded to {cache_dir}")

        except Exception as e:
            print(f"✗ Failed to download model: {e}")
