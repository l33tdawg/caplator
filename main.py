#!/usr/bin/env python3
"""
Real-Time Audio Translator

Captures system audio, transcribes and translates it locally using
faster-whisper and Ollama, then displays captions in a floating overlay.

100% local, 100% free, 100% private.
"""

import sys
import signal
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui import OverlayWindow
from src.ui.settings import SettingsDialog
from src.utils import Config
from src.audio import AudioCapture
from src.providers import Translator


class TranslatorApp:
    """Main application orchestrating audio capture, translation, and display."""

    def __init__(self):
        """Initialize the translator application."""
        self.config = Config()
        self.translator = None
        self.is_running = False

        # Initialize audio capture
        self.audio_capture = AudioCapture(
            sample_rate=self.config.get("audio.sample_rate", 44100),
            chunk_duration=self.config.get("audio.chunk_duration", 2.0),
            device_name=self.config.get("audio_device"),
            silence_threshold=self.config.get("audio.silence_threshold", 100),
        )
        self.audio_capture.audio_ready.connect(self._on_audio_ready)

        # Initialize UI
        self.overlay = OverlayWindow(self.config)

        # Connect overlay signals
        self.overlay.start_clicked.connect(self.start)
        self.overlay.stop_clicked.connect(self.stop)
        self.overlay.settings_clicked.connect(self._show_settings)

        # Thread pool for async translation
        self._executor = ThreadPoolExecutor(max_workers=2)

    def _init_translator(self) -> bool:
        """Initialize the translator. Returns True if successful."""
        try:
            self.overlay.set_text_immediate("Loading Whisper model...")

            self.translator = Translator(
                target_language=self.config.get("target_language", "English"),
                whisper_model=self.config.get("whisper_model", "small"),
                whisper_backend=self.config.get("whisper_backend", "auto"),
                ollama_model=self.config.get("ollama_model", "llama3.2"),
                ollama_host=self.config.get("ollama_host", "http://localhost:11434"),
                device=self.config.get("device", "auto"),
            )

            # Check Ollama is running
            self.overlay.set_text_immediate("Checking Ollama...")
            ok, msg = self.translator.check_ollama()
            if not ok:
                self.overlay.set_text_immediate(f"⚠️ {msg}")
                return False

            return True

        except Exception as e:
            self.overlay.set_text_immediate(f"⚠️ Error: {str(e)[:80]}")
            return False

    def _on_audio_ready(self, audio_data: bytes):
        """Handle audio chunk ready for processing."""
        if not self.is_running or self.translator is None:
            return
        self._executor.submit(self._process_audio, audio_data)

    def _process_audio(self, audio_data: bytes):
        """Process audio in background thread."""
        try:
            sample_rate = self.config.get("audio.sample_rate", 44100)
            result = self.translator.transcribe_and_translate(audio_data, sample_rate)
            
            if result:
                self.overlay.set_text_immediate(result)
            else:
                # No speech detected but audio was playing = likely music
                self.overlay.set_text_immediate("[MUSIC]")
                
        except Exception as e:
            import traceback
            print(f"Translation error: {e}")
            traceback.print_exc()

    def _show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self.config)
        dialog.settings_saved.connect(self._on_settings_changed)
        dialog.exec()

    def _on_settings_changed(self):
        """Handle settings change."""
        try:
            # Reload config from file
            self.config = Config()
            self.overlay.apply_settings(self.config)

            # Reinitialize translator with new settings
            was_running = self.is_running
            if was_running:
                self.stop()
            
            # Reset translator to pick up new settings
            self.translator = None
            
            if was_running:
                self.start()
        except Exception as e:
            print(f"Error applying settings: {e}")
            import traceback
            traceback.print_exc()

    def start(self):
        """Start translation."""
        if self.is_running:
            return

        self.overlay.set_text_immediate("Initializing...")

        if not self._init_translator():
            return

        # Clear context for fresh start
        if self.translator:
            self.translator.clear_context()
        self.overlay.clear_history()

        self.is_running = True
        self.audio_capture.start()
        self.overlay.set_text_immediate("🎧 Listening for audio...")
        self.overlay.set_running(True)

    def stop(self):
        """Stop translation."""
        if not self.is_running:
            return

        self.is_running = False
        self.audio_capture.stop()
        self.overlay.set_text_immediate("Stopped. Click ▶ Start to resume.")
        self.overlay.set_running(False)

    def run(self):
        """Start the application."""
        self.overlay.show()

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self._executor.shutdown(wait=False)


def main():
    """Entry point."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    app.setApplicationName("Real-Time Translator")

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    translator = TranslatorApp()
    translator.run()

    exit_code = app.exec()
    translator.cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
