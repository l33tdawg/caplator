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

from PyQt6.QtWidgets import QApplication, QDialog
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
                translation_mode=self.config.get("translation_mode", "auto"),
                beam_size=self.config.get("whisper_beam_size", 3),
                best_of=self.config.get("whisper_best_of", 2),
            )

            # Check Ollama is running (only if we need it)
            translation_mode = self.config.get("translation_mode", "auto")
            if translation_mode != "transcribe_only":
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

    def _on_streaming_token(self, text: str):
        """Handle streaming translation token (called from background thread)."""
        # Use Qt signal mechanism to safely update UI from background thread
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.overlay, 
            "set_streaming_text", 
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, text)
        )
    
    def _process_audio(self, audio_data: bytes):
        """Process audio in background thread."""
        try:
            sample_rate = self.config.get("audio.sample_rate", 44100)
            
            # Use streaming callback for Ollama translations
            result = self.translator.transcribe_and_translate(
                audio_data, 
                sample_rate,
                on_token=self._on_streaming_token
            )
            
            if result:
                # Finalize the streaming text
                from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
                QMetaObject.invokeMethod(
                    self.overlay, 
                    "finalize_streaming_text", 
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, result)
                )
                
                # Update stats display
                stats = self.translator.get_stats()
                self.overlay.update_stats(
                    language=stats.get("language"),
                    mode=stats.get("mode"),
                    confidence=stats.get("confidence"),
                    whisper_model=stats.get("whisper_model"),
                    ollama_model=stats.get("ollama_model"),
                )
            else:
                # No speech detected but audio was playing = likely music
                self.overlay.set_text_immediate("[MUSIC]")
                
        except Exception as e:
            import traceback
            print(f"Translation error: {e}")
            traceback.print_exc()

    def _show_settings(self):
        """Show settings dialog."""
        # Pass overlay as parent so dialog closing doesn't quit the app
        dialog = SettingsDialog(self.config, parent=self.overlay)
        dialog.settings_saved.connect(self._on_settings_changed)
        result = dialog.exec()
        
        # If dialog was accepted (OK clicked), ensure settings are applied
        if result == QDialog.DialogCode.Accepted:
            self._on_settings_changed()

    def _on_settings_changed(self):
        """Handle settings change."""
        try:
            # Stop if running
            was_running = self.is_running
            if was_running:
                self.stop()

            # Reload config from file
            self.config = Config()
            
            # Apply visual settings to overlay immediately
            self.overlay.apply_settings(self.config)

            # Recreate audio capture with new settings
            self.audio_capture = AudioCapture(
                sample_rate=self.config.get("audio.sample_rate", 44100),
                chunk_duration=self.config.get("audio.chunk_duration", 2.0),
                device_name=self.config.get("audio_device"),
                silence_threshold=self.config.get("audio.silence_threshold", 100),
            )
            self.audio_capture.audio_ready.connect(self._on_audio_ready)

            # Reset translator to pick up new settings
            self.translator = None

            # Restart if it was running
            if was_running:
                self.start()
                
        except Exception as e:
            print(f"Error applying settings: {e}")
            import traceback
            traceback.print_exc()
            self.overlay.set_text_immediate(f"⚠️ Settings error: {str(e)[:50]}")

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
        
        # Show initial mode and models
        mode = self.config.get("translation_mode", "auto")
        whisper_model = self.config.get("whisper_model", "small")
        ollama_model = self.config.get("ollama_model", "llama3.2")
        self.overlay.update_stats(
            mode=mode,
            whisper_model=whisper_model,
            ollama_model=ollama_model if mode != "transcribe_only" else None,
        )

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
    app.setApplicationName("Translator")
    app.setApplicationDisplayName("Real-Time Translator")
    
    # Set dock icon on macOS
    try:
        from AppKit import NSApplication, NSImage
        icon_path = Path(__file__).parent / "assets" / "icon.icns"
        if icon_path.exists():
            ns_app = NSApplication.sharedApplication()
            icon = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
            if icon:
                ns_app.setApplicationIconImage_(icon)
    except ImportError:
        pass  # AppKit not available (not macOS or pyobjc not installed)
    except Exception:
        pass  # Icon setting failed, continue anyway

    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    translator = TranslatorApp()
    translator.run()

    exit_code = app.exec()
    translator.cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
