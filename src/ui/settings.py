"""Settings dialog for the Real-Time Audio Translator."""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QComboBox,
    QSlider,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QLineEdit,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

import sounddevice as sd


class SettingsDialog(QDialog):
    """Settings dialog for configuring the translator."""

    settings_saved = pyqtSignal()

    LANGUAGES = [
        "English", "Spanish", "French", "German", "Italian", "Portuguese",
        "Dutch", "Russian", "Chinese (Simplified)", "Japanese", "Korean",
        "Arabic", "Hindi", "Vietnamese", "Thai", "Indonesian", "Turkish",
        "Polish", "Ukrainian", "Swedish", "Danish", "Norwegian", "Finnish",
        "Greek", "Hebrew", "Czech", "Romanian", "Hungarian",
    ]

    WHISPER_MODELS = [
        ("tiny", "Tiny (~75MB) - Fastest, lower accuracy"),
        ("base", "Base (~145MB) - Fast, decent accuracy"),
        ("small", "Small (~488MB) - Balanced (recommended)"),
        ("medium", "Medium (~1.5GB) - High accuracy, slower"),
        ("large-v3", "Large-v3 (~3GB) - Best accuracy, needs GPU"),
    ]

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the UI."""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)

        # Translation settings
        trans_group = QGroupBox("Translation")
        trans_layout = QFormLayout(trans_group)

        self.translation_mode = QComboBox()
        self.translation_mode.addItem("Transcription Only (Whisper)", "transcribe_only")
        self.translation_mode.addItem("Translate with Ollama", "ollama")
        self.translation_mode.addItem("Auto (Whisper→English, else Ollama)", "auto")
        self.translation_mode.currentIndexChanged.connect(self._on_mode_changed)
        trans_layout.addRow("Mode:", self.translation_mode)

        self.target_lang = QComboBox()
        self.target_lang.addItems(self.LANGUAGES)
        self.target_lang.setEditable(True)
        trans_layout.addRow("Translate to:", self.target_lang)

        layout.addWidget(trans_group)

        # Whisper settings
        whisper_group = QGroupBox("Transcription (Whisper)")
        whisper_layout = QFormLayout(whisper_group)

        self.whisper_model = QComboBox()
        for model_id, label in self.WHISPER_MODELS:
            self.whisper_model.addItem(label, model_id)
        whisper_layout.addRow("Model:", self.whisper_model)

        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        whisper_layout.addRow("Device:", self.device)

        layout.addWidget(whisper_group)

        # Ollama settings
        ollama_group = QGroupBox("Translation (Ollama)")
        ollama_layout = QFormLayout(ollama_group)

        self.ollama_host = QLineEdit()
        self.ollama_host.setPlaceholderText("http://localhost:11434")
        self.ollama_host.textChanged.connect(self._on_host_changed)
        ollama_layout.addRow("Host:", self.ollama_host)

        # Model dropdown with refresh button
        model_row = QHBoxLayout()
        self.ollama_model = QComboBox()
        self.ollama_model.setMinimumWidth(200)
        self.ollama_model.setEditable(True)  # Allow custom model entry
        model_row.addWidget(self.ollama_model, 1)

        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(28, 28)
        refresh_btn.setToolTip("Refresh available models")
        refresh_btn.clicked.connect(self._refresh_ollama_models)
        model_row.addWidget(refresh_btn)
        ollama_layout.addRow("Model:", model_row)

        self.ollama_status = QLabel("Checking Ollama...")
        self.ollama_status.setStyleSheet("color: #888; font-size: 11px;")
        ollama_layout.addRow("", self.ollama_status)

        layout.addWidget(ollama_group)

        # Audio settings
        audio_group = QGroupBox("Audio Input")
        audio_layout = QFormLayout(audio_group)

        self.audio_device = QComboBox()
        self._populate_audio_devices()
        audio_layout.addRow("Device:", self.audio_device)

        self.chunk_duration = QSpinBox()
        self.chunk_duration.setRange(1, 10)
        self.chunk_duration.setSuffix(" seconds")
        audio_layout.addRow("Chunk size:", self.chunk_duration)

        layout.addWidget(audio_group)

        # Appearance
        appearance_group = QGroupBox("Overlay Appearance")
        appearance_layout = QFormLayout(appearance_group)

        opacity_row = QHBoxLayout()
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(20, 100)
        self.opacity_label = QLabel("85%")
        self.opacity.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        opacity_row.addWidget(self.opacity)
        opacity_row.addWidget(self.opacity_label)
        appearance_layout.addRow("Opacity:", opacity_row)

        self.font_size = QSpinBox()
        self.font_size.setRange(12, 48)
        appearance_layout.addRow("Font size:", self.font_size)

        layout.addWidget(appearance_group)

        # Buttons
        buttons = QHBoxLayout()
        buttons.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        apply_btn.setToolTip("Apply settings without closing")
        buttons.addWidget(apply_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._save_and_close)
        buttons.addWidget(ok_btn)

        layout.addLayout(buttons)

    def _populate_audio_devices(self):
        """List available audio input devices."""
        self.audio_device.clear()
        self.audio_device.addItem("Auto-detect (BlackHole)", None)

        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    self.audio_device.addItem(dev["name"], dev["name"])
        except Exception:
            pass

    def _on_host_changed(self, text):
        """Handle Ollama host change - trigger model refresh."""
        # Don't auto-refresh while typing, user can click refresh
        pass

    def _on_mode_changed(self, index):
        """Handle translation mode change - enable/disable Ollama settings."""
        mode = self.translation_mode.currentData()
        ollama_enabled = mode in ("ollama", "auto")
        
        # Enable/disable Ollama controls
        self.ollama_host.setEnabled(ollama_enabled)
        self.ollama_model.setEnabled(ollama_enabled)
        
        # Also update target language based on mode
        if mode == "transcribe_only":
            self.target_lang.setEnabled(False)
            self.target_lang.setToolTip("Target language disabled in transcription-only mode")
        else:
            self.target_lang.setEnabled(True)
            self.target_lang.setToolTip("")

    def _refresh_ollama_models(self):
        """Fetch available models from Ollama server."""
        host = self.ollama_host.text() or "http://localhost:11434"
        current_model = self.ollama_model.currentText()

        self.ollama_model.clear()
        self.ollama_status.setText("Connecting to Ollama...")
        self.ollama_status.setStyleSheet("color: #888; font-size: 11px;")

        # Process events to show the status update
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            import ollama
            client = ollama.Client(host=host)
            response = client.list()

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

            if model_names:
                self.ollama_model.addItems(model_names)
                self.ollama_status.setText(f"✓ Found {len(model_names)} model(s)")
                self.ollama_status.setStyleSheet("color: #4CAF50; font-size: 11px;")

                # Try to restore the previously selected model
                if current_model:
                    idx = self.ollama_model.findText(current_model)
                    if idx >= 0:
                        self.ollama_model.setCurrentIndex(idx)
                    else:
                        # Check for partial match
                        for i in range(self.ollama_model.count()):
                            if current_model in self.ollama_model.itemText(i):
                                self.ollama_model.setCurrentIndex(i)
                                break
            else:
                self.ollama_model.addItem("llama3.2")
                self.ollama_status.setText("⚠ No models found. Run: ollama pull llama3.2")
                self.ollama_status.setStyleSheet("color: #f44336; font-size: 11px;")

        except ImportError:
            self.ollama_model.addItem("llama3.2")
            self.ollama_status.setText("⚠ ollama package not installed")
            self.ollama_status.setStyleSheet("color: #f44336; font-size: 11px;")
        except Exception as e:
            self.ollama_model.addItem("llama3.2")
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                self.ollama_status.setText("⚠ Ollama not running. Start: ollama serve")
            else:
                self.ollama_status.setText(f"⚠ Error: {str(e)[:40]}")
            self.ollama_status.setStyleSheet("color: #f44336; font-size: 11px;")

    def _load_settings(self):
        """Load current settings into form."""
        # Translation mode
        mode = self.config.get("translation_mode", "auto")
        idx = self.translation_mode.findData(mode)
        if idx >= 0:
            self.translation_mode.setCurrentIndex(idx)
        self._on_mode_changed(0)  # Update UI state based on mode

        # Target language
        lang = self.config.get("target_language", "English")
        idx = self.target_lang.findText(lang)
        if idx >= 0:
            self.target_lang.setCurrentIndex(idx)
        else:
            self.target_lang.setCurrentText(lang)

        # Whisper model
        model = self.config.get("whisper_model", "small")
        idx = self.whisper_model.findData(model)
        if idx >= 0:
            self.whisper_model.setCurrentIndex(idx)

        # Device
        device = self.config.get("device", "auto")
        idx = self.device.findText(device)
        if idx >= 0:
            self.device.setCurrentIndex(idx)

        # Ollama - load host first, then refresh models
        self.ollama_host.setText(self.config.get("ollama_host", "http://localhost:11434"))
        
        # Store the configured model to select after refresh
        self._configured_ollama_model = self.config.get("ollama_model", "llama3.2")
        
        # Refresh models from Ollama server
        self._refresh_ollama_models()
        
        # Try to select the configured model
        idx = self.ollama_model.findText(self._configured_ollama_model)
        if idx >= 0:
            self.ollama_model.setCurrentIndex(idx)
        else:
            # Partial match or set as custom text
            found = False
            for i in range(self.ollama_model.count()):
                if self._configured_ollama_model in self.ollama_model.itemText(i):
                    self.ollama_model.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.ollama_model.setCurrentText(self._configured_ollama_model)

        # Audio
        audio_dev = self.config.get("audio_device")
        if audio_dev:
            idx = self.audio_device.findData(audio_dev)
            if idx >= 0:
                self.audio_device.setCurrentIndex(idx)

        self.chunk_duration.setValue(int(self.config.get("audio.chunk_duration", 2)))

        # Appearance
        self.opacity.setValue(int(self.config.get("overlay.opacity", 0.85) * 100))
        self.font_size.setValue(self.config.get("overlay.font_size", 18))

    def _apply(self):
        """Apply settings without closing dialog."""
        try:
            self.config.set("translation_mode", self.translation_mode.currentData())
            self.config.set("target_language", self.target_lang.currentText())
            self.config.set("whisper_model", self.whisper_model.currentData())
            self.config.set("device", self.device.currentText())
            self.config.set("ollama_model", self.ollama_model.currentText() or "llama3.2")
            self.config.set("ollama_host", self.ollama_host.text() or "http://localhost:11434")
            self.config.set("audio_device", self.audio_device.currentData())
            self.config.set("audio.chunk_duration", float(self.chunk_duration.value()))
            self.config.set("overlay.opacity", self.opacity.value() / 100.0)
            self.config.set("overlay.font_size", self.font_size.value())

            self.config.save()
            self.settings_saved.emit()
            
            # Show brief confirmation
            self.ollama_status.setText("✓ Settings applied!")
            self.ollama_status.setStyleSheet("color: #4CAF50; font-size: 11px;")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Settings",
                f"Failed to save settings:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _save_and_close(self):
        """Save settings and close dialog."""
        try:
            self.config.set("translation_mode", self.translation_mode.currentData())
            self.config.set("target_language", self.target_lang.currentText())
            self.config.set("whisper_model", self.whisper_model.currentData())
            self.config.set("device", self.device.currentText())
            self.config.set("ollama_model", self.ollama_model.currentText() or "llama3.2")
            self.config.set("ollama_host", self.ollama_host.text() or "http://localhost:11434")
            self.config.set("audio_device", self.audio_device.currentData())
            self.config.set("audio.chunk_duration", float(self.chunk_duration.value()))
            self.config.set("overlay.opacity", self.opacity.value() / 100.0)
            self.config.set("overlay.font_size", self.font_size.value())

            self.config.save()
            
            # Close dialog - main app will check result and apply settings
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Settings",
                f"Failed to save settings:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
