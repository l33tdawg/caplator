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

        self.ollama_model = QLineEdit()
        self.ollama_model.setPlaceholderText("llama3.2")
        ollama_layout.addRow("Model:", self.ollama_model)

        self.ollama_host = QLineEdit()
        self.ollama_host.setPlaceholderText("http://localhost:11434")
        ollama_layout.addRow("Host:", self.ollama_host)

        info = QLabel("Make sure Ollama is running: ollama serve")
        info.setStyleSheet("color: #888; font-size: 11px;")
        ollama_layout.addRow("", info)

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

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        buttons.addWidget(save_btn)

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

    def _load_settings(self):
        """Load current settings into form."""
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

        # Ollama
        self.ollama_model.setText(self.config.get("ollama_model", "llama3.2"))
        self.ollama_host.setText(self.config.get("ollama_host", "http://localhost:11434"))

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

    def _save(self):
        """Save settings."""
        self.config.set("target_language", self.target_lang.currentText())
        self.config.set("whisper_model", self.whisper_model.currentData())
        self.config.set("device", self.device.currentText())
        self.config.set("ollama_model", self.ollama_model.text() or "llama3.2")
        self.config.set("ollama_host", self.ollama_host.text() or "http://localhost:11434")
        self.config.set("audio_device", self.audio_device.currentData())
        self.config.set("audio.chunk_duration", float(self.chunk_duration.value()))
        self.config.set("overlay.opacity", self.opacity.value() / 100.0)
        self.config.set("overlay.font_size", self.font_size.value())

        self.config.save()
        self.settings_saved.emit()
        self.accept()
