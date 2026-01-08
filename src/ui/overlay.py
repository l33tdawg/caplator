"""Floating transparent overlay window for displaying captions."""

from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSizeGrip,
    QGraphicsOpacityEffect,
    QScrollArea,
    QFrame,
)
from PyQt6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QBrush


class OverlayWindow(QWidget):
    """Floating transparent window for displaying real-time captions."""

    # Signals for controls
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    settings_clicked = pyqtSignal()

    def __init__(self, config, parent=None):
        """Initialize the overlay window."""
        super().__init__(parent)

        self.config = config
        self._drag_position = QPoint()
        self._is_running = False
        
        # Rolling caption history
        self._caption_history = []
        self._max_history = 50  # Keep lots of history for scrolling
        self._last_text_time = 0
        self._paragraph_threshold = 5.0  # Seconds of silence before new paragraph (increased)
        self._last_text = ""  # For smart combining

        self._setup_window()
        self._setup_ui()
        self._apply_config()

    def _setup_window(self):
        """Configure window properties for floating overlay."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(400, 120)

    def _setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(8)

        # Control buttons row
        controls = QHBoxLayout()
        controls.setSpacing(8)

        # Start/Stop button
        self.start_stop_btn = QPushButton("▶ Start")
        self.start_stop_btn.setFixedSize(80, 28)
        self.start_stop_btn.clicked.connect(self._on_start_stop)
        self.start_stop_btn.setStyleSheet(self._button_style("#4CAF50"))
        controls.addWidget(self.start_stop_btn)

        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(28, 28)
        settings_btn.clicked.connect(self.settings_clicked.emit)
        settings_btn.setStyleSheet(self._button_style("#666"))
        settings_btn.setToolTip("Settings")
        controls.addWidget(settings_btn)

        controls.addStretch()

        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        controls.addWidget(self.status_label)

        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self._on_close)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #666;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #ff6b6b;
            }
        """)
        controls.addWidget(close_btn)

        main_layout.addLayout(controls)

        # Stats bar - shows detected language, mode, confidence, models
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)
        
        # Language indicator
        self.lang_label = QLabel("--")
        self.lang_label.setStyleSheet("color: #888; font-size: 10px;")
        self.lang_label.setToolTip("Detected source language")
        stats_layout.addWidget(self.lang_label)
        
        # Confidence indicator
        self.confidence_label = QLabel("--%")
        self.confidence_label.setStyleSheet("color: #888; font-size: 10px;")
        self.confidence_label.setToolTip("Language detection confidence")
        stats_layout.addWidget(self.confidence_label)
        
        # Separator
        sep1 = QLabel("|")
        sep1.setStyleSheet("color: #444; font-size: 10px;")
        stats_layout.addWidget(sep1)
        
        # Mode indicator
        self.mode_label = QLabel("--")
        self.mode_label.setStyleSheet("color: #888; font-size: 10px;")
        self.mode_label.setToolTip("Translation mode")
        stats_layout.addWidget(self.mode_label)
        
        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #444; font-size: 10px;")
        stats_layout.addWidget(sep2)
        
        # Whisper model indicator
        self.whisper_model_label = QLabel("Whisper: --")
        self.whisper_model_label.setStyleSheet("color: #888; font-size: 10px;")
        self.whisper_model_label.setToolTip("Whisper model for transcription")
        stats_layout.addWidget(self.whisper_model_label)
        
        # Ollama model indicator
        self.ollama_model_label = QLabel("LLM: --")
        self.ollama_model_label.setStyleSheet("color: #888; font-size: 10px;")
        self.ollama_model_label.setToolTip("Ollama model for translation")
        stats_layout.addWidget(self.ollama_model_label)
        
        stats_layout.addStretch()
        main_layout.addLayout(stats_layout)

        # Scrollable caption area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.viewport().setStyleSheet("background: transparent;")
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.1);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Caption label inside scroll area
        self.caption_label = QLabel("Click ▶ Start to begin translating...")
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.caption_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.caption_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 16px;
                line-height: 1.6;
                padding: 8px 12px;
                background-color: transparent;
            }
        """)
        
        self.scroll_area.setWidget(self.caption_label)
        main_layout.addWidget(self.scroll_area, 1)

        # Size grip
        grip_layout = QHBoxLayout()
        grip_layout.addStretch()
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(16, 16)
        grip_layout.addWidget(self.size_grip)
        main_layout.addLayout(grip_layout)

        # Setup animation
        self._setup_animation()

    def _button_style(self, color: str) -> str:
        """Generate button stylesheet."""
        return f"""
            QPushButton {{
                background: {color};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background: {color};
                opacity: 0.85;
            }}
            QPushButton:pressed {{
                background: {color};
                opacity: 0.7;
            }}
        """

    def _setup_animation(self):
        """Setup fade animation for smooth text updates."""
        self.opacity_effect = QGraphicsOpacityEffect(self.caption_label)
        self.caption_label.setGraphicsEffect(self.opacity_effect)

        self.fade_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.fade_animation.setDuration(150)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def _apply_config(self):
        """Apply configuration settings."""
        overlay_config = self.config.get("overlay", {})

        pos = overlay_config.get("position", {"x": 100, "y": 100})
        size = overlay_config.get("size", {"width": 600, "height": 150})

        self.move(pos.get("x", 100), pos.get("y", 100))
        self.resize(size.get("width", 600), size.get("height", 150))

        self._window_opacity = overlay_config.get("opacity", 0.85)

        font_size = overlay_config.get("font_size", 18)
        font = QFont("Helvetica Neue", font_size)
        font.setWeight(QFont.Weight.Medium)
        self.caption_label.setFont(font)

        self.caption_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 240);
                padding: 4px;
                background: transparent;
            }
        """)

        self.update()

    def apply_settings(self, config):
        """Apply new settings from config."""
        self.config = config
        self._apply_config()

    def paintEvent(self, event):
        """Paint the semi-transparent rounded background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        opacity = int(self._window_opacity * 255)
        painter.fillPath(path, QBrush(QColor(20, 20, 25, opacity)))

        painter.setPen(QColor(60, 60, 70, 100))
        painter.drawPath(path)

    def mousePressEvent(self, event):
        """Handle mouse press for dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release - save position."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.pos()
            self.config.set("overlay.position.x", pos.x())
            self.config.set("overlay.position.y", pos.y())
            self.config.save()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to toggle size."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.width() < 700:
                self.resize(800, 200)
            else:
                self.resize(600, 150)

            self.config.set("overlay.size.width", self.width())
            self.config.set("overlay.size.height", self.height())
            self.config.save()

    def _on_start_stop(self):
        """Handle start/stop button click."""
        if self._is_running:
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    def _on_close(self):
        """Handle close button - quit app."""
        from PyQt6.QtWidgets import QApplication
        QApplication.quit()

    def set_running(self, running: bool):
        """Update running state and button."""
        self._is_running = running
        if running:
            self.start_stop_btn.setText("■ Stop")
            self.start_stop_btn.setStyleSheet(self._button_style("#f44336"))
            self.status_label.setText("🎧 Listening")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")
            self._caption_history = []  # Clear history on start
            self.caption_label.setText("🎧 Listening for audio...")
            # Reset language stats (models will be set separately)
            self.lang_label.setText("--")
            self.lang_label.setStyleSheet("color: #888; font-size: 10px;")
            self.confidence_label.setText("--%")
            self.confidence_label.setStyleSheet("color: #888; font-size: 10px;")
        else:
            self.start_stop_btn.setText("▶ Start")
            self.start_stop_btn.setStyleSheet(self._button_style("#4CAF50"))
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("color: #888; font-size: 11px;")
            # Clear all stats
            self.lang_label.setText("--")
            self.lang_label.setStyleSheet("color: #888; font-size: 10px;")
            self.mode_label.setText("--")
            self.mode_label.setStyleSheet("color: #888; font-size: 10px;")
            self.confidence_label.setText("--%")
            self.confidence_label.setStyleSheet("color: #888; font-size: 10px;")
            self.whisper_model_label.setText("Whisper: --")
            self.whisper_model_label.setStyleSheet("color: #888; font-size: 10px;")
            self.ollama_model_label.setText("LLM: --")
            self.ollama_model_label.setStyleSheet("color: #888; font-size: 10px;")

    def update_stats(self, language: str = None, mode: str = None, confidence: float = None,
                     whisper_model: str = None, ollama_model: str = None):
        """Update the stats bar with current translation info."""
        if language:
            # Map language codes to names
            lang_names = {
                "en": "English", "es": "Spanish", "fr": "French", "de": "German",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
                "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
            }
            lang_display = lang_names.get(language, language.upper())
            self.lang_label.setText(f"🌐 {lang_display}")
            self.lang_label.setStyleSheet("color: #64B5F6; font-size: 10px;")
        
        if mode:
            mode_display = {
                "transcribe_only": "📝 Transcribe",
                "ollama": "🤖 Ollama",
                "auto": "⚡ Auto",
                "whisper": "🎯 Whisper→EN",
            }.get(mode, mode)
            self.mode_label.setText(mode_display)
            self.mode_label.setStyleSheet("color: #81C784; font-size: 10px;")
        
        if confidence is not None:
            conf_pct = int(confidence * 100)
            # Color based on confidence level
            if conf_pct >= 80:
                color = "#4CAF50"  # Green
            elif conf_pct >= 60:
                color = "#FFC107"  # Yellow
            else:
                color = "#FF5722"  # Orange
            self.confidence_label.setText(f"🎯 {conf_pct}%")
            self.confidence_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        
        if whisper_model:
            self.whisper_model_label.setText(f"🎤 {whisper_model}")
            self.whisper_model_label.setStyleSheet("color: #CE93D8; font-size: 10px;")
        
        if ollama_model:
            # Shorten long model names (e.g., "llama3.2:latest" -> "llama3.2")
            short_name = ollama_model.split(":")[0] if ":" in ollama_model else ollama_model
            self.ollama_model_label.setText(f"🧠 {short_name}")
            self.ollama_model_label.setStyleSheet("color: #FFB74D; font-size: 10px;")

    def update_text(self, text: str):
        """Update caption text with fade animation."""
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.3)
        self.fade_animation.start()
        self.fade_animation.finished.connect(lambda: self._set_text(text))

    def _set_text(self, text: str):
        """Set text and fade back in."""
        try:
            self.fade_animation.finished.disconnect()
        except TypeError:
            pass

        self.caption_label.setText(text)
        self.fade_animation.setStartValue(0.3)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.start()

    def set_text_immediate(self, text: str):
        """Set text immediately without animation, with smart sentence combining."""
        import time
        
        # Status messages - just display directly
        if not text or text.startswith("🎧") or text.startswith("Click"):
            self.caption_label.setText(text)
            return
        
        # Music indicator
        if text == "[MUSIC]":
            if not self._caption_history or self._caption_history[-1] != "♪ ♪ ♪":
                self._caption_history.append("♪ ♪ ♪")
            self._update_display()
            return
        
        current_time = time.time()
        silence_duration = current_time - self._last_text_time if self._last_text_time > 0 else 0
        
        # Check if last text was an incomplete sentence (no ending punctuation)
        def is_complete_sentence(s):
            s = s.strip()
            return s and s[-1] in '.?!。？！'
        
        def should_combine(prev, curr):
            """Determine if current text should be combined with previous."""
            if not prev:
                return False
            prev = prev.strip()
            curr = curr.strip()
            # Combine if previous doesn't end with sentence-ending punctuation
            # and the pause wasn't too long
            if not is_complete_sentence(prev) and silence_duration < self._paragraph_threshold:
                return True
            # Combine if current starts with lowercase (continuation)
            if curr and curr[0].islower():
                return True
            return False
        
        # Smart combining with previous text
        if self._caption_history and should_combine(self._last_text, text):
            # Combine with the last entry
            last_entry = self._caption_history[-1]
            if last_entry and last_entry != "♪ ♪ ♪" and last_entry != "":
                # Merge the texts
                combined = f"{last_entry} {text.strip()}"
                self._caption_history[-1] = combined
                self._last_text = combined
                self._last_text_time = current_time
                self._update_display()
                return
        
        # Check for paragraph break (long silence after complete sentence)
        if self._last_text_time > 0 and silence_duration > self._paragraph_threshold:
            if self._caption_history and self._caption_history[-1] != "":
                # Only add paragraph if last sentence was complete
                if is_complete_sentence(self._last_text):
                    self._caption_history.append("")
        
        self._last_text_time = current_time
        self._last_text = text
        
        # Add to history (avoid duplicates)
        if not self._caption_history or self._caption_history[-1] != text:
            self._caption_history.append(text)
        
        self._update_display()
    
    def _update_display(self):
        """Update the caption display."""
        # Keep only recent items
        if len(self._caption_history) > self._max_history:
            self._caption_history = self._caption_history[-self._max_history:]
        
        # Format display text
        display_text = "\n".join(self._caption_history)
        self.caption_label.setText(display_text)
        
        # Auto-scroll to bottom
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(10, self._scroll_to_bottom)
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the caption area."""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_history(self):
        """Clear caption history."""
        self._caption_history = []
        self._last_text_time = 0
        self.caption_label.setText("🎧 Listening...")

    def closeEvent(self, event):
        """Handle window close."""
        pos = self.pos()
        self.config.set("overlay.position.x", pos.x())
        self.config.set("overlay.position.y", pos.y())
        self.config.set("overlay.size.width", self.width())
        self.config.set("overlay.size.height", self.height())
        self.config.save()
        super().closeEvent(event)
