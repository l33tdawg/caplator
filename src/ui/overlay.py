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
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, QPoint, QPropertyAnimation, QEasingCurve, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath, QBrush, QShortcut, QKeySequence


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
        self._max_history = 50
        self._last_text_time = 0
        self._paragraph_threshold = 5.0
        self._last_text = ""

        self._setup_window()
        self._setup_ui()
        self._setup_shortcuts()
        self._apply_config()

    def _setup_window(self):
        """Configure window properties for floating overlay."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(500, 150)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts for common actions."""
        # Cmd/Ctrl+Shift+T - Toggle translation on/off
        toggle_shortcut = QShortcut(QKeySequence("Ctrl+Shift+T"), self)
        toggle_shortcut.activated.connect(self._on_start_stop)
        
        # Cmd/Ctrl+Shift+C - Clear history
        clear_shortcut = QShortcut(QKeySequence("Ctrl+Shift+C"), self)
        clear_shortcut.activated.connect(self._on_clear_history)
        
        # Cmd/Ctrl+Shift+S - Save/Export captions
        save_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
        save_shortcut.activated.connect(self.export_captions)
        
        # Cmd/Ctrl+Shift+, - Open settings
        settings_shortcut = QShortcut(QKeySequence("Ctrl+,"), self)
        settings_shortcut.activated.connect(self.settings_clicked.emit)
        
        # Escape - Stop translation
        escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        escape_shortcut.activated.connect(self._on_escape)
        
        # Space - Toggle start/stop when not in text field
        space_shortcut = QShortcut(QKeySequence("Space"), self)
        space_shortcut.activated.connect(self._on_start_stop)
    
    def _on_clear_history(self):
        """Handle clear history shortcut."""
        self.clear_history()
        self.caption_label.setText("History cleared. Press ▶ to start.")
    
    def _on_escape(self):
        """Handle escape key - stop if running."""
        if self._is_running:
            self.stop_clicked.emit()

    def _setup_ui(self):
        """Set up the UI components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # Top bar with controls
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        # Play/Pause button (icon style)
        self.start_stop_btn = QPushButton("▶")
        self.start_stop_btn.setFixedSize(32, 32)
        self.start_stop_btn.clicked.connect(self._on_start_stop)
        self.start_stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_stop_btn.setToolTip("Start/Stop")
        self._update_play_button_style(False)
        top_bar.addWidget(self.start_stop_btn)

        # Settings button
        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(32, 32)
        settings_btn.clicked.connect(self.settings_clicked.emit)
        settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_btn.setToolTip("Settings")
        settings_btn.setStyleSheet(self._icon_button_style())
        top_bar.addWidget(settings_btn)
        
        # Export button
        export_btn = QPushButton("💾")
        export_btn.setFixedSize(32, 32)
        export_btn.clicked.connect(self.export_captions)
        export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        export_btn.setToolTip("Export captions to file")
        export_btn.setStyleSheet(self._icon_button_style())
        top_bar.addWidget(export_btn)

        top_bar.addStretch()

        # Status indicator (right side)
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                font-weight: 500;
                padding: 4px 12px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 12px;
            }
        """)
        top_bar.addWidget(self.status_label)

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(28, 28)
        close_btn.clicked.connect(self._on_close)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #555;
                border: none;
                font-size: 20px;
                font-weight: 400;
            }
            QPushButton:hover {
                color: #ff6b6b;
            }
        """)
        top_bar.addWidget(close_btn)

        main_layout.addLayout(top_bar)

        # Stats bar with pill badges
        stats_bar = QHBoxLayout()
        stats_bar.setSpacing(8)

        # Language pill
        self.lang_label = self._create_stat_pill("--", "#64B5F6")
        stats_bar.addWidget(self.lang_label)

        # Confidence pill
        self.confidence_label = self._create_stat_pill("--%", "#888")
        stats_bar.addWidget(self.confidence_label)

        # Mode pill
        self.mode_label = self._create_stat_pill("--", "#81C784")
        stats_bar.addWidget(self.mode_label)

        stats_bar.addStretch()

        # Model info (right side, subtle)
        self.whisper_model_label = QLabel("--")
        self.whisper_model_label.setStyleSheet("color: #666; font-size: 10px;")
        self.whisper_model_label.setToolTip("Whisper model")
        stats_bar.addWidget(self.whisper_model_label)

        sep = QLabel("·")
        sep.setStyleSheet("color: #444; font-size: 10px; margin: 0 4px;")
        stats_bar.addWidget(sep)

        self.ollama_model_label = QLabel("--")
        self.ollama_model_label.setStyleSheet("color: #666; font-size: 10px;")
        self.ollama_model_label.setToolTip("LLM model")
        stats_bar.addWidget(self.ollama_model_label)

        main_layout.addLayout(stats_bar)

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
                background: rgba(255, 255, 255, 0.05);
                width: 6px;
                border-radius: 3px;
                margin: 4px 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # Caption label
        self.caption_label = QLabel("Press ▶ to start listening...")
        self.caption_label.setWordWrap(True)
        self.caption_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.caption_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.caption_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.95);
                font-size: 16px;
                line-height: 1.5;
                padding: 4px 0;
                background: transparent;
            }
        """)
        
        self.scroll_area.setWidget(self.caption_label)
        main_layout.addWidget(self.scroll_area, 1)

        # Bottom bar with resize grip
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.addStretch()
        
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(16, 16)
        self.size_grip.setStyleSheet("background: transparent;")
        bottom_bar.addWidget(self.size_grip)
        
        main_layout.addLayout(bottom_bar)

        self._setup_animation()

    def _create_stat_pill(self, text: str, color: str) -> QLabel:
        """Create a pill-style stat label."""
        label = QLabel(text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 11px;
                font-weight: 500;
                padding: 3px 10px;
                background: rgba(255, 255, 255, 0.06);
                border-radius: 10px;
            }}
        """)
        return label

    def _update_stat_pill(self, label: QLabel, text: str, color: str):
        """Update a stat pill's text and color."""
        label.setText(text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 11px;
                font-weight: 500;
                padding: 3px 10px;
                background: rgba(255, 255, 255, 0.06);
                border-radius: 10px;
            }}
        """)

    def _icon_button_style(self) -> str:
        """Style for icon buttons."""
        return """
            QPushButton {
                background: rgba(255, 255, 255, 0.08);
                color: #aaa;
                border: none;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.15);
                color: #fff;
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.1);
            }
        """

    def _update_play_button_style(self, is_running: bool):
        """Update play/pause button style based on state."""
        if is_running:
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(244, 67, 54, 0.2);
                    color: #f44336;
                    border: 1px solid rgba(244, 67, 54, 0.3);
                    border-radius: 8px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background: rgba(244, 67, 54, 0.3);
                    border-color: rgba(244, 67, 54, 0.5);
                }
            """)
        else:
            self.start_stop_btn.setStyleSheet("""
                QPushButton {
                    background: rgba(76, 175, 80, 0.2);
                    color: #4CAF50;
                    border: 1px solid rgba(76, 175, 80, 0.3);
                    border-radius: 8px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: rgba(76, 175, 80, 0.3);
                    border-color: rgba(76, 175, 80, 0.5);
                }
            """)

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
        size = overlay_config.get("size", {"width": 700, "height": 180})

        self.move(pos.get("x", 100), pos.get("y", 100))
        self.resize(size.get("width", 700), size.get("height", 180))

        self._window_opacity = overlay_config.get("opacity", 0.88)

        font_size = overlay_config.get("font_size", 18)
        font = QFont("SF Pro Display", font_size)
        font.setWeight(QFont.Weight.Normal)
        self.caption_label.setFont(font)
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
        path.addRoundedRect(0, 0, self.width(), self.height(), 16, 16)

        opacity = int(self._window_opacity * 255)
        painter.fillPath(path, QBrush(QColor(15, 15, 20, opacity)))

        # Subtle border
        painter.setPen(QColor(255, 255, 255, 15))
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
            if self.width() < 800:
                self.resize(900, 250)
            else:
                self.resize(700, 180)

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
            self.start_stop_btn.setText("■")
            self._update_play_button_style(True)
            self.status_label.setText("● Listening")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #4CAF50;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 4px 12px;
                    background: rgba(76, 175, 80, 0.15);
                    border-radius: 12px;
                }
            """)
            self._caption_history = []
            self.caption_label.setText("Listening for audio...")
            # Reset stats
            self._update_stat_pill(self.lang_label, "--", "#888")
            self._update_stat_pill(self.confidence_label, "--%", "#888")
        else:
            self.start_stop_btn.setText("▶")
            self._update_play_button_style(False)
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 12px;
                    font-weight: 500;
                    padding: 4px 12px;
                    background: rgba(255, 255, 255, 0.08);
                    border-radius: 12px;
                }
            """)
            # Reset all stats
            self._update_stat_pill(self.lang_label, "--", "#888")
            self._update_stat_pill(self.confidence_label, "--%", "#888")
            self._update_stat_pill(self.mode_label, "--", "#888")
            self.whisper_model_label.setText("--")
            self.whisper_model_label.setStyleSheet("color: #555; font-size: 10px;")
            self.ollama_model_label.setText("--")
            self.ollama_model_label.setStyleSheet("color: #555; font-size: 10px;")

    def update_stats(self, language: str = None, mode: str = None, confidence: float = None,
                     whisper_model: str = None, ollama_model: str = None):
        """Update the stats bar with current translation info."""
        if language:
            lang_names = {
                "en": "English", "es": "Spanish", "fr": "French", "de": "German",
                "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
                "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
                "hi": "Hindi", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
                "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
            }
            lang_display = lang_names.get(language, language.upper())
            self._update_stat_pill(self.lang_label, f"🌐 {lang_display}", "#64B5F6")
        
        if mode:
            mode_display = {
                "transcribe_only": "📝 Transcribe",
                "ollama": "🤖 Ollama",
                "auto": "⚡ Auto",
                "whisper": "🎯 Whisper",
            }.get(mode, mode)
            self._update_stat_pill(self.mode_label, mode_display, "#81C784")
        
        if confidence is not None:
            conf_pct = int(confidence * 100)
            if conf_pct >= 80:
                color = "#4CAF50"
            elif conf_pct >= 60:
                color = "#FFC107"
            else:
                color = "#FF5722"
            self._update_stat_pill(self.confidence_label, f"{conf_pct}%", color)
        
        if whisper_model:
            self.whisper_model_label.setText(f"🎤 {whisper_model}")
            self.whisper_model_label.setStyleSheet("color: #9575CD; font-size: 10px;")
        
        if ollama_model:
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
        
        if not text or text.startswith("🎧") or text.startswith("Press") or text.startswith("Listening"):
            self.caption_label.setText(text)
            return
        
        if text == "[MUSIC]":
            if not self._caption_history or self._caption_history[-1] != "♪ ♪ ♪":
                self._caption_history.append("♪ ♪ ♪")
            self._update_display()
            return
        
        current_time = time.time()
        silence_duration = current_time - self._last_text_time if self._last_text_time > 0 else 0
        
        def is_complete_sentence(s):
            s = s.strip()
            return s and s[-1] in '.?!。？！'
        
        def should_combine(prev, curr):
            if not prev:
                return False
            prev = prev.strip()
            curr = curr.strip()
            if not is_complete_sentence(prev) and silence_duration < self._paragraph_threshold:
                return True
            if curr and curr[0].islower():
                return True
            return False
        
        if self._caption_history and should_combine(self._last_text, text):
            last_entry = self._caption_history[-1]
            if last_entry and last_entry != "♪ ♪ ♪" and last_entry != "":
                combined = f"{last_entry} {text.strip()}"
                self._caption_history[-1] = combined
                self._last_text = combined
                self._last_text_time = current_time
                self._update_display()
                return
        
        if self._last_text_time > 0 and silence_duration > self._paragraph_threshold:
            if self._caption_history and self._caption_history[-1] != "":
                if is_complete_sentence(self._last_text):
                    self._caption_history.append("")
        
        self._last_text_time = current_time
        self._last_text = text
        
        if not self._caption_history or self._caption_history[-1] != text:
            self._caption_history.append(text)
        
        self._update_display()
    
    def _update_display(self):
        """Update the caption display."""
        if len(self._caption_history) > self._max_history:
            self._caption_history = self._caption_history[-self._max_history:]
        
        display_text = "\n".join(self._caption_history)
        self.caption_label.setText(display_text)
        
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
        self.caption_label.setText("Listening for audio...")
    
    @pyqtSlot(str)
    def set_streaming_text(self, text: str):
        """Update text during streaming translation (shows partial results).
        
        This updates the last line in history with the streaming text,
        giving the user faster feedback during translation.
        """
        if not text:
            return
        
        # Update the last entry or add new one for streaming
        if self._caption_history and self._caption_history[-1].startswith("⏳"):
            # Update existing streaming entry
            self._caption_history[-1] = f"⏳ {text}"
        else:
            # Add new streaming entry
            self._caption_history.append(f"⏳ {text}")
        
        self._update_display()
    
    @pyqtSlot(str)
    def finalize_streaming_text(self, text: str):
        """Finalize streaming text (remove streaming indicator)."""
        if not text:
            return
        
        # Replace streaming entry with final text
        if self._caption_history and self._caption_history[-1].startswith("⏳"):
            self._caption_history[-1] = text
        else:
            self._caption_history.append(text)
        
        self._update_display()
    
    def export_captions(self):
        """Export caption history to a text file."""
        if not self._caption_history:
            QMessageBox.information(
                self,
                "Export Captions",
                "No captions to export yet. Start translating first!"
            )
            return
        
        # Get save location from user
        from datetime import datetime
        default_name = f"captions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Captions",
            default_name,
            "Text Files (*.txt);;SRT Subtitle (*.srt);;All Files (*)"
        )
        
        if not filepath:
            return  # User cancelled
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if filepath.endswith('.srt'):
                    # Export as SRT format
                    for i, caption in enumerate(self._caption_history, 1):
                        if caption and caption != "♪ ♪ ♪":
                            f.write(f"{i}\n")
                            f.write(f"00:00:{i:02d},000 --> 00:00:{i+1:02d},000\n")
                            f.write(f"{caption}\n\n")
                else:
                    # Export as plain text
                    for caption in self._caption_history:
                        if caption:  # Skip empty entries
                            f.write(f"{caption}\n")
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Captions exported to:\n{filepath}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export captions:\n{str(e)}"
            )

    def closeEvent(self, event):
        """Handle window close."""
        pos = self.pos()
        self.config.set("overlay.position.x", pos.x())
        self.config.set("overlay.position.y", pos.y())
        self.config.set("overlay.size.width", self.width())
        self.config.set("overlay.size.height", self.height())
        self.config.save()
        super().closeEvent(event)
