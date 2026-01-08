"""System tray icon for the Real-Time Audio Translator."""

from PyQt6.QtWidgets import (
    QSystemTrayIcon,
    QMenu,
    QApplication,
)
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen

from .settings import SettingsDialog


class SystemTray(QObject):
    """System tray icon with controls for the translator."""

    # Signals
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    settings_changed = pyqtSignal()

    def __init__(self, config, overlay_window, parent=None):
        """Initialize system tray.

        Args:
            config: Application configuration
            overlay_window: Reference to overlay window
            parent: Parent object
        """
        super().__init__(parent)

        self.config = config
        self.overlay = overlay_window
        self._is_running = False

        self._create_tray_icon()
        self._create_menu()

    def _create_tray_icon(self):
        """Create the system tray icon."""
        self.tray_icon = QSystemTrayIcon()
        self.tray_icon.setIcon(self._create_icon(running=False))
        self.tray_icon.setToolTip("Real-Time Translator")

        # Double-click to toggle
        self.tray_icon.activated.connect(self._on_activated)

    def _create_icon(self, running: bool = False) -> QIcon:
        """Create a simple icon for the tray.

        Args:
            running: Whether translation is active

        Returns:
            QIcon for the tray
        """
        # Create a simple colored circle icon
        size = 64
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw circle
        if running:
            # Green when running
            painter.setBrush(QBrush(QColor(50, 205, 50)))
            painter.setPen(QPen(QColor(30, 150, 30), 2))
        else:
            # Gray when stopped
            painter.setBrush(QBrush(QColor(128, 128, 128)))
            painter.setPen(QPen(QColor(100, 100, 100), 2))

        margin = 4
        painter.drawEllipse(margin, margin, size - 2 * margin, size - 2 * margin)

        # Draw "T" for translator
        painter.setPen(QPen(QColor(255, 255, 255), 3))
        font = painter.font()
        font.setPixelSize(32)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), 0x0084, "T")  # AlignCenter

        painter.end()

        return QIcon(pixmap)

    def _create_menu(self):
        """Create the context menu."""
        self.menu = QMenu()

        # Start/Stop action
        self.start_stop_action = self.menu.addAction("Start Translation")
        self.start_stop_action.triggered.connect(self._toggle_translation)

        self.menu.addSeparator()

        # Show/Hide overlay
        self.show_overlay_action = self.menu.addAction("Show Overlay")
        self.show_overlay_action.setCheckable(True)
        self.show_overlay_action.setChecked(True)
        self.show_overlay_action.triggered.connect(self._toggle_overlay)

        self.menu.addSeparator()

        # Settings
        settings_action = self.menu.addAction("Settings...")
        settings_action.triggered.connect(self._show_settings)

        self.menu.addSeparator()

        # Quit
        quit_action = self.menu.addAction("Quit")
        quit_action.triggered.connect(self._quit)

        self.tray_icon.setContextMenu(self.menu)

    def _on_activated(self, reason):
        """Handle tray icon activation.

        Args:
            reason: Activation reason
        """
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._toggle_translation()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            # Single click on macOS shows menu
            pass

    def _toggle_translation(self):
        """Toggle translation on/off."""
        if self._is_running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _toggle_overlay(self, checked: bool):
        """Toggle overlay visibility.

        Args:
            checked: Whether to show overlay
        """
        if checked:
            self.overlay.show()
        else:
            self.overlay.hide()

    def _show_settings(self):
        """Show the settings dialog."""
        # Pass overlay as parent so dialog closing doesn't quit the app
        dialog = SettingsDialog(self.config, parent=self.overlay)
        dialog.settings_saved.connect(self.settings_changed.emit)
        dialog.exec()

    def _quit(self):
        """Quit the application."""
        # Stop translation if running
        if self._is_running:
            self.stop_requested.emit()

        # Hide tray icon
        self.tray_icon.hide()

        # Quit application
        QApplication.quit()

    def set_running(self, running: bool):
        """Update the running state.

        Args:
            running: Whether translation is active
        """
        self._is_running = running
        self.tray_icon.setIcon(self._create_icon(running=running))

        if running:
            self.start_stop_action.setText("Stop Translation")
            self.tray_icon.setToolTip("Real-Time Translator (Running)")
        else:
            self.start_stop_action.setText("Start Translation")
            self.tray_icon.setToolTip("Real-Time Translator (Stopped)")

    def show(self):
        """Show the tray icon."""
        self.tray_icon.show()

    def hide(self):
        """Hide the tray icon."""
        self.tray_icon.hide()

    def show_message(self, title: str, message: str, icon=None):
        """Show a notification message.

        Args:
            title: Message title
            message: Message body
            icon: Message icon type
        """
        if icon is None:
            icon = QSystemTrayIcon.MessageIcon.Information

        self.tray_icon.showMessage(title, message, icon, 3000)
