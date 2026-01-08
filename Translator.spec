# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Translator app.
Build with: pyinstaller Translator.spec
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

# Find PyQt6 location for plugins
import PyQt6
pyqt6_path = Path(PyQt6.__file__).parent
qt_plugins_path = pyqt6_path / "Qt6" / "plugins"
qt_lib_path = pyqt6_path / "Qt6" / "lib"

# Collect Qt plugins
qt_plugins = []
if qt_plugins_path.exists():
    for plugin_dir in ["platforms", "styles", "imageformats", "iconengines"]:
        plugin_path = qt_plugins_path / plugin_dir
        if plugin_path.exists():
            for plugin_file in plugin_path.glob("*.dylib"):
                qt_plugins.append((str(plugin_file), f"PyQt6/Qt6/plugins/{plugin_dir}"))

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=qt_plugins,
    datas=[
        ('config.json', '.'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        'numpy',
        'sounddevice',
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'ollama',
        'httpx',
        'httpcore',
        'anyio',
        'src',
        'src.audio',
        'src.audio.capture',
        'src.providers',
        'src.providers.base',
        'src.providers.translator',
        'src.ui',
        'src.ui.overlay',
        'src.ui.settings',
        'src.ui.tray',
        'src.utils',
        'src.utils.config',
    ],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=['hooks/hook-pyqt6.py'],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'PIL',
        'cv2',
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Translator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disable UPX on macOS - can cause issues
    console=False,  # No terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Translator',
)

app = BUNDLE(
    coll,
    name='Translator.app',
    icon='assets/icon.icns' if (project_root / 'assets' / 'icon.icns').exists() else None,
    bundle_identifier='com.translator.app',
    info_plist={
        'CFBundleName': 'Translator',
        'CFBundleDisplayName': 'Real-Time Translator',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Translator needs microphone access for audio capture.',
        'LSUIElement': False,
        'NSRequiresAquaSystemAppearance': False,  # Support dark mode
    },
)
