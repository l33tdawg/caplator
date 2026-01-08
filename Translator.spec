# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Translator app.
Build with: pyinstaller Translator.spec
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ('config.json', '.'),
        ('src', 'src'),
    ],
    hiddenimports=[
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'sounddevice',
        'faster_whisper',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
        'ollama',
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
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='Translator',
)

app = BUNDLE(
    coll,
    name='Translator.app',
    icon='assets/icon.icns',
    bundle_identifier='com.translator.app',
    info_plist={
        'CFBundleName': 'Translator',
        'CFBundleDisplayName': 'Translator',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Translator needs audio access for transcription.',
        'LSUIElement': False,
    },
)
