"""
Setup script for building macOS app bundle using py2app.

Usage:
    python setup.py py2app

This creates a standalone .app bundle in the dist/ folder.
"""

from setuptools import setup

APP = ['main.py']
APP_NAME = 'Translator'

DATA_FILES = [
    ('', ['config.json']),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'assets/icon.icns',
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleIdentifier': 'com.translator.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'LSBackgroundOnly': False,
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Translator needs microphone access to capture audio for transcription.',
        'LSUIElement': False,  # Show in dock
    },
    'packages': [
        'PyQt6',
        'numpy',
        'sounddevice',
        'faster_whisper',
        'ollama',
        'ctranslate2',
        'huggingface_hub',
        'tokenizers',
    ],
    'includes': [
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
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy',
        'PIL',
        'cv2',
    ],
    'resources': [],
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
