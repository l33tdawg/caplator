"""
py2app setup script for building Translator.app

Usage:
    python setup_app.py py2app
"""

from setuptools import setup

APP = ['main.py']
DATA_FILES = [
    ('', ['config.json']),
]

OPTIONS = {
    'argv_emulation': False,
    'iconfile': 'assets/icon.icns',
    'plist': {
        'CFBundleName': 'Translator',
        'CFBundleDisplayName': 'Real-Time Translator',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '12.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'Translator needs audio access for transcription.',
        'LSUIElement': False,
        'NSRequiresAquaSystemAppearance': False,
    },
    'packages': [
        'PyQt6',
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
    ],
    'includes': [
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
        'PyQt6.sip',
    ],
    'excludes': [
        'tkinter',
        'matplotlib',
        'scipy',
        'PIL',
        'cv2',
        'IPython',
        'jupyter',
        'mlx_whisper',
        'mlx',
    ],
    'semi_standalone': False,
}

setup(
    name='Translator',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
