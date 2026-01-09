# Real-Time Audio Translator

A macOS desktop app that captures system audio, transcribes and translates it in real-time, and displays captions in a beautiful floating overlay.

**100% local. 100% free. 100% private.**

No API keys. No cloud services. No subscription costs. Everything runs on your machine.

![Translator Screenshot](screenshot.png)

## Features

- **Real-time transcription and translation** - See captions as people speak
- **Multi-language support** - Translate from any language to English (and more)
- **Fast local processing** - Uses optimized Whisper for speech-to-text
- **Scrollable caption history** - Scroll back to see previous translations
- **Context-aware** - Uses previous sentences for better accuracy
- **Smart sentence combining** - Handles "um", "ah" pauses gracefully
- **Music detection** - Shows indicator when music is playing (no speech)
- **Beautiful overlay** - Draggable, resizable, semi-transparent window
- **Highly configurable** - Adjust model size, chunk duration, appearance

## What's New in v1.1.0

### Accuracy Improvements

- **Higher beam search** - Default beam_size increased from 1 to 3 for significantly better transcription accuracy
- **Larger context window** - Now uses 7 previous sentences (up from 3) for better contextual understanding
- **High-quality audio resampling** - Uses scipy polyphase resampling instead of linear interpolation
- **Enhanced translation prompts** - Few-shot examples improve Ollama translation quality
- **RMS-based silence detection** - More robust speech/silence classification

### New Features

- **Quality Profiles** - Choose from Fast, Balanced, or Accurate presets in settings
- **Streaming Translation** - See translations appear word-by-word as they're generated
- **Audio Normalization** - Automatic gain adjustment for quiet audio sources
- **Caption Export** - Save your caption history to text or SRT subtitle files
- **Keyboard Shortcuts**:
  - `Ctrl+Shift+T` or `Space` - Toggle translation on/off
  - `Ctrl+Shift+C` - Clear history
  - `Ctrl+Shift+S` - Export captions
  - `Ctrl+,` - Open settings
  - `Escape` - Stop translation

### Quality Profiles

| Profile | Beam Size | Chunk Duration | Best For |
|---------|-----------|----------------|----------|
| Fast | 1 | 2s | Live conversations |
| Balanced | 3 | 3s | Movies/videos (default) |
| Accurate | 5 | 4s | Important recordings |

## How It Works

```
System Audio -> BlackHole -> Whisper -> Captions
   (VLC, Zoom, etc.)  (virtual audio)  (transcribe + translate)
```

For **English output**: Whisper handles both transcription AND translation in a single pass (fastest!)

For **other languages**: Whisper transcribes, then Ollama translates

## Quick Start

### 1. Install BlackHole (Audio Routing)

```bash
brew install blackhole-16ch
```

Then set up Multi-Output Device:
1. Open **Audio MIDI Setup** (Spotlight -> "Audio MIDI Setup")
2. Click **+** -> **Create Multi-Output Device**
3. Check both:
   - Your speakers/headphones
   - BlackHole 16ch
4. Set **BlackHole 16ch** as the **Master Device** (dropdown at top)
5. Go to **System Settings -> Sound -> Output** -> Select **Multi-Output Device**

### 2. Install Ollama (Optional - only for non-English targets)

```bash
brew install ollama
ollama pull llama3.2
ollama serve  # Run in background
```

### 3. Install the App

```bash
# Clone and enter directory
cd translator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Run

```bash
python main.py
```

## Usage

1. **Click the Start button** to begin capturing and translating
2. **Play audio** from any app (VLC, YouTube, Zoom, etc.)
3. **Watch captions** appear in real-time
4. **Scroll up** to see previous translations
5. **Drag** the overlay to reposition
6. **Resize** by dragging the corner
7. **Click the gear icon** for settings
8. **Click the save icon** to export captions

## Configuration

Edit `config.json` or use the Settings dialog:

```json
{
  "target_language": "English",
  "whisper_model": "small",
  "whisper_backend": "faster-whisper",
  "quality_profile": "balanced",
  "whisper_beam_size": 3,
  "whisper_best_of": 2,
  "audio": {
    "sample_rate": 44100,
    "chunk_duration": 3.0,
    "silence_threshold": 100
  }
}
```

### Whisper Models

| Model | Size | Speed | Accuracy | Recommendation |
|-------|------|-------|----------|----------------|
| `tiny` | 75MB | Fastest | Low | Low-end hardware |
| `base` | 145MB | Fast | Medium | Good balance |
| **`small`** | 488MB | Medium | High | **Default** |
| `medium` | 1.5GB | Slow | Very High | High accuracy |
| `large-v3` | 3GB | Slowest | Best | Best accuracy (needs GPU) |
| **`large-v3-turbo`** | 809MB | Fast | Near-best | **Recommended for quality** |

**New: large-v3-turbo** - Released by OpenAI in late 2024, this model offers near-best accuracy at 8x the speed of large-v3, with only half the memory requirements (~6GB VRAM vs ~10GB). It's the sweet spot for users who want high quality without the slowdown of large-v3.

### Chunk Duration

| Duration | Latency | Accuracy | Feel |
|----------|---------|----------|------|
| 1.5s | Low | Lower | Fast but choppy |
| 2.0s | Medium | Good | Balanced |
| **3.0s** | Higher | Better | **Recommended** |
| 4.0s+ | High | Best | Feels delayed |

### Whisper Backend

- `faster-whisper` - **Recommended**, works on all platforms, very stable
- `mlx` - Apple Silicon only, known to cause crashes - NOT RECOMMENDED

## Smart Features

### Context-Aware Transcription
The app passes up to 7 recent transcriptions to Whisper as context, improving:
- Recognition of technical terms
- Consistency of names and proper nouns
- Overall coherence

### Smart Sentence Combining
Automatically combines sentence fragments:
- "like a drum?" + "Then we can tune it" -> Combined if no complete sentence
- Won't break on filler words like "um", "ah", "you know"

### Audio Normalization
Automatically boosts quiet audio sources (up to 10x gain) for better transcription of low-volume content.

### Hallucination Filtering
Detects and filters common Whisper hallucinations like "Thank you for watching", music notation symbols, and other artifacts.

### Paragraph Breaks
Only inserts paragraph breaks when:
- Previous sentence ends with `.?!`
- AND there's 5+ seconds of silence

## Troubleshooting

### No audio captured
```bash
# Check BlackHole is installed
brew list blackhole-16ch

# Verify in Audio MIDI Setup that Multi-Output Device includes BlackHole
```

Make sure your app (VLC, etc.) is using the system audio output, not a specific device.

### Translation is slow
1. Use a smaller Whisper model: `"whisper_model": "base"`
2. Switch to "Fast" quality profile in settings
3. For English output, Ollama is skipped automatically
4. Reduce chunk duration to 2.0s (less accurate but faster)

### App crashes on settings save
This was fixed. If it happens, restart the app.

### Whisper model downloading
First run downloads the model (~500MB for `small`). This is a one-time download.

## Project Structure

```
translator/
├── main.py              # App entry point
├── config.json          # User configuration
├── requirements.txt     # Python dependencies
├── tests/               # Unit tests
│   └── test_improvements.py  # Tests for v1.1 features
└── src/
    ├── audio/
    │   └── capture.py   # Audio capture via BlackHole
    ├── providers/
    │   ├── base.py      # Provider interface
    │   └── translator.py # Whisper + Ollama integration
    ├── ui/
    │   ├── overlay.py   # Floating caption window
    │   ├── settings.py  # Settings dialog
    │   └── tray.py      # System tray (optional)
    └── utils/
        └── config.py    # Configuration management
```

## Building the App

To create a standalone `.app` bundle you can double-click:

```bash
# Install build dependencies
pip3 install pyinstaller pillow

# Create the icon
python3 scripts/create_icon.py

# Build the app
pyinstaller Translator.spec --noconfirm

# The app will be in dist/Translator.app
open dist/Translator.app
```

To install to Applications:
```bash
cp -r dist/Translator.app /Applications/
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only the v1.1 improvement tests
python -m pytest tests/test_improvements.py -v
```

## Requirements

- **macOS 12+** (tested on macOS 14+)
- **Python 3.10+** (for development/building only)
- **BlackHole 16ch** virtual audio driver
- **~1GB disk space** for app + Whisper model

## Dependencies

- PyQt6 - GUI framework
- faster-whisper - Optimized Whisper implementation
- ollama - Local LLM client
- sounddevice - Audio capture
- numpy - Audio processing
- scipy - High-quality audio resampling

## License

MIT License - Use freely!

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [Ollama](https://ollama.ai) - Local LLM runner
- [BlackHole](https://existential.audio/blackhole/) - Virtual audio driver
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework

## Changelog

### v1.2.0 (2026-01-09)
- Added Whisper large-v3-turbo model support (8x faster than large-v3, near-best accuracy)
- Updated MLX backend warnings (known crash issues, not recommended)
- Documentation improvements for model selection

### v1.1.0 (2026-01-09)
- Added quality profiles (Fast/Balanced/Accurate)
- Increased default beam_size from 1 to 3 for better accuracy
- Added streaming translation display
- Added caption export to TXT/SRT files
- Added keyboard shortcuts
- Improved audio resampling with scipy
- Added RMS-based silence detection
- Added audio normalization for quiet sources
- Enhanced Ollama prompts with few-shot examples
- Expanded context window from 3 to 7 sentences
- Added comprehensive unit tests

### v1.0.0
- Initial release
