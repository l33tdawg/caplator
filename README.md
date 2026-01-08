# Real-Time Audio Translator

A macOS desktop app that captures system audio, transcribes and translates it in real-time, and displays captions in a beautiful floating overlay.

**100% local. 100% free. 100% private.**

No API keys. No cloud services. No subscription costs. Everything runs on your machine.

![Translator Screenshot](screenshot.png)

## Features

- 🎧 **Real-time transcription & translation** - See captions as people speak
- 🌍 **Multi-language support** - Translate from any language to English (and more)
- 🚀 **Fast local processing** - Uses optimized Whisper for speech-to-text
- 📜 **Scrollable caption history** - Scroll back to see previous translations
- 🧠 **Context-aware** - Uses previous sentences for better accuracy
- 📝 **Smart sentence combining** - Handles "um", "ah" pauses gracefully
- ♪ **Music detection** - Shows indicator when music is playing (no speech)
- 🎨 **Beautiful overlay** - Draggable, resizable, semi-transparent window
- ⚙️ **Configurable** - Adjust model size, chunk duration, appearance

## How It Works

```
System Audio → BlackHole → Whisper → Captions
   (VLC, Zoom, etc.)  (virtual audio)  (transcribe + translate)
```

For **English output**: Whisper handles both transcription AND translation in a single pass (fastest!)

For **other languages**: Whisper transcribes → Ollama translates

## Quick Start

### 1. Install BlackHole (Audio Routing)

```bash
brew install blackhole-16ch
```

Then set up Multi-Output Device:
1. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
2. Click **+** → **Create Multi-Output Device**
3. Check both:
   - ✅ Your speakers/headphones
   - ✅ BlackHole 16ch
4. Set **BlackHole 16ch** as the **Master Device** (dropdown at top)
5. Go to **System Settings → Sound → Output** → Select **Multi-Output Device**

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

1. **Click ▶ Start** to begin capturing and translating
2. **Play audio** from any app (VLC, YouTube, Zoom, etc.)
3. **Watch captions** appear in real-time
4. **Scroll up** to see previous translations
5. **Drag** the overlay to reposition
6. **Resize** by dragging the corner
7. **Click ⚙** for settings

## Configuration

Edit `config.json` or use the Settings dialog:

```json
{
  "target_language": "English",
  "whisper_model": "small",
  "whisper_backend": "faster-whisper",
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
| `tiny` | 75MB | ⚡⚡⚡⚡ | ★★☆☆ | Low-end hardware |
| `base` | 145MB | ⚡⚡⚡ | ★★★☆ | Good balance |
| **`small`** | 488MB | ⚡⚡ | ★★★★ | **Recommended** |
| `medium` | 1.5GB | ⚡ | ★★★★★ | High accuracy |
| `large-v3` | 3GB | 🐌 | ★★★★★ | Best accuracy (GPU) |

### Chunk Duration

| Duration | Latency | Accuracy | Feel |
|----------|---------|----------|------|
| 1.5s | Low | Lower | Fast but choppy |
| 2.0s | Medium | Good | Balanced |
| **3.0s** | Higher | Better | **Recommended** |
| 4.0s+ | High | Best | Feels delayed |

### Whisper Backend

- `faster-whisper` - **Default**, works on all platforms
- `mlx` - Optimized for Apple Silicon (M1/M2/M3), experimental

## Smart Features

### Context-Aware Transcription
The app passes recent transcriptions to Whisper as context, improving:
- Recognition of technical terms
- Consistency of names and proper nouns
- Overall coherence

### Smart Sentence Combining
Automatically combines sentence fragments:
- "like a drum?" + "Then we can tune it" → Combined if no complete sentence
- Won't break on filler words like "um", "ah", "you know"

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
2. For English output, Ollama is skipped automatically
3. Reduce chunk duration to 2.0s (less accurate but faster)

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

## Requirements

- **macOS 12+** (tested on macOS 14+)
- **Python 3.10+** (for development/building only)
- **BlackHole 16ch** virtual audio driver
- **~1GB disk space** for app + Whisper model

## License

MIT License - Use freely!

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [Ollama](https://ollama.ai) - Local LLM runner
- [BlackHole](https://existential.audio/blackhole/) - Virtual audio driver
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
