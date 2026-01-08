"""Audio capture module for capturing system audio via BlackHole."""

import threading
import time
import logging
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal

# Setup file logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/translator_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AudioCapture(QObject):
    """Captures audio from system input (BlackHole) and emits chunks for processing."""

    # Signal emitted when audio chunk is ready for processing
    audio_ready = pyqtSignal(bytes)

    # Audio format constants
    CHANNELS = 1
    DTYPE = np.int16
    BYTES_PER_SAMPLE = 2

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        device_name: Optional[str] = None,
        silence_threshold: int = 500,
    ):
        """Initialize audio capture.

        Args:
            sample_rate: Audio sample rate in Hz (default 16000 for speech)
            chunk_duration: Duration of audio chunks in seconds
            device_name: Name of audio device to use (None = auto-detect BlackHole)
            silence_threshold: Amplitude threshold for voice activity detection
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device_name = device_name
        self.silence_threshold = silence_threshold

        # Calculate frames per chunk
        self.frames_per_buffer = 1024
        self.chunks_per_emission = int(
            (sample_rate * chunk_duration) / self.frames_per_buffer
        )

        # Stream
        self._stream: Optional[sd.InputStream] = None

        # Threading
        self._capture_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Buffer for accumulating audio
        self._audio_buffer: List[bytes] = []
        self._chunk_count = 0

    def get_available_devices(self) -> List[Tuple[int, str]]:
        """Get list of available audio input devices.

        Returns:
            List of tuples (device_index, device_name)
        """
        devices = []
        device_list = sd.query_devices()
        for i, dev in enumerate(device_list):
            if dev["max_input_channels"] > 0:
                devices.append((i, dev["name"]))
        return devices

    def find_blackhole_device(self) -> Optional[int]:
        """Find BlackHole audio device index.

        Returns:
            Device index or None if not found
        """
        devices = self.get_available_devices()
        for idx, name in devices:
            if "blackhole" in name.lower():
                return idx
        return None

    def find_device_by_name(self, name: str) -> Optional[int]:
        """Find audio device by name.

        Args:
            name: Device name (partial match)

        Returns:
            Device index or None if not found
        """
        devices = self.get_available_devices()
        name_lower = name.lower()
        for idx, dev_name in devices:
            if name_lower in dev_name.lower():
                return idx
        return None

    def _get_device_index(self) -> Optional[int]:
        """Get the device index to use for capture.

        Returns:
            Device index or None to use default
        """
        if self.device_name:
            return self.find_device_by_name(self.device_name)
        return self.find_blackhole_device()

    def _is_silent(self, audio_data: bytes) -> bool:
        """Check if audio chunk is silent using simple amplitude threshold.

        Args:
            audio_data: Raw audio bytes

        Returns:
            True if audio is below silence threshold
        """
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        amplitude = np.abs(audio_array).mean()
        is_silent = amplitude < self.silence_threshold
        logger.info(f"Amplitude: {amplitude:.1f} (threshold: {self.silence_threshold}) -> {'SILENT' if is_silent else 'SPEECH'}")
        return is_silent

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio status: {status}")

        # Convert to bytes
        audio_bytes = indata.tobytes()
        self._audio_buffer.append(audio_bytes)
        self._chunk_count += 1

        # Emit when we have enough chunks
        if self._chunk_count >= self.chunks_per_emission:
            combined = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            self._chunk_count = 0

            # Only emit if not silent
            if not self._is_silent(combined):
                self.audio_ready.emit(combined)

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        device_index = self._get_device_index()

        try:
            self._stream = sd.InputStream(
                device=device_index,
                channels=self.CHANNELS,
                samplerate=self.sample_rate,
                blocksize=self.frames_per_buffer,
                dtype=self.DTYPE,
                callback=self._audio_callback,
            )

            logger.info(f"Audio capture started (device index: {device_index})")
            self._stream.start()

            while self._is_running:
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}")
        finally:
            self._cleanup_stream()

    def _cleanup_stream(self):
        """Clean up audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def start(self):
        """Start audio capture."""
        if self._is_running:
            return

        self._is_running = True
        self._audio_buffer.clear()
        self._chunk_count = 0
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """Stop audio capture."""
        self._is_running = False

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._cleanup_stream()
        self._audio_buffer.clear()

    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._is_running
