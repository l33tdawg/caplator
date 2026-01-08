"""Tests for audio capture module."""

import time
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from src.audio.capture import AudioCapture


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice for testing."""
    with patch('src.audio.capture.sd') as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_sd.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
        ]

        yield mock_sd


class TestAudioCapture:
    """Test cases for AudioCapture class."""

    def test_initialization(self):
        """Test AudioCapture initialization with default parameters."""
        capture = AudioCapture()

        assert capture.sample_rate == 16000
        assert capture.chunk_duration == 2.0
        assert capture.device_name is None
        assert capture.silence_threshold == 500
        assert not capture.is_running()

    def test_initialization_custom_params(self):
        """Test AudioCapture with custom parameters."""
        capture = AudioCapture(
            sample_rate=44100,
            chunk_duration=3.0,
            device_name="Test Device",
            silence_threshold=1000,
        )

        assert capture.sample_rate == 44100
        assert capture.chunk_duration == 3.0
        assert capture.device_name == "Test Device"
        assert capture.silence_threshold == 1000

    def test_get_available_devices(self, mock_sounddevice):
        """Test getting list of available audio devices."""
        capture = AudioCapture()
        devices = capture.get_available_devices()

        assert len(devices) == 2
        assert devices[0] == (0, "Built-in Microphone")
        assert devices[1] == (1, "BlackHole 2ch")

    def test_find_blackhole_device(self, mock_sounddevice):
        """Test finding BlackHole device automatically."""
        capture = AudioCapture()
        device_idx = capture.find_blackhole_device()

        assert device_idx == 1

    def test_find_blackhole_device_not_found(self, mock_sounddevice):
        """Test when BlackHole device is not found."""
        mock_sounddevice.query_devices.return_value = [
            {"name": "Built-in Microphone", "max_input_channels": 2},
            {"name": "USB Audio", "max_input_channels": 2},
        ]

        capture = AudioCapture()
        device_idx = capture.find_blackhole_device()

        assert device_idx is None

    def test_find_device_by_name(self, mock_sounddevice):
        """Test finding device by name."""
        capture = AudioCapture()

        idx = capture.find_device_by_name("blackhole")
        assert idx == 1

        idx = capture.find_device_by_name("microphone")
        assert idx == 0

    def test_is_silent_with_silence(self):
        """Test silence detection with silent audio."""
        capture = AudioCapture(silence_threshold=500)

        silent_audio = b'\x00\x00' * 1000
        assert capture._is_silent(silent_audio) == True

    def test_is_silent_with_audio(self):
        """Test silence detection with actual audio."""
        capture = AudioCapture(silence_threshold=500)

        loud_samples = np.array([10000] * 1000, dtype=np.int16)
        loud_audio = loud_samples.tobytes()

        assert capture._is_silent(loud_audio) == False

    def test_start_stop(self, mock_sounddevice):
        """Test starting and stopping capture."""
        capture = AudioCapture()

        capture.start()
        assert capture.is_running() is True

        time.sleep(0.1)

        capture.stop()
        assert capture.is_running() is False

    def test_double_start(self, mock_sounddevice):
        """Test that double start is handled gracefully."""
        capture = AudioCapture()

        capture.start()
        capture.start()

        assert capture.is_running() is True

        capture.stop()

    def test_double_stop(self, mock_sounddevice):
        """Test that double stop is handled gracefully."""
        capture = AudioCapture()

        capture.start()
        time.sleep(0.1)

        capture.stop()
        capture.stop()

        assert capture.is_running() is False

    def test_frames_per_buffer_calculation(self):
        """Test that frames per emission is calculated correctly."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=2.0)

        expected_chunks = int((16000 * 2.0) / 1024)
        assert capture.chunks_per_emission == expected_chunks
