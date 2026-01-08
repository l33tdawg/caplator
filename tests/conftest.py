"""Pytest configuration and shared fixtures."""

import os
import sys
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_audio_data():
    """Create sample audio data (1 second of silence)."""
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    return b'\x00\x00' * num_samples


@pytest.fixture
def sample_wav_data(sample_audio_data):
    """Create sample WAV file bytes."""
    sample_rate = 16000
    channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(sample_audio_data)
    file_size = 36 + data_size

    wav_header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b'data',
        data_size
    )

    return wav_header + sample_audio_data


@pytest.fixture
def mock_pyaudio():
    """Mock PyAudio for testing audio capture."""
    with patch('pyaudio.PyAudio') as mock_pa:
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00\x00' * 1024

        mock_instance = MagicMock()
        mock_instance.get_device_count.return_value = 2
        mock_instance.get_device_info_by_index.side_effect = [
            {"name": "Built-in Microphone", "maxInputChannels": 2},
            {"name": "BlackHole 2ch", "maxInputChannels": 2},
        ]
        mock_instance.open.return_value = mock_stream
        mock_pa.return_value = mock_instance

        yield mock_pa
