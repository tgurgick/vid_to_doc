"""Pytest configuration and common fixtures."""

import pytest
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def sample_video_path(tmp_path):
    """Create a sample video file path for testing."""
    return tmp_path / "sample_video.mp4"


@pytest.fixture
def sample_audio_path(tmp_path):
    """Create a sample audio file path for testing."""
    return tmp_path / "sample_audio.wav"


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    mock_response = Mock()
    mock_response.__str__ = lambda self: "This is a test transcript."
    return mock_response


@pytest.fixture
def mock_video_clip():
    """Create a mock VideoFileClip."""
    mock_video = Mock()
    mock_video.duration = 60.0
    mock_audio = Mock()
    mock_video.audio = mock_audio
    return mock_video, mock_audio 