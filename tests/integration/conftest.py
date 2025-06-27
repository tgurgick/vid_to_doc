"""Pytest configuration for integration tests."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a test data directory for integration tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_data"
        test_dir.mkdir()
        
        # Create sample files for testing
        (test_dir / "videos").mkdir()
        (test_dir / "audio").mkdir()
        (test_dir / "output").mkdir()
        
        yield test_dir


@pytest.fixture(scope="session")
def sample_video_files(test_data_dir):
    """Create sample video files for testing."""
    video_dir = test_data_dir / "videos"
    
    # Create mock video files
    video_files = [
        video_dir / "sample1.mp4",
        video_dir / "sample2.avi",
        video_dir / "sample3.mov",
    ]
    
    for video_file in video_files:
        video_file.touch()  # Create empty files
    
    return video_files


@pytest.fixture(scope="session")
def sample_audio_files(test_data_dir):
    """Create sample audio files for testing."""
    audio_dir = test_data_dir / "audio"
    
    # Create mock audio files
    audio_files = [
        audio_dir / "sample1.wav",
        audio_dir / "sample2.mp3",
        audio_dir / "sample3.flac",
    ]
    
    for audio_file in audio_files:
        audio_file.touch()  # Create empty files
    
    return audio_files


@pytest.fixture(scope="session")
def mock_openai_responses():
    """Provide consistent mock OpenAI responses for integration tests."""
    return {
        "transcription": "This is a test transcription generated for integration testing purposes.",
        "summary": "Test summary of the video content with key points and insights.",
        "engineering_doc": "# Engineering Documentation\n\n## Overview\n\nThis is test engineering documentation.",
        "meeting_notes": "# Meeting Notes\n\n## Agenda\n\n- Test agenda item 1\n- Test agenda item 2",
        "action_items": "# Action Items\n\n## Tasks\n\n- [ ] Test task 1\n- [ ] Test task 2",
    }


@pytest.fixture(scope="function")
def mock_openai_service(mock_openai_responses):
    """Mock OpenAI service for integration tests."""
    with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.__str__ = lambda self: mock_openai_responses["transcription"]
        mock_openai.audio.transcriptions.create.return_value = mock_transcription
        
        # Mock chat completion responses
        mock_summary_response = Mock()
        mock_summary_response.choices = [Mock(message=Mock(content=mock_openai_responses["summary"]))]
        
        mock_eng_doc_response = Mock()
        mock_eng_doc_response.choices = [Mock(message=Mock(content=mock_openai_responses["engineering_doc"]))]
        
        mock_meeting_response = Mock()
        mock_meeting_response.choices = [Mock(message=Mock(content=mock_openai_responses["meeting_notes"]))]
        
        mock_action_response = Mock()
        mock_action_response.choices = [Mock(message=Mock(content=mock_openai_responses["action_items"]))]
        
        # Set up side effect for different document types
        mock_openai.chat.completions.create.side_effect = [
            mock_summary_response,    # For summarization
            mock_eng_doc_response,    # For engineering docs
            mock_meeting_response,    # For meeting notes
            mock_action_response,     # For action items
        ]
        
        yield mock_openai


@pytest.fixture(scope="function")
def mock_moviepy():
    """Mock MoviePy for video processing tests."""
    with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
        mock_video_instance = Mock()
        mock_video_instance.audio = Mock()
        mock_video_instance.audio.write_audiofile = Mock()
        mock_video.return_value.__enter__.return_value = mock_video_instance
        
        yield mock_video


@pytest.fixture(scope="function")
def mock_soundfile():
    """Mock soundfile for audio processing tests."""
    with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
        # Mock audio data (1 second, 44.1kHz, mono)
        mock_sf.read.return_value = ([0.1] * 44100, 44100)
        
        yield mock_sf


@pytest.fixture(scope="function")
def temp_working_dir():
    """Create a temporary working directory for each test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        yield Path(temp_dir)
        
        os.chdir(original_cwd)


@pytest.fixture(scope="function")
def mock_file_system():
    """Mock file system operations for integration tests."""
    with patch("pathlib.Path.exists", return_value=True) as mock_exists, \
         patch("pathlib.Path.stat") as mock_stat, \
         patch("pathlib.Path.mkdir") as mock_mkdir, \
         patch("pathlib.Path.unlink") as mock_unlink:
        
        # Set up default stat values
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        
        yield {
            "exists": mock_exists,
            "stat": mock_stat,
            "mkdir": mock_mkdir,
            "unlink": mock_unlink,
        }


@pytest.fixture(scope="function")
def test_config():
    """Provide test configuration for integration tests."""
    return {
        "openai": {
            "api_key": "test-api-key",
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
        },
        "output": {
            "format": "markdown",
            "directory": "./output",
            "include_timestamps": True,
        },
        "transcription": {
            "chunk_size_mb": 25,
            "retry_attempts": 3,
        },
    }


@pytest.fixture(scope="function")
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-env-api-key",
        "VID_TO_DOC_CONFIG_PATH": "/tmp/test_config.yaml",
        "VID_TO_DOC_LOG_LEVEL": "DEBUG",
    }):
        yield


# Markers for different types of integration tests
def pytest_configure(config):
    """Configure pytest markers for integration tests."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI integration tests"
    )
    config.addinivalue_line(
        "markers", "pipeline: marks tests as full pipeline integration tests"
    )


# Skip slow tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Skip slow tests by default."""
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow) 