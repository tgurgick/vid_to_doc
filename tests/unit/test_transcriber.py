"""Unit tests for the Transcriber class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.vid_to_doc.core.transcriber import Transcriber
from src.vid_to_doc.models.exceptions import TranscriptionError


class TestTranscriber:
    """Test cases for Transcriber class."""

    def test_init_with_api_key(self):
        """Test Transcriber initialization with API key."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            transcriber = Transcriber()
            assert transcriber.api_key == "test_key"
            assert transcriber.model == "gpt-4o-transcribe"

    def test_init_without_api_key(self):
        """Test Transcriber initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(TranscriptionError, match="OpenAI API key is required"):
                Transcriber()

    def test_init_with_custom_model(self):
        """Test Transcriber initialization with custom model."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            transcriber = Transcriber(model="whisper-1")
            assert transcriber.model == "whisper-1"

    @patch("src.vid_to_doc.core.transcriber.openai")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_small_file(self, mock_file, mock_exists, mock_stat, mock_openai):
        """Test transcription of a small file."""
        # Setup mocks
        mock_stat.return_value.st_size = 1024 * 1024  # 1MB
        mock_response = Mock()
        mock_response.__str__ = lambda self: "Test transcript"
        mock_openai.audio.transcriptions.create.return_value = mock_response

        # Test
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            transcriber = Transcriber()
            result = transcriber.transcribe(Path("test_audio.wav"))

        # Assertions
        assert result == "Test transcript"
        mock_openai.audio.transcriptions.create.assert_called_once()

    @patch("src.vid_to_doc.core.transcriber.openai")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_large_file(self, mock_file, mock_exists, mock_stat, mock_openai):
        """Test transcription of a large file (chunking)."""
        # Setup mocks
        mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50MB
        mock_response = Mock()
        mock_response.__str__ = lambda self: "Chunk transcript"
        mock_openai.audio.transcriptions.create.return_value = mock_response

        # Calculate chunking to guarantee at least 2 chunks
        class FakeData(list):
            @property
            def itemsize(self):
                return 2  # bytes per sample
        # Make data length so that chunk_count = 2
        max_samples_per_chunk = Transcriber.MAX_BYTES // 2
        fake_data = FakeData([0.1] * (max_samples_per_chunk + 1))  # Just over 1 chunk

        with patch("src.vid_to_doc.core.transcriber.sf") as mock_sf:
            mock_sf.read.return_value = (fake_data, 44100)

            # Test
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
                transcriber = Transcriber()
                result = transcriber.transcribe(Path("test_audio.wav"))

            # Assertions
            assert result == "Chunk transcript\nChunk transcript"
            assert mock_openai.audio.transcriptions.create.call_count == 2

    @patch("src.vid_to_doc.core.transcriber.openai")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_with_custom_model(self, mock_file, mock_exists, mock_stat, mock_openai):
        """Test transcription with custom model."""
        # Setup mocks
        mock_stat.return_value.st_size = 1024 * 1024
        mock_response = Mock()
        mock_response.__str__ = lambda self: "Test transcript"
        mock_openai.audio.transcriptions.create.return_value = mock_response

        # Test
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            transcriber = Transcriber()
            result = transcriber.transcribe(Path("test_audio.wav"), model="whisper-1")

        # Assertions
        assert result == "Test transcript"
        # Check that the call was made with the correct model and response_format
        called_args, called_kwargs = mock_openai.audio.transcriptions.create.call_args
        assert called_kwargs["model"] == "whisper-1"
        assert called_kwargs["response_format"] == "text"
        # The file argument should be a file-like object (from mock_open)
        assert hasattr(called_kwargs["file"], "read")

    @patch("src.vid_to_doc.core.transcriber.openai")
    @patch("pathlib.Path.stat")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_api_error(self, mock_file, mock_exists, mock_stat, mock_openai):
        """Test transcription when API call fails."""
        # Setup mocks
        mock_stat.return_value.st_size = 1024 * 1024
        mock_openai.audio.transcriptions.create.side_effect = Exception("API Error")

        # Test
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"}):
            transcriber = Transcriber()
            with pytest.raises(TranscriptionError, match="Transcription failed"):
                transcriber.transcribe(Path("test_audio.wav"))

    def test_max_bytes_constant(self):
        """Test that MAX_BYTES is correctly calculated."""
        assert Transcriber.MAX_BYTES == 25 * 1024 * 1024  # 25MB in bytes

    def test_default_model_constant(self):
        """Test that DEFAULT_MODEL is set correctly."""
        assert Transcriber.DEFAULT_MODEL == "gpt-4o-transcribe" 