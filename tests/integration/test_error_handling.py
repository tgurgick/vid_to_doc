"""Integration tests for error handling and edge cases."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import requests

from src.vid_to_doc.core.audio_extractor import AudioExtractor
from src.vid_to_doc.core.transcriber import Transcriber
from src.vid_to_doc.core.summarizer import Summarizer
from src.vid_to_doc.core.doc_generator import DocGenerator
from src.vid_to_doc.models.exceptions import (
    AudioExtractionError, 
    TranscriptionError, 
    SummarizationError, 
    DocumentationError,
    APIError
)


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_network_failure_during_transcription(self, temp_dir):
        """Test handling of network failures during transcription."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate network error
                mock_openai.audio.transcriptions.create.side_effect = requests.exceptions.RequestException("Network error")
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Failed to transcribe audio"):
                    transcriber.transcribe(audio_file)

    def test_api_rate_limit_handling(self, temp_dir):
        """Test handling of API rate limits."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate rate limit error
                from openai import RateLimitError
                mock_openai.audio.transcriptions.create.side_effect = RateLimitError(
                    "Rate limit exceeded", 
                    response=Mock(status_code=429), 
                    body=""
                )
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Rate limit exceeded"):
                    transcriber.transcribe(audio_file)

    def test_invalid_api_key_handling(self, temp_dir):
        """Test handling of invalid API keys."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate authentication error
                from openai import AuthenticationError
                mock_openai.audio.transcriptions.create.side_effect = AuthenticationError(
                    "Invalid API key", 
                    response=Mock(status_code=401), 
                    body=""
                )
                
                transcriber = Transcriber(api_key="invalid-key")
                
                with pytest.raises(TranscriptionError, match="Authentication failed"):
                    transcriber.transcribe(audio_file)

    def test_corrupted_video_file_handling(self, temp_dir):
        """Test handling of corrupted video files."""
        video_file = temp_dir / "corrupted_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                # Simulate corrupted file error
                mock_video.side_effect = Exception("Invalid video file format")
                
                extractor = AudioExtractor()
                
                with pytest.raises(AudioExtractionError, match="Failed to extract audio"):
                    extractor.extract_audio(video_file)

    def test_large_file_chunking_failure(self, temp_dir):
        """Test handling of large file chunking failures."""
        audio_file = temp_dir / "large_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
                
                with patch("src.vid_to_doc.core.transcriber.sf") as mock_sf:
                    # Simulate chunking failure
                    mock_sf.read.side_effect = Exception("Failed to read audio file")
                    
                    transcriber = Transcriber(api_key="test-key")
                    
                    with pytest.raises(TranscriptionError, match="Failed to process audio chunks"):
                        transcriber.transcribe(audio_file)

    def test_disk_space_insufficient(self, temp_dir):
        """Test handling of insufficient disk space."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                # Simulate disk space error
                mock_video_instance.audio.write_audiofile.side_effect = OSError("No space left on device")
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                extractor = AudioExtractor()
                
                with pytest.raises(AudioExtractionError, match="Failed to save audio file"):
                    extractor.extract_audio(video_file)

    def test_permission_denied_error(self, temp_dir):
        """Test handling of permission denied errors."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.parent") as mock_parent:
                # Simulate permission error
                mock_parent.mkdir.side_effect = PermissionError("Permission denied")
                
                extractor = AudioExtractor()
                
                with pytest.raises(AudioExtractionError, match="Permission denied"):
                    extractor.extract_audio(video_file)

    def test_malformed_openai_response(self, temp_dir):
        """Test handling of malformed OpenAI API responses."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate malformed response
                mock_response = Mock()
                mock_response.__str__ = lambda self: None  # Malformed response
                mock_openai.audio.transcriptions.create.return_value = mock_response
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Invalid transcription response"):
                    transcriber.transcribe(audio_file)

    def test_empty_transcript_handling(self, temp_dir):
        """Test handling of empty transcription results."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate empty transcription
                mock_response = Mock()
                mock_response.__str__ = lambda self: ""
                mock_openai.audio.transcriptions.create.return_value = mock_response
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Empty transcription result"):
                    transcriber.transcribe(audio_file)

    def test_summarization_with_empty_content(self, temp_dir):
        """Test handling of empty content during summarization."""
        summarizer = Summarizer(api_key="test-key")
        
        with pytest.raises(SummarizationError, match="Content cannot be empty"):
            summarizer.summarize("")

    def test_documentation_generation_failure(self, temp_dir):
        """Test handling of documentation generation failures."""
        summary = "Test summary"
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate documentation generation failure
                mock_openai.chat.completions.create.side_effect = Exception("Documentation generation failed")
                
                doc_generator = DocGenerator(api_key="test-key")
                
                with pytest.raises(DocumentationError, match="Failed to generate documentation"):
                    doc_generator.generate_engineering_doc(summary, video_file)

    def test_concurrent_api_calls_handling(self, temp_dir):
        """Test handling of concurrent API calls."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate concurrent request error
                from openai import ConflictError
                mock_openai.audio.transcriptions.create.side_effect = ConflictError(
                    "Request in progress", 
                    response=Mock(status_code=409), 
                    body=""
                )
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Request in progress"):
                    transcriber.transcribe(audio_file)

    def test_timeout_handling(self, temp_dir):
        """Test handling of API timeouts."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate timeout error
                mock_openai.audio.transcriptions.create.side_effect = requests.exceptions.Timeout("Request timeout")
                
                transcriber = Transcriber(api_key="test-key")
                
                with pytest.raises(TranscriptionError, match="Request timeout"):
                    transcriber.transcribe(audio_file)

    def test_unsupported_video_format(self, temp_dir):
        """Test handling of unsupported video formats."""
        video_file = temp_dir / "unsupported.xyz"
        
        with patch("pathlib.Path.exists", return_value=True):
            extractor = AudioExtractor()
            
            with pytest.raises(AudioExtractionError, match="Unsupported video format"):
                extractor.extract_audio(video_file)

    def test_missing_dependencies_handling(self, temp_dir):
        """Test handling of missing dependencies."""
        video_file = temp_dir / "test_video.mp4"
        
        # Test missing MoviePy
        with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip", side_effect=ImportError("No module named 'moviepy'")):
            extractor = AudioExtractor()
            
            with pytest.raises(AudioExtractionError, match="MoviePy is required"):
                extractor.extract_audio(video_file)

    def test_graceful_degradation_on_partial_failure(self, temp_dir):
        """Test graceful degradation when some components fail."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Audio extraction succeeds, but transcription fails
                with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                    mock_openai.audio.transcriptions.create.side_effect = Exception("Transcription failed")
                    
                    extractor = AudioExtractor()
                    transcriber = Transcriber(api_key="test-key")
                    
                    # Audio extraction should succeed
                    audio_path = extractor.extract_audio(video_file)
                    assert audio_path is not None
                    
                    # Transcription should fail gracefully
                    with pytest.raises(TranscriptionError):
                        transcriber.transcribe(audio_path)

    @pytest.mark.slow
    def test_retry_mechanism_on_temporary_failures(self, temp_dir):
        """Test retry mechanism for temporary failures."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
                # Simulate temporary failure followed by success
                mock_openai.audio.transcriptions.create.side_effect = [
                    requests.exceptions.RequestException("Temporary error"),
                    Mock(__str__=lambda self: "Successful transcription")
                ]
                
                transcriber = Transcriber(api_key="test-key", retry_attempts=2)
                
                # Should succeed after retry
                transcript = transcriber.transcribe(audio_file)
                assert transcript == "Successful transcription"
                assert mock_openai.audio.transcriptions.create.call_count == 2 