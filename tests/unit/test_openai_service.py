"""Unit tests for the OpenAI service wrapper."""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.vid_to_doc.services.openai_service import OpenAIService
from src.vid_to_doc.models.exceptions import (
    APIError, 
    TranscriptionError, 
    SummarizationError, 
    DocumentationError
)


class TestOpenAIService:
    """Test cases for OpenAI service wrapper."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        service = OpenAIService(api_key="test-key")
        assert service.api_key == "test-key"
        assert service.models["transcription"] == "gpt-4o-transcribe"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_init_with_env_api_key(self):
        """Test initialization with environment API key."""
        service = OpenAIService()
        assert service.api_key == "env-key"

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(APIError, match="OpenAI API key is required"):
                OpenAIService()

    def test_init_with_custom_models(self):
        """Test initialization with custom models."""
        custom_models = {
            "transcription": "whisper-1",
            "summarization": "gpt-4",
            "documentation": "gpt-4"
        }
        service = OpenAIService(api_key="test-key", default_models=custom_models)
        assert service.models == custom_models

    def test_init_with_custom_retry_config(self):
        """Test initialization with custom retry configuration."""
        service = OpenAIService(
            api_key="test-key", 
            retry_attempts=5, 
            retry_delay=3
        )
        assert service.retry_attempts == 5
        assert service.retry_delay == 3

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_transcribe_audio_success(self, mock_openai):
        """Test successful audio transcription."""
        mock_response = Mock()
        mock_response.__str__ = lambda self: "Transcribed text"
        mock_openai.audio.transcriptions.create.return_value = mock_response
        
        service = OpenAIService(api_key="test-key")
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                result = service.transcribe_audio(Path("test.wav"))
                assert result == "Transcribed text"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_transcribe_audio_with_custom_model(self, mock_openai):
        """Test transcription with custom model."""
        mock_response = Mock()
        mock_response.__str__ = lambda self: "Transcribed text"
        mock_openai.audio.transcriptions.create.return_value = mock_response
        
        service = OpenAIService(api_key="test-key")
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                result = service.transcribe_audio(Path("test.wav"), model="whisper-1")
                assert result == "Transcribed text"
                mock_openai.audio.transcriptions.create.assert_called_once()

    def test_transcribe_audio_file_not_exists(self):
        """Test transcription with non-existent file."""
        service = OpenAIService(api_key="test-key")
        
        with pytest.raises(TranscriptionError, match="Audio file does not exist"):
            service.transcribe_audio(Path("nonexistent.wav"))

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_transcribe_audio_with_retry(self, mock_openai):
        """Test transcription with retry logic."""
        # First call fails, second succeeds
        mock_openai.audio.transcriptions.create.side_effect = [
            Exception("API Error"),
            Mock(__str__=lambda self: "Transcribed text")
        ]
        
        service = OpenAIService(api_key="test-key", retry_attempts=2, retry_delay=0)
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                with patch("time.sleep"):  # Mock sleep to speed up test
                    result = service.transcribe_audio(Path("test.wav"))
                    assert result == "Transcribed text"
                    assert mock_openai.audio.transcriptions.create.call_count == 2

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_transcribe_audio_max_retries_exceeded(self, mock_openai):
        """Test transcription with max retries exceeded."""
        mock_openai.audio.transcriptions.create.side_effect = Exception("API Error")
        
        service = OpenAIService(api_key="test-key", retry_attempts=2, retry_delay=0)
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                with patch("time.sleep"):  # Mock sleep to speed up test
                    with pytest.raises(TranscriptionError, match="Transcription failed after 2 attempts"):
                        service.transcribe_audio(Path("test.wav"))

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_generate_summary_success(self, mock_openai):
        """Test successful summary generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated summary"))]
        mock_openai.chat.completions.create.return_value = mock_response
        
        service = OpenAIService(api_key="test-key")
        
        result = service.generate_summary("Test text")
        assert result == "Generated summary"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_generate_summary_with_custom_prompt(self, mock_openai):
        """Test summary generation with custom prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Custom summary"))]
        mock_openai.chat.completions.create.return_value = mock_response
        
        service = OpenAIService(api_key="test-key")
        
        result = service.generate_summary(
            "Test text", 
            system_prompt="Custom prompt",
            model="gpt-4",
            max_tokens=1000
        )
        assert result == "Custom summary"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_generate_summary_with_retry(self, mock_openai):
        """Test summary generation with retry logic."""
        mock_openai.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Generated summary"))])
        ]
        
        service = OpenAIService(api_key="test-key", retry_attempts=2, retry_delay=0)
        
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = service.generate_summary("Test text")
            assert result == "Generated summary"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_generate_documentation_success(self, mock_openai):
        """Test successful documentation generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated documentation"))]
        mock_openai.chat.completions.create.return_value = mock_response
        
        service = OpenAIService(api_key="test-key")
        
        result = service.generate_documentation("Test content", "Documentation prompt")
        assert result == "Generated documentation"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_generate_documentation_with_retry(self, mock_openai):
        """Test documentation generation with retry logic."""
        mock_openai.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Generated documentation"))])
        ]
        
        service = OpenAIService(api_key="test-key", retry_attempts=2, retry_delay=0)
        
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = service.generate_documentation("Test content", "Documentation prompt")
            assert result == "Generated documentation"

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_get_usage_stats(self, mock_openai):
        """Test getting usage statistics."""
        service = OpenAIService(api_key="test-key")
        
        result = service.get_usage_stats()
        assert "note" in result
        assert result["api_key"] == "test-key..."

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_validate_api_key_success(self, mock_openai):
        """Test successful API key validation."""
        mock_openai.models.list.return_value = Mock(data=[Mock(id="gpt-3.5-turbo")])
        
        service = OpenAIService(api_key="test-key")
        
        result = service.validate_api_key()
        assert result is True

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_validate_api_key_failure(self, mock_openai):
        """Test API key validation failure."""
        mock_openai.models.list.side_effect = Exception("Invalid API key")
        
        service = OpenAIService(api_key="test-key")
        
        with pytest.raises(APIError, match="Invalid API key"):
            service.validate_api_key()

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_get_available_models(self, mock_openai):
        """Test getting available models."""
        mock_models = [Mock(id="gpt-3.5-turbo"), Mock(id="gpt-4")]
        mock_openai.models.list.return_value = Mock(data=mock_models)
        
        service = OpenAIService(api_key="test-key")
        
        result = service.get_available_models()
        assert result == ["gpt-3.5-turbo", "gpt-4"]

    @patch("src.vid_to_doc.services.openai_service.openai")
    def test_get_available_models_error(self, mock_openai):
        """Test getting available models with error."""
        mock_openai.models.list.side_effect = Exception("API Error")
        
        service = OpenAIService(api_key="test-key")
        
        with pytest.raises(APIError, match="Failed to get available models"):
            service.get_available_models()

    def test_update_model(self):
        """Test updating model for operation."""
        service = OpenAIService(api_key="test-key")
        
        service.update_model("transcription", "whisper-1")
        assert service.models["transcription"] == "whisper-1"

    def test_update_model_invalid_operation(self):
        """Test updating model for invalid operation."""
        service = OpenAIService(api_key="test-key")
        
        with pytest.raises(ValueError, match="Unknown operation"):
            service.update_model("invalid", "model")

    def test_get_model(self):
        """Test getting model for operation."""
        service = OpenAIService(api_key="test-key")
        
        assert service.get_model("transcription") == "gpt-4o-transcribe"
        assert service.get_model("summarization") == "gpt-3.5-turbo"
        assert service.get_model("unknown") == "gpt-3.5-turbo"  # Default

    def test_set_retry_config(self):
        """Test setting retry configuration."""
        service = OpenAIService(api_key="test-key")
        
        service.set_retry_config(attempts=5, delay=3)
        assert service.retry_attempts == 5
        assert service.retry_delay == 3

    def test_default_constants(self):
        """Test default constants are set correctly."""
        assert OpenAIService.DEFAULT_MODELS["transcription"] == "gpt-4o-transcribe"
        assert OpenAIService.DEFAULT_MODELS["summarization"] == "gpt-3.5-turbo"
        assert OpenAIService.DEFAULT_MODELS["documentation"] == "gpt-3.5-turbo"
        assert OpenAIService.DEFAULT_MAX_TOKENS["summarization"] == 500
        assert OpenAIService.DEFAULT_MAX_TOKENS["documentation"] == 700
        assert OpenAIService.DEFAULT_RETRY_ATTEMPTS == 3
        assert OpenAIService.DEFAULT_RETRY_DELAY == 2 