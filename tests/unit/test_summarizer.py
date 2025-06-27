"""Unit tests for the Summarizer class."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.vid_to_doc.core.summarizer import Summarizer
from src.vid_to_doc.models.exceptions import SummarizationError


class TestSummarizer:
    """Test cases for Summarizer class."""

    def test_default_constants(self):
        """Test that default constants are set correctly."""
        assert Summarizer.DEFAULT_MODEL == "gpt-4o"
        assert Summarizer.DEFAULT_MAX_TOKENS == 500
        assert Summarizer.DEFAULT_SYSTEM_PROMPT == "Summarize the following transcription into concise knowledge notes."

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        summarizer = Summarizer()
        assert summarizer.api_key == "test-api-key"
        assert summarizer.model == Summarizer.DEFAULT_MODEL

    def test_init_with_provided_api_key(self):
        """Test initialization with provided API key."""
        summarizer = Summarizer(api_key="provided-key")
        assert summarizer.api_key == "provided-key"
        assert summarizer.model == Summarizer.DEFAULT_MODEL

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        summarizer = Summarizer(api_key="test-key", model="gpt-4")
        assert summarizer.api_key == "test-key"
        assert summarizer.model == "gpt-4"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(SummarizationError, match="OpenAI API key is required for summarization"):
            Summarizer()

    @patch("openai.chat.completions.create")
    def test_summarize_success(self, mock_create):
        """Test successful summarization."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This is a summary of the text."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("This is a long text that needs to be summarized.")

        # Assertions
        assert result == "This is a summary of the text."
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["max_tokens"] == 500
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][0]["content"] == Summarizer.DEFAULT_SYSTEM_PROMPT
        assert call_args[1]["messages"][1]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == "This is a long text that needs to be summarized."

    @patch("openai.chat.completions.create")
    def test_summarize_with_custom_model(self, mock_create):
        """Test summarization with custom model."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Custom model summary."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("Test text", model="gpt-4")

        # Assertions
        assert result == "Custom model summary."
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "gpt-4"

    @patch("openai.chat.completions.create")
    def test_summarize_with_custom_max_tokens(self, mock_create):
        """Test summarization with custom max tokens."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Custom tokens summary."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("Test text", max_tokens=1000)

        # Assertions
        assert result == "Custom tokens summary."
        call_args = mock_create.call_args
        assert call_args[1]["max_tokens"] == 1000

    @patch("openai.chat.completions.create")
    def test_summarize_with_custom_system_prompt(self, mock_create):
        """Test summarization with custom system prompt."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Custom prompt summary."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        custom_prompt = "Create a technical summary of the following text."
        result = summarizer.summarize("Test text", system_prompt=custom_prompt)

        # Assertions
        assert result == "Custom prompt summary."
        call_args = mock_create.call_args
        assert call_args[1]["messages"][0]["content"] == custom_prompt

    @patch("openai.chat.completions.create")
    def test_summarize_with_empty_text(self, mock_create):
        """Test summarization with empty text."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Empty text summary."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("")

        # Assertions
        assert result == "Empty text summary."
        call_args = mock_create.call_args
        assert call_args[1]["messages"][1]["content"] == ""

    @patch("openai.chat.completions.create")
    def test_summarize_with_whitespace_text(self, mock_create):
        """Test summarization with whitespace-only text."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Whitespace summary."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("   \n\t   ")

        # Assertions
        assert result == "Whitespace summary."
        call_args = mock_create.call_args
        assert call_args[1]["messages"][1]["content"] == "   \n\t   "

    @patch("openai.chat.completions.create")
    def test_summarize_strips_response_content(self, mock_create):
        """Test that response content is properly stripped."""
        # Setup mock response with extra whitespace
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "  \n  This is a summary with extra whitespace.  \n  "
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("Test text")

        # Assertions
        assert result == "This is a summary with extra whitespace."

    @patch("openai.chat.completions.create")
    def test_summarize_openai_api_error(self, mock_create):
        """Test that OpenAI API errors are properly handled."""
        # Setup mock to raise exception
        mock_create.side_effect = Exception("API rate limit exceeded")

        # Test
        summarizer = Summarizer(api_key="test-key")
        with pytest.raises(SummarizationError, match="Failed to summarize text: API rate limit exceeded"):
            summarizer.summarize("Test text")

    @patch("openai.chat.completions.create")
    def test_summarize_openai_authentication_error(self, mock_create):
        """Test that OpenAI authentication errors are properly handled."""
        # Setup mock to raise authentication error
        mock_create.side_effect = Exception("Invalid API key")

        # Test
        summarizer = Summarizer(api_key="invalid-key")
        with pytest.raises(SummarizationError, match="Failed to summarize text: Invalid API key"):
            summarizer.summarize("Test text")

    @patch("openai.chat.completions.create")
    def test_summarize_openai_network_error(self, mock_create):
        """Test that OpenAI network errors are properly handled."""
        # Setup mock to raise network error
        mock_create.side_effect = Exception("Connection timeout")

        # Test
        summarizer = Summarizer(api_key="test-key")
        with pytest.raises(SummarizationError, match="Failed to summarize text: Connection timeout"):
            summarizer.summarize("Test text")

    @patch("openai.chat.completions.create")
    def test_summarize_empty_response_choices(self, mock_create):
        """Test handling of empty response choices."""
        # Setup mock response with empty choices
        mock_response = Mock()
        mock_response.choices = []
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        with pytest.raises(SummarizationError, match="Failed to summarize text"):
            summarizer.summarize("Test text")

    @patch("openai.chat.completions.create")
    def test_summarize_none_response_content(self, mock_create):
        """Test handling of None response content."""
        # Setup mock response with None content
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        summarizer = Summarizer(api_key="test-key")
        result = summarizer.summarize("Test text")

        # Assertions
        assert result == ""

    def test_summarize_uses_instance_model_by_default(self):
        """Test that summarization uses the instance model by default."""
        with patch("openai.chat.completions.create") as mock_create:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test summary."
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response

            # Test with custom instance model
            summarizer = Summarizer(api_key="test-key", model="gpt-4")
            summarizer.summarize("Test text")

            # Assertions
            call_args = mock_create.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_summarize_uses_instance_model_when_no_model_provided(self):
        """Test that summarization uses instance model when no model parameter provided."""
        with patch("openai.chat.completions.create") as mock_create:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test summary."
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response

            # Test
            summarizer = Summarizer(api_key="test-key", model="gpt-4")
            summarizer.summarize("Test text")

            # Assertions
            call_args = mock_create.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_summarize_uses_provided_model_over_instance_model(self):
        """Test that provided model parameter overrides instance model."""
        with patch("openai.chat.completions.create") as mock_create:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test summary."
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response

            # Test
            summarizer = Summarizer(api_key="test-key", model="gpt-3.5-turbo")
            summarizer.summarize("Test text", model="gpt-4")

            # Assertions
            call_args = mock_create.call_args
            assert call_args[1]["model"] == "gpt-4"

    def test_summarize_uses_default_system_prompt_when_none_provided(self):
        """Test that default system prompt is used when none provided."""
        with patch("openai.chat.completions.create") as mock_create:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test summary."
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response

            # Test
            summarizer = Summarizer(api_key="test-key")
            summarizer.summarize("Test text")

            # Assertions
            call_args = mock_create.call_args
            assert call_args[1]["messages"][0]["content"] == Summarizer.DEFAULT_SYSTEM_PROMPT

    def test_summarize_uses_provided_system_prompt(self):
        """Test that provided system prompt is used."""
        with patch("openai.chat.completions.create") as mock_create:
            # Setup mock response
            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test summary."
            mock_response.choices = [mock_choice]
            mock_create.return_value = mock_response

            # Test
            summarizer = Summarizer(api_key="test-key")
            custom_prompt = "Custom system prompt for testing."
            summarizer.summarize("Test text", system_prompt=custom_prompt)

            # Assertions
            call_args = mock_create.call_args
            assert call_args[1]["messages"][0]["content"] == custom_prompt 