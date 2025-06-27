"""Unit tests for the DocGenerator class."""

import pytest
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.vid_to_doc.core.doc_generator import DocGenerator
from src.vid_to_doc.models.exceptions import DocumentationError


class TestDocGenerator:
    """Test cases for DocGenerator class."""

    def test_default_constants(self):
        """Test that default constants are set correctly."""
        assert DocGenerator.DEFAULT_MODEL == "gpt-3.5-turbo"
        assert DocGenerator.DEFAULT_MAX_TOKENS == 700

    def test_templates_structure(self):
        """Test that templates are properly structured."""
        templates = DocGenerator.TEMPLATES
        expected_keys = {"engineering", "meeting_notes", "action_items", "summary"}
        assert set(templates.keys()) == expected_keys
        
        for template in templates.values():
            assert "system_prompt" in template
            assert "max_tokens" in template
            assert isinstance(template["system_prompt"], str)
            assert isinstance(template["max_tokens"], int)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        doc_generator = DocGenerator()
        assert doc_generator.api_key == "test-api-key"
        assert doc_generator.model == DocGenerator.DEFAULT_MODEL

    def test_init_with_provided_api_key(self):
        """Test initialization with provided API key."""
        doc_generator = DocGenerator(api_key="provided-key")
        assert doc_generator.api_key == "provided-key"
        assert doc_generator.model == DocGenerator.DEFAULT_MODEL

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        doc_generator = DocGenerator(api_key="test-key", model="gpt-4")
        assert doc_generator.api_key == "test-key"
        assert doc_generator.model == "gpt-4"

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(DocumentationError, match="OpenAI API key is required for documentation generation"):
            DocGenerator()

    @patch("openai.chat.completions.create")
    def test_generate_engineering_doc_success(self, mock_create):
        """Test successful engineering documentation generation."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "# Engineering Documentation\n\nThis is the generated documentation."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_engineering_doc("This is a summary of the video.")

        # Assertions
        assert result == "# Engineering Documentation\n\nThis is the generated documentation."
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["max_tokens"] == DocGenerator.TEMPLATES["engineering"]["max_tokens"]
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][0]["content"] == DocGenerator.TEMPLATES["engineering"]["system_prompt"]
        assert call_args[1]["messages"][1]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == "This is a summary of the video."

    @patch("openai.chat.completions.create")
    def test_generate_meeting_notes_success(self, mock_create):
        """Test successful meeting notes generation."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "# Meeting Notes\n\n## Agenda\n- Item 1\n- Item 2"
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_meeting_notes("This is the meeting transcript.")

        # Assertions
        assert result == "# Meeting Notes\n\n## Agenda\n- Item 1\n- Item 2"
        call_args = mock_create.call_args
        assert call_args[1]["max_tokens"] == DocGenerator.TEMPLATES["meeting_notes"]["max_tokens"]
        assert call_args[1]["messages"][0]["content"] == DocGenerator.TEMPLATES["meeting_notes"]["system_prompt"]

    @patch("openai.chat.completions.create")
    def test_generate_action_items_success(self, mock_create):
        """Test successful action items extraction."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "# Action Items\n\n1. [ ] Task 1 - John\n2. [ ] Task 2 - Jane"
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_action_items("This is the meeting transcript.")

        # Assertions
        assert result == "# Action Items\n\n1. [ ] Task 1 - John\n2. [ ] Task 2 - Jane"
        call_args = mock_create.call_args
        assert call_args[1]["max_tokens"] == DocGenerator.TEMPLATES["action_items"]["max_tokens"]
        assert call_args[1]["messages"][0]["content"] == DocGenerator.TEMPLATES["action_items"]["system_prompt"]

    @patch("openai.chat.completions.create")
    def test_generate_summary_success(self, mock_create):
        """Test successful summary generation."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "This is a comprehensive summary of the transcript."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_summary("This is the transcript.")

        # Assertions
        assert result == "This is a comprehensive summary of the transcript."
        call_args = mock_create.call_args
        assert call_args[1]["max_tokens"] == DocGenerator.TEMPLATES["summary"]["max_tokens"]
        assert call_args[1]["messages"][0]["content"] == DocGenerator.TEMPLATES["summary"]["system_prompt"]

    @patch("openai.chat.completions.create")
    def test_generate_custom_doc_success(self, mock_create):
        """Test successful custom documentation generation."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Custom documentation content."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        custom_prompt = "Generate a custom document from this content."
        result = doc_generator.generate_custom_doc("Content to process.", custom_prompt)

        # Assertions
        assert result == "Custom documentation content."
        call_args = mock_create.call_args
        assert call_args[1]["messages"][0]["content"] == custom_prompt
        assert call_args[1]["max_tokens"] == DocGenerator.DEFAULT_MAX_TOKENS

    @patch("openai.chat.completions.create")
    def test_generate_doc_with_custom_model(self, mock_create):
        """Test documentation generation with custom model."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Documentation with custom model."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_engineering_doc("Summary text.", model="gpt-4")

        # Assertions
        assert result == "Documentation with custom model."
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "gpt-4"

    @patch("openai.chat.completions.create")
    def test_generate_doc_with_custom_max_tokens(self, mock_create):
        """Test documentation generation with custom max tokens."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Documentation with custom tokens."
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_engineering_doc("Summary text.", max_tokens=1000)

        # Assertions
        assert result == "Documentation with custom tokens."
        call_args = mock_create.call_args
        assert call_args[1]["max_tokens"] == 1000

    @patch("openai.chat.completions.create")
    def test_generate_doc_api_error(self, mock_create):
        """Test that API errors are properly handled."""
        # Setup mock to raise exception
        mock_create.side_effect = Exception("API rate limit exceeded")

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        with pytest.raises(DocumentationError, match="Failed to generate documentation: API rate limit exceeded"):
            doc_generator.generate_engineering_doc("Summary text.")

    @patch("openai.chat.completions.create")
    def test_generate_doc_none_response_content(self, mock_create):
        """Test handling of None response content."""
        # Setup mock response with None content
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = None
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_engineering_doc("Summary text.")

        # Assertions
        assert result == ""

    @patch("openai.chat.completions.create")
    def test_generate_doc_strips_response_content(self, mock_create):
        """Test that response content is properly stripped."""
        # Setup mock response with extra whitespace
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "  \n  Documentation with extra whitespace.  \n  "
        mock_response.choices = [mock_choice]
        mock_create.return_value = mock_response

        # Test
        doc_generator = DocGenerator(api_key="test-key")
        result = doc_generator.generate_engineering_doc("Summary text.")

        # Assertions
        assert result == "Documentation with extra whitespace."

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_engineering_doc(self, mock_mkdir, mock_file):
        """Test saving engineering documentation."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        result = doc_generator.save_document(
            "# Engineering Doc\n\nContent here.", 
            base_path, 
            "engineering_doc"
        )

        # Assertions
        expected_path = base_path.parent / "output" / "test_video_engineering_doc.md"
        assert result == expected_path
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file.assert_called_once_with(expected_path, 'w', encoding='utf-8')

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_meeting_notes(self, mock_mkdir, mock_file):
        """Test saving meeting notes."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        result = doc_generator.save_document(
            "# Meeting Notes\n\nContent here.", 
            base_path, 
            "meeting_notes"
        )

        # Assertions
        expected_path = base_path.parent / "output" / "test_video_meeting_notes.md"
        assert result == expected_path

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_action_items(self, mock_mkdir, mock_file):
        """Test saving action items."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        result = doc_generator.save_document(
            "# Action Items\n\nContent here.", 
            base_path, 
            "action_items"
        )

        # Assertions
        expected_path = base_path.parent / "output" / "test_video_action_items.md"
        assert result == expected_path

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_summary(self, mock_mkdir, mock_file):
        """Test saving summary."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        result = doc_generator.save_document(
            "This is a summary.", 
            base_path, 
            "summary"
        )

        # Assertions
        expected_path = base_path.parent / "output" / "test_video_summary.txt"
        assert result == expected_path

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_custom_type(self, mock_mkdir, mock_file):
        """Test saving custom document type."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        result = doc_generator.save_document(
            "Custom content.", 
            base_path, 
            "custom_type"
        )

        # Assertions
        expected_path = base_path.parent / "output" / "test_video_custom_type.md"
        assert result == expected_path

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.mkdir")
    def test_save_document_custom_output_dir(self, mock_mkdir, mock_file):
        """Test saving document with custom output directory."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        custom_output = Path("/custom/output")
        
        result = doc_generator.save_document(
            "Content here.", 
            base_path, 
            "engineering_doc",
            custom_output
        )

        # Assertions
        expected_path = custom_output / "test_video_engineering_doc.md"
        assert result == expected_path
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("builtins.open", side_effect=Exception("Permission denied"))
    @patch("pathlib.Path.mkdir")
    def test_save_document_error(self, mock_mkdir, mock_file):
        """Test that save errors are properly handled."""
        doc_generator = DocGenerator(api_key="test-key")
        base_path = Path("test_video.mp4")
        
        with pytest.raises(DocumentationError, match="Failed to save document: Permission denied"):
            doc_generator.save_document("Content here.", base_path, "engineering_doc")

    def test_get_available_templates(self):
        """Test getting available templates."""
        doc_generator = DocGenerator(api_key="test-key")
        templates = doc_generator.get_available_templates()
        
        # Should return a copy of the templates
        assert templates == DocGenerator.TEMPLATES
        assert templates is not DocGenerator.TEMPLATES  # Should be a copy

    def test_add_template(self):
        """Test adding a custom template."""
        doc_generator = DocGenerator(api_key="test-key")
        custom_prompt = "Generate a custom document."
        
        doc_generator.add_template("custom", custom_prompt, 1000)
        
        # Check that template was added
        templates = doc_generator.get_available_templates()
        assert "custom" in templates
        assert templates["custom"]["system_prompt"] == custom_prompt
        assert templates["custom"]["max_tokens"] == 1000

    def test_add_template_default_max_tokens(self):
        """Test adding template with default max tokens."""
        doc_generator = DocGenerator(api_key="test-key")
        custom_prompt = "Generate a custom document."
        
        doc_generator.add_template("custom", custom_prompt)
        
        # Check that default max tokens was used
        templates = doc_generator.get_available_templates()
        assert templates["custom"]["max_tokens"] == DocGenerator.DEFAULT_MAX_TOKENS 