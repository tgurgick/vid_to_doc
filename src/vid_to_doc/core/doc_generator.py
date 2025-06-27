"""Documentation generation logic for the vid_to_doc package."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import openai

from ..models.exceptions import DocumentationError


class DocGenerator:
    """Handles generation of various types of documentation from transcriptions and summaries."""

    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_MAX_TOKENS = 700

    # Predefined templates for different document types
    TEMPLATES = {
        "engineering": {
            "system_prompt": "You are an expert technical writer. Given the following summary of a video transcription, generate a concise engineering documentation section for future engineering teams. Focus on technical insights, process, and key takeaways. Use Markdown formatting.",
            "max_tokens": 700
        },
        "meeting_notes": {
            "system_prompt": "You are a professional meeting note taker. Convert the following transcription into well-structured meeting notes with clear sections for agenda, discussion points, decisions made, and action items. Use Markdown formatting.",
            "max_tokens": 800
        },
        "action_items": {
            "system_prompt": "Extract and organize action items from the following transcription. For each action item, identify the responsible party (if mentioned), the task description, and any deadlines or context. Format as a clear list with Markdown.",
            "max_tokens": 500
        },
        "summary": {
            "system_prompt": "Create a comprehensive summary of the following transcription. Include key points, main topics discussed, important decisions, and any notable insights. Use clear, professional language.",
            "max_tokens": 600
        }
    }

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the DocGenerator.

        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            model: Model name to use (optional, defaults to gpt-3.5-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            raise DocumentationError("OpenAI API key is required for documentation generation.")
        openai.api_key = self.api_key

    def generate_engineering_doc(
        self, 
        summary_text: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate engineering documentation from a summary.

        Args:
            summary_text: The summary text to base the documentation on
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated engineering documentation as a string

        Raises:
            DocumentationError: If documentation generation fails
        """
        template = self.TEMPLATES["engineering"]
        return self._generate_document(
            summary_text, 
            template["system_prompt"],
            model=model,
            max_tokens=max_tokens or template["max_tokens"]
        )

    def generate_meeting_notes(
        self, 
        transcript_text: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate meeting notes from a transcript.

        Args:
            transcript_text: The transcript text to convert to meeting notes
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated meeting notes as a string

        Raises:
            DocumentationError: If documentation generation fails
        """
        template = self.TEMPLATES["meeting_notes"]
        return self._generate_document(
            transcript_text, 
            template["system_prompt"],
            model=model,
            max_tokens=max_tokens or template["max_tokens"]
        )

    def generate_action_items(
        self, 
        transcript_text: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Extract action items from a transcript.

        Args:
            transcript_text: The transcript text to extract action items from
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Extracted action items as a formatted string

        Raises:
            DocumentationError: If documentation generation fails
        """
        template = self.TEMPLATES["action_items"]
        return self._generate_document(
            transcript_text, 
            template["system_prompt"],
            model=model,
            max_tokens=max_tokens or template["max_tokens"]
        )

    def generate_summary(
        self, 
        transcript_text: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a comprehensive summary from a transcript.

        Args:
            transcript_text: The transcript text to summarize
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated summary as a string

        Raises:
            DocumentationError: If documentation generation fails
        """
        template = self.TEMPLATES["summary"]
        return self._generate_document(
            transcript_text, 
            template["system_prompt"],
            model=model,
            max_tokens=max_tokens or template["max_tokens"]
        )

    def generate_custom_doc(
        self, 
        content: str, 
        system_prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate documentation using a custom system prompt.

        Args:
            content: The content to process
            system_prompt: Custom system prompt for the generation
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated documentation as a string

        Raises:
            DocumentationError: If documentation generation fails
        """
        return self._generate_document(
            content, 
            system_prompt,
            model=model,
            max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS
        )

    def _generate_document(
        self, 
        content: str, 
        system_prompt: str, 
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Internal method to generate documentation using OpenAI's API.

        Args:
            content: The content to process
            system_prompt: System prompt for the generation
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated documentation as a string

        Raises:
            DocumentationError: If documentation generation fails
        """
        model = model or self.model
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=max_tokens,
            )
            result_content = response.choices[0].message.content
            if result_content is None:
                return ""
            return result_content.strip()
        except Exception as e:
            raise DocumentationError(f"Failed to generate documentation: {e}")

    def save_document(
        self, 
        document_content: str, 
        base_path: Path, 
        doc_type: str = "engineering_doc",
        output_dir: Optional[Path] = None
    ) -> Path:
        """Save generated documentation to a file.

        Args:
            document_content: The document content to save
            base_path: Base path for the original file (used for naming)
            doc_type: Type of document (used for file extension)
            output_dir: Output directory (optional, defaults to 'output' folder)

        Returns:
            Path to the saved file

        Raises:
            DocumentationError: If saving fails
        """
        try:
            # Determine output directory
            if output_dir is None:
                output_dir = base_path.parent / "output"
            else:
                output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            base_name = base_path.stem
            if doc_type == "engineering_doc":
                filename = f"{base_name}_engineering_doc.md"
            elif doc_type == "meeting_notes":
                filename = f"{base_name}_meeting_notes.md"
            elif doc_type == "action_items":
                filename = f"{base_name}_action_items.md"
            elif doc_type == "summary":
                filename = f"{base_name}_summary.txt"
            else:
                filename = f"{base_name}_{doc_type}.md"

            output_path = output_dir / filename

            # Save the document
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(document_content)

            return output_path
        except Exception as e:
            raise DocumentationError(f"Failed to save document: {e}")

    def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available document templates.

        Returns:
            Dictionary of available templates with their configurations
        """
        return self.TEMPLATES.copy()

    def add_template(
        self, 
        name: str, 
        system_prompt: str, 
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        """Add a custom template.

        Args:
            name: Template name
            system_prompt: System prompt for the template
            max_tokens: Maximum tokens for the template
        """
        self.TEMPLATES[name] = {
            "system_prompt": system_prompt,
            "max_tokens": max_tokens
        } 