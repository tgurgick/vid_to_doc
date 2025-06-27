"""OpenAI API service wrapper for the vid_to_doc package."""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import openai

from ..models.exceptions import APIError, TranscriptionError, SummarizationError, DocumentationError


class OpenAIService:
    """Service wrapper for OpenAI API interactions."""

    DEFAULT_MODELS = {
        "transcription": "gpt-4o-transcribe",
        "summarization": "gpt-3.5-turbo",
        "documentation": "gpt-3.5-turbo",
    }

    DEFAULT_MAX_TOKENS = {
        "summarization": 500,
        "documentation": 700,
    }

    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 2

    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_models: Optional[Dict[str, str]] = None,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_delay: int = DEFAULT_RETRY_DELAY
    ):
        """Initialize the OpenAI service.

        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            default_models: Dictionary of default models for different operations
            retry_attempts: Number of retry attempts for failed API calls
            retry_delay: Base delay for retry attempts (will be exponential)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIError("OpenAI API key is required")
        
        openai.api_key = self.api_key
        
        self.models = default_models or self.DEFAULT_MODELS.copy()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def transcribe_audio(
        self, 
        audio_file: Path, 
        model: Optional[str] = None,
        response_format: str = "text"
    ) -> str:
        """Transcribe audio using OpenAI's Whisper API.

        Args:
            audio_file: Path to the audio file
            model: Model to use (optional, uses default if not provided)
            response_format: Response format ('text' or 'verbose')

        Returns:
            Transcribed text

        Raises:
            TranscriptionError: If transcription fails
        """
        model = model or self.models["transcription"]
        audio_file = Path(audio_file)

        if not audio_file.exists():
            raise TranscriptionError(f"Audio file does not exist: {audio_file}")

        for attempt in range(1, self.retry_attempts + 1):
            try:
                with open(audio_file, "rb") as file:
                    response = openai.audio.transcriptions.create(
                        file=file,
                        model=model,
                        response_format=response_format,
                    )
                
                if response_format == "text":
                    return str(response)
                else:
                    return response.text

            except Exception as e:
                if attempt < self.retry_attempts:
                    wait_time = self.retry_delay ** attempt
                    time.sleep(wait_time)
                else:
                    raise TranscriptionError(f"Transcription failed after {self.retry_attempts} attempts: {e}")

    def generate_summary(
        self, 
        text: str, 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a summary using OpenAI's GPT API.

        Args:
            text: Text to summarize
            system_prompt: Custom system prompt (optional)
            model: Model to use (optional, uses default if not provided)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated summary

        Raises:
            SummarizationError: If summarization fails
        """
        model = model or self.models["summarization"]
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS["summarization"]
        
        if not system_prompt:
            system_prompt = "Summarize the following text into concise knowledge notes."

        return self._generate_chat_completion(
            text, 
            system_prompt, 
            model, 
            max_tokens,
            error_type=SummarizationError,
            operation="summarization"
        )

    def generate_documentation(
        self, 
        content: str, 
        system_prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate documentation using OpenAI's GPT API.

        Args:
            content: Content to process
            system_prompt: System prompt for the generation
            model: Model to use (optional, uses default if not provided)
            max_tokens: Maximum tokens for the response (optional)

        Returns:
            Generated documentation

        Raises:
            DocumentationError: If documentation generation fails
        """
        model = model or self.models["documentation"]
        max_tokens = max_tokens or self.DEFAULT_MAX_TOKENS["documentation"]

        return self._generate_chat_completion(
            content, 
            system_prompt, 
            model, 
            max_tokens,
            error_type=DocumentationError,
            operation="documentation generation"
        )

    def _generate_chat_completion(
        self, 
        content: str, 
        system_prompt: str,
        model: str,
        max_tokens: int,
        error_type: type,
        operation: str
    ) -> str:
        """Internal method to generate chat completions with retry logic.

        Args:
            content: Content to process
            system_prompt: System prompt
            model: Model to use
            max_tokens: Maximum tokens
            error_type: Type of error to raise
            operation: Operation name for error messages

        Returns:
            Generated response

        Raises:
            error_type: If generation fails
        """
        for attempt in range(1, self.retry_attempts + 1):
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
                if attempt < self.retry_attempts:
                    wait_time = self.retry_delay ** attempt
                    time.sleep(wait_time)
                else:
                    raise error_type(f"{operation.capitalize()} failed after {self.retry_attempts} attempts: {e}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics.

        Returns:
            Dictionary with usage statistics

        Raises:
            APIError: If unable to retrieve usage stats
        """
        try:
            # Note: This requires the OpenAI API to support usage endpoints
            # The actual implementation may vary based on OpenAI's API
            return {
                "note": "Usage statistics endpoint not implemented in this version",
                "api_key": self.api_key[:8] + "..." if self.api_key else None
            }
        except Exception as e:
            raise APIError(f"Failed to get usage stats: {e}")

    def validate_api_key(self) -> bool:
        """Validate that the API key is working.

        Returns:
            True if the API key is valid

        Raises:
            APIError: If API key is invalid
        """
        try:
            # Make a simple API call to test the key
            response = openai.models.list()
            return True
        except Exception as e:
            raise APIError(f"Invalid API key: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available models.

        Returns:
            List of available model names

        Raises:
            APIError: If unable to retrieve models
        """
        try:
            response = openai.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            raise APIError(f"Failed to get available models: {e}")

    def update_model(self, operation: str, model: str) -> None:
        """Update the default model for an operation.

        Args:
            operation: Operation type ('transcription', 'summarization', 'documentation')
            model: Model name to use
        """
        if operation not in self.models:
            raise ValueError(f"Unknown operation: {operation}")
        
        self.models[operation] = model

    def get_model(self, operation: str) -> str:
        """Get the default model for an operation.

        Args:
            operation: Operation type

        Returns:
            Model name for the operation
        """
        return self.models.get(operation, "gpt-3.5-turbo")

    def set_retry_config(self, attempts: int, delay: int) -> None:
        """Update retry configuration.

        Args:
            attempts: Number of retry attempts
            delay: Base delay for retries
        """
        self.retry_attempts = attempts
        self.retry_delay = delay 