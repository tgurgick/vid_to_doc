"""Summarization logic for the vid_to_doc package."""

import os
from typing import Optional
import openai

from ..models.exceptions import SummarizationError


class Summarizer:
    """Handles text summarization using OpenAI GPT models."""

    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_MAX_TOKENS = 500
    DEFAULT_SYSTEM_PROMPT = "Summarize the following transcription into concise knowledge notes."

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            raise SummarizationError("OpenAI API key is required for summarization.")
        openai.api_key = self.api_key

    def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Summarize the given text using OpenAI's GPT API.

        Args:
            text: The text to summarize
            model: Model name to use (optional)
            max_tokens: Maximum tokens for the summary
            system_prompt: Custom system prompt (optional)
        Returns:
            The summary as a string
        Raises:
            SummarizationError: If summarization fails
        """
        model = model or self.model
        system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content is None:
                return ""
            summary = content.strip()
            return summary
        except Exception as e:
            raise SummarizationError(f"Failed to summarize text: {e}") 