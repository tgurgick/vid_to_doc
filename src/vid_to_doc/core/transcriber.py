"""Transcription logic for the vid_to_doc package."""

import os
import time
import math
import random
from pathlib import Path
from typing import Optional, Dict, Any

import openai
import soundfile as sf
from openai import APIConnectionError, APITimeoutError, RateLimitError

from ..models.exceptions import TranscriptionError


class Transcriber:
    """Handles transcription of audio files using OpenAI Whisper API."""

    MAX_MB = 25
    MAX_BYTES = MAX_MB * 1024 * 1024
    DEFAULT_MODEL = "gpt-4o-transcribe"
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_RETRY_DELAY = 2

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 retry_attempts: Optional[int] = None, retry_delay: Optional[int] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.retry_attempts = retry_attempts or self.DEFAULT_RETRY_ATTEMPTS
        self.retry_delay = retry_delay or self.DEFAULT_RETRY_DELAY
        
        if not self.api_key:
            raise TranscriptionError("OpenAI API key is required for transcription.")
        openai.api_key = self.api_key

    def _transcribe_with_retry(self, audio_file, model: str) -> str:
        """Transcribe audio with retry logic for connection errors.
        
        Args:
            audio_file: Audio file object
            model: Model to use for transcription
            
        Returns:
            Transcript as string
            
        Raises:
            TranscriptionError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = openai.audio.transcriptions.create(
                    file=audio_file,
                    model=model,
                    response_format="text",
                )
                return str(response)
                
            except (APIConnectionError, APITimeoutError) as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise TranscriptionError(f"Connection error after {self.retry_attempts} attempts: {e}")
                    
            except RateLimitError as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    # Longer delay for rate limits
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    time.sleep(delay)
                    continue
                else:
                    raise TranscriptionError(f"Rate limit exceeded after {self.retry_attempts} attempts: {e}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    delay = self.retry_delay + random.uniform(0, 1)
                    time.sleep(delay)
                    continue
                else:
                    raise TranscriptionError(f"Transcription failed after {self.retry_attempts} attempts: {e}")
        
        # This should never be reached, but just in case
        raise TranscriptionError(f"Transcription failed after {self.retry_attempts} attempts: {last_exception}")

    def transcribe(self, audio_path: Path, model: Optional[str] = None) -> str:
        """Transcribe audio using OpenAI's Whisper API, chunking if >25MB.

        Args:
            audio_path: Path to the audio file
            model: Model name to use (optional)
        Returns:
            Transcript as a string
        Raises:
            TranscriptionError: If transcription fails
        """
        audio_path = Path(audio_path)
        model = model or self.model
        file_size = audio_path.stat().st_size
        transcript = None
        start_time = time.time()
        
        if file_size <= self.MAX_BYTES:
            try:
                with open(audio_path, "rb") as audio_file:
                    transcript = self._transcribe_with_retry(audio_file, model)
            except Exception as e:
                raise TranscriptionError(f"Transcription failed: {e}")
        else:
            # Chunking for large files
            try:
                data, samplerate = sf.read(str(audio_path))
                total_samples = len(data)
                bytes_per_sample = data.itemsize if hasattr(data, "itemsize") else 2
                max_samples_per_chunk = self.MAX_BYTES // bytes_per_sample
                chunk_count = math.ceil(total_samples / max_samples_per_chunk)
                transcripts = []
                
                for i in range(chunk_count):
                    start = i * max_samples_per_chunk
                    end = min((i + 1) * max_samples_per_chunk, total_samples)
                    chunk_data = data[start:end]
                    chunk_path = audio_path.with_name(f"{audio_path.stem}_chunk{i+1}.wav")
                    sf.write(str(chunk_path), chunk_data, samplerate)
                    
                    try:
                        with open(chunk_path, "rb") as chunk_file:
                            chunk_transcript = self._transcribe_with_retry(chunk_file, model)
                        transcripts.append(chunk_transcript)
                    finally:
                        # Clean up chunk file
                        try:
                            os.remove(chunk_path)
                        except Exception:
                            pass
                
                transcript = "\n".join(transcripts)
            except Exception as e:
                raise TranscriptionError(f"Chunked transcription failed: {e}")
        
        elapsed = time.time() - start_time
        return transcript 