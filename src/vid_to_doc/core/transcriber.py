"""Transcription logic for the vid_to_doc package."""

import os
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any

import openai
import soundfile as sf

from ..models.exceptions import TranscriptionError


class Transcriber:
    """Handles transcription of audio files using OpenAI Whisper API."""

    MAX_MB = 25
    MAX_BYTES = MAX_MB * 1024 * 1024
    DEFAULT_MODEL = "gpt-4o-transcribe"
    RETRY_ATTEMPTS = 3

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        if not self.api_key:
            raise TranscriptionError("OpenAI API key is required for transcription.")
        openai.api_key = self.api_key

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
                    response = openai.audio.transcriptions.create(
                        file=audio_file,
                        model=model,
                        response_format="text",
                    )
                transcript = str(response)
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
                    chunk_transcript = None
                    for attempt in range(1, self.RETRY_ATTEMPTS + 1):
                        try:
                            with open(chunk_path, "rb") as chunk_file:
                                response = openai.audio.transcriptions.create(
                                    file=chunk_file,
                                    model=model,
                                    response_format="text",
                                )
                            chunk_transcript = str(response)
                            break
                        except Exception as e:
                            if attempt < self.RETRY_ATTEMPTS:
                                time.sleep(2 ** attempt)
                            else:
                                raise TranscriptionError(f"Chunk {i+1} failed after retries: {e}")
                    transcripts.append(chunk_transcript)
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
                transcript = "\n".join(transcripts)
            except Exception as e:
                raise TranscriptionError(f"Chunked transcription failed: {e}")
        elapsed = time.time() - start_time
        # Optionally, you could return stats as well
        return transcript 