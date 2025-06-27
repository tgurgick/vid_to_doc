"""Audio extraction functionality for the vid_to_doc package."""

import os
from pathlib import Path
from typing import Optional, Union

from moviepy import VideoFileClip

from ..models.exceptions import AudioExtractionError


class AudioExtractor:
    """Handles audio extraction from video files."""

    SUPPORTED_FORMATS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"]

    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported video formats."""
        return cls.SUPPORTED_FORMATS.copy()

    @classmethod
    def extract_audio(
        cls, video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """Extract audio from a video file and save as a .wav file.
        
        Args:
            video_path: Path to the video file
            output_path: Path for the output audio file (optional)
            
        Returns:
            Path to the extracted audio file
            
        Raises:
            AudioExtractionError: If audio extraction fails
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        if output_path is None:
            output_path = video_path.with_suffix(".wav")
        else:
            output_path = Path(output_path)
        
        try:
            with VideoFileClip(str(video_path)) as video:
                audio = video.audio
                if audio is None:
                    raise AudioExtractionError(f"No audio track found in {video_path}")
                
                audio.write_audiofile(str(output_path), codec="pcm_s16le")
            
            return output_path
            
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio from {video_path}: {e}")

    @classmethod
    def extract_sample(
        cls,
        video_path: Union[str, Path],
        duration: int,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Extract a sample audio clip from the middle of the video.
        
        Args:
            video_path: Path to the video file
            duration: Duration of the sample in seconds
            output_path: Path for the output audio file (optional)
            
        Returns:
            Path to the extracted sample audio file
            
        Raises:
            AudioExtractionError: If sample extraction fails
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        if output_path is None:
            output_path = video_path.with_name(
                f"{video_path.stem}_sample_{duration}s.wav"
            )
        else:
            output_path = Path(output_path)
        
        try:
            with VideoFileClip(str(video_path)) as video:
                if video.duration < duration:
                    raise AudioExtractionError(
                        f"Video too short for sampling: {video_path} "
                        f"(duration: {video.duration:.2f}s, requested: {duration}s)"
                    )
                
                # Calculate start time for middle sample
                start_time = (video.duration - duration) / 2
                end_time = start_time + duration
                
                # Extract sample clip
                try:
                    # Try subclipped method (MoviePy v2+)
                    sample_clip = video.subclipped(start_time, end_time)
                except AttributeError:
                    # Fallback to subclip method (MoviePy v1)
                    sample_clip = video.subclip(start_time, end_time)
                
                audio = sample_clip.audio
                if audio is None:
                    raise AudioExtractionError(f"No audio track found in {video_path}")
                
                audio.write_audiofile(str(output_path), codec="pcm_s16le")
            
            return output_path
            
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract sample from {video_path}: {e}") 