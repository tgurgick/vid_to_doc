"""Audio processing utilities for the vid_to_doc package."""

import wave
import math
from pathlib import Path
from typing import Tuple, Optional, List
import soundfile as sf

from ..models.exceptions import AudioError


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds, or None if duration cannot be determined
        
    Raises:
        AudioError: If file cannot be read
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    try:
        # Try using wave module first (for WAV files)
        with wave.open(str(audio_path), 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        try:
            # Fallback to soundfile for other formats
            data, samplerate = sf.read(str(audio_path))
            return len(data) / samplerate
        except Exception as e:
            raise AudioError(f"Failed to get audio duration for {audio_path}: {e}")


def get_audio_info(audio_path: Path) -> dict:
    """Get comprehensive information about an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing audio information (duration, sample_rate, channels, etc.)
        
    Raises:
        AudioError: If file cannot be read
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    try:
        data, samplerate = sf.read(str(audio_path))
        
        # Determine number of channels
        if len(data.shape) == 1:
            channels = 1
        else:
            channels = data.shape[1]
        
        # Calculate duration
        duration = len(data) / samplerate
        
        # Get file size
        file_size = audio_path.stat().st_size
        
        return {
            "duration": duration,
            "sample_rate": samplerate,
            "channels": channels,
            "samples": len(data),
            "file_size_bytes": file_size,
            "file_size_mb": file_size / (1024 * 1024),
            "format": audio_path.suffix.lower()
        }
    except Exception as e:
        raise AudioError(f"Failed to get audio info for {audio_path}: {e}")


def split_audio_file(
    audio_path: Path, 
    chunk_duration: float, 
    output_dir: Optional[Path] = None
) -> List[Path]:
    """Split an audio file into chunks of specified duration.
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds
        output_dir: Output directory for chunks (optional)
        
    Returns:
        List of paths to the chunk files
        
    Raises:
        AudioError: If splitting fails
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    if chunk_duration <= 0:
        raise AudioError("Chunk duration must be positive")
    
    try:
        # Read audio data
        data, samplerate = sf.read(str(audio_path))
        
        # Calculate chunk parameters
        samples_per_chunk = int(chunk_duration * samplerate)
        total_samples = len(data)
        num_chunks = math.ceil(total_samples / samples_per_chunk)
        
        # Determine output directory
        if output_dir is None:
            output_dir = audio_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_paths = []
        
        for i in range(num_chunks):
            start_sample = i * samples_per_chunk
            end_sample = min((i + 1) * samples_per_chunk, total_samples)
            
            # Extract chunk data
            chunk_data = data[start_sample:end_sample]
            
            # Generate chunk filename
            chunk_filename = f"{audio_path.stem}_chunk{i+1:03d}{audio_path.suffix}"
            chunk_path = output_dir / chunk_filename
            
            # Save chunk
            sf.write(str(chunk_path), chunk_data, samplerate)
            chunk_paths.append(chunk_path)
        
        return chunk_paths
        
    except Exception as e:
        raise AudioError(f"Failed to split audio file {audio_path}: {e}")


def validate_audio_file(audio_path: Path) -> bool:
    """Validate that an audio file can be read and processed.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        True if the file is valid
        
    Raises:
        AudioError: If file is invalid
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    if audio_path.stat().st_size == 0:
        raise AudioError(f"Audio file is empty: {audio_path}")
    
    try:
        # Try to read the file
        data, samplerate = sf.read(str(audio_path))
        
        if len(data) == 0:
            raise AudioError(f"Audio file contains no data: {audio_path}")
        
        if samplerate <= 0:
            raise AudioError(f"Invalid sample rate: {samplerate}")
        
        return True
        
    except Exception as e:
        raise AudioError(f"Invalid audio file {audio_path}: {e}")


def convert_audio_format(
    input_path: Path, 
    output_path: Path, 
    target_format: str = "wav",
    sample_rate: Optional[int] = None
) -> Path:
    """Convert an audio file to a different format.
    
    Args:
        input_path: Path to the input audio file
        output_path: Path for the output file
        target_format: Target format (e.g., 'wav', 'mp3', 'flac')
        sample_rate: Target sample rate (optional, keeps original if not specified)
        
    Returns:
        Path to the converted file
        
    Raises:
        AudioError: If conversion fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise AudioError(f"Input audio file does not exist: {input_path}")
    
    try:
        # Read input file
        data, original_sample_rate = sf.read(str(input_path))
        
        # Use specified sample rate or keep original
        target_sample_rate = sample_rate or original_sample_rate
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output file
        sf.write(str(output_path), data, target_sample_rate, format=target_format.upper())
        
        return output_path
        
    except Exception as e:
        raise AudioError(f"Failed to convert audio file {input_path}: {e}")


def get_audio_chunk_size_for_bytes(
    audio_path: Path, 
    max_bytes: int
) -> Tuple[int, int]:
    """Calculate the number of samples per chunk to stay under a byte limit.
    
    Args:
        audio_path: Path to the audio file
        max_bytes: Maximum bytes per chunk
        
    Returns:
        Tuple of (samples_per_chunk, total_chunks)
        
    Raises:
        AudioError: If calculation fails
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    try:
        data, samplerate = sf.read(str(audio_path))
        
        # Calculate bytes per sample
        bytes_per_sample = data.itemsize if hasattr(data, 'itemsize') else 2
        
        # Calculate samples per chunk
        samples_per_chunk = max_bytes // bytes_per_sample
        
        # Calculate total chunks
        total_samples = len(data)
        total_chunks = math.ceil(total_samples / samples_per_chunk)
        
        return samples_per_chunk, total_chunks
        
    except Exception as e:
        raise AudioError(f"Failed to calculate chunk size for {audio_path}: {e}")


def normalize_audio_levels(audio_path: Path, output_path: Optional[Path] = None) -> Path:
    """Normalize audio levels to prevent clipping and improve quality.
    
    Args:
        audio_path: Path to the input audio file
        output_path: Path for the normalized file (optional)
        
    Returns:
        Path to the normalized file
        
    Raises:
        AudioError: If normalization fails
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise AudioError(f"Audio file does not exist: {audio_path}")
    
    if output_path is None:
        output_path = audio_path.parent / f"{audio_path.stem}_normalized{audio_path.suffix}"
    else:
        output_path = Path(output_path)
    
    try:
        # Read audio data
        data, samplerate = sf.read(str(audio_path))
        
        # Find the maximum absolute value
        max_val = abs(data).max()
        
        if max_val > 0:
            # Normalize to prevent clipping (scale to 0.95 of max)
            normalized_data = data * (0.95 / max_val)
        else:
            # If all values are zero, keep as is
            normalized_data = data
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save normalized audio
        sf.write(str(output_path), normalized_data, samplerate)
        
        return output_path
        
    except Exception as e:
        raise AudioError(f"Failed to normalize audio file {audio_path}: {e}") 