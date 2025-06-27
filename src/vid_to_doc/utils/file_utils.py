"""File operation utilities for the vid_to_doc package."""

import os
from pathlib import Path
from typing import List

from ..models.exceptions import FileError


def ensure_output_directory(base_path: Path) -> Path:
    """Ensure an output directory exists next to the given base path.
    
    Args:
        base_path: Base file path to create output directory for
        
    Returns:
        Path to the output directory
    """
    base_path = Path(base_path)
    output_dir = base_path.parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def get_video_files_from_folder(folder_path: Path) -> List[Path]:
    """Get a list of video file paths in the given folder.
    
    Args:
        folder_path: Path to folder containing video files
        
    Returns:
        List of video file paths
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileError(f"Folder does not exist: {folder_path}")
    
    if not folder_path.is_dir():
        raise FileError(f"Path is not a directory: {folder_path}")
    
    supported_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"}
    
    video_files = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            video_files.append(file_path)
    
    return sorted(video_files)


def save_text_to_file(text: str, file_path: Path, encoding: str = "utf-8") -> Path:
    """Save text content to a file.
    
    Args:
        text: Text content to save
        file_path: Path where to save the file
        encoding: File encoding (default: utf-8)
        
    Returns:
        Path to the saved file
        
    Raises:
        FileError: If file cannot be saved
    """
    file_path = Path(file_path)
    
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(file_path, "w", encoding=encoding) as f:
            f.write(text)
        
        return file_path
    except Exception as e:
        raise FileError(f"Failed to save file {file_path}: {e}")


def get_file_size(file_path: Path) -> int:
    """Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes
        
    Raises:
        FileError: If file does not exist or cannot be accessed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileError(f"File does not exist: {file_path}")
    
    try:
        return file_path.stat().st_size
    except Exception as e:
        raise FileError(f"Failed to get file size for {file_path}: {e}")


def is_audio_file(file_path: Path) -> bool:
    """Check if a file is an audio file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is an audio file
    """
    file_path = Path(file_path)
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mpga", ".webm"}
    return file_path.suffix.lower() in audio_extensions


def is_video_file(file_path: Path) -> bool:
    """Check if a file is a video file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a video file
    """
    file_path = Path(file_path)
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"}
    return file_path.suffix.lower() in video_extensions


def create_temp_file(suffix: str = ".tmp", prefix: str = "vid_to_doc_") -> Path:
    """Create a temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        
    Returns:
        Path to the temporary file
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, prefix=prefix, delete=False)
    temp_file.close()
    return Path(temp_file.name)


def cleanup_temp_file(file_path: Path) -> None:
    """Clean up a temporary file.
    
    Args:
        file_path: Path to the temporary file
    """
    file_path = Path(file_path)
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        # Ignore cleanup errors
        pass 