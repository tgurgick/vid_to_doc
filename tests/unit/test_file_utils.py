"""Unit tests for the file_utils module."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, Mock

from src.vid_to_doc.utils.file_utils import (
    ensure_output_directory,
    get_video_files_from_folder,
    save_text_to_file,
    get_file_size,
    is_audio_file,
    is_video_file,
    create_temp_file,
    cleanup_temp_file,
)
from src.vid_to_doc.models.exceptions import FileError


class TestFileUtils:
    """Test cases for file utilities."""

    def test_ensure_output_directory_success(self):
        """Test creating output directory."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = ensure_output_directory(Path("/path/to/file.txt"))
            expected_path = Path("/path/to/output")
            assert result == expected_path
            mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_get_video_files_from_folder_success(self):
        """Test getting video files from folder."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.iterdir") as mock_iterdir:
                    # Mock directory contents with proper Path objects
                    mock_files = [
                        Path("/videos/file1.mp4"),
                        Path("/videos/file2.avi"),
                        Path("/videos/file3.txt"),  # Non-video file
                        Path("/videos/file4.MP4"),  # Case insensitive
                    ]
                    mock_iterdir.return_value = mock_files
                    with patch("pathlib.Path.is_file", return_value=True):
                        result = get_video_files_from_folder(Path("/videos"))
                        # Should return only video files, sorted
                        assert len(result) == 3
                        assert all(f.suffix.lower() in [".mp4", ".avi"] for f in result)

    def test_get_video_files_from_folder_not_exists(self):
        """Test getting video files from non-existent folder."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileError, match="Folder does not exist"):
                get_video_files_from_folder(Path("/nonexistent"))

    def test_get_video_files_from_folder_not_directory(self):
        """Test getting video files from path that's not a directory."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=False):
                with pytest.raises(FileError, match="Path is not a directory"):
                    get_video_files_from_folder(Path("/file.txt"))

    def test_save_text_to_file_success(self):
        """Test saving text to file."""
        with patch("pathlib.Path.parent") as mock_parent:
            mock_parent.mkdir = Mock()
            with patch("builtins.open", mock_open()) as mock_file:
                result = save_text_to_file("Hello world", Path("/tmp/test.txt"))
                assert result == Path("/tmp/test.txt")
                mock_file.assert_called_once_with(Path("/tmp/test.txt"), "w", encoding="utf-8")

    def test_save_text_to_file_custom_encoding(self):
        """Test saving text to file with custom encoding."""
        with patch("pathlib.Path.parent") as mock_parent:
            mock_parent.mkdir = Mock()
            with patch("builtins.open", mock_open()) as mock_file:
                save_text_to_file("Hello world", Path("/tmp/test.txt"), "latin-1")
                mock_file.assert_called_once_with(Path("/tmp/test.txt"), "w", encoding="latin-1")

    def test_save_text_to_file_error(self):
        """Test saving text to file with error."""
        with patch("pathlib.Path.parent") as mock_parent:
            mock_parent.mkdir = Mock()
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                with pytest.raises(FileError, match="Failed to save file"):
                    save_text_to_file("Hello world", Path("/tmp/test.txt"))

    def test_get_file_size_success(self):
        """Test getting file size."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024
                result = get_file_size(Path("/tmp/test.txt"))
                assert result == 1024

    def test_get_file_size_not_exists(self):
        """Test getting size of non-existent file."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileError, match="File does not exist"):
                get_file_size(Path("/nonexistent.txt"))

    def test_get_file_size_error(self):
        """Test getting file size with error."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", side_effect=PermissionError("Permission denied")):
                with pytest.raises(FileError, match="Failed to get file size"):
                    get_file_size(Path("/tmp/test.txt"))

    def test_is_audio_file_true(self):
        """Test identifying audio files."""
        assert is_audio_file(Path("test.wav")) is True
        assert is_audio_file(Path("test.MP3")) is True  # Case insensitive
        assert is_audio_file(Path("test.flac")) is True

    def test_is_audio_file_false(self):
        """Test identifying non-audio files."""
        assert is_audio_file(Path("test.txt")) is False
        assert is_audio_file(Path("test.mp4")) is False

    def test_is_video_file_true(self):
        """Test identifying video files."""
        assert is_video_file(Path("test.mp4")) is True
        assert is_video_file(Path("test.AVI")) is True  # Case insensitive
        assert is_video_file(Path("test.mkv")) is True

    def test_is_video_file_false(self):
        """Test identifying non-video files."""
        assert is_video_file(Path("test.txt")) is False
        assert is_video_file(Path("test.wav")) is False

    def test_create_temp_file(self):
        """Test creating temporary file."""
        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.name = "/tmp/vid_to_doc_test_123.tmp"
            mock_temp.return_value.close = Mock()
            
            result = create_temp_file(suffix=".wav", prefix="test_")
            
            assert result == Path("/tmp/vid_to_doc_test_123.tmp")
            mock_temp.assert_called_once_with(
                suffix=".wav", 
                prefix="test_", 
                delete=False
            )

    def test_cleanup_temp_file_exists(self):
        """Test cleaning up existing temporary file."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.unlink") as mock_unlink:
                cleanup_temp_file(Path("/tmp/test.tmp"))
                mock_unlink.assert_called_once()

    def test_cleanup_temp_file_not_exists(self):
        """Test cleaning up non-existent temporary file."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("pathlib.Path.unlink") as mock_unlink:
                cleanup_temp_file(Path("/tmp/test.tmp"))
                mock_unlink.assert_not_called()

    def test_cleanup_temp_file_error(self):
        """Test cleaning up temporary file with error."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.unlink", side_effect=PermissionError("Permission denied")):
                # Should not raise exception, just ignore the error
                cleanup_temp_file(Path("/tmp/test.tmp"))

    def test_video_file_extensions(self):
        """Test all supported video file extensions."""
        supported_videos = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"]
        for ext in supported_videos:
            assert is_video_file(Path(f"test{ext}")) is True
            assert is_video_file(Path(f"test{ext.upper()}")) is True

    def test_audio_file_extensions(self):
        """Test all supported audio file extensions."""
        supported_audio = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mpga", ".webm"]
        for ext in supported_audio:
            assert is_audio_file(Path(f"test{ext}")) is True
            assert is_audio_file(Path(f"test{ext.upper()}")) is True 