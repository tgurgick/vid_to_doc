"""Unit tests for the AudioExtractor class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.vid_to_doc.core.audio_extractor import AudioExtractor
from src.vid_to_doc.models.exceptions import AudioExtractionError


class TestAudioExtractor:
    """Test cases for AudioExtractor class."""

    def test_get_supported_formats(self):
        """Test that get_supported_formats returns the expected list."""
        formats = AudioExtractor.get_supported_formats()
        expected_formats = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm"]
        assert formats == expected_formats

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_audio_success(self, mock_video_clip, mock_exists):
        """Test successful audio extraction."""
        # Setup mock
        mock_video = Mock()
        mock_audio = Mock()
        mock_video.audio = mock_audio
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        result = AudioExtractor.extract_audio(video_path)

        # Assertions
        assert result == video_path.with_suffix(".wav")
        mock_video_clip.assert_called_once_with(str(video_path))
        mock_audio.write_audiofile.assert_called_once_with(
            str(video_path.with_suffix(".wav")), codec="pcm_s16le"
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_audio_with_custom_output_path(self, mock_video_clip, mock_exists):
        """Test audio extraction with custom output path."""
        # Setup mock
        mock_video = Mock()
        mock_audio = Mock()
        mock_video.audio = mock_audio
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        output_path = Path("custom_output.wav")
        result = AudioExtractor.extract_audio(video_path, output_path)

        # Assertions
        assert result == output_path
        mock_audio.write_audiofile.assert_called_once_with(
            str(output_path), codec="pcm_s16le"
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_audio_no_audio_track(self, mock_video_clip, mock_exists):
        """Test audio extraction when video has no audio track."""
        # Setup mock
        mock_video = Mock()
        mock_video.audio = None
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        with pytest.raises(AudioExtractionError, match="No audio track found"):
            AudioExtractor.extract_audio(video_path)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_audio_videoclip_error(self, mock_video_clip, mock_exists):
        """Test audio extraction when VideoFileClip raises an exception."""
        # Setup mock to raise exception
        mock_video_clip.side_effect = Exception("Video file error")

        # Test
        video_path = Path("test_video.mp4")
        with pytest.raises(AudioExtractionError, match="Failed to extract audio"):
            AudioExtractor.extract_audio(video_path)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_sample_success(self, mock_video_clip, mock_exists):
        """Test successful sample extraction."""
        # Setup mock
        mock_video = Mock()
        mock_video.duration = 60.0
        mock_sample_clip = Mock()
        mock_audio = Mock()
        mock_sample_clip.audio = mock_audio
        mock_video.subclipped.return_value = mock_sample_clip
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        result = AudioExtractor.extract_sample(video_path, 15)

        # Assertions
        expected_output = video_path.with_name("test_video_sample_15s.wav")
        assert result == expected_output
        mock_video.subclipped.assert_called_once_with(22.5, 37.5)
        mock_audio.write_audiofile.assert_called_once_with(
            str(expected_output), codec="pcm_s16le"
        )

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_sample_with_subclip_fallback(self, mock_video_clip, mock_exists):
        """Test sample extraction using subclip method (MoviePy v1)."""
        # Setup mock
        mock_video = Mock()
        mock_video.duration = 60.0
        mock_sample_clip = Mock()
        mock_audio = Mock()
        mock_sample_clip.audio = mock_audio
        # Remove subclipped method to test fallback
        del mock_video.subclipped
        mock_video.subclip.return_value = mock_sample_clip
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        result = AudioExtractor.extract_sample(video_path, 15)

        # Assertions
        expected_output = video_path.with_name("test_video_sample_15s.wav")
        assert result == expected_output
        mock_video.subclip.assert_called_once_with(22.5, 37.5)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_sample_video_too_short(self, mock_video_clip, mock_exists):
        """Test sample extraction when video is too short."""
        # Setup mock
        mock_video = Mock()
        mock_video.duration = 10.0  # Shorter than requested sample
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        with pytest.raises(AudioExtractionError, match="Video too short for sampling"):
            AudioExtractor.extract_sample(video_path, 15)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_sample_no_audio_track(self, mock_video_clip, mock_exists):
        """Test sample extraction when video has no audio track."""
        # Setup mock
        mock_video = Mock()
        mock_video.duration = 60.0
        mock_sample_clip = Mock()
        mock_sample_clip.audio = None
        mock_video.subclipped.return_value = mock_sample_clip
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        with pytest.raises(AudioExtractionError, match="No audio track found"):
            AudioExtractor.extract_sample(video_path, 15)

    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.vid_to_doc.core.audio_extractor.VideoFileClip")
    def test_extract_sample_with_custom_output_path(self, mock_video_clip, mock_exists):
        """Test sample extraction with custom output path."""
        # Setup mock
        mock_video = Mock()
        mock_video.duration = 60.0
        mock_sample_clip = Mock()
        mock_audio = Mock()
        mock_sample_clip.audio = mock_audio
        mock_video.subclipped.return_value = mock_sample_clip
        mock_video_clip.return_value.__enter__.return_value = mock_video

        # Test
        video_path = Path("test_video.mp4")
        output_path = Path("custom_sample.wav")
        result = AudioExtractor.extract_sample(video_path, 15, output_path)

        # Assertions
        assert result == output_path
        mock_audio.write_audiofile.assert_called_once_with(
            str(output_path), codec="pcm_s16le"
        )

    @patch("pathlib.Path.exists", return_value=True)
    def test_extract_audio_path_conversion(self, mock_exists):
        """Test that string paths are converted to Path objects."""
        with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video_clip:
            # Setup mock
            mock_video = Mock()
            mock_audio = Mock()
            mock_video.audio = mock_audio
            mock_video_clip.return_value.__enter__.return_value = mock_video

            # Test with string path
            result = AudioExtractor.extract_audio("test_video.mp4")
            assert isinstance(result, Path)
            assert result == Path("test_video.wav")

    @patch("pathlib.Path.exists", return_value=True)
    def test_extract_sample_path_conversion(self, mock_exists):
        """Test that string paths are converted to Path objects in sample extraction."""
        with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video_clip:
            # Setup mock
            mock_video = Mock()
            mock_video.duration = 60.0
            mock_sample_clip = Mock()
            mock_audio = Mock()
            mock_sample_clip.audio = mock_audio
            mock_video.subclipped.return_value = mock_sample_clip
            mock_video_clip.return_value.__enter__.return_value = mock_video

            # Test with string path
            result = AudioExtractor.extract_sample("test_video.mp4", 15)
            assert isinstance(result, Path)
            assert result == Path("test_video_sample_15s.wav") 