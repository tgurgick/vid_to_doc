"""Unit tests for the audio_utils module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.vid_to_doc.utils.audio_utils import (
    get_audio_duration,
    get_audio_info,
    split_audio_file,
    validate_audio_file,
    convert_audio_format,
    get_audio_chunk_size_for_bytes,
    normalize_audio_levels,
)
from src.vid_to_doc.models.exceptions import AudioError


class TestAudioUtils:
    """Test cases for audio utilities."""

    @patch("pathlib.Path.exists", return_value=True)
    def test_get_audio_duration_wave_success(self, mock_exists):
        """Test getting audio duration using wave module."""
        with patch("wave.open") as mock_wave:
            mock_wf = Mock()
            mock_wf.getnframes.return_value = 44100  # 1 second at 44.1kHz
            mock_wf.getframerate.return_value = 44100
            mock_wave.return_value.__enter__.return_value = mock_wf
            
            result = get_audio_duration(Path("test.wav"))
            assert result == 1.0

    @patch("pathlib.Path.exists", return_value=True)
    @patch("wave.open", side_effect=Exception("Not a WAV file"))
    def test_get_audio_duration_soundfile_fallback(self, mock_wave, mock_exists):
        """Test getting audio duration using soundfile fallback."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            mock_sf.read.return_value = (np.array([0.1] * 44100), 44100)
            
            result = get_audio_duration(Path("test.mp3"))
            assert result == 1.0

    def test_get_audio_duration_file_not_exists(self):
        """Test getting audio duration for non-existent file."""
        with pytest.raises(AudioError, match="Audio file does not exist"):
            get_audio_duration(Path("nonexistent.wav"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_get_audio_info_success(self, mock_exists):
        """Test getting comprehensive audio information."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock audio data (1 second, 44.1kHz, mono)
            mock_data = np.array([0.1] * 44100)
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 88200  # 2 bytes per sample
                
                result = get_audio_info(Path("test.wav"))
                
                assert result["duration"] == 1.0
                assert result["sample_rate"] == 44100
                assert result["channels"] == 1
                assert result["samples"] == 44100
                assert result["file_size_bytes"] == 88200
                assert result["file_size_mb"] == 88200 / (1024 * 1024)
                assert result["format"] == ".wav"

    @patch("pathlib.Path.exists", return_value=True)
    def test_get_audio_info_stereo(self, mock_exists):
        """Test getting audio info for stereo file."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock stereo audio data
            mock_data = np.array([[0.1, 0.2]] * 44100)
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 176400
                
                result = get_audio_info(Path("test.wav"))
                assert result["channels"] == 2

    def test_get_audio_info_file_not_exists(self):
        """Test getting audio info for non-existent file."""
        with pytest.raises(AudioError, match="Audio file does not exist"):
            get_audio_info(Path("nonexistent.wav"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_split_audio_file_success(self, mock_exists):
        """Test splitting audio file into chunks."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock 3-second audio data
            mock_data = np.array([0.1] * (44100 * 3))
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.mkdir"):
                with patch("src.vid_to_doc.utils.audio_utils.sf.write") as mock_write:
                    result = split_audio_file(Path("test.wav"), 1.0)
                    
                    # Should create 3 chunks of 1 second each
                    assert len(result) == 3
                    assert mock_write.call_count == 3

    @patch("pathlib.Path.exists", return_value=True)
    def test_split_audio_file_invalid_duration(self, mock_exists):
        """Test splitting audio file with invalid duration."""
        with pytest.raises(AudioError, match="Chunk duration must be positive"):
            split_audio_file(Path("test.wav"), -1.0)

    def test_split_audio_file_file_not_exists(self):
        """Test splitting non-existent audio file."""
        with pytest.raises(AudioError, match="Audio file does not exist"):
            split_audio_file(Path("nonexistent.wav"), 1.0)

    @patch("pathlib.Path.exists", return_value=True)
    def test_validate_audio_file_success(self, mock_exists):
        """Test validating a valid audio file."""
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1000
            
            with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
                mock_sf.read.return_value = (np.array([0.1] * 1000), 44100)
                
                result = validate_audio_file(Path("test.wav"))
                assert result is True

    @patch("pathlib.Path.exists", return_value=True)
    def test_validate_audio_file_empty(self, mock_exists):
        """Test validating an empty audio file."""
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 0
            
            with pytest.raises(AudioError, match="Audio file is empty"):
                validate_audio_file(Path("test.wav"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_validate_audio_file_no_data(self, mock_exists):
        """Test validating audio file with no data."""
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1000
            
            with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
                mock_sf.read.return_value = (np.array([]), 44100)
                
                with pytest.raises(AudioError, match="Audio file contains no data"):
                    validate_audio_file(Path("test.wav"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_validate_audio_file_invalid_sample_rate(self, mock_exists):
        """Test validating audio file with invalid sample rate."""
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1000
            
            with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
                mock_sf.read.return_value = (np.array([0.1] * 1000), 0)
                
                with pytest.raises(AudioError, match="Invalid sample rate"):
                    validate_audio_file(Path("test.wav"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_convert_audio_format_success(self, mock_exists):
        """Test converting audio format."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            mock_sf.read.return_value = (np.array([0.1] * 1000), 44100)
            
            with patch("pathlib.Path.mkdir"):
                with patch("src.vid_to_doc.utils.audio_utils.sf.write") as mock_write:
                    result = convert_audio_format(
                        Path("input.wav"), 
                        Path("output.mp3"), 
                        "mp3"
                    )
                    
                    assert result == Path("output.mp3")
                    mock_write.assert_called_once()

    def test_convert_audio_format_input_not_exists(self):
        """Test converting non-existent audio file."""
        with pytest.raises(AudioError, match="Input audio file does not exist"):
            convert_audio_format(Path("nonexistent.wav"), Path("output.mp3"))

    @patch("pathlib.Path.exists", return_value=True)
    def test_get_audio_chunk_size_for_bytes_success(self, mock_exists):
        """Test calculating chunk size for byte limit."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock data with 4 bytes per sample (float32)
            mock_data = np.array([0.1] * 1000, dtype=np.float32)
            mock_sf.read.return_value = (mock_data, 44100)
            
            samples_per_chunk, total_chunks = get_audio_chunk_size_for_bytes(
                Path("test.wav"), 
                1000
            )
            
            # With 4 bytes per sample, 1000 bytes = 250 samples
            assert samples_per_chunk == 250
            assert total_chunks == 4  # 1000 samples / 250 per chunk

    def test_get_audio_chunk_size_for_bytes_file_not_exists(self):
        """Test calculating chunk size for non-existent file."""
        with pytest.raises(AudioError, match="Audio file does not exist"):
            get_audio_chunk_size_for_bytes(Path("nonexistent.wav"), 1000)

    @patch("pathlib.Path.exists", return_value=True)
    def test_normalize_audio_levels_success(self, mock_exists):
        """Test normalizing audio levels."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock audio data with high amplitude
            mock_data = np.array([0.8, -0.9, 0.7, -0.6])
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.mkdir"):
                with patch("src.vid_to_doc.utils.audio_utils.sf.write") as mock_write:
                    result = normalize_audio_levels(Path("input.wav"))
                    
                    # Should create normalized file
                    expected_path = Path("input_normalized.wav")
                    assert result == expected_path
                    mock_write.assert_called_once()

    @patch("pathlib.Path.exists", return_value=True)
    def test_normalize_audio_levels_zero_amplitude(self, mock_exists):
        """Test normalizing audio with zero amplitude."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            # Mock audio data with zero amplitude
            mock_data = np.array([0.0, 0.0, 0.0, 0.0])
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.mkdir"):
                with patch("src.vid_to_doc.utils.audio_utils.sf.write") as mock_write:
                    result = normalize_audio_levels(Path("input.wav"))
                    
                    # Should still create file with original data
                    assert result == Path("input_normalized.wav")
                    mock_write.assert_called_once()

    @patch("pathlib.Path.exists", return_value=True)
    def test_normalize_audio_levels_custom_output(self, mock_exists):
        """Test normalizing audio with custom output path."""
        with patch("src.vid_to_doc.utils.audio_utils.sf") as mock_sf:
            mock_data = np.array([0.5, -0.5])
            mock_sf.read.return_value = (mock_data, 44100)
            
            with patch("pathlib.Path.mkdir"):
                with patch("src.vid_to_doc.utils.audio_utils.sf.write") as mock_write:
                    result = normalize_audio_levels(
                        Path("input.wav"), 
                        Path("custom_output.wav")
                    )
                    
                    assert result == Path("custom_output.wav")

    def test_normalize_audio_levels_file_not_exists(self):
        """Test normalizing non-existent audio file."""
        with pytest.raises(AudioError, match="Audio file does not exist"):
            normalize_audio_levels(Path("nonexistent.wav")) 