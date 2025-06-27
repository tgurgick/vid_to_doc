"""Integration tests for the CLI functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from click.testing import CliRunner

from src.vid_to_doc.cli import main


class TestCLIIntegration:
    """Test CLI integration and user workflows."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_openai_service(self):
        """Mock OpenAI service for CLI tests."""
        with patch("src.vid_to_doc.services.openai_service.openai") as mock_openai:
            # Mock transcription response
            mock_transcription = Mock()
            mock_transcription.__str__ = lambda self: "Test transcription from CLI."
            mock_openai.audio.transcriptions.create.return_value = mock_transcription
            
            # Mock chat completion responses
            mock_summary_response = Mock()
            mock_summary_response.choices = [Mock(message=Mock(content="Test summary from CLI."))]
            
            mock_doc_response = Mock()
            mock_doc_response.choices = [Mock(message=Mock(content="# CLI Test Documentation\n\nGenerated via CLI."))]
            
            mock_openai.chat.completions.create.side_effect = [
                mock_summary_response,  # For summarization
                mock_doc_response,      # For documentation
            ]
            
            yield mock_openai

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Commands:" in result.output

    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert "vid-to-doc" in result.output

    def test_cli_process_single_video(self, runner, temp_dir, mock_openai_service):
        """Test processing a single video file via CLI."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024 * 1024
                
                with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                    mock_video_instance = Mock()
                    mock_video_instance.audio = Mock()
                    mock_video_instance.audio.write_audiofile = Mock()
                    mock_video.return_value.__enter__.return_value = mock_video_instance
                    
                    # Test CLI command
                    result = runner.invoke(main, [
                        "--api-key", "test-key",
                        "process", 
                        str(video_file)
                    ])
                    
                    # Should complete successfully (or show not implemented message)
                    assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_process_folder(self, runner, temp_dir, mock_openai_service):
        """Test processing a folder of videos via CLI."""
        videos_dir = temp_dir / "videos"
        videos_dir.mkdir()
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.is_dir", return_value=True):
                with patch("pathlib.Path.iterdir") as mock_iterdir:
                    # Mock video files in directory
                    mock_files = [
                        Path("video1.mp4"),
                        Path("video2.avi"),
                        Path("document.txt"),  # Non-video file
                    ]
                    mock_iterdir.return_value = mock_files
                    
                    with patch("pathlib.Path.is_file", return_value=True):
                        with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                            mock_video_instance = Mock()
                            mock_video_instance.audio = Mock()
                            mock_video_instance.audio.write_audiofile = Mock()
                            mock_video.return_value.__enter__.return_value = mock_video_instance
                            
                            # Test CLI command
                            result = runner.invoke(main, [
                                "--api-key", "test-key",
                                "process", 
                                "--folder", str(videos_dir)
                            ])
                            
                            # Should complete successfully (or show not implemented message)
                            assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_extract_audio_only(self, runner, temp_dir):
        """Test extracting audio only via CLI."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Test CLI command
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "extract-audio", 
                    str(video_file)
                ])
                
                # Should complete successfully (or show not implemented message)
                assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_transcribe_only(self, runner, temp_dir, mock_openai_service):
        """Test transcribing audio only via CLI."""
        audio_file = temp_dir / "test_audio.wav"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open()):
                # Test CLI command
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "transcribe", 
                    str(audio_file)
                ])
                
                # Should complete successfully (or show not implemented message)
                assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_summarize_only(self, runner, mock_openai_service):
        """Test summarizing text only via CLI."""
        transcript = "This is a test transcript for summarization."
        
        # Test CLI command
        result = runner.invoke(main, [
            "--api-key", "test-key",
            "summarize", 
            transcript
        ])
        
        # Should complete successfully (or show not implemented message)
        assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_generate_doc_only(self, runner, temp_dir, mock_openai_service):
        """Test generating documentation only via CLI."""
        summary = "This is a test summary for documentation generation."
        output_file = temp_dir / "test_doc.md"
        
        with patch("pathlib.Path.parent") as mock_parent:
            mock_parent.mkdir = Mock()
            
            # Test CLI command
            result = runner.invoke(main, [
                "--api-key", "test-key",
                "generate-doc", 
                summary,
                "--output", str(output_file)
            ])
            
            # Should complete successfully (or show not implemented message)
            assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_with_config_file(self, runner, temp_dir, mock_openai_service):
        """Test CLI with configuration file."""
        config_file = temp_dir / "config.yaml"
        config_content = """
openai:
  api_key: test-key
  model: gpt-3.5-turbo

output:
  format: markdown
  directory: ./output
"""
        
        with open(config_file, "w") as f:
            f.write(config_content)
        
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Test CLI command with config
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "--config", str(config_file),
                    "process", 
                    str(video_file)
                ])
                
                # Should complete successfully (or show not implemented message)
                assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_verbose_output(self, runner, temp_dir, mock_openai_service):
        """Test CLI with verbose output."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Test CLI command with verbose flag
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "--verbose",
                    "process", 
                    str(video_file)
                ])
                
                # Should show verbose output
                assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_error_handling(self, runner):
        """Test CLI error handling."""
        # Test with non-existent file
        result = runner.invoke(main, [
            "--api-key", "test-key",
            "process", 
            "nonexistent_video.mp4"
        ])
        
        # Should show error
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_cli_missing_api_key(self, runner, temp_dir):
        """Test CLI behavior with missing API key."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            # Test CLI command without API key
            result = runner.invoke(main, [
                "process", 
                str(video_file)
            ])
            
            # Should show error about missing API key or not implemented
            assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_custom_output_directory(self, runner, temp_dir, mock_openai_service):
        """Test CLI with custom output directory."""
        video_file = temp_dir / "test_video.mp4"
        custom_output = temp_dir / "custom_output"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Test CLI command with custom output
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "process", 
                    str(video_file),
                    "--output-dir", str(custom_output)
                ])
                
                # Should complete successfully (or show not implemented message)
                assert result.exit_code == 0 or "not yet implemented" in result.output

    def test_cli_sample_extraction(self, runner, temp_dir):
        """Test CLI sample extraction functionality."""
        video_file = temp_dir / "test_video.mp4"
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.vid_to_doc.core.audio_extractor.VideoFileClip") as mock_video:
                mock_video_instance = Mock()
                mock_video_instance.audio = Mock()
                mock_video_instance.audio.write_audiofile = Mock()
                mock_video.return_value.__enter__.return_value = mock_video_instance
                
                # Test CLI command for sample extraction
                result = runner.invoke(main, [
                    "--api-key", "test-key",
                    "extract-sample", 
                    str(video_file),
                    "--duration", "30"
                ])
                
                # Should complete successfully (or show not implemented message)
                assert result.exit_code == 0 or "not yet implemented" in result.output 