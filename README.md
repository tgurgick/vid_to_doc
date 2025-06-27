# Video to Documentation Pipeline

Convert video files to documentation using AI transcription and summarization.

## ⚠️ Important Disclaimer

**YouTube Download Functionality**: The YouTube download features in this tool are provided for **educational and non-commercial testing purposes only**. Users are responsible for ensuring compliance with YouTube's Terms of Service and applicable copyright laws. This tool should only be used with content you own or have permission to download.

## Features

- 🎥 **Video Processing**: Extract audio from various video formats
- 🎤 **AI Transcription**: Transcribe audio using OpenAI's GPT-4o transcription model
- 📝 **Smart Summarization**: Generate concise summaries using GPT-4o
- 📚 **Documentation Generation**: Create engineering documentation from summaries
- ⚙️ **Configurable**: Flexible configuration via files, environment variables, or CLI
- 🚀 **CLI Interface**: Easy-to-use command-line interface
- 📊 **Progress Tracking**: Visual progress indicators for long-running operations
- 📥 **YouTube Integration**: Download videos for testing (educational use only)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vid-to-doc.git
cd vid-to-doc

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

1. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Process a video file**:
   ```bash
   vid-to-doc process video.mp4
   ```

3. **Process all videos in a folder**:
   ```bash
   vid-to-doc process videos/
   ```

4. **Extract just the audio**:
   ```bash
   vid-to-doc extract-audio video.mp4
   ```

5. **Transcribe an audio file**:
   ```bash
   vid-to-doc transcribe audio.wav
   ```

## YouTube Integration (Educational Use Only)

⚠️ **Important**: YouTube download functionality is for **educational and non-commercial testing purposes only**. Ensure compliance with YouTube's Terms of Service and copyright laws.

### YouTube Commands

```bash
# Get video information (no download)
vid-to-doc info "https://youtube.com/watch?v=VIDEO_ID"

# Download video for testing
vid-to-doc download "https://youtube.com/watch?v=VIDEO_ID" --output-dir ./videos --quality 720p

# Download and process in one command
vid-to-doc --api-key "your-key" download-and-process "https://youtube.com/watch?v=VIDEO_ID"
```

### YouTube Download Options

```bash
# Download with specific quality
vid-to-doc download URL --quality 480p

# Download audio only
vid-to-doc download URL --audio-only

# Custom filename
vid-to-doc download URL --filename "my_video"

# Specify output directory
vid-to-doc download URL --output-dir ./downloads
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional
VID_TO_DOC_LOG_LEVEL=INFO
VID_TO_DOC_OUTPUT_DIR=./output
VID_TO_DOC_CONFIG_FILE=config.yaml
```

### Configuration File

Create a `config.yaml` file for more advanced configuration:

```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-transcribe

transcription:
  chunk_size_mb: 25
  retry_attempts: 3
  retry_delay: 2

output:
  format: markdown
  directory: ./output
  include_timestamps: true

logging:
  level: INFO
  file: vid_to_doc.log
```

### CLI Configuration

```bash
# Show current configuration
vid-to-doc config show

# Validate configuration
vid-to-doc config validate

# Set configuration values
vid-to-doc config set openai.model gpt-4o-transcribe
```

## Advanced Usage

### Process with Custom Options

```bash
# Specify output directory and format
vid-to-doc process video.mp4 --output-dir ./docs --format markdown

# Process only a sample of the video
vid-to-doc process video.mp4 --sample-duration 30

# Use verbose logging
vid-to-doc process video.mp4 --verbose

# Use custom configuration file
vid-to-doc process video.mp4 --config my_config.yaml
```

### Individual Steps

```bash
# Extract audio from video
vid-to-doc extract-audio video.mp4 --output-path audio.wav

# Transcribe audio file
vid-to-doc transcribe audio.wav

# Summarize text file
vid-to-doc summarize transcript.txt
```

## Output Files

For each processed video, the following files are generated:

- `{video_name}_transcription.txt`: Full transcription
- `{video_name}_summary.txt`: Concise summary
- `{video_name}_engineering_doc.md`: Engineering documentation

## Supported Formats

### Video Formats
- MP4, MOV, AVI, MKV, FLV, WMV, WebM

### Audio Formats
- WAV, MP3, M4A, FLAC, OGG, MPGA, WebM

### Output Formats
- Markdown (default)
- Plain text
- JSON

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

### Project Structure

```
vid_to_doc/
├── src/vid_to_doc/          # Main package
│   ├── core/               # Core functionality
│   ├── utils/              # Utility functions
│   ├── models/             # Data models
│   ├── services/           # External service wrappers
│   └── cli.py              # Command-line interface
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Example files
└── config/                 # Configuration files
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/vid_to_doc

# Run specific test categories
pytest -m unit
pytest -m integration

# Run slow tests
pytest -m slow
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Run linting: `black src/ tests/ && isort src/ tests/ && flake8 src/ tests/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Google Drive integration
- [ ] Web interface
- [ ] Batch processing improvements
- [ ] More output formats
- [ ] Custom summarization prompts
- [ ] Speaker identification
- [ ] Timestamp extraction
- [ ] Multi-language support

## Troubleshooting

### Common Issues

**"OpenAI API key is required"**
- Set the `OPENAI_API_KEY` environment variable
- Or add it to your configuration file

**"File not found"**
- Ensure the video file exists and is accessible
- Check file permissions

**"Transcription failed"**
- Verify your OpenAI API key is valid
- Check your internet connection
- Ensure the audio file is not corrupted

**"Audio extraction failed"**
- Verify the video file has an audio track
- Check that the video format is supported
- Ensure sufficient disk space

### Getting Help

- Check the [documentation](docs/)
- Search [existing issues](https://github.com/yourusername/vid-to-doc/issues)
- Create a [new issue](https://github.com/yourusername/vid-to-doc/issues/new)

## Acknowledgments

- OpenAI for providing the GPT-4o transcription and completion APIs
- MoviePy for video processing capabilities
- Click for the excellent CLI framework
- Rich for beautiful terminal output