# Refactor Plan: Video to Documentation Pipeline

## Current State Analysis

The current codebase is a functional but monolithic script that:
- Processes video files to extract audio, transcribe, summarize, and generate engineering docs
- Has all functionality in a single 325-line `main.py` file
- Lacks proper error handling, logging, and configuration management
- Has minimal testing and documentation
- Uses hardcoded values and lacks flexibility

## Refactor Goals

1. **Modular Architecture**: Break down the monolithic script into focused, testable modules
2. **Configuration Management**: Externalize settings and make the tool configurable
3. **Error Handling & Logging**: Add proper error handling, logging, and user feedback
4. **Testing**: Comprehensive test suite with proper mocking
5. **Documentation**: Clear API docs, usage examples, and contribution guidelines
6. **CLI Improvements**: Better argument parsing, progress indicators, and help text
7. **Code Quality**: Type hints, linting, and consistent code style

## Proposed Directory Structure

```
vid_to_doc/
├── README.md                    # Project overview and quick start
├── CONTRIBUTING.md              # Development guidelines
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore patterns
├── pyproject.toml              # Project metadata and dependencies
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
├── setup.py                     # Package installation
├── .env.example                 # Environment variables template
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configuration management
│   └── logging_config.py        # Logging configuration
├── src/
│   └── vid_to_doc/
│       ├── __init__.py
│       ├── cli.py               # Command-line interface
│       ├── core/
│       │   ├── __init__.py
│       │   ├── audio_extractor.py    # Audio extraction logic
│       │   ├── transcriber.py        # Transcription logic
│       │   ├── summarizer.py         # Text summarization
│       │   └── doc_generator.py      # Documentation generation
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── file_utils.py         # File operations
│       │   ├── audio_utils.py        # Audio processing utilities
│       │   └── text_utils.py         # Text processing utilities
│       ├── models/
│       │   ├── __init__.py
│       │   ├── config.py             # Data models and config classes
│       │   └── exceptions.py         # Custom exceptions
│       └── services/
│           ├── __init__.py
│           ├── openai_service.py     # OpenAI API wrapper
│           └── google_service.py     # Google Drive integration (future)
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_audio_extractor.py
│   │   ├── test_transcriber.py
│   │   ├── test_summarizer.py
│   │   └── test_doc_generator.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_end_to_end.py
│   │   └── test_cli.py
│   └── fixtures/
│       ├── sample_video.mp4
│       ├── sample_audio.wav
│       └── sample_transcript.txt
├── docs/
│   ├── api.md                   # API documentation
│   ├── cli.md                   # CLI usage guide
│   └── examples/
│       ├── basic_usage.md
│       └── advanced_usage.md
└── examples/
    ├── sample_videos/           # Example video files
    └── output_samples/          # Example outputs
```

## Module Breakdown

### 1. Core Modules (`src/vid_to_doc/core/`)

#### `audio_extractor.py`
- **Responsibilities**: Extract audio from video files, handle different formats
- **Key Classes**: `AudioExtractor`
- **Methods**: 
  - `extract_audio(video_path, output_path=None)`
  - `extract_sample(video_path, duration, output_path=None)`
  - `get_supported_formats()`

#### `transcriber.py`
- **Responsibilities**: Handle transcription using OpenAI Whisper API
- **Key Classes**: `Transcriber`
- **Methods**:
  - `transcribe(audio_path, model="gpt-4o-transcribe")`
  - `transcribe_chunked(audio_path, chunk_size_mb=25)`
  - `get_transcription_stats()`

#### `summarizer.py`
- **Responsibilities**: Generate summaries from transcriptions
- **Key Classes**: `Summarizer`
- **Methods**:
  - `summarize(text, max_tokens=500)`
  - `summarize_with_prompt(text, prompt, max_tokens=500)`

#### `doc_generator.py`
- **Responsibilities**: Generate engineering documentation
- **Key Classes**: `DocGenerator`
- **Methods**:
  - `generate_engineering_doc(summary, template="default")`
  - `generate_meeting_notes(transcript)`
  - `generate_action_items(transcript)`

### 2. Utility Modules (`src/vid_to_doc/utils/`)

#### `file_utils.py`
- **Responsibilities**: File operations, path handling, output management
- **Key Functions**:
  - `ensure_output_directory(base_path)`
  - `get_video_files_from_folder(folder_path)`
  - `save_text_to_file(text, file_path)`

#### `audio_utils.py`
- **Responsibilities**: Audio processing utilities
- **Key Functions**:
  - `get_audio_duration(audio_path)`
  - `split_audio_file(audio_path, chunk_duration)`
  - `validate_audio_file(audio_path)`

#### `text_utils.py`
- **Responsibilities**: Text processing and formatting
- **Key Functions**:
  - `clean_transcript(text)`
  - `format_timestamp(seconds)`
  - `extract_key_points(text)`

### 3. Service Modules (`src/vid_to_doc/services/`)

#### `openai_service.py`
- **Responsibilities**: OpenAI API interactions with proper error handling
- **Key Classes**: `OpenAIService`
- **Methods**:
  - `transcribe_audio(audio_file, model)`
  - `generate_summary(text, prompt)`
  - `generate_documentation(summary, template)`

### 4. Configuration (`src/vid_to_doc/models/`)

#### `config.py`
- **Responsibilities**: Configuration management and validation
- **Key Classes**: `Config`, `TranscriptionConfig`, `SummarizationConfig`
- **Features**:
  - Environment variable loading
  - Configuration validation
  - Default value management

## CLI Improvements

### New CLI Structure
```bash
# Basic usage
vid-to-doc process video.mp4
vid-to-doc process --folder videos/
vid-to-doc process --sample video.mp4 --duration 30

# Advanced options
vid-to-doc process video.mp4 --output-dir ./docs --format markdown
vid-to-doc process video.mp4 --config config.yaml --verbose

# Utility commands
vid-to-doc extract-audio video.mp4
vid-to-doc transcribe audio.wav
vid-to-doc summarize transcript.txt
vid-to-doc generate-doc summary.txt

# Configuration
vid-to-doc config show
vid-to-doc config set openai.api_key YOUR_KEY
vid-to-doc config validate
```

### CLI Features
- **Progress indicators**: Show progress for long-running operations
- **Verbose logging**: Multiple log levels (quiet, normal, verbose, debug)
- **Configuration management**: CLI commands for managing settings
- **Output formatting**: Multiple output formats (text, markdown, json)
- **Batch processing**: Process multiple files with progress tracking

## Configuration Management

### Configuration Sources (in order of precedence)
1. Command-line arguments
2. Configuration file (`config.yaml`)
3. Environment variables
4. Default values

### Configuration File Example (`config.yaml`)
```yaml
openai:
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-transcribe
  max_tokens: 500
  temperature: 0.7

transcription:
  chunk_size_mb: 25
  retry_attempts: 3
  retry_delay: 2

output:
  format: markdown
  directory: ./output
  include_timestamps: true
  include_confidence: false

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: vid_to_doc.log
```

## Testing Strategy

### Unit Tests
- **Mock external dependencies**: OpenAI API, file system operations
- **Test edge cases**: Large files, network errors, invalid inputs
- **Test configuration**: Config validation and loading
- **Test utilities**: File operations, text processing

### Integration Tests
- **End-to-end workflows**: Complete video processing pipeline
- **CLI testing**: Command-line interface functionality
- **Error handling**: Network failures, API rate limits
- **Performance testing**: Large file processing

### Test Fixtures
- **Sample files**: Small video/audio files for testing
- **Mock responses**: OpenAI API response examples
- **Configuration files**: Test configuration scenarios

## Documentation Improvements

### README.md Enhancements
- **Quick start guide**: Get running in 5 minutes
- **Installation options**: pip, conda, docker
- **Usage examples**: Common use cases with examples
- **Configuration guide**: How to customize behavior
- **Troubleshooting**: Common issues and solutions

### API Documentation
- **Module documentation**: Each module's purpose and usage
- **Class documentation**: All public classes and methods
- **Type hints**: Complete type annotations
- **Examples**: Code examples for each major function

### CLI Documentation
- **Command reference**: All available commands and options
- **Usage examples**: Real-world usage scenarios
- **Configuration**: CLI configuration options

## Code Quality Improvements

### Type Hints
- Add complete type annotations to all functions
- Use `typing` module for complex types
- Include type checking in CI/CD pipeline

### Linting and Formatting
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Code quality checks
- **mypy**: Type checking
- **pre-commit hooks**: Automated quality checks

### Error Handling
- **Custom exceptions**: Domain-specific error types
- **Graceful degradation**: Handle failures gracefully
- **User-friendly messages**: Clear error messages
- **Recovery strategies**: Automatic retry and fallback

## Migration Strategy

### Phase 1: Foundation (Week 1)
1. Set up new project structure
2. Create configuration management
3. Set up logging system
4. Create basic CLI framework

### Phase 2: Core Modules (Week 2)
1. Extract audio extraction logic
2. Extract transcription logic
3. Extract summarization logic
4. Extract documentation generation

### Phase 3: Utilities and Services (Week 3)
1. Create utility modules
2. Create service wrappers
3. Add error handling
4. Implement configuration validation

### Phase 4: Testing and Documentation (Week 4)
1. Write comprehensive tests
2. Create documentation
3. Set up CI/CD pipeline
4. Performance optimization

### Phase 5: Polish and Release (Week 5)
1. Final testing and bug fixes
2. Documentation review
3. Release preparation
4. Migration guide for existing users

## Benefits of Refactor

1. **Maintainability**: Modular code is easier to understand and modify
2. **Testability**: Smaller, focused modules are easier to test
3. **Reusability**: Components can be used independently
4. **Configurability**: Users can customize behavior without code changes
5. **Reliability**: Better error handling and logging
6. **Scalability**: Easier to add new features and integrations
7. **Community**: Better structure encourages contributions

## Risk Mitigation

1. **Backward Compatibility**: Maintain existing CLI interface during transition
2. **Gradual Migration**: Migrate functionality piece by piece
3. **Comprehensive Testing**: Ensure no functionality is lost
4. **Documentation**: Clear migration guide for existing users
5. **Rollback Plan**: Ability to revert to previous version if needed

This refactor plan transforms the current monolithic script into a professional, maintainable, and extensible Python package that will be much more suitable for open-source sharing and community contributions. 