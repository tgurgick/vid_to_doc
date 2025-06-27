"""Configuration management for the vid_to_doc package."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from .exceptions import ConfigurationError


@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings."""
    
    model: str = "gpt-4o-transcribe"
    chunk_size_mb: int = 25
    retry_attempts: int = 3
    retry_delay: int = 2
    max_tokens: int = 500
    temperature: float = 0.7


@dataclass
class SummarizationConfig:
    """Configuration for summarization settings."""
    
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    system_prompt: str = "Summarize the following transcription into concise knowledge notes."


@dataclass
class OutputConfig:
    """Configuration for output settings."""
    
    format: str = "markdown"
    directory: str = "./output"
    include_timestamps: bool = True
    include_confidence: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


@dataclass
class Config:
    """Main configuration class for the application."""
    
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_credentials_path: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_CREDENTIALS_PATH")
    )
    
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate the configuration."""
        if not self.openai_api_key:
            raise ConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in config file."
            )
        
        if self.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError(f"Invalid logging level: {self.logging.level}")
        
        if self.output.format not in ["markdown", "text", "json"]:
            raise ConfigurationError(f"Invalid output format: {self.output.format}")
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "Config":
        """Create configuration from a dictionary."""
        # Handle environment variable substitution
        config_data = cls._substitute_env_vars(config_data)
        
        # Extract nested configurations
        transcription_data = config_data.get("transcription", {})
        summarization_data = config_data.get("summarization", {})
        output_data = config_data.get("output", {})
        logging_data = config_data.get("logging", {})
        
        return cls(
            openai_api_key=config_data.get("openai", {}).get("api_key", ""),
            google_credentials_path=config_data.get("google", {}).get("credentials_path"),
            transcription=TranscriptionConfig(**transcription_data),
            summarization=SummarizationConfig(**summarization_data),
            output=OutputConfig(**output_data),
            logging=LoggingConfig(**logging_data),
        )
    
    @staticmethod
    def _substitute_env_vars(data: Any) -> Any:
        """Recursively substitute environment variables in configuration data."""
        if isinstance(data, dict):
            return {k: Config._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var, data)
        else:
            return data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "openai": {
                "api_key": self.openai_api_key,
            },
            "google": {
                "credentials_path": self.google_credentials_path,
            },
            "transcription": {
                "model": self.transcription.model,
                "chunk_size_mb": self.transcription.chunk_size_mb,
                "retry_attempts": self.transcription.retry_attempts,
                "retry_delay": self.transcription.retry_delay,
                "max_tokens": self.transcription.max_tokens,
                "temperature": self.transcription.temperature,
            },
            "summarization": {
                "model": self.summarization.model,
                "max_tokens": self.summarization.max_tokens,
                "temperature": self.summarization.temperature,
                "system_prompt": self.summarization.system_prompt,
            },
            "output": {
                "format": self.output.format,
                "directory": self.output.directory,
                "include_timestamps": self.output.include_timestamps,
                "include_confidence": self.output.include_confidence,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
            },
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration file: {e}")


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment variables."""
    # Load environment variables
    load_dotenv()
    
    if config_path:
        return Config.from_file(config_path)
    
    # Try to load from default config file
    default_config_path = Path("config.yaml")
    if default_config_path.exists():
        return Config.from_file(default_config_path)
    
    # Fall back to environment variables only
    return Config() 