"""Logging utilities for the vid_to_doc package."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

from ..models.config import LoggingConfig


def setup_logging(config: LoggingConfig, console: Optional[Console] = None) -> None:
    """Set up logging with the given configuration.
    
    Args:
        config: Logging configuration
        console: Rich console for colored output (optional)
    """
    # Create logger
    logger = logging.getLogger("vid_to_doc")
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Console handler with Rich formatting if available
    if console:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, config.level.upper()))
        logger.addHandler(console_handler)
    else:
        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler if specified
    if config.file:
        file_path = Path(config.file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set up third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("moviepy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized with level: {config.level}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"vid_to_doc.{name}") 