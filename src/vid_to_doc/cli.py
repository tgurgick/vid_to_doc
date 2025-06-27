"""Command-line interface for the vid_to_doc package."""

import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from .models.config import Config, load_config
from .core.audio_extractor import AudioExtractor
from .core.transcriber import Transcriber
from .core.summarizer import Summarizer
from .core.doc_generator import DocGenerator
from .utils.file_utils import get_video_files_from_folder, save_text_to_file, ensure_output_directory
from config.logging_config import setup_logging, get_logger

# Initialize console for rich output
console = Console()


@dataclass
class MinimalConfig:
    """Minimal configuration for simple CLI commands."""
    openai_api_key: str = ""
    logging_level: str = "INFO"
    
    @property
    def logging(self):
        from .models.config import LoggingConfig
        return LoggingConfig(level=self.logging_level)
    
    @property
    def output(self):
        from .models.config import OutputConfig
        return OutputConfig()
    
    @property
    def transcription(self):
        from .models.config import TranscriptionConfig
        return TranscriptionConfig()
    
    @property
    def summarization(self):
        from .models.config import SummarizationConfig
        return SummarizationConfig()


# Global configuration object
class CLIContext:
    def __init__(self):
        self.config = None
        self.api_key = None
        self.verbose = False
        self.quiet = False

def pass_cli_context(f):
    """Decorator to pass CLI context to commands."""
    return click.pass_obj(f)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option(
    "--api-key",
    type=str,
    help="OpenAI API key (overrides config file and environment variable)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress output except errors",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path], api_key: Optional[str], verbose: bool, quiet: bool) -> None:
    """Video to Documentation Pipeline.
    
    Convert video files to documentation using AI transcription and summarization.
    """
    # Create CLI context object
    cli_ctx = CLIContext()
    cli_ctx.api_key = api_key
    cli_ctx.verbose = verbose
    cli_ctx.quiet = quiet
    
    # Skip configuration loading for simple commands
    if ctx.invoked_subcommand in ['version', 'help']:
        # Create minimal config for simple commands
        cfg = MinimalConfig()
        if verbose:
            cfg.logging_level = "DEBUG"
        elif quiet:
            cfg.logging_level = "ERROR"
        
        # Set up logging
        setup_logging(cfg.logging, console)
        cli_ctx.config = cfg
        ctx.obj = cli_ctx
        return
    
    # Load configuration for other commands
    try:
        cfg = load_config(config)
        
        # Override API key if provided via CLI
        if api_key:
            cfg.openai_api_key = api_key
        
        # Override log level based on CLI options
        if verbose:
            cfg.logging.level = "DEBUG"
        elif quiet:
            cfg.logging.level = "ERROR"
        
        # Set up logging
        setup_logging(cfg.logging, console)
        
        # Store configuration in context
        cli_ctx.config = cfg
        
    except Exception as e:
        # If config loading fails, create a minimal config for basic commands
        console.print(f"[yellow]Warning: Configuration error: {e}[/yellow]")
        console.print("[yellow]Using minimal configuration. Some commands may require API key.[/yellow]")
        
        # Create minimal config
        cfg = MinimalConfig()
        if api_key:
            cfg.openai_api_key = api_key
        if verbose:
            cfg.logging_level = "DEBUG"
        elif quiet:
            cfg.logging_level = "ERROR"
        
        # Set up logging
        setup_logging(cfg.logging, console)
        
        # Store configuration in context
        cli_ctx.config = cfg
    
    ctx.obj = cli_ctx


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--folder",
    "-f",
    is_flag=True,
    help="Treat input_path as a folder containing video files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for generated files",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "text", "json"]),
    default="markdown",
    help="Output format for documentation",
)
@click.option(
    "--sample-duration",
    "-d",
    type=int,
    default=0,
    help="Extract only a sample of specified duration (seconds). 0 = full video",
)
@pass_cli_context
def process(
    cli_ctx: CLIContext,
    input_path: Path,
    folder: bool,
    output_dir: Optional[Path],
    format: str,
    sample_duration: int,
) -> None:
    """Process a video file and generate documentation.
    
    INPUT_PATH can be a video file or a directory containing video files.
    Use --folder flag to explicitly treat input_path as a folder.
    """
    logger = get_logger("cli.process")
    config = cli_ctx.config
    
    # Check if API key is available
    if not config.openai_api_key:
        console.print("[red]Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable, provide in config file, or use --api-key option.[/red]")
        sys.exit(1)
    
    # Override output directory if specified
    if output_dir:
        config.output.directory = str(output_dir)
    
    # Override output format if specified
    config.output.format = format
    
    try:
        # Determine if input is file or directory
        is_directory = folder or input_path.is_dir()
        
        if not is_directory and input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            console.print(f"[blue]Processing {input_path}...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # Step 1: Extract audio
                task1 = progress.add_task("Extracting audio...", total=None)
                extractor = AudioExtractor()
                if sample_duration > 0:
                    audio_path = extractor.extract_sample(input_path, sample_duration)
                else:
                    audio_path = extractor.extract_audio(input_path)
                progress.update(task1, completed=True)
                
                # Step 2: Transcribe audio
                task2 = progress.add_task("Transcribing audio...", total=None)
                transcriber = Transcriber(
                    api_key=config.openai_api_key,
                    model=config.transcription.model
                )
                transcript = transcriber.transcribe(audio_path)
                progress.update(task2, completed=True)
                
                # Step 3: Summarize transcript
                task3 = progress.add_task("Generating summary...", total=None)
                summarizer = Summarizer(
                    api_key=config.openai_api_key,
                    model=config.summarization.model
                )
                summary = summarizer.summarize(transcript)
                progress.update(task3, completed=True)
                
                # Step 4: Generate documentation
                task4 = progress.add_task("Creating documentation...", total=None)
                doc_generator = DocGenerator(
                    api_key=config.openai_api_key,
                    model=config.summarization.model
                )
                doc_path = doc_generator.generate_engineering_doc(summary, input_path)
                progress.update(task4, completed=True)
            
            console.print(f"[green]✓ Processing complete![/green]")
            console.print(f"[green]Documentation saved to: {doc_path}[/green]")
            
        elif is_directory:
            logger.info(f"Processing directory: {input_path}")
            console.print(f"[blue]Processing directory {input_path}...[/blue]")
            
            # Get all video files in directory
            video_files = get_video_files_from_folder(input_path)
            if not video_files:
                console.print("[yellow]No video files found in directory[/yellow]")
                return
            
            console.print(f"[blue]Found {len(video_files)} video files[/blue]")
            
            # Process each video file
            for i, video_file in enumerate(video_files, 1):
                console.print(f"\n[blue]Processing {i}/{len(video_files)}: {video_file.name}[/blue]")
                
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        # Extract audio
                        task1 = progress.add_task("Extracting audio...", total=None)
                        extractor = AudioExtractor()
                        if sample_duration > 0:
                            audio_path = extractor.extract_sample(video_file, sample_duration)
                        else:
                            audio_path = extractor.extract_audio(video_file)
                        progress.update(task1, completed=True)
                        
                        # Transcribe
                        task2 = progress.add_task("Transcribing...", total=None)
                        transcriber = Transcriber(
                            api_key=config.openai_api_key,
                            model=config.transcription.model
                        )
                        transcript = transcriber.transcribe(audio_path)
                        progress.update(task2, completed=True)
                        
                        # Summarize
                        task3 = progress.add_task("Summarizing...", total=None)
                        summarizer = Summarizer(
                            api_key=config.openai_api_key,
                            model=config.summarization.model
                        )
                        summary = summarizer.summarize(transcript)
                        progress.update(task3, completed=True)
                        
                        # Generate documentation
                        task4 = progress.add_task("Creating docs...", total=None)
                        doc_generator = DocGenerator(
                            api_key=config.openai_api_key,
                            model=config.summarization.model
                        )
                        doc_path = doc_generator.generate_engineering_doc(summary, video_file)
                        progress.update(task4, completed=True)
                    
                    console.print(f"[green]✓ {video_file.name} completed[/green]")
                    
                except Exception as e:
                    console.print(f"[red]✗ Failed to process {video_file.name}: {e}[/red]")
                    logger.error(f"Failed to process {video_file}: {e}")
                    continue
            
            console.print(f"\n[green]✓ Directory processing complete![/green]")
            
        else:
            raise click.BadParameter(f"Input path must be a file or directory: {input_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for extracted audio",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    help="Extract only a sample of specified duration (seconds)",
)
@pass_cli_context
def extract_audio(cli_ctx: CLIContext, video_path: Path, output_path: Optional[Path], duration: Optional[int]) -> None:
    """Extract audio from a video file."""
    logger = get_logger("cli.extract_audio")
    
    try:
        console.print(f"[blue]Extracting audio from {video_path}...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting audio...", total=None)
            
            extractor = AudioExtractor()
            if duration:
                audio_path = extractor.extract_sample(video_path, duration)
                console.print(f"[blue]Extracting {duration}-second sample...[/blue]")
            else:
                audio_path = extractor.extract_audio(video_path, output_path)
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Audio extracted successfully![/green]")
        console.print(f"[green]Audio saved to: {audio_path}[/green]")
        
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for extracted audio",
)
@click.option(
    "--duration",
    "-d",
    type=int,
    required=True,
    help="Extract only a sample of specified duration (seconds)",
)
@pass_cli_context
def extract_sample(cli_ctx: CLIContext, video_path: Path, output_path: Optional[Path], duration: int) -> None:
    """Extract a sample of audio from a video file."""
    logger = get_logger("cli.extract_sample")
    
    try:
        console.print(f"[blue]Extracting {duration}-second sample from {video_path}...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting audio sample...", total=None)
            
            extractor = AudioExtractor()
            audio_path = extractor.extract_sample(video_path, duration, output_path)
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Audio sample extracted successfully![/green]")
        console.print(f"[green]Audio saved to: {audio_path}[/green]")
        
    except Exception as e:
        logger.error(f"Audio sample extraction failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for transcript",
)
@pass_cli_context
def transcribe(cli_ctx: CLIContext, audio_path: Path, output_path: Optional[Path]) -> None:
    """Transcribe an audio file."""
    logger = get_logger("cli.transcribe")
    config = cli_ctx.config
    
    # Check if API key is available
    if not config.openai_api_key:
        console.print("[red]Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable, provide in config file, or use --api-key option.[/red]")
        sys.exit(1)
    
    try:
        console.print(f"[blue]Transcribing {audio_path}...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Transcribing...", total=None)
            
            transcriber = Transcriber(
                api_key=config.openai_api_key,
                model=config.transcription.model
            )
            transcript = transcriber.transcribe(audio_path)
            
            progress.update(task, completed=True)
        
        # Save transcript to file if output path specified
        if output_path:
            save_text_to_file(transcript, output_path)
            console.print(f"[green]✓ Transcript saved to: {output_path}[/green]")
        else:
            # Save to default location
            output_path = audio_path.with_suffix('.txt')
            save_text_to_file(transcript, output_path)
            console.print(f"[green]✓ Transcript saved to: {output_path}[/green]")
        
        console.print(f"[green]✓ Transcription complete![/green]")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("text", type=str)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for summary",
)
@pass_cli_context
def summarize(cli_ctx: CLIContext, text: str, output_path: Optional[Path]) -> None:
    """Summarize text content."""
    logger = get_logger("cli.summarize")
    config = cli_ctx.config
    
    # Check if API key is available
    if not config.openai_api_key:
        console.print("[red]Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable, provide in config file, or use --api-key option.[/red]")
        sys.exit(1)
    
    try:
        console.print(f"[blue]Summarizing text...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating summary...", total=None)
            
            summarizer = Summarizer(
                api_key=config.openai_api_key,
                model=config.summarization.model
            )
            summary = summarizer.summarize(text)
            
            progress.update(task, completed=True)
        
        # Save summary to file if output path specified
        if output_path:
            save_text_to_file(summary, output_path)
            console.print(f"[green]✓ Summary saved to: {output_path}[/green]")
        else:
            # Print to console
            console.print(f"\n[bold]Summary:[/bold]")
            console.print(summary)
        
        console.print(f"[green]✓ Summarization complete![/green]")
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("content", type=str)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for documentation",
)
@click.option(
    "--type",
    "-t",
    type=click.Choice(["engineering", "meeting", "action"]),
    default="engineering",
    help="Type of documentation to generate",
)
@pass_cli_context
def generate_doc(cli_ctx: CLIContext, content: str, output: Path, type: str) -> None:
    """Generate documentation from content."""
    logger = get_logger("cli.generate_doc")
    config = cli_ctx.config
    
    # Check if API key is available
    if not config.openai_api_key:
        console.print("[red]Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable, provide in config file, or use --api-key option.[/red]")
        sys.exit(1)
    
    try:
        console.print(f"[blue]Generating {type} documentation...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating documentation...", total=None)
            
            doc_generator = DocGenerator(
                api_key=config.openai_api_key,
                model=config.summarization.model
            )
            
            if type == "engineering":
                doc_path = doc_generator.generate_engineering_doc(content, output)
            elif type == "meeting":
                doc_path = doc_generator.generate_meeting_notes(content, output)
            elif type == "action":
                doc_path = doc_generator.generate_action_items(content, output)
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Documentation generated successfully![/green]")
        console.print(f"[green]Documentation saved to: {doc_path}[/green]")
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command()
@click.pass_context
def show(ctx: click.Context) -> None:
    """Show current configuration."""
    config = ctx.obj["config"]
    
    console.print("\n[bold]Current Configuration:[/bold]")
    console.print(f"OpenAI API Key: {'*' * 20 if config.openai_api_key else 'Not set'}")
    console.print(f"Output Directory: {config.output.directory}")
    console.print(f"Output Format: {config.output.format}")
    console.print(f"Log Level: {config.logging.level}")
    console.print(f"Transcription Model: {config.transcription.model}")
    console.print(f"Summarization Model: {config.summarization.model}")


@config.command()
@click.argument("key")
@click.argument("value")
@click.pass_context
def set(ctx: click.Context, key: str, value: str) -> None:
    """Set a configuration value."""
    # TODO: Implement configuration setting
    console.print(f"[yellow]Setting {key} = {value}[/yellow]")
    console.print("[yellow]Configuration setting not yet implemented[/yellow]")


@config.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate current configuration."""
    config = ctx.obj["config"]
    
    try:
        # Configuration is validated on load, so if we get here it's valid
        console.print("[green]✓ Configuration is valid[/green]")
        
        # Additional checks
        if not config.openai_api_key:
            console.print("[red]✗ OpenAI API key is not set[/red]")
        else:
            console.print("[green]✓ OpenAI API key is set[/green]")
            
    except Exception as e:
        console.print(f"[red]✗ Configuration validation failed: {e}[/red]")
        sys.exit(1)


@main.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"vid-to-doc version {__version__}")


# Create a standalone version command that doesn't require configuration
@click.command()
def version_standalone() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"vid-to-doc version {__version__}")


if __name__ == "__main__":
    main() 