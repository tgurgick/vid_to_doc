"""YouTube video downloading utilities.

⚠️ IMPORTANT DISCLAIMER ⚠️
This module is provided for EDUCATIONAL AND NON-COMMERCIAL TESTING PURPOSES ONLY.
Users are responsible for ensuring compliance with YouTube's Terms of Service 
and applicable copyright laws. Only download content you own or have permission to download.

For more information, see YouTube's Terms of Service: https://www.youtube.com/t/terms
"""

import os
import re
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse, parse_qs

import yt_dlp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..models.exceptions import YouTubeDownloadError

console = Console()


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL.
    
    Args:
        url: The URL to check
        
    Returns:
        True if it's a YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def extract_video_id(url: str) -> Optional[str]:
    """Extract the video ID from a YouTube URL.
    
    Args:
        url: The YouTube URL
        
    Returns:
        The video ID or None if not found
    """
    # Handle youtu.be URLs
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    
    # Handle youtube.com URLs
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[-1]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[-1]
    
    return None


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing invalid characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def download_youtube_video(
    url: str,
    output_dir: Optional[Union[str, Path]] = None,
    quality: str = "best",
    audio_only: bool = False,
    filename: Optional[str] = None
) -> Path:
    """Download a YouTube video.
    
    ⚠️ EDUCATIONAL USE ONLY: This function is for educational and non-commercial 
    testing purposes only. Ensure compliance with YouTube's Terms of Service and 
    copyright laws. Only download content you own or have permission to download.
    
    Args:
        url: The YouTube URL to download
        output_dir: Directory to save the video (default: current directory)
        quality: Video quality preference (best, worst, 720p, 480p, etc.)
        audio_only: If True, download only audio
        filename: Custom filename (without extension)
        
    Returns:
        Path to the downloaded file
        
    Raises:
        YouTubeDownloadError: If download fails
    """
    # Display educational use warning
    console.print("[yellow]⚠️  EDUCATIONAL USE ONLY: Ensure compliance with YouTube's Terms of Service and copyright laws.[/yellow]")
    
    if not is_youtube_url(url):
        raise YouTubeDownloadError(f"Invalid YouTube URL: {url}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    # Set quality preferences
    if audio_only:
        ydl_opts['format'] = 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio'
        ydl_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }]
    else:
        if quality == "best":
            ydl_opts['format'] = 'best[ext=mp4]/best[ext=webm]/best'
        elif quality == "worst":
            ydl_opts['format'] = 'worst[ext=mp4]/worst[ext=webm]/worst'
        else:
            # Try to match specific quality
            ydl_opts['format'] = f'best[height<={quality}]/best[ext=mp4]/best'
    
    # Use custom filename if provided
    if filename:
        sanitized_filename = sanitize_filename(filename)
        ydl_opts['outtmpl'] = str(output_dir / f'{sanitized_filename}.%(ext)s')
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Downloading video...", total=None)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')
                
                progress.update(task, description=f"Downloading: {video_title}")
                
                # Download the video
                ydl.download([url])
            
            progress.update(task, completed=True)
        
        # Find the downloaded file
        if filename:
            # Look for files with our custom filename
            pattern = f"{sanitized_filename}.*"
            downloaded_files = list(output_dir.glob(pattern))
        else:
            # Look for files with the video title
            sanitized_title = sanitize_filename(video_title)
            pattern = f"{sanitized_title}.*"
            downloaded_files = list(output_dir.glob(pattern))
        
        if not downloaded_files:
            # Fallback: look for recently created video files
            video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.wav', '.mp3', '.m4a']
            all_files = []
            for ext in video_extensions:
                all_files.extend(output_dir.glob(f"*{ext}"))
            
            if all_files:
                # Get the most recently created file
                downloaded_file = max(all_files, key=lambda f: f.stat().st_mtime)
            else:
                raise YouTubeDownloadError("Could not find downloaded file")
        else:
            downloaded_file = downloaded_files[0]
        
        console.print(f"[green]✓ Downloaded: {downloaded_file.name}[/green]")
        return downloaded_file
        
    except Exception as e:
        raise YouTubeDownloadError(f"Failed to download video: {e}")


def get_video_info(url: str) -> dict:
    """Get information about a YouTube video without downloading.
    
    ⚠️ EDUCATIONAL USE ONLY: This function is for educational and non-commercial 
    testing purposes only. Ensure compliance with YouTube's Terms of Service.
    
    Args:
        url: The YouTube URL
        
    Returns:
        Dictionary containing video information
        
    Raises:
        YouTubeDownloadError: If info extraction fails
    """
    if not is_youtube_url(url):
        raise YouTubeDownloadError(f"Invalid YouTube URL: {url}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown'),
                'description': info.get('description', '')[:200] + '...' if info.get('description', '') else '',
                'formats': [f.get('format_note', 'Unknown') for f in info.get('formats', [])[:5]]
            }
    except Exception as e:
        raise YouTubeDownloadError(f"Failed to get video info: {e}")


def list_available_formats(url: str) -> list:
    """List available formats for a YouTube video.
    
    ⚠️ EDUCATIONAL USE ONLY: This function is for educational and non-commercial 
    testing purposes only. Ensure compliance with YouTube's Terms of Service.
    
    Args:
        url: The YouTube URL
        
    Returns:
        List of available formats
        
    Raises:
        YouTubeDownloadError: If format listing fails
    """
    if not is_youtube_url(url):
        raise YouTubeDownloadError(f"Invalid YouTube URL: {url}")
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            formats = []
            
            for f in info.get('formats', []):
                format_info = {
                    'format_id': f.get('format_id', 'Unknown'),
                    'ext': f.get('ext', 'Unknown'),
                    'resolution': f.get('resolution', 'Unknown'),
                    'filesize': f.get('filesize', 'Unknown'),
                    'format_note': f.get('format_note', 'Unknown'),
                }
                formats.append(format_info)
            
            return formats
    except Exception as e:
        raise YouTubeDownloadError(f"Failed to list formats: {e}") 