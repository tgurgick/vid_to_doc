"""Text processing utilities for the vid_to_doc package."""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import timedelta

from ..models.exceptions import TextProcessingError


def clean_transcript(text: str) -> str:
    """Clean and normalize transcript text.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript text
        
    Raises:
        TextProcessingError: If cleaning fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcription artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
        text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical content
        text = re.sub(r'<.*?>', '', text)    # Remove angle bracket content
        
        # Clean up punctuation
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
        text = re.sub(r'!{2,}', '!', text)   # Multiple exclamation marks to single
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks to single
        
        # Fix common transcription errors
        text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)  # Fix spaced initials
        text = re.sub(r'\b([a-z])\s+([A-Z])\b', r'\1. \2', text)  # Add periods after initials
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
        
    except Exception as e:
        raise TextProcessingError(f"Failed to clean transcript: {e}")


def format_timestamp(seconds: float, format_type: str = "standard") -> str:
    """Format seconds into a human-readable timestamp.
    
    Args:
        seconds: Time in seconds
        format_type: Format type ('standard', 'compact', 'detailed')
        
    Returns:
        Formatted timestamp string
        
    Raises:
        TextProcessingError: If formatting fails
    """
    if not isinstance(seconds, (int, float)) or seconds < 0:
        raise TextProcessingError("Seconds must be a non-negative number")
    
    try:
        # Convert to timedelta for easier formatting
        delta = timedelta(seconds=seconds)
        
        if format_type == "standard":
            # Format: HH:MM:SS
            total_seconds = int(seconds)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        elif format_type == "compact":
            # Format: MM:SS (for durations under 1 hour)
            if seconds < 3600:
                minutes = int(seconds) // 60
                secs = int(seconds) % 60
                return f"{minutes:02d}:{secs:02d}"
            else:
                return format_timestamp(seconds, "standard")
        
        elif format_type == "detailed":
            # Format: HH:MM:SS.mmm
            total_seconds = int(seconds)
            milliseconds = int((seconds - total_seconds) * 1000)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        
        else:
            raise TextProcessingError(f"Unknown format type: {format_type}")
            
    except Exception as e:
        raise TextProcessingError(f"Failed to format timestamp: {e}")


def extract_key_points(text: str, max_points: int = 10) -> List[str]:
    """Extract key points from text using simple heuristics.
    
    Args:
        text: Text to extract key points from
        max_points: Maximum number of key points to extract
        
    Returns:
        List of key points
        
    Raises:
        TextProcessingError: If extraction fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    try:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences based on various criteria
        scored_sentences = []
        
        for sentence in sentences:
            score = 0
            
            # Longer sentences get higher scores (up to a point)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 2
            elif 5 <= length < 10:
                score += 1
            
            # Sentences with numbers get higher scores
            if re.search(r'\d+', sentence):
                score += 1
            
            # Sentences with technical terms get higher scores
            technical_terms = ['api', 'function', 'method', 'class', 'interface', 'database', 
                             'server', 'client', 'protocol', 'algorithm', 'framework']
            if any(term in sentence.lower() for term in technical_terms):
                score += 2
            
            # Sentences with action words get higher scores
            action_words = ['implement', 'create', 'build', 'develop', 'design', 'optimize',
                           'deploy', 'configure', 'test', 'debug', 'fix', 'update']
            if any(word in sentence.lower() for word in action_words):
                score += 1
            
            # Sentences that start with capital letters (likely proper nouns)
            if re.match(r'^[A-Z][a-z]', sentence):
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        key_points = [sentence for sentence, _ in scored_sentences[:max_points]]
        
        return key_points
        
    except Exception as e:
        raise TextProcessingError(f"Failed to extract key points: {e}")


def extract_action_items(text: str) -> List[Dict[str, str]]:
    """Extract action items from text.
    
    Args:
        text: Text to extract action items from
        
    Returns:
        List of action items with 'task', 'assignee', and 'deadline' fields
        
    Raises:
        TextProcessingError: If extraction fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    try:
        action_items = []
        
        # Common action item patterns
        patterns = [
            r'(?:need to|should|must|will|going to)\s+([^.!?]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:will|should|needs to)\s+([^.!?]+)',
            r'(?:TODO|TASK|ACTION):\s*([^.!?]+)',
            r'([^.!?]*(?:by|before|until)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[^.!?]*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                task = match.group(1).strip()
                
                # Try to extract assignee and deadline
                assignee = None
                deadline = None
                
                # Look for assignee patterns
                assignee_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:will|should|needs to)', task)
                if assignee_match:
                    assignee = assignee_match.group(1)
                
                # Look for deadline patterns
                deadline_match = re.search(r'(?:by|before|until)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2})', task)
                if deadline_match:
                    deadline = deadline_match.group(1)
                
                action_items.append({
                    'task': task,
                    'assignee': assignee,
                    'deadline': deadline
                })
        
        return action_items
        
    except Exception as e:
        raise TextProcessingError(f"Failed to extract action items: {e}")


def extract_speakers(text: str) -> List[str]:
    """Extract speaker names from transcript text.
    
    Args:
        text: Transcript text
        
    Returns:
        List of unique speaker names
        
    Raises:
        TextProcessingError: If extraction fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    try:
        # Common speaker patterns
        patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):',  # Speaker: format
            r'\[([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\]',  # [Speaker] format
            r'\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)',  # (Speaker) format
        ]
        
        speakers = set()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                speaker = match.group(1).strip()
                if len(speaker.split()) <= 3:  # Reasonable name length
                    speakers.add(speaker)
        
        return sorted(list(speakers))
        
    except Exception as e:
        raise TextProcessingError(f"Failed to extract speakers: {e}")


def split_text_into_chunks(text: str, max_chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into chunks for processing.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
        
    Raises:
        TextProcessingError: If splitting fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    if max_chunk_size <= 0:
        raise TextProcessingError("Max chunk size must be positive")
    
    if overlap < 0:
        raise TextProcessingError("Overlap must be non-negative")
    
    try:
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_chunk_size * 0.7:  # Only break if we find a good boundary
                    end = sentence_end + 1
                else:
                    # Look for other natural break points
                    for break_char in ['\n', ';', ':', '!', '?']:
                        break_pos = text.rfind(break_char, start, end)
                        if break_pos > start + max_chunk_size * 0.7:
                            end = break_pos + 1
                            break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
        
    except Exception as e:
        raise TextProcessingError(f"Failed to split text into chunks: {e}")


def calculate_readability_score(text: str) -> Dict[str, float]:
    """Calculate readability metrics for text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with readability metrics
        
    Raises:
        TextProcessingError: If calculation fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    try:
        # Clean the text
        clean_text = clean_transcript(text)
        
        # Count sentences
        sentences = re.split(r'[.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        num_sentences = len(sentences)
        
        # Count words
        words = clean_text.split()
        num_words = len(words)
        
        # Count syllables (approximate)
        num_syllables = 0
        for word in words:
            word = word.lower()
            if word.endswith('e'):
                word = word[:-1]
            num_syllables += len(re.findall(r'[aeiouy]+', word))
            if num_syllables == 0:
                num_syllables = 1
        
        # Calculate metrics
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0
        avg_syllables_per_word = num_syllables / num_words if num_words > 0 else 0
        
        # Flesch Reading Ease Score
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        flesch_score = max(0, min(100, flesch_score))  # Clamp between 0 and 100
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        fk_grade = max(0, fk_grade)  # Don't go below 0
        
        return {
            'flesch_reading_ease': round(flesch_score, 2),
            'flesch_kincaid_grade': round(fk_grade, 1),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_syllables_per_word': round(avg_syllables_per_word, 2),
            'num_sentences': num_sentences,
            'num_words': num_words,
            'num_syllables': num_syllables
        }
        
    except Exception as e:
        raise TextProcessingError(f"Failed to calculate readability score: {e}")


def save_text_analysis(text: str, output_path: Path) -> Path:
    """Save comprehensive text analysis to a file.
    
    Args:
        text: Text to analyze
        output_path: Path to save the analysis
        
    Returns:
        Path to the saved analysis file
        
    Raises:
        TextProcessingError: If analysis fails
    """
    if not isinstance(text, str):
        raise TextProcessingError("Input must be a string")
    
    output_path = Path(output_path)
    
    try:
        # Perform analysis
        cleaned_text = clean_transcript(text)
        key_points = extract_key_points(cleaned_text)
        action_items = extract_action_items(cleaned_text)
        speakers = extract_speakers(cleaned_text)
        readability = calculate_readability_score(cleaned_text)
        
        # Generate analysis report
        report = f"""# Text Analysis Report

## Basic Statistics
- **Original length**: {len(text)} characters
- **Cleaned length**: {len(cleaned_text)} characters
- **Sentences**: {readability['num_sentences']}
- **Words**: {readability['num_words']}
- **Syllables**: {readability['num_syllables']}

## Readability Metrics
- **Flesch Reading Ease**: {readability['flesch_reading_ease']} (0-100, higher is easier)
- **Flesch-Kincaid Grade Level**: {readability['flesch_kincaid_grade']} (approximate grade level)
- **Average sentence length**: {readability['avg_sentence_length']} words
- **Average syllables per word**: {readability['avg_syllables_per_word']}

## Key Points
"""
        
        for i, point in enumerate(key_points, 1):
            report += f"{i}. {point}\n"
        
        report += "\n## Action Items\n"
        for i, item in enumerate(action_items, 1):
            report += f"{i}. **Task**: {item['task']}\n"
            if item['assignee']:
                report += f"   **Assignee**: {item['assignee']}\n"
            if item['deadline']:
                report += f"   **Deadline**: {item['deadline']}\n"
            report += "\n"
        
        if speakers:
            report += f"## Speakers Identified\n"
            for speaker in speakers:
                report += f"- {speaker}\n"
        
        # Save the report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return output_path
        
    except Exception as e:
        raise TextProcessingError(f"Failed to save text analysis: {e}") 