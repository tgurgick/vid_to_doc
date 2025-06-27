import argparse
import os
from dotenv import load_dotenv
from moviepy import VideoFileClip
import openai
import time

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables.")

def get_video_files_from_folder(folder_path):
    """Return a list of video file paths in the given folder."""
    supported_exts = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv'}
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_exts
    ]

def ensure_output_folder(base_path):
    """Ensure an 'output' folder exists next to the given base path."""
    output_dir = os.path.join(os.path.dirname(base_path), 'output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_audio_from_video(video_path, output_audio_path=None):
    """Extract audio from a video file and save as a .wav file."""
    if output_audio_path is None:
        output_audio_path = os.path.splitext(video_path)[0] + ".wav"
    try:
        with VideoFileClip(video_path) as video:
            audio = video.audio
            if audio is None:
                print(f"No audio track found in {video_path}.")
                return None
            audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        print(f"Audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Failed to extract audio from {video_path}: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI's Whisper API, chunking if >25MB. Prints video length, model, token usage, and processing time. Adds retry logic for chunked API calls."""
    import math
    import soundfile as sf
    import time
    import traceback
    MAX_MB = 25
    MAX_BYTES = MAX_MB * 1024 * 1024
    MODEL_NAME = "gpt-4o-transcribe"
    file_size = os.path.getsize(audio_path)
    # Get audio duration
    try:
        import wave
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
    except Exception:
        duration = None
    start_time = time.time()
    if file_size <= MAX_BYTES:
        with open(audio_path, 'rb') as audio_file:
            transcript = openai.audio.transcriptions.create(
                file=audio_file,
                model=MODEL_NAME,
                response_format='text',
            )
        elapsed = time.time() - start_time
        print(f"Transcription complete for {audio_path}")
        if duration:
            print(f"Audio length: {duration:.2f} seconds")
        print(f"Model used: {MODEL_NAME}")
        print(f"Processing time: {elapsed:.2f} seconds")
        # Print token usage if available
        if hasattr(transcript, 'usage'):
            usage = transcript.usage
            print(f"Input tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Output tokens: {usage.get('completion_tokens', 'N/A')}")
        return transcript
    else:
        print(f"Audio file {audio_path} is larger than 25MB, chunking...")
        data, samplerate = sf.read(audio_path)
        total_samples = len(data)
        bytes_per_sample = data.itemsize if hasattr(data, 'itemsize') else 2
        max_samples_per_chunk = MAX_BYTES // bytes_per_sample
        chunk_count = math.ceil(total_samples / max_samples_per_chunk)
        transcripts = []
        for i in range(chunk_count):
            chunk_start_time = time.time()
            start = i * max_samples_per_chunk
            end = min((i + 1) * max_samples_per_chunk, total_samples)
            chunk_data = data[start:end]
            chunk_path = f"{os.path.splitext(audio_path)[0]}_chunk{i+1}.wav"
            sf.write(chunk_path, chunk_data, samplerate)
            # Retry logic for API call
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    with open(chunk_path, 'rb') as chunk_file:
                        chunk_transcript = openai.audio.transcriptions.create(
                            file=chunk_file,
                            model=MODEL_NAME,
                            response_format='text',
                        )
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"Error transcribing chunk {i+1}/{chunk_count} (attempt {attempt}): {e}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        print(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to transcribe chunk {i+1} after {max_retries} attempts. Skipping this chunk.")
                        chunk_transcript = None
                        traceback.print_exc()
            chunk_elapsed = time.time() - chunk_start_time
            print(f"Chunk {i+1}/{chunk_count}:")
            if duration:
                print(f"Chunk duration: {duration/chunk_count:.2f} seconds")
            print(f"Model used: {MODEL_NAME}")
            print(f"Processing time: {chunk_elapsed:.2f} seconds")
            if chunk_transcript is not None:
                if hasattr(chunk_transcript, 'usage'):
                    usage = chunk_transcript.usage
                    print(f"Input tokens: {usage.get('prompt_tokens', 'N/A')}")
                    print(f"Output tokens: {usage.get('completion_tokens', 'N/A')}")
                transcripts.append(chunk_transcript)
            else:
                print(f"Warning: Skipped chunk {i+1} due to repeated errors.")
            os.remove(chunk_path)
        total_elapsed = time.time() - start_time
        full_transcript = '\n'.join(str(t) for t in transcripts)
        print(f"Transcription complete for {audio_path} (chunked)")
        print(f"Total processing time: {total_elapsed:.2f} seconds")
        return full_transcript

def save_transcription(transcript_text, video_path):
    """Save the transcription text to a file in the output folder."""
    output_dir = ensure_output_folder(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, base + "_transcription.txt")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        print(f"Transcription saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to save transcription for {video_path}: {e}")
        return None

def summarize_text(text):
    """Summarize the given text using OpenAI's GPT API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Summarize the following transcription into concise knowledge notes."},
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )
        summary = response.choices[0].message.content.strip()
        print("Summarization complete.")
        return summary
    except Exception as e:
        print(f"Failed to summarize transcription: {e}")
        return None

def save_summary(summary_text, video_path):
    """Save the summary text to a file in the output folder."""
    output_dir = ensure_output_folder(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, base + "_summary.txt")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"Summary saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to save summary for video {video_path}: {e}")
        return None

def generate_engineering_doc(summary_text, video_path):
    """Generate an engineering-focused documentation summary from the content summary and save as a Markdown file."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert technical writer. Given the following summary of a video transcription, generate a concise engineering documentation section for future engineering teams. Focus on technical insights, process, and key takeaways. Use Markdown formatting."},
                {"role": "user", "content": summary_text}
            ],
            max_tokens=700
        )
        doc = response.choices[0].message.content.strip()
        output_dir = ensure_output_folder(video_path)
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, base + "_engineering_doc.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f"Engineering documentation saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to generate engineering documentation for {video_path}: {e}")
        return None

def process_videos_in_folder(folder_path):
    video_files = get_video_files_from_folder(folder_path)
    if not video_files:
        print(f"No supported video files found in {folder_path}.")
        return
    for video_path in video_files:
        print(f"Processing: {video_path}")
        audio_path = extract_audio_from_video(video_path)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            if transcript:
                save_transcription(transcript, video_path)
                summary = summarize_text(transcript)
                if summary:
                    save_summary(summary, video_path)
                    generate_engineering_doc(summary, video_path)

def extract_sample_from_video(video_path, sample_duration=15, output_audio_path=None):
    """Extract a sample audio clip from the middle of the video."""
    try:
        with VideoFileClip(video_path) as video:
            if video.duration < sample_duration:
                print(f"Video too short for sampling: {video_path}")
                return None
            start = (video.duration - sample_duration) / 2
            end = start + sample_duration
            # Use 'subclipped' instead of 'subclip' for MoviePy v2+
            sample_clip = video.subclipped(start, end)
            audio = sample_clip.audio
            if audio is None:
                print(f"No audio track found in {video_path}.")
                return None
            if output_audio_path is None:
                base = os.path.splitext(video_path)[0]
                output_audio_path = base + f"_sample_{int(sample_duration)}s.wav"
            audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        print(f"Sample audio extracted to {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Failed to extract sample audio from {video_path}: {e}")
        return None

def process_sample_from_video(video_path, sample_duration=15, full=False):
    print(f"Processing sample from: {video_path}")
    ext = os.path.splitext(video_path)[1].lower()
    audio_exts = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mpga', '.webm'}
    if ext in audio_exts:
        # Directly transcribe audio files
        transcript = transcribe_audio(video_path)
        if transcript:
            print("Sample transcription:")
            print(transcript)
            summary = summarize_text(transcript)
            if summary:
                print("Sample summary:")
                print(summary)
            # Save outputs to output folder
            save_transcription(transcript, video_path)
            if summary:
                save_summary(summary, video_path)
                generate_engineering_doc(summary, video_path)
    else:
        if full:
            # Extract full audio from video and transcribe
            audio_path = extract_audio_from_video(video_path)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                if transcript:
                    print("Full transcription:")
                    print(transcript)
                    summary = summarize_text(transcript)
                    if summary:
                        print("Full summary:")
                        print(summary)
                    # Save outputs to output folder
                    save_transcription(transcript, video_path)
                    if summary:
                        save_summary(summary, video_path)
                        generate_engineering_doc(summary, video_path)
        else:
            # Extract sample from video files
            sample_audio_path = extract_sample_from_video(video_path, sample_duration)
            if sample_audio_path:
                transcript = transcribe_audio(sample_audio_path)
                if transcript:
                    print("Sample transcription:")
                    print(transcript)
                    summary = summarize_text(transcript)
                    if summary:
                        print("Sample summary:")
                        print(summary)
                    # Save outputs to output folder
                    save_transcription(transcript, sample_audio_path)
                    if summary:
                        save_summary(summary, sample_audio_path)
                        generate_engineering_doc(summary, sample_audio_path)

# Placeholder for main workflow
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and summarize videos in a folder or test a sample from a video.")
    parser.add_argument('--folder', type=str, help='Path to folder containing video files')
    parser.add_argument('--sample', type=str, help='Path to a video file to sample')
    parser.add_argument('--duration', type=int, default=15, help='Sample duration in seconds (default: 15)')
    parser.add_argument('--full', action='store_true', help='Process the full video instead of a sample')
    args = parser.parse_args()
    if args.sample:
        process_sample_from_video(args.sample, args.duration, args.full)
    elif args.folder:
        process_videos_in_folder(args.folder)
    else:
        parser.print_help()
