# Transcription Project

This project allows you to:
- Input video files (local, with future plans for Google Drive integration)
- Transcribe videos using OpenAI APIs (Whisper, with chunking for large files)
- Summarize transcriptions into concise knowledge documents
- Automatically generate engineering documentation from summaries

## Setup
1. Ensure you have Python 3.9+ and a virtual environment activated.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```sh
   export OPENAI_API_KEY=your-key-here
   ```
4. (Optional) Set up Google Drive API credentials for video access.

## Usage

### Transcribe and Summarize All Videos in a Folder
1. Place your video files (supported formats: .mp4, .mov, .avi, .mkv, .flv, .wmv) in a folder (e.g., `videos/`).
2. Ensure your OpenAI API key is set as an environment variable:
   ```sh
   export OPENAI_API_KEY=your-key-here
   ```
3. Run the script with the folder path:
   ```sh
   python main.py --folder videos
   ```

### Transcribe and Summarize a Sample or Full Video
To process a sample (e.g., 15 seconds from the middle) of a video:
```sh
python main.py --sample videos/your_video.mp4 --duration 15
```
To process the full video (not just a sample):
```sh
python main.py --sample videos/your_video.mp4 --full
```

## Output
For each video or sample, the script will:
- Extract the audio track and save it as a `.wav` file.
- Transcribe the audio using OpenAI Whisper and save the transcription as a `.txt` file in the `output/` directory.
- Summarize the transcription using OpenAI GPT and save the summary as a `.txt` file in the `output/` directory.
- Generate an engineering documentation summary (Markdown) for future engineering teams and save it as a `.md` file in the `output/` directory.

**Output files:**
- `<video_name>_transcription.txt`: Full transcription of the video or sample.
- `<video_name>_summary.txt`: Concise summary/knowledge notes from the transcription.
- `<video_name>_engineering_doc.md`: Engineering-focused documentation generated from the summary.

## CLI Options
- `--folder <folder>`: Process all videos in a folder.
- `--sample <file>`: Process a single video file (sample or full).
- `--duration <seconds>`: Sample duration in seconds (default: 15, used with `--sample`).
- `--full`: Process the full video instead of a sample (used with `--sample`).

## Roadmap
- [x] Local video/audio file support
- [x] Audio extraction and chunked transcription
- [x] Summarization and engineering doc generation
- [x] CLI for batch and sample processing
- [ ] Google Drive video link ingestion
- [ ] Enhanced error handling/logging
- [ ] Advanced CLI options (output dir, verbosity, etc.)
- [ ] More automated tests

---

*This README will be updated as features are implemented.*
