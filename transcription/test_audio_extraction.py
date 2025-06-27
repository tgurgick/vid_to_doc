import os
from main import extract_audio_from_video

def test_extract_audio_from_video():
    # Use a small sample video file for testing
    sample_video = 'sample.mp4'
    sample_audio = 'sample.wav'
    # Create a dummy video file if it doesn't exist (1s silent video)
    if not os.path.exists(sample_video):
        from moviepy.editor import ColorClip
        clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=1)
        clip = clip.set_audio(None)
        clip.write_videofile(sample_video, fps=24)
    # Test extraction (should print a warning about no audio)
    result = extract_audio_from_video(sample_video, sample_audio)
    assert result is None, "Should return None for video with no audio track"
    # Clean up
    if os.path.exists(sample_video):
        os.remove(sample_video)
    if os.path.exists(sample_audio):
        os.remove(sample_audio)
    print("test_extract_audio_from_video passed.")

if __name__ == "__main__":
    test_extract_audio_from_video()
