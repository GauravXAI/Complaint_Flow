import os
import tempfile
from typing import Optional


def _whisper_available() -> bool:
    try:
        import whisper  # noqa
        return True
    except ImportError:
        return False


def _moviepy_available() -> bool:
    try:
        from moviepy.editor import VideoFileClip  # noqa
        return True
    except ImportError:
        return False


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio file using local Whisper model.
    Supports: mp3, wav, m4a, ogg, flac
    """
    if not _whisper_available():
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def transcribe_video(video_path: str, model_size: str = "base") -> str:
    """
    Extract audio from video and transcribe.
    Supports: mp4, avi, mov, mkv
    """
    if not _moviepy_available():
        raise RuntimeError(
            "moviepy is not installed. Run: pip install moviepy"
        )
    if not _whisper_available():
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    from moviepy.editor import VideoFileClip

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_audio_path = tmp.name

    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(tmp_audio_path, logger=None)
        clip.close()
        text = transcribe_audio(tmp_audio_path, model_size)
    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

    return text


def is_audio_file(filename: str) -> bool:
    return filename.lower().split(".")[-1] in {"mp3", "wav", "m4a", "ogg", "flac", "opus"}


def is_video_file(filename: str) -> bool:
    return filename.lower().split(".")[-1] in {"mp4", "avi", "mov", "mkv", "webm"}


def transcribe_uploaded_file(file_path: str, filename: str, model_size: str = "base") -> str:
    """
    Unified entry point: auto-detects audio vs video.
    Returns transcribed text.
    """
    if is_audio_file(filename):
        return transcribe_audio(file_path, model_size)
    elif is_video_file(filename):
        return transcribe_video(file_path, model_size)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
