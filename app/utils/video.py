import os
import subprocess
import imageio_ffmpeg
from pathlib import Path
from app.config import AUDIO_DIR

def get_local_ffmpeg() -> str:
    """Returns the absolute path to the locally installed static FFmpeg binary."""
    return imageio_ffmpeg.get_ffmpeg_exe()

def extract_audio_from_video(video_path: Path) -> Path:
    """
    Extracts the audio track from a video file and saves it as a 16kHz mono WAV file.
    This format is optimized for speech recognition (Whisper) and acoustic feature analysis (Librosa).
    Uses a highly memory-efficient FFmpeg subprocess to prevent crashes on large video files.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    audio_output_name = f"{video_path.stem}_extracted.wav"
    audio_output_path = AUDIO_DIR / audio_output_name
    
    # FFmpeg command to extract 16kHz mono audio.
    # Teams recordings in this project are very low bitrate, so we normalize speech
    # and remove rumble/high-frequency noise before ASR sees the waveform.
    # -y: overwrite output
    # -i: input file
    # -vn: disable video recording
    # -acodec pcm_s16le: 16-bit PCM WAV
    # -ar 16000: 16kHz sampling rate
    # -ac 1: mono channel
    cmd = [
        get_local_ffmpeg(), "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=7800,dynaudnorm=f=150:g=15",
        str(audio_output_path)
    ]
    
    try:
        # Run FFmpeg process silently
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        try:
            print(f"Successfully extracted audio to: {audio_output_path}")
        except Exception:
            print("[OK] Successfully extracted audio track.")
        return audio_output_path
    except subprocess.CalledProcessError as e:
        try:
            print(f"Error during audio extraction: {e.stderr}")
        except Exception:
            print("[-] Error during audio extraction.")
        # Try a basic fallback if PCM WAV fails (e.g., to MP3)
        raise RuntimeError("FFmpeg audio extraction failed.")

def slice_video_segment(video_path: Path, start_seconds: float, end_seconds: float, output_path: Path) -> Path:
    """
    Cuts a segment from a video file from start_seconds to end_seconds without re-encoding.
    Very fast and memory efficient.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    duration = end_seconds - start_seconds
    
    # FFmpeg fast-seek clip slicing:
    # -ss: start time
    # -t: duration
    # -c copy: stream copy without re-encoding (extremely fast!)
    cmd = [
        get_local_ffmpeg(), "-y",
        "-ss", str(start_seconds),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",
        str(output_path)
    ]
    
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg video slicing failed: {e.stderr}")
