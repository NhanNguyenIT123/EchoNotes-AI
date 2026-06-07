import os
import sys
from pathlib import Path
import torch
from app.env import load_local_env

load_local_env()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Windows DLL injection for local PIP-installed CUDA binaries
if os.name == 'nt':
    venv_base = BASE_DIR / "venv"
    if venv_base.exists():
        # Support both CUDA 11 and CUDA 12 library locations
        cuda_dirs = [
            venv_base / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
            venv_base / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
            venv_base / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin"
        ]
        for cdir in cuda_dirs:
            if cdir.exists():
                try:
                    os.add_dll_directory(str(cdir))
                    # Inject directly into PATH for legacy Win32 LoadLibrary support
                    os.environ["PATH"] = str(cdir) + os.pathsep + os.environ["PATH"]
                except Exception:
                    pass

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"
FRAMES_DIR = PROCESSED_DIR / "frames"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Create directories if they do not exist
for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, AUDIO_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware Acceleration Configuration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
CPU_THREADS = 4

# Whisper STT Settings
# "small" is the minimum practical default for low-bitrate Vietnamese lectures.
WHISPER_MODEL_DEFAULT = "small"
WHISPER_COMPUTE_TYPE = "float16" if CUDA_AVAILABLE else "int8"

# EasyOCR Settings
OCR_DEVICE = CUDA_AVAILABLE  # True for GPU if PyTorch supports it, False for CPU

# Acoustic Analysis Thresholds
PITCH_HIGH_PERCENTILE = 90      # Trigger emphasis when pitch > 90th percentile
VOLUME_HIGH_PERCENTILE = 90     # Trigger emphasis when RMS volume > 90th percentile
SPEECH_RATE_SLOW_THRESHOLD = 2.2 # Words per second below this are flagged as slow & deliberate

# Slide Detection Settings
SSIM_THRESHOLD = 0.94           # Slide transition occurs when SSIM drops below this
FRAME_CHECK_INTERVAL = 2.0      # Sample one frame every 2.0 seconds (very lightweight)

# Ollama API Configuration
OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
OLLAMA_FALLBACK_MODEL = "llama3:latest"
