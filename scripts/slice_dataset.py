# -*- coding: utf-8 -*-
"""
EchoNotes AI - Automated Dataset Slicing & Preparation Pipeline
This script automatically slices a 2.5-hour lecture video/audio into clean 10-30s audio clips
using timestamps from the enriched segments cache, applies Vinglish spelling corrections
to the text labels, writes metadata.csv, and packages everything into a ready-to-run Colab zip file!

Usage:
    python scripts/slice_dataset.py
"""

import os
import json
import subprocess
import zipfile
import re
import sys
import imageio_ffmpeg
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "Meeting in General -20260515_085012-Bản ghi cuộc họp.mp4"
CACHE_PATH = BASE_DIR / "data" / "processed" / "transcripts" / "enriched_segments_cache.json"
OUTPUT_DIR = BASE_DIR / "lecture_dataset"
ZIP_PATH = BASE_DIR / "lecture_dataset.zip"
TEMP_WAV = BASE_DIR / "temp_audio_full.wav"

# Add parent to path to import corrector
sys.path.append(str(BASE_DIR))
from app.utils.corrector import corrector

def get_ffmpeg_exe() -> str:
    """Returns the bundled static FFmpeg executable path."""
    return imageio_ffmpeg.get_ffmpeg_exe()

def check_requirements():
    """Checks if video and cache exist."""
    if not VIDEO_PATH.exists():
        print(f"[Loi] Khong tim thay file video tai: {VIDEO_PATH}")
        return False
    if not CACHE_PATH.exists():
        print(f"[Loi] Khong tim thay cache phan doan tai: {CACHE_PATH}")
        return False
    return True

def extract_full_audio():
    """Extracts raw mono 16kHz audio from the video using FFmpeg (takes ~10 seconds)."""
    print("[1/6] Dang trich xuat toan bo am thanh tu video 2.5 gio (mono, 16kHz)...")
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", str(VIDEO_PATH),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(TEMP_WAV)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] Da trich xuat xong am thanh day du!")
        return True
    except Exception as e:
        print(f"[Loi] Loi khi trich xuat am thanh bang FFmpeg: {e}")
        return False

def clean_label(text: str) -> str:
    """Cleans transcript text: removes noise tags and applies technical glossary corrections."""
    if not text:
        return ""
    # Remove tape noise tags [Tạp âm], [Cười], etc.
    cleaned = re.sub(r'\[[^\]]+\]', '', text)
    # Apply master Vinglish glossary replacements (plóc -> block, ai nốt -> inode...)
    cleaned = corrector.correct_text(cleaned)
    # Clean up excess spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def slice_audio_segments():
    """Slices the full audio file into individual aligned clips and writes metadata.csv."""
    print("[2/6] Dang tai cache phan doan va tinh toan thoi gian...")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_output_dir = OUTPUT_DIR / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    metadata.append("file_name,sentence")
    
    print(f"[3/6] Bat dau cat am thanh bang FFmpeg (Tong so {len(segments)} phan doan)...")
    valid_count = 0
    ffmpeg_bin = get_ffmpeg_exe()
    
    for idx, seg in enumerate(segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = end - start
        raw_text = seg.get("text", "").strip()
        
        # Skip noise-only or extremely short segments (< 1.5 seconds)
        if duration < 1.5 or not raw_text or raw_text == "[Tạp âm]":
            continue
            
        clean_text = clean_label(raw_text)
        if not clean_text or len(clean_text.split()) < 3: # Skip segments with < 3 words
            continue
            
        # File name inside the dataset folder
        file_name = f"audio/chunk_{valid_count + 1:04d}.wav"
        output_file_path = OUTPUT_DIR / file_name
        
        # Slice segment instantly using FFmpeg copy (no re-encoding = lightning fast!)
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(TEMP_WAV),
            "-acodec", "copy",
            str(output_file_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Add to metadata CSV (double quote sentence to handle commas correctly)
            safe_text = clean_text.replace('"', '""')
            metadata.append(f"{file_name},\"{safe_text}\"")
            valid_count += 1
        except Exception:
            # Skip if a segment slice fails
            continue
            
        if (idx + 1) % 50 == 0:
            print(f"   * Da xu ly {idx + 1}/{len(segments)} phan doan (Da xuat {valid_count} file)...")
            
    # Write metadata.csv
    print("[4/6] Ghi tap tin chi muc metadata.csv...")
    with open(OUTPUT_DIR / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
        
    print(f"[OK] Da cat va gan nhan thanh cong {valid_count} phan doan bai giang!")
    return valid_count

def package_to_zip():
    """Zips the sliced lecture_dataset into a single ready-to-run zip file."""
    print(f"[5/6] Dang nen toan bo tap du lieu thanh '{ZIP_PATH.name}'...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(OUTPUT_DIR):
                for file in files:
                    file_path = Path(root) / file
                    # Store relative path inside zip
                    arcname = file_path.relative_to(OUTPUT_DIR)
                    zip_file.write(file_path, arcname)
        print(f"[OK] Da nen thanh cong! File luu tai: {ZIP_PATH}")
        return True
    except Exception as e:
        print(f"[Loi] Loi khi nen file zip: {e}")
        return False

def cleanup():
    """Cleans up temporary WAV files to save space."""
    print("[6/6] Dang don dep cac tep tin tam thoi...")
    if TEMP_WAV.exists():
        os.remove(TEMP_WAV)
    print("[OK] Da don dep xong!")

def main():
    if not check_requirements():
        return
        
    print("=" * 70)
    print("BAT DAU TIEN TRINH TU DONG CAT VA GAN NHAN DU LIEU BAI GIANG 2.5 GIO")
    print("=" * 70)
    
    if extract_full_audio():
        count = slice_audio_segments()
        if count > 0:
            if package_to_zip():
                print("\n" + "=" * 70)
                print("[THANH CONG] DA TU DONG TAO XONG BO DU LIEU HUAN LUYEN!")
                print(f"File ZIP luu tai: {ZIP_PATH}")
                print("Ban chi can keo tha file 'lecture_dataset.zip' nay len Google Colab!")
                print("=" * 70)
        cleanup()

if __name__ == "__main__":
    main()
