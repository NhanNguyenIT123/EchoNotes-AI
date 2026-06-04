# -*- coding: utf-8 -*-
"""
EchoNotes AI - Reusable Dataset Factory Utility
This module provides a programmatic, cross-platform dataset extraction pipeline.
Slices raw audio tracks into individual WAV chunks, applies glossary corrections,
generates metadata.csv, and packages them into a ZIP file.
"""

import os
import json
import subprocess
import zipfile
import re
import shutil
import imageio_ffmpeg
from pathlib import Path
from typing import List, Dict, Any, Callable
from app.utils.corrector import corrector

def get_ffmpeg_exe() -> str:
    """Returns the bundled static FFmpeg executable path."""
    return imageio_ffmpeg.get_ffmpeg_exe()

def generate_whisper_dataset(
    video_path: Path,
    segments: List[Dict[str, Any]],
    output_zip_path: Path,
    progress_callback: Callable[[float], None] = None
) -> int:
    """
    Programmatically slices video/audio into standard Hugging Face AudioFolder format
    (metadata.csv + aligned WAV clips) and packages it into a ZIP file.
    
    Returns the total number of successfully exported segments.
    """
    video_path = Path(video_path)
    output_zip_path = Path(output_zip_path)
    
    # Create temporary folders inside data directory
    temp_dir = video_path.parent / "temp_dataset_factory"
    audio_output_dir = temp_dir / "audio"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_wav_path = temp_dir / "full_audio_track.wav"
    ffmpeg_exe = get_ffmpeg_exe()
    
    # Step 1: Extract full audio
    if progress_callback: progress_callback(0.1)
    extract_cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(temp_wav_path)
    ]
    subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Step 2: Slice audio segments and generate metadata csv
    metadata = []
    metadata.append("file_name,sentence")
    
    valid_count = 0
    total_segs = len(segments)
    
    for idx, seg in enumerate(segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        duration = end - start
        raw_text = seg.get("text", "").strip()
        
        # Skip noise-only or extremely short segments
        if duration < 1.5 or not raw_text or raw_text == "[Tạp âm]":
            continue
            
        # Clean text using the master Vinglish spelling corrector
        cleaned_text = re.sub(r'\[[^\]]+\]', '', raw_text)
        cleaned_text = corrector.correct_text(cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        if not cleaned_text or len(cleaned_text.split()) < 3:
            continue
            
        # Define output chunk file path
        file_name = f"audio/chunk_{valid_count + 1:04d}.wav"
        output_file_path = temp_dir / file_name
        
        # Slice segment instantly using FFmpeg copy (no re-encoding = lightning fast!)
        slice_cmd = [
            ffmpeg_exe, "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", str(temp_wav_path),
            "-acodec", "copy",
            str(output_file_path)
        ]
        
        try:
            subprocess.run(slice_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            safe_text = cleaned_text.replace('"', '""')
            metadata.append(f"{file_name},\"{safe_text}\"")
            valid_count += 1
        except Exception:
            continue
            
        # Report progress
        if progress_callback and total_segs > 0:
            # Map remaining progress between 0.2 and 0.8
            progress_callback(0.2 + (idx / total_segs) * 0.6)
            
    # Write metadata.csv
    if progress_callback: progress_callback(0.85)
    with open(temp_dir / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
        
    # Step 3: Package into ZIP
    if progress_callback: progress_callback(0.9)
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                # Skip the raw full temporary wav in the ZIP package
                if file_path == temp_wav_path:
                    continue
                arcname = file_path.relative_to(temp_dir)
                zip_file.write(file_path, arcname)
                
    # Clean up temp files
    if progress_callback: progress_callback(0.98)
    shutil.rmtree(temp_dir)
    
    if progress_callback: progress_callback(1.0)
    return valid_count
