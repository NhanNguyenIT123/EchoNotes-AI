# -*- coding: utf-8 -*-
"""
Import a Microsoft Teams transcript and turn it into EchoNotes/Whisper data.

Supported inputs:
- .vtt with timestamp cues
- .srt with timestamp cues
- .txt with Teams-style timestamp lines when available

Typical usage:
  venv\\Scripts\\python.exe scripts\\import_teams_transcript.py ^
    --transcript "path\\to\\teams_transcript.vtt" ^
    --video "data\\raw\\Meeting.mp4" ^
    --dataset-output teams_corrected_dataset ^
    --cache-output data\\processed\\transcripts\\teams_segments_cache.json
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import imageio_ffmpeg


TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?)"
)


def parse_time(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timestamp: {value}")


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"^\s*[-–]?\s*[^:\n]{1,60}:\s+", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt_or_srt(path: Path) -> List[Dict]:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    segments = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        match = TIMESTAMP_RE.search(line)
        if not match:
            i += 1
            continue

        start = parse_time(match.group("start"))
        end = parse_time(match.group("end"))
        i += 1

        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        text = clean_text(" ".join(text_lines))
        if text:
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "words": [],
                "source": "teams_transcript",
            })
        i += 1

    return merge_short_segments(segments)


def parse_txt(path: Path) -> List[Dict]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    segments = []

    # Handles lines like:
    # 00:01:23 Speaker Name: text...
    # [00:01:23] Speaker Name: text...
    line_re = re.compile(r"^\[?(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]?\s+(?P<text>.+)$")
    for idx, line in enumerate(lines):
        match = line_re.match(line)
        if not match:
            continue

        start = parse_time(match.group("time"))
        next_start = None
        for next_line in lines[idx + 1:]:
            next_match = line_re.match(next_line)
            if next_match:
                next_start = parse_time(next_match.group("time"))
                break

        end = next_start if next_start and next_start > start else start + 8.0
        text = clean_text(match.group("text"))
        if text:
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "words": [],
                "source": "teams_transcript",
            })

    if segments:
        return merge_short_segments(segments)

    raise ValueError(
        "Could not parse timestamps from TXT. Export Teams transcript as .vtt when possible."
    )


def merge_short_segments(segments: List[Dict], min_duration: float = 1.5, max_duration: float = 20.0) -> List[Dict]:
    merged = []
    current = None

    for seg in segments:
        duration = seg["end"] - seg["start"]
        if current is None:
            current = dict(seg)
            continue

        current_duration = current["end"] - current["start"]
        gap = seg["start"] - current["end"]
        should_merge = (
            current_duration < min_duration
            or duration < min_duration
            or (gap <= 0.8 and current_duration + gap + duration <= max_duration)
        )

        if should_merge:
            current["end"] = seg["end"]
            current["text"] = clean_text(current["text"] + " " + seg["text"])
        else:
            merged.append(current)
            current = dict(seg)

    if current:
        merged.append(current)
    return merged


def parse_transcript(path: Path) -> List[Dict]:
    suffix = path.suffix.lower()
    if suffix in {".vtt", ".srt"}:
        return parse_vtt_or_srt(path)
    if suffix == ".txt":
        return parse_txt(path)
    raise ValueError("Supported transcript formats: .vtt, .srt, .txt")


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(video_path: Path, output_wav: Path) -> None:
    cmd = [
        ffmpeg_exe(), "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "highpass=f=80,lowpass=f=7800,dynaudnorm=f=150:g=15",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_dataset(video_path: Path, segments: List[Dict], output_dir: Path) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        full_wav = Path(tmp) / "full_audio.wav"
        extract_audio(video_path, full_wav)

        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            duration = end - start
            text = clean_text(seg["text"])
            if duration < 1.0 or duration > 30.0 or len(text.split()) < 2:
                continue

            file_name = f"audio/teams_{len(rows) + 1:04d}.wav"
            out_audio = output_dir / file_name
            cmd = [
                ffmpeg_exe(), "-y",
                "-ss", str(start),
                "-t", str(duration),
                "-i", str(full_wav),
                "-acodec", "copy",
                str(out_audio),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            rows.append({"file_name": file_name, "sentence": text})

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Import Teams transcript into EchoNotes data.")
    parser.add_argument("--transcript", required=True, help="Teams transcript file (.vtt, .srt, .txt).")
    parser.add_argument("--video", help="Matching video file. Required when --dataset-output is used.")
    parser.add_argument("--cache-output", default="data/processed/transcripts/teams_segments_cache.json")
    parser.add_argument("--dataset-output", help="Optional clean AudioFolder dataset output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    segments = parse_transcript(transcript_path)
    if not segments:
        raise ValueError("No transcript segments were parsed.")

    cache_output = Path(args.cache_output)
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_output, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    print(f"[OK] Parsed {len(segments)} transcript segments.")
    print(f"[OK] Saved EchoNotes cache: {cache_output.resolve()}")

    if args.dataset_output:
        if not args.video:
            raise ValueError("--video is required when --dataset-output is set.")
        count = build_dataset(Path(args.video), segments, Path(args.dataset_output))
        print(f"[OK] Built clean Teams-labeled dataset with {count} clips: {Path(args.dataset_output).resolve()}")


if __name__ == "__main__":
    main()
