# -*- coding: utf-8 -*-
"""
Create a manual correction pack from an existing AudioFolder dataset.

Input:
  lecture_dataset/
    metadata.csv        columns: file_name,sentence
    audio/chunk_0001.wav

Output:
  correction_workspace/
    metadata_to_correct.csv
    audio/chunk_0001.wav

Edit only the corrected_sentence column. Leave rows blank when you do not want to
use that clip for training/evaluation.
"""

import argparse
import csv
import shutil
import wave
from pathlib import Path


def get_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Create a CSV/audio pack for manual transcript correction.")
    parser.add_argument("--source", default="lecture_dataset", help="Existing AudioFolder dataset directory.")
    parser.add_argument("--output", default="correction_workspace", help="Output correction workspace.")
    parser.add_argument("--limit", type=int, default=180, help="Max clips to export. 180 clips is usually ~30-60 minutes.")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Skip clips shorter than this many seconds.")
    parser.add_argument("--max-duration", type=float, default=18.0, help="Skip clips longer than this many seconds.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source)
    metadata_path = source_dir / "metadata.csv"
    output_dir = Path(args.output)
    output_audio_dir = output_dir / "audio"

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    rows_out = []

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if exported >= args.limit:
                break

            file_name = (row.get("file_name") or "").strip()
            draft = (row.get("sentence") or "").strip()
            if not file_name or not draft:
                continue

            src_audio = source_dir / file_name
            if not src_audio.exists():
                continue

            duration = get_duration_seconds(src_audio)
            if duration < args.min_duration or duration > args.max_duration:
                continue

            dst_name = f"audio/clip_{exported + 1:04d}{src_audio.suffix.lower()}"
            dst_audio = output_dir / dst_name
            shutil.copy2(src_audio, dst_audio)

            rows_out.append({
                "file_name": dst_name,
                "duration_sec": f"{duration:.2f}",
                "draft_sentence": draft,
                "corrected_sentence": "",
                "notes": "",
            })
            exported += 1

    output_csv = output_dir / "metadata_to_correct.csv"
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "duration_sec", "draft_sentence", "corrected_sentence", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] Exported {exported} clips to: {output_dir.resolve()}")
    print(f"[NEXT] Edit this file: {output_csv.resolve()}")
    print("       Fill corrected_sentence with the exact transcript you hear.")


if __name__ == "__main__":
    main()
