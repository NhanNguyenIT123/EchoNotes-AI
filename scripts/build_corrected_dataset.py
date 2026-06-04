# -*- coding: utf-8 -*-
"""
Build a clean Hugging Face AudioFolder dataset from a manually corrected CSV.

Rows are included only when corrected_sentence is non-empty. The output can be
used directly with scripts/train_whisper_local.py.
"""

import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a clean AudioFolder dataset from corrected transcripts.")
    parser.add_argument("--correction-dir", default="correction_workspace", help="Directory from create_correction_pack.py.")
    parser.add_argument("--output", default="corrected_lecture_dataset", help="Output clean dataset directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    correction_dir = Path(args.correction_dir)
    input_csv = correction_dir / "metadata_to_correct.csv"
    output_dir = Path(args.output)
    output_audio_dir = output_dir / "audio"

    if not input_csv.exists():
        raise FileNotFoundError(f"Correction CSV not found: {input_csv}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corrected = (row.get("corrected_sentence") or "").strip()
            file_name = (row.get("file_name") or "").strip()
            if not corrected or not file_name:
                continue

            src_audio = correction_dir / file_name
            if not src_audio.exists():
                continue

            dst_name = f"audio/clean_{len(rows) + 1:04d}{src_audio.suffix.lower()}"
            shutil.copy2(src_audio, output_dir / dst_name)
            rows.append({"file_name": dst_name, "sentence": corrected})

    output_csv = output_dir / "metadata.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "sentence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Built clean dataset with {len(rows)} clips: {output_dir.resolve()}")
    if len(rows) < 80:
        print("[WARN] This is probably too small for useful fine-tuning. Aim for at least 100-200 corrected clips.")


if __name__ == "__main__":
    main()
