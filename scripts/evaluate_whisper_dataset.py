# -*- coding: utf-8 -*-
"""
Evaluate a Faster-Whisper model on a corrected AudioFolder dataset.

This script computes simple WER/CER without external dependencies, so it works
even when jiwer is not installed.
"""

import argparse
import csv
import re
import unicodedata
from pathlib import Path
from faster_whisper import WhisperModel


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster-Whisper on a corrected dataset.")
    parser.add_argument("--dataset", default="corrected_lecture_dataset", help="AudioFolder dataset with metadata.csv.")
    parser.add_argument("--model", default="small", help="Faster-Whisper model name or local CT2 path.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    parser.add_argument("--compute-type", default="float16", help="float16/int8/float32.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=4)

    total_word_edits = total_words = 0
    total_char_edits = total_chars = 0
    rows_done = 0

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit and rows_done >= args.limit:
                break

            audio_path = dataset_dir / row["file_name"]
            reference = row["sentence"]
            segments, _ = model.transcribe(
                str(audio_path),
                language="vi",
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                hotwords="inode block pointer file system single indirect double indirect triple indirect FreeBSD UNIX cấp phát lưu trữ con trỏ trực tiếp gián tiếp",
            )
            hypothesis = " ".join(seg.text.strip() for seg in segments)

            ref_norm = normalize_text(reference)
            hyp_norm = normalize_text(hypothesis)
            ref_words = ref_norm.split()
            hyp_words = hyp_norm.split()

            word_edits = edit_distance(ref_words, hyp_words)
            char_edits = edit_distance(ref_norm, hyp_norm)

            total_word_edits += word_edits
            total_words += max(1, len(ref_words))
            total_char_edits += char_edits
            total_chars += max(1, len(ref_norm))
            rows_done += 1

            print(f"\n[{rows_done}] {row['file_name']}")
            print(f"REF: {reference}")
            print(f"HYP: {hypothesis}")

    wer = total_word_edits / total_words if total_words else 0.0
    cer = total_char_edits / total_chars if total_chars else 0.0
    print("\n" + "=" * 60)
    print(f"Rows: {rows_done}")
    print(f"WER : {wer:.2%}")
    print(f"CER : {cer:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
