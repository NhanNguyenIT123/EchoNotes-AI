import re
from pathlib import Path
from typing import Dict, List


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


def clean_transcript_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"^\s*[-–]?\s*[^:\n]{1,60}:\s+", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def merge_short_segments(
    segments: List[Dict],
    min_duration: float = 1.5,
    max_duration: float = 20.0,
) -> List[Dict]:
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
            current["text"] = clean_transcript_text(current["text"] + " " + seg["text"])
        else:
            merged.append(current)
            current = dict(seg)

    if current:
        merged.append(current)
    return merged


def parse_vtt_or_srt(path: Path) -> List[Dict]:
    lines = Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines()
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

        text = clean_transcript_text(" ".join(text_lines))
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
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        if line.strip()
    ]
    segments = []
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
        text = clean_transcript_text(match.group("text"))
        if text:
            segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "text": text,
                "words": [],
                "source": "teams_transcript",
            })

    if not segments:
        raise ValueError("Could not parse timestamps from TXT. Export Teams transcript as .vtt when possible.")
    return merge_short_segments(segments)


def parse_teams_transcript(path: Path) -> List[Dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".vtt", ".srt"}:
        return parse_vtt_or_srt(path)
    if suffix == ".txt":
        return parse_txt(path)
    raise ValueError("Supported transcript formats: .vtt, .srt, .txt")
