from __future__ import annotations

import re
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List


def infer_speaker_roles(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lightweight speaker-role pass.

    This is not neural diarization. It uses speaker labels when Teams/VTT provides
    them, otherwise it classifies likely instructor/student turns from discourse cues.
    WhisperX/pyannote can replace this module later without changing the API.
    """
    labeled = []
    speakers = {}
    for idx, seg in enumerate(segments or []):
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker") or _extract_speaker_prefix(text) or "unknown"
        role = _role_from_text(text, idx)
        duration = max(0.0, float(seg.get("end", 0) or 0) - float(seg.get("start", 0) or 0))
        speakers.setdefault(speaker, {"segments": 0, "role_votes": {}, "duration": 0.0, "words": 0, "texts": []})
        speakers[speaker]["segments"] += 1
        speakers[speaker]["duration"] += duration
        speakers[speaker]["words"] += len(text.split())
        if text:
            speakers[speaker]["texts"].append(text)
        speakers[speaker]["role_votes"][role] = speakers[speaker]["role_votes"].get(role, 0) + 1
        labeled.append({**seg, "speaker": speaker, "speaker_role": role})

    summary = []
    ranked_by_duration = sorted(speakers.items(), key=lambda item: item[1]["duration"], reverse=True)
    lead_speaker = ranked_by_duration[0][0] if ranked_by_duration else None
    for speaker, info in speakers.items():
        role = _aggregate_role_for_speaker(speaker, info, speaker == lead_speaker)
        summary.append({
            "speaker": speaker,
            "role": role,
            "segments": info["segments"],
            "duration_sec": round(info["duration"], 2),
            "words": info["words"],
        })

    return {
        "available": bool(labeled),
        "engine": "speaker-role pass over transcript labels; pyannote labels are used when available",
        "speakers": summary,
        "segments": labeled,
    }


def apply_optional_diarization(audio_path: Path, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Attach speaker labels using pyannote.audio when configured.

    Requirements:
    - ECHONOTES_DIARIZATION_PROVIDER=pyannote
    - HUGGINGFACE_TOKEN or PYANNOTE_AUTH_TOKEN with access to the pyannote model
    - pyannote.audio installed from requirements-prod.txt

    If the runtime is not configured, this returns the original segments plus a
    clear diagnostic instead of failing the whole lecture pipeline.
    """
    provider = os.getenv("ECHONOTES_DIARIZATION_PROVIDER", "heuristic").strip().lower()
    if provider != "pyannote":
        role_info = infer_speaker_roles(segments)
        return {
            "segments": role_info.get("segments", segments),
            "metadata": {
                "provider": "heuristic",
                "status": "fallback",
                "reason": "Set ECHONOTES_DIARIZATION_PROVIDER=pyannote to enable neural diarization.",
            },
        }

    token = os.getenv("PYANNOTE_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        role_info = infer_speaker_roles(segments)
        return {
            "segments": role_info.get("segments", segments),
            "metadata": {
                "provider": "pyannote",
                "status": "fallback",
                "reason": "Missing PYANNOTE_AUTH_TOKEN/HUGGINGFACE_TOKEN.",
            },
        }

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as exc:
        role_info = infer_speaker_roles(segments)
        return {
            "segments": role_info.get("segments", segments),
            "metadata": {
                "provider": "pyannote",
                "status": "fallback",
                "reason": f"pyannote.audio is not installed or failed to import: {exc}",
            },
        }

    try:
        import torch
        model_name = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            pipeline.to(torch.device(device))
        except Exception:
            device = "cpu"
        diarization = pipeline(str(audio_path))
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})

        labeled = []
        for segment in segments or []:
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", start + 1) or start + 1)
            speaker = _best_overlap_speaker(start, end, turns)
            labeled.append({**segment, "speaker": speaker or segment.get("speaker") or "unknown"})

        role_info = infer_speaker_roles(labeled)
        return {
            "segments": role_info.get("segments", labeled),
            "metadata": {
                "provider": "pyannote",
                "status": "completed",
                "model": model_name,
                "device": device,
                "turns": len(turns),
                "speakers": len({turn["speaker"] for turn in turns}),
            },
        }
    except Exception as exc:
        role_info = infer_speaker_roles(segments)
        return {
            "segments": role_info.get("segments", segments),
            "metadata": {
                "provider": "pyannote",
                "status": "fallback",
                "reason": str(exc),
            },
        }


def _best_overlap_speaker(start: float, end: float, turns: List[Dict[str, Any]]) -> str | None:
    overlaps: Dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))
        if overlap > 0:
            overlaps[turn["speaker"]] = overlaps.get(turn["speaker"], 0.0) + overlap
    if not overlaps:
        return None
    return max(overlaps.items(), key=lambda item: item[1])[0]


def _extract_speaker_prefix(text: str) -> str | None:
    match = re.match(r"^([A-Z][A-Za-z .'-]{1,40}):\s+", text or "")
    return match.group(1).strip() if match else None


def _role_from_text(text: str, idx: int) -> str:
    lower = _normalize_text(text)
    instructor_markers = [
        "let's", "we will", "i will show", "you can see", "for example", "remember",
        "correct", "this means", "what we do", "the idea is", "you need to",
        "chung ta", "cac ban", "vi du", "dau tien", "tiep theo", "nhu vay",
        "deadline", "lab",
    ]
    student_markers = [
        "can you repeat", "i have a question", "yes", "no", "okay", "sorry", "i don't understand",
        "da", "co a", "em", "khong nghe", "khong thay", "chua hieu",
    ]
    score_i = sum(1 for marker in instructor_markers if marker in lower)
    score_s = sum(1 for marker in student_markers if marker in lower)
    if idx < 8 and any(marker in lower for marker in ["can you see my screen", "before we start"]):
        score_i += 1
    if score_i >= score_s and score_i > 0:
        return "instructor"
    if score_s > 0:
        return "student"
    return "participant"


def _aggregate_role_for_speaker(speaker: str, info: Dict[str, Any], is_lead_speaker: bool) -> str:
    if speaker == "unknown":
        return max(info["role_votes"].items(), key=lambda item: item[1])[0] if info["role_votes"] else "participant"

    texts = _normalize_text(" ".join(info.get("texts", [])[:80]))
    duration = float(info.get("duration", 0.0) or 0.0)
    segments = int(info.get("segments", 0) or 0)
    words = int(info.get("words", 0) or 0)
    avg_words = words / max(1, segments)

    instructor_cues = [
        "let's", "we will", "i will show", "you can see", "for example", "remember",
        "this means", "you need to", "quantum", "round robin", "feedback", "queue",
        "chung ta", "cac ban", "vi du", "thay", "co", "minh se", "ta se", "bai nay",
        "dau tien", "tiep theo", "nhu vay", "deadline", "lab",
    ]
    student_cues = [
        "da", "co a", "em", "yes", "no", "okay", "sorry", "i have a question",
        "can you repeat", "chua hieu", "hoi", "khong nghe", "khong thay",
    ]
    instructor_score = sum(1 for cue in instructor_cues if cue in texts)
    student_score = sum(1 for cue in student_cues if cue in texts)

    if is_lead_speaker and (duration >= 60 or segments >= 12 or instructor_score >= student_score):
        return "instructor"
    if avg_words <= 6 and student_score > 0 and duration < 120:
        return "student"
    if instructor_score >= student_score + 1 and (duration >= 30 or avg_words >= 8):
        return "instructor"
    if student_score > instructor_score:
        return "student"
    return "participant"


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    without_marks = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_marks.replace("đ", "d").replace("Đ", "D").lower()
