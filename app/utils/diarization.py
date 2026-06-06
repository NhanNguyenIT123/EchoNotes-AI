from __future__ import annotations

import re
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
        speakers.setdefault(speaker, {"segments": 0, "role_votes": {}})
        speakers[speaker]["segments"] += 1
        speakers[speaker]["role_votes"][role] = speakers[speaker]["role_votes"].get(role, 0) + 1
        labeled.append({**seg, "speaker": speaker, "speaker_role": role})

    summary = []
    for speaker, info in speakers.items():
        role = max(info["role_votes"].items(), key=lambda item: item[1])[0]
        summary.append({"speaker": speaker, "role": role, "segments": info["segments"]})

    return {
        "available": bool(labeled),
        "engine": "heuristic speaker-role pass; WhisperX/pyannote-ready",
        "speakers": summary,
        "segments": labeled,
    }


def _extract_speaker_prefix(text: str) -> str | None:
    match = re.match(r"^([A-Z][A-Za-z .'-]{1,40}):\s+", text or "")
    return match.group(1).strip() if match else None


def _role_from_text(text: str, idx: int) -> str:
    lower = (text or "").lower()
    instructor_markers = [
        "let's", "we will", "i will show", "you can see", "for example", "remember",
        "correct", "this means", "what we do", "the idea is", "you need to",
    ]
    student_markers = ["can you repeat", "i have a question", "yes", "no", "okay", "sorry", "i don't understand"]
    score_i = sum(1 for marker in instructor_markers if marker in lower)
    score_s = sum(1 for marker in student_markers if marker in lower)
    if idx < 8 and any(marker in lower for marker in ["can you see my screen", "before we start"]):
        score_i += 1
    if score_i >= score_s and score_i > 0:
        return "instructor"
    if score_s > 0:
        return "student"
    return "participant"
