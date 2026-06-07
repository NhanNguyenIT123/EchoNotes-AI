from __future__ import annotations

import re
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein_distance(left: List[str] | str, right: List[str] | str) -> int:
    if isinstance(left, str):
        left = list(left)
    if isinstance(right, str):
        right = list(right)
    prev = list(range(len(right) + 1))
    for i, l_item in enumerate(left, start=1):
        curr = [i]
        for j, r_item in enumerate(right, start=1):
            cost = 0 if l_item == r_item else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = normalize_text(reference).replace(" ", "")
    hyp_chars = normalize_text(hypothesis).replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def segments_to_text(segments: List[Dict[str, Any]]) -> str:
    return " ".join((seg.get("text") or "").strip() for seg in segments or [] if (seg.get("text") or "").strip())


def _quality_grade(score: float | None) -> str:
    if score is None:
        return "unavailable"
    if score <= 0.15:
        return "excellent"
    if score <= 0.35:
        return "usable"
    if score <= 0.6:
        return "needs review"
    return "poor"


def _nearest_segment(target_start: float, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not segments:
        return {}
    return min(segments, key=lambda seg: abs(float(seg.get("start", 0) or 0) - target_start))


def transcript_comparison_samples(
    reference_segments: List[Dict[str, Any]],
    hypothesis_segments: List[Dict[str, Any]],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    if not reference_segments or not hypothesis_segments:
        return []
    step = max(1, len(reference_segments) // limit)
    samples = []
    for ref in reference_segments[::step][:limit]:
        start = float(ref.get("start", 0) or 0)
        hyp = _nearest_segment(start, hypothesis_segments)
        ref_text = (ref.get("text") or "").strip()
        hyp_text = (hyp.get("text") or "").strip()
        samples.append(
            {
                "start": start,
                "reference": ref_text,
                "hypothesis": hyp_text,
                "wer": round(wer(ref_text, hyp_text), 4) if ref_text and hyp_text else None,
                "cer": round(cer(ref_text, hyp_text), 4) if ref_text and hyp_text else None,
            }
        )
    return samples


def evaluate_transcript_quality(
    reference_segments: List[Dict[str, Any]],
    hypothesis_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reference_text = segments_to_text(reference_segments)
    hypothesis_text = segments_to_text(hypothesis_segments)
    wer_score = round(wer(reference_text, hypothesis_text), 4) if reference_text and hypothesis_text else None
    cer_score = round(cer(reference_text, hypothesis_text), 4) if reference_text and hypothesis_text else None
    return {
        "available": bool(reference_text and hypothesis_text),
        "wer": wer_score,
        "cer": cer_score,
        "grade": _quality_grade(wer_score),
        "reference_words": len(normalize_text(reference_text).split()),
        "hypothesis_words": len(normalize_text(hypothesis_text).split()),
        "reference_segments": len(reference_segments or []),
        "hypothesis_segments": len(hypothesis_segments or []),
        "samples": transcript_comparison_samples(reference_segments, hypothesis_segments),
    }
