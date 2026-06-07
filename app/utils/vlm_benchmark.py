from __future__ import annotations

import re
from typing import Any, Dict, List


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text or "")
    }


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _confidence(text: str, aligned_text: str, source: str) -> float:
    token_count = len(_tokens(text))
    density = min(1.0, token_count / (35 if source == "ocr" else 55))
    alignment = _overlap(text, aligned_text)
    penalty = 0.0
    lower = (text or "").lower()
    if "unavailable" in lower or "error" in lower or "unsupported" in lower:
        penalty = 0.35
    return round(max(0.0, min(1.0, density * 0.72 + alignment * 0.28 - penalty)), 3)


def _specificity(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    technical_markers = {
        "table", "field", "relation", "order", "purchase", "sales", "invoice",
        "request", "vendor", "customer", "workflow", "code", "line", "header",
        "business", "central", "approval", "supplier", "quote", "operation",
    }
    marker_score = min(1.0, len(tokens & technical_markers) / 5)
    length_score = min(1.0, len(tokens) / 45)
    return round(0.55 * length_score + 0.45 * marker_score, 3)


def _confidence_label(score: float) -> str:
    if score >= 0.78:
        return "strong"
    if score >= 0.5:
        return "usable"
    if score > 0:
        return "weak"
    return "missing"


def build_vlm_benchmark(slides: List[Dict[str, Any]], transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for idx, slide in enumerate(slides or [], start=1):
        start = float(slide.get("timestamp_sec", 0) or 0)
        next_start = (
            float(slides[idx].get("timestamp_sec", start + 90) or start + 90)
            if idx < len(slides or [])
            else start + 90
        )
        aligned = " ".join(
            (seg.get("text") or "").strip()
            for seg in transcript or []
            if start <= float(seg.get("start", 0) or 0) < next_start
        )
        ocr = slide.get("ocr_text") or ""
        vlm = slide.get("vlm_description") or ""
        ocr_conf = _confidence(ocr, aligned, "ocr") if ocr else 0.0
        vlm_conf = _confidence(vlm, aligned, "vlm") if vlm else 0.0
        visual_specificity = max(_specificity(ocr), _specificity(vlm))
        transcript_alignment = round(max(_overlap(ocr, aligned), _overlap(vlm, aligned)), 3)
        fused_conf = round(min(1.0, max(ocr_conf, vlm_conf) * 0.72 + visual_specificity * 0.16 + transcript_alignment * 0.12), 3)
        rows.append(
            {
                "index": idx,
                "timestamp_sec": start,
                "timestamp": slide.get("timestamp_formatted") or _format_time(start),
                "has_ocr": bool(ocr.strip()),
                "has_vlm": bool(vlm.strip()),
                "ocr_confidence": ocr_conf,
                "vlm_confidence": vlm_conf,
                "ocr_vlm_confidence": fused_conf,
                "confidence_level": _confidence_label(fused_conf),
                "alignment_score": transcript_alignment,
                "rubric": {
                    "coverage": round((1 if ocr.strip() else 0) * 0.35 + (1 if vlm.strip() else 0) * 0.65, 3),
                    "specificity": visual_specificity,
                    "transcript_alignment": transcript_alignment,
                    "score": fused_conf,
                },
                "review_note": (
                    "Visual context is grounded enough for report synthesis."
                    if fused_conf >= 0.5
                    else "Needs OCR/VLM rerun or manual review before treating as strong visual evidence."
                ),
            }
        )
    count = max(1, len(rows))
    return {
        "benchmark_engine": "deterministic visual-grounding rubric over OCR, image-understanding text, and aligned transcript",
        "slides_evaluated": len(rows),
        "ocr_coverage": round(sum(1 for row in rows if row["has_ocr"]) / count, 3),
        "vlm_coverage": round(sum(1 for row in rows if row["has_vlm"]) / count, 3),
        "avg_ocr_confidence": round(sum(row["ocr_confidence"] for row in rows) / count, 3),
        "avg_vlm_confidence": round(sum(row["vlm_confidence"] for row in rows) / count, 3),
        "avg_ocr_vlm_confidence": round(sum(row["ocr_vlm_confidence"] for row in rows) / count, 3),
        "strong_visual_evidence": sum(1 for row in rows if row["confidence_level"] == "strong"),
        "usable_visual_evidence": sum(1 for row in rows if row["confidence_level"] in {"strong", "usable"}),
        "rows": rows,
    }


def _format_time(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
