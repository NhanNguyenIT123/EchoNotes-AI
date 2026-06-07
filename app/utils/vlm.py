from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

import requests

from app.config import OLLAMA_API_URL


VISION_MODEL_HINTS = ("llava", "bakllava", "moondream", "minicpm-v", "qwen2-vl", "qwen2.5vl", "qwen-vl")


def looks_like_vision_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(hint in name for hint in VISION_MODEL_HINTS)


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def analyze_keyframe_with_ollama_vlm(
    image_path: str,
    model_name: str,
    nearby_transcript: str = "",
    timeout_sec: int = 120,
) -> str:
    """Ask an Ollama vision model to describe a lecture keyframe."""
    image_b64 = image_to_base64(image_path)
    if not image_b64:
        return ""

    prompt = (
        "Analyze this lecture/video keyframe directly from the image. "
        "Focus on visible UI, slide content, diagrams, code, tables, highlighted regions, and what the instructor is likely demonstrating. "
        "Do not invent text that is not visible. Do not output Chinese characters. "
        "Answer in concise English bullets.\n\n"
    )
    if nearby_transcript.strip():
        prompt += (
            "Nearby transcript is provided only as supporting context, not as a replacement for visual inspection:\n"
            f"{nearby_transcript[:900]}\n"
        )

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 260,
        },
    }

    response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload, timeout=(10, timeout_sec))
    if response.status_code != 200:
        raise RuntimeError(f"Ollama vision model returned HTTP {response.status_code}: {response.text[:300]}")
    return (response.json().get("message", {}).get("content") or "").strip()


def enrich_keyframes_with_vlm(
    slides: List[Dict[str, Any]],
    transcript_segments: List[Dict[str, Any]],
    model_name: str,
    max_frames: int = 12,
) -> List[Dict[str, Any]]:
    """
    Adds image-understanding descriptions to keyframes using a local Ollama vision model.
    The cap keeps long lectures usable on local machines.
    """
    enriched: List[Dict[str, Any]] = []
    if not model_name:
        return slides

    for idx, slide in enumerate(slides):
        slide_copy = dict(slide)
        image_path = slide_copy.get("image_path") or ""
        if idx >= max_frames or not image_path:
            slide_copy.setdefault("vlm_description", "")
            enriched.append(slide_copy)
            continue

        start = float(slide_copy.get("timestamp_sec", 0) or 0)
        end = float(slides[idx + 1].get("timestamp_sec", start + 90) or start + 90) if idx + 1 < len(slides) else start + 90
        nearby = " ".join(
            (seg.get("text") or "").strip()
            for seg in transcript_segments
            if start <= float(seg.get("start", 0) or 0) < end
        )

        try:
            slide_copy["vlm_description"] = analyze_keyframe_with_ollama_vlm(
                image_path=image_path,
                model_name=model_name,
                nearby_transcript=nearby,
            )
        except Exception as exc:
            slide_copy["vlm_description"] = f"[Image understanding unavailable for this keyframe: {exc}]"
        enriched.append(slide_copy)

    return enriched
