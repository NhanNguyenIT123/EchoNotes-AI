from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Set


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by", "can", "could",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "him", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "just", "like", "me", "my", "not", "of", "on",
    "or", "our", "so", "that", "the", "their", "them", "then", "there", "this", "to", "was",
    "we", "what", "when", "where", "which", "who", "will", "with", "would", "you", "your",
    "yeah", "yes", "okay", "ok", "right", "actually", "basically", "now", "one", "two",
}


DOMAIN_HINTS = {
    "business", "central", "table", "relation", "vendor", "sales", "purchase", "request",
    "invoice", "field", "code", "payment", "terms", "customer", "header", "line", "lookup",
    "extension", "page", "source", "record", "database", "workflow", "confirm", "order",
}


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{1,}", text or "")
        if token.lower() not in STOPWORDS
    ]


def keyword_list(text: str, limit: int = 10) -> List[str]:
    counts = Counter(tokenize(text))
    if not counts:
        return []
    ranked = sorted(
        counts.items(),
        key=lambda item: (-(item[1] + (1.5 if item[0] in DOMAIN_HINTS else 0)), item[0]),
    )
    return [word for word, _ in ranked[:limit]]


def _token_set(segments: Iterable[Dict[str, Any]]) -> Set[str]:
    text = " ".join((seg.get("text") or "") for seg in segments)
    return set(keyword_list(text, limit=28))


def _overlap(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _title_from_keywords(keywords: List[str]) -> str:
    if not keywords:
        return "Lecture Topic"
    title_terms = [kw.replace("_", " ").title() for kw in keywords[:4]]
    return " / ".join(title_terms)


def build_semantic_topic_blocks(
    transcript_segments: List[Dict[str, Any]],
    min_words: int = 120,
    max_words: int = 420,
    max_duration_sec: int = 420,
    semantic_shift_threshold: float = 0.18,
) -> List[Dict[str, Any]]:
    """
    Build topic blocks from timestamped transcript using lexical semantic-shift signals.

    This is deterministic and local, so it is stable enough for tests and does not need
    cloud embeddings. It groups neighboring utterances until the topic vocabulary shifts,
    or the block becomes too long for summarization/retrieval.
    """
    clean_segments = [
        seg for seg in transcript_segments or []
        if (seg.get("text") or "").strip()
    ]
    if not clean_segments:
        return []

    blocks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for idx, seg in enumerate(clean_segments):
        current.append(seg)
        text = " ".join((item.get("text") or "") for item in current)
        word_count = len(text.split())
        start = float(current[0].get("start", 0) or 0)
        end = float(current[-1].get("end", current[-1].get("start", start)) or start)
        duration = end - start

        should_close = False
        if idx + 1 < len(clean_segments) and word_count >= min_words:
            current_terms = _token_set(current)
            lookahead_terms = _token_set(clean_segments[idx + 1: idx + 7])
            should_close = _overlap(current_terms, lookahead_terms) < semantic_shift_threshold
        if word_count >= max_words or duration >= max_duration_sec:
            should_close = True

        if should_close:
            blocks.append(_make_block(current, len(blocks) + 1))
            current = []

    if current:
        blocks.append(_make_block(current, len(blocks) + 1))

    return blocks


def _make_block(segments: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    start = float(segments[0].get("start", 0) or 0)
    end = float(segments[-1].get("end", segments[-1].get("start", start)) or start)
    text = " ".join((seg.get("text") or "").strip() for seg in segments)
    keywords = keyword_list(text, limit=12)
    return {
        "index": index,
        "title": _title_from_keywords(keywords),
        "start": start,
        "end": end,
        "timestamp": format_timestamp(start),
        "end_timestamp": format_timestamp(end),
        "keywords": keywords,
        "text": text,
        "segment_count": len(segments),
        "word_count": len(text.split()),
    }
