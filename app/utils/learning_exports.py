from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def build_quiz_bank(topic_blocks: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    quiz = []
    for block in (topic_blocks or [])[:limit]:
        keywords = block.get("keywords") or []
        title = block.get("title") or "Lecture topic"
        if keywords:
            question = f"What is the role of {keywords[0]} in the topic '{title}'?"
        else:
            question = f"Summarize the main idea of '{title}'."
        quiz.append(
            {
                "type": "short_answer",
                "timestamp": block.get("timestamp"),
                "topic": title,
                "question": question,
                "answer_hint": (block.get("text") or "")[:420],
                "keywords": keywords[:8],
            }
        )
    return quiz


def write_quiz_json(topic_blocks: List[Dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_quiz_bank(topic_blocks), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_anki_tsv(topic_blocks: List[Dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in build_quiz_bank(topic_blocks):
        front = f"[{item.get('timestamp')}] {item['question']}"
        back = f"{item['answer_hint']}\n\nKeywords: {', '.join(item.get('keywords') or [])}"
        rows.append([front, back, item.get("topic") or "EchoNotes"])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)
    return output_path
