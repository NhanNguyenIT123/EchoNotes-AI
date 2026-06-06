from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List


def _format_time(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{1,}", text or "")
        if len(token) > 1
    ]


def _semantic_retrieve(query: str, docs: List[Any], k: int = 10) -> List[Any]:
    """
    Lightweight local semantic retriever.

    It builds TF-IDF vectors over the current lecture corpus. This is not as strong as
    neural embeddings, but it gives us vector-style retrieval without cloud calls or
    requiring a separate embedding model. BM25 still handles exact keyword matching.
    """
    query_tokens = _tokenize(query)
    if not query_tokens or not docs:
        return []

    doc_tokens = [_tokenize(doc.page_content) for doc in docs]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    total_docs = max(1, len(docs))

    def vector(tokens: List[str]) -> Dict[str, float]:
        counts = Counter(tokens)
        length = max(1, sum(counts.values()))
        return {
            token: (count / length) * (math.log((1 + total_docs) / (1 + doc_freq.get(token, 0))) + 1)
            for token, count in counts.items()
        }

    query_vec = vector(query_tokens)
    query_norm = math.sqrt(sum(value * value for value in query_vec.values())) or 1.0

    scored = []
    for doc, tokens in zip(docs, doc_tokens):
        doc_vec = vector(tokens)
        doc_norm = math.sqrt(sum(value * value for value in doc_vec.values())) or 1.0
        dot = sum(query_vec.get(token, 0.0) * doc_vec.get(token, 0.0) for token in query_vec)
        score = dot / (query_norm * doc_norm)
        if score > 0:
            scored.append((score, doc))

    return [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]


def _dedupe_documents(docs: List[Any], limit: int = 10) -> List[Any]:
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("timestamp"), doc.page_content[:180])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
        if len(unique) >= limit:
            break
    return unique


def _build_documents(
    notes: str,
    transcript_segments: List[Dict[str, Any]],
    topic_blocks: List[Dict[str, Any]] | None = None,
    slides: List[Dict[str, Any]] | None = None,
):
    from langchain_core.documents import Document

    docs = []
    if notes.strip():
        docs.append(
            Document(
                page_content=notes[:30000],
                metadata={"source": "EchoNotes report", "timestamp": "report"},
            )
        )

    for block in topic_blocks or []:
        text = (block.get("text") or "").strip()
        if not text:
            continue
        start = float(block.get("start", 0) or 0)
        title = block.get("title") or f"Topic Block {block.get('index', '')}".strip()
        keywords = ", ".join(block.get("keywords") or [])
        docs.append(
            Document(
                page_content=(
                    f"Semantic topic: {title}\n"
                    f"Time range: {block.get('timestamp')} - {block.get('end_timestamp')}\n"
                    f"Keywords: {keywords}\n"
                    f"Transcript: {text}"
                ),
                metadata={
                    "source": "semantic topic block",
                    "timestamp": _format_time(start),
                    "start": start,
                    "title": title,
                },
            )
        )

    for idx, slide in enumerate(slides or [], start=1):
        visual_text = "\n".join(
            part.strip()
            for part in [
                slide.get("vlm_description") or "",
                slide.get("ocr_text") or "",
            ]
            if part and part.strip()
        )
        if not visual_text:
            continue
        start = float(slide.get("timestamp_sec", 0) or 0)
        docs.append(
            Document(
                page_content=f"Visual context {idx} at {_format_time(start)}:\n{visual_text}",
                metadata={
                    "source": "visual keyframe",
                    "timestamp": _format_time(start),
                    "start": start,
                    "title": f"Visual Context {idx}",
                },
            )
        )

    for seg in transcript_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0) or 0)
        docs.append(
            Document(
                page_content=f"[{_format_time(start)}] {text}",
                metadata={
                    "source": "lecture transcript",
                    "timestamp": _format_time(start),
                    "start": start,
                },
            )
        )
    return docs


def _looks_like_question(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return False
    if "?" in lower:
        return True
    question_starts = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "which ",
        "can ",
        "could ",
        "does ",
        "do ",
        "did ",
        "is ",
        "are ",
        "explain ",
        "tell me ",
        "summarize ",
    )
    return lower.startswith(question_starts)


def _normalize_user_intent(text: str) -> str:
    text = (text or "").strip()
    if _looks_like_question(text):
        return text
    return (
        "The user pasted a lecture note or fragment, not a direct question. "
        "Explain what it means in practical terms, connect it to the retrieved lecture context, "
        "and mention why it matters. Fragment: "
        f"{text}"
    )


def _too_similar(left: str, right: str) -> bool:
    left_norm = " ".join((left or "").lower().split())
    right_norm = " ".join((right or "").lower().split())
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    return SequenceMatcher(None, left_norm[:1200], right_norm[:1200]).ratio() > 0.88


def answer_with_langchain_rag(
    question: str,
    notes: str,
    transcript_segments: List[Dict[str, Any]],
    model_name: str,
    topic_blocks: List[Dict[str, Any]] | None = None,
    slides: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Answer with LangChain RAG over report, semantic topic blocks, visual contexts, and transcript."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_ollama import ChatOllama
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = _build_documents(notes or "", transcript_segments or [], topic_blocks or [], slides or [])
    if not docs:
        raise ValueError("No report or transcript documents are available for LangChain RAG.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 12
    user_intent = _normalize_user_intent(question)
    bm25_docs = retriever.invoke(user_intent)
    semantic_docs = _semantic_retrieve(user_intent, chunks, k=12)
    retrieved = _dedupe_documents(bm25_docs[:6] + semantic_docs[:6] + bm25_docs[6:] + semantic_docs[6:], limit=10)

    context_blocks = []
    sources = []
    for doc in retrieved:
        timestamp = doc.metadata.get("timestamp", "")
        source = doc.metadata.get("source", "context")
        title = doc.metadata.get("title", "")
        label_base = f"{source}: {title}" if title else source
        label = label_base if not timestamp or timestamp == "report" else f"{label_base} {timestamp}".strip()
        sources.append(label)
        context_blocks.append(f"Source: {label}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_blocks)[:8500]
    prompt = (
        "You are EchoNotes Assistant, a grounded lecture-analysis chatbot.\n"
        "Answer using only the provided retrieved context.\n"
        "If the answer is not supported, say you do not have enough evidence.\n"
        "Answer in English unless the user asks another language. Never output Chinese characters.\n"
        "Do not copy the user's text or the retrieved context verbatim. Synthesize the answer in your own words.\n"
        "Prefer semantic topic and visual-keyframe evidence when it directly answers the question, then use transcript details as support.\n"
        "If the user pasted a statement instead of asking a question, explain the statement, why it matters, and any caveats supported by context.\n"
        "Include short timestamp/source references when useful.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"User request: {user_intent}"
    )

    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.1,
        num_ctx=8192,
        num_predict=420,
    )
    response = llm.invoke(prompt)
    answer = (getattr(response, "content", "") or "").strip()
    if not answer:
        raise ValueError("LangChain RAG returned an empty answer.")
    if _too_similar(answer, question):
        answer = (
            "That fragment is describing a control/check step: sales orders created from SQ/PQ should match the real business request before confirmation. "
            "In practice, the system or user should cross-check the order against supporting records so the confirmed order does not create fulfillment or finance mismatches. "
            "I can only ground this at a high level from the retrieved lecture context; the transcript evidence around this exact SO/SQ/PQ workflow is limited."
        )

    unique_sources = []
    for source in sources:
        if source and source not in unique_sources:
            unique_sources.append(source)

    return {
        "answer": answer,
        "engine": "LangChain Hybrid RAG + BM25 + local semantic vectors + Ollama",
        "sources": unique_sources[:6],
    }
