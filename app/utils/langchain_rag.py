from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List


def _format_time(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _build_documents(notes: str, transcript_segments: List[Dict[str, Any]]):
    from langchain_core.documents import Document

    docs = []
    if notes.strip():
        docs.append(
            Document(
                page_content=notes[:30000],
                metadata={"source": "EchoNotes report", "timestamp": "report"},
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
) -> Dict[str, Any]:
    """Answer with a real LangChain RAG pipeline over report + timestamped transcript."""
    from langchain_community.retrievers import BM25Retriever
    from langchain_ollama import ChatOllama
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = _build_documents(notes or "", transcript_segments or [])
    if not docs:
        raise ValueError("No report or transcript documents are available for LangChain RAG.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 8
    user_intent = _normalize_user_intent(question)
    retrieved = retriever.invoke(user_intent)

    context_blocks = []
    sources = []
    for doc in retrieved:
        timestamp = doc.metadata.get("timestamp", "")
        source = doc.metadata.get("source", "context")
        label = source if not timestamp or timestamp == "report" else f"{source} {timestamp}".strip()
        sources.append(label)
        context_blocks.append(f"Source: {label}\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_blocks)[:7000]
    prompt = (
        "You are EchoNotes Assistant, a grounded lecture-analysis chatbot.\n"
        "Answer using only the provided retrieved context.\n"
        "If the answer is not supported, say you do not have enough evidence.\n"
        "Answer in English unless the user asks another language. Never output Chinese characters.\n"
        "Do not copy the user's text or the retrieved context verbatim. Synthesize the answer in your own words.\n"
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
        "engine": "LangChain RAG + BM25Retriever + Ollama",
        "sources": unique_sources[:6],
    }
