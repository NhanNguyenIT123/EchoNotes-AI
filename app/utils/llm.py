import json
import requests
import re
from pathlib import Path
from typing import List, Dict, Any, Generator
from app.config import OLLAMA_API_URL, OLLAMA_DEFAULT_MODEL, OLLAMA_FALLBACK_MODEL
from app.utils.corrector import TECH_VOCABULARY
from app.utils.topic_segmentation import build_semantic_topic_blocks

def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text or ""))

def strip_cjk_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        if not contains_cjk(line):
            lines.append(line)
    return "\n".join(lines).strip()

def infer_note_language(segments: List[Dict[str, Any]]) -> str:
    for seg in segments or []:
        meta = seg.get("asr_metadata") or {}
        requested = (meta.get("requested_language") or "").lower()
        detected = (meta.get("language") or meta.get("detected_language") or "").lower()
        if requested == "en" or detected == "en":
            return "English"
        if requested == "vi" or detected == "vi":
            return "Vietnamese"
    return "English"

def get_installed_ollama_models() -> List[str]:
    """
    Queries the local Ollama server to list all downloaded/installed models.
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
    except Exception:
        pass
    return []

def align_content_by_slides(
    slides: List[Dict[str, Any]], 
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Aligns spoken transcript segments and acoustic highlights with their corresponding slide timeline.
    Returns a structured list where each slide has its OCR text, screenshot, and transcript segments.
    """
    aligned = []
    num_slides = len(slides)
    
    for i, slide in enumerate(slides):
        start_time = slide["timestamp_sec"]
        # The next slide's timestamp marks the end of the current slide's segment
        end_time = slides[i + 1]["timestamp_sec"] if i + 1 < num_slides else float("inf")
        
        slide_segments = []
        highlights = []
        
        for seg in segments:
            # Check if segment falls within the slide's active timeline
            if start_time <= seg["start"] < end_time:
                slide_segments.append(seg["text"])
                
                # Check for acoustic emphasis
                if seg.get("acoustics", {}).get("is_important", False):
                    # Format high-priority moment
                    time_fmt = seg.get("acoustics", {}).get("timestamp_formatted", "")
                    if not time_fmt:
                        m = int(seg["start"] // 60)
                        s = int(seg["start"] % 60)
                        time_fmt = f"{m:02d}:{s:02d}"
                        
                    emphasis_type = []
                    ac = seg["acoustics"]
                    if ac.get("is_loud"): emphasis_type.append("louder/emphasized speech")
                    if ac.get("is_high_pitch"): emphasis_type.append("rising pitch")
                    if ac.get("is_slow"): emphasis_type.append("slower detailed explanation")
                    
                    type_str = ", ".join(emphasis_type) if emphasis_type else "emphasized speech"
                    highlights.append({
                        "timestamp": time_fmt,
                        "quote": seg["text"],
                        "type": type_str
                    })
                    
        aligned.append({
            "timestamp_formatted": slide["timestamp_formatted"],
            "image_path": slide["image_path"],
            "ocr_text": slide["ocr_text"],
            "vlm_description": slide.get("vlm_description", ""),
            "speech_text": " ".join(slide_segments),
            "highlights": highlights
        })
        
    return aligned

def has_visual_context(slides: List[Dict[str, Any]]) -> bool:
    """Returns True only when the pipeline produced real slide/OCR context."""
    for slide in slides or []:
        image_path = (slide.get("image_path") or "").strip()
        ocr_text = (slide.get("ocr_text") or "").strip()
        if image_path or ocr_text:
            return True
    return False

def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def chunk_segments_by_time(
    segments: List[Dict[str, Any]],
    chunk_minutes: int = 8,
    max_words: int = 850
) -> List[Dict[str, Any]]:
    """Groups timestamped transcript segments into NLP-friendly study chunks."""
    chunks = []
    current = []
    start_time = None
    word_count = 0
    window = chunk_minutes * 60

    for seg in segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        seg_start = float(seg.get("start", 0) or 0)
        if start_time is None:
            start_time = seg_start

        words = text.split()
        should_flush = current and (
            seg_start - start_time >= window or word_count + len(words) > max_words
        )
        if should_flush:
            chunks.append({
                "start": start_time,
                "end": float(current[-1].get("end", current[-1].get("start", start_time)) or start_time),
                "segments": current,
                "text": " ".join((x.get("text") or "").strip() for x in current if (x.get("text") or "").strip())
            })
            current = []
            start_time = seg_start
            word_count = 0

        current.append(seg)
        word_count += len(words)

    if current:
        chunks.append({
            "start": start_time if start_time is not None else 0,
            "end": float(current[-1].get("end", current[-1].get("start", start_time or 0)) or 0),
            "segments": current,
            "text": " ".join((x.get("text") or "").strip() for x in current if (x.get("text") or "").strip())
        })

    return chunks

def extract_keywords_simple(text: str, limit: int = 12) -> List[str]:
    """Lightweight keyword extraction for offline NLP notes without extra dependencies."""
    stopwords = {
        "the", "and", "for", "that", "this", "with", "you", "are", "was", "were", "from",
        "mot", "cua", "cac", "cho", "thi", "la", "va", "co", "khong", "duoc", "trong",
        "nhung", "nhu", "neu", "minh", "chung", "ta", "nay", "do", "de", "se", "toi",
        "can", "phai", "khi", "vao", "ra", "len", "xuong", "roi", "nua", "day", "ben"
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text.lower())
    counts: Dict[str, int] = {}
    for token in tokens:
        if token in stopwords:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]

def summarize_chunk_offline(text: str, max_sentences: int = 4) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]
    if not sentences:
        words = text.split()
        return [" ".join(words[:80]).strip()] if words else []

    keywords = set(extract_keywords_simple(text, limit=16))
    scored = []
    for idx, sentence in enumerate(sentences):
        score = sum(1 for kw in keywords if kw in sentence.lower())
        score += 1 if any(mark in sentence.lower() for mark in ["important", "exam", "note", "remember", "key", "error", "problem"]) else 0
        score += max(0, 3 - idx) * 0.2
        scored.append((score, idx, sentence))

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]

def generate_offline_transcript_notes(segments: List[Dict[str, Any]]) -> str:
    """Professional NLP-first fallback when the user runs transcript-only mode."""
    chunks = build_semantic_topic_blocks(segments)
    all_text = " ".join((s.get("text") or "").strip() for s in segments or [])
    total_words = len(all_text.split())
    duration = max((float(s.get("end", s.get("start", 0)) or 0) for s in segments or [{"end": 0}]), default=0)
    source = "Microsoft Teams transcript" if any(s.get("source") == "teams_transcript" for s in segments or []) else "ASR transcript"
    top_keywords = extract_keywords_simple(all_text, limit=18)

    md = []
    md.append("# EchoNotes AI - NLP Study Notes\n\n")
    md.append(f"> Source: **{source}** | Duration: **{format_timestamp(duration)}** | Segments: **{len(segments or [])}** | Words: **{total_words}**\n\n")
    md.append("## Session Overview\n\n")
    if top_keywords:
        md.append("**Core keywords:** " + ", ".join(f"`{kw}`" for kw in top_keywords) + "\n\n")
    md.append("These notes are generated from the transcript first. Slide/OCR context is intentionally skipped for this run, so the structure below focuses on topic flow, terminology, and reusable NLP data.\n\n")

    md.append("## Semantic Topic Map\n\n")
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        chunk_keywords = chunk.get("keywords") or extract_keywords_simple(text, limit=8)
        bullets = summarize_chunk_offline(text, max_sentences=4)
        md.append(f"### {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])} | {chunk.get('title') or f'Topic Block {idx}'}\n\n")
        if chunk_keywords:
            md.append("**Keywords:** " + ", ".join(f"`{kw}`" for kw in chunk_keywords) + "\n\n")
        md.append("**Lecture summary:**\n")
        for bullet in bullets:
            md.append(f"- {bullet}\n")
        md.append("\n**NLP learning signals:**\n")
        md.append(f"- Semantic block contains **{chunk.get('segment_count', 0)}** timestamped utterances and **{chunk.get('word_count', 0)}** words.\n")
        md.append("- Good candidate for topic labeling, keyphrase extraction, QA generation, and hybrid search indexing.\n\n")

    md.append("## Dataset Hooks\n\n")
    md.append("- `topic_segmentation`: use each Topic Block as one labeled training/evaluation unit.\n")
    md.append("- `keyword_extraction`: compare extracted keywords against manually curated course terms.\n")
    md.append("- `summarization`: use the Lecture summary bullets as draft labels, then correct them by hand.\n")
    md.append("- `question_generation`: generate review questions from each Topic Block after manual cleanup.\n")
    return "".join(md)

def generate_offline_study_notes(slides: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> str:
    """
    Generates a high-fidelity offline markdown study notebook directly from slide OCR, 
    cleaned speech transcripts, and acoustic highlights.
    Used as an elegant fail-safe fallback when the local Ollama LLM times out or is offline.
    """
    if not has_visual_context(slides):
        return generate_offline_transcript_notes(segments)

    aligned_data = align_content_by_slides(slides, segments)


    md = []
    md.append("# EchoNotes AI - Offline Lecture Report\n\n")
    md.append("> Local Ollama is unavailable or timed out, so EchoNotes used a deterministic offline synthesis path.\n")
    md.append("> The report is grounded in slide OCR, keyframes, and the cleaned transcript.\n\n")
    md.append("---\n\n")
    
    for idx, item in enumerate(aligned_data):
        md.append(f"## 🖼️ Phân Đoạn Slide {idx + 1} - Xuất hiện tại [{item['timestamp_formatted']}]\n\n")
        
        # Embed slide screenshot when available
        md.append(markdown_image_or_empty(item.get("image_path", "")))
        
        # OCR Text block
        if item["ocr_text"]:
            md.append("### 🔍 Nội dung hiển thị trên Slide (OCR Text)\n")
            md.append(f"```text\n{item['ocr_text'].strip()}\n```\n\n")
            
        # Cleaned Spoken Speech block
        if item["speech_text"]:
            md.append("### 🎓 Tóm tắt lời giảng tương ứng của giáo viên\n")
            md.append(f"{item['speech_text'].strip()}\n\n")
            
        # Acoustic Highlights
        if item["highlights"]:
            md.append("### 🔥 Điểm nhấn giọng nói đặc biệt cần lưu ý\n")
            for hl in item["highlights"]:
                md.append(f"- **⏱️ [{hl['timestamp']}] ({hl['type']})**: *\"{hl['quote']}\"*\n")
            md.append("\n")
            
        md.append("---\n\n")
        
    return "".join(md)

def generate_offline_study_notes(slides: List[Dict[str, Any]], transcript: List[Dict[str, Any]]) -> str:
    """Deterministic transcript-first report used when local LLM is unavailable."""
    aligned_data = align_content_by_slides(slides, transcript)
    topic_blocks = build_semantic_topic_blocks(transcript)
    md = []
    md.append("# EchoNotes AI - Offline Lecture Report\n\n")
    md.append("> Generated with the deterministic NLP pipeline. The report is grounded in transcript segments, visual keyframes, and optional extracted slide text when available.\n\n")
    md.append("---\n\n")

    if topic_blocks:
        md.append("## Semantic Topic Map\n\n")
        for block in topic_blocks[:14]:
            keywords = ", ".join(f"`{kw}`" for kw in block.get("keywords", [])[:8])
            md.append(f"### {block['timestamp']} - {block['end_timestamp']} | {block['title']}\n\n")
            if keywords:
                md.append(f"**Keywords:** {keywords}\n\n")
            for sentence in summarize_chunk_offline(block.get("text", ""), max_sentences=2):
                md.append(f"- {sentence}\n")
            md.append("\n")
        md.append("---\n\n")

    for idx, item in enumerate(aligned_data):
        md.append(f"## Visual Context {idx + 1} - {item['timestamp_formatted']}\n\n")
        md.append(markdown_image_or_empty(item.get("image_path", "")))

        md.append("### Keyframe Context\n")
        for bullet in build_grounded_visual_analysis(item):
            md.append(f"- {bullet}\n")
        md.append("\n")

        if item.get("vlm_description"):
            md.append("### Visual Image Understanding\n")
            md.append(f"{item['vlm_description'].strip()}\n\n")

        if item.get("speech_text"):
            md.append("### Lecture Interpretation\n")
            bullets = summarize_chunk_offline(item["speech_text"], max_sentences=3)
            if bullets:
                for bullet in bullets:
                    md.append(f"- {bullet}\n")
            else:
                md.append(f"{item['speech_text'].strip()}\n")
            md.append("\n")

        if item.get("ocr_text"):
            md.append("### Extracted Visual Evidence\n")
            md.append(f"```text\n{item['ocr_text'].strip()[:1200]}\n```\n\n")

        if item.get("highlights"):
            md.append("### Acoustic Highlights\n")
            for hl in item["highlights"]:
                md.append(f"- **[{hl['timestamp']}] ({hl['type']})**: \"{hl['quote']}\"\n")
            md.append("\n")

        md.append("---\n\n")

    return "".join(md)

def generate_offline_study_notes_from_index(
    slides: List[Dict[str, Any]],
    transcript: List[Dict[str, Any]],
    start_index: int,
) -> str:
    """Generate deterministic fallback only for visual contexts not yet synthesized."""
    aligned_data = align_content_by_slides(slides, transcript)
    remaining = aligned_data[max(0, start_index):]
    if not remaining:
        return ""

    md = []
    md.append("## Deterministic Completion\n\n")
    md.append("> Local AI synthesis stopped early, so EchoNotes completed only the remaining visual contexts with the offline NLP pipeline.\n\n")

    for offset, item in enumerate(remaining, start=max(0, start_index) + 1):
        md.append(generate_offline_slide_block(item, offset))
        md.append("\n\n---\n\n")

    return "".join(md)

def compress_text_for_llm(text: str, max_words: int = 180) -> str:
    """
    Compresses long speech transcripts to keep the prompt highly dense and concise.
    Filters out filler words and focuses strictly on sentences containing core OS technical vocabulary.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
        
    # Split text into sentences using standard regex boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    selected_sentences = []
    
    # Check each sentence against the technical term vocabulary list
    for sentence in sentences:
        has_tech = any(term in sentence.lower() for term in TECH_VOCABULARY)
        if has_tech:
            selected_sentences.append(sentence)
            
    # Fallback to cropping if no technical sentences were found
    if not selected_sentences:
        half = max_words // 2
        return " ".join(words[:half]) + " ... [Cắt bớt lời thoại phụ] ... " + " ".join(words[-half:])
        
    # Reassemble and limit to maximum word boundary
    joined = " ".join(selected_sentences)
    joined_words = joined.split()
    if len(joined_words) > max_words:
        return " ".join(joined_words[:max_words]) + " ... [Cắt bớt lời thoại phụ]"
        
    return joined

def convert_local_images_to_base64(markdown_text: str) -> str:
    """
    Finds all markdown images pointing to local filesystem paths,
    converts them to base64 data URIs, and replaces them on-the-fly.
    This bypasses browser security restrictions blocking absolute local paths!
    """
    if not markdown_text:
        return markdown_text
        
    from pathlib import Path
    
    # Regex to match ![alt_text](image_path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_match(match):
        alt_text = match.group(1)
        img_path_str = match.group(2).strip()
        
        # Clean path quotes if any
        if img_path_str.startswith('"') and img_path_str.endswith('"'):
            img_path_str = img_path_str[1:-1]
        elif img_path_str.startswith("'") and img_path_str.endswith("'"):
            img_path_str = img_path_str[1:-1]
            
        img_path = Path(img_path_str)
        
        if img_path.exists() and img_path.is_file():
            return image_to_markdown_data_uri(img_path, alt_text) or ""
        return ""
        
    return re.sub(pattern, replace_match, markdown_text)

def image_to_markdown_data_uri(image_path: str, alt_text: str = "Slide Screenshot") -> str:
    if not image_path:
        return ""
    import base64
    from pathlib import Path
    img_path = Path(image_path)
    if not img_path.exists() or not img_path.is_file():
        return ""
    try:
        ext = img_path.suffix.lower().replace(".", "")
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            return ""
        mime_type = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"
        with open(img_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        return f"![{alt_text}](data:{mime_type};base64,{b64_data})\n\n"
    except Exception:
        return ""

def markdown_image_or_empty(image_path: str, alt_text: str = "Slide Screenshot") -> str:
    if not image_path:
        return ""
    img_path = Path(image_path)
    if not img_path.exists() or not img_path.is_file():
        return ""
    return f"![{alt_text}]({img_path})\n\n"

def build_grounded_visual_analysis(item: Dict[str, Any], max_bullets: int = 4) -> List[str]:
    """Create deterministic keyframe context bullets grounded in extracted text and nearby speech."""
    ocr_text = (item.get("ocr_text") or "").strip()
    vlm_text = (item.get("vlm_description") or "").strip()
    speech_text = (item.get("speech_text") or "").strip()
    combined = f"{ocr_text}\n{vlm_text}\n{speech_text}".strip()
    keywords = extract_keywords_simple(combined, limit=10)

    bullets: List[str] = []
    if vlm_text:
        bullets.append(f"Visual AI interpretation: {vlm_text[:260]}")
    if ocr_text:
        ocr_lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
        visual_terms = ", ".join(f"`{kw}`" for kw in keywords[:6]) if keywords else "visible UI/text elements"
        bullets.append(f"The captured keyframe contains extracted text/context around {visual_terms}.")
        if ocr_lines:
            bullets.append(f"Most visible slide/screen text: {ocr_lines[0][:180]}.")
    elif not vlm_text:
        bullets.append("This keyframe is retained as visual reference; no detailed slide text was extracted in the current fast visual mode.")

    if speech_text:
        summaries = summarize_chunk_offline(speech_text, max_sentences=max_bullets)
        for summary in summaries[: max(1, max_bullets - len(bullets))]:
            bullets.append(summary)

    return bullets[:max_bullets] or ["No reliable visual or transcript evidence is available for this frame."]

def generate_offline_slide_block(item: Dict[str, Any], slide_num: int) -> str:
    """
    Helper to generate a beautiful markdown block for a single slide in offline fallback mode.
    """
    md = []
    md.append(f"## 🖼️ Slide {slide_num} - Xuất hiện tại [{item['timestamp_formatted']}]\n\n")
    md.append(markdown_image_or_empty(item.get("image_path", "")))
    
    if item["ocr_text"]:
        md.append("### 🔍 Nội dung hiển thị trên Slide (OCR Text)\n")
        md.append(f"```text\n{item['ocr_text'].strip()}\n```\n\n")
        
    if item["speech_text"]:
        md.append("### 🎓 Tóm tắt lời giảng tương ứng của giáo viên\n")
        md.append(f"{item['speech_text'].strip()}\n\n")
        
    if item["highlights"]:
        md.append("### 🔥 Điểm nhấn giọng nói đặc biệt cần lưu ý\n")
        for hl in item["highlights"]:
            md.append(f"- **⏱️ [{hl['timestamp']}] ({hl['type']})**: *\"{hl['quote']}\"*\n")
        md.append("\n")
        
    return "".join(md)

def generate_offline_slide_block(item: Dict[str, Any], slide_num: int) -> str:
    """Grounded fallback block for one visual context."""
    md = []
    md.append(f"## Visual Context {slide_num} - {item.get('timestamp_formatted', '')}\n\n")
    md.append(markdown_image_or_empty(item.get("image_path", "")))

    md.append("### Keyframe Context\n")
    for bullet in build_grounded_visual_analysis(item):
        md.append(f"- {bullet}\n")
    md.append("\n")

    if item.get("vlm_description"):
        md.append("### Visual Image Understanding\n")
        md.append(f"{item['vlm_description'].strip()}\n\n")

    if item.get("speech_text"):
        md.append("### Lecture Interpretation\n")
        for bullet in summarize_chunk_offline(item["speech_text"], max_sentences=3):
            md.append(f"- {bullet}\n")
        md.append("\n")

    if item.get("ocr_text"):
        md.append("### Extracted Visual Evidence\n")
        md.append(f"```text\n{item['ocr_text'].strip()[:1200]}\n```\n\n")

    if item.get("highlights"):
        md.append("### Acoustic Highlights\n")
        for hl in item["highlights"]:
            md.append(f"- **[{hl['timestamp']}] ({hl['type']})**: \"{hl['quote']}\"\n")
        md.append("\n")

    if not item.get("ocr_text") and not item.get("speech_text"):
        md.append("- No reliable visual or transcript content is available for this frame.\n\n")

    return "".join(md)

def is_meeting_interface_slide(ocr_text: str) -> bool:
    """
    Detects if the slide is just a MS Teams / Zoom / Webex participant screen, lobby or waiting room.
    Checks for high density of initials, common student names, and absolute lack of technical keywords.
    """
    if not ocr_text:
        return False
        
    ocr_lower = ocr_text.lower()
    
    # Common meeting signatures
    meeting_words = [
        "microsoft teams", "recorded by", "organized by", "meeting in", 
        "waiting room", "lobby", "participants", "unmute", "share screen"
    ]
    if any(word in ocr_lower for word in meeting_words):
        return True
        
    words = ocr_text.split()
    if len(words) < 5:
        return False
        
    # Check for initials pattern (e.g. "NH HD NK CP VS" - two-letter uppercase words)
    initials_count = sum(1 for w in words if len(w) == 2 and w.isupper() and w.isalpha())
    initials_ratio = initials_count / len(words)
    
    # Common Vietnamese name tokens in lowercase
    name_tokens = {
        "nguyen", "nguyễn", "tran", "trần", "le", "lê", "pham", "phạm", 
        "hoang", "hoàng", "huy", "vu", "vũ", "do", "đỗ", "ha", "hà", 
        "bao", "bảo", "long", "son", "sơn", "truong", "trương", "viet", "việt", 
        "duc", "đức", "tri", "trí", "danh", "huong", "hương", "dinh", "định",
        "phat", "phát", "loi", "lợi", "nhat", "nhật", "thu", "thư", "lam", "lâm",
        "phong", "hai", "hải", "dang", "đăng", "nien", "niên", "huynh", "huỳnh",
        "ngoc", "ngọc", "thao", "thảo", "vuong", "vương", "quan", "quân", "sieu", 
        "siêu", "trinh", "trịnh", "phuc", "phúc", "thinh", "thịnh", "minh", 
        "khoi", "khôi", "xuan", "xuân", "dung", "dũng", "hien", "hiền"
    }
    
    name_count = sum(1 for w in words if w.lower() in name_tokens)
    name_ratio = name_count / len(words)
    
    # Common technical terms to avoid false positives on technical text
    tech_keywords = {
        "inode", "pointer", "block", "file", "system", "memory", "cpu", "process", 
        "thread", "allocation", "direct", "indirect", "directory", "struct", "class", 
        "function", "variable", "code", "programming", "database", "table", "query"
    }
    tech_count = sum(1 for w in words if w.lower() in tech_keywords)
    
    if (initials_ratio > 0.15 or name_ratio > 0.35) and tech_count == 0:
        return True
        
    return False

def generate_smart_notes_stream(
    slides: List[Dict[str, Any]], 
    segments: List[Dict[str, Any]], 
    model_name: str = OLLAMA_DEFAULT_MODEL
) -> Generator[str, None, None]:
    """
    Sends the aligned slide and acoustic context to local Ollama.
    Streams the structured AI Smart Notes response in real-time, slide-by-slide,
    providing highly robust, detailed and fast compilation without timeouts or small LLM looping.
    """
    aligned_data = align_content_by_slides(slides, segments)
    note_language = infer_note_language(segments)

    if not has_visual_context(slides):
        chunks = chunk_segments_by_time(segments)
        all_text = " ".join((s.get("text") or "").strip() for s in segments or [])
        top_keywords = extract_keywords_simple(all_text, limit=18)

        yield "# EchoNotes AI - NLP Smart Notes\n\n"
        yield f"*Transcript-first synthesis using local model: `{model_name}`*\n\n"
        if top_keywords:
            yield "**Detected keywords:** " + ", ".join(f"`{kw}`" for kw in top_keywords) + "\n\n"
        yield "---\n\n"

        for idx, chunk in enumerate(chunks, start=1):
            yield f"## {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])} | Topic Block {idx}\n\n"

            system_prompt = (
                f"You are an NLP learning assistant. Write concise, professional {note_language} study notes from a timestamped lecture transcript.\n"
                "Focus on topic segmentation, key concepts, terminology, learning outcomes, and possible review questions.\n"
                "Do not mention slides, screenshots, OCR, or Teams intro because this run is transcript-only.\n"
                "Return Markdown only with these sections: Summary, Key Terms, What To Remember, Review Questions.\n"
                "Do not hallucinate content that is not supported by the transcript.\n"
                "Never output Chinese characters. If unsure, stay in English."
            )
            user_context = (
                f"Transcript time window: {format_timestamp(chunk['start'])} - {format_timestamp(chunk['end'])}\n"
                f"Raw transcript:\n{chunk['text'][:6000]}"
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_context}
                ],
                "stream": True,
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.2,
                    "repeat_penalty": 1.35,
                    "repeat_last_n": 128,
                    "presence_penalty": 0.4,
                    "frequency_penalty": 0.4
                }
            }

            try:
                response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload, stream=True, timeout=(15, 90))
                if response.status_code != 200:
                    yield "\n\n*[Ollama failed for this block. Using offline NLP fallback.]*\n\n"
                    for bullet in summarize_chunk_offline(chunk["text"], max_sentences=4):
                        yield f"- {bullet}\n"
                    yield "\n\n---\n\n"
                    continue

                block_text = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    response_chunk = json.loads(line.decode("utf-8"))
                    response_text = response_chunk.get("message", {}).get("content", "")
                    block_text += response_text
                    if contains_cjk(block_text):
                        yield "\n\n*[Model output contained unsupported CJK text and was discarded for this block. Using offline fallback.]*\n\n"
                        for bullet in summarize_chunk_offline(chunk["text"], max_sentences=4):
                            yield f"- {bullet}\n"
                        break
                    if len(block_text) > 2200:
                        yield "\n\n*[Output limit reached for this topic block.]*\n"
                        break
                    yield response_text
                yield "\n\n---\n\n"
            except Exception:
                yield "\n\n*[Ollama timed out. Using offline NLP fallback.]*\n\n"
                for bullet in summarize_chunk_offline(chunk["text"], max_sentences=4):
                    yield f"- {bullet}\n"
                yield "\n\n---\n\n"
        return

    yield "# EchoNotes AI - Smart Lecture Notes\n\n"
    yield f"*Generated with local AI model: `{model_name}`*\n\n"
    yield "---\n\n"
    
    total_slides = len(aligned_data)
    
    for idx, item in enumerate(aligned_data):
        slide_num = idx + 1
        
        # State-Aware Slide Classification
        dense_speech = compress_text_for_llm(item['speech_text'], max_words=180)
        lower_ocr = item['ocr_text'].lower() if item['ocr_text'] else ""
        
        # Detect Intro / Setup slides (usually Slide 1 or empty Teams screens)
        is_intro = False
        if not dense_speech.strip():
            if "microsoft teams" in lower_ocr or "meeting in" in lower_ocr or "recorded by" in lower_ocr or len(lower_ocr) < 40:
                is_intro = True
                
        # Detect Real Silent slides (meaning no lecture audio is available for this frame)
        is_silent = not dense_speech.strip()
        
        if is_intro:
            # Bypasses the LLM completely to save compute and prevent Slide 1 loops
            yield f"## Visual Context {slide_num}: Meeting Setup\n\n"
            yield markdown_image_or_empty(item.get("image_path", ""))
            yield "### Content\n"
            yield "- Meeting lobby or screen-sharing setup. No substantive lecture content is available for this moment.\n\n"
            yield "### Practical Notes\n"
            yield "- Skip this section unless you need meeting context.\n\n"
            yield "\n\n---\n\n"
            continue
            
        # Dynamically intercept and handle MS Teams participant gallery list screens
        if is_meeting_interface_slide(item['ocr_text']):
            yield f"## Visual Context {slide_num}: Meeting Interface\n\n"
            yield markdown_image_or_empty(item.get("image_path", ""))
            yield "### Content\n"
            yield "- Microsoft Teams participant or meeting interface. This frame is not a lecture slide.\n"
            if item['ocr_text']:
                yield f"- **Detected interface text:** `{item['ocr_text'].strip()}`\n"
            yield "- **Status:** No technical visual material is available in this frame.\n\n"
            
            if item['speech_text']:
                yield "### Transcript Context\n"
                yield f"{item['speech_text'].strip()}\n\n"
            else:
                yield "### Key Signal\n"
                yield "- No substantive lecture transcript is aligned to this frame.\n\n"
                
            yield "### Practical Notes\n"
            yield "- Treat this as meeting UI context, not learning content.\n\n"
            yield "\n\n---\n\n"
            continue

        # Insert the visual header and screenshot outside the LLM so the model cannot invent image paths.
        yield f"## Visual Context {slide_num}: Deep Synthesis\n\n"
        yield markdown_image_or_empty(item.get("image_path", ""))
        
        if is_silent:
            # Uses a specialized simple prompt focusing only on OCR text and strictly forbidding teacher content hallucination
            system_prompt = (
                "You are an AI learning assistant.\n"
                f"Write concise, professional {note_language} notes for a silent visual context {slide_num}/{total_slides}.\n\n"
                "Important: no teacher transcript is available for this frame. Use only extracted visual text if present.\n\n"
                "Required Markdown sections:\n"
                "### Visual Content\n"
                "   - Analyze the technical terms, structures, and processes that actually appear in the extracted visual text. Use only terms supported by the data; do not force a fixed domain.\n"
                "### Practical Notes\n"
                "   - Explain only useful implementation or learning notes supported by the available visual/text evidence.\n\n"
                "Strict rules:\n"
                "- Never output Chinese characters.\n"
                "- Do not invent teacher speech or highlights.\n"
                "- Do not mention exams unless the transcript explicitly mentions an exam."
            )
            
            user_context = (
                f"Data for visual context {slide_num}:\n"
                f"- **Extracted visual text:** \"{item['ocr_text']}\"\n"
                f"- **Visual image understanding:** \"{item.get('vlm_description', '')}\"\n"
                "- **Transcript:** No teacher speech is aligned to this frame.\n"
            )
            
        else:
            # Rich lecture slide prompt
            system_prompt = (
                "You are an AI learning assistant.\n"
                f"Write concise, professional {note_language} notes for visual context {slide_num}/{total_slides}.\n\n"
                "Required Markdown sections:\n"
                "### Summary\n"
                "   - Explain the slide/transcript content using only supported evidence.\n"
                "### Key Concepts\n"
                "   - Extract important technical concepts, identifiers, commands, tables, fields, or workflows.\n"
                "### Practical Notes\n"
                "   - Include implementation details, mistakes to avoid, or follow-up actions. Do not mention exams unless the transcript explicitly mentions an exam.\n\n"
                "Strict rules:\n"
                f"- Output language: {note_language}.\n"
                "- Never output Chinese characters. If unsure, answer in English.\n"
                "- Do not hallucinate details not supported by extracted visual text or transcript.\n"
                "- Do not add slide titles; go directly into the required sections."
            )
            
            user_context = (
                f"Data for visual context {slide_num}:\n"
                f"- **Extracted visual text:** \"{item['ocr_text']}\"\n"
                f"- **Visual image understanding:** \"{item.get('vlm_description', '')}\"\n"
                f"- **Aligned transcript:** \"{dense_speech}\"\n"
            )
            
            if item["highlights"]:
                user_context += "- **Acoustic highlights:**\n"
                for hl in item["highlights"]:
                    user_context += f"  * At [{hl['timestamp']}] ({hl['type']}): \"{hl['quote']}\"\n"
                    
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_context}
            ],
            "stream": True,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.3,
                "repeat_penalty": 1.4, # Strongly penalize repetitions for 1.5B models
                "repeat_last_n": 128,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.5,
                "stop": ["## Slide", "## 🖼️ Slide", "---", "### SLIDE", "Slide "] # Strict stop tokens
            }
        }
        
        try:
            # Query via chat API (using chat instead of generate to apply native model templates)
            response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload, stream=True, timeout=(15, 120))
            if response.status_code != 200:
                yield f"\n\n*Local AI returned HTTP {response.status_code}. Using grounded offline notes for this visual context.*\n\n"
                yield generate_offline_slide_block(item, slide_num)
                continue
                
            slide_text = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    response_text = chunk.get("message", {}).get("content", "")
                    
                    slide_text += response_text
                    
                    # On-the-fly repetition cut-off filter
                    if len(slide_text) > 120:
                        has_loop = False
                        n = len(slide_text)
                        
                        # Hard output safety limit: if AI writes excessively for a single slide, cut it off to protect data
                        if n > 2500:
                            yield "\n\n*Local AI output was truncated for this visual context because it exceeded the safety limit.*\n\n"
                            break
                            
                        # Check for repeating consecutive blocks of length 60 to 1200 characters (can be multi-paragraph)
                        for k in range(60, min(1200, n // 2) + 1):
                            p1 = slide_text[-k:]
                            p2 = slide_text[-2*k:-k]
                            if p1 == p2:
                                has_loop = True
                                break
                        if has_loop:
                            yield "\n\n*Local AI output was stopped because repetition was detected.*\n\n"
                            break
                            
                    if chunk.get("done", False):
                        break

            cleaned_slide_text = strip_cjk_lines(slide_text)
            if not cleaned_slide_text.strip() or contains_cjk(cleaned_slide_text):
                yield "*Model output was discarded because it contained unsupported Chinese text. Using grounded fallback notes instead.*\n\n"
                if dense_speech:
                    for bullet in summarize_chunk_offline(item["speech_text"], max_sentences=4):
                        yield f"- {bullet}\n"
                elif item.get("vlm_description"):
                    yield f"### Visual Image Understanding\n\n{item['vlm_description'].strip()}\n"
                elif item.get("ocr_text"):
                    yield f"### Visual Content\n\n{item['ocr_text'].strip()}\n"
                else:
                    yield "- No reliable visual or transcript content is available for this frame.\n"
            else:
                yield cleaned_slide_text
            yield "\n\n---\n\n"
            
        except Exception as e:
            yield f"\n\n*Could not reach local AI for visual context {slide_num}: {e}. Using grounded offline notes.*\n\n"
            yield generate_offline_slide_block(item, slide_num)
            yield "\n\n---\n\n"

def post_process_transcript_with_llm(
    segments: List[Dict[str, Any]], 
    model_name: str = OLLAMA_DEFAULT_MODEL
) -> List[Dict[str, Any]]:
    """
    Level 3 Post-ASR Correction: Groups transcript segments into paragraphs, 
    sends them to the local Ollama LLM to correct phonetic Vinglish errors, 
    fix grammar, and restore technical terminology on the fly.
    """
    if not segments:
        return segments
        
    import copy
    corrected_segments = copy.deepcopy(segments)
    
    # Group segments into chunks of 8 sentences to maintain context while keeping it fast
    chunk_size = 8
    num_segments = len(segments)
    
    # Simple system prompt for precise correction
    system_prompt = (
        "Bạn là một biên tập viên kỹ thuật chuyên nghiệp. Nhiệm vụ của bạn là đọc đoạn văn bản "
        "lời giảng thô (được chuyển đổi từ giọng nói ASR) và sửa lại các lỗi chính tả, lỗi phát âm sai "
        "thuật ngữ tiếng Anh-Việt (Vinglish) sang thuật ngữ tiếng Anh chuẩn xác (như: block, inode, pointer, single indirect, "
        "double, triple, file, partition...) và chỉnh sửa ngữ pháp tiếng Việt cho trôi chảy, chuyên nghiệp.\n\n"
        "YÊU CẦU CỰC KỲ NGHIÊM NGẶT:\n"
        "- BẮT BUỘC TRẢ LỜI BẰNG TIẾNG VIỆT 100%. Tuyệt đối KHÔNG sử dụng tiếng Trung (Chinese, chữ Hán) dưới bất kỳ hình thức nào. DO NOT OUTPUT CHINESE.\n"
        "- Chỉ trả về đoạn văn bản đã được chỉnh sửa, KHÔNG thêm lời giải thích, KHÔNG thêm lời mở đầu hay kết thúc.\n"
        "- Giữ nguyên ý nghĩa gốc của bài giảng."
    )
    
    for i in range(0, num_segments, chunk_size):
        chunk = corrected_segments[i:i + chunk_size]
        raw_text = " ".join([seg["text"] for seg in chunk])
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Đoạn văn bản cần sửa:\n\"{raw_text}\""}
            ],
            "stream": False,  # Non-streaming for clean single-shot return
            "options": {
                "num_ctx": 4096,
                "temperature": 0.3,
                "repeat_penalty": 1.25,
                "repeat_last_n": 64,
                "presence_penalty": 0.3,
                "frequency_penalty": 0.3
            }
        }
        
        try:
            response = requests.post(f"{OLLAMA_API_URL}/chat", json=payload, timeout=45)
            if response.status_code == 200:
                corrected_text = response.json().get("message", {}).get("content", "").strip()
                if corrected_text:
                    # Distribute the corrected text back to the chunk segments
                    corrected_sentences = re.split(r'(?<=[.!?])\s+', corrected_text)
                    
                    # Align sentences back to segments
                    for idx, seg in enumerate(chunk):
                        if idx < len(corrected_sentences):
                            seg["text"] = corrected_sentences[idx]
                            
                    # Append any leftover sentences to the last segment in the chunk
                    if len(corrected_sentences) > len(chunk):
                        chunk[-1]["text"] += " " + " ".join(corrected_sentences[len(chunk):])
        except Exception as e:
            # Fallback: if LLM fails, we keep the original text
            print(f"Error correcting chunk: {e}")
            
    return corrected_segments
