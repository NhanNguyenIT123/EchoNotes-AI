import os
import json
import time
import re
import html
import shutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Generator

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import project configurations and NLP/Vision utilities
from app.config import (
    RAW_DIR, OUTPUTS_DIR, FRAMES_DIR, TRANSCRIPTS_DIR, DATA_DIR,
    WHISPER_MODEL_DEFAULT, OLLAMA_DEFAULT_MODEL, SSIM_THRESHOLD, FRAME_CHECK_INTERVAL
)
from app.db import (
    add_chat_message, add_evaluation_record, create_project_for_video, db_health,
    get_latest_project_for_video, get_project_payload, init_db, list_chat_messages,
    list_evaluation_records, list_projects, rate_chat_message, save_project_artifacts,
    update_project_input, update_project_metadata
)
from app.utils.video import extract_audio_from_video
from app.utils.audio import transcribe_audio
from app.utils.acoustic import analyze_audio_acoustics
from app.utils.vision import detect_slide_transitions
from app.utils.llm import (
    get_installed_ollama_models, generate_smart_notes_stream,
    generate_offline_study_notes, convert_local_images_to_base64,
    generate_offline_study_notes_from_index, chunk_segments_by_time,
    extract_keywords_simple, summarize_chunk_offline
)
from app.utils.corrector import corrector, correct_transcript_segment
from app.utils.dataset_factory import generate_whisper_dataset
from app.utils.model_sync import download_file_from_google_drive, extract_and_install_zip, auto_detect_google_drive_paths
from app.utils.teams_transcript import parse_teams_transcript
from app.utils.graph_import import complete_device_flow, create_device_flow, download_teams_recording_assets, get_default_graph_config
from app.utils.langchain_rag import answer_with_langchain_rag
from app.utils.learning_exports import write_anki_tsv, write_quiz_json
from app.utils.pdf_export import export_notes_pdf
from app.utils.topic_segmentation import build_semantic_topic_blocks
from app.utils.evaluation import evaluate_transcript_quality
from app.utils.diarization import apply_optional_diarization, infer_speaker_roles
from app.utils.vlm_benchmark import build_vlm_benchmark
from app.utils.vlm import enrich_keyframes_with_vlm
from app.job_queue import (
    configured_job_backend, create_job, enqueue_rq_job,
    read_job_status, update_job_status
)
from app.storage import get_storage_backend

app = FastAPI(title="EchoNotes AI - Backend API")

# Enable CORS for React dev server running on port 5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Paths matching app/ui.py
enriched_cache_path = TRANSCRIPTS_DIR / "enriched_segments_cache.json"
slides_cache_path = TRANSCRIPTS_DIR / "slides_cache.json"
topic_blocks_cache_path = TRANSCRIPTS_DIR / "topic_blocks_cache.json"
smart_notes_cache_path = TRANSCRIPTS_DIR / "smart_notes_cache.md"
pdf_text_cache_path = TRANSCRIPTS_DIR / "report_pdf_text_cache.txt"
regression_set_path = DATA_DIR / "regression" / "echonotes_smoke.json"

# Global workstation state
state = {
    "status": "idle",               # idle, processing, completed, error
    "stage": "",                    # Text indicating active step
    "progress": 0,                  # Progress bar percentage 0 - 100
    "error": None,
    "active_video_path": None,
    "active_video_name": None,
    "active_transcript_path": None,
    "active_project_id": None,
    "database": {"connected": False},
    
    # Teams Integration state
    "graph_device_flow": None,
    "graph_access_token": None,
    "teams_link_video_path": None,
    "teams_link_transcript_path": None,
    
    # Report compilation state
    "smart_notes": "",
    "generating_report": False,
    "report_started_at": None,
    
    # Dataset generation state
    "dataset_status": "idle",       # idle, exporting, completed, error
    "dataset_progress": 0,
    "last_metrics": {},
    "active_job_id": None,
    "storage_artifacts": {},
}

# Thread lock for pipeline execution
pipeline_lock = threading.Lock()

try:
    init_db()
    state["database"] = db_health()
except Exception as exc:
    state["database"] = {"connected": False, "error": str(exc)}

def clear_result_caches():
    """Remove processed outputs that should not leak across videos."""
    for cache_file in [enriched_cache_path, slides_cache_path, topic_blocks_cache_path, smart_notes_cache_path, pdf_text_cache_path]:
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass

def load_initial_cache():
    """Restores the latest finished processing state from cache if available."""
    global state
    raw_videos = sorted(
        list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mkv")) + list(RAW_DIR.glob("*.avi")) + list(RAW_DIR.glob("*.mov")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if raw_videos:
        state["active_video_path"] = str(raw_videos[0])
        state["active_video_name"] = raw_videos[0].name

    if enriched_cache_path.exists() and slides_cache_path.exists():
        state["status"] = "completed"
        state["stage"] = "Loaded from cache"
        state["progress"] = 100
        
        # Load cached report markdown if exists
        if smart_notes_cache_path.exists():
            with open(smart_notes_cache_path, "r", encoding="utf-8") as f:
                state["smart_notes"] = f.read()

def markdown_for_client(markdown_text: str) -> str:
    """Convert local markdown image paths to backend-served URLs for the browser."""
    def replace_image(match):
        alt_text = match.group(1)
        target = (match.group(2) or "").strip().strip('"').strip("'")
        if target.startswith("data:image/") or target.startswith("http://") or target.startswith("https://"):
            return f"![{alt_text}]({target})"

        image_path = Path(target)
        if image_path.exists() and image_path.is_file():
            return f"![{alt_text}](http://localhost:8000/api/keyframes/{image_path.name})"

        return ""

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, markdown_text or "")


def sync_storage_artifact(local_path: Path, object_name: str) -> Optional[str]:
    try:
        if not local_path.exists():
            return None
        url = get_storage_backend().put_file(local_path, object_name)
        state.setdefault("storage_artifacts", {})[object_name] = url
        return url
    except Exception as exc:
        state.setdefault("storage_artifacts", {})[object_name] = f"storage sync failed: {exc}"
        return None

load_initial_cache()

# --- BACKGROUND WORKER PIPELINE ---

def run_pipeline_thread(
    video_path: Path,
    whisper_model: str,
    initial_prompt: str,
    teams_transcript_path: Optional[Path],
    hotwords: str,
    use_glossary: bool,
    visual_mode: str,
    ssim_thresh: float,
    min_keyframe_gap_sec: float,
    max_keyframes: int,
    frame_check_interval_sec: float,
    analyze_acoustics: bool,
    speech_language: str,
    vision_model: str,
    project_id: Optional[str] = None,
    diarization_enabled: bool = False,
    job_id: Optional[str] = None,
):
    global state
    try:
        run_started = time.perf_counter()
        stage_times: Dict[str, float] = {}
        use_teams_transcript = bool(teams_transcript_path and teams_transcript_path.exists())
        needs_audio = (not use_teams_transcript) or analyze_acoustics or diarization_enabled
        
        # Stage 1: Audio extraction
        state["status"] = "processing"
        state["stage"] = "Extracting 16kHz audio from the video..."
        state["progress"] = 10
        if job_id:
            update_job_status(job_id, status="processing", stage=state["stage"], progress=10)
        audio_path = None
        stage_start = time.perf_counter()
        if needs_audio:
            audio_path = extract_audio_from_video(video_path)
        stage_times["audio_extraction_sec"] = round(time.perf_counter() - stage_start, 3)
        time.sleep(0.5)
        
        # Stage 2: Transcribe
        state["stage"] = "Transcribing speech to text..."
        state["progress"] = 25
        if job_id:
            update_job_status(job_id, status="processing", stage=state["stage"], progress=25)
        
        stage_start = time.perf_counter()
        if use_teams_transcript:
            state["stage"] = "Importing Teams transcript..."
            teams_segments = parse_teams_transcript(teams_transcript_path)
            raw_transcript = {
                "metadata": {
                    "source": "microsoft_teams_transcript",
                    "requested_language": "provided_by_teams",
                    "detected_language": "provided_by_teams",
                    "transcript_file": str(teams_transcript_path),
                    "duration": round(max((seg["end"] for seg in teams_segments), default=0.0), 2),
                },
                "segments": teams_segments,
                "full_text": " ".join(seg["text"] for seg in teams_segments)
            }
        else:
            def whisper_progress(current_sec, total_sec):
                percent = min(1.0, current_sec / total_sec) if total_sec > 0 else 0.0
                state["progress"] = int(25 + percent * 30)
                m_curr, s_curr = int(current_sec // 60), int(current_sec % 60)
                m_tot, s_tot = int(total_sec // 60), int(total_sec % 60)
                state["stage"] = f"Whisper transcribing: {m_curr:02d}:{s_curr:02d} / {m_tot:02d}:{s_tot:02d} ({percent * 100:.1f}%)"
                if job_id:
                    update_job_status(job_id, status="processing", stage=state["stage"], progress=state["progress"])
            
            raw_transcript = transcribe_audio(
                audio_path,
                model_size=whisper_model,
                progress_callback=whisper_progress,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                language=speech_language
            )
        stage_times["asr_sec"] = round(time.perf_counter() - stage_start, 3)
        
        # Stage 3: Acoustic labels
        state["stage"] = "Analyzing acoustic signals and emphasis..."
        state["progress"] = 60
        if job_id:
            update_job_status(job_id, status="processing", stage=state["stage"], progress=60)
        stage_start = time.perf_counter()
        if analyze_acoustics and audio_path:
            enriched_segments = analyze_audio_acoustics(audio_path, raw_transcript["segments"])
            asr_metadata = raw_transcript.get("metadata", {})
            for seg in enriched_segments:
                seg["asr_metadata"] = asr_metadata
        else:
            enriched_segments = []
            asr_metadata = raw_transcript.get("metadata", {})
            for seg in raw_transcript["segments"]:
                copied = seg.copy()
                copied["asr_metadata"] = asr_metadata
                copied["acoustics"] = {
                    "volume_rms": 0.0,
                    "pitch_hz": 0.0,
                    "speech_rate_wps": 0.0,
                    "is_loud": False,
                    "is_high_pitch": False,
                    "is_slow": False,
                    "emphasis_score": 0,
                    "has_semantic_clue": False,
                    "is_important": False
                }
                enriched_segments.append(copied)
        stage_times["acoustic_sec"] = round(time.perf_counter() - stage_start, 3)
        time.sleep(0.5)

        # Optional speaker diarization. Uses pyannote when configured, otherwise
        # attaches heuristic role metadata without failing the pipeline.
        stage_start = time.perf_counter()
        diarization_metadata = {"enabled": bool(diarization_enabled), "status": "skipped"}
        if diarization_enabled and audio_path:
            state["stage"] = "Running speaker diarization..."
            state["progress"] = 70
            if job_id:
                update_job_status(job_id, status="processing", stage=state["stage"], progress=70)
            diarized = apply_optional_diarization(Path(audio_path), enriched_segments)
            enriched_segments = diarized.get("segments", enriched_segments)
            diarization_metadata = diarized.get("metadata", diarization_metadata)
        stage_times["diarization_sec"] = round(time.perf_counter() - stage_start, 3)
        
        # Stage 4: Visuals
        state["stage"] = f"Analyzing visual keyframes ({visual_mode})..."
        state["progress"] = 80
        if job_id:
            update_job_status(job_id, status="processing", stage=state["stage"], progress=80)
        stage_start = time.perf_counter()
        if visual_mode.startswith("Transcript only"):
            slide_keyframes = [{
                "timestamp_sec": 0.0,
                "timestamp_formatted": "00:00:00",
                "image_path": "",
                "ocr_text": ""
            }]
        else:
            slide_keyframes = detect_slide_transitions(
                video_path,
                progress_callback=None,
                ssim_threshold=ssim_thresh,
                frame_check_interval=float(frame_check_interval_sec),
                min_transition_gap_sec=float(min_keyframe_gap_sec),
                run_ocr=visual_mode.startswith("Full"),
                max_slides=int(max_keyframes)
            )

            if visual_mode.startswith("VLM"):
                state["stage"] = f"Running visual image understanding ({vision_model})..."
                if job_id:
                    update_job_status(job_id, status="processing", stage=state["stage"], progress=86)
                slide_keyframes = enrich_keyframes_with_vlm(
                    slide_keyframes,
                    enriched_segments,
                    model_name=vision_model,
                    max_frames=min(int(max_keyframes), 12),
                )
        stage_times["visual_sec"] = round(time.perf_counter() - stage_start, 3)
        time.sleep(0.5)
        
        # Stage 5: Finalization & Cache Save
        state["stage"] = "Synchronizing transcript and slides..."
        state["progress"] = 95
        if job_id:
            update_job_status(job_id, status="processing", stage=state["stage"], progress=95)
        
        # Apply glossary corrections on the fly
        cleaned_enriched_segments = [
            correct_transcript_segment(seg, use_glossary=use_glossary, use_fuzzy=use_glossary)
            for seg in enriched_segments
        ]
        cleaned_slide_keyframes = []
        for slide in slide_keyframes:
            if use_glossary and "ocr_text" in slide and slide["ocr_text"]:
                slide["ocr_text"] = corrector.correct_text(slide["ocr_text"])
            cleaned_slide_keyframes.append(slide)

        topic_blocks = build_semantic_topic_blocks(cleaned_enriched_segments)
            
        with open(enriched_cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_enriched_segments, f, ensure_ascii=False, indent=2)
        with open(topic_blocks_cache_path, "w", encoding="utf-8") as f:
            json.dump(topic_blocks, f, ensure_ascii=False, indent=2)
        with open(slides_cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_slide_keyframes, f, ensure_ascii=False, indent=2)

        metrics = {
            **stage_times,
            "total_pipeline_sec": round(time.perf_counter() - run_started, 3),
            "transcript_segments": len(cleaned_enriched_segments),
            "semantic_topic_blocks": len(topic_blocks),
            "keyframes": len(cleaned_slide_keyframes),
            "transcript_source": raw_transcript.get("metadata", {}).get("source", "whisper"),
            "visual_mode": visual_mode,
            "vision_model": vision_model if visual_mode.startswith("VLM") else None,
            "acoustic_enabled": analyze_acoustics,
            "diarization": diarization_metadata,
            "job_backend": configured_job_backend(),
        }
        state["last_metrics"] = metrics
        if project_id:
            try:
                save_project_artifacts(
                    project_id,
                    transcript=cleaned_enriched_segments,
                    topic_blocks=topic_blocks,
                    slides=cleaned_slide_keyframes,
                    metrics=metrics,
                    status="analyzed",
                )
            except Exception as db_exc:
                state["database"] = {"connected": False, "error": str(db_exc)}
            
        # Clean old report cache on new video processing run
        if smart_notes_cache_path.exists():
            smart_notes_cache_path.unlink()
            
        state["smart_notes"] = ""
        state["status"] = "completed"
        state["stage"] = "Analysis complete"
        state["progress"] = 100
        state["error"] = None
        if job_id:
            update_job_status(job_id, status="completed", stage=state["stage"], progress=100, metrics=metrics)
        
    except Exception as e:
        state["status"] = "error"
        state["stage"] = "Pipeline crashed"
        state["progress"] = 0
        state["error"] = str(e)
        if job_id:
            update_job_status(job_id, status="error", stage=state["stage"], progress=0, error=str(e))

# --- CHAT GROUNDED RETRIEVAL HELPERS (Ported from app/ui.py) ---

def _format_time(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"

def _retrieve_chat_evidence(question: str, notes: str, transcript_segments, max_segments: int = 10) -> str:
    question_l = (question or "").lower()
    query_terms = [
        term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", question_l)
        if term not in {
            "what", "who", "when", "where", "which", "with", "from", "this", "that",
            "like", "name", "does", "have", "about", "report", "lecture", "instructor",
            "teacher", "speaker", "course", "video", "his", "her", "the", "and", "for"
        }
    ]

    if any(word in question_l for word in ["instructor", "teacher", "speaker", "professor", "his name", "her name"]):
        query_terms.extend(["instructor", "teacher", "speaker", "professor", "name"])

    scored = []
    for seg in transcript_segments or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        text_l = text.lower()
        score = sum(2 for term in query_terms if term in text_l)
        if any(pattern in text_l for pattern in ["my name is", "i am ", "i'm ", "this is ", "instructor", "teacher", "professor"]):
            score += 3
        if any(pattern in text_l for pattern in ["hello", "can you see", "before we start", "welcome"]):
            score += 1
        if score:
            scored.append((score, float(seg.get("start", 0) or 0), text))

    scored.sort(key=lambda item: (-item[0], item[1]))
    evidence_lines = [f"[{_format_time(start)}] {text}" for _, start, text in scored[:max_segments]]

    if not evidence_lines and notes:
        # Grounding fallback using report lines
        note_lines = [line.strip() for line in notes.splitlines() if line.strip()]
        evidence_lines = note_lines[:12]

    return "\n".join(evidence_lines)

def _local_evidence_answer(question: str, evidence: str) -> str:
    q = (question or "").lower()
    if not evidence.strip():
        return "I do not have enough evidence in the current report/transcript to answer that."

    if any(word in q for word in ["instructor", "teacher", "speaker", "professor", "his name", "her name"]):
        name_patterns = [
            r"\b(?:my name is|i am|i'm|this is|teacher is|instructor is|professor is)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
            r"\b(?:Mr\.|Ms\.|Mrs\.|Dr\.|Prof\.)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, evidence)
            if match:
                return f"The transcript suggests the instructor/speaker may be **{match.group(1).strip()}**.\n\nEvidence:\n{evidence[:900]}"
        return "I cannot confidently identify the instructor's name from the current transcript. The relevant evidence I found is:\n\n" + evidence[:1200]

    return "Relevant evidence from the report/transcript:\n\n" + evidence[:1500]

def answer_notes_question(question: str, notes: str, transcript_segments, model_name: str) -> str:
    import requests as req_sync
    evidence = _retrieve_chat_evidence(question, notes, transcript_segments)
    local_fallback = _local_evidence_answer(question, evidence)

    context = (notes or "")[:2500] + "\n\nRelevant transcript evidence:\n" + evidence[:2500]
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided EchoNotes report/transcript context. "
                    "Answer in English unless the user asks another language. Never output Chinese characters. "
                    "If the answer is not in the context, say you do not have enough evidence."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 320},
    }
    try:
        res = req_sync.post("http://localhost:11434/api/chat", json=payload, timeout=(10, 180))
        if res.status_code == 200:
            answer = res.json().get("message", {}).get("content", "").strip()
            return answer or local_fallback
        return f"{local_fallback}\n\n_Local AI returned HTTP {res.status_code}, so EchoNotes used transcript retrieval instead._"
    except req_sync.exceptions.Timeout:
        return f"{local_fallback}\n\n_Local AI timed out, so EchoNotes used transcript retrieval instead._"
    except Exception as exc:
        return f"{local_fallback}\n\n_Local AI was unavailable: {exc}_"


# --- FastAPI ENDPOINTS ---

@app.get("/api/status")
def get_status():
    """Queries current workstation and pipeline state."""
    # Check if files physically exist to sync completed state if changed on disk
    if state["status"] == "idle" and enriched_cache_path.exists() and slides_cache_path.exists():
        load_initial_cache()
    active_job = read_job_status(state.get("active_job_id"))
    if active_job and active_job.get("status") in {"queued", "processing", "fallback"}:
        state["status"] = "processing"
        state["stage"] = active_job.get("stage") or state["stage"]
        state["progress"] = int(active_job.get("progress") or state["progress"] or 0)
        state["error"] = active_job.get("error")
    elif active_job and active_job.get("status") == "completed" and state["status"] == "processing":
        state["status"] = "completed"
        state["stage"] = active_job.get("stage", "Analysis complete")
        state["progress"] = 100
        state["last_metrics"] = active_job.get("metrics") or state.get("last_metrics", {})
    elif active_job and active_job.get("status") == "error":
        state["status"] = "error"
        state["stage"] = active_job.get("stage", "Worker failed")
        state["progress"] = 0
        state["error"] = active_job.get("error")
    
    return {
        "status": state["status"],
        "stage": state["stage"],
        "progress": state["progress"],
        "error": state["error"],
        "active_video_name": state["active_video_name"],
        "active_project_id": state.get("active_project_id"),
        "database": state.get("database", {"connected": False}),
        "has_transcript_upload": state["active_transcript_path"] is not None,
        "has_teams_video": state["teams_link_video_path"] is not None,
        "generating_report": state["generating_report"],
        "report_started_at": state.get("report_started_at"),
        "dataset_status": state["dataset_status"],
        "dataset_progress": state["dataset_progress"],
        "last_metrics": state.get("last_metrics", {}),
        "active_job_id": state.get("active_job_id"),
        "active_job": active_job,
    }

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Uploads lecture video and sets it as the active session input."""
    global state
    try:
        video_path = RAW_DIR / file.filename
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        state["active_video_path"] = str(video_path)
        state["active_video_name"] = file.filename
        try:
            project = create_project_for_video(video_path, source_mode="upload")
            state["active_project_id"] = project.id
            state["database"] = db_health()
        except Exception as db_exc:
            state["active_project_id"] = None
            state["database"] = {"connected": False, "error": str(db_exc)}
        
        # Reset completed run status on new uploads
        clear_result_caches()
        state["status"] = "idle"
        state["stage"] = "Video uploaded"
        state["progress"] = 0
        state["smart_notes"] = ""
        state["active_transcript_path"] = None

        storage_url = sync_storage_artifact(video_path, f"raw/{video_path.name}")
        return {"filename": file.filename, "status": "uploaded", "storage_url": storage_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload video: {str(e)}")

@app.post("/api/upload/transcript")
async def upload_transcript(file: UploadFile = File(...)):
    """Uploads standard Teams transcript or subtitles text file."""
    global state
    try:
        transcript_path = RAW_DIR / file.filename
        with open(transcript_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        state["active_transcript_path"] = str(transcript_path)
        if state.get("active_project_id"):
            try:
                update_project_input(state["active_project_id"], transcript_path=str(transcript_path))
            except Exception as db_exc:
                state["database"] = {"connected": False, "error": str(db_exc)}
        storage_url = sync_storage_artifact(transcript_path, f"transcripts/source/{transcript_path.name}")
        return {"filename": file.filename, "status": "uploaded", "storage_url": storage_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload transcript: {str(e)}")

class AnalysisSettings(BaseModel):
    whisper_model: str = WHISPER_MODEL_DEFAULT
    selected_llm: str = OLLAMA_DEFAULT_MODEL
    speech_language: str = "en"
    lecture_profile: str = "General / no forced terminology"
    whisper_prompt: str = ""
    whisper_hotwords: str = ""
    use_os_glossary: bool = True
    vision_mode: str = "Fast: capture keyframes, no OCR"
    vision_model: str = "llava:7b"
    ssim_thresh: float = SSIM_THRESHOLD
    min_slide_gap: float = 20.0
    max_slide_count: int = 80
    frame_sample_interval: float = FRAME_CHECK_INTERVAL
    analyze_acoustics_enabled: bool = False
    diarization_enabled: bool = False

@app.post("/api/analyze")
def start_analysis(settings: AnalysisSettings, background_tasks: BackgroundTasks):
    """Spawns pipeline processing in a background worker thread."""
    global state
    if state["status"] == "processing":
        raise HTTPException(status_code=400, detail="An analysis run is already in progress.")
        
    video_path_str = state["active_video_path"]
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(status_code=400, detail="No video file has been uploaded for analysis.")
        
    teams_path = None
    if state["active_transcript_path"]:
        teams_path = Path(state["active_transcript_path"])

    # Force a clean result set for every new analysis run.
    clear_result_caches()
        
    payload = {
        "video_path": str(Path(video_path_str)),
        "whisper_model": settings.whisper_model,
        "initial_prompt": settings.whisper_prompt,
        "teams_transcript_path": str(teams_path) if teams_path else None,
        "hotwords": settings.whisper_hotwords,
        "use_glossary": settings.use_os_glossary,
        "visual_mode": settings.vision_mode,
        "ssim_thresh": settings.ssim_thresh,
        "min_keyframe_gap_sec": settings.min_slide_gap,
        "max_keyframes": settings.max_slide_count,
        "frame_check_interval_sec": settings.frame_sample_interval,
        "analyze_acoustics": settings.analyze_acoustics_enabled,
        "speech_language": settings.speech_language,
        "vision_model": settings.vision_model,
        "project_id": state.get("active_project_id"),
        "diarization_enabled": settings.diarization_enabled,
    }
    job_id = create_job("analysis", payload)
    state["active_job_id"] = job_id

    if configured_job_backend() == "redis":
        try:
            queue_info = enqueue_rq_job(job_id, payload)
            state["status"] = "processing"
            state["stage"] = "Queued in Redis/RQ worker"
            state["progress"] = 2
            return {"status": "queued", "job_id": job_id, **queue_info}
        except Exception as exc:
            update_job_status(job_id, status="fallback", stage="Redis queue unavailable; using local background task", error=str(exc))

    background_tasks.add_task(
        run_pipeline_thread,
        video_path=Path(payload["video_path"]),
        whisper_model=payload["whisper_model"],
        initial_prompt=payload["initial_prompt"],
        teams_transcript_path=teams_path,
        hotwords=payload["hotwords"],
        use_glossary=payload["use_glossary"],
        visual_mode=payload["visual_mode"],
        ssim_thresh=payload["ssim_thresh"],
        min_keyframe_gap_sec=payload["min_keyframe_gap_sec"],
        max_keyframes=payload["max_keyframes"],
        frame_check_interval_sec=payload["frame_check_interval_sec"],
        analyze_acoustics=payload["analyze_acoustics"],
        speech_language=payload["speech_language"],
        vision_model=payload["vision_model"],
        project_id=payload["project_id"],
        diarization_enabled=payload["diarization_enabled"],
        job_id=job_id,
    )

    return {"status": "started", "job_id": job_id, "backend": "local-background"}

@app.get("/api/results")
def get_results():
    """Retrieves synchronized transcripts, slide events, and metadata."""
    if not (enriched_cache_path.exists() and slides_cache_path.exists()):
        raise HTTPException(status_code=404, detail="No processed results available. Run analysis first.")
        
    try:
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
        if topic_blocks_cache_path.exists():
            with open(topic_blocks_cache_path, "r", encoding="utf-8") as f:
                topic_blocks = json.load(f)
        else:
            topic_blocks = build_semantic_topic_blocks(transcript)
            with open(topic_blocks_cache_path, "w", encoding="utf-8") as f:
                json.dump(topic_blocks, f, ensure_ascii=False, indent=2)
            
        # Clean paths for client consumption (avoid exposing full local windows paths)
        for slide in slides:
            if "image_path" in slide and slide["image_path"]:
                slide["image_path"] = Path(slide["image_path"]).name
                
        return {"transcript": transcript, "slides": slides, "topic_blocks": topic_blocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {str(e)}")


@app.get("/api/projects")
def get_projects(search: str = ""):
    """List saved lecture projects from PostgreSQL."""
    try:
        state["database"] = db_health()
        if not state["database"].get("connected"):
            return {"database": state["database"], "projects": []}
        return {"database": state["database"], "projects": list_projects(query=search)}
    except Exception as exc:
        state["database"] = {"connected": False, "error": str(exc)}
        return {"database": state["database"], "projects": []}


@app.post("/api/projects/{project_id}/load")
def load_project(project_id: str):
    """Load a saved project into the active workstation session."""
    global state
    payload = get_project_payload(project_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Project not found.")

    transcript = payload.get("transcript") or []
    topic_blocks = payload.get("topic_blocks") or build_semantic_topic_blocks(transcript)
    slides = payload.get("slides") or []
    report_markdown = payload.get("report_markdown") or ""

    with open(enriched_cache_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    with open(topic_blocks_cache_path, "w", encoding="utf-8") as f:
        json.dump(topic_blocks, f, ensure_ascii=False, indent=2)
    with open(slides_cache_path, "w", encoding="utf-8") as f:
        json.dump(slides, f, ensure_ascii=False, indent=2)
    with open(smart_notes_cache_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    state["active_project_id"] = project_id
    state["active_video_path"] = payload.get("video_path")
    state["active_video_name"] = payload.get("video_filename")
    state["active_transcript_path"] = payload.get("transcript_path")
    state["smart_notes"] = report_markdown
    state["status"] = "completed" if transcript and slides else payload.get("status", "idle")
    state["stage"] = "Loaded project from database"
    state["progress"] = 100 if transcript and slides else 0
    state["error"] = None

    return {"status": "loaded", "project": payload, "chat": list_chat_messages(project_id)}


@app.post("/api/projects/save-current")
def save_current_project():
    """Persist active cache artifacts back into the current project record."""
    project_id = state.get("active_project_id")
    if not project_id:
        video_path_str = state.get("active_video_path")
        if not video_path_str or not Path(video_path_str).exists():
            raise HTTPException(status_code=400, detail="No active database project or local video is attached to this session.")
        try:
            project = get_latest_project_for_video(Path(video_path_str)) or create_project_for_video(Path(video_path_str), source_mode="cache")
            project_id = project.id
            state["active_project_id"] = project_id
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    transcript = []
    slides = []
    if enriched_cache_path.exists():
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    if slides_cache_path.exists():
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
    topic_blocks = []
    if topic_blocks_cache_path.exists():
        with open(topic_blocks_cache_path, "r", encoding="utf-8") as f:
            topic_blocks = json.load(f)
    elif transcript:
        topic_blocks = build_semantic_topic_blocks(transcript)
        with open(topic_blocks_cache_path, "w", encoding="utf-8") as f:
            json.dump(topic_blocks, f, ensure_ascii=False, indent=2)
    save_project_artifacts(
        project_id,
        transcript=transcript,
        topic_blocks=topic_blocks,
        slides=slides,
        report_markdown=state.get("smart_notes", ""),
        status="reported" if state.get("smart_notes") else "analyzed",
    )
    return {"status": "saved", "project_id": project_id}


class ProjectMetadataUpdate(BaseModel):
    title: Optional[str] = None
    course_name: Optional[str] = None
    tags: List[str] = []
    description: Optional[str] = None


@app.post("/api/projects/{project_id}/metadata")
def update_project_metadata_endpoint(project_id: str, metadata: ProjectMetadataUpdate):
    project = update_project_metadata(
        project_id,
        title=metadata.title,
        course_name=metadata.course_name,
        tags=metadata.tags,
        description=metadata.description,
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")
    return {"status": "saved", "project": project}

@app.get("/api/report/markdown")
def get_report_markdown():
    """Retrieves the compiled Markdown smart notes study guide."""
    return {"markdown": markdown_for_client(state["smart_notes"])}

class GenerateReportSettings(BaseModel):
    method: str = "fast" # fast, local_ai
    model_name: str = OLLAMA_DEFAULT_MODEL

def generate_report_worker(method: str, model_name: str):
    global state
    state["generating_report"] = True
    state["report_started_at"] = time.time()
    try:
        if not (enriched_cache_path.exists() and slides_cache_path.exists()):
            raise Exception("No analysis results found. Run pipeline before note synthesis.")
            
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
            
        if method == "fast":
            notes = generate_offline_study_notes(slides, transcript)
            state["smart_notes"] = notes
        else:
            # Stream/compile using local LLM generator. Keep updating state so the UI
            # can show partial notes instead of looking frozen while Ollama works.
            generator = generate_smart_notes_stream(slides, transcript, model_name=model_name)
            compiled = []
            completed_contexts = 0
            deadline = time.time() + 240
            for chunk in generator:
                compiled.append(chunk)
                completed_contexts += chunk.count("\n\n---\n\n")
                state["smart_notes"] = "".join(compiled)
                if time.time() > deadline:
                    compiled.append(
                        "\n\n---\n\n"
                        "*Local AI synthesis reached the 4-minute safety limit. "
                        "EchoNotes completed only the remaining visual contexts with deterministic offline notes.*\n\n"
                    )
                    compiled.append(generate_offline_study_notes_from_index(slides, transcript, completed_contexts))
                    break
            state["smart_notes"] = "".join(compiled)
            
        # Save to markdown cache file
        with open(smart_notes_cache_path, "w", encoding="utf-8") as f:
            f.write(state["smart_notes"])
        if state.get("active_project_id"):
            try:
                save_project_artifacts(
                    state["active_project_id"],
                    report_markdown=state["smart_notes"],
                    status="reported",
                )
            except Exception as db_exc:
                state["database"] = {"connected": False, "error": str(db_exc)}
            
    except Exception as e:
        state["smart_notes"] = f"### Synthesis Failed\nAn error occurred while generating notes:\n```\n{str(e)}\n```"
    finally:
        state["generating_report"] = False
        state["report_started_at"] = None

@app.post("/api/report/generate")
def generate_report(settings: GenerateReportSettings, background_tasks: BackgroundTasks):
    """Spawns report generation task in a background thread."""
    global state
    if state["generating_report"]:
        raise HTTPException(status_code=400, detail="Report generation is already running.")
    background_tasks.add_task(generate_report_worker, method=settings.method, model_name=settings.model_name)
    return {"status": "started"}

@app.post("/api/report/reset")
def reset_report_generation_state():
    """Reset a stuck report status after an interrupted local Ollama run."""
    global state
    state["generating_report"] = False
    state["report_started_at"] = None
    return {"status": "reset", "markdown_length": len(state.get("smart_notes") or "")}

@app.post("/api/report/save")
def save_report(data: Dict[str, str]):
    """Allows manual editing edits to be saved back to backend cache."""
    global state
    new_notes = data.get("markdown", "")
    state["smart_notes"] = new_notes
    try:
        with open(smart_notes_cache_path, "w", encoding="utf-8") as f:
            f.write(new_notes)
        if state.get("active_project_id"):
            save_project_artifacts(state["active_project_id"], report_markdown=new_notes, status="reported")
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report/pdf")
def download_report_pdf():
    """Compiles and downloads study guide PDF."""
    report_markdown = state.get("smart_notes") or (
        smart_notes_cache_path.read_text(encoding="utf-8") if smart_notes_cache_path.exists() else ""
    )
    if not report_markdown:
        raise HTTPException(status_code=400, detail="Generate the report markdown notes first.")
        
    try:
        # Load slides coordinates to render inline images inside report PDF
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
            
        pdf_path = OUTPUTS_DIR / "EchoNotes_Report.pdf"
        export_notes_pdf(report_markdown, slides, pdf_path)
        
        if not pdf_path.exists():
            raise HTTPException(status_code=500, detail="PDF compiler completed but did not produce a file.")

        sync_storage_artifact(pdf_path, f"reports/{pdf_path.name}")
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename="EchoNotes_Report.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF compiler error: {str(e)}")


@app.post("/api/report/ingest-pdf")
def ingest_report_pdf():
    """Extract text from the exported PDF artifact and add it to chat retrieval context."""
    pdf_path = OUTPUTS_DIR / "EchoNotes_Report.pdf"
    report_markdown = state.get("smart_notes") or (
        smart_notes_cache_path.read_text(encoding="utf-8") if smart_notes_cache_path.exists() else ""
    )
    if not pdf_path.exists():
        if not report_markdown:
            raise HTTPException(status_code=400, detail="Generate report notes before ingesting PDF.")
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slides = json.load(f)
        export_notes_pdf(report_markdown, slides, pdf_path)

    extracted = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        extracted = report_markdown

    extracted = (extracted or "").strip()
    if not extracted and report_markdown:
        extracted = report_markdown.strip()
    if not extracted:
        raise HTTPException(status_code=500, detail="PDF ingestion produced no text.")
    pdf_text_cache_path.write_text(extracted, encoding="utf-8")
    return {"status": "ingested", "characters": len(extracted), "source": str(pdf_path)}

class ChatRequest(BaseModel):
    question: str
    model: str = OLLAMA_DEFAULT_MODEL
    mode: str = "explain"

@app.post("/api/chat")
def run_chat(req: ChatRequest):
    """Queries grounded chat over notes and segments."""
    chat_started = time.perf_counter()
    if not smart_notes_cache_path.exists() and not state["smart_notes"]:
        # Generate instant fallback notes to ground chat context
        if enriched_cache_path.exists() and slides_cache_path.exists():
            with open(enriched_cache_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            with open(slides_cache_path, "r", encoding="utf-8") as f:
                slides = json.load(f)
            state["smart_notes"] = generate_offline_study_notes(slides, transcript)
            with open(smart_notes_cache_path, "w", encoding="utf-8") as f:
                f.write(state["smart_notes"])
        else:
            raise HTTPException(status_code=400, detail="Synthesize report context before chatting.")
            
    try:
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        if topic_blocks_cache_path.exists():
            with open(topic_blocks_cache_path, "r", encoding="utf-8") as f:
                topic_blocks = json.load(f)
        else:
            topic_blocks = build_semantic_topic_blocks(transcript)
            with open(topic_blocks_cache_path, "w", encoding="utf-8") as f:
                json.dump(topic_blocks, f, ensure_ascii=False, indent=2)
        slides = []
        if slides_cache_path.exists():
            with open(slides_cache_path, "r", encoding="utf-8") as f:
                slides = json.load(f)
        pdf_text = pdf_text_cache_path.read_text(encoding="utf-8") if pdf_text_cache_path.exists() else ""

        try:
            rag_result = answer_with_langchain_rag(
                req.question,
                state["smart_notes"],
                transcript,
                req.model,
                topic_blocks=topic_blocks,
                slides=slides,
                pdf_text=pdf_text,
                query_mode=req.mode,
            )
            latency_ms = int((time.perf_counter() - chat_started) * 1000)
            rag_result["latency_ms"] = latency_ms
            rag_result["mode"] = req.mode
            if state.get("active_project_id"):
                add_chat_message(state["active_project_id"], "user", req.question, query_mode=req.mode)
                assistant_id = add_chat_message(
                    state["active_project_id"],
                    "assistant",
                    rag_result.get("answer", ""),
                    query_mode=req.mode,
                    citations=rag_result.get("citations", []),
                    latency_ms=latency_ms,
                )
                rag_result["message_id"] = assistant_id
            return rag_result
        except Exception as rag_error:
            # Keep the app usable even when Ollama embeddings or LangChain indexing fails.
            fallback_note = f"\n\n_Engine fallback: LangChain RAG was unavailable ({rag_error}). Used custom transcript retrieval._"

        answer = answer_notes_question(
            req.question,
            state["smart_notes"],
            transcript,
            req.model
        )
        if state.get("active_project_id"):
            add_chat_message(state["active_project_id"], "user", req.question, query_mode=req.mode)
            assistant_id = add_chat_message(
                state["active_project_id"],
                "assistant",
                answer,
                query_mode=req.mode,
                citations=[],
                latency_ms=int((time.perf_counter() - chat_started) * 1000),
            )
        else:
            assistant_id = None
        return {
            "answer": answer + fallback_note,
            "engine": "Custom grounded retrieval fallback",
            "sources": [],
            "citations": [],
            "mode": req.mode,
            "latency_ms": int((time.perf_counter() - chat_started) * 1000),
            "message_id": assistant_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    transcript, slides, topic_blocks = load_current_evaluation_artifacts()
    reference_segments = []
    if state.get("active_transcript_path") and Path(state["active_transcript_path"]).exists():
        try:
            reference_segments = parse_teams_transcript(Path(state["active_transcript_path"]))
        except Exception:
            reference_segments = []

    transcript_quality = evaluate_transcript_quality(reference_segments, transcript) if reference_segments else {
        "available": False,
        "reason": "Upload a Teams VTT/SRT/TXT transcript as reference to compute WER/CER.",
    }
    speaker_roles = infer_speaker_roles(transcript)
    speaker_role_preview = {
        key: value for key, value in speaker_roles.items()
        if key != "segments"
    }
    speaker_role_preview["sample_segments"] = [
        {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": (segment.get("text") or "")[:220],
            "speaker": segment.get("speaker"),
            "speaker_role": segment.get("speaker_role"),
        }
        for segment in speaker_roles.get("segments", [])[:20]
    ]
    vlm_benchmark = build_vlm_benchmark(slides, transcript)
    ablation = build_ablation_snapshot(transcript, slides, topic_blocks, vlm_benchmark, transcript_quality)
    regression_set = load_regression_set_summary()
    if not regression_set.get("available"):
        regression_set["buildable_from_current_lecture"] = bool(topic_blocks)
        regression_set["candidate_cases"] = min(12, len(topic_blocks or []))
    chat_messages = list_chat_messages(state.get("active_project_id"), limit=200) if state.get("active_project_id") else []
    chat_latencies = [
        int(message["latency_ms"])
        for message in chat_messages
        if message.get("role") == "assistant" and isinstance(message.get("latency_ms"), int)
    ]
    metrics = {
        **(state.get("last_metrics") or {}),
        "transcript_segments": len(transcript),
        "topic_blocks": len(topic_blocks),
        "keyframes": len(slides),
        "pdf_ingested": pdf_text_cache_path.exists(),
        "chat_messages": len(chat_messages),
        "rag_answer_latency_avg_ms": round(sum(chat_latencies) / len(chat_latencies), 1) if chat_latencies else None,
        "evaluations_stored": len(list_evaluation_records(state.get("active_project_id"), limit=20)) if state.get("active_project_id") else 0,
    }
    speaker_role_preview["diarization_status"] = metrics.get("diarization") or {
        "provider": "unknown",
        "status": "not_available",
        "reason": "Run the pipeline with Enable Speaker Diarization turned on.",
    }
    return {
        "metrics": metrics,
        "transcript_quality": transcript_quality,
        "speaker_roles": speaker_role_preview,
        "vlm_benchmark": vlm_benchmark,
        "ablation": ablation,
        "regression_set": regression_set,
    }


def build_ablation_snapshot(
    transcript: List[Dict[str, Any]],
    slides: List[Dict[str, Any]],
    topic_blocks: List[Dict[str, Any]],
    vlm_benchmark: Dict[str, Any],
    transcript_quality: Dict[str, Any],
) -> Dict[str, Any]:
    transcript_conf = min(1.0, len(topic_blocks or []) / 8) if transcript else 0.0
    ocr_conf = float(vlm_benchmark.get("avg_ocr_confidence") or 0.0)
    vlm_conf = float(vlm_benchmark.get("avg_vlm_confidence") or 0.0)
    fused_conf = float(vlm_benchmark.get("avg_ocr_vlm_confidence") or max(ocr_conf, vlm_conf))
    teams_bonus = 0.1 if transcript_quality.get("available") else 0.0
    return {
        "engine": "deterministic ablation proxy; use human rubric/regression set for final scoring",
        "rows": [
            {
                "mode": "transcript_only",
                "available": bool(transcript),
                "evidence_units": len(transcript or []),
                "confidence": round(transcript_conf, 3),
            },
            {
                "mode": "ocr_only",
                "available": any((slide.get("ocr_text") or "").strip() for slide in slides or []),
                "evidence_units": sum(1 for slide in slides or [] if (slide.get("ocr_text") or "").strip()),
                "confidence": round(ocr_conf, 3),
            },
            {
                "mode": "vlm_only",
                "available": any((slide.get("vlm_description") or "").strip() for slide in slides or []),
                "evidence_units": sum(1 for slide in slides or [] if (slide.get("vlm_description") or "").strip()),
                "confidence": round(vlm_conf, 3),
            },
            {
                "mode": "ocr_plus_vlm_plus_transcript",
                "available": bool(transcript and slides),
                "evidence_units": len(transcript or []) + len(slides or []),
                "confidence": round(min(1.0, max(transcript_conf, fused_conf) + teams_bonus), 3),
            },
        ],
    }


def load_regression_set_summary() -> Dict[str, Any]:
    if not regression_set_path.exists():
        return {"available": False, "cases": 0, "path": str(regression_set_path)}
    try:
        payload = json.loads(regression_set_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "cases": 0, "path": str(regression_set_path), "error": str(exc)}
    cases = payload.get("cases") or []
    return {
        "available": True,
        "name": payload.get("name", "EchoNotes regression set"),
        "cases": len(cases),
        "expected_topics": sum(len(case.get("expected_topics") or []) for case in cases),
        "expected_terms": sum(len(case.get("expected_terms") or []) for case in cases),
        "path": str(regression_set_path),
    }


def load_current_evaluation_artifacts() -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    transcript: List[Dict[str, Any]] = []
    slides: List[Dict[str, Any]] = []
    topic_blocks: List[Dict[str, Any]] = []
    if enriched_cache_path.exists():
        transcript = json.loads(enriched_cache_path.read_text(encoding="utf-8"))
    if slides_cache_path.exists():
        slides = json.loads(slides_cache_path.read_text(encoding="utf-8"))
    if topic_blocks_cache_path.exists():
        topic_blocks = json.loads(topic_blocks_cache_path.read_text(encoding="utf-8"))
    return transcript, slides, topic_blocks


def build_regression_set_payload(
    transcript: List[Dict[str, Any]],
    slides: List[Dict[str, Any]],
    topic_blocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cases = []
    for block in (topic_blocks or [])[:12]:
        start = float(block.get("start", 0) or 0)
        end = float(block.get("end", start + 90) or start + 90)
        block_text = block.get("text") or ""
        keywords = block.get("keywords") or extract_keywords_simple(block_text, limit=8)
        aligned_slides = [
            {
                "timestamp": slide.get("timestamp_formatted"),
                "timestamp_sec": slide.get("timestamp_sec"),
                "has_ocr": bool((slide.get("ocr_text") or "").strip()),
                "has_image_understanding": bool((slide.get("vlm_description") or "").strip()),
            }
            for slide in slides or []
            if start <= float(slide.get("timestamp_sec", 0) or 0) <= end
        ][:5]
        cases.append(
            {
                "id": f"topic-{block.get('index', len(cases) + 1)}",
                "title": block.get("title") or f"Topic {len(cases) + 1}",
                "time_range": {
                    "start": start,
                    "end": end,
                    "label": f"{block.get('timestamp')} - {block.get('end_timestamp')}",
                },
                "expected_topics": [block.get("title") or "lecture topic"],
                "expected_terms": keywords[:8],
                "retrieval_queries": [
                    f"Explain {keywords[0]}" if keywords else f"Explain {block.get('title', 'this topic')}",
                    f"Find exact moment about {keywords[0]}" if keywords else f"Find exact moment for {block.get('title', 'this topic')}",
                    f"Generate quiz for {block.get('title', 'this topic')}",
                ],
                "visual_expectations": aligned_slides,
                "reference_excerpt": block_text[:900],
                "acceptance": {
                    "must_return_timestamp": True,
                    "must_include_at_least_one_expected_term": True,
                    "must_stay_grounded": True,
                },
            }
        )

    return {
        "name": "EchoNotes active lecture regression set",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active_project_id": state.get("active_project_id"),
        "active_video_name": state.get("active_video_name"),
        "source": "generated_from_current_lecture",
        "coverage": {
            "transcript_segments": len(transcript or []),
            "topic_blocks": len(topic_blocks or []),
            "visual_keyframes": len(slides or []),
            "cases": len(cases),
        },
        "cases": cases,
    }


class ChatFeedbackRequest(BaseModel):
    message_id: Optional[str] = None
    rating: str
    comment: str = ""


@app.post("/api/chat/feedback")
def submit_chat_feedback(feedback: ChatFeedbackRequest):
    if feedback.message_id:
        rate_chat_message(feedback.message_id, feedback.rating)
    record_id = add_evaluation_record(
        state.get("active_project_id"),
        "chat_feedback",
        {"message_id": feedback.message_id, "rating": feedback.rating, "comment": feedback.comment},
    )
    return {"status": "saved", "evaluation_id": record_id}


@app.get("/api/export/quiz")
def export_quiz_bank():
    if not topic_blocks_cache_path.exists():
        raise HTTPException(status_code=400, detail="Run analysis first to create semantic topic blocks.")
    topic_blocks = json.loads(topic_blocks_cache_path.read_text(encoding="utf-8"))
    path = OUTPUTS_DIR / "EchoNotes_Quiz_Bank.json"
    write_quiz_json(topic_blocks, path)
    sync_storage_artifact(path, f"exports/{path.name}")
    return FileResponse(str(path), media_type="application/json", filename=path.name)


@app.get("/api/export/anki")
def export_anki_cards():
    if not topic_blocks_cache_path.exists():
        raise HTTPException(status_code=400, detail="Run analysis first to create semantic topic blocks.")
    topic_blocks = json.loads(topic_blocks_cache_path.read_text(encoding="utf-8"))
    path = OUTPUTS_DIR / "EchoNotes_Anki.tsv"
    write_anki_tsv(topic_blocks, path)
    sync_storage_artifact(path, f"exports/{path.name}")
    return FileResponse(str(path), media_type="text/tab-separated-values", filename=path.name)


@app.get("/api/evaluation/regression-set")
def get_regression_set():
    if not regression_set_path.exists():
        raise HTTPException(status_code=404, detail="Regression smoke set is not available.")
    return FileResponse(str(regression_set_path), media_type="application/json", filename=regression_set_path.name)


@app.post("/api/evaluation/regression-set/build")
def build_regression_set_from_current():
    transcript, slides, topic_blocks = load_current_evaluation_artifacts()
    if not topic_blocks:
        raise HTTPException(status_code=400, detail="Run analysis first to create semantic topic blocks.")
    regression_set_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_regression_set_payload(transcript, slides, topic_blocks)
    regression_set_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    record_id = add_evaluation_record(state.get("active_project_id"), "regression_set", payload)
    summary = load_regression_set_summary()
    return {"status": "saved", "evaluation_id": record_id, "regression_set": summary}


@app.get("/api/deployment/readiness")
def deployment_readiness():
    storage_provider = os.getenv("ECHONOTES_STORAGE_PROVIDER", "local").lower()
    vector_backend = os.getenv("ECHONOTES_VECTOR_BACKEND", "auto").lower()
    job_backend = configured_job_backend()
    return {
        "worker_queue": {
            "current": "Redis/RQ" if job_backend == "redis" else "FastAPI BackgroundTasks/local",
            "planned": "Redis/RQ worker service",
            "ready": job_backend == "redis",
        },
        "storage": {
            "current": storage_provider,
            "planned": "local, Azure Blob Storage, or S3 adapter",
            "ready": storage_provider in {"local", "azure", "s3"},
        },
        "vector_retrieval": {
            "current": vector_backend,
            "planned": "auto Chroma/FAISS embeddings with local TF-IDF fallback",
            "ready": vector_backend in {"auto", "local", "tfidf", "chroma", "faiss"},
        },
        "diarization": {
            "current": os.getenv("ECHONOTES_DIARIZATION_PROVIDER", "heuristic"),
            "planned": "heuristic fallback or pyannote.audio with Hugging Face token",
            "ready": os.getenv("ECHONOTES_DIARIZATION_PROVIDER", "heuristic").lower() in {"heuristic", "pyannote"},
        },
        "docker": {
            "current": "PostgreSQL plus full-profile backend/frontend/Redis/RQ worker",
            "planned": "docker compose --profile full up --build",
            "ready": True,
        },
        "auth": {
            "current": "Microsoft Graph device flow for Teams import only",
            "planned": "local user profile / enterprise auth",
            "ready": False,
        },
    }


@app.post("/api/storage/sync-current")
def sync_current_artifacts():
    artifacts = {}
    candidates = [
        (enriched_cache_path, f"transcripts/{enriched_cache_path.name}"),
        (topic_blocks_cache_path, f"transcripts/{topic_blocks_cache_path.name}"),
        (slides_cache_path, f"visual/{slides_cache_path.name}"),
        (smart_notes_cache_path, f"reports/{smart_notes_cache_path.name}"),
        (OUTPUTS_DIR / "EchoNotes_Report.pdf", "reports/EchoNotes_Report.pdf"),
        (OUTPUTS_DIR / "EchoNotes_Quiz_Bank.json", "exports/EchoNotes_Quiz_Bank.json"),
        (OUTPUTS_DIR / "EchoNotes_Anki.tsv", "exports/EchoNotes_Anki.tsv"),
    ]
    for path, object_name in candidates:
        url = sync_storage_artifact(path, object_name)
        if url:
            artifacts[object_name] = url
    return {"provider": os.getenv("ECHONOTES_STORAGE_PROVIDER", "local"), "artifacts": artifacts}

@app.get("/api/video")
def get_video_stream(request: Request):
    """Streams the active video file with byte-range support for HTML5 playback."""
    video_path_str = state["active_video_path"]
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(status_code=404, detail="No video file loaded in session.")

    video_path = Path(video_path_str)
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    media_type = _video_media_type(video_path)

    if not range_header:
        return FileResponse(
            str(video_path),
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        )

    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not match:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    start_s, end_s = match.groups()
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else file_size - 1
    end = min(end, file_size - 1)
    if start >= file_size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    chunk_size = end - start + 1

    def iter_file() -> Generator[bytes, None, None]:
        with video_path.open("rb") as f:
            f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                data = f.read(min(1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Cache-Control": "no-store",
        },
    )


def _video_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".webm":
        return "video/webm"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix in {".mkv", ".avi"}:
        return "video/mp4"
    return "video/mp4"


@app.head("/api/video")
def head_video_stream():
    """Returns video metadata headers for browsers that probe before playback."""
    video_path_str = state["active_video_path"]
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(status_code=404, detail="No video file loaded in session.")
    video_path = Path(video_path_str)
    return Response(
        status_code=200,
        media_type=_video_media_type(video_path),
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(video_path.stat().st_size),
            "Cache-Control": "no-store",
        },
    )

@app.get("/api/keyframes/{filename}")
def serve_keyframe_image(filename: str):
    """Serves slide screenshot files from frame extraction folders."""
    img_path = FRAMES_DIR / filename
    if not img_path.exists():
        # Fallback to check nested absolute path configurations
        raise HTTPException(status_code=404, detail="Keyframe image not found on disk.")
    return FileResponse(str(img_path))

# --- TEAMS GRAPH DEVICE FLOW INTEGRATION ---

@app.post("/api/teams/device-code")
def initiate_device_code():
    """Initiates device flow and registers access keys."""
    global state
    try:
        cfg = get_default_graph_config()
        scopes = cfg["scopes"].split()
        flow = create_device_flow(cfg["client_id"], cfg["tenant_id"], scopes)
        state["graph_device_flow"] = flow
        return {
            "user_code": flow["user_code"],
            "verification_uri": flow["verification_uri"],
            "message": flow["message"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not initiate Microsoft sign-in: {str(e)}")

@app.post("/api/teams/complete-login")
def complete_teams_login():
    """Acquires access tokens once device code matches verification."""
    global state
    flow = state["graph_device_flow"]
    if not flow:
        raise HTTPException(status_code=400, detail="No Microsoft sign-in flow active. Generate device code first.")
        
    try:
        result = complete_device_flow(flow)
        state["graph_access_token"] = result["access_token"]
        state["graph_device_flow"] = None
        return {"status": "authenticated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/teams/download")
def download_teams_link(data: Dict[str, str], background_tasks: BackgroundTasks):
    """Downloads Teams SharePoint link in background."""
    global state
    token = state["graph_access_token"]
    if not token:
        raise HTTPException(status_code=400, detail="Authenticate using Microsoft Graph first.")
        
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="No recording URL provided.")
        
    def download_worker(graph_token: str, sharing_url: str):
        global state
        state["status"] = "processing"
        state["stage"] = "Connecting to Microsoft Graph and downloading assets..."
        state["progress"] = 5
        try:
            assets = download_teams_recording_assets(graph_token, sharing_url, RAW_DIR)
            state["teams_link_video_path"] = str(assets["video_path"])
            state["teams_link_transcript_path"] = str(assets["transcript_path"]) if assets["transcript_path"] else None
            
            state["active_video_path"] = str(assets["video_path"])
            state["active_video_name"] = assets["video_path"].name
            if assets["transcript_path"]:
                state["active_transcript_path"] = str(assets["transcript_path"])
                
            state["status"] = "idle"
            state["stage"] = "Recording assets downloaded successfully"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "error"
            state["stage"] = "Microsoft Graph download failed"
            state["error"] = str(e)
            
    background_tasks.add_task(download_worker, token, url)
    return {"status": "started"}

# --- DATASET FACTORY ENDPOINTS ---

def dataset_worker(video_path: Path, segments: List[Dict[str, Any]], zip_path: Path):
    global state
    state["dataset_status"] = "exporting"
    state["dataset_progress"] = 0
    try:
        def on_prog(p):
            state["dataset_progress"] = int(p * 100)
            
        generate_whisper_dataset(
            video_path=video_path,
            segments=segments,
            output_zip_path=zip_path,
            progress_callback=on_prog
        )
        state["dataset_status"] = "completed"
    except Exception as e:
        state["dataset_status"] = "error"
        state["dataset_stage"] = f"Extraction failed: {str(e)}"

@app.post("/api/dataset/package")
def package_dataset(background_tasks: BackgroundTasks):
    """Slices and compiles dataset ZIP."""
    global state
    if state["dataset_status"] == "exporting":
        raise HTTPException(status_code=400, detail="Dataset packaging already in progress.")
        
    video_path_str = state["active_video_path"]
    if not video_path_str or not Path(video_path_str).exists():
        raise HTTPException(status_code=400, detail="No active video file to create dataset from.")
        
    if not enriched_cache_path.exists():
        raise HTTPException(status_code=400, detail="No transcript details found. Run analysis first.")
        
    try:
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
            
        video_name = Path(video_path_str).name
        export_zip_name = f"{Path(video_name).stem}_dataset.zip"
        export_zip_path = OUTPUTS_DIR / export_zip_name
        
        background_tasks.add_task(
            dataset_worker,
            video_path=Path(video_path_str),
            segments=segments,
            zip_path=export_zip_path
        )
        return {"status": "started", "filename": export_zip_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/download")
def download_dataset():
    """Downloads packaged Hugging Face AudioFolder ZIP."""
    video_path_str = state["active_video_path"]
    if not video_path_str:
        raise HTTPException(status_code=400, detail="No video loaded in session.")
        
    video_name = Path(video_path_str).name
    export_zip_name = f"{Path(video_name).stem}_dataset.zip"
    export_zip_path = OUTPUTS_DIR / export_zip_name
    
    if not export_zip_path.exists():
        raise HTTPException(status_code=404, detail="Dataset package does not exist. Package it first.")
        
    return FileResponse(
        str(export_zip_path),
        media_type="application/zip",
        filename=export_zip_name
    )

@app.post("/api/dataset/train-gpu")
def start_gpu_finetune():
    """Starts local 1-Click GPU fine-tuning if PyTorch CUDA is active."""
    video_path_str = state["active_video_path"]
    if not video_path_str:
        raise HTTPException(status_code=400, detail="No video loaded in session.")
        
    video_name = Path(video_path_str).name
    export_zip_name = f"{Path(video_name).stem}_dataset.zip"
    export_zip_path = OUTPUTS_DIR / export_zip_name
    
    if not export_zip_path.exists():
        raise HTTPException(status_code=400, detail="Create the dataset ZIP package before fine-tuning.")
        
    import torch
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="NVIDIA GPU/CUDA not available in host Python env.")
        
    # Launch training process
    python_exe = str(Path("venv/Scripts/python.exe").resolve())
    train_script = str(Path("scripts/train_whisper_local.py").resolve())
    
    temp_train_dir = DATA_DIR / "processed" / "temp_local_train"
    if temp_train_dir.exists():
        shutil.rmtree(temp_train_dir)
    temp_train_dir.mkdir(parents=True, exist_ok=True)
    
    import zipfile
    with zipfile.ZipFile(export_zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_train_dir)
        
    cmd = [
        python_exe,
        train_script,
        "--dataset_path", str(temp_train_dir.resolve())
    ]
    
    try:
        # Run asynchronously, let training proceed
        subprocess.Popen(cmd)
        return {"status": "started", "message": "GPU Whisper training spawned in background process."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch training: {str(e)}")

# --- SYSTEM UTILITIES ---

@app.get("/api/ollama-models")
def get_ollama_models():
    """Retrieves installed Ollama LLM models."""
    return {"models": get_installed_ollama_models()}

@app.post("/api/clear")
def clear_session_cache():
    """Clears all session caches, raw uploads, and output data."""
    global state
    for cache_file in [enriched_cache_path, slides_cache_path, topic_blocks_cache_path, smart_notes_cache_path, pdf_text_cache_path]:
        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass
            
    # Remove files in raw directory
    for f in RAW_DIR.glob("*"):
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass
            
    state = {
        "status": "idle",
        "stage": "Caches cleared",
        "progress": 0,
        "error": None,
        "active_video_path": None,
        "active_video_name": None,
        "active_transcript_path": None,
        
        "graph_device_flow": None,
        "graph_access_token": None,
        "teams_link_video_path": None,
        "teams_link_transcript_path": None,
        
        "smart_notes": "",
        "generating_report": False,
        
        "dataset_status": "idle",
        "dataset_progress": 0,
        "last_metrics": {}
    }
    
    return {"status": "cleared"}

@app.get("/api/demo-init")
def initialize_demo_mode():
    """Mocks transformer demo data (equivalent to ui.py run demo)."""
    global state
    
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    slide_files = ["slide_00_00_00.jpg", "slide_00_05_12.jpg", "slide_00_12_40.jpg", "slide_00_22_15.jpg"]
    for sf in slide_files:
        open(FRAMES_DIR / sf, "a").close()
        
    slides_data = [
        {
            "timestamp_sec": 0.0,
            "timestamp_formatted": "00:00:00",
            "image_path": str(FRAMES_DIR / "slide_00_00_00.jpg"),
            "ocr_text": "DEEP LEARNING WORKSHOP: SEQUENCE TO SEQUENCE & ATTENTION SYSTEM. Introduction to RNNs, Seq2Seq limits, and Attention mechanisms. Speaker: Prof. Alex Johnson."
        },
        {
            "timestamp_sec": 312.0,
            "timestamp_formatted": "00:05:12",
            "image_path": str(FRAMES_DIR / "slide_00_05_12.jpg"),
            "ocr_text": "LIMITS OF RECURRENT NEURAL NETWORKS (RNNS). 1. Vanishing Gradients on long sequences. 2. Sequential Bottleneck (cannot parallelize training). 3. Information loss in single context vector."
        },
        {
            "timestamp_sec": 760.0,
            "timestamp_formatted": "00:12:40",
            "image_path": str(FRAMES_DIR / "slide_00_12_40.jpg"),
            "ocr_text": "THE SELF-ATTENTION MECHANISM. Formulas: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V. Query, Key, Value vectors explained. Scaled dot-product attention diagram."
        },
        {
            "timestamp_sec": 1335.0,
            "timestamp_formatted": "00:22:15",
            "image_path": str(FRAMES_DIR / "slide_00_22_15.jpg"),
            "ocr_text": "TRANSFORMER ENCODER-DECODER ARCHITECTURE. Multi-Head Attention blocks, Feed Forward networks, Positional Encoding, Residual Connections & Layer Normalization."
        }
    ]
    
    transcript_data = [
        {
            "start": 10.0, "end": 25.0,
            "text": "Welcome to today's deep learning workshop. We will explore sequence modeling architectures, moving from recurrent neural networks to attention-based models.",
            "acoustics": {"volume_rms": 0.012, "pitch_hz": 120.0, "speech_rate_wps": 2.5, "is_loud": False, "is_high_pitch": False, "is_slow": False, "emphasis_score": 0, "has_semantic_clue": False, "is_important": False}
        },
        {
            "start": 320.0, "end": 335.0,
            "text": "Pay close attention to this limitation: the critical weakness of RNNs is the sequential bottleneck. Training cannot be fully parallelized because each token depends on previous time steps.",
            "acoustics": {"volume_rms": 0.045, "pitch_hz": 215.0, "speech_rate_wps": 1.8, "is_loud": True, "is_high_pitch": True, "is_slow": True, "emphasis_score": 3, "has_semantic_clue": True, "is_important": True}
        },
        {
            "start": 780.0, "end": 795.0,
            "text": "The self-attention formula uses Query, Key, and Value vectors. Attention of Q, K, and V is computed as softmax of Q times K transpose divided by the square root of d_k, then multiplied by V.",
            "acoustics": {"volume_rms": 0.038, "pitch_hz": 190.0, "speech_rate_wps": 1.9, "is_loud": True, "is_high_pitch": False, "is_slow": True, "emphasis_score": 2, "has_semantic_clue": True, "is_important": True}
        },
        {
            "start": 810.0, "end": 820.0,
            "text": "Why do we divide by the square root of d_k? The scaling term prevents dot-product values from becoming too large, which would push the softmax function into saturated regions.",
            "acoustics": {"volume_rms": 0.052, "pitch_hz": 230.0, "speech_rate_wps": 1.5, "is_loud": True, "is_high_pitch": True, "is_slow": True, "emphasis_score": 3, "has_semantic_clue": True, "is_important": True}
        },
        {
            "start": 1350.0, "end": 1365.0,
            "text": "In the Transformer architecture, residual connections and layer normalization are essential for stable optimization, faster convergence, and reducing gradient degradation in deep networks.",
            "acoustics": {"volume_rms": 0.025, "pitch_hz": 130.0, "speech_rate_wps": 2.3, "is_loud": False, "is_high_pitch": False, "is_slow": False, "emphasis_score": 0, "has_semantic_clue": True, "is_important": False}
        }
    ]
    
    with open(enriched_cache_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    with open(slides_cache_path, "w", encoding="utf-8") as f:
        json.dump(slides_data, f, ensure_ascii=False, indent=2)
        
    state["active_video_name"] = "Transformer_DeepLearning_Demo.mp4"
    state["active_video_path"] = None  # No real playback file in mock demo mode
    state["smart_notes"] = ""
    state["status"] = "completed"
    state["stage"] = "Demo initialized"
    state["progress"] = 100
    
    return {"status": "initialized"}
