import sys
import os

# Global UTF-8 encoding safeguard for Windows terminal prints
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import time
import re
import html
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from urllib.parse import urlparse
import requests

# Import pipeline components
from app.config import (
    RAW_DIR, OUTPUTS_DIR, FRAMES_DIR, 
    WHISPER_MODEL_DEFAULT, OLLAMA_DEFAULT_MODEL, SSIM_THRESHOLD, FRAME_CHECK_INTERVAL
)
from app.utils.video import extract_audio_from_video
from app.utils.audio import transcribe_audio
from app.utils.acoustic import analyze_audio_acoustics
from app.utils.vision import detect_slide_transitions
from app.utils.llm import get_installed_ollama_models, generate_smart_notes_stream, generate_offline_study_notes, convert_local_images_to_base64, chunk_segments_by_time, extract_keywords_simple, summarize_chunk_offline
from app.utils.corrector import corrector, correct_transcript_segment
from app.utils.dataset_factory import generate_whisper_dataset
from app.utils.model_sync import download_file_from_google_drive, extract_and_install_zip, auto_detect_google_drive_paths
from app.utils.teams_transcript import parse_teams_transcript
from app.utils.graph_import import complete_device_flow, create_device_flow, download_teams_recording_assets, get_default_graph_config
from app.utils.pdf_export import export_notes_pdf


# Page configuration with custom title & icon
st.set_page_config(
    page_title="EchoNotes AI - Multimodal Lecture Synthesizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Premium custom CSS styling (Dark-glassmorphism and modern fonts)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Header styling */
    .title-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .title-container::after {
        content: '';
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 50%);
        pointer-events: none;
    }
    .tagline {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Modern Card UI */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        border-color: rgba(30, 60, 114, 0.4);
    }
    .badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 5px;
    }
    .badge-important {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }
    .badge-pitch {
        background-color: rgba(155, 89, 182, 0.15);
        color: #9b59b6;
        border: 1px solid rgba(155, 89, 182, 0.3);
    }
    .badge-volume {
        background-color: rgba(241, 196, 15, 0.15);
        color: #f1c40f;
        border: 1px solid rgba(241, 196, 15, 0.3);
    }
    .badge-slow {
        background-color: rgba(52, 152, 219, 0.15);
        color: #3498db;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }

    /* Keep reruns visually stable. Streamlit can briefly fade containers while
       Python is busy; that reads like a broken UI during long local AI calls. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stVerticalBlock"],
    [data-testid="stElementContainer"],
    [data-testid="stMarkdownContainer"] {
        opacity: 1 !important;
        transition: none !important;
        animation: none !important;
    }
    [data-testid="stStatusWidget"] {
        opacity: 0.95 !important;
    }
    .echonotes-loading {
        border: 1px solid rgba(96, 165, 250, 0.22);
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(15, 23, 42, 0.08));
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin: 0.75rem 0 1rem 0;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .echonotes-loading strong {
        color: #93c5fd;
    }
    .echonotes-loading .bar {
        height: 4px;
        margin-top: 0.75rem;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(148, 163, 184, 0.2);
    }
    .echonotes-loading .bar::before {
        content: "";
        display: block;
        height: 100%;
        width: 42%;
        background: linear-gradient(90deg, #60a5fa, #22c55e);
        animation: echonotesSlide 1.2s ease-in-out infinite;
    }
    @keyframes echonotesSlide {
        0% { transform: translateX(-110%); }
        100% { transform: translateX(260%); }
    }
    .chat-panel {
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(15, 23, 42, 0.42));
        border-radius: 12px;
        padding: 1rem;
        margin-top: 0.75rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }
    .chat-hint {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 0.85rem;
    }
    .chat-row {
        display: flex;
        gap: 0.8rem;
        align-items: flex-start;
        margin: 0.7rem 0;
    }
    .chat-row.user {
        flex-direction: row-reverse;
    }
    .chat-avatar {
        width: 34px;
        height: 34px;
        border-radius: 9px;
        display: grid;
        place-items: center;
        font-size: 1rem;
        flex: 0 0 34px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.12);
    }
    .chat-avatar.user {
        background: #ef4444;
    }
    .chat-avatar.assistant {
        background: #f59e0b;
    }
    .chat-bubble {
        max-width: 82%;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        line-height: 1.55;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .chat-row.user .chat-bubble {
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.24);
    }
    .chat-row.assistant .chat-bubble {
        background: rgba(30, 41, 59, 0.72);
    }
    .chat-role {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
import json

enriched_cache_path = Path("data/processed/transcripts/enriched_segments_cache.json")
slides_cache_path = Path("data/processed/transcripts/slides_cache.json")

has_offline_cache = enriched_cache_path.exists() and slides_cache_path.exists()

if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = has_offline_cache

# Cache Auto-Corrector logic
def apply_glossary_corrections_to_cache(use_glossary: bool = True):
    if has_offline_cache:
        try:
            # 1. Load, correct and write back enriched_segments_cache.json
            with open(enriched_cache_path, "r", encoding="utf-8") as f:
                raw_segments = json.load(f)
            cleaned_segments = [
                correct_transcript_segment(seg, use_glossary=use_glossary, use_fuzzy=use_glossary)
                for seg in raw_segments
            ]
            with open(enriched_cache_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_segments, f, ensure_ascii=False, indent=2)
                
            # 2. Load, correct and write back slides_cache.json
            with open(slides_cache_path, "r", encoding="utf-8") as f:
                raw_slides = json.load(f)
            cleaned_slides = []
            for slide in raw_slides:
                if use_glossary and "ocr_text" in slide and slide["ocr_text"]:
                    slide["ocr_text"] = corrector.correct_text(slide["ocr_text"])
                cleaned_slides.append(slide)
            with open(slides_cache_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_slides, f, ensure_ascii=False, indent=2)
                
            # Update session state with fresh cleaned data
            st.session_state.transcript_data = cleaned_segments
            st.session_state.slides_data = cleaned_slides
            st.session_state.cache_auto_corrected = True
            return True
        except Exception as e:
            st.sidebar.error(f"Automatic cache cleanup failed: {e}")
    return False

# Trigger startup auto-correction once
if has_offline_cache and "cache_auto_corrected" not in st.session_state:
    st.session_state.cache_auto_corrected = True

if "transcript_data" not in st.session_state:
    if has_offline_cache:
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            st.session_state.transcript_data = json.load(f)
    else:
        st.session_state.transcript_data = None
if "slides_data" not in st.session_state:
    if has_offline_cache:
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            st.session_state.slides_data = json.load(f)
    else:
        st.session_state.slides_data = None
if "smart_notes" not in st.session_state:
    st.session_state.smart_notes = ""
if "video_start_time" not in st.session_state:
    st.session_state.video_start_time = 0
if "notes_chat_history" not in st.session_state:
    st.session_state.notes_chat_history = []
if "pending_notes_question" not in st.session_state:
    st.session_state.pending_notes_question = None
if "pending_notes_question_ready" not in st.session_state:
    st.session_state.pending_notes_question_ready = False

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 🎙️ EchoNotes AI Dashboard")
st.sidebar.caption("Multimodal Video Lecture Summarizer")

def reset_pipeline_state(clear_disk_cache: bool = True):
    if clear_disk_cache:
        for cache_file in [enriched_cache_path, slides_cache_path]:
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except Exception:
                pass
        try:
            for cache_file in Path("data/processed/transcripts").glob("*_transcript.json"):
                cache_file.unlink()
        except Exception:
            pass

    st.session_state.pipeline_complete = False
    st.session_state.smart_notes = ""
    st.session_state.slides_data = None
    st.session_state.transcript_data = None
    st.session_state.pop("teams_link_video_path", None)
    st.session_state.pop("teams_link_transcript_path", None)
    st.session_state.pop("active_video_name", None)

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
        res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=(10, 180))
        if res.status_code == 200:
            answer = res.json().get("message", {}).get("content", "").strip()
            return answer or local_fallback
        return f"{local_fallback}\n\n_Local AI returned HTTP {res.status_code}, so EchoNotes used transcript retrieval instead._"
    except requests.exceptions.Timeout:
        return f"{local_fallback}\n\n_Local AI timed out, so EchoNotes used transcript retrieval instead._"
    except Exception as exc:
        return f"{local_fallback}\n\n_Local AI was unavailable: {exc}_"

if st.sidebar.button("🧹 Clear cache / start new video", type="primary", use_container_width=True):
    reset_pipeline_state(clear_disk_cache=True)
    st.rerun()

# Input Type Selection
source_mode = st.sidebar.radio(
    "Data source",
    [
        "Manual video/transcript upload",
        "Teams Link Import (Advanced - requires Microsoft login)",
        "Video Demo"
    ],
    index=0
)
use_demo = source_mode == "Video Demo"

uploaded_file = None
uploaded_transcript = None
teams_recording_url = ""
manual_video_path = None
manual_transcript_path = None
teams_link_video_path = st.session_state.get("teams_link_video_path")
teams_link_transcript_path = st.session_state.get("teams_link_transcript_path")

if source_mode == "Manual video/transcript upload":
    uploaded_file = st.sidebar.file_uploader("Upload lecture video (MP4/MKV)", type=["mp4", "mkv", "avi", "mov"])
    if uploaded_file:
        manual_video_path = RAW_DIR / uploaded_file.name
        with open(manual_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.active_video_path = str(manual_video_path)
        if st.session_state.get("active_video_name") and st.session_state.active_video_name != uploaded_file.name:
            st.session_state.pipeline_complete = False
            st.session_state.smart_notes = ""
            st.session_state.slides_data = None
            st.session_state.transcript_data = None
        st.sidebar.success(f"Received video: {uploaded_file.name}")

    uploaded_transcript = st.sidebar.file_uploader(
        "Upload Teams transcript (optional: VTT/SRT/TXT)",
        type=["vtt", "srt", "txt"],
        help="If you have the official Microsoft Teams transcript, EchoNotes will use it instead of Whisper for better accuracy."
    )
    if uploaded_transcript:
        manual_transcript_path = RAW_DIR / uploaded_transcript.name
        with open(manual_transcript_path, "wb") as f:
            f.write(uploaded_transcript.getbuffer())
        st.sidebar.success(f"Received Teams transcript: {uploaded_transcript.name}")

elif source_mode.startswith("Teams Link Import"):
    st.sidebar.caption("Advanced: download from SharePoint/OneDrive when Microsoft Graph login is available.")
    graph_defaults = get_default_graph_config()
    tenant_id = st.sidebar.text_input("Microsoft Tenant ID", graph_defaults["tenant_id"])
    client_id = st.sidebar.text_input("Microsoft Client ID", graph_defaults["client_id"])
    scopes_text = st.sidebar.text_input("Graph scopes", graph_defaults["scopes"])
    teams_recording_url = st.sidebar.text_input(
        "Paste Teams/SharePoint recording link",
        "",
        placeholder="https://...sharepoint.com/.../Recordings/....mp4",
        help="Use a SharePoint/OneDrive/Stream recording link, not a login.microsoftonline.com OAuth URL."
    )

    if teams_recording_url:
        parsed_recording_url = urlparse(teams_recording_url if "://" in teams_recording_url else f"https://{teams_recording_url}")
        recording_host = parsed_recording_url.netloc.lower()
        if "login.microsoftonline.com" in recording_host:
            st.sidebar.error("This is a Microsoft OAuth login URL, not a recording link.")
        elif any(host in recording_host for host in ["sharepoint.com", "onedrive.live.com", "office.com", "microsoftstream.com"]):
            st.sidebar.info("The recording link format looks valid. Sign in with Microsoft Graph to try downloading it.")
        else:
            st.sidebar.warning("This does not look like a typical SharePoint/OneDrive/Stream recording link.")

    if st.sidebar.button("Create Microsoft sign-in code", use_container_width=True):
        if not client_id.strip():
            st.sidebar.error("Missing Microsoft Client ID. You need a public-client Azure App registration.")
        else:
            try:
                scopes = [scope.strip() for scope in scopes_text.split() if scope.strip()]
                flow = create_device_flow(client_id.strip(), tenant_id.strip(), scopes)
                st.session_state.graph_device_flow = flow
                st.sidebar.success("Microsoft sign-in code created.")
            except Exception as e:
                st.sidebar.error(f"Could not create Microsoft sign-in code: {e}")

    if st.session_state.get("graph_device_flow") and not st.session_state.get("graph_access_token"):
        flow = st.session_state.graph_device_flow
        verification_uri = flow.get("verification_uri") or "https://microsoft.com/devicelogin"
        user_code = flow.get("user_code", "")
        st.sidebar.info("Open the link below, enter the code, then return here and complete sign-in.")
        st.sidebar.code(user_code)
        st.sidebar.markdown(f"[Open Microsoft sign-in page]({verification_uri})")
        if st.sidebar.button("✅ Complete Microsoft sign-in", type="primary", use_container_width=True, key="complete_ms_login_sidebar"):
            try:
                token_result = complete_device_flow(flow)
                st.session_state.graph_access_token = token_result["access_token"]
                st.session_state.pop("graph_device_flow", None)
                st.sidebar.success("Microsoft Graph sign-in completed.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Sign-in is not complete or consent was denied: {e}")

    if st.session_state.get("graph_access_token"):
        st.sidebar.success("Microsoft Graph token is ready in this session.")

    if st.sidebar.button("Download recording from Teams link", use_container_width=True):
        if not teams_recording_url:
            st.sidebar.error("No recording link was provided.")
        elif not st.session_state.get("graph_access_token"):
            st.sidebar.error("Sign in with Microsoft Graph first.")
        else:
            with st.sidebar.status("Downloading recording through Microsoft Graph...", expanded=True):
                try:
                    assets = download_teams_recording_assets(
                        st.session_state.graph_access_token,
                        teams_recording_url,
                        RAW_DIR
                    )
                    st.session_state.teams_link_video_path = str(assets["video_path"])
                    st.session_state.teams_link_transcript_path = str(assets["transcript_path"]) if assets["transcript_path"] else None
                    st.write(f"Video: {assets['video_path'].name}")
                    if assets["transcript_path"]:
                        st.write(f"Transcript: {assets['transcript_path'].name}")
                    else:
                        st.write("No transcript file was found in the same folder. Upload a .vtt/.srt fallback below.")
                    st.sidebar.success("Recording downloaded from Teams link.")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Could not download from Teams link: {e}")

    fallback_transcript = st.sidebar.file_uploader(
        "Fallback: upload transcript Teams (VTT/SRT/TXT)",
        type=["vtt", "srt", "txt"],
        help="Use this when Graph can download the video but cannot find the automatic transcript."
    )
    if fallback_transcript:
        manual_transcript_path = RAW_DIR / fallback_transcript.name
        with open(manual_transcript_path, "wb") as f:
            f.write(fallback_transcript.getbuffer())
        st.session_state.teams_link_transcript_path = str(manual_transcript_path)
        st.sidebar.success(f"Received fallback transcript: {fallback_transcript.name}")

    if teams_link_video_path:
        st.sidebar.success(f"Video ready: {Path(teams_link_video_path).name}")
    if st.session_state.get("teams_link_transcript_path"):
        st.sidebar.success(f"Transcript ready: {Path(st.session_state.teams_link_transcript_path).name}")

    with st.sidebar.expander("How to get the correct Teams link/file"):
        st.markdown(
            """
1. Open the meeting chat in Teams.
2. Click the recording.
3. Choose **Open in Stream** or **Open in OneDrive/SharePoint**.
4. Copy the SharePoint/OneDrive recording file link.
5. If Graph cannot fetch the transcript, download the `.vtt` manually and upload it as fallback.
            """
        )

else:
    st.sidebar.info("💡 Demo mode is enabled. EchoNotes will load a sample Transformer & Self-Attention lecture analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ AI Engine")

# STT Model
whisper_model_option = st.sidebar.selectbox(
    "Whisper model (Speech-to-Text)",
    ["tiny", "base", "small", "medium", "Custom trained model"],
    index=2,
    help="Choose a built-in model or load your own fine-tuned Whisper model."
)

whisper_model = whisper_model_option
if whisper_model_option == "Custom trained model":
    default_custom_path = str(Path("d:/GITHUB/EchoNotes-AI/data/whisper-vinglish-model/whisper-vinglish-ct2").resolve())
    whisper_model = st.sidebar.text_input(
        "Custom model directory path:",
        default_custom_path,
        help="Absolute path to the directory containing model.bin, config.json, and related model files."
    )
    
    # 🔄 AUTOMATED MODEL SYNC & DOWNLOAD SYSTEM
    with st.sidebar.expander("🪄 Automatic model sync"):
        st.caption("Download or unpack a custom model without manually moving files.")
        
        sync_method = st.radio(
            "Sync method:",
            ["⚡ Google Drive Desktop (Recommended)", "🔗 Google Drive share link"],
            index=0
        )
        
        if sync_method.startswith("⚡"):
            # Auto-detect local Google Drive Desktop zip paths
            detected_paths = auto_detect_google_drive_paths()
            default_zip_path = detected_paths[0] if detected_paths else r"G:\My Drive\whisper-vinglish-ct2.zip"
            
            local_zip_path = st.text_input(
                "Local Drive ZIP path:",
                default_zip_path,
                help="Path to a ZIP file synchronized by Google Drive Desktop on this machine."
            )
            
            if st.button("🔄 Sync and update model", use_container_width=True):
                if not Path(local_zip_path).exists():
                    st.error("❌ ZIP file was not found at this path. Make sure Google Drive Desktop has finished syncing.")
                else:
                    with st.spinner("Extracting and updating the custom model..."):
                        try:
                            success = extract_and_install_zip(Path(local_zip_path), Path(whisper_model))
                            if success:
                                st.success("✅ Custom model was synced and updated.")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            gdrive_url = st.text_input(
                "Google Drive share link:",
                "",
                placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
                help="Share link with public or anyone-with-link access."
            )
            
            if st.button("📥 Download and install automatically", use_container_width=True):
                if not gdrive_url:
                    st.warning("Enter a Google Drive share link first.")
                else:
                    temp_download_zip = Path("data/processed/temp_gdrive_download.zip")
                    with st.spinner("Downloading model from Google Drive. This may take 15-30 seconds..."):
                        try:
                            # 1. Download
                            success_dl = download_file_from_google_drive(gdrive_url, temp_download_zip)
                            if success_dl:
                                # 2. Extract and install
                                success_install = extract_and_install_zip(temp_download_zip, Path(whisper_model))
                                # Cleanup download zip
                                if temp_download_zip.exists():
                                    os.remove(temp_download_zip)
                                    
                                if success_install:
                                    st.success("✅ Model downloaded, extracted, and installed.")
                                    time.sleep(1.5)
                                    st.rerun()
                            else:
                                st.error("❌ Download failed. Check the network connection or share link.")
                        except Exception as e:
                            if temp_download_zip.exists():
                                os.remove(temp_download_zip)
                            st.error(f"Error: {e}")


# Ollama settings
local_models = get_installed_ollama_models()
if not local_models:
    st.sidebar.warning("⚠️ Ollama is not running. Start Ollama to use local AI synthesis/chat.")
    selected_llm = st.sidebar.text_input("Ollama model name:", OLLAMA_DEFAULT_MODEL)
else:
    # Ensure default model is first if available
    default_idx = local_models.index(OLLAMA_DEFAULT_MODEL) if OLLAMA_DEFAULT_MODEL in local_models else 0
    selected_llm = st.sidebar.selectbox("LLM model (Ollama)", local_models, index=default_idx)

# Whisper language and prompt
speech_language_label = st.sidebar.selectbox(
    "Spoken language in video",
    [
        "English",
        "Auto detect / preserve original language",
        "Vietnamese",
    ],
    index=0,
    help="For APSS/Business Central videos that are fully English, choose English. Auto detect may misread accents as Malay/Indonesian."
)
speech_language = {
    "English": "en",
    "Auto detect / preserve original language": "auto",
    "Vietnamese": "vi",
}[speech_language_label]

lecture_profile = st.sidebar.selectbox(
    "Lecture profile",
    [
        "General / no forced terminology",
        "Operating systems / inode-block-pointer"
    ],
    index=0,
    help="Only choose the OS profile when the video is actually about inode/block/pointer. Wrong profiles bias the transcript."
)

if speech_language == "en":
    default_prompt = "English technical lecture. Preserve exact English terms, code identifiers, product names, and acronyms. Do not translate."
elif speech_language == "vi":
    default_prompt = (
        "Vietnamese lecture with possible English technical terms. Preserve clear English terms as-is."
        if lecture_profile.startswith("General")
        else "Vietnamese operating systems lecture with UNIX, FreeBSD, inode, block, pointer, file system, direct, and indirect terms."
    )
else:
    default_prompt = "Technical lecture. Preserve the speaker's original language. Keep English technical terms and code identifiers unchanged. Do not translate."
default_hotwords = (
    ""
    if lecture_profile.startswith("General")
    else "UNIX FreeBSD inode block pointer file system single indirect double indirect triple indirect allocation storage pointer direct indirect"
)
use_os_glossary = st.sidebar.checkbox(
    "Enable OS/Vinglish glossary",
    value=not lecture_profile.startswith("General"),
    help="Keep this off for non-OS videos to avoid biasing transcript terms toward inode/block/pointer."
)
whisper_prompt = st.sidebar.text_area(
    "📝 ASR context prompt",
    default_prompt,
    height=80,
    key=f"whisper_prompt_{speech_language}_{lecture_profile}",
    help="Very specific prompts can bias ASR. Use a neutral prompt for general videos."
)
whisper_hotwords = st.sidebar.text_area(
    "Whisper hotwords",
    default_hotwords,
    height=70,
    key=f"whisper_hotwords_{speech_language}_{lecture_profile}",
    help="Only add terms that definitely appear in the video. Leave blank for general videos."
)

# Slide settings sliders
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Visual analysis")
vision_mode = st.sidebar.selectbox(
    "Visual analysis mode",
    [
        "Transcript only: skip slides/keyframes",
        "Fast: capture keyframes, no OCR",
        "Full: capture keyframes + OCR"
    ],
    index=0,
    help="For coding/screen recordings, use Fast or Transcript-only. Full OCR is slower."
)
ssim_thresh = st.sidebar.slider("Slide-change sensitivity (SSIM)", 0.85, 0.99, SSIM_THRESHOLD, step=0.01, 
                                help="Lower SSIM means a larger screen change is required before capturing a new keyframe.")
min_slide_gap = st.sidebar.slider(
    "Minimum gap between keyframes (seconds)",
    5, 60, 20, step=5,
    help="Increase this for code/editor videos to avoid too many keyframes."
)
max_slide_count = st.sidebar.slider(
    "Maximum keyframes",
    20, 300, 80, step=20,
    help="Stop visual scanning after this many keyframes."
)
frame_sample_interval = st.sidebar.slider(
    "Frame check interval (seconds)",
    2, 30, 10, step=2,
    help="Increase to 10-30 seconds for long videos. Decrease if screens change quickly."
)
analyze_acoustics_enabled = st.sidebar.checkbox(
    "Enable acoustic emphasis analysis",
    value=False,
    help="Enable this when you need red/important transcript rows based on voice emphasis."
)

# Glossary Controls Sidebar Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Optional correction glossary")
st.sidebar.caption(f"Loaded **{len(corrector.rules)}** optional technical correction rules from data/glossary.json.")

if st.sidebar.button("🔄 Reload glossary corrections", use_container_width=True,
                      help="Only use this for OS/Vinglish lectures. Keep it off for general videos."):
    corrector.load_glossary()
    if not use_os_glossary:
        st.sidebar.warning("The OS/Vinglish glossary is disabled, so cached transcript segments were not modified.")
    elif apply_glossary_corrections_to_cache(use_glossary=use_os_glossary):
        st.sidebar.success("✅ Glossary corrections were applied to the cached transcript.")
        st.rerun()
    else:
        st.sidebar.warning("No transcript cache is available yet.")

st.sidebar.write("")
if st.sidebar.button("🪄 AI post-ASR cleanup", use_container_width=True,
                      help="Runs a local LLM over transcript segments for cleanup. Use carefully because it can rewrite wording."):
    if st.session_state.transcript_data:
        with st.spinner("AI is cleaning transcript segments. This may take 30-40 seconds..."):
            from app.utils.llm import post_process_transcript_with_llm
            cleaned_segments = post_process_transcript_with_llm(st.session_state.transcript_data, model_name=selected_llm)
            
            # Save back to cache
            try:
                with open(enriched_cache_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned_segments, f, ensure_ascii=False, indent=2)
                st.session_state.transcript_data = cleaned_segments
                st.sidebar.success("✅ AI transcript cleanup completed.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Cache save failed: {e}")
    else:
        st.sidebar.warning("No transcript data is available yet.")

# ----------------- MAIN TITLE HEADER -----------------
st.markdown("""
<div class="title-container">
    <h1 style="margin:0; font-size:2.8rem; font-weight:800;">🎙️ EchoNotes AI</h1>
    <div class="tagline">Multimodal lecture intelligence: transcript analysis, visual keyframes, acoustic cues, PDF reports, and local AI Q&A.</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.get("graph_device_flow") and not st.session_state.get("graph_access_token"):
    flow = st.session_state.graph_device_flow
    verification_uri = flow.get("verification_uri") or "https://microsoft.com/devicelogin"
    st.info("After entering the Microsoft code, click the button below so EchoNotes can retrieve the access token.")
    st.code(flow.get("user_code", ""), language="text")
    st.markdown(f"[Open Microsoft sign-in page]({verification_uri})")
    if st.button("✅ Complete Microsoft sign-in", type="primary", use_container_width=True, key="complete_ms_login_main"):
        try:
            token_result = complete_device_flow(flow)
            st.session_state.graph_access_token = token_result["access_token"]
            st.session_state.pop("graph_device_flow", None)
            st.success("Microsoft Graph sign-in completed.")
            st.rerun()
        except Exception as e:
            st.error(f"Sign-in is not complete or consent was denied: {e}")

# ----------------- PROCESS LOGIC OR DEMO MOCK -----------------

if st.session_state.pipeline_complete:
    if st.button("🧹 Clear current results and upload a new video", type="primary", use_container_width=True, key="reset_top_main"):
        reset_pipeline_state(clear_disk_cache=True)
        st.rerun()
    if (
        st.session_state.get("active_video_path")
        and st.session_state.get("slides_data")
        and all(not slide.get("image_path") for slide in st.session_state.slides_data)
    ):
        st.warning("Current results are transcript-only, so no visual preview is available.")
        if st.button("🖼️ Generate quick keyframes from current video", use_container_width=True, key="generate_keyframes_now"):
            with st.spinner("Capturing quick keyframes without OCR..."):
                try:
                    keyframes = detect_slide_transitions(
                        Path(st.session_state.active_video_path),
                        progress_callback=None,
                        ssim_threshold=0.88,
                        frame_check_interval=20.0,
                        min_transition_gap_sec=45.0,
                        run_ocr=False,
                        max_slides=40
                    )
                    with open(slides_cache_path, "w", encoding="utf-8") as f:
                        json.dump(keyframes, f, ensure_ascii=False, indent=2)
                    st.session_state.slides_data = keyframes
                    st.success(f"Generated {len(keyframes)} quick keyframes.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not generate keyframes: {e}")

def run_multimodal_pipeline(
    video_path: Path,
    initial_prompt: str = None,
    teams_transcript_path: Path = None,
    hotwords: str = None,
    use_glossary: bool = True,
    visual_mode: str = "Fast: capture keyframes, no OCR",
    min_keyframe_gap_sec: float = 20.0,
    max_keyframes: int = 80,
    frame_check_interval_sec: float = 10.0,
    analyze_acoustics: bool = False,
    speech_language: str = "auto"
):
    """Executes the pipeline stage by stage and shows progress."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    teams_transcript_path = Path(teams_transcript_path) if teams_transcript_path else None
    use_teams_transcript = bool(teams_transcript_path and teams_transcript_path.exists())
    needs_audio = (not use_teams_transcript) or analyze_acoustics
    
    # Stage 1: Audio extraction
    status_text.markdown("🔄 **Step 1/5:** Preparing input data...")
    progress_bar.progress(10)
    audio_path = None
    if needs_audio:
        status_text.markdown("🔄 **Step 1/5:** Extracting 16kHz audio from the video with FFmpeg...")
        audio_path = extract_audio_from_video(video_path)
    time.sleep(1)
    
    # Stage 2: Transcript source
    if use_teams_transcript:
        status_text.markdown("🔄 **Step 2/5:** Importing official Microsoft Teams transcript...")
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
        status_text.markdown("🔄 **Step 2/5:** Faster-Whisper is transcribing the audio...")
        
        # Create real-time progress indicators
        sub_progress_bar = st.progress(0.0)
        sub_status_text = st.empty()
        
        def whisper_progress_callback(current_sec, total_sec):
            percent = min(1.0, current_sec / total_sec) if total_sec > 0 else 0.0
            sub_progress_bar.progress(percent)
            
            m_curr, s_curr = int(current_sec // 60), int(current_sec % 60)
            m_tot, s_tot = int(total_sec // 60), int(total_sec % 60)
            sub_status_text.markdown(
                f"⏳ Speech recognition progress: **{m_curr:02d}:{s_curr:02d}** / {m_tot:02d}:{s_tot:02d} "
                f"({percent * 100:.1f}%)"
            )
            
        raw_transcript = transcribe_audio(
            audio_path, 
            model_size=whisper_model, 
            progress_callback=whisper_progress_callback,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            language=speech_language
        )
        
        # Clear sub-indicators upon stage completion
        sub_progress_bar.empty()
        sub_status_text.empty()
    progress_bar.progress(35)
    time.sleep(1)
    
    # Stage 3: Acoustic Profiling
    status_text.markdown("🔄 **Step 3/5:** Processing acoustic labels...")
    progress_bar.progress(60)
    if analyze_acoustics and audio_path:
        status_text.markdown("🔄 **Step 3/5:** Librosa is calculating pitch (F0) and loudness (RMS)...")
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
    time.sleep(1)
    
    # Stage 4: Slide transition & OCR
    status_text.markdown("🔄 **Step 4/5:** Analyzing visuals/keyframes with the selected mode...")
    progress_bar.progress(85)
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
    time.sleep(1)
    
    # Stage 5: Finish basic pipeline
    status_text.markdown("✅ **Raw streams processed.** Synchronizing data for LLM summarization...")
    progress_bar.progress(100)
    time.sleep(1.5)
    status_text.empty()
    progress_bar.empty()
    
    # Save cache files to disk for permanent offline loading
    try:
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
            
        with open(enriched_cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_enriched_segments, f, ensure_ascii=False, indent=2)
        with open(slides_cache_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_slide_keyframes, f, ensure_ascii=False, indent=2)
            
        st.session_state.transcript_data = cleaned_enriched_segments
        st.session_state.slides_data = cleaned_slide_keyframes
    except Exception as e:
        st.session_state.transcript_data = enriched_segments
        st.session_state.slides_data = slide_keyframes
        
        st.session_state.pipeline_complete = True
        st.session_state.active_video_name = video_path.name
        st.session_state.active_video_path = str(video_path)

# Action handler
if has_offline_cache and not st.session_state.pipeline_complete:
    st.success("Found the latest processed result in cache.")
    if st.button("📂 Open processed results", type="primary", use_container_width=True):
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            st.session_state.transcript_data = json.load(f)
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            st.session_state.slides_data = json.load(f)
        if not st.session_state.get("active_video_path"):
            raw_videos = sorted(
                list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mkv")) + list(RAW_DIR.glob("*.avi")) + list(RAW_DIR.glob("*.mov")),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if raw_videos:
                st.session_state.active_video_path = str(raw_videos[0])
        st.session_state.pipeline_complete = True
        st.rerun()

if not st.session_state.pipeline_complete:
    if use_demo:
        if st.button("🚀 Run demo analysis", type="primary", use_container_width=True):
            # Load mock database representing a rich tutorial on Transformers
            with st.spinner("Initializing demo Deep Learning lecture data..."):
                time.sleep(2)
                
                # Mock slide keyframes
                mock_img_dir = FRAMES_DIR
                mock_img_dir.mkdir(parents=True, exist_ok=True)
                
                # Create empty dummy files for demonstration (representing slides)
                slide_files = ["slide_00_00_00.jpg", "slide_00_05_12.jpg", "slide_00_12_40.jpg", "slide_00_22_15.jpg"]
                for sf in slide_files:
                    open(mock_img_dir / sf, "a").close() # Touch files
                    
                st.session_state.slides_data = [
                    {
                        "timestamp_sec": 0.0,
                        "timestamp_formatted": "00:00:00",
                        "image_path": str(mock_img_dir / "slide_00_00_00.jpg"),
                        "ocr_text": "DEEP LEARNING WORKSHOP: SEQUENCE TO SEQUENCE & ATTENTION SYSTEM. Introduction to RNNs, Seq2Seq limits, and Attention mechanisms. Speaker: Prof. Alex Johnson."
                    },
                    {
                        "timestamp_sec": 312.0,
                        "timestamp_formatted": "00:05:12",
                        "image_path": str(mock_img_dir / "slide_00_05_12.jpg"),
                        "ocr_text": "LIMITS OF RECURRENT NEURAL NETWORKS (RNNS). 1. Vanishing Gradients on long sequences. 2. Sequential Bottleneck (cannot parallelize training). 3. Information loss in single context vector."
                    },
                    {
                        "timestamp_sec": 760.0,
                        "timestamp_formatted": "00:12:40",
                        "image_path": str(mock_img_dir / "slide_00_12_40.jpg"),
                        "ocr_text": "THE SELF-ATTENTION MECHANISM. Formulas: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V. Query, Key, Value vectors explained. Scaled dot-product attention diagram."
                    },
                    {
                        "timestamp_sec": 1335.0,
                        "timestamp_formatted": "00:22:15",
                        "image_path": str(mock_img_dir / "slide_00_22_15.jpg"),
                        "ocr_text": "TRANSFORMER ENCODER-DECODER ARCHITECTURE. Multi-Head Attention blocks, Feed Forward networks, Positional Encoding, Residual Connections & Layer Normalization."
                    }
                ]
                
                # Mock enriched transcript segments
                st.session_state.transcript_data = [
                    {
                        "start": 10.0, "end": 25.0,
                        "text": "Chào mừng các bạn đến với buổi Workshop về Deep Learning hôm nay. Chúng ta sẽ cùng nhau tìm hiểu về các kiến trúc xử lý chuỗi từ RNN đến Attention.",
                        "acoustics": {"volume_rms": 0.012, "pitch_hz": 120.0, "speech_rate_wps": 2.5, "is_loud": False, "is_high_pitch": False, "is_slow": False, "emphasis_score": 0, "has_semantic_clue": False, "is_important": False}
                    },
                    {
                        "start": 320.0, "end": 335.0,
                        "text": "Các bạn đặc biệt lưu ý cho mình chỗ này! Nhược điểm chí mạng của RNN chính là sequential bottleneck, nghĩa là bạn không thể train song song được, bắt buộc từ phải đi qua từng bước tuần tự.",
                        "acoustics": {"volume_rms": 0.045, "pitch_hz": 215.0, "speech_rate_wps": 1.8, "is_loud": True, "is_high_pitch": True, "is_slow": True, "emphasis_score": 3, "has_semantic_clue": True, "is_important": True}
                    },
                    {
                        "start": 780.0, "end": 795.0,
                        "text": "Công thức tính Self-Attention bắt buộc các bạn phải thuộc lòng khi đi thi. Attention của Q, K, V sẽ bằng Softmax của Q nhân K chuyển vị chia cho căn bậc hai của d_k, tất cả nhân với V.",
                        "acoustics": {"volume_rms": 0.038, "pitch_hz": 190.0, "speech_rate_wps": 1.9, "is_loud": True, "is_high_pitch": False, "is_slow": True, "emphasis_score": 2, "has_semantic_clue": True, "is_important": True}
                    },
                    {
                        "start": 810.0, "end": 820.0,
                        "text": "Vì sao phải chia cho căn bậc hai của d_k? Đây là câu hỏi phỏng vấn rất hay gặp tại các tập đoàn lớn như Bosch! Chia cho căn d_k để tránh việc giá trị dot-product quá lớn khiến Softmax bị bão hòa.",
                        "acoustics": {"volume_rms": 0.052, "pitch_hz": 230.0, "speech_rate_wps": 1.5, "is_loud": True, "is_high_pitch": True, "is_slow": True, "emphasis_score": 3, "has_semantic_clue": True, "is_important": True}
                    },
                    {
                        "start": 1350.0, "end": 1365.0,
                        "text": "Trong kiến trúc Transformer, các đường kết nối tắt Residual Connection và Layer Normalization đóng vai trò cực kỳ quan trọng giúp mô hình hội tụ nhanh hơn và không bị tiêu biến gradient.",
                        "acoustics": {"volume_rms": 0.025, "pitch_hz": 130.0, "speech_rate_wps": 2.3, "is_loud": False, "is_high_pitch": False, "is_slow": False, "emphasis_score": 0, "has_semantic_clue": True, "is_important": False}
                    }
                ]
                
                st.session_state.pipeline_complete = True
                st.rerun()
    else:
        if source_mode == "Manual video/transcript upload" and manual_video_path:
            if st.button("🚀 Run analysis pipeline", type="primary", use_container_width=True):
                run_multimodal_pipeline(
                    Path(manual_video_path),
                    initial_prompt=whisper_prompt,
                    teams_transcript_path=manual_transcript_path,
                    hotwords=whisper_hotwords,
                    use_glossary=use_os_glossary,
                    visual_mode=vision_mode,
                    min_keyframe_gap_sec=min_slide_gap,
                    max_keyframes=max_slide_count,
                    frame_check_interval_sec=frame_sample_interval,
                    analyze_acoustics=analyze_acoustics_enabled,
                    speech_language=speech_language
                )
                st.rerun()
        elif source_mode.startswith("Teams Link Import") and teams_link_video_path:
            if st.button("🚀 Analyze recording downloaded from Teams link", type="primary", use_container_width=True):
                run_multimodal_pipeline(
                    Path(teams_link_video_path),
                    initial_prompt=whisper_prompt,
                    teams_transcript_path=Path(st.session_state.teams_link_transcript_path) if st.session_state.get("teams_link_transcript_path") else None,
                    hotwords=whisper_hotwords,
                    use_glossary=use_os_glossary,
                    visual_mode=vision_mode,
                    min_keyframe_gap_sec=min_slide_gap,
                    max_keyframes=max_slide_count,
                    frame_check_interval_sec=frame_sample_interval,
                    analyze_acoustics=analyze_acoustics_enabled,
                    speech_language=speech_language
                )
                st.rerun()
        else:
            if source_mode.startswith("Teams Link Import"):
                st.info("👈 Sign in with Microsoft Graph and download the recording, or switch to manual upload.")
            else:
                st.info("👈 Upload a lecture video in the sidebar, or choose Video Demo to try the app.")

# ----------------- MAIN VIEW DASHBOARD -----------------
if st.session_state.pipeline_complete:
    
    # 1. Highlights metrics row
    visual_items = [
        slide for slide in st.session_state.slides_data
        if (slide.get("image_path") or "").strip() or (slide.get("ocr_text") or "").strip()
    ]
    total_slides = len(visual_items)
    total_segments = len(st.session_state.transcript_data)
    imp_segments = sum(1 for s in st.session_state.transcript_data if s.get("acoustics", {}).get("is_important", False))
    transcript_source = "Microsoft Teams" if any(s.get("source") == "teams_transcript" for s in st.session_state.transcript_data) else "Whisper/Faster-Whisper"
    transcript_only = total_slides == 0
    nlp_topic_blocks = len(chunk_segments_by_time(st.session_state.transcript_data))
    visual_label = "NLP Context" if transcript_only else "Visual Keyframes"
    visual_value = "Transcript-only" if transcript_only else f"{total_slides} Keyframes"
    visual_caption = "Slide/OCR skipped to prioritize NLP pipeline" if transcript_only else "Screenshots/OCR aligned with transcript timeline"
    signal_label = "NLP Topic Blocks" if transcript_only else "Acoustic Highlights"
    signal_value = f"{nlp_topic_blocks} Blocks" if transcript_only else f"{imp_segments} Moments"
    signal_caption = "Time-window segmentation for summary, QA, and dataset generation" if transcript_only else "Loud, slow, or emphasized speech moments"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.9rem; opacity:0.75; text-transform:uppercase;">{visual_label}</span>
            <h2 style="margin: 5px 0 0 0; font-size:2.2rem; color:#3498db;">{visual_value}</h2>
            <span style="font-size:0.8rem; opacity:0.6;">{visual_caption}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.9rem; opacity:0.75; text-transform:uppercase;">{signal_label}</span>
            <h2 style="margin: 5px 0 0 0; font-size:2.2rem; color:#e74c3c;">{signal_value}</h2>
            <span style="font-size:0.8rem; opacity:0.6;">{signal_caption}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.9rem; opacity:0.75; text-transform:uppercase;">Transcript Units</span>
            <h2 style="margin: 5px 0 0 0; font-size:2.2rem; color:#2ecc71;">{total_segments} Segments</h2>
            <span style="font-size:0.8rem; opacity:0.6;">Timestamped text ready for NLP analysis</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.caption(f"Transcript source: **{transcript_source}**")
    asr_meta = next((s.get("asr_metadata") for s in st.session_state.transcript_data if s.get("asr_metadata")), None)
    if asr_meta:
        requested_lang = asr_meta.get("requested_language", "unknown")
        detected_lang = asr_meta.get("language") or asr_meta.get("detected_language", "unknown")
        lang_prob = asr_meta.get("language_probability")
        lang_note = f"ASR language: requested **{requested_lang}**, detected **{detected_lang}**"
        if lang_prob is not None:
            lang_note += f" ({lang_prob})"
        st.caption(lang_note)
    
    # 2. Main Tabs
    tab_notes, tab_slides, tab_highlights, tab_transcript, tab_dataset = st.tabs([
        "NLP Smart Notes", 
        "Visual Context", 
        "Key Moments", 
        "Lecture Playback",
        "NLP Dataset Factory"
    ])
    
    # ------------- TAB 1: SMART NOTES (LLM SYNTHESIS) -------------
    with tab_notes:
        st.markdown("### NLP Smart Notes")
        st.caption("Transcript-first study notes with topic segmentation, key terms, review questions, and dataset hooks.")
        
        # Mode selector to avoid hardware-related local LLM timeouts completely
        synthesis_mode = st.radio(
            "Synthesis method:",
            [
                "⚡ Fast offline synthesis",
                "🪄 Local AI synthesis with Ollama"
            ],
            index=0,
            help="Use offline synthesis for speed, or Ollama for deeper analysis."
        )
        
        st.write("")
        
        # Action button to trigger synthesis based on selected mode
        if st.session_state.smart_notes == "":
            if synthesis_mode.startswith("⚡"):
                if st.button("⚡ Generate report instantly", type="primary", use_container_width=True):
                    loading_box = st.empty()
                    loading_box.markdown(
                        '<div class="echonotes-loading"><strong>Generating offline report...</strong><div class="bar"></div></div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.5)
                    notes = generate_offline_study_notes(st.session_state.slides_data, st.session_state.transcript_data)
                    st.session_state.smart_notes = notes
                    loading_box.empty()
                    st.rerun()
            else:
                if st.button("🪄 Generate report with local AI", type="primary", use_container_width=True):
                    notes_placeholder = st.empty()
                    loading_box = st.empty()
                    stream_text = ""
                    last_render_at = 0.0
                    
                    loading_box.markdown(
                        '<div class="echonotes-loading"><strong>Ollama is synthesizing the lecture context...</strong><div class="bar"></div></div>',
                        unsafe_allow_html=True,
                    )
                    # Call LLM streaming generator
                    for chunk in generate_smart_notes_stream(
                        st.session_state.slides_data,
                        st.session_state.transcript_data,
                        model_name=selected_llm
                    ):
                        stream_text += chunk
                        now = time.monotonic()
                        if now - last_render_at >= 0.8:
                            notes_placeholder.markdown(convert_local_images_to_base64(stream_text))
                            last_render_at = now
                            
                    loading_box.empty()
                    notes_placeholder.markdown(convert_local_images_to_base64(stream_text))
                    st.session_state.smart_notes = stream_text
                    st.rerun()
        else:
            # Display cached notes
            st.markdown(convert_local_images_to_base64(st.session_state.smart_notes))
            
            # Export options
            st.write("---")
            col_ex1, col_ex2, col_ex3 = st.columns([1.2, 1.2, 3])
            with col_ex1:
                st.download_button(
                    label="Download Markdown",
                    data=st.session_state.smart_notes,
                    file_name="EchoNotes_SmartNotes.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_ex2:
                pdf_path = OUTPUTS_DIR / "EchoNotes_Report.pdf"
                try:
                    export_notes_pdf(st.session_state.smart_notes, st.session_state.slides_data or [], pdf_path)
                    st.download_button(
                        label="Download PDF",
                        data=pdf_path.read_bytes(),
                        file_name="EchoNotes_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"PDF export unavailable: {e}")
            with col_ex3:
                if st.button("Regenerate report", help="Clear cached report and synthesize again"):
                    st.session_state.smart_notes = ""
                    st.rerun()

            st.write("---")
            st.markdown("### Chat with this report")
            st.markdown(
                '<div class="chat-panel"><div class="chat-hint">Ask grounded questions about the generated report, transcript, key moments, or visual context.</div>',
                unsafe_allow_html=True,
            )

            def render_chat_message(role: str, content: str):
                safe = html.escape(content or "").replace("\n", "<br>")
                avatar = "☻" if role == "user" else "▣"
                label = "You" if role == "user" else "EchoNotes"
                st.markdown(
                    f"""
                    <div class="chat-row {role}">
                        <div class="chat-avatar {role}">{avatar}</div>
                        <div class="chat-bubble">
                            <div class="chat-role">{label}</div>
                            <div>{safe}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            for msg in st.session_state.notes_chat_history[-10:]:
                render_chat_message(msg["role"], msg["content"])

            pending_question = st.session_state.get("pending_notes_question")
            if pending_question:
                loading_box = st.empty()
                loading_label = (
                    "Question queued. Preparing the report context..."
                    if not st.session_state.get("pending_notes_question_ready")
                    else "Reading the report and transcript..."
                )
                loading_box.markdown(
                    f"""
                    <div class="chat-row assistant">
                        <div class="chat-avatar assistant">▣</div>
                        <div class="chat-bubble">
                            <div class="chat-role">EchoNotes</div>
                            <div class="echonotes-loading" style="margin:0; padding:0.7rem 0.8rem;">
                                <strong>{loading_label}</strong>
                                <div class="bar"></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not st.session_state.get("pending_notes_question_ready"):
                    st.session_state.pending_notes_question_ready = True
                    time.sleep(0.35)
                    st.rerun()

                answer = answer_notes_question(
                    pending_question,
                    st.session_state.smart_notes,
                    st.session_state.transcript_data,
                    selected_llm,
                )
                st.session_state.notes_chat_history.append({"role": "assistant", "content": answer})
                st.session_state.pending_notes_question = None
                st.session_state.pending_notes_question_ready = False
                loading_box.empty()
                st.rerun()

            with st.form("notes_chat_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Message",
                    placeholder="Ask about a concept, timestamp, key moment, or report detail...",
                    label_visibility="collapsed",
                )
                send_clicked = st.form_submit_button("Send", use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            if send_clicked and user_question.strip():
                question = user_question.strip()
                st.session_state.notes_chat_history.append({"role": "user", "content": question})
                st.session_state.pending_notes_question = question
                st.session_state.pending_notes_question_ready = False
                st.rerun()
                    
    # ------------- TAB 2: SLIDES TIMELINE -------------
    with tab_slides:
        if transcript_only:
            st.markdown("### Visual Context")
            st.info("This run is in transcript-only mode, so no keyframes, OCR, or images were generated.")
            st.markdown(
                """
                **When should Visual Context be enabled?**
                - You need screenshots for a portfolio or demo.
                - The video contains important slides or code screens.
                - You want transcript alignment with screen changes.

                **Recommended for long videos:** choose `Fast: keyframes only, no OCR`, set minimum gap to 20-30 seconds, and cap keyframes around 40-80.
                """
            )
        else:
            st.markdown("### 🖼️ Visual Keyframe Timeline")
            st.caption("Screenshots/keyframes aligned to the transcript timeline.")

            for idx, slide in enumerate(visual_items):
                with st.container():
                    st.markdown(f"#### Keyframe {idx + 1} - {slide['timestamp_formatted']}")
                    col_img, col_info = st.columns([1, 1])

                    with col_img:
                        if use_demo:
                            st.image(
                                "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=600&auto=format&fit=crop&q=60",
                                caption=f"Slide Screenshot at {slide['timestamp_formatted']}",
                                use_container_width=True
                            )
                        elif os.path.exists(slide.get("image_path", "")):
                            st.image(slide["image_path"], use_container_width=True)
                        else:
                            st.warning("Keyframe image file was not found.")

                    with col_info:
                        if slide.get("ocr_text"):
                            st.markdown("**OCR / visual text:**")
                            st.info(slide["ocr_text"])
                        else:
                            st.caption("OCR is disabled for this run. The keyframe is still available as visual context.")

                        st.markdown("**Transcript near this moment:**")
                        slide_speech = []
                        start_time = slide["timestamp_sec"]
                        next_slide = visual_items[idx + 1] if idx + 1 < len(visual_items) else None
                        end_time = next_slide["timestamp_sec"] if next_slide else float("inf")
                        for seg in st.session_state.transcript_data:
                            if start_time <= seg["start"] < end_time:
                                slide_speech.append(seg["text"])
                        st.write((" ".join(slide_speech)[:500] + "...") if slide_speech else "No transcript is aligned to this interval.")
                    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
                
    # ------------- TAB 3: KEY MOMENTS / HIGHLIGHTS -------------
    with tab_highlights:
        if transcript_only:
            st.markdown("### NLP Key Moments")
            st.caption("Transcript-only mode uses topic windows and keyword density instead of acoustic pitch/volume.")
            chunks = chunk_segments_by_time(st.session_state.transcript_data)
            for idx, chunk in enumerate(chunks[:10], start=1):
                keywords = extract_keywords_simple(chunk["text"], limit=8)
                bullets = summarize_chunk_offline(chunk["text"], max_sentences=2)
                start_m = int(chunk["start"] // 60)
                start_s = int(chunk["start"] % 60)
                end_m = int(chunk["end"] // 60)
                end_s = int(chunk["end"] % 60)
                jump_col, text_col = st.columns([1, 5])
                with jump_col:
                    if st.button(f"Jump {start_m:02d}:{start_s:02d}", key=f"jump_topic_{idx}"):
                        st.session_state.video_start_time = int(chunk["start"])
                        st.rerun()
                with text_col:
                    st.markdown(f"#### {start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d} | Topic Window {idx}")
                if keywords:
                    st.markdown(" ".join(f"`{kw}`" for kw in keywords))
                for bullet in bullets:
                    st.write(f"- {bullet}")
        else:
            st.markdown("### 🔥 Acoustic Highlights")
            st.caption("Segments detected from volume, pitch, speech rate, and semantic trigger words.")

            highlights_exist = False
            for idx, seg in enumerate(st.session_state.transcript_data):
                ac = seg.get("acoustics", {})
                if ac.get("is_important", False):
                    highlights_exist = True

                    m = int(seg["start"] // 60)
                    s = int(seg["start"] % 60)
                    time_fmt = f"{m:02d}:{s:02d}"

                    badge_html = ""
                    if ac.get("is_loud"): badge_html += '<span class="badge badge-volume">LOUD</span>'
                    if ac.get("is_high_pitch"): badge_html += '<span class="badge badge-pitch">HIGH PITCH</span>'
                    if ac.get("is_slow"): badge_html += '<span class="badge badge-slow">SLOW</span>'
                    if ac.get("has_semantic_clue"): badge_html += '<span class="badge badge-important">KEYWORD</span>'

                    if st.button(f"Jump to {time_fmt}", key=f"jump_acoustic_{idx}"):
                        st.session_state.video_start_time = int(seg["start"])
                        st.rerun()
                    st.markdown(f"""
                    <div style="background: rgba(231, 76, 60, 0.04); border-left: 5px solid #e74c3c; padding: 1.2rem; border-radius: 4px; margin-bottom: 1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 800; color: #e74c3c; font-size:1.1rem;">⏱️ Timestamp [{time_fmt}]</span>
                            <div>{badge_html}</div>
                        </div>
                        <p style="font-style: italic; font-size: 1.05rem; margin: 0; color: rgba(255,255,255,0.85); line-height: 1.5;">
                            "{seg['text']}"
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            if not highlights_exist:
                st.info("No acoustic highlights are available. Enable acoustic analysis in the sidebar if you need voice emphasis signals.")
            
    # ------------- TAB 4: VIDEO + TRANSCRIPT NAVIGATOR -------------
    with tab_transcript:
        st.markdown("### Lecture Playback")
        st.caption("Play the lecture, search transcript text, and jump to any timestamp. Blue rows are normal speech; red rows are acoustic/key moments when acoustic analysis is enabled.")

        active_video = st.session_state.get("active_video_path")
        if active_video and Path(active_video).exists():
            # Passing bytes forces Streamlit to remount the media element after timestamp jumps.
            st.video(Path(active_video).read_bytes(), start_time=int(st.session_state.get("video_start_time", 0)))
            st.caption(f"Current jump target: {int(st.session_state.get('video_start_time', 0))}s")
        else:
            st.info("No local video file is available for playback in this session.")

        search_query = st.text_input("Search transcript", "")
        max_rows = st.slider("Rows to show", 25, 250, 120, step=25)
        shown = 0

        for idx, seg in enumerate(st.session_state.transcript_data):
            text = seg.get("text", "")
            if search_query and search_query.lower() not in text.lower():
                continue
            if shown >= max_rows:
                break
            shown += 1

            start = float(seg.get("start", 0) or 0)
            m = int(start // 60)
            s = int(start % 60)
            time_fmt = f"{m:02d}:{s:02d}"
            ac = seg.get("acoustics", {})
            is_imp = bool(ac.get("is_important", False))
            color = "#ef4444" if is_imp else "#2563eb"
            bg = "rgba(239, 68, 68, 0.08)" if is_imp else "rgba(37, 99, 235, 0.08)"

            jump_col, text_col = st.columns([1, 7])
            with jump_col:
                if st.button(time_fmt, key=f"jump_transcript_{idx}"):
                    st.session_state.video_start_time = int(start)
                    st.rerun()
            with text_col:
                st.markdown(
                    f"""
                    <div style="border-left:4px solid {color}; background:{bg}; padding:0.65rem 0.9rem; border-radius:6px; margin-bottom:0.35rem;">
                        <span style="color:{color}; font-family:monospace; font-weight:700;">[{time_fmt}]</span>
                        <span style="margin-left:0.75rem;">{text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
    # ------------- TAB 5: DATASET EXPORTER / FACTORY -------------
    with tab_dataset:
        st.markdown("### NLP Dataset Factory")
        st.caption("Export timestamped audio/text segments for ASR or NLP evaluation datasets.")
        
        st.markdown("""
        The Dataset Factory can export processed lecture data into reusable training/evaluation assets:
        - Timestamped transcript segments.
        - Optional audio clips aligned to each segment.
        - Metadata suitable for ASR/NLP experiments.
        
        Use this when you want a clean dataset for Whisper evaluation, transcript correction, summarization, or retrieval experiments.
        """)
        
        if use_demo:
            st.warning("⚠️ Demo mode uses simulated data and has no real video file. Upload a real video in the sidebar to export a dataset.")
        else:
            # Crash-proof fallback if no file is currently uploaded in Streamlit session but we are using cached results
            if uploaded_file:
                video_name = uploaded_file.name
                current_video = RAW_DIR / video_name
            else:
                # Search RAW_DIR for any previously uploaded videos
                raw_videos = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.mkv")) + list(RAW_DIR.glob("*.avi"))
                if raw_videos:
                    current_video = raw_videos[0]
                    video_name = current_video.name
                else:
                    video_name = "Meeting in General -20260515_085012-recording.mp4"
                    current_video = RAW_DIR / video_name
            
            st.info(f"🎥 Active video: `{video_name}` ({total_segments} transcript segments)")
            
            # Destination path
            export_zip_name = f"{Path(video_name).stem}_dataset.zip"
            export_zip_path = OUTPUTS_DIR / export_zip_name
            
            # Check if dataset zip already exists
            if export_zip_path.exists():
                st.success("✅ A dataset package already exists for this video.")
                # Download button
                with open(export_zip_path, "rb") as f:
                    st.download_button(
                        label="🎁 Download dataset package (.zip)",
                        data=f,
                        file_name=export_zip_name,
                        mime="application/zip",
                        use_container_width=True
                    )
                st.write("")
                
                # Check if CUDA is available locally to display the correct notice
                import torch
                import subprocess
                cuda_is_available = torch.cuda.is_available()
                
                st.markdown("---")
                st.markdown("### 🚀 Local Fine-Tuning & Model Install (1-Click GPU)")
                
                if not cuda_is_available:
                    st.warning("⚠️ Your current Python environment is using CPU-only PyTorch. To enable NVIDIA GPU fine-tuning, run this CUDA PyTorch install command in the terminal:")
                    st.code("venv\\Scripts\\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121", language="bash")
                    st.info("💡 After upgrading, the local GPU training button below will become available and run fully offline.")
                else:
                    st.success("✅ NVIDIA GPU is available for local training.")
                
                # We let the button be clicked, but show warning if CUDA is not installed
                if st.button("🚀 Start local GPU fine-tuning", type="primary", use_container_width=True, disabled=not cuda_is_available):
                    log_area = st.empty()
                    status_indicator = st.info("🔄 Initializing training data pipeline...")
                    
                    try:
                        import zipfile
                        import shutil
                        temp_train_dir = Path("data/processed/temp_local_train")
                        if temp_train_dir.exists():
                            shutil.rmtree(temp_train_dir)
                        temp_train_dir.mkdir(parents=True, exist_ok=True)
                        
                        status_indicator.info("📂 Extracting dataset package...")
                        with zipfile.ZipFile(export_zip_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_train_dir)
                            
                        status_indicator.info("🔥 GPU fine-tuning has started. Follow the log below.")
                        
                        # Command execution
                        python_exe = str(Path("venv/Scripts/python.exe").resolve())
                        train_script = str(Path("scripts/train_whisper_local.py").resolve())
                        
                        cmd = [
                            python_exe,
                            train_script,
                            "--dataset_path", str(temp_train_dir.resolve())
                        ]
                        
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            encoding="utf-8",
                            errors="ignore"
                        )
                        
                        log_text = ""
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                break
                            log_text += line
                            log_area.code(log_text)
                            
                        process.wait()
                        
                        # Cleanup temp training directory
                        if temp_train_dir.exists():
                            shutil.rmtree(temp_train_dir)
                            
                        if process.returncode == 0:
                            status_indicator.success("🎉 Fine-tuning completed. The merged model was installed into the project model directory.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            status_indicator.error(f"❌ Training failed with exit code: {process.returncode}")
                    except Exception as e:
                        status_indicator.error(f"❌ System error: {e}")
                        
                st.write("")
                if st.button("🔄 Regenerate dataset package"):
                    export_zip_path.unlink()
                    st.rerun()
            else:
                if st.button("🎁 Extract and package dataset", type="primary", use_container_width=True):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    try:
                        def on_progress(p):
                            progress_bar.progress(p)
                            if p < 0.2:
                                status_text.text("Extracting full WAV audio stream (16kHz mono)...")
                            elif p < 0.8:
                                status_text.text(f"Cutting aligned audio segments ({int(p*100)}%)...")
                            elif p < 0.9:
                                status_text.text("Writing metadata.csv index...")
                            elif p < 0.98:
                                status_text.text("Compressing dataset into ZIP...")
                            else:
                                status_text.text("Done.")
                                
                        with st.spinner("Extracting and labeling dataset..."):
                            count = generate_whisper_dataset(
                                video_path=current_video,
                                segments=st.session_state.transcript_data,
                                output_zip_path=export_zip_path,
                                progress_callback=on_progress
                            )
                        st.success(f"🎉 Dataset exported with {count} aligned audio files and metadata.csv.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Dataset extraction failed: {e}")
                        
    # Reset button at bottom
    st.write("")
    if st.button("🧹 Clear all analysis results and upload a new video"):
        reset_pipeline_state(clear_disk_cache=True)
        st.rerun()
