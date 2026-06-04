import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Callable
from faster_whisper import WhisperModel
from app.config import TRANSCRIPTS_DIR, DEVICE, WHISPER_COMPUTE_TYPE, WHISPER_MODEL_DEFAULT, CPU_THREADS

def _safe_cache_token(value: str) -> str:
    value = str(value or "").replace("\\", "/")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_")[:80] or "default"

def _build_transcript_cache_path(audio_path: Path, model_size: str, initial_prompt: str, hotwords: str, language: str = None) -> Path:
    cache_profile = "asr-v4-language-aware"
    prompt_hash = hashlib.sha1(
        f"{cache_profile}\n{language or 'auto'}\n{initial_prompt or ''}\n{hotwords or ''}".encode("utf-8")
    ).hexdigest()[:10]
    model_token = _safe_cache_token(model_size)
    return TRANSCRIPTS_DIR / f"{audio_path.stem}_{model_token}_{prompt_hash}_transcript.json"

def _trim_repeated_tail(text: str, max_repeats: int = 3) -> str:
    words = text.split()
    if len(words) < max_repeats + 2:
        return text.strip()

    while len(words) >= max_repeats + 1:
        tail = words[-1].strip(".,!?;:").lower()
        if not tail:
            break

        repeat_count = 1
        for prev in reversed(words[:-1]):
            if prev.strip(".,!?;:").lower() != tail:
                break
            repeat_count += 1

        if repeat_count <= max_repeats:
            break

        words = words[:-(repeat_count - max_repeats)]

    return " ".join(words).strip()

def transcribe_audio(
    audio_path: Path, 
    model_size: str = WHISPER_MODEL_DEFAULT,
    progress_callback: Callable[[float, float], None] = None,
    initial_prompt: str = None,
    hotwords: str = None,
    language: str = None
) -> Dict[str, Any]:
    """
    Transcribes a WAV/MP3 audio file using Faster-Whisper.
    Enables GPU acceleration automatically based on config.
    Saves and caches the result in JSON format.
    """
    audio_path = Path(audio_path)
    if initial_prompt is None:
        initial_prompt = "Technical lecture transcript. Preserve the speaker's original language. Do not translate."

    hotwords = (hotwords or "").strip()
    language = (language or "").strip().lower() or None
    if language in {"auto", "detect", "auto-detect"}:
        language = None
    transcript_cache_path = _build_transcript_cache_path(audio_path, model_size, initial_prompt, hotwords, language)
    
    # Check if cached transcript already exists
    if transcript_cache_path.exists():
        try:
            print(f"Loading cached transcript from: {transcript_cache_path}")
        except Exception:
            print("[OK] Loading cached transcript.")
        with open(transcript_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    print(f"Loading Faster-Whisper model '{model_size}' on '{DEVICE}' with '{WHISPER_COMPUTE_TYPE}' and {CPU_THREADS} CPU threads...")
    
    # Initialize the model
    # On Windows, we need to handle potential library errors (like zlibwapi.dll) 
    # by gracefully falling back or providing descriptive errors.
    try:
        model = WhisperModel(
            model_size, 
            device=DEVICE, 
            compute_type=WHISPER_COMPUTE_TYPE, 
            cpu_threads=CPU_THREADS
        )
    except Exception as e:
        print(f"Error loading Faster-Whisper model on {DEVICE}: {e}")
        print("Falling back to standard CPU / float32 execution...")
        model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="float32", 
            cpu_threads=CPU_THREADS
        )
        
    try:
        print(f"Transcribing audio: {audio_path}...")
    except Exception:
        print("[*] Transcribing audio track...")
        
    # Run transcription with Voice Activity Detection (VAD).
    # Keep task="transcribe" and never force Vietnamese for English lectures.
    segments, info = model.transcribe(
        str(audio_path),
        task="transcribe",
        beam_size=5,
        temperature=0.0,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        word_timestamps=True,
        language=language,
        vad_filter=True,                   # Enable Voice Activity Detection (VAD) to skip long silences and MS Teams hums
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=700,
            speech_pad_ms=250
        ),
        condition_on_previous_text=False,  # Essential: prevents infinite repetition loops (e.g. repeating 'Bà bài' or 'lỡ lỡ')
        compression_ratio_threshold=2.2,   # Discards segments with high repetition
        no_speech_threshold=0.6,           # Filters out silence and ambient noise
        hallucination_silence_threshold=1.0,
        initial_prompt=initial_prompt,     # Guides spelling based on customized session type
        hotwords=hotwords or None
    )
    
    processed_segments = []
    full_text_list = []
    
    for segment in segments:
        segment_dict = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": _trim_repeated_tail(segment.text.strip()),
            "words": []
        }
        
        full_text_list.append(segment_dict["text"])
        
        # Add word level timestamps
        if segment.words:
            for word in segment.words:
                segment_dict["words"].append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "probability": round(word.probability, 4)
                })
                
        processed_segments.append(segment_dict)
        
        # Run callback to report real-time transcription progress
        if progress_callback:
            try:
                progress_callback(segment.end, info.duration)
            except Exception:
                pass
        
    result = {
        "metadata": {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 2),
            "model_size": model_size,
            "device": DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "requested_language": language or "auto",
            "initial_prompt_sha1": hashlib.sha1(initial_prompt.encode("utf-8")).hexdigest(),
            "hotwords_sha1": hashlib.sha1(hotwords.encode("utf-8")).hexdigest() if hotwords else None
        },
        "segments": processed_segments,
        "full_text": " ".join(full_text_list)
    }
    
    # Save cache file
    with open(transcript_cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    try:
        print(f"Saved transcribed text cache to: {transcript_cache_path}")
    except Exception:
        print("[OK] Saved transcribed text cache.")
    return result
