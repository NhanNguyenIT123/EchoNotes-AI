import json
import sys
from pathlib import Path

# Configure stdout and stderr for UTF-8 Windows terminal support
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import app.config  # Activates local CUDA DLL path injections
from app.utils.video import extract_audio_from_video
from app.utils.audio import transcribe_audio
from app.utils.acoustic import analyze_audio_acoustics
from app.utils.vision import detect_slide_transitions

def main():
    video_path = Path("Meeting in General -20260515_085012-Bản ghi cuộc họp.mp4")
    if not video_path.exists():
        print(f"[-] Error: Video file not found at: {video_path.absolute()}")
        return
        
    print("=" * 65)
    print("🎙️  EchoNotes AI - Offline Multimodal Preprocessing Pipeline  🎙️")
    print("=" * 65)
    print(f"[*] Ingesting Target Video: {video_path.name}")
    
    # Stage 1: Audio extraction
    print("\n[+] Stage 1/4: Extracting audio track...")
    audio_path = extract_audio_from_video(video_path)
    print(f"[OK] Audio extracted: {audio_path.name}")
    
    # Stage 2: Whisper Transcribe
    print("\n[+] Stage 2/4: Transcribing speech with optimized CPU Faster-Whisper...")
    raw_transcript = transcribe_audio(audio_path, model_size="base")
    print("[OK] Speech-to-text transcription complete.")
    
    # Stage 3: Acoustic Profiling
    enriched_cache_path = Path("data/processed/transcripts/enriched_segments_cache.json")
    if enriched_cache_path.exists():
        print(f"\n[+] Stage 3/4: Loading cached acoustic analysis from {enriched_cache_path.name}...")
        with open(enriched_cache_path, "r", encoding="utf-8") as f:
            enriched_segments = json.load(f)
        print("[OK] Acoustic analysis loaded from cache.")
    else:
        print("\n[+] Stage 3/4: Analyzing vocal acoustics (pitch, volume, speech rate)...")
        enriched_segments = analyze_audio_acoustics(audio_path, raw_transcript["segments"])
        
        # Cache enriched segments to disk
        with open(enriched_cache_path, "w", encoding="utf-8") as f:
            json.dump(enriched_segments, f, ensure_ascii=False, indent=2)
        print(f"[OK] Acoustic analysis complete. Cache saved: {enriched_cache_path.name}")
    
    # Stage 4: Slide transition & OCR
    slides_cache_path = Path("data/processed/transcripts/slides_cache.json")
    if slides_cache_path.exists():
        print(f"\n[+] Stage 4/4: Loading cached slide keyframes from {slides_cache_path.name}...")
        with open(slides_cache_path, "r", encoding="utf-8") as f:
            slide_keyframes = json.load(f)
        print("[OK] Slide transitions loaded from cache.")
    else:
        print("\n[+] Stage 4/4: Scanning video frames for slide transitions & running OCR...")
        slide_keyframes = detect_slide_transitions(video_path, progress_callback=None)
        
        # Cache slide keyframes to disk
        with open(slides_cache_path, "w", encoding="utf-8") as f:
            json.dump(slide_keyframes, f, ensure_ascii=False, indent=2)
        print(f"[OK] Slide transitions complete. Cache saved: {slides_cache_path.name}")
    
    print("\n" + "=" * 65)
    print("🎉  [SUCCESS] All pipeline stages successfully processed offline!  🎉")
    print("=" * 65)

if __name__ == "__main__":
    main()
