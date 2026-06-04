import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Any
from app.config import PITCH_HIGH_PERCENTILE, VOLUME_HIGH_PERCENTILE, SPEECH_RATE_SLOW_THRESHOLD

def analyze_audio_acoustics(audio_path: Path, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes the acoustic features of each transcribed segment.
    Calculates:
      1. Volume (RMS Energy): to detect shouting or vocal force.
      2. Pitch (F0 using YIN): to detect high-intonation emphasis.
      3. Speech Rate: words per second to detect slow, deliberate speaking.
    
    Operates on segment-level audio slices to prevent memory overflow and speed up computation.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    print(f"Loading audio for acoustic feature extraction: {audio_path}...")
    # Load audio at a light sampling rate of 16kHz for speed
    y, sr = librosa.load(str(audio_path), sr=16000)
    
    # Store metrics per segment
    segment_metrics = []
    
    print(f"Analyzing {len(segments)} segments...")
    for idx, seg in enumerate(segments):
        start_sec = seg["start"]
        end_sec = seg["end"]
        duration = end_sec - start_sec
        
        # Avoid division by zero and tiny segments
        if duration < 0.1:
            segment_metrics.append({
                "index": idx,
                "rms": 0.0,
                "pitch": 0.0,
                "speech_rate": 0.0
            })
            continue
            
        # Map time to sample indices
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        
        # Extract audio slice
        y_slice = y[start_idx:end_idx]
        
        if len(y_slice) < 64:  # Too short for analysis
            segment_metrics.append({
                "index": idx,
                "rms": 0.0,
                "pitch": 0.0,
                "speech_rate": 0.0
            })
            continue
            
        # 1. Volume (RMS Energy)
        rms_frames = librosa.feature.rms(y=y_slice, frame_length=512, hop_length=256)
        mean_rms = float(np.mean(rms_frames)) if rms_frames.size > 0 else 0.0
        
        # 2. Pitch (Fundamental Frequency - F0)
        # Using YIN algorithm bounded by normal human voice frequencies (50Hz - 400Hz)
        mean_pitch = 0.0
        try:
            # Only calculate pitch if segment isn't complete silence
            if mean_rms > 0.005:
                # To prevent slow computation on very long segments, downsample slice if necessary
                pitches = librosa.yin(
                    y=y_slice, 
                    fmin=60, 
                    fmax=350, 
                    sr=sr,
                    frame_length=1024,
                    hop_length=512
                )
                # Filter out zero pitches (unvoiced frames)
                voiced_pitches = pitches[pitches > 0]
                if voiced_pitches.size > 0:
                    mean_pitch = float(np.mean(voiced_pitches))
        except Exception:
            # Fallback if YIN fails due to signal characteristics
            mean_pitch = 0.0
            
        # 3. Speech Rate (Words Per Second)
        num_words = len(seg.get("words", []))
        if num_words == 0:
            # Fallback: estimate from text length (5 characters ~ 1 word)
            num_words = len(seg["text"].split())
            
        speech_rate = float(num_words / duration)
        
        segment_metrics.append({
            "index": idx,
            "rms": mean_rms,
            "pitch": mean_pitch,
            "speech_rate": speech_rate
        })
        
    # Calculate global percentiles across all voiced segments to set dynamic, adaptive thresholds
    all_rms = [m["rms"] for m in segment_metrics if m["rms"] > 0]
    all_pitches = [m["pitch"] for m in segment_metrics if m["pitch"] > 0]
    
    rms_threshold = np.percentile(all_rms, VOLUME_HIGH_PERCENTILE) if all_rms else 0.05
    pitch_threshold = np.percentile(all_pitches, PITCH_HIGH_PERCENTILE) if all_pitches else 200.0
    
    print(f"Dynamic Thresholds -> RMS Volume threshold ({VOLUME_HIGH_PERCENTILE}th percentile): {rms_threshold:.4f}")
    print(f"Dynamic Thresholds -> Pitch threshold ({PITCH_HIGH_PERCENTILE}th percentile): {pitch_threshold:.1f} Hz")
    
    # Enriched segments with acoustic labels
    enriched_segments = []
    for idx, seg in enumerate(segments):
        metrics = segment_metrics[idx]
        
        is_loud = metrics["rms"] >= rms_threshold and metrics["rms"] > 0.01
        is_high_pitch = metrics["pitch"] >= pitch_threshold and metrics["pitch"] > 0
        is_slow = 0.5 < metrics["speech_rate"] <= SPEECH_RATE_SLOW_THRESHOLD
        
        # Calculate a composite acoustic score (0 to 3) representing vocal emphasis
        score = 0
        if is_loud: score += 1
        if is_high_pitch: score += 1
        if is_slow: score += 1
        
        # Check semantic clues in the text
        semantic_clues = ["quan trọng", "lưu ý", "thi", "lỗi", "chú ý", "nhớ", "important", "remember", "exam", "mistake"]
        text_lower = seg["text"].lower()
        has_semantic_clue = any(clue in text_lower for clue in semantic_clues)
        
        is_important = (score >= 2) or (score >= 1 and has_semantic_clue)
        
        enriched_seg = seg.copy()
        enriched_seg["acoustics"] = {
            "volume_rms": round(metrics["rms"], 4),
            "pitch_hz": round(metrics["pitch"], 1),
            "speech_rate_wps": round(metrics["speech_rate"], 2),
            "is_loud": bool(is_loud),
            "is_high_pitch": bool(is_high_pitch),
            "is_slow": bool(is_slow),
            "emphasis_score": score,
            "has_semantic_clue": has_semantic_clue,
            "is_important": bool(is_important)
        }
        enriched_segments.append(enriched_seg)
        
    print(f"Acoustic analysis complete. Flagged {sum(1 for s in enriched_segments if s['acoustics']['is_important'])} important moments.")
    return enriched_segments
