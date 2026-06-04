import cv2
import easyocr
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable
from skimage.metrics import structural_similarity as ssim
from app.config import FRAMES_DIR, SSIM_THRESHOLD, FRAME_CHECK_INTERVAL, OCR_DEVICE

def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH_MM_SS representation."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}_{m:02d}_{s:02d}"

def detect_slide_transitions(
    video_path: Path, 
    progress_callback: Callable[[float], None] = None,
    ssim_threshold: float = SSIM_THRESHOLD,
    frame_check_interval: float = FRAME_CHECK_INTERVAL,
    min_transition_gap_sec: float = 8.0,
    run_ocr: bool = True,
    max_slides: int = 120
) -> List[Dict[str, Any]]:
    """
    Scans a video file, downsamples frames, and uses Structural Similarity Index (SSIM)
    to detect when a slide transition occurs. Saves screenshots of transitions and
    extracts visual text using EasyOCR with Vietnamese & English support.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
        
    reader = None
    if run_ocr:
        print(f"Initializing EasyOCR Reader (Vietnamese & English) on Device (GPU={OCR_DEVICE})...")
        # Initialize EasyOCR reader (handles both English and Vietnamese)
        reader = easyocr.Reader(['vi', 'en'], gpu=OCR_DEVICE)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video Details -> Duration: {duration:.1f}s, FPS: {fps:.2f}, Total Frames: {total_frames}")
    
    # We sample every frame_check_interval seconds.
    frame_step = max(1, int(fps * frame_check_interval))
    
    prev_gray = None
    slide_keyframes = []
    
    # Read first frame as initial baseline keyframe
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
        
    # Resize and convert to grayscale for super fast SSIM comparison
    # Downsampling to 640x360 is perfect for structural changes and runs 10x faster
    prev_resized = cv2.resize(first_frame, (640, 360))
    prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
    
    # Save the initial frame as first slide
    first_time_sec = 0.0
    first_img_name = f"slide_{format_timestamp(first_time_sec)}.jpg"
    first_img_path = FRAMES_DIR / first_img_name
    cv2.imwrite(str(first_img_path), first_frame)
    
    if reader:
        print("Extracting text from first frame slide...")
        ocr_result = reader.readtext(str(first_img_path), detail=0)
        ocr_text = " ".join(ocr_result)
    else:
        ocr_result = []
        ocr_text = ""
    
    slide_keyframes.append({
        "timestamp_sec": first_time_sec,
        "timestamp_formatted": "00:00:00",
        "image_path": str(first_img_path),
        "ocr_text": ocr_text
    })
    
    frame_idx = frame_step
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        time_sec = frame_idx / fps
        
        # Report progress
        if progress_callback:
            progress_callback(time_sec / duration)
            
        # Resize and grayscale comparison frame
        resized = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Compute Structural Similarity Index (SSIM)
        score, _ = ssim(prev_gray, gray, full=True)
        
        # If similarity drops below threshold, a new slide has transitioned!
        last_slide_time = slide_keyframes[-1]["timestamp_sec"] if slide_keyframes else -float("inf")
        if score < ssim_threshold and (time_sec - last_slide_time) >= min_transition_gap_sec:
            img_name = f"slide_{format_timestamp(time_sec)}.jpg"
            img_path = FRAMES_DIR / img_name
            cv2.imwrite(str(img_path), frame)
            
            # Extract OCR text offline
            if reader:
                ocr_result = reader.readtext(str(img_path), detail=0)
                ocr_text = " ".join(ocr_result)
            else:
                ocr_result = []
                ocr_text = ""
            
            # Print slide transition notification
            print(f"Slide transition detected at {time_sec:.1f}s (SSIM: {score:.3f}). OCR words: {len(ocr_result)}")
            
            slide_keyframes.append({
                "timestamp_sec": time_sec,
                "timestamp_formatted": format_timestamp(time_sec).replace("_", ":"),
                "image_path": str(img_path),
                "ocr_text": ocr_text
            })
            
            # Update baseline frame
            prev_gray = gray

            if max_slides and len(slide_keyframes) >= max_slides:
                print(f"Reached max_slides={max_slides}; stopping slide scan early.")
                break
            
        frame_idx += frame_step
        
    cap.release()
    print(f"Slide transition detection complete. Extracted {len(slide_keyframes)} slides.")
    return slide_keyframes
