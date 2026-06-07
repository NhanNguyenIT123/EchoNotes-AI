from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from app.job_queue import update_job_status


def run_analysis_job(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    RQ entry point for production-style background analysis.

    The job writes status to data/jobs/*.json and persists pipeline outputs to the
    usual cache/database paths, so the FastAPI process can reload results without
    sharing Python memory with the worker.
    """
    from backend.main import run_pipeline_thread, state

    update_job_status(job_id, status="processing", stage="Worker started", progress=3)
    try:
        run_pipeline_thread(
            video_path=Path(payload["video_path"]),
            whisper_model=payload["whisper_model"],
            initial_prompt=payload.get("initial_prompt", ""),
            teams_transcript_path=Path(payload["teams_transcript_path"]) if payload.get("teams_transcript_path") else None,
            hotwords=payload.get("hotwords", ""),
            use_glossary=bool(payload.get("use_glossary", True)),
            visual_mode=payload.get("visual_mode", "Fast: capture keyframes, no OCR"),
            ssim_thresh=float(payload.get("ssim_thresh", 0.94)),
            min_keyframe_gap_sec=float(payload.get("min_keyframe_gap_sec", 20)),
            max_keyframes=int(payload.get("max_keyframes", 80)),
            frame_check_interval_sec=float(payload.get("frame_check_interval_sec", 10)),
            analyze_acoustics=bool(payload.get("analyze_acoustics", False)),
            speech_language=payload.get("speech_language", "en"),
            vision_model=payload.get("vision_model", "llava:7b"),
            project_id=payload.get("project_id"),
            diarization_enabled=bool(payload.get("diarization_enabled", False)),
            job_id=job_id,
        )
        if state.get("status") == "error":
            raise RuntimeError(state.get("error") or "Pipeline failed")
        update_job_status(
            job_id,
            status="completed",
            stage="Analysis complete",
            progress=100,
            metrics=state.get("last_metrics", {}),
        )
        return {"status": "completed", "metrics": state.get("last_metrics", {})}
    except Exception as exc:
        update_job_status(job_id, status="error", stage="Worker failed", progress=0, error=str(exc))
        raise
