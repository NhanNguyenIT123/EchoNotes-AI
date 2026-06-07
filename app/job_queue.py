from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import DATA_DIR

JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def create_job(kind: str, payload: Dict[str, Any]) -> str:
    job_id = str(uuid.uuid4())
    write_job_status(
        job_id,
        {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "stage": "Queued",
            "progress": 0,
            "payload": payload,
            "created_at": time.time(),
            "updated_at": time.time(),
            "error": None,
            "backend": configured_job_backend(),
        },
    )
    return job_id


def configured_job_backend() -> str:
    return os.getenv("ECHONOTES_JOB_BACKEND", "local").strip().lower() or "local"


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def read_job_status(job_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not job_id:
        return None
    path = job_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_job_status(job_id: str, status: Dict[str, Any]) -> None:
    status["updated_at"] = time.time()
    job_path(job_id).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def update_job_status(job_id: str, **updates: Any) -> None:
    current = read_job_status(job_id) or {"id": job_id}
    current.update(updates)
    write_job_status(job_id, current)


def latest_job_status() -> Optional[Dict[str, Any]]:
    files = sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return read_job_status(files[0].stem) if files else None


def enqueue_rq_job(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    from redis import Redis
    from rq import Queue

    redis_url = os.getenv("ECHONOTES_REDIS_URL", "redis://127.0.0.1:6379/0")
    queue_name = os.getenv("ECHONOTES_RQ_QUEUE", "echonotes")
    redis_conn = Redis.from_url(redis_url)
    queue = Queue(queue_name, connection=redis_conn)
    rq_job = queue.enqueue(
        "backend.queue_jobs.run_analysis_job",
        job_id,
        payload,
        job_timeout=int(os.getenv("ECHONOTES_RQ_JOB_TIMEOUT", "14400")),
        result_ttl=86400,
        failure_ttl=86400,
    )
    update_job_status(job_id, backend="redis-rq", rq_job_id=rq_job.id)
    return {"backend": "redis-rq", "rq_job_id": rq_job.id, "queue": queue_name}
