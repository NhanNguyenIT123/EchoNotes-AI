from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DEFAULT_DATABASE_URL = "postgresql+psycopg://echonotes:echonotes@127.0.0.1:5432/echonotes"
DATABASE_URL = os.getenv("ECHONOTES_DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    pass


class LectureProject(Base):
    __tablename__ = "lecture_projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_mode: Mapped[str] = mapped_column(String(64), default="upload")
    status: Mapped[str] = mapped_column(String(64), default="draft")
    video_filename: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    slides_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    report_markdown: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


engine_kwargs: Dict[str, Any] = {"pool_pre_ping": True, "future": True}
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs["connect_args"] = {"connect_timeout": 2}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def db_health() -> Dict[str, Any]:
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("select 1")
        return {"connected": True, "url": _redact_url(DATABASE_URL)}
    except SQLAlchemyError as exc:
        return {"connected": False, "url": _redact_url(DATABASE_URL), "error": str(exc)}


def _redact_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    return f"{scheme}://***@{rest.split('@', 1)[1]}"


@contextmanager
def db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def project_to_dict(project: LectureProject) -> Dict[str, Any]:
    return {
        "id": project.id,
        "title": project.title,
        "source_mode": project.source_mode,
        "status": project.status,
        "video_filename": project.video_filename,
        "video_path": project.video_path,
        "transcript_path": project.transcript_path,
        "metrics": _json_load(project.metrics_json, {}),
        "has_transcript": bool(project.transcript_json),
        "has_slides": bool(project.slides_json),
        "has_report": bool(project.report_markdown),
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    }


def create_project_for_video(video_path: Path, source_mode: str = "upload") -> LectureProject:
    with db_session() as session:
        project = LectureProject(
            id=str(uuid.uuid4()),
            title=video_path.stem,
            source_mode=source_mode,
            status="draft",
            video_filename=video_path.name,
            video_path=str(video_path),
        )
        session.add(project)
        session.flush()
        session.refresh(project)
        return project


def update_project_input(project_id: str, **fields: Any) -> None:
    with db_session() as session:
        project = session.get(LectureProject, project_id)
        if not project:
            return
        for key, value in fields.items():
            if hasattr(project, key):
                setattr(project, key, value)
        project.updated_at = datetime.now(timezone.utc)


def list_projects(limit: int = 50) -> List[Dict[str, Any]]:
    with db_session() as session:
        rows = session.scalars(
            select(LectureProject)
            .order_by(LectureProject.updated_at.desc())
            .limit(limit)
        ).all()
        return [project_to_dict(row) for row in rows]


def get_project(project_id: str) -> Optional[LectureProject]:
    with db_session() as session:
        project = session.get(LectureProject, project_id)
        if not project:
            return None
        session.expunge(project)
        return project


def get_project_payload(project_id: str) -> Optional[Dict[str, Any]]:
    project = get_project(project_id)
    if not project:
        return None
    payload = project_to_dict(project)
    payload.update(
        {
            "transcript": _json_load(project.transcript_json, []),
            "slides": _json_load(project.slides_json, []),
            "report_markdown": project.report_markdown or "",
        }
    )
    return payload


def save_project_artifacts(
    project_id: str,
    transcript: Optional[List[Dict[str, Any]]] = None,
    slides: Optional[List[Dict[str, Any]]] = None,
    report_markdown: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
) -> None:
    with db_session() as session:
        project = session.get(LectureProject, project_id)
        if not project:
            return
        if transcript is not None:
            project.transcript_json = _json_dump(transcript)
        if slides is not None:
            project.slides_json = _json_dump(slides)
        if report_markdown is not None:
            project.report_markdown = report_markdown
        if metrics is not None:
            project.metrics_json = _json_dump(metrics)
        if status:
            project.status = status
        project.updated_at = datetime.now(timezone.utc)


def add_chat_message(project_id: str, role: str, content: str) -> None:
    with db_session() as session:
        session.add(
            ChatMessage(
                id=str(uuid.uuid4()),
                project_id=project_id,
                role=role,
                content=content,
            )
        )


def list_chat_messages(project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    with db_session() as session:
        rows = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.project_id == project_id)
            .order_by(ChatMessage.created_at.asc())
            .limit(limit)
        ).all()
        return [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
