# 비동기 분석 작업 상태 저장소
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from app.core.config import JOBS_DB


def _connect() -> sqlite3.Connection:
    JOBS_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(JOBS_DB), timeout=10)
    connection.row_factory = sqlite3.Row
    # API와 worker가 동시에 여는 파일이라 WAL + busy_timeout으로 "database is locked"을 줄인다.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id TEXT PRIMARY KEY,
            celery_task_id TEXT,
            user_id TEXT,
            status TEXT NOT NULL,
            analysis_id TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        )
        """
    )
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in row.keys()}


def create_job(job_id: str, *, user_id: str | None, celery_task_id: str | None = None) -> dict:
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO analysis_jobs (job_id, celery_task_id, user_id, status, created_at)
            VALUES (?, ?, ?, 'queued', ?)
            """,
            (job_id, celery_task_id, user_id, _now()),
        )
    job = get_job(job_id)
    assert job is not None
    return job


def get_job(job_id: str) -> dict | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM analysis_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def set_celery_task_id(job_id: str, celery_task_id: str) -> None:
    """디버깅과 작업 추적에 사용할 Celery 태스크 ID를 기록한다."""
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE analysis_jobs SET celery_task_id=? WHERE job_id=?",
            (celery_task_id, job_id),
        )


def mark_running(job_id: str) -> bool:
    """queued 또는 running 상태를 running으로 전환한다."""
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE analysis_jobs SET status='running', started_at=?
             WHERE job_id=? AND status IN ('queued', 'running')
            """,
            (_now(), job_id),
        )
    return cursor.rowcount > 0


def mark_completed(job_id: str, analysis_id: str) -> bool:
    """running 상태를 completed로 전환하고 성공 여부를 반환한다."""
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE analysis_jobs SET status='completed', analysis_id=?, completed_at=?
             WHERE job_id=? AND status='running'
            """,
            (analysis_id, _now(), job_id),
        )
    return cursor.rowcount > 0


def mark_failed(job_id: str, error_code: str, error_message: str) -> bool:
    """미완료 작업을 failed로 전환하고 성공 여부를 반환한다."""
    with closing(_connect()) as connection, connection:
        cursor = connection.execute(
            """
            UPDATE analysis_jobs SET status='failed', error_code=?, error_message=?, completed_at=?
             WHERE job_id=? AND status IN ('queued', 'running')
            """,
            (error_code, error_message, _now(), job_id),
        )
    return cursor.rowcount > 0


def cleanup_stale_queued(max_age_minutes: int = 5) -> list[str]:
    """오래 정체된 queued 작업을 실패 처리하고 ID 목록을 반환한다."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    with closing(_connect()) as connection:
        stale_ids = [
            row["job_id"]
            for row in connection.execute(
                "SELECT job_id FROM analysis_jobs WHERE status='queued' AND created_at < ?",
                (cutoff,),
            )
        ]
    cleaned = [job_id for job_id in stale_ids if mark_failed(job_id, "STALE_JOB", "잡이 queued 상태로 정체되어 자동 정리되었습니다")]
    return cleaned


def cleanup_stale_running(max_age_minutes: int = 15) -> list[str]:
    """오래 정체된 running 작업을 실패 처리하고 ID 목록을 반환한다."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    with closing(_connect()) as connection:
        stale_ids = [
            row["job_id"]
            for row in connection.execute(
                "SELECT job_id FROM analysis_jobs WHERE status='running' AND started_at < ?",
                (cutoff,),
            )
        ]
    cleaned = [
        job_id for job_id in stale_ids
        if mark_failed(job_id, "WORKER_LOST", "워커가 처리 도중 응답을 멈춰 자동 정리되었습니다")
    ]
    return cleaned
