# 비동기 분석 잡 상태 저장소 — analysis_jobs 테이블이 API가 제공하는 공식 상태다.
# docs/speed/celery-async-development-plan.md §2.2
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from app.core.config import JOBS_DB


def _connect() -> sqlite3.Connection:
    JOBS_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(JOBS_DB))
    connection.row_factory = sqlite3.Row
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
    """디버깅·상관관계 추적용 메타데이터 기록. 상태 전이가 아니므로 조건부 UPDATE가 아니다."""
    with closing(_connect()) as connection, connection:
        connection.execute(
            "UPDATE analysis_jobs SET celery_task_id=? WHERE job_id=?",
            (celery_task_id, job_id),
        )


def mark_running(job_id: str) -> bool:
    """queued 또는 running -> running. running도 허용해 재배달된 잡이 이전 시도의
    running을 만나 실행 불가가 되는 것을 막는다(계획 §2.2)."""
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
    """running -> completed. 0행이면(취소·재배달 경합에서 짐) 결과를 쓰지 않는다."""
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
    """queued 또는 running -> failed. 이미 completed/failed면 아무것도 하지 않는다."""
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
    """N분 이상 queued로 정체된 잡을 failed(STALE_JOB)로 정리하고 job_id 목록을 반환한다.

    enqueue 보상(§2.1) 자체가 실패했을 때의 마지막 안전망 — celery beat가 주기 호출한다
    (§2.1b). 파일 정리는 호출자(태스크)가 반환된 job_id로 uploads.delete_job_upload를
    부른다 — 이 함수는 DB 상태 전이만 책임진다.
    """
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
