# 비동기 분석 작업 상태 저장소
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from app.core.config import JOBS_DB
from app.core.database import connect, execute, fetchall, fetchone, using_mysql


def _affected_rows(result: object) -> int:
    """Return affected-row count for both sqlite3 and PyMySQL execute()."""
    if isinstance(result, int):
        return result
    return int(getattr(result, "rowcount", 0))


@contextmanager
def _connection():
    JOBS_DB.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(JOBS_DB)) as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        if not using_mysql():
            execute(connection, "PRAGMA journal_mode=WAL")
            execute(connection, "PRAGMA busy_timeout=10000")
        execute(connection, f"""
            CREATE TABLE IF NOT EXISTS {table} (
                job_id VARCHAR(64) PRIMARY KEY,
                celery_task_id VARCHAR(255),
                user_id VARCHAR(255),
                status VARCHAR(32) NOT NULL,
                analysis_id VARCHAR(64),
                error_code VARCHAR(64),
                error_message TEXT,
                created_at VARCHAR(40) NOT NULL,
                started_at VARCHAR(40),
                completed_at VARCHAR(40)
            )
        """)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return dict(row) if isinstance(row, dict) else {key: row[key] for key in row.keys()}


def create_job(job_id: str, *, user_id: str | None, celery_task_id: str | None = None) -> dict:
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        insert = "INSERT IGNORE" if using_mysql() else "INSERT OR IGNORE"
        execute(connection, f"""
            {insert} INTO {table} (job_id, celery_task_id, user_id, status, created_at)
            VALUES (?, ?, ?, 'queued', ?)
        """, (job_id, celery_task_id, user_id, _now()))
    job = get_job(job_id)
    assert job is not None
    return job


def get_job(job_id: str) -> dict | None:
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        row = fetchone(connection, f"SELECT * FROM {table} WHERE job_id = ?", (job_id,))
    return _row_to_dict(row) if row else None


def set_celery_task_id(job_id: str, celery_task_id: str) -> None:
    """디버깅과 작업 추적에 사용할 Celery 태스크 ID를 기록한다."""
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        execute(connection, f"UPDATE {table} SET celery_task_id=? WHERE job_id=?", (celery_task_id, job_id))


def mark_running(job_id: str) -> bool:
    """queued 또는 running 상태를 running으로 전환한다."""
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        cursor = execute(connection, """
            UPDATE {table} SET status='running', started_at=?
             WHERE job_id=? AND status IN ('queued', 'running')
        """.format(table=table), (_now(), job_id))
    return _affected_rows(cursor) > 0


def mark_completed(job_id: str, analysis_id: str) -> bool:
    """running 상태를 completed로 전환하고 성공 여부를 반환한다."""
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        cursor = execute(connection, """
            UPDATE {table} SET status='completed', analysis_id=?, completed_at=?
             WHERE job_id=? AND status='running'
        """.format(table=table), (analysis_id, _now(), job_id))
    return _affected_rows(cursor) > 0


def mark_failed(job_id: str, error_code: str, error_message: str) -> bool:
    """미완료 작업을 failed로 전환하고 성공 여부를 반환한다."""
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        cursor = execute(connection, """
            UPDATE {table} SET status='failed', error_code=?, error_message=?, completed_at=?
             WHERE job_id=? AND status IN ('queued', 'running')
        """.format(table=table), (error_code, error_message, _now(), job_id))
    return _affected_rows(cursor) > 0


def _cleanup(status: str, max_age_minutes: int, error_code: str, message: str, field: str) -> list[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    with _connection() as connection:
        table = "ai_analysis_jobs" if using_mysql() else "analysis_jobs"
        rows = fetchall(connection, f"SELECT job_id FROM {table} WHERE status=? AND {field} < ?", (status, cutoff))
    return [job_id for job_id in (row["job_id"] for row in rows) if mark_failed(job_id, error_code, message)]


def cleanup_stale_queued(max_age_minutes: int = 5) -> list[str]:
    return _cleanup("queued", max_age_minutes, "STALE_JOB", "잡이 queued 상태로 정체되어 자동 정리되었습니다", "created_at")


def cleanup_stale_running(max_age_minutes: int = 15) -> list[str]:
    return _cleanup("running", max_age_minutes, "WORKER_LOST", "워커가 처리 도중 응답을 멈춰 자동 정리되었습니다", "started_at")
