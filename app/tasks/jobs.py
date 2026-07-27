"""Celery beat가 호출하는 정체 작업 정리 태스크."""
from __future__ import annotations

from app.celery_app import celery_app


@celery_app.task(name="jobs.cleanup_stale")
def cleanup_stale_jobs(max_age_minutes: int = 5) -> list[str]:
    from app.core import uploads
    from app.services import jobs

    cleaned = jobs.cleanup_stale_queued(max_age_minutes=max_age_minutes)
    for job_id in cleaned:
        uploads.delete_job_upload(job_id)
    return cleaned
