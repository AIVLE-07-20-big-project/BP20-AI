"""Celery beat가 호출하는 정리 태스크 모음: 정체 작업 정리, 만료된 업로드 원본 삭제."""
from __future__ import annotations

from app.celery_app import celery_app


@celery_app.task(name="jobs.cleanup_stale")
def cleanup_stale_jobs(queued_max_age_minutes: int = 5, running_max_age_minutes: int = 15) -> list[str]:
    from app.core import uploads
    from app.services import jobs

    cleaned = jobs.cleanup_stale_queued(max_age_minutes=queued_max_age_minutes)
    cleaned += jobs.cleanup_stale_running(max_age_minutes=running_max_age_minutes)
    for job_id in cleaned:
        uploads.delete_job_upload(job_id)
    return cleaned


@celery_app.task(name="jobs.purge_expired_uploads")
def purge_expired_uploads(max_age_days: int = 7) -> int:
    """실패 후 재현 목적으로 남겨둔 업로드 원본을 보존기간 경과 시 정리한다."""
    from app.core import uploads

    return uploads.purge_expired_uploads(max_age_days=max_age_days)
