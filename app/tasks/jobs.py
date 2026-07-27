"""잡 정리 태스크 — celery beat가 주기 호출한다.

docs/speed/celery-async-development-plan.md §2.1b.
enqueue 보상(§2.1) 자체가 실패하면 잡이 영원히 queued로 남을 수 있어, 정체된 잡을
주기적으로 청소하는 마지막 안전망이다.
"""
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
