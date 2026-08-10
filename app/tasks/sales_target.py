"""Celery beat가 호출하는 신규 가맹점 영업 타겟 그래프 체크포인트 정리 태스크.

app/tasks/jobs.py의 cleanup_stale_jobs와 동일한 패턴(순수 로직은 별도 모듈에 두고, 이 파일은
Celery 태스크 등록 + 설정값 주입만 담당)."""
from __future__ import annotations

from app.celery_app import celery_app


@celery_app.task(name="sales_target.cleanup_checkpoints")
def cleanup_sales_target_checkpoints(older_than_days: int | None = None) -> list[str]:
    from app.core.config import SALES_TARGET_CHECKPOINT_RETENTION_DAYS
    from app.sales_target.cleanup import delete_old_terminal_threads
    from app.sales_target.graph import get_checkpointer

    days = older_than_days if older_than_days is not None else SALES_TARGET_CHECKPOINT_RETENTION_DAYS
    return delete_old_terminal_threads(get_checkpointer(), older_than_days=days)
