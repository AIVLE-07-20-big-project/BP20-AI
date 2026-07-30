from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows Celery 워커에서도 지연 로딩하는 scripts.*를 찾도록 프로젝트 루트를 추가한다.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from celery import Celery

# 워커는 main.py를 거치지 않으므로 여기서 .env를 로드한다.
from app.core import bootstrap  # noqa: F401

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "bp20_ai",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.tasks.analysis", "app.tasks.jobs"],
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=540,
    worker_prefetch_multiplier=1,
    # 워커가 처리 도중 죽어도 메시지가 유실되지 않도록 완료 후 ack한다.
    # run_analysis_task는 mark_running/INSERT OR IGNORE로 재실행에도 안전하다.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    # 전송 또는 워커 중단으로 정체된 작업을 주기적으로 실패 처리한다.
    # 단일 워커(prefetch=1)에서는 작업이 직렬로 처리되므로, 대기가 몰리면 정상 queued 작업도
    # 5분을 쉽게 넘길 수 있다 — task_time_limit(10분) 기준 몇 건 밀려도 견디도록 기본값을 늘렸다.
    beat_schedule={
        "cleanup-stale-queued-jobs": {
            "task": "jobs.cleanup_stale",
            "schedule": 60.0,
            "kwargs": {
                "queued_max_age_minutes": int(os.getenv("CELERY_QUEUED_STALE_MINUTES", "20")),
                "running_max_age_minutes": int(os.getenv("CELERY_RUNNING_STALE_MINUTES", "15")),
            },
        },
        "purge-expired-uploads": {
            "task": "jobs.purge_expired_uploads",
            "schedule": 86400.0,
            "kwargs": {"max_age_days": int(os.getenv("UPLOAD_PURGE_MAX_AGE_DAYS", "7"))},
        },
    },
)
