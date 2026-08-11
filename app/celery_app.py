from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows Celery 워커에서도 지연 로딩하는 scripts.*를 찾도록 프로젝트 루트를 추가한다.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from celery import Celery
from celery.schedules import crontab

# 워커는 main.py를 거치지 않으므로 여기서 .env를 로드한다.
from app.core import bootstrap  # noqa: F401
from app.core.huggingface_assets import sync_huggingface_assets

CONTAINER_ROLE = os.getenv("CONTAINER_ROLE", "worker").strip().lower()

# Beat는 스케줄 메시지만 발행하므로 대용량 매출 분석·RAG 자산이 필요하지 않다.
# API와 Worker만 S3 자산을 로컬 캐시에 동기화한다.
if CONTAINER_ROLE != "beat":
    sync_huggingface_assets()

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "bp20_ai",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.tasks.analysis", "app.tasks.jobs", "app.tasks.sales_target"],
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
        # AI_Agent_전환_가이드라인.md 4단계 — 완료/반려된 지 오래된 영업 타겟 그래프 thread의
        # 체크포인트 데이터를 매일 새벽 4시(Asia/Seoul)에 정리한다. 보존 기간은
        # SALES_TARGET_CHECKPOINT_RETENTION_DAYS(기본 30일)로 조절.
        "cleanup-sales-target-checkpoints": {
            "task": "sales_target.cleanup_checkpoints",
            "schedule": crontab(hour=4, minute=0),
        },
    },
)
