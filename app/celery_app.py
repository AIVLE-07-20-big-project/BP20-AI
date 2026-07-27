"""Redis를 브로커와 결과 백엔드로 사용하는 Celery 앱.

실행(로컬, 이 저장소 루트에서):
    uvicorn app.main:app --reload
    celery -A app.celery_app worker --loglevel=info --pool=solo
    celery -A app.celery_app beat --loglevel=info

Windows solo 풀은 개발용이다. 병렬 처리와 타임아웃은 Linux prefork에서 검증한다.
작업 재전달(task_acks_late)은 아직 사용하지 않는다.
"""
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
    result_expires=3600,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    # 전송 또는 워커 중단으로 정체된 작업을 주기적으로 실패 처리한다.
    beat_schedule={
        "cleanup-stale-queued-jobs": {
            "task": "jobs.cleanup_stale",
            "schedule": 60.0,  # 1분마다
            "kwargs": {"max_age_minutes": 5},
        },
    },
)
