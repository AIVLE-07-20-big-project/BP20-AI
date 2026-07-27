"""Celery 앱 정의 — 무거운 AI 작업을 워커 프로세스로 오프로드하는 진입점.

브로커·결과 백엔드는 Redis(기본값 localhost:6379, BE compose의 redis 컨테이너).
상세 배경은 docs/speed/celery-async-development-plan.md 참고.

실행 방법(로컬 개발 전용):
    # 1) Redis 컨테이너 (BP20-BE)
    #    docker compose up -d redis
    # 2) API 서버
    uvicorn app.main:app --reload
    # 3) 워커 (별도 프로세스). Windows는 prefork가 불안정하므로 solo 풀:
    celery -A app.celery_app worker --loglevel=info --pool=solo
    # 4) beat (또 별도 프로세스, 정체된 queued 잡 청소용 — §2.1b):
    celery -A app.celery_app beat --loglevel=info
    # 연결 확인:
    #   python -c "from app.tasks.analysis import ping; print(ping.delay().get(timeout=10))"

주의: 위 solo 풀은 개발 확인용이며 운영 기준으로 삼지 않는다. solo는 병렬 처리가 없고,
Windows에는 SIGALRM이 없어 아래 task_time_limit/task_soft_time_limit이 실제로 강제되지
않는다. 타임아웃·동시성·장애 복구 검증은 Linux prefork 환경에서 수행한다(계획 문서 §3.8).

미구현(계획 문서 §3.5): 워커 장애 시 재배달을 위한 task_acks_late,
task_reject_on_worker_lost, broker_transport_options={"visibility_timeout": 660}.
이들은 at-least-once가 되므로 job_id 기반 멱등성을 먼저 구현한 뒤에 켠다.
"""
from __future__ import annotations

import os

from celery import Celery

# 워커는 main.py를 거치지 않으므로 여기서 .env를 직접 로드한다(OPENAI_API_KEY 등).
from app.core import bootstrap  # noqa: F401

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "bp20_ai",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.tasks.analysis", "app.tasks.jobs"],  # 태스크 모듈 등록(자동 발견)
)

celery_app.conf.update(
    task_track_started=True,        # queued → running(STARTED) 상태를 결과 백엔드에 노출
    task_time_limit=600,            # 하드 타임아웃 10분(초과 시 워커 강제 종료)
    task_soft_time_limit=540,       # 소프트 타임아웃 9분(예외로 정리 기회 부여)
    worker_prefetch_multiplier=1,   # 무거운 작업은 워커당 한 번에 하나씩만 선점
    result_expires=3600,            # 결과는 1시간 후 만료(잡 상태 조회용, 결과물은 별도 저장)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    enable_utc=True,
    # 정체된 queued 잡 청소(§2.1b). 하드 타임아웃(10분)보다 훨씬 짧게 잡아 오탐을 줄인다.
    beat_schedule={
        "cleanup-stale-queued-jobs": {
            "task": "jobs.cleanup_stale",
            "schedule": 60.0,  # 1분마다
            "kwargs": {"max_age_minutes": 5},
        },
    },
)
