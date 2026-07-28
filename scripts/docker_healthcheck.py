"""Docker HEALTHCHECK 엔트리포인트.

api/worker/beat가 같은 이미지를 공유하므로 healthcheck도 역할에 맞게 갈라져야 한다.
BE compose가 컨테이너별로 CONTAINER_ROLE 환경변수를 지정한다(미지정 시 "api"로 동작).
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path


def check_api() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=3)
        return True
    except Exception:
        return False


def check_worker() -> bool:
    from app.celery_app import celery_app

    try:
        return bool(celery_app.control.ping(timeout=3))
    except Exception:
        return False


def check_beat() -> bool:
    # beat는 매 tick(60초)마다 스케줄 상태 파일을 갱신한다 — 최근 갱신 여부로 생존을 판단한다.
    candidates = [
        Path("celerybeat-schedule"),
        Path("celerybeat-schedule.dat"),
    ]
    now = time.time()
    return any(path.exists() and now - path.stat().st_mtime < 90 for path in candidates)


_CHECKS = {"api": check_api, "worker": check_worker, "beat": check_beat}


def main() -> int:
    role = os.environ.get("CONTAINER_ROLE", "api")
    check = _CHECKS.get(role, check_api)
    return 0 if check() else 1


if __name__ == "__main__":
    sys.exit(main())
