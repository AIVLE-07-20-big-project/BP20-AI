# 비동기 분석 잡의 상태 조회 — job_id 폴링용(계획 문서 §3)
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.services import jobs

router = APIRouter(tags=["작업 상태"])


def _assert_owner(job: dict, user_id: Optional[str]) -> None:
    owner = job.get("user_id")
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="해당 작업에 접근할 권한이 없습니다")


@router.get(
    "/jobs/{job_id}",
    summary="비동기 분석 잡 상태 조회",
    description="queued/running/completed/failed 상태와, 완료 시 analysis_id를 반환합니다.",
)
def get_job_status(job_id: str, x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> dict:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    _assert_owner(job, x_user_id)
    return job
