"""POST /api/v1/campaign-logs, GET /api/v1/campaign-logs/quality

계획 §1단계 — 승인된 agent-run(thread_id)의 체크포인트에서 결정 시점 값을 자동으로 읽어
실행 결과를 기록한다. 스키마·데이터 계약 검증은 app/services/response/campaign_logs.py 참고.
"""
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from app.schemas.campaign_log import CampaignLogQualityResponse, CampaignLogRequest
from app.services.response.campaign_logs import (
    DecisionNotApproved, DecisionNotFound, DecisionOwnershipMismatch, append_log, validate_logs,
)

router = APIRouter()


@router.post("/campaign-logs")
def create_campaign_log(
    payload: CampaignLogRequest, x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict:
    try:
        return append_log(
            thread_id=payload.thread_id, executed=payload.executed,
            treatment_yyqu_cd=payload.treatment_yyqu_cd, revenue_after=payload.revenue_after,
            user_id=x_user_id,
        )
    except DecisionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DecisionNotApproved as e:
        raise HTTPException(status_code=409, detail=str(e))
    except DecisionOwnershipMismatch as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("/campaign-logs/quality", response_model=CampaignLogQualityResponse)
def get_campaign_log_quality() -> dict:
    return validate_logs()
