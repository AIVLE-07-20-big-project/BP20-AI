# POST /api/v1/campaign-logs, GET /api/v1/campaign-logs/quality
from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from app.schemas.campaign_log import CampaignLogQualityResponse, CampaignLogRequest
from app.services.response.campaign_logs import (
    DecisionNotApproved, DecisionNotFound, DecisionOwnershipMismatch, append_log, validate_logs,
)

router = APIRouter(tags=["캠페인 학습"])


@router.post("/campaign-logs")
def create_campaign_log(
    payload: CampaignLogRequest, x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict:
    try:
        return append_log(
            thread_id=payload.thread_id, executed=payload.executed,
            treatment_yyqu_cd=payload.treatment_yyqu_cd, net_sales_after=payload.net_sales_after,
            variable_cost_before=payload.variable_cost_before,
            variable_cost_after=payload.variable_cost_after,
            campaign_cost=payload.campaign_cost, measurement_days=payload.measurement_days,
            execution_started_at=payload.execution_started_at,
            execution_ended_at=payload.execution_ended_at,
            measurement_started_at=payload.measurement_started_at,
            measurement_ended_at=payload.measurement_ended_at,
            baseline_period_start=payload.baseline_period_start,
            baseline_period_end=payload.baseline_period_end,
            control_store_ids=payload.control_store_ids,
            user_id=x_user_id,
        )
    except DecisionNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except DecisionNotApproved as e:
        raise HTTPException(status_code=409, detail=str(e))
    except DecisionOwnershipMismatch as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/campaign-logs/quality", response_model=CampaignLogQualityResponse)
def get_campaign_log_quality() -> dict:
    return validate_logs()
