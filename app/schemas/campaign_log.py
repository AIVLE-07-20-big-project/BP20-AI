from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel


class CampaignLogRequest(BaseModel):
    thread_id: str
    executed: bool
    treatment_yyqu_cd: int
    # Reward v2 계산에 필요 — 없으면 reward_status=incomplete로 기록되고 학습에서 제외
    net_sales_after: Optional[float] = None
    variable_cost_before: Optional[float] = None
    variable_cost_after: Optional[float] = None
    campaign_cost: Optional[float] = None
    measurement_days: Optional[int] = None
    execution_started_at: Optional[datetime] = None
    execution_ended_at: Optional[datetime] = None
    measurement_started_at: Optional[date] = None
    measurement_ended_at: Optional[date] = None
    baseline_period_start: Optional[date] = None
    baseline_period_end: Optional[date] = None
    control_store_ids: Optional[list[str]] = None


class CampaignLogResponse(BaseModel):
    decision_id: str

    model_config = {"extra": "allow"}


class CampaignLogQualityResponse(BaseModel):
    총행수: int
    유효행수: int
    제외행수: int
    제외사유: dict[str, Any]

    model_config = {"extra": "allow"}
