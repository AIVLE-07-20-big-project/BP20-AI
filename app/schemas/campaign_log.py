from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class CampaignLogRequest(BaseModel):
    thread_id: str
    executed: bool
    treatment_yyqu_cd: int
    revenue_after: Optional[float] = None


class CampaignLogResponse(BaseModel):
    decision_id: str

    model_config = {"extra": "allow"}


class CampaignLogQualityResponse(BaseModel):
    총행수: int
    유효행수: int
    제외행수: int
    제외사유: dict[str, Any]

    model_config = {"extra": "allow"}
