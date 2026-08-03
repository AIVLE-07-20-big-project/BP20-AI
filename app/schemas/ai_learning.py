from __future__ import annotations

from pydantic import BaseModel, Field


class AiSalesFeedbackRequest(BaseModel):
    thread_id: str
    before_sales: float = Field(ge=0)
    after_sales: float
    measurement_days: int = Field(gt=0)

