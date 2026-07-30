from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from app.schemas.ai_learning import AiSalesFeedbackRequest
from app.services.ai_learning import record_sales_feedback

router = APIRouter(tags=["AI 온라인 학습"])


@router.post("/ai-learning/sales-feedback")
def create_sales_feedback(
    payload: AiSalesFeedbackRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict:
    try:
        return record_sales_feedback(
            thread_id=payload.thread_id,
            before_sales=payload.before_sales,
            after_sales=payload.after_sales,
            measurement_days=payload.measurement_days,
            user_id=x_user_id,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

