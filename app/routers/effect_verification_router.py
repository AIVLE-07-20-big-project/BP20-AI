from fastapi import APIRouter, HTTPException

from app.schemas.effect_verification_schema import (
    EffectVerificationFromAnalysesRequest,
    EffectVerificationFromAnalysesResponse,
    EffectVerificationRequest,
    EffectVerificationResponse,
)
from app.services import analyses
from app.services.effect_verification_service import (
    MissingVerificationAggregatesError,
    verify_effect,
    verify_effect_from_analyses,
)


router = APIRouter(
    prefix="/effect-verification",
    tags=["Effect Verification"],
)


@router.get("/")
def test():
    return {
        "status": "success",
        "message": "Effect Verification API",
    }


@router.post(
    "/verify",
    response_model=EffectVerificationResponse,
)
def verify_recommendation_effect(
    request: EffectVerificationRequest,
) -> EffectVerificationResponse:
    return verify_effect(request)


# 매출형 전략검증 — 이미 /analyses로 저장된 적용전/적용후 분석 두 건을 비교한다.
@router.post(
    "/verify-from-analyses",
    response_model=EffectVerificationFromAnalysesResponse,
)
def verify_recommendation_effect_from_analyses(
    request: EffectVerificationFromAnalysesRequest,
) -> EffectVerificationFromAnalysesResponse:
    before_analysis = analyses.get_analysis(request.before_analysis_id)
    if before_analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {request.before_analysis_id}")
    after_analysis = analyses.get_analysis(request.after_analysis_id)
    if after_analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {request.after_analysis_id}")

    try:
        return verify_effect_from_analyses(
            before_analysis, after_analysis,
            store_id=request.store_id, recommendation_id=request.recommendation_id,
            start_hour=request.start_hour, end_hour=request.end_hour,
            period_days=request.period_days,
        )
    except (MissingVerificationAggregatesError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
