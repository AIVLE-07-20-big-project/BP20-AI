from fastapi import APIRouter

from app.schemas.effect_verification_schema import (
    EffectVerificationRequest,
    EffectVerificationResponse,
)
from app.services.effect_verification_service import verify_effect


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