from fastapi import APIRouter, Depends
from app.schemas.review import ReviewRequest, ReviewAnalyzeResponse
from app.services.absa_service import ABSAService

router = APIRouter()

@router.post("/analyze", response_model=ReviewAnalyzeResponse)
def analyze_review_endpoint(payload: ReviewRequest, service: ABSAService = Depends()):
    return service.analyze_review(payload.human_prompt, payload.review_id)