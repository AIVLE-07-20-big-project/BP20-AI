from fastapi import APIRouter, Request, HTTPException
from schemas.review import ReviewRequest, ReviewResponse
from services.absa_service import ABSAService

router = APIRouter()

@router.post("/predict", response_model=ReviewResponse, summary="리뷰 ABSA 감성 분석")
async def predict_review(payload: ReviewRequest, request: Request):
    review_text = payload.review_text.strip()
    if not review_text:
        raise HTTPException(status_code=400, detail="리뷰 텍스트를 입력해주세요.")

    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device

    service = ABSAService(model=model, tokenizer=tokenizer, device=device)
    results = service.predict(review_text)

    return ReviewResponse(review_text=review_text, results=results)