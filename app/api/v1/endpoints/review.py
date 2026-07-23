from fastapi import APIRouter, Request, HTTPException
from typing import List
from schemas.review import ReviewRequest, ReviewResponse
from services.absa_service import ABSAService

router = APIRouter()

@router.post("/analyze", response_model=List[ReviewResponse], summary="리뷰 ABSA 감성 분석 (배치)")
async def analyze_reviews(payloads: List[ReviewRequest], request: Request):
    if not payloads:
        raise HTTPException(status_code=400, detail="분석할 리뷰 목록이 비어있습니다.")

    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    service = ABSAService(model=model, tokenizer=tokenizer, device=device)

    responses = []
    
    for payload in payloads:
        review_text = payload.review_text.strip()
        if not review_text:
            continue
            
        results = service.analyze(review_text)
        
        responses.append(
            ReviewResponse(
                review_id=payload.review_id, 
                review_text=review_text, 
                results=results
            )
        )

    return responses