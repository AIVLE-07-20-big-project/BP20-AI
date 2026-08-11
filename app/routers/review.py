from fastapi import APIRouter, Depends, Request, HTTPException
from typing import List
import os
from app.schemas.review import ReviewRequest, ReviewResponse
from app.schemas.review import ABSAAgentResponse, BatchReviewRequest, MonthlyReviewReportRequest
from app.services.absa_service import ABSAService
from app.services.review_analyze_agent import ABSAGraphRunner

router = APIRouter(prefix="/review", tags=["Review ABSA"])

_absa_service = None
_graph_runner = None

def get_graph_runner(request: Request) -> ABSAGraphRunner:
    global _absa_service, _graph_runner
    if _graph_runner is None:

        _absa_service = ABSAService(
            model=request.app.state.model,
            tokenizer=request.app.state.tokenizer,
            device=request.app.state.device,
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="Openai key가 환경변수에 없습니다.",
            )

        _graph_runner = ABSAGraphRunner(
            absa_service=_absa_service, openai_api_key=api_key
        )

    return _graph_runner

@router.post("/analyze-graph", response_model=ABSAAgentResponse)
async def analyze_reviews_with_graph(
    payload: BatchReviewRequest,
    runner: ABSAGraphRunner = Depends(get_graph_runner),
):
    """30개 리뷰를 RoBERTa + LangGraph LLM 키워드 추출 & 클러스터링 배치를 수행합니다"""
    try:
        response = await runner.run(payload)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"에이전트 그래프 실행 중 에러 발생: {str(e)}"
        )

@router.post("/analyze", response_model=List[ReviewResponse], summary="리뷰 ABSA 감성 분석 (배치)")
async def analyze_reviews(payloads: List[ReviewRequest], request: Request):
    if not payloads:
        raise HTTPException(status_code=400, detail="분석할 리뷰 목록이 비어있습니다.")

    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="리뷰 감성분석 모델을 사용할 수 없습니다. ROBERTA_MODEL_PATH와 ROBERTA_HF_TOKEN을 확인하세요.",
        )
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
                # review_text=review_text, 
                results=results
            )
        )

    return responses

@router.post("/monthly-report", response_model=ABSAAgentResponse)
async def generate_monthly_review_report(
    payload: MonthlyReviewReportRequest,
    runner: ABSAGraphRunner = Depends(get_graph_runner),
):
    """Generate a monthly report from ABSA results already saved by Spring Boot."""
    try:
        return await runner.run_monthly_report(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monthly report generation failed: {str(e)}")
