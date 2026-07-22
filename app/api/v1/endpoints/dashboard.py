from fastapi import APIRouter, Depends, Request
from schemas.dashboard import ReviewAnalysisRequest, ReviewAnalysisResponse  # DTO 경로에 맞게 수정
from services.dashboard_service import DashboardService

router = APIRouter()

# ★ Request 객체에서 main.py가 저장한 app.state 자원들을 바로 꺼냅니다.
def get_dashboard_service(request: Request) -> DashboardService:
    return DashboardService(
        model=request.app.state.model,
        tokenizer=request.app.state.tokenizer,
        device=request.app.state.device
    )

@router.post("/analyze", response_model=ReviewAnalysisResponse)
async def analyze_dashboard(
    data: ReviewAnalysisRequest,
    service: DashboardService = Depends(get_dashboard_service)
):
    return service.analyze_reviews_pipeline(data.reviews)