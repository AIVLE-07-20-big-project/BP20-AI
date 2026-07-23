from fastapi import FastAPI

from app.core import bootstrap  # noqa: F401
from app.core.errors import ErrorResponse, register_error_handlers
from app.ocr import router as ocr
from app.routers import agent_runs, analysis, campaign_logs

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (400, 401, 403, 404, 409, 413, 415, 422, 500, 503)
}
OPENAPI_TAGS = [
    {"name": "매출 분석", "description": "매출 CSV 분석, 저장 및 이력 조회"},
    {"name": "전략 추천", "description": "대응방안 추천, 상태 조회 및 승인 워크플로우"},
    {"name": "캠페인 학습", "description": "실행 결과 기록과 학습 데이터 품질 확인"},
    {"name": "OCR", "description": "영수증 인식, 비용 분석 및 리포트 생성"},
    {"name": "상태 확인", "description": "통합 FastAPI 서비스 상태 확인"},
]

app = FastAPI(
    title="20BG AI 서비스",
    version="1.0.0",
    description="매출 분석, 고객 대응방안 추천·검증, 영수증 OCR 통합 API",
    openapi_tags=OPENAPI_TAGS,
    responses=ERROR_RESPONSES,
)
register_error_handlers(app)
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
app.include_router(ocr.router)


# 최초 OCR 요청이 지연되지 않도록 서버 시작 시 PaddleOCR 모델을 미리 로드한다.
@app.on_event("startup")
async def warm_up_ocr_engine() -> None:
    from app.ocr.pipeline import _get_ocr_engine

    print("서버 시작 - PaddleOCR 모델 예열 중...")
    _get_ocr_engine()
    print("PaddleOCR 모델 예열 완료.")
