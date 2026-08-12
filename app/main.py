import sys
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.config import settings
from app.core.huggingface_assets import sync_huggingface_assets
from app.core import bootstrap  # noqa: F401
from app.core.errors import ErrorResponse, register_error_handlers
from app.ocr import router as ocr
from app.online_trend import router as online_trend
from app.product_image import router as product_image
from app.sales_target import router as sales_target
from app.routers import (
    agent_runs,
    analysis,
    campaign_logs,
    ai_learning,
    effect_verification_router,
    forecast,
    review,
    jobs,
    locations,
    industries,
)

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (400, 401, 403, 404, 409, 413, 415, 422, 429, 500, 502, 503, 504)
}
OPENAPI_TAGS = [
    {"name": "매출 분석", "description": "매출 CSV 분석, 저장 및 이력 조회"},
    {"name": "전략 추천", "description": "대응방안 추천, 상태 조회 및 승인 워크플로우"},
    {"name": "신규 가맹점 영업 타겟", "description": "영업 타겟 후보 생성(그래프), 상태 조회 및 승인 워크플로우"},
    {"name": "캠페인 학습", "description": "실행 결과 기록과 학습 데이터 품질 확인"},
    {"name": "OCR", "description": "영수증 인식, 비용 분석 및 리포트 생성"},
    {"name": "작업 상태", "description": "비동기 분석 잡 상태 조회"},
    {"name": "상태 확인", "description": "통합 FastAPI 서비스 상태 확인"},
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        asset_status = sync_huggingface_assets()
        print(f"Hugging Face 아티팩트 상태: {asset_status}")
    except Exception as error:
        # 분석 아티팩트 동기화 실패가 OCR 등 독립 API의 시작을 막지 않게 한다.
        print(f"Hugging Face 아티팩트 동기화 실패. 서버를 계속 시작합니다: {error}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    app.state.tokenizer = None
    app.state.model = None

    absa_enabled = os.getenv("ABSA_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if absa_enabled:
        print(f"RoBERTa ABSA is loading (Device: {device})")
        model_path = getattr(
            settings, "ROBERTA_MODEL_PATH", "thadus2/roberta-absa-best-4class"
        )
        hf_token = getattr(settings, "ROBERTA_HF_TOKEN", None) or os.getenv(
            "ROBERTA_HF_TOKEN"
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token or None)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, token=hf_token or None
            )
            model.to(device)
            model.eval()
            app.state.tokenizer = tokenizer
            app.state.model = model
            print("RoBERTa 모델 로드 완료!")
        except Exception as error:  # ABSA is optional; OCR and other APIs must still start.
            print(
                "RoBERTa ABSA 모델을 불러오지 못했습니다. "
                f"리뷰 분석만 비활성화하고 서버를 계속 시작합니다: {error}"
            )
    else:
        print("ABSA_ENABLED=false: RoBERTa 리뷰 분석 모델 로딩을 건너뜁니다.")
    try:
        from app.ocr.pipeline import _get_ocr_engine

        print("서버 시작 - PaddleOCR 모델 예열 중...")
        _get_ocr_engine()
        print("PaddleOCR 모델 예열 완료.")
    except Exception as e:
        print(f"PaddleOCR 예열 중 알림: {e}")

    yield
    
    app.state.tokenizer = None
    app.state.model = None

app = FastAPI(
    title="20BG AI 서비스",
    version="1.0.1",
    description="모델 및 FastAPI 서버 구조 1차 통합 version",
    openapi_tags=OPENAPI_TAGS,
    responses=ERROR_RESPONSES,
    lifespan=lifespan,
)

register_error_handlers(app)

app.include_router(review.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
app.include_router(ai_learning.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(locations.router, prefix="/api/v1")
app.include_router(industries.router, prefix="/api/v1")
app.include_router(ocr.router)
app.include_router(effect_verification_router.router)
app.include_router(forecast.router)
app.include_router(online_trend.router)
app.include_router(product_image.router)
app.include_router(sales_target.router, prefix="/api/v1")

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "BP Team 20 AI Server is running!"}


@app.get("/health")
def health_probe():
    """ALB/ECS/Docker가 사용하는 경량 liveness probe."""
    return {"status": "ok"}
