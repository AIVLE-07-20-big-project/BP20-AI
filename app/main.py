import sys
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.config import settings
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
    effect_verification_router,
    review,
    jobs,
)

ERROR_RESPONSES = {
    status: {"model": ErrorResponse}
    for status in (400, 401, 403, 404, 409, 413, 415, 422, 500, 503)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RoBERTa ABSA is loading (Device: {device})")

    model_path = getattr(settings, "MODEL_PATH", "thadus2/roberta-absa-best-4class")
    hf_token = getattr(settings, "HF_TOKEN", None) or os.getenv("HF_TOKEN")

    token_arg = hf_token if hf_token else None

    app.state.tokenizer = None
    app.state.model = None
    app.state.device = device
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=token_arg)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, token=token_arg)

        model.to(device)
        model.eval()

        app.state.tokenizer = tokenizer
        app.state.model = model

        print("RoBERTa 모델 로드 완료!")
    except Exception as e:
        # ABSA(리뷰 감성분석) 전용 모델이라 다른 라우터(신규 가맹점 영업 타겟 등)와 무관하다.
        # HF_TOKEN 미설정/만료 등으로 여기서 죽으면 서버 전체가 못 뜨는 게 더 큰 문제라서,
        # 로드 실패 시 경고만 남기고 계속 진행한다 — /api/v1/review* 엔드포인트만 영향받는다.
        print(f"RoBERTa ABSA 로드 실패, 리뷰 감성분석 기능 없이 계속 진행합니다: {type(e).__name__}: {e}")
    try:
        from app.ocr.pipeline import _get_ocr_engine

        print("서버 시작 - PaddleOCR 모델 예열 중...")
        _get_ocr_engine()
        print("PaddleOCR 모델 예열 완료.")
    except Exception as e:
        print(f"PaddleOCR 예열 중 알림: {e}")

    yield
    
    del app.state.tokenizer
    del app.state.model

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
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(ocr.router)
app.include_router(effect_verification_router.router)
app.include_router(online_trend.router)
app.include_router(product_image.router)
app.include_router(sales_target.router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "BP Team 20 AI Server is running!"}
