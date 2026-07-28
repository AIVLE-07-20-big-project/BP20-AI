import sys
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.core.config import settings

from app.core import bootstrap  # noqa: F401
from app.ocr import router as ocr
from app.online_trend import router as online_trend
from app.product_image import router as product_image
from app.routers import (
    agent_runs,
    analysis,
    campaign_logs,
    effect_verification_router,
    review,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RoBERTa ABSA is loading (Device: {device})")
    
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_PATH)
    model.to(device)
    model.eval()

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    
    print("RoBERTa 모델 로드 완료! FastAPI 서비스를 시작합니다.")
    yield
    
    del app.state.tokenizer
    del app.state.model

app = FastAPI(
    title="20BG AI 서비스",
    lifespan=lifespan,
)
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
app.include_router(review.router, prefix="/api/v1")
app.include_router(ocr.router)
app.include_router(effect_verification_router.router)
app.include_router(online_trend.router)
app.include_router(product_image.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "BP Team 20 AI Server is running!"}
