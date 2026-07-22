import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from core.config import settings
from api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RoBERTa ABSA is loading (Device: {device})")
    
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(settings.MODEL_PATH)
    model.to(device)
    model.eval()

    # FastAPI app.state 메모리에 전역 저장
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = device
    
    print("RoBERTa 모델 로드 완료! FastAPI 서비스를 시작합니다.")
    yield
    
    # 서버 종료 시 메모리 정리
    del app.state.tokenizer
    del app.state.model

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# API v1 라우터 등록 (/api/v1/review/predict)
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RoBERTa ABSA API Server is running!"}