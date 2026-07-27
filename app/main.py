from fastapi import FastAPI
from app.core import bootstrap  # noqa: F401
from app.ocr import router as ocr
from app.online_trend import router as online_trend
from app.product_image import router as product_image
from app.routers import (
    agent_runs,
    analysis,
    campaign_logs,
    effect_verification_router,
)
app = FastAPI(title="20BG AI 서비스")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
app.include_router(ocr.router)
app.include_router(effect_verification_router.router)
app.include_router(online_trend.router)
app.include_router(product_image.router)