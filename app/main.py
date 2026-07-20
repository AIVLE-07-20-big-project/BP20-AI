from fastapi import FastAPI

from app.core import bootstrap  # noqa: F401
from app.routers import agent_runs, analysis, campaign_logs

app = FastAPI(title="20BG 매출분석 서비스")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
