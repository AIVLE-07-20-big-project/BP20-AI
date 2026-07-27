from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core import bootstrap  # noqa: F401
from app.forecast_service import ForecastService
from app.ocr import router as ocr
from app.routers import (
    agent_runs,
    analysis,
    campaign_logs,
    effect_verification_router,
)
from app.schemas.forecast import ForecastRequest, ForecastResponse

app = FastAPI(title="20BG AI 서비스")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(agent_runs.router, prefix="/api/v1")
app.include_router(campaign_logs.router, prefix="/api/v1")
app.include_router(ocr.router)
app.include_router(effect_verification_router.router)

forecast_service = ForecastService()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors()}),
    )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "UP"}


@app.post("/api/v1/forecasts", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    forecasts = [
        forecast_service.predict_product(
            product,
            request.forecastDays,
            request.weather,
        )
        for product in request.products
    ]

    return ForecastResponse(
        selectedModel="Temporary-Average-Forecast",
        forecasts=forecasts,
    )
