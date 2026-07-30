from fastapi import APIRouter

from app.forecast_service import ForecastService
from app.schemas.forecast import ForecastRequest, ForecastResponse

router = APIRouter(prefix="/api/v1", tags=["Forecast"])
forecast_service = ForecastService()


@router.post("/forecasts", response_model=ForecastResponse)
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
