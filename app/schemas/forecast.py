from datetime import date, datetime

from pydantic import BaseModel


class DailySalesValue(BaseModel):
    saleDate: date
    salesQuantity: int
    unitPrice: int


class ProductSalesHistory(BaseModel):
    productCode: str
    productName: str
    salesHistory: list[DailySalesValue]


class WeatherFeature(BaseModel):
    forecastDateTime: datetime | None = None
    temperature: float | None = None
    windSpeed: float | None = None
    sky: str | None = None
    precipitationType: str | None = None
    rainProbability: int | None = None
    humidity: int | None = None


class ForecastRequest(BaseModel):
    forecastDays: int
    orderDateTime: datetime
    weather: WeatherFeature | None = None
    products: list[ProductSalesHistory]


class ProductForecast(BaseModel):
    productCode: str
    predictedSalesQuantity: int
    lowerBound: int
    upperBound: int
    confidenceScore: float


class ForecastResponse(BaseModel):
    selectedModel: str
    forecasts: list[ProductForecast]
