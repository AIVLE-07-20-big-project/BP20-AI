import math

class ForecastService:

    def predict_product(self, product, forecast_days, weather) -> dict:

        quantities = [
            item.salesQuantity
            for item in product.salesHistory
        ]

        if not quantities:
            predicted_quantity = 0

        else:
            average_daily_sales = sum(quantities) / len(quantities)

            factor = self.weather_factor(weather)

            predicted_quantity = math.ceil(
                average_daily_sales
                * forecast_days
                * factor
            )

        # Confidence 계산
        confidence = 0.70

        if len(quantities) >= 30:
            confidence += 0.10

        if weather is not None:
            confidence += 0.05

        confidence = min(confidence, 0.95)

        return {
            "productCode": product.productCode,
            "predictedSalesQuantity": predicted_quantity,
            "lowerBound": max(
                0,
                math.floor(predicted_quantity * 0.85)
            ),
            "upperBound": math.ceil(
                predicted_quantity * 1.15
            ),
            "confidenceScore": confidence
        }

    def weather_factor(self, weather):

        factor = 1.0

        if weather is None:
            return factor

        # 기온
        if weather.temperature is not None:

            if weather.temperature >= 30:
                factor += 0.10

            elif weather.temperature >= 25:
                factor += 0.05

            elif weather.temperature <= 5:
                factor += 0.08

        # 강수확률
        if weather.rainProbability is not None:

            if weather.rainProbability >= 80:
                factor -= 0.08

            elif weather.rainProbability >= 60:
                factor -= 0.05

        # 습도
        if weather.humidity is not None:

            if weather.humidity >= 80:
                factor += 0.03

        return factor