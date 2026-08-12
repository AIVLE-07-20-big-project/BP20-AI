import unittest

from pydantic import ValidationError

from app.schemas.effect_verification_schema import (
    EffectVerificationRequest,
    PeriodMetrics,
    RecommendationType,
    ReviewMetrics,
    SalesMetrics,
    VerificationCondition,
)
from app.services.effect_verification_service import (
    determine_verdict,
    verify_effect,
)


class EffectVerificationServiceTests(unittest.TestCase):

    def test_sales_verification_returns_effective_result(self):
        after = self.sales_metrics(1_300_000, 5_500_000)
        after.visit_count = 120
        after.average_order_value = 10_800
        after.revisit_rate = 25
        after.coupon_usage_rate = 18
        after.new_customer_count = 24
        after.dormant_customer_return_count = 7
        request = EffectVerificationRequest(
            store_id=1,
            recommendation_id=100,
            recommendation_type=RecommendationType.SALES,
            condition=VerificationCondition(
                period_days=14,
                start_hour=14,
                end_hour=17,
            ),
            before=PeriodMetrics(sales=self.sales_metrics(1_000_000, 5_000_000)),
            after=PeriodMetrics(sales=after),
        )

        response = verify_effect(request)

        self.assertEqual(response.verdict, "EFFECTIVE")
        self.assertGreaterEqual(response.effect_score, 70)
        self.assertEqual(len(response.metric_results), 8)
        self.assertIn("인과효과를 확정하지 않습니다", response.summary)

    def test_review_verification_supports_improvement_scenario(self):
        request = EffectVerificationRequest(
            store_id=1,
            recommendation_id=101,
            recommendation_type=RecommendationType.REVIEW,
            condition=VerificationCondition(
                period_days=14,
                target_aspect="대기시간",
            ),
            before=PeriodMetrics(review=self.review_metrics(
                rating=3.5,
                negative_rate=40,
                aspect_count=50,
                aspect_negative_rate=60,
                revisit_rate=20,
                sales=1_000_000,
            )),
            after=PeriodMetrics(review=self.review_metrics(
                rating=4.1,
                negative_rate=25,
                aspect_count=35,
                aspect_negative_rate=35,
                revisit_rate=26,
                sales=1_150_000,
            )),
        )

        response = verify_effect(request)

        self.assertEqual(response.verdict, "EFFECTIVE")
        self.assertGreaterEqual(response.effect_score, 70)
        self.assertEqual(len(response.metric_results), 8)

    def test_review_verification_accepts_string_id_and_missing_optional_metrics(self):
        before = self.review_metrics(3.5, 40, 50, 60, 20, 1_000_000)
        after = self.review_metrics(4.0, 25, 35, 30, 25, 1_100_000)
        before = ReviewMetrics(**{
            **before.model_dump(),
            "revisit_rate": None,
            "sales": None,
            "target_aspect_average_confidence": 91,
        })
        after = ReviewMetrics(**{
            **after.model_dump(),
            "revisit_rate": None,
            "sales": None,
            "target_aspect_average_confidence": 93,
        })

        response = verify_effect(EffectVerificationRequest(
            store_id=2,
            recommendation_id="review-3-1",
            recommendation_type=RecommendationType.REVIEW,
            condition=VerificationCondition(period_days=30, target_aspect="food"),
            before=PeriodMetrics(review=before),
            after=PeriodMetrics(review=after),
        ))

        self.assertEqual(response.recommendation_id, "review-3-1")
        metric_names = {metric.metric_name for metric in response.metric_results}
        self.assertNotIn("revisit_rate", metric_names)
        self.assertNotIn("sales", metric_names)
        self.assertGreater(response.effect_score, 0)
        confidence = next(
            metric for metric in response.metric_results
            if metric.metric_name == "target_aspect_average_confidence"
        )
        self.assertEqual(confidence.before_value, 0.91)
        self.assertEqual(confidence.after_value, 0.93)

    def test_sales_verification_returns_not_effective_when_metrics_decline(self):
        request = EffectVerificationRequest(
            store_id=1,
            recommendation_id=105,
            recommendation_type=RecommendationType.SALES,
            condition=VerificationCondition(period_days=14),
            before=PeriodMetrics(sales=self.sales_metrics(1_000_000, 5_000_000)),
            after=PeriodMetrics(sales=self.sales_metrics(800_000, 4_500_000)),
        )

        response = verify_effect(request)

        self.assertEqual(response.verdict, "NOT_EFFECTIVE")
        self.assertLess(response.effect_score, 40)

    def test_verdict_boundaries(self):
        self.assertEqual(determine_verdict(70), "EFFECTIVE")
        self.assertEqual(determine_verdict(69.99), "PARTIALLY_EFFECTIVE")
        self.assertEqual(determine_verdict(40), "PARTIALLY_EFFECTIVE")
        self.assertEqual(determine_verdict(39.99), "NOT_EFFECTIVE")

    def test_zero_baseline_does_not_divide_by_zero(self):
        before = self.sales_metrics(0, 0)
        after = self.sales_metrics(100_000, 200_000)
        request = EffectVerificationRequest(
            store_id=1,
            recommendation_id=102,
            recommendation_type=RecommendationType.SALES,
            condition=VerificationCondition(period_days=14),
            before=PeriodMetrics(sales=before),
            after=PeriodMetrics(sales=after),
        )

        response = verify_effect(request)

        target_sales = next(
            metric for metric in response.metric_results
            if metric.metric_name == "target_sales"
        )
        self.assertIsNone(target_sales.change_rate)

    def test_sales_request_rejects_missing_sales_metrics(self):
        with self.assertRaises(ValidationError):
            EffectVerificationRequest(
                store_id=1,
                recommendation_id=103,
                recommendation_type=RecommendationType.SALES,
                condition=VerificationCondition(period_days=14),
                before=PeriodMetrics(),
                after=PeriodMetrics(),
            )

    def test_review_request_requires_target_aspect(self):
        review = self.review_metrics(3.5, 40, 50, 60, 20, 1_000_000)
        with self.assertRaises(ValidationError):
            EffectVerificationRequest(
                store_id=1,
                recommendation_id=104,
                recommendation_type=RecommendationType.REVIEW,
                condition=VerificationCondition(period_days=14),
                before=PeriodMetrics(review=review),
                after=PeriodMetrics(review=review),
            )

    @staticmethod
    def sales_metrics(target_sales: float, total_sales: float) -> SalesMetrics:
        return SalesMetrics(
            target_sales=target_sales,
            visit_count=100 if target_sales else 0,
            average_order_value=10_000 if target_sales else 0,
            revisit_rate=20,
            coupon_usage_rate=10,
            new_customer_count=20 if target_sales else 0,
            dormant_customer_return_count=5 if target_sales else 0,
            total_sales=total_sales,
        )

    @staticmethod
    def review_metrics(
        rating: float,
        negative_rate: float,
        aspect_count: int,
        aspect_negative_rate: float,
        revisit_rate: float,
        sales: float,
    ) -> ReviewMetrics:
        return ReviewMetrics(
            average_rating=rating,
            negative_review_rate=negative_rate,
            target_aspect_review_count=aspect_count,
            target_aspect_negative_rate=aspect_negative_rate,
            target_aspect_average_confidence=0.9,
            review_count=100,
            revisit_rate=revisit_rate,
            sales=sales,
        )


if __name__ == "__main__":
    unittest.main()
