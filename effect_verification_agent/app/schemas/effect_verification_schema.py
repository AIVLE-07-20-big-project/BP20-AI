from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RecommendationType(str, Enum):
    SALES = "SALES"
    REVIEW = "REVIEW"


class SalesMetrics(BaseModel):
    """매출 기반 추천 효과 검증 지표"""

    target_sales: float = Field(
        ge=0,
        description="추천 대상 시간대의 총매출",
    )
    visit_count: int = Field(
        ge=0,
        description="방문 건수",
    )
    average_order_value: float = Field(
        ge=0,
        description="객단가",
    )
    revisit_rate: float = Field(
        ge=0,
        le=100,
        description="재방문율(%)",
    )
    coupon_usage_rate: float = Field(
        ge=0,
        le=100,
        description="쿠폰 사용률(%)",
    )
    new_customer_count: int = Field(
        ge=0,
        description="신규 고객 수",
    )
    dormant_customer_return_count: int = Field(
        ge=0,
        description="장기 미방문 고객 재방문 수",
    )
    total_sales: float = Field(
        ge=0,
        description="동일 기간 전체 매출",
    )


class ReviewMetrics(BaseModel):
    """리뷰 기반 추천 효과 검증 지표"""

    average_rating: float = Field(
        ge=0,
        le=5,
        description="평균 별점",
    )

    negative_review_rate: float = Field(
        ge=0,
        le=100,
        description="전체 부정 리뷰 비율(%)",
    )

    target_aspect_review_count: int = Field(
        ge=0,
        description="검증 대상 속성이 분석된 리뷰 수",
    )

    target_aspect_negative_rate: float = Field(
        ge=0,
        le=100,
        description="검증 대상 속성의 부정 비율(%)",
    )

    target_aspect_average_confidence: float = Field(
        ge=0,
        le=1,
        description="검증 대상 속성 분석의 평균 신뢰도",
    )

    review_count: int = Field(
        ge=0,
        description="전체 리뷰 수",
    )

    revisit_rate: float = Field(
        ge=0,
        le=100,
        description="재방문율(%)",
    )

    sales: float = Field(
        ge=0,
        description="동일 기간 매출",
    )


class PeriodMetrics(BaseModel):
    """실행 전 또는 실행 후 집계 데이터"""

    sales: Optional[SalesMetrics] = None
    review: Optional[ReviewMetrics] = None


class VerificationCondition(BaseModel):
    """전후 비교 조건"""

    period_days: int = Field(
        default=14,
        ge=1,
        description="실행 전후 비교 기간",
    )
    start_hour: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="추천 대상 시간대 시작 시각",
    )
    end_hour: Optional[int] = Field(
        default=None,
        ge=0,
        le=23,
        description="추천 대상 시간대 종료 시각",
    )
    compare_same_weekday: bool = Field(
        default=True,
        description="동일 요일 기준 비교 여부",
    )
    target_aspect: Optional[str] = Field(
        default=None,
        description="검증할 리뷰 속성",
    )


class EffectVerificationRequest(BaseModel):
    """AI 추천 실행 효과 검증 요청"""

    store_id: int
    recommendation_id: int
    recommendation_type: RecommendationType
    condition: VerificationCondition
    before: PeriodMetrics
    after: PeriodMetrics

    @model_validator(mode="after")
    def validate_metrics_for_recommendation_type(self):
        if self.recommendation_type == RecommendationType.SALES:
            if self.before.sales is None or self.after.sales is None:
                raise ValueError(
                    "SALES recommendation requires before.sales and after.sales"
                )
        elif self.recommendation_type == RecommendationType.REVIEW:
            if self.before.review is None or self.after.review is None:
                raise ValueError(
                    "REVIEW recommendation requires before.review and after.review"
                )

        has_start = self.condition.start_hour is not None
        has_end = self.condition.end_hour is not None
        if has_start != has_end:
            raise ValueError(
                "start_hour and end_hour must be provided together"
            )
        if has_start and self.condition.start_hour == self.condition.end_hour:
            raise ValueError("start_hour and end_hour must be different")
        if (
            self.recommendation_type == RecommendationType.REVIEW
            and not self.condition.target_aspect
        ):
            raise ValueError("REVIEW recommendation requires target_aspect")
        return self


class MetricResult(BaseModel):
    metric_name: str
    before_value: float
    after_value: float
    change_value: float
    change_rate: Optional[float] = None
    improved: bool


class EffectVerificationResponse(BaseModel):
    store_id: int
    recommendation_id: int
    recommendation_type: RecommendationType
    effect_score: float
    verdict: str
    metric_results: list[MetricResult]
    summary: str
