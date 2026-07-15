from app.schemas.effect_verification_schema import (
    EffectVerificationRequest,
    EffectVerificationResponse,
    MetricResult,
    RecommendationType,
)


def calculate_change_rate(before: float, after: float) -> float | None:
    """
    실행 전 대비 실행 후 증감률을 계산한다.

    예:
    100 -> 120 = 20%
    100 -> 80 = -20%

    실행 전 값이 0이면 증감률을 계산할 수 없으므로 None을 반환한다.
    """
    if before == 0:
        return None

    return round(((after - before) / before) * 100, 2)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """값을 지정된 범위 안으로 제한한다."""
    return max(minimum, min(value, maximum))


def create_metric_result(
    metric_name: str,
    before_value: float,
    after_value: float,
    higher_is_better: bool = True,
) -> MetricResult:
    """지표별 실행 전후 변화 결과를 생성한다."""

    change_value = round(after_value - before_value, 2)
    change_rate = calculate_change_rate(before_value, after_value)

    improved = (
        after_value > before_value
        if higher_is_better
        else after_value < before_value
    )

    return MetricResult(
        metric_name=metric_name,
        before_value=before_value,
        after_value=after_value,
        change_value=change_value,
        change_rate=change_rate,
        improved=improved,
    )


def normalize_positive_change(
    change_rate: float | None,
    target_rate: float,
) -> float:
    """
    증가할수록 좋은 지표를 0~100점으로 변환한다.

    목표 증가율만큼 증가하면 100점이다.
    """
    if change_rate is None or change_rate <= 0:
        return 0.0

    return round(
        clamp((change_rate / target_rate) * 100, 0, 100),
        2,
    )


def normalize_negative_change(
    change_rate: float | None,
    target_decrease_rate: float,
) -> float:
    """
    감소할수록 좋은 지표를 0~100점으로 변환한다.

    예:
    실제 감소율 -30%, 목표 감소율 30%
    -> 100점
    """
    if change_rate is None or change_rate >= 0:
        return 0.0

    decrease_rate = abs(change_rate)

    return round(
        clamp(
            (decrease_rate / target_decrease_rate) * 100,
            0,
            100,
        ),
        2,
    )


def normalize_point_increase(
    before: float,
    after: float,
    target_point: float,
) -> float:
    """
    별점, 재방문율처럼 차이값으로 판단하는 지표를 점수화한다.
    """
    point_change = after - before

    if point_change <= 0:
        return 0.0

    return round(
        clamp((point_change / target_point) * 100, 0, 100),
        2,
    )


def calculate_sales_effect_score(
    request: EffectVerificationRequest,
) -> tuple[float, list[MetricResult], str]:
    """매출 기반 추천의 효과 점수를 계산한다."""

    if request.before.sales is None or request.after.sales is None:
        raise ValueError(
            "매출 기반 추천은 before.sales와 after.sales가 필요합니다."
        )

    before = request.before.sales
    after = request.after.sales

    target_sales_change = calculate_change_rate(
        before.target_sales,
        after.target_sales,
    )
    total_sales_change = calculate_change_rate(
        before.total_sales,
        after.total_sales,
    )
    visit_count_change = calculate_change_rate(
        before.visit_count,
        after.visit_count,
    )
    average_order_value_change = calculate_change_rate(
        before.average_order_value,
        after.average_order_value,
    )
    new_customer_change = calculate_change_rate(
        before.new_customer_count,
        after.new_customer_count,
    )
    dormant_return_change = calculate_change_rate(
        before.dormant_customer_return_count,
        after.dormant_customer_return_count,
    )

    if target_sales_change is None:
        adjusted_target_sales_change = 0.0
    elif total_sales_change is None:
        adjusted_target_sales_change = target_sales_change
    else:
        adjusted_target_sales_change = (
            target_sales_change - total_sales_change
        )

    metric_results = [
        create_metric_result(
            "target_sales",
            before.target_sales,
            after.target_sales,
        ),
        create_metric_result(
            "total_sales",
            before.total_sales,
            after.total_sales,
        ),
        create_metric_result(
            "visit_count",
            before.visit_count,
            after.visit_count,
        ),
        create_metric_result(
            "average_order_value",
            before.average_order_value,
            after.average_order_value,
        ),
        create_metric_result(
            "revisit_rate",
            before.revisit_rate,
            after.revisit_rate,
        ),
        create_metric_result(
            "coupon_usage_rate",
            before.coupon_usage_rate,
            after.coupon_usage_rate,
        ),
        create_metric_result(
            "new_customer_count",
            before.new_customer_count,
            after.new_customer_count,
        ),
        create_metric_result(
            "dormant_customer_return_count",
            before.dormant_customer_return_count,
            after.dormant_customer_return_count,
        ),
    ]

    scores = {
        "adjusted_target_sales": normalize_positive_change(
            adjusted_target_sales_change,
            target_rate=15,
        ),
        "visit_count": normalize_positive_change(
            visit_count_change,
            target_rate=15,
        ),
        "average_order_value": normalize_positive_change(
            average_order_value_change,
            target_rate=10,
        ),
        "revisit_rate": normalize_point_increase(
            before.revisit_rate,
            after.revisit_rate,
            target_point=5,
        ),
        "coupon_usage_rate": normalize_point_increase(
            before.coupon_usage_rate,
            after.coupon_usage_rate,
            target_point=10,
        ),
        "new_customer_count": normalize_positive_change(
            new_customer_change,
            target_rate=15,
        ),
        "dormant_customer_return_count": normalize_positive_change(
            dormant_return_change,
            target_rate=20,
        ),
    }

    weights = {
        "adjusted_target_sales": 0.35,
        "visit_count": 0.15,
        "average_order_value": 0.10,
        "revisit_rate": 0.15,
        "coupon_usage_rate": 0.10,
        "new_customer_count": 0.05,
        "dormant_customer_return_count": 0.10,
    }

    effect_score = round(
        sum(scores[key] * weights[key] for key in weights),
        2,
    )

    target_change_text = (
        f"{target_sales_change}%"
        if target_sales_change is not None
        else "계산 불가"
    )
    total_change_text = (
        f"{total_sales_change}%"
        if total_sales_change is not None
        else "계산 불가"
    )

    summary = (
        f"추천 대상 시간대 매출은 {target_change_text} 변화했고, "
        f"같은 기간 전체 매출은 {total_change_text} 변화했습니다. "
        f"전체 매출 변화를 제외한 대상 시간대의 추가 개선 폭은 "
        f"{round(adjusted_target_sales_change, 2)}%p입니다."
    )

    return effect_score, metric_results, summary


def calculate_review_effect_score(
    request: EffectVerificationRequest,
) -> tuple[float, list[MetricResult], str]:
    """ReviewAnalysis 기반 리뷰 개선 효과 점수를 계산한다."""

    if request.before.review is None or request.after.review is None:
        raise ValueError(
            "리뷰 기반 추천은 before.review와 after.review가 필요합니다."
        )

    before = request.before.review
    after = request.after.review

    negative_review_change = calculate_change_rate(
        before.negative_review_rate,
        after.negative_review_rate,
    )
    target_aspect_negative_change = calculate_change_rate(
        before.target_aspect_negative_rate,
        after.target_aspect_negative_rate,
    )
    review_count_change = calculate_change_rate(
        before.review_count,
        after.review_count,
    )
    target_aspect_review_count_change = calculate_change_rate(
        before.target_aspect_review_count,
        after.target_aspect_review_count,
    )
    sales_change = calculate_change_rate(
        before.sales,
        after.sales,
    )

    metric_results = [
        create_metric_result(
            "average_rating",
            before.average_rating,
            after.average_rating,
        ),
        create_metric_result(
            "negative_review_rate",
            before.negative_review_rate,
            after.negative_review_rate,
            higher_is_better=False,
        ),
        create_metric_result(
            "target_aspect_review_count",
            before.target_aspect_review_count,
            after.target_aspect_review_count,
            higher_is_better=False,
        ),
        create_metric_result(
            "target_aspect_negative_rate",
            before.target_aspect_negative_rate,
            after.target_aspect_negative_rate,
            higher_is_better=False,
        ),
        create_metric_result(
            "target_aspect_average_confidence",
            before.target_aspect_average_confidence,
            after.target_aspect_average_confidence,
        ),
        create_metric_result(
            "review_count",
            before.review_count,
            after.review_count,
        ),
        create_metric_result(
            "revisit_rate",
            before.revisit_rate,
            after.revisit_rate,
        ),
        create_metric_result(
            "sales",
            before.sales,
            after.sales,
        ),
    ]

    scores = {
        "average_rating": normalize_point_increase(
            before.average_rating,
            after.average_rating,
            target_point=0.5,
        ),
        "negative_review_rate": normalize_negative_change(
            negative_review_change,
            target_decrease_rate=30,
        ),
        "target_aspect_negative_rate": normalize_negative_change(
            target_aspect_negative_change,
            target_decrease_rate=30,
        ),
        "target_aspect_review_count": normalize_negative_change(
            target_aspect_review_count_change,
            target_decrease_rate=20,
        ),
        "revisit_rate": normalize_point_increase(
            before.revisit_rate,
            after.revisit_rate,
            target_point=5,
        ),
        "sales": normalize_positive_change(
            sales_change,
            target_rate=10,
        ),
        "review_count": normalize_positive_change(
            review_count_change,
            target_rate=20,
        ),
    }

    weights = {
        "average_rating": 0.10,
        "negative_review_rate": 0.20,
        "target_aspect_negative_rate": 0.30,
        "target_aspect_review_count": 0.10,
        "revisit_rate": 0.15,
        "sales": 0.10,
        "review_count": 0.05,
    }

    effect_score = round(
        sum(scores[key] * weights[key] for key in weights),
        2,
    )

    negative_change_text = (
        f"{negative_review_change}%"
        if negative_review_change is not None
        else "계산 불가"
    )
    aspect_negative_change_text = (
        f"{target_aspect_negative_change}%"
        if target_aspect_negative_change is not None
        else "계산 불가"
    )

    summary = (
        f"전체 부정 리뷰 비율은 {negative_change_text} 변화했고, "
        f"대상 속성의 부정 비율은 {aspect_negative_change_text} 변화했습니다. "
        f"대상 속성 리뷰 수는 "
        f"{before.target_aspect_review_count}건에서 "
        f"{after.target_aspect_review_count}건으로 변했습니다. "
        f"분석 신뢰도는 "
        f"{before.target_aspect_average_confidence}에서 "
        f"{after.target_aspect_average_confidence}로 변했습니다."
    )

    return effect_score, metric_results, summary


def determine_verdict(effect_score: float) -> str:
    """효과 점수를 기준으로 최종 판정을 반환한다."""

    if effect_score >= 70:
        return "EFFECTIVE"

    if effect_score >= 40:
        return "PARTIALLY_EFFECTIVE"

    return "NOT_EFFECTIVE"


def verify_effect(
    request: EffectVerificationRequest,
) -> EffectVerificationResponse:
    """추천 유형에 따라 효과 검증을 수행한다."""

    if request.recommendation_type == RecommendationType.SALES:
        effect_score, metric_results, summary = (
            calculate_sales_effect_score(request)
        )
    elif request.recommendation_type == RecommendationType.REVIEW:
        effect_score, metric_results, summary = (
            calculate_review_effect_score(request)
        )
    else:
        raise ValueError("지원하지 않는 추천 유형입니다.")

    verdict = determine_verdict(effect_score)

    verdict_message = {
        "EFFECTIVE": "추천 실행 이후 유의미한 개선 효과가 확인되었습니다.",
        "PARTIALLY_EFFECTIVE": "일부 지표에서 개선 효과가 확인되었습니다.",
        "NOT_EFFECTIVE": "현재 데이터에서는 뚜렷한 개선 효과가 확인되지 않았습니다.",
    }[verdict]

    final_summary = (
        f"{verdict_message} "
        f"종합 효과 점수는 {effect_score}점입니다. "
        f"{summary}"
    )

    return EffectVerificationResponse(
        store_id=request.store_id,
        recommendation_id=request.recommendation_id,
        recommendation_type=request.recommendation_type,
        effect_score=effect_score,
        verdict=verdict,
        metric_results=metric_results,
        summary=final_summary,
    )