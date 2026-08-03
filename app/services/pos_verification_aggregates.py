# 매출형 전략검증(/effect-verification/verify-from-analyses)이 재사용할 POS 집계
#
# /analyses 실행 시 원본 POS CSV에서 이 집계를 뽑아 분석 결과(detailed_analysis)에
# 함께 저장해둔다. 원본 CSV는 분석 완료 후 삭제되므로, 나중에 검증 시점에는
# 여기서 만든 소량의 집계만으로 SalesMetrics를 다시 구성한다.
from __future__ import annotations

import io

from app.schemas.effect_verification_schema import SalesMetrics
from scripts.modeling.detailed_sales_external_analysis import (
    DetailedSalesDataError,
    load_pos_transactions,
)

POS_CUSTOMER_COLUMNS = {"customer_id", "coupon_used"}
_TRUE_TOKENS = {"true", "1", "yes", "y"}


class PosVerificationDataError(ValueError):
    """매출형 전략검증에 필요한 POS 컬럼·형식이 부족할 때 발생한다."""


def build_verification_aggregates(raw_bytes: bytes) -> dict:
    """POS CSV 원본에서 검증에 필요한 소량 집계만 뽑아 저장 가능한 dict로 반환한다.

    필요 컬럼(customer_id, coupon_used)이 없으면 PosVerificationDataError를 낸다 —
    호출부(detailed_sales.analyze_uploaded_sales)는 이를 잡아 경고로만 남기고
    분석 자체는 계속 진행한다.
    """

    try:
        frame = load_pos_transactions(io.BytesIO(raw_bytes))
    except DetailedSalesDataError as exc:
        raise PosVerificationDataError(str(exc)) from exc

    missing = sorted(POS_CUSTOMER_COLUMNS - set(frame.columns))
    if missing:
        raise PosVerificationDataError(
            f"매출형 전략검증에 필요한 컬럼이 POS CSV에 없습니다: {missing}"
        )

    frame = frame.copy()
    frame["customer_id"] = frame["customer_id"].astype(str)
    frame["coupon_used"] = (
        frame["coupon_used"].astype(str).str.strip().str.lower().isin(_TRUE_TOKENS)
    )

    # detailed_sales_external_analysis.analyze_internal_drivers와 같은 규칙으로
    # "이번 달"만 검증 대상 구간으로 쓴다. 업로드 CSV에 그 이전 달 데이터가 섞여
    # 있어도(내부 원인 분석의 baseline용) 매출형 전략검증 수치에는 섞이지 않는다.
    current_start = frame["date"].max().normalize().replace(day=1)
    frame = frame[frame["date"] >= current_start]

    total_sales = float(frame["Total_Bill"].sum())
    visit_count = int(frame["transaction_id"].nunique())
    average_order_value = round(total_sales / visit_count, 2) if visit_count else 0.0

    visits_per_customer = frame.groupby("customer_id")["transaction_id"].nunique()
    revisit_rate = (
        round(float((visits_per_customer >= 2).mean() * 100), 2)
        if len(visits_per_customer) else 0.0
    )
    coupon_usage_rate = (
        round(float(frame["coupon_used"].mean() * 100), 2) if len(frame) else 0.0
    )

    hourly_sales = frame.groupby("hour")["Total_Bill"].sum()
    hourly_sales_by_str_hour = {str(h): float(v) for h, v in hourly_sales.items()}

    return {
        "total_sales": round(total_sales, 2),
        "visit_count": visit_count,
        "average_order_value": average_order_value,
        "revisit_rate": revisit_rate,
        "coupon_usage_rate": coupon_usage_rate,
        "customer_ids": sorted(set(frame["customer_id"])),
        "hourly_sales": hourly_sales_by_str_hour,
        "period_days": int((frame["date"].max() - frame["date"].min()).days) + 1,
        "period_start": frame["date"].min().date().isoformat(),
        "period_end": frame["date"].max().date().isoformat(),
    }


def sales_metrics_from_aggregates(
    aggregates: dict,
    *,
    start_hour: int | None,
    end_hour: int | None,
    known_customer_ids: set[str] | None,
) -> SalesMetrics:
    """저장된 집계(build_verification_aggregates 결과)로 SalesMetrics를 구성한다."""

    total_sales = float(aggregates["total_sales"])
    hourly_sales: dict[str, float] = aggregates["hourly_sales"]

    if start_hour is not None and end_hour is not None:
        if start_hour <= end_hour:
            window_hours = range(start_hour, end_hour + 1)
        else:
            # 자정을 넘기는 시간대(예: 22시~2시)도 지원한다.
            window_hours = [*range(start_hour, 24), *range(0, end_hour + 1)]
        target_sales = sum(hourly_sales.get(str(h), 0.0) for h in window_hours)
    else:
        target_sales = total_sales

    customer_ids = set(aggregates["customer_ids"])
    new_customer_count = (
        0 if known_customer_ids is None else len(customer_ids - known_customer_ids)
    )

    return SalesMetrics(
        target_sales=round(target_sales, 2),
        total_sales=round(total_sales, 2),
        visit_count=int(aggregates["visit_count"]),
        average_order_value=float(aggregates["average_order_value"]),
        revisit_rate=float(aggregates["revisit_rate"]),
        coupon_usage_rate=float(aggregates["coupon_usage_rate"]),
        new_customer_count=new_customer_count,
        # 적용전/적용후 두 분석만으로는 "장기 미방문 후 재방문"을 판별할 이력이 없어
        # 항상 0으로 둔다 — 효과 점수 가중치 10%가 이 지표에서는 반영되지 않는다.
        dormant_customer_return_count=0,
    )
