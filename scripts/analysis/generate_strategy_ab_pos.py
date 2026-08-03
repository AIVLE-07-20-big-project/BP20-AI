# 매출형 전략검증 테스트용 POS CSV(적용전/적용후)를 생성한다.
#
# /analyses의 내부 원인 분석(analyze_internal_drivers)은 업로드 CSV 안에서 자동으로
# "이번 달(current) vs 지난 달(baseline)"을 잘라 비교하므로, 각 파일은 최소 2개월치
# 데이터를 포함해야 한다. 매출형 전략검증(build_verification_aggregates)은 그중
# "이번 달"만 집계 대상으로 쓴다.
#
#   before.csv: 2026-04-01 ~ 05-31 → 검증 대상 구간 = 5월(전략 적용 전, 4월은 baseline용)
#   after.csv : 2026-05-04 ~ 06-28 → 검증 대상 구간 = 6월(전략 적용 후, 5월은 baseline용)
#
# 적용 전략은 '타임세일'(14~17시 한정 할인쿠폰)이며, 6월의 해당 시간대에만
# 추가 주문·쿠폰 사용을 반영한다.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "samples" / "verification"

STORE_ID = 1
STORE_LOCATION = "강남역점"
CATEGORIES = [
    ("커피", 5000),
    ("논커피음료", 5500),
    ("베이커리", 4500),
    ("디저트", 6500),
]
TARGET_START_HOUR, TARGET_END_HOUR = 14, 17
HOUR_WEIGHTS = np.array([
    0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.5, 2.5, 2.0, 1.5, 2.0,   # 0~11시
    3.0, 2.5, 0.8, 0.7, 0.6, 0.9, 2.5, 3.0, 2.0, 1.2, 0.6, 0.3,   # 12~23시
])
CUSTOMER_POOL = [f"C-{i:04d}" for i in range(1, 201)]

rng = np.random.default_rng(20260731)


def _generate_base_transactions(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    rows = []
    next_id = 1
    for date in dates:
        daily_mean = 22 if date.dayofweek < 5 else 27
        n = rng.poisson(daily_mean)
        hours = rng.choice(24, size=n, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
        for hour in hours:
            minute, second = rng.integers(0, 60), rng.integers(0, 60)
            customer_id = CUSTOMER_POOL[rng.integers(0, len(CUSTOMER_POOL))]
            category, list_price = CATEGORIES[rng.integers(0, len(CATEGORIES))]
            qty = int(rng.integers(1, 3))
            total_bill = qty * list_price
            rows.append({
                "transaction_id": next_id,
                "transaction_date": date.strftime("%d-%m-%Y"),
                "transaction_time": f"{hour:02d}:{minute:02d}:{second:02d}",
                "store_id": STORE_ID,
                "store_location": STORE_LOCATION,
                "product_id": rng.integers(1, 20),
                "customer_id": customer_id,
                "transaction_qty": qty,
                "unit_price": list_price,
                "Total_Bill": total_bill,
                "coupon_used": False,
                "product_category": category,
                "_date": date,
                "_hour": int(hour),
            })
            next_id += 1
    return pd.DataFrame(rows)


def _apply_time_sale(frame: pd.DataFrame, month_start: pd.Timestamp) -> pd.DataFrame:
    """month_start가 속한 달의 14~17시에 타임세일 효과(쿠폰·추가 주문)를 반영한다."""

    frame = frame.copy()
    in_month = frame["_date"] >= month_start
    in_target_hour = frame["_hour"].between(TARGET_START_HOUR, TARGET_END_HOUR)
    boosted = in_month & in_target_hour

    coupon_mask = boosted & (rng.random(len(frame)) < 0.4)
    frame.loc[coupon_mask, "coupon_used"] = True
    frame.loc[coupon_mask, "unit_price"] = (frame.loc[coupon_mask, "unit_price"] * 0.85).round(-1)
    frame.loc[coupon_mask, "Total_Bill"] = (
        frame.loc[coupon_mask, "transaction_qty"] * frame.loc[coupon_mask, "unit_price"]
    )

    month_dates = pd.date_range(month_start, frame.loc[in_month, "_date"].max(), freq="D")
    next_id = int(frame["transaction_id"].max()) + 1
    extra_rows = []
    new_customer_pool = [f"C-NEW-{i:03d}" for i in range(1, 61)]
    for i, date in enumerate(month_dates):
        n_extra = rng.poisson(6)  # 대상 시간대 주문 증분
        hours = rng.integers(TARGET_START_HOUR, TARGET_END_HOUR + 1, size=n_extra)
        for hour in hours:
            minute, second = rng.integers(0, 60), rng.integers(0, 60)
            is_new_customer = rng.random() < 0.5
            customer_id = (
                new_customer_pool[rng.integers(0, len(new_customer_pool))]
                if is_new_customer
                else CUSTOMER_POOL[rng.integers(0, len(CUSTOMER_POOL))]
            )
            category, list_price = CATEGORIES[rng.integers(0, len(CATEGORIES))]
            qty = int(rng.integers(1, 3))
            unit_price = round(list_price * 0.85, -1)
            extra_rows.append({
                "transaction_id": next_id,
                "transaction_date": date.strftime("%d-%m-%Y"),
                "transaction_time": f"{int(hour):02d}:{minute:02d}:{second:02d}",
                "store_id": STORE_ID,
                "store_location": STORE_LOCATION,
                "product_id": rng.integers(1, 20),
                "customer_id": customer_id,
                "transaction_qty": qty,
                "unit_price": unit_price,
                "Total_Bill": qty * unit_price,
                "coupon_used": True,
                "product_category": category,
                "_date": date,
                "_hour": int(hour),
            })
            next_id += 1
    return pd.concat([frame, pd.DataFrame(extra_rows)], ignore_index=True)


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "transaction_id", "transaction_date", "transaction_time", "store_id",
        "store_location", "product_id", "customer_id", "transaction_qty",
        "unit_price", "Total_Bill", "coupon_used", "product_category",
    ]
    return frame.sort_values("transaction_id")[columns]


def main() -> None:
    base = _generate_base_transactions("2026-04-01", "2026-06-28")

    before = base[(base["_date"] >= "2026-04-01") & (base["_date"] <= "2026-05-31")]
    _finalize(before).to_csv(OUTPUT_DIR / "pos_strategy_ab_before.csv", index=False)

    after_source = base[(base["_date"] >= "2026-05-04") & (base["_date"] <= "2026-06-28")]
    after = _apply_time_sale(after_source, month_start=pd.Timestamp("2026-06-01"))
    _finalize(after).to_csv(OUTPUT_DIR / "pos_strategy_ab_after.csv", index=False)

    print(f"before: {len(before)}건, after: {len(after)}건")


if __name__ == "__main__":
    main()
