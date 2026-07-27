"""Generate a POS-like cafe sample using KMA 2026 Q2 Seoul weather.

The weather comes from KMA daily observations. Every receipt, item, channel,
payment method, quantity, and sales amount is synthetic test data.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEATHER_DATA = ROOT / "data" / "source" / "weather_seoul_daily_recent.csv"
SAMPLE_DIR = ROOT / "samples"
OMNICHANNEL_OUTPUT = SAMPLE_DIR / "sample_cafe_pos_omnichannel_weather_2026Q2.csv"

PRODUCTS = [
    (1, "커피", "아메리카노", "핫", 4500),
    (2, "커피", "카페라떼", "아이스", 5000),
    (3, "논커피음료", "핫초코", "핫", 5200),
    (4, "디저트", "크루아상", "플레인", 3800),
    (5, "베이커리", "스콘", "플레인", 4200),
    (6, "디저트", "치즈케이크", "조각", 6500),
]
DAY_NAMES = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def _load_weather() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_DATA, encoding="utf-8-sig")
    weather["date"] = pd.to_datetime(weather["date"], errors="raise")
    weather["temperature_c"] = pd.to_numeric(
        weather["temperature_c"],
        errors="raise",
    )
    weather["precipitation_mm"] = pd.to_numeric(
        weather["precipitation_mm"],
        errors="coerce",
    ).fillna(0)
    return weather.loc[
        weather["date"].between("2026-04-01", "2026-06-30"),
    ].reset_index(drop=True)


def _transaction_row(
    *,
    day,
    order_index: int,
    transaction_id: int,
    channel: str,
    rng: random.Random,
) -> dict:
    product_id, category, product_type, detail, unit_price = PRODUCTS[
        (order_index + day.date.day + transaction_id) % len(PRODUCTS)
    ]
    is_delivery = channel == "배달"
    if is_delivery:
        hour = 11 + (order_index * 7 + day.date.day) % 11
        order_type = "배달"
        payment_method = rng.choice(["배달앱선결제", "카드"])
        platform = rng.choice(["배달의민족", "요기요", "쿠팡이츠"])
        terminal_id = "DELIVERY-01"
    else:
        hour = 8 + (order_index * 5 + day.date.day) % 14
        order_type = rng.choices(["매장식사", "포장"], weights=[0.68, 0.32])[0]
        payment_method = rng.choices(
            ["카드", "현금", "카카오페이", "네이버페이"],
            weights=[0.67, 0.08, 0.15, 0.10],
        )[0]
        platform = ""
        terminal_id = rng.choice(["POS-01", "POS-02"])
    minute = rng.randrange(60)
    second = rng.randrange(60)
    quantity = 1 + int(rng.random() < 0.18)
    return {
        "transaction_id": transaction_id,
        "receipt_number": f"R-{day.date:%Y%m%d}-{transaction_id:06d}",
        "transaction_date": day.date.strftime("%d-%m-%Y"),
        "transaction_time": f"{hour:02d}:{minute:02d}:{second:02d}",
        "store_id": 1,
        "store_location": "강남역점",
        "terminal_id": terminal_id,
        "sales_channel": channel,
        "order_type": order_type,
        "delivery_platform": platform,
        "payment_method": payment_method,
        "product_id": product_id,
        "transaction_qty": quantity,
        "unit_price": unit_price,
        "Total_Bill": quantity * unit_price,
        "product_category": category,
        "product_type": product_type,
        "product_detail": detail,
        "Size": "중형",
        "Month Name": f"{day.date.month}월",
        "Day Name": DAY_NAMES[day.date.dayofweek],
        "Hour": hour,
        "Month": day.date.month,
        "Day of Week": day.date.dayofweek,
    }


def _generate_rows(weather: pd.DataFrame) -> list[dict]:
    rng = random.Random(20260727)
    rows: list[dict] = []
    transaction_id = 600_001
    for day in weather.itertuples(index=False):
        weekend_bonus = 3 if day.date.dayofweek >= 5 else 0
        rain_bonus = min(7, round(float(day.precipitation_mm) * 0.12))
        delivery_orders = max(
            8,
            round(
                35
                - 0.75 * float(day.temperature_c)
                + weekend_bonus
                + rain_bonus
            ),
        )
        store_orders = max(
            14,
            round(19 + 0.18 * float(day.temperature_c) + weekend_bonus),
        )
        for channel, order_count in (
            ("배달", delivery_orders),
            ("매장", store_orders),
        ):
            for order_index in range(order_count):
                rows.append(
                    _transaction_row(
                        day=day,
                        order_index=order_index,
                        transaction_id=transaction_id,
                        channel=channel,
                        rng=rng,
                    ),
                )
                transaction_id += 1
    return rows


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    delivery_count = sum(row["sales_channel"] == "배달" for row in rows)
    print(f"생성 완료: {path}")
    print(
        f"거래 {len(rows):,}건 "
        f"(배달 {delivery_count:,}건, 매장 {len(rows) - delivery_count:,}건)",
    )


def main() -> None:
    weather = _load_weather()
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    _write(OMNICHANNEL_OUTPUT, _generate_rows(weather))
    print(f"기간: {weather['date'].min().date()}~{weather['date'].max().date()}")
    print(
        "주의: 날씨는 기상청 관측자료이며 POS 거래와 매출은 모두 합성 데이터입니다.",
    )


if __name__ == "__main__":
    main()
