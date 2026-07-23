from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POS_PATH = ROOT / "data" / "agent" / "coffee_sample.csv"
DEFAULT_NOAA_PATH = (
    ROOT / "data" / "agent" / "2023_lat_39__42__lon_-76__-72_1.1M.csv"
)
NYC_CENTRAL_PARK_STATION = 72505394728

POS_REQUIRED_COLUMNS = {
    "transaction_id",
    "transaction_date",
    "transaction_time",
    "store_id",
    "store_location",
    "product_id",
    "transaction_qty",
    "unit_price",
    "Total_Bill",
    "product_category",
}
EXTERNAL_KEY_COLUMNS = {"date", "store_id"}
EXTERNAL_METADATA_COLUMNS = {
    "source",
    "observed_at",
    "collected_at",
    "spatial_scope",
    "quality_status",
    "holiday_name",
    "nearest_event_distance_m",
    "transit_active_hours",
    "subway_joined",
    "event_geocoded",
    "event_geocode_match_rate",
}


class DetailedSalesDataError(ValueError):
    pass


def _parse_isd_scaled(
    values: pd.Series,
    *,
    value_index: int = 0,
    quality_index: int = 1,
    missing: str,
    scale: float,
) -> pd.Series:
    parts = values.fillna("").astype(str).str.split(",")
    raw = parts.str[value_index]
    quality = parts.str[quality_index]
    parsed = pd.to_numeric(raw.where(raw.ne(missing)), errors="coerce") / scale
    return parsed.where(quality.isin({"0", "1", "4", "5"}))


def load_noaa_isd_daily(
    path: str | Path = DEFAULT_NOAA_PATH,
    *,
    station: int = NYC_CENTRAL_PARK_STATION,
    start_date: str = "2023-01-01",
    end_date: str = "2023-06-30",
) -> pd.DataFrame:
    usecols = ["STATION", "DATE", "TMP", "DEW", "WND", "VIS", "SLP", "AA1", "AA2", "AA3"]
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False):
        station_ids = pd.to_numeric(chunk["STATION"], errors="coerce")
        match = chunk.loc[station_ids.eq(station)].copy()
        if not match.empty:
            selected.append(match)
    if not selected:
        raise DetailedSalesDataError(f"NOAA 관측소를 찾을 수 없습니다: {station}")

    weather = pd.concat(selected, ignore_index=True)
    utc = pd.to_datetime(weather["DATE"], errors="coerce", utc=True)
    weather["observed_at"] = utc.dt.tz_convert("America/New_York")
    weather["date"] = weather["observed_at"].dt.tz_localize(None).dt.normalize()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    weather = weather[weather["date"].between(start, end)].copy()
    if weather.empty:
        raise DetailedSalesDataError("요청 기간에 NOAA 날씨 관측이 없습니다")

    weather["temperature_c"] = _parse_isd_scaled(
        weather["TMP"], missing="+9999", scale=10,
    )
    weather["dew_point_c"] = _parse_isd_scaled(
        weather["DEW"], missing="+9999", scale=10,
    )
    weather["wind_speed_ms"] = _parse_isd_scaled(
        weather["WND"], value_index=3, quality_index=4, missing="9999", scale=10,
    )
    weather["visibility_m"] = _parse_isd_scaled(
        weather["VIS"], missing="999999", scale=1,
    )
    weather["sea_level_pressure_hpa"] = _parse_isd_scaled(
        weather["SLP"], missing="99999", scale=10,
    )

    precipitation_parts: list[pd.Series] = []
    for column in ("AA1", "AA2", "AA3"):
        parts = weather[column].fillna("").astype(str).str.split(",")
        one_hour = parts.str[0].eq("01")
        depth = pd.to_numeric(
            parts.str[1].where(parts.str[1].ne("9999")), errors="coerce",
        ) / 10
        quality = parts.str[3]
        precipitation_parts.append(depth.where(one_hour & quality.isin({"0", "1", "4", "5"})))
    weather["precipitation_1h_mm"] = pd.concat(
        precipitation_parts, axis=1,
    ).max(axis=1, skipna=True)

    daily = (
        weather.groupby("date", as_index=False)
        .agg(
            temperature_c=("temperature_c", "mean"),
            temperature_min_c=("temperature_c", "min"),
            temperature_max_c=("temperature_c", "max"),
            dew_point_c=("dew_point_c", "mean"),
            wind_speed_ms=("wind_speed_ms", "mean"),
            visibility_m=("visibility_m", "mean"),
            sea_level_pressure_hpa=("sea_level_pressure_hpa", "mean"),
            weather_observations=("observed_at", "size"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    precipitation = (
        weather.groupby("date")["precipitation_1h_mm"].sum(min_count=1)
    )
    daily["precipitation_mm"] = daily["date"].map(precipitation)
    daily["source"] = f"NOAA ISD station {station}"
    return daily


def load_pos_transactions(path: str | Path = DEFAULT_POS_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(POS_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise DetailedSalesDataError(f"POS 필수 컬럼 누락: {missing}")
    if frame.empty:
        raise DetailedSalesDataError("POS 데이터가 비어 있습니다")
    if frame["transaction_id"].duplicated().any():
        raise DetailedSalesDataError("중복 transaction_id가 있습니다")

    frame = frame.copy()
    day_first_dates = pd.to_datetime(
        frame["transaction_date"], format="%d-%m-%Y", errors="coerce",
    )
    iso_dates = pd.to_datetime(
        frame["transaction_date"], format="%Y-%m-%d", errors="coerce",
    )
    frame["date"] = day_first_dates.fillna(iso_dates)
    parsed_time = pd.to_datetime(
        frame["transaction_time"], format="%H:%M:%S", errors="coerce",
    )
    if frame["date"].isna().any() or parsed_time.isna().any():
        raise DetailedSalesDataError("날짜 또는 시간 형식이 올바르지 않습니다")
    frame["hour"] = parsed_time.dt.hour

    numeric_columns = ["transaction_qty", "unit_price", "Total_Bill"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if frame[numeric_columns].isna().any().any():
        raise DetailedSalesDataError("수량·단가·결제액에 숫자가 아닌 값이 있습니다")
    if (frame[numeric_columns] <= 0).any().any():
        raise DetailedSalesDataError("수량·단가·결제액은 0보다 커야 합니다")

    expected_bill = frame["transaction_qty"] * frame["unit_price"]
    if not np.allclose(expected_bill, frame["Total_Bill"], rtol=0, atol=0.01):
        raise DetailedSalesDataError("수량 × 단가와 Total_Bill이 일치하지 않습니다")
    return frame


def aggregate_daily_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    daily = (
        transactions.groupby(
            ["store_id", "store_location", "date"], observed=True, as_index=False,
        )
        .agg(
            revenue=("Total_Bill", "sum"),
            transaction_count=("transaction_id", "nunique"),
            quantity=("transaction_qty", "sum"),
        )
        .sort_values(["store_id", "date"])
        .reset_index(drop=True)
    )
    daily["average_order_value"] = daily["revenue"] / daily["transaction_count"]
    daily["average_unit_price"] = daily["revenue"] / daily["quantity"]
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day_index"] = (daily["date"] - daily["date"].min()).dt.days
    return daily


def validate_aggregation(transactions: pd.DataFrame, daily: pd.DataFrame) -> None:
    checks = {
        "총매출": (transactions["Total_Bill"].sum(), daily["revenue"].sum()),
        "총수량": (transactions["transaction_qty"].sum(), daily["quantity"].sum()),
        "거래건수": (
            transactions["transaction_id"].nunique(),
            daily["transaction_count"].sum(),
        ),
    }
    mismatched = [
        name for name, (raw, aggregated) in checks.items()
        if not np.isclose(raw, aggregated, rtol=0, atol=0.01)
    ]
    if mismatched:
        raise DetailedSalesDataError(f"원본과 집계 결과 불일치: {mismatched}")


def _contribution_rows(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    dimension: str,
    revenue_change: float,
) -> list[dict[str, Any]]:
    baseline_sales = baseline.groupby(dimension, observed=True)["Total_Bill"].sum()
    current_sales = current.groupby(dimension, observed=True)["Total_Bill"].sum()
    values = baseline_sales.index.union(current_sales.index)
    rows = []
    for value in values:
        before = float(baseline_sales.get(value, 0))
        after = float(current_sales.get(value, 0))
        contribution = after - before
        rows.append({
            "value": str(value),
            "baselineRevenue": round(before, 2),
            "currentRevenue": round(after, 2),
            "contributionAmount": round(contribution, 2),
            "contributionPct": (
                round(contribution / revenue_change * 100, 3)
                if not np.isclose(revenue_change, 0) else None
            ),
            "direction": "positive" if contribution > 0 else "negative" if contribution < 0 else "neutral",
            "evidenceType": "decomposition",
        })
    return sorted(rows, key=lambda row: abs(row["contributionAmount"]), reverse=True)


def _price_quantity_decomposition(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    revenue_change: float,
) -> dict[str, Any]:
    def product_totals(frame: pd.DataFrame) -> pd.DataFrame:
        totals = frame.groupby("product_id", observed=True).agg(
            quantity=("transaction_qty", "sum"),
            revenue=("Total_Bill", "sum"),
        )
        totals["price"] = totals["revenue"] / totals["quantity"]
        return totals

    base = product_totals(baseline)
    curr = product_totals(current)
    products = base.index.union(curr.index)
    quantity_effect = 0.0
    price_effect = 0.0
    product_entry_effect = 0.0
    details = []
    for product_id in products:
        in_base = product_id in base.index
        in_current = product_id in curr.index
        if in_base and in_current:
            q0, p0 = float(base.at[product_id, "quantity"]), float(base.at[product_id, "price"])
            q1, p1 = float(curr.at[product_id, "quantity"]), float(curr.at[product_id, "price"])
            q_effect = (q1 - q0) * (p0 + p1) / 2
            p_effect = (p1 - p0) * (q0 + q1) / 2
            entry_effect = 0.0
        elif in_current:
            q_effect = p_effect = 0.0
            entry_effect = float(curr.at[product_id, "revenue"])
        else:
            q_effect = p_effect = 0.0
            entry_effect = -float(base.at[product_id, "revenue"])
        quantity_effect += q_effect
        price_effect += p_effect
        product_entry_effect += entry_effect
        details.append({
            "productId": str(product_id),
            "quantityEffect": round(q_effect, 2),
            "priceEffect": round(p_effect, 2),
            "productEntryEffect": round(entry_effect, 2),
        })

    effects = {
        "quantityEffect": round(quantity_effect, 2),
        "priceEffect": round(price_effect, 2),
        "productEntryEffect": round(product_entry_effect, 2),
    }
    effects["reconciledChange"] = round(sum(effects.values()), 2)
    effects["revenueChange"] = round(revenue_change, 2)
    effects["details"] = sorted(
        details,
        key=lambda row: abs(
            row["quantityEffect"] + row["priceEffect"] + row["productEntryEffect"],
        ),
        reverse=True,
    )
    return effects


def analyze_internal_drivers(
    transactions: pd.DataFrame,
    *,
    baseline_start: str | pd.Timestamp | None = None,
    baseline_end: str | pd.Timestamp | None = None,
    current_start: str | pd.Timestamp | None = None,
    current_end: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    transactions = transactions.copy()
    if "date" not in transactions:
        transactions["date"] = pd.to_datetime(
            transactions["transaction_date"], format="%d-%m-%Y", errors="coerce",
        )
    if "hour" not in transactions:
        transactions["hour"] = pd.to_datetime(
            transactions["transaction_time"], format="%H:%M:%S", errors="coerce",
        ).dt.hour
    if transactions[["date", "hour"]].isna().any().any():
        raise DetailedSalesDataError("내부 원인 분석의 날짜 또는 시간이 올바르지 않습니다")

    max_date = transactions["date"].max().normalize()
    inferred_current_start = max_date.replace(day=1)
    inferred_baseline_end = inferred_current_start - pd.Timedelta(days=1)
    current_start_ts = pd.Timestamp(current_start or inferred_current_start)
    current_end_ts = pd.Timestamp(current_end or max_date)
    baseline_end_ts = pd.Timestamp(baseline_end or inferred_baseline_end)
    comparison_days = (current_end_ts - current_start_ts).days + 1
    inferred_baseline_start = baseline_end_ts - pd.Timedelta(days=comparison_days - 1)
    baseline_start_ts = pd.Timestamp(baseline_start or inferred_baseline_start)

    baseline = transactions[
        transactions["date"].between(baseline_start_ts, baseline_end_ts)
    ]
    current = transactions[
        transactions["date"].between(current_start_ts, current_end_ts)
    ]
    if baseline.empty or current.empty:
        raise DetailedSalesDataError("내부 원인 분해에 필요한 비교 기간 데이터가 부족합니다")

    baseline_revenue = float(baseline["Total_Bill"].sum())
    current_revenue = float(current["Total_Bill"].sum())
    baseline_count = int(baseline["transaction_id"].nunique())
    current_count = int(current["transaction_id"].nunique())
    baseline_aov = baseline_revenue / baseline_count
    current_aov = current_revenue / current_count
    revenue_change = current_revenue - baseline_revenue

    count_effect = (current_count - baseline_count) * (baseline_aov + current_aov) / 2
    aov_effect = (current_aov - baseline_aov) * (baseline_count + current_count) / 2
    summary = {
        "baselineRevenue": round(baseline_revenue, 2),
        "currentRevenue": round(current_revenue, 2),
        "revenueChange": round(revenue_change, 2),
        "revenueChangePct": round(revenue_change / baseline_revenue * 100, 3),
        "baselineTransactionCount": baseline_count,
        "currentTransactionCount": current_count,
        "transactionChangePct": round((current_count / baseline_count - 1) * 100, 3),
        "baselineAverageOrderValue": round(baseline_aov, 4),
        "currentAverageOrderValue": round(current_aov, 4),
        "averageOrderValueChangePct": round((current_aov / baseline_aov - 1) * 100, 3),
    }
    factor_decomposition = [
        {
            "factor": "transaction_count",
            "contributionAmount": round(count_effect, 2),
            "contributionPct": (
                round(count_effect / revenue_change * 100, 3)
                if not np.isclose(revenue_change, 0) else None
            ),
            "evidenceType": "decomposition",
        },
        {
            "factor": "average_order_value",
            "contributionAmount": round(aov_effect, 2),
            "contributionPct": (
                round(aov_effect / revenue_change * 100, 3)
                if not np.isclose(revenue_change, 0) else None
            ),
            "evidenceType": "decomposition",
        },
    ]
    return {
        "period": {
            "baselineStart": baseline_start_ts.date().isoformat(),
            "baselineEnd": baseline_end_ts.date().isoformat(),
            "currentStart": current_start_ts.date().isoformat(),
            "currentEnd": current_end_ts.date().isoformat(),
        },
        "summary": summary,
        "revenueFormulaDrivers": factor_decomposition,
        "dimensionDrivers": {
            "store": _contribution_rows(
                baseline, current, "store_location", revenue_change,
            ),
            "hour": _contribution_rows(baseline, current, "hour", revenue_change),
            "category": _contribution_rows(
                baseline, current, "product_category", revenue_change,
            ),
        },
        "priceQuantityDrivers": _price_quantity_decomposition(
            baseline, current, revenue_change,
        ),
    }


FACTOR_LABELS = {
    "is_weekend": "주말",
    "is_federal_holiday": "연방 공휴일",
    "is_month_start": "월초",
    "is_month_end": "월말",
    "days_to_nearest_holiday": "공휴일과의 날짜 간격",
    "is_holiday_eve": "공휴일 전날",
    "subway_ridership": "인근 지하철 이용량",
    "subway_exposure": "거리 가중 지하철 유동인구",
    "event_count_2km": "인근 행사 수",
    "event_exposure": "인근 행사 노출도",
    "temperature_c": "평균 기온",
    "temperature_min_c": "최저 기온",
    "temperature_max_c": "최고 기온",
    "dew_point_c": "이슬점",
    "wind_speed_ms": "풍속",
    "visibility_m": "가시거리",
    "sea_level_pressure_hpa": "해면기압",
    "precipitation_mm": "강수량",
}


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    total = len(values)
    for rank_index in range(total - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * total / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def _confidence(
    adjusted_p_value: float,
    join_rate: float,
    validation_gain_pct: float,
) -> str:
    if join_rate < 0.8 or adjusted_p_value > 0.1 or validation_gain_pct <= 0:
        return "low"
    if adjusted_p_value <= 0.01 and join_rate >= 0.95:
        return "high"
    if adjusted_p_value <= 0.05:
        return "medium"
    return "low"


def _time_series_validation(frame: pd.DataFrame, factor: str) -> float:
    columns = ["date", "revenue", factor, "store_id", "day_of_week", "month", "day_index"]
    data = frame[columns].dropna().sort_values(["date", "store_id"]).copy()
    dates = pd.Index(data["date"].drop_duplicates().sort_values())
    if len(dates) < 60 or data[factor].nunique() < 2:
        return 0.0

    categorical_controls = ["store_id", "month"]
    if factor != "is_weekend":
        categorical_controls.append("day_of_week")
    controls = pd.concat(
        [
            data[["day_index"]].reset_index(drop=True),
            pd.get_dummies(
                data[categorical_controls].astype(str).reset_index(drop=True),
                drop_first=True,
                dtype=float,
            ),
        ],
        axis=1,
    )
    factor_values = data[factor].reset_index(drop=True).astype(float)
    if not set(factor_values.unique()).issubset({0, 1}):
        std = float(factor_values.std(ddof=0))
        if std == 0:
            return 0.0
        factor_values = (factor_values - factor_values.mean()) / std
    full = controls.copy()
    full["_factor"] = factor_values
    target = np.log1p(data["revenue"].reset_index(drop=True).astype(float))
    row_dates = data["date"].reset_index(drop=True)

    baseline_errors = []
    full_errors = []
    for train_date_indexes, test_date_indexes in TimeSeriesSplit(n_splits=3).split(dates):
        train_dates = set(dates[train_date_indexes])
        test_dates = set(dates[test_date_indexes])
        train_mask = row_dates.isin(train_dates)
        test_mask = row_dates.isin(test_dates)
        baseline_model = LinearRegression().fit(controls[train_mask], target[train_mask])
        full_model = LinearRegression().fit(full[train_mask], target[train_mask])
        baseline_errors.append(mean_absolute_error(
            target[test_mask],
            baseline_model.predict(controls[test_mask]),
        ))
        full_errors.append(mean_absolute_error(
            target[test_mask],
            full_model.predict(full[test_mask]),
        ))
    baseline_mae = float(np.mean(baseline_errors))
    full_mae = float(np.mean(full_errors))
    return (baseline_mae - full_mae) / baseline_mae * 100 if baseline_mae else 0.0


def _external_period_attribution(
    frame: pd.DataFrame,
    factor_columns: list[str],
    internal: dict[str, Any],
    join_rates: dict[str, float],
) -> list[dict[str, Any]]:
    period = internal["period"]
    baseline = frame[frame["date"].between(period["baselineStart"], period["baselineEnd"])]
    current = frame[frame["date"].between(period["currentStart"], period["currentEnd"])]
    baseline_revenue = float(internal["summary"]["baselineRevenue"])
    candidates = []
    for factor in factor_columns:
        model_result = _fit_factor(frame, factor, "revenue")
        if model_result is None:
            continue
        usable = frame[["date", factor]].dropna()
        unique_values = set(usable[factor].unique())
        is_binary = unique_values.issubset({0, 1})
        factor_std = float(usable[factor].std(ddof=0))
        before = float(baseline[factor].mean())
        after = float(current[factor].mean())
        if is_binary:
            model_delta = after - before
        elif factor_std > 0:
            model_delta = (after - before) / factor_std
        else:
            continue
        effect_for_period_pct = float(
            np.expm1(np.log1p(model_result["effectPct"] / 100) * model_delta) * 100,
        )
        contribution = baseline_revenue * effect_for_period_pct / 100
        validation_gain = _time_series_validation(frame, factor)
        candidates.append({
            "factor": factor,
            "label": FACTOR_LABELS.get(factor, factor),
            "baselineValue": round(before, 4),
            "currentValue": round(after, 4),
            "changeValue": round(after - before, 4),
            "estimatedContributionAmount": round(contribution, 2),
            "estimatedEffectPct": round(effect_for_period_pct, 3),
            "rawPValue": model_result["pValue"],
            "joinRate": join_rates.get(factor, 0.0),
            "validationGainPct": round(validation_gain, 3),
            "effectUnit": model_result["effectUnit"],
            "evidenceType": "association",
        })

    adjusted = _benjamini_hochberg([item["rawPValue"] for item in candidates])
    for item, adjusted_p in zip(candidates, adjusted):
        item["adjustedPValue"] = round(adjusted_p, 6)
        item["confidence"] = _confidence(
            adjusted_p,
            item["joinRate"],
            item["validationGainPct"],
        )
        item["direction"] = (
            "positive" if item["estimatedContributionAmount"] > 0
            else "negative" if item["estimatedContributionAmount"] < 0
            else "neutral"
        )
    return sorted(
        candidates,
        key=lambda item: abs(item["estimatedContributionAmount"]),
        reverse=True,
    )


def _driver_sentence(driver: dict[str, Any]) -> str:
    direction = "상승" if driver["estimatedContributionAmount"] > 0 else "하락"
    return (
        f"{driver['label']} 변화도 매출 {direction}과 관련된 것으로 보입니다"
        f"(추정 영향 {driver['estimatedEffectPct']:+.1f}%)."
    )


def build_root_cause_analysis(
    internal: dict[str, Any],
    external_drivers: list[dict[str, Any]],
    *,
    data_warnings: list[str],
) -> dict[str, Any]:
    summary = internal["summary"]
    change = float(summary["revenueChange"])
    direction = "increase" if change > 0 else "decrease" if change < 0 else "stable"
    direction_ko = "상승" if change > 0 else "하락" if change < 0 else "유지"
    aligned_direction = "positive" if change > 0 else "negative"

    formula_drivers = sorted(
        internal["revenueFormulaDrivers"],
        key=lambda item: abs(item["contributionAmount"]),
        reverse=True,
    )
    primary_formula = next(
        (
            item for item in formula_drivers
            if (item["contributionAmount"] > 0) == (change > 0)
        ),
        formula_drivers[0],
    )
    internal_label = {
        "transaction_count": "거래 건수",
        "average_order_value": "객단가",
    }[primary_formula["factor"]]

    reliable_external = [
        item for item in external_drivers
        if item["confidence"] in {"high", "medium"}
        and item["direction"] == aligned_direction
    ]
    excluded_external = [
        {
            "factor": item["factor"],
            "reason": (
                "data_quality" if item["joinRate"] < 0.8
                else "multiple_testing_not_significant"
            ),
        }
        for item in external_drivers
        if item["confidence"] == "low"
    ]

    detailed_internal = []
    detail_labels = {
        "store": lambda value: f"{value} 매장",
        "hour": lambda value: f"{value}시 시간대",
        "category": lambda value: f"{value} 카테고리",
    }
    for dimension, drivers in internal.get("dimensionDrivers", {}).items():
        aligned = next(
            (
                item for item in drivers
                if (item["contributionAmount"] > 0) == (change > 0)
            ),
            None,
        )
        if aligned:
            detailed_internal.append({
                **aligned,
                "dimension": dimension,
                "label": detail_labels[dimension](aligned["value"]),
            })

    narrative = (
        f"분석 기간 매출은 비교 기간보다 {abs(summary['revenueChangePct']):.1f}% "
        f"{direction_ko}했습니다. {internal_label} 변화가 매출 {direction_ko}분에서 "
        f"가장 큰 내부 요인으로 확인됐습니다."
    )
    if detailed_internal:
        detail_text = ", ".join(item["label"] for item in detailed_internal[:3])
        narrative += f" 세부적으로는 {detail_text}의 변화가 같은 방향으로 기여했습니다."
    if reliable_external:
        narrative += " " + " ".join(_driver_sentence(item) for item in reliable_external[:2])
    else:
        narrative += " 품질과 다중검정 기준을 통과한 외부 원인은 확인되지 않았습니다."
    if data_warnings:
        narrative += " 일부 외부 데이터는 품질이 낮아 사용자 설명에서 제외했습니다."

    return {
        "change": {
            "direction": direction,
            "amount": summary["revenueChange"],
            "ratePct": summary["revenueChangePct"],
        },
        "headline": f"{internal_label} 변화를 중심으로 매출이 {direction_ko}한 것으로 분석됩니다.",
        "narrative": narrative,
        "internalDrivers": formula_drivers,
        "internalDetailedDrivers": detailed_internal,
        "externalDrivers": reliable_external,
        "excludedExternalDrivers": excluded_external,
        "limitations": data_warnings,
        "evidencePolicy": {
            "decomposition": "실제 매출 합계와 일치하는 산술 분해",
            "association": "통제 회귀와 다중검정을 통과한 연관성 추정",
            "causal": "현재 결과에는 포함되지 않음",
        },
    }


def load_external_factors(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    factors = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in factors:
        raise DetailedSalesDataError("외부요인 필수 컬럼 누락: date")

    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    if factors["date"].isna().any():
        raise DetailedSalesDataError("외부요인 date 형식이 올바르지 않습니다")

    join_keys = ["date"]
    if "store_id" in factors:
        factors["store_id"] = pd.to_numeric(factors["store_id"], errors="coerce")
        if factors["store_id"].isna().any():
            raise DetailedSalesDataError("외부요인 store_id 형식이 올바르지 않습니다")
        join_keys.insert(0, "store_id")
    if factors.duplicated(join_keys).any():
        raise DetailedSalesDataError(f"외부요인 결합 키가 중복됩니다: {join_keys}")

    candidates = [
        column for column in factors.columns
        if column not in EXTERNAL_KEY_COLUMNS | EXTERNAL_METADATA_COLUMNS
    ]
    if not candidates:
        raise DetailedSalesDataError("분석할 외부요인 컬럼이 없습니다")
    for column in candidates:
        factors[column] = pd.to_numeric(factors[column], errors="coerce")
        if factors[column].notna().sum() == 0:
            raise DetailedSalesDataError(f"외부요인 컬럼이 숫자가 아닙니다: {column}")
    return factors, candidates


def merge_external_factors(
    daily: pd.DataFrame,
    factors: pd.DataFrame,
    factor_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    join_keys = ["date"]
    if "store_id" in factors:
        join_keys.insert(0, "store_id")
    merged = daily.merge(factors, on=join_keys, how="left", validate="many_to_one")
    join_rates = {
        column: round(float(merged[column].notna().mean()), 4)
        for column in factor_columns
    }
    return merged, join_rates


def _fit_factor(
    frame: pd.DataFrame,
    factor: str,
    target: str,
) -> dict[str, Any] | None:
    columns = [target, factor, "store_id", "date", "day_of_week", "month", "day_index"]
    data = frame[columns].dropna().copy()
    if len(data) < 30 or data[factor].nunique() < 2:
        return None

    is_binary = set(data[factor].unique()).issubset({0, 1})
    if is_binary:
        data["_factor_model"] = data[factor].astype(float)
        effect_unit = "binary_0_to_1"
    else:
        factor_std = float(data[factor].std(ddof=0))
        if factor_std == 0:
            return None
        data["_factor_model"] = (data[factor] - data[factor].mean()) / factor_std
        effect_unit = "one_standard_deviation"
    y = np.log1p(data[target].astype(float))
    categorical_controls = ["store_id", "month"]
    if factor != "is_weekend":
        categorical_controls.append("day_of_week")
    controls = pd.concat(
        [
            data[["_factor_model", "day_index"]],
            pd.get_dummies(
                data[categorical_controls].astype(str),
                drop_first=True,
                dtype=float,
            ),
        ],
        axis=1,
    )
    model = sm.OLS(y, sm.add_constant(controls, has_constant="add")).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["date"]},
    )
    coefficient = float(model.params["_factor_model"])
    low, high = model.conf_int().loc["_factor_model"].astype(float)
    return {
        "factor": factor,
        "target": target,
        "effectPct": round(float(np.expm1(coefficient) * 100), 3),
        "effectUnit": effect_unit,
        "confidenceIntervalPct": [
            round(float(np.expm1(low) * 100), 3),
            round(float(np.expm1(high) * 100), 3),
        ],
        "pValue": round(float(model.pvalues["_factor_model"]), 6),
        "observations": int(model.nobs),
        "evidenceType": "association",
    }


def analyze_external_factors(
    daily: pd.DataFrame,
    factor_columns: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for factor in factor_columns:
        for target in ("revenue", "transaction_count", "average_order_value"):
            result = _fit_factor(daily, factor, target)
            if result is not None:
                results.append(result)
    return results


def run_analysis(
    pos_path: str | Path = DEFAULT_POS_PATH,
    external_path: str | Path | None = None,
    noaa_path: str | Path | None = None,
    noaa_station: int = NYC_CENTRAL_PARK_STATION,
) -> dict[str, Any]:
    transactions = load_pos_transactions(pos_path)
    daily = aggregate_daily_sales(transactions)
    validate_aggregation(transactions, daily)
    daily_sales = (
        daily.groupby("date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            transactionCount=("transaction_count", "sum"),
            quantity=("quantity", "sum"),
        )
        .sort_values("date")
    )
    daily_sales["averageOrderValue"] = (
        daily_sales["revenue"] / daily_sales["transactionCount"]
    )

    result: dict[str, Any] = {
        "dataSummary": {
            "transactionRows": int(len(transactions)),
            "aggregatedRows": int(len(daily)),
            "storeCount": int(daily["store_id"].nunique()),
            "startDate": daily["date"].min().date().isoformat(),
            "endDate": daily["date"].max().date().isoformat(),
            "totalRevenue": round(float(daily["revenue"].sum()), 2),
            "totalQuantity": int(daily["quantity"].sum()),
            "totalTransactions": int(daily["transaction_count"].sum()),
        },
        "dailySales": [
            {
                "date": row.date.date().isoformat(),
                "revenue": round(float(row.revenue), 2),
                "transactionCount": int(row.transactionCount),
                "quantity": int(row.quantity),
                "averageOrderValue": round(float(row.averageOrderValue), 2),
            }
            for row in daily_sales.itertuples(index=False)
        ],
        "externalFactors": [],
        "internalAnalysis": analyze_internal_drivers(transactions),
        "dataQuality": {
            "aggregationValidated": True,
            "externalJoinRate": {},
            "warnings": [],
        },
    }
    if external_path is None and noaa_path is None:
        result["dataQuality"]["warnings"].append(
            "외부요인 데이터가 없어 POS 집계만 수행했습니다",
        )
        result["rootCauseAnalysis"] = build_root_cause_analysis(
            result["internalAnalysis"],
            [],
            data_warnings=result["dataQuality"]["warnings"],
        )
        return result

    factors: pd.DataFrame
    factor_columns: list[str]
    if external_path is not None:
        factors, factor_columns = load_external_factors(external_path)
    else:
        factors = pd.DataFrame()
        factor_columns = []
    if noaa_path is not None:
        weather = load_noaa_isd_daily(
            noaa_path,
            station=noaa_station,
            start_date=daily["date"].min().date().isoformat(),
            end_date=daily["date"].max().date().isoformat(),
        )
        weather_columns = [
            column for column in weather.columns
            if column not in {"date", "source", "weather_observations"}
        ]
        weather = weather.drop(columns=["source"], errors="ignore")
        if factors.empty:
            factors = weather
        else:
            factors = factors.merge(weather, on="date", how="left", validate="many_to_one")
        factor_columns.extend(weather_columns)
    merged, join_rates = merge_external_factors(daily, factors, factor_columns)
    result["externalFactors"] = analyze_external_factors(merged, factor_columns)
    result["dataQuality"]["externalJoinRate"] = join_rates
    for factor, rate in join_rates.items():
        if rate < 0.8:
            result["dataQuality"]["warnings"].append(
                f"{factor} 외부요인 결합률이 낮습니다: {rate:.1%}",
            )
    if "event_geocode_match_rate" in merged:
        match_rate = float(merged["event_geocode_match_rate"].dropna().min())
        if match_rate < 0.8:
            result["dataQuality"]["warnings"].append(
                f"행사 장소 좌표 매칭률이 낮습니다: {match_rate:.1%}",
            )
    external_attribution = _external_period_attribution(
        merged,
        factor_columns,
        result["internalAnalysis"],
        join_rates,
    )
    result["rootCauseAnalysis"] = build_root_cause_analysis(
        result["internalAnalysis"],
        external_attribution,
        data_warnings=result["dataQuality"]["warnings"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="상세 POS 매출과 외부요인 연관성 분석")
    parser.add_argument("--pos", type=Path, default=DEFAULT_POS_PATH)
    parser.add_argument("--external", type=Path)
    parser.add_argument("--noaa", type=Path)
    parser.add_argument("--noaa-station", type=int, default=NYC_CENTRAL_PARK_STATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = run_analysis(args.pos, args.external, args.noaa, args.noaa_station)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
