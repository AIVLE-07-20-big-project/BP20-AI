from __future__ import annotations

import io
from typing import Any

import pandas as pd

from app.core.config import (
    FOOT_TRAFFIC,
    SEOUL_EVENT_EXPOSURE,
    SEOUL_SUBWAY_EXPOSURE,
    SEOUL_WEATHER_MONTHLY,
)
from scripts.modeling.detailed_sales_external_analysis import (
    load_pos_transactions,
    run_analysis,
)


def _quarter_code(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 10 + ((dates.dt.month - 1) // 3 + 1)


def _lookup_by_quarter(path, trdar_cd: str, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["TRDAR_CD"] = pd.to_numeric(frame["TRDAR_CD"], errors="coerce")
    selected = frame.loc[
        frame["TRDAR_CD"].eq(int(trdar_cd)),
        ["STDR_YYQU_CD", *columns],
    ].copy()
    return selected.drop_duplicates("STDR_YYQU_CD")


def _build_seoul_factors(file_bytes: bytes, trdar_cd: str) -> tuple[pd.DataFrame, list[str]]:
    transactions = load_pos_transactions(io.BytesIO(file_bytes))
    dates = pd.DataFrame(
        {"date": pd.date_range(transactions["date"].min(), transactions["date"].max())},
    )
    dates["STDR_YYQU_CD"] = _quarter_code(dates["date"])
    dates["is_weekend"] = dates["date"].dt.dayofweek.ge(5).astype(int)
    dates["is_month_start"] = dates["date"].dt.is_month_start.astype(int)
    dates["is_month_end"] = dates["date"].dt.is_month_end.astype(int)
    limitations: list[str] = []

    if SEOUL_WEATHER_MONTHLY.exists():
        weather = pd.read_csv(SEOUL_WEATHER_MONTHLY)
        weather = weather.rename(
            columns={
                "rn": "precipitation_mm_monthly",
                "rn_day": "rain_days_monthly",
                "ws": "wind_speed_ms_monthly",
                "tm_max": "temperature_max_c_monthly",
            },
        )
        weather_columns = [
            column for column in (
                "precipitation_mm_monthly",
                "rain_days_monthly",
                "wind_speed_ms_monthly",
                "temperature_max_c_monthly",
            )
            if column in weather
        ]
        dates = dates.merge(
            weather[["year", "month", *weather_columns]],
            left_on=[dates["date"].dt.year, dates["date"].dt.month],
            right_on=["year", "month"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["key_0", "key_1", "year", "month"], errors="ignore")
        if weather_columns and dates[weather_columns].notna().any().any():
            limitations.append("서울 날씨는 월 단위 값을 POS 일자에 확장해 사용했습니다.")
        else:
            limitations.append("POS 기간에 해당하는 서울 월별 날씨 데이터가 없습니다.")

    quarterly_sources = [
        (FOOT_TRAFFIC, ["TOT_FLPOP_CO"], {"TOT_FLPOP_CO": "foot_traffic_quarterly"}),
        (
            SEOUL_EVENT_EXPOSURE,
            ["event_count", "event_days", "event_exposure"],
            {
                "event_count": "event_count_quarterly",
                "event_days": "event_days_quarterly",
                "event_exposure": "event_exposure_quarterly",
            },
        ),
        (
            SEOUL_SUBWAY_EXPOSURE,
            ["subway_exposure", "subway_station_count"],
            {
                "subway_exposure": "subway_exposure_quarterly",
                "subway_station_count": "subway_station_count_quarterly",
            },
        ),
    ]
    matched_quarterly = False
    for path, columns, rename in quarterly_sources:
        lookup = _lookup_by_quarter(path, trdar_cd, columns)
        if lookup.empty:
            continue
        dates = dates.merge(
            lookup.rename(columns=rename),
            on="STDR_YYQU_CD",
            how="left",
            validate="many_to_one",
        )
        matched_columns = list(rename.values())
        matched_quarterly = (
            matched_quarterly or dates[matched_columns].notna().any().any()
        )
    if matched_quarterly:
        limitations.append(
            "서울 유동인구·행사·지하철 요인은 상권·분기 단위 값을 POS 일자에 확장해 사용했습니다.",
        )
    else:
        limitations.append("POS 기간·상권에 해당하는 서울 분기 외부요인 데이터가 없습니다.")

    factor_columns = [
        column for column in dates.columns
        if column not in {"date", "STDR_YYQU_CD"} and dates[column].notna().any()
    ]
    return dates[["date", *factor_columns]], limitations


def analyze_uploaded_sales(file_bytes: bytes, trdar_cd: str) -> dict[str, Any]:
    factors, limitations = _build_seoul_factors(file_bytes, trdar_cd)
    factor_bytes = io.BytesIO(factors.to_csv(index=False).encode("utf-8-sig"))
    result = run_analysis(
        pos_path=io.BytesIO(file_bytes),
        external_path=factor_bytes,
    )
    result["dataQuality"]["externalDataRegion"] = "서울"
    result["dataQuality"]["warnings"].extend(limitations)
    root_cause = result["rootCauseAnalysis"]
    root_cause["limitations"] = list(dict.fromkeys(result["dataQuality"]["warnings"]))
    root_cause["narrative"] += " " + " ".join(limitations)
    return result
