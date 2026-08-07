from __future__ import annotations

import io
import sys
from typing import Any

import pandas as pd

from app.core.config import (
    ROOT,
    FOOT_TRAFFIC,
    SEOUL_EVENT_EXPOSURE,
    SEOUL_EVENT_DETAILS,
    SEOUL_SUBWAY_EXPOSURE,
    SEOUL_WEATHER_DAILY,
    SEOUL_WEATHER_MONTHLY,
)

# Celery 워커의 지연 임포트에서도 scripts.*를 찾도록 프로젝트 루트를 추가한다.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.detailed_sales_external_analysis import (
    load_pos_transactions,
    run_analysis,
)
from app.services.pos_verification_aggregates import (
    PosVerificationDataError,
    build_verification_aggregates,
)


def _quarter_code(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 10 + ((dates.dt.month - 1) // 3 + 1)


def _lookup_by_quarter(path, trdar_cd: str, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["TRDAR_CD"] = pd.to_numeric(frame["TRDAR_CD"], errors="coerce")
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns or "STDR_YYQU_CD" not in frame.columns:
        return pd.DataFrame()
    selected = frame.loc[
        frame["TRDAR_CD"].eq(int(trdar_cd)),
        ["STDR_YYQU_CD", *available_columns],
    ].copy()
    return selected.drop_duplicates("STDR_YYQU_CD")


def _build_seoul_factors(
    file_bytes: bytes,
    trdar_cd: str,
) -> tuple[pd.DataFrame, list[str]]:
    transactions = load_pos_transactions(io.BytesIO(file_bytes))
    dates = pd.DataFrame(
        {"date": pd.date_range(transactions["date"].min(), transactions["date"].max())},
    )
    dates["STDR_YYQU_CD"] = _quarter_code(dates["date"])
    dates["is_weekend"] = dates["date"].dt.dayofweek.ge(5).astype(int)
    dates["is_month_start"] = dates["date"].dt.is_month_start.astype(int)
    dates["is_month_end"] = dates["date"].dt.is_month_end.astype(int)
    limitations: list[str] = []
    daily_weather_matched = False

    if SEOUL_WEATHER_DAILY.exists():
        recent_weather = pd.read_csv(SEOUL_WEATHER_DAILY)
        recent_weather["date"] = pd.to_datetime(
            recent_weather["date"],
            errors="coerce",
        )
        recent_columns = [
            column
            for column in (
                "temperature_c",
                "temperature_min_c",
                "temperature_max_c",
                "precipitation_mm",
                "wind_speed_ms",
            )
            if column in recent_weather
        ]
        dates = dates.merge(
            recent_weather[["date", *recent_columns]],
            on="date",
            how="left",
            validate="one_to_one",
        )
        if recent_columns and dates[recent_columns].notna().any().any():
            daily_weather_matched = True
            has_model_substitution = (
                "contains_model_substitution" in recent_weather
                and recent_weather["contains_model_substitution"]
                .astype(str)
                .str.lower()
                .eq("true")
                .any()
            )
            if has_model_substitution:
                limitations.append(
                    "최근 서울 일별 날씨에는 모델 대체값이 포함되어 있습니다.",
                )

    if SEOUL_WEATHER_MONTHLY.exists():
        weather = pd.read_csv(SEOUL_WEATHER_MONTHLY)
        weather = weather.rename(
            columns={
                "rn_day": "precipitation_mm_monthly",
                "rn_day_cnt1": "rain_days_monthly",
                "ws": "wind_speed_ms_monthly",
                "taavg": "temperature_c",
                "avgtamin": "temperature_min_c",
                "avgtamax": "temperature_max_c_monthly",
            },
        )
        weather_columns = [
            column for column in (
                "precipitation_mm_monthly",
                "rain_days_monthly",
                "wind_speed_ms_monthly",
                "temperature_c",
                "temperature_min_c",
                "temperature_max_c_monthly",
            )
            if column in weather and column not in dates
        ]
        if weather_columns:
            dates = dates.merge(
                weather[["year", "month", *weather_columns]],
                left_on=[dates["date"].dt.year, dates["date"].dt.month],
                right_on=["year", "month"],
                how="left",
                validate="many_to_one",
            ).drop(columns=["key_0", "key_1", "year", "month"], errors="ignore")
        if not (
            weather_columns and dates[weather_columns].notna().any().any()
        ) and not daily_weather_matched:
            pos_start = dates["date"].min().date().isoformat()
            pos_end = dates["date"].max().date().isoformat()
            weather_periods = pd.to_datetime(
                {
                    "year": weather["year"],
                    "month": weather["month"],
                    "day": 1,
                },
                errors="coerce",
            ).dropna()
            weather_start = weather_periods.min().strftime("%Y-%m")
            weather_end = weather_periods.max().strftime("%Y-%m")
            limitations.append(
                f"POS 기간({pos_start}~{pos_end})에 해당하는 서울 월별 날씨가 "
                f"없습니다(보유 범위: {weather_start}~{weather_end}).",
            )

    quarterly_sources = [
        (FOOT_TRAFFIC, ["TOT_FLPOP_CO"], {"TOT_FLPOP_CO": "foot_traffic_quarterly"}),
        (
            SEOUL_EVENT_EXPOSURE,
            ["event_count", "event_days", "event_exposure", "event_details_json"],
            {
                "event_count": "event_count_quarterly",
                "event_days": "event_days_quarterly",
                "event_exposure": "event_exposure_quarterly",
                "event_details_json": "event_details_quarterly",
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
    if SEOUL_EVENT_DETAILS.exists():
        event_details = _lookup_by_quarter(
            SEOUL_EVENT_DETAILS,
            trdar_cd,
            ["event_details_json"],
        )
        if not event_details.empty:
            dates = dates.merge(
                event_details.rename(columns={"event_details_json": "event_details_quarterly"}),
                on="STDR_YYQU_CD",
                how="left",
                validate="many_to_one",
            )
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
        # rename 대상 컬럼이 원본 CSV에 없으면(예: event_details_json 미포함) merge
        # 후에도 dates에 생기지 않으므로, 실제로 존재하는 컬럼만 검사한다.
        matched_columns = [c for c in rename.values() if c in dates.columns]
        matched_quarterly = (
            matched_quarterly or (
                bool(matched_columns) and dates[matched_columns].notna().any().any()
            )
        )
    if not matched_quarterly:
        limitations.append("POS 기간·상권에 해당하는 서울 분기 외부요인 데이터가 없습니다.")

    factor_columns = [
        column for column in dates.columns
        if column not in {
            "date",
            "STDR_YYQU_CD",
            # 행사 상세 JSON은 보고서 표시용 메타데이터이며
            # 숫자형 외부요인 분석 대상이 아니다.
            "event_details_quarterly",
        }
        and dates[column].notna().any()
    ]
    return dates[["date", *factor_columns]], limitations


def analyze_uploaded_sales(file_bytes: bytes, trdar_cd: str) -> dict[str, Any]:
    factors, limitations = _build_seoul_factors(
        file_bytes,
        trdar_cd,
    )
    factor_bytes = io.BytesIO(factors.to_csv(index=False).encode("utf-8-sig"))
    result = run_analysis(
        pos_path=io.BytesIO(file_bytes),
        external_path=factor_bytes,
    )
    result["dataQuality"]["externalDataRegion"] = "서울"
    result["dataQuality"]["warnings"].extend(limitations)

    try:
        result["verificationAggregates"] = build_verification_aggregates(file_bytes)
    except PosVerificationDataError as exc:
        result["verificationAggregates"] = None
        result["dataQuality"]["warnings"].append(
            f"매출형 전략검증용 집계를 만들 수 없습니다({exc}) — "
            "이 분석은 나중에 전략검증 비교 대상으로 쓸 수 없습니다.",
        )

    root_cause = result["rootCauseAnalysis"]
    root_cause["limitations"] = list(dict.fromkeys(result["dataQuality"]["warnings"]))
    return result
