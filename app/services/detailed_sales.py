from __future__ import annotations

import io
import sys
from typing import Any

import pandas as pd

from app.core.config import (
    ROOT,
    AGENT_DATA,
    FOOT_TRAFFIC,
    SEOUL_EVENT_EXPOSURE,
    SEOUL_SUBWAY_EXPOSURE,
    SEOUL_WEATHER_DAILY,
    SEOUL_WEATHER_MONTHLY,
)

# Celery 워커(Windows, `celery -A app.celery_app worker`)로 이 모듈이 지연 임포트될 때
# "ModuleNotFoundError: No module named 'scripts'"가 난다. Celery의 cwd_in_path()가
# 앱 로딩 시점에만 cwd를 sys.path에 임시로 넣었다가 빼기 때문에(celery/utils/imports.py),
# 이 모듈처럼 그 이후 지연 로딩되는 임포트는 그 혜택을 못 받는다. cwd 문자열 매칭에
# 의존하지 않도록, 이미 정상 해석된 app.core.config.ROOT를 이 시점에 직접 넣는다.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.detailed_sales_external_analysis import (
    load_pos_transactions,
    run_analysis,
)


CAFE_DELIVERY_TEMPERATURE_REFERENCE = {
    "id": "cafe_delivery_weather_2021_rainy_days",
    "label": "기온 변화(카페 배달 과거자료)",
    "sourceFiles": [
        "03카페배달매출데이터_2021.xlsx",
        "03날씨데이터_2021.xls",
    ],
    "sourcePeriod": "2021년",
    "population": "비가 온 날의 카페 배달 매출",
    "condition": "일강수량 > 0mm",
    "metric": "최저기온",
    "sampleSize": 101,
    "correlation": -0.312425,
    "testStatistic": -3.272399,
    "degreesOfFreedom": 99,
    "pValue": 0.001469,
    "confidenceInterval95": [-0.47864, -0.12459],
    "interpretation": (
        "2021년 비가 온 날의 카페 배달 자료에서는 최저기온이 낮을수록 "
        "배달 매출이 높아지는 약한 음의 상관관계가 관측됐습니다."
    ),
    "applicability": (
        "현재 매출에 배달 주문이 포함되고, 현재 기온과 매출의 변화 방향이 "
        "과거 관측 패턴과 일치할 때 가능한 원인 후보로만 사용합니다."
    ),
    "causal": False,
}

_TEMPERATURE_COLUMNS = {
    "temperature_c": "평균기온",
    "temperature_min_c": "최저기온",
    "temperature_max_c_monthly": "월 최고기온",
}


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


def _build_seoul_factors(
    file_bytes: bytes,
    trdar_cd: str,
) -> tuple[pd.DataFrame, list[str], bool | None]:
    transactions = load_pos_transactions(io.BytesIO(file_bytes))
    channel_column = next(
        (
            column
            for column in ("sales_channel", "order_channel", "판매채널", "주문채널")
            if column in transactions
        ),
        None,
    )
    is_delivery_sales = None
    if channel_column is not None:
        channel_values = (
            transactions[channel_column]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if not channel_values.empty:
            is_delivery_sales = channel_values.str.contains(
                "배달|delivery",
                regex=True,
            ).any()
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
            else:
                provider = (
                    str(recent_weather["provider"].dropna().iloc[0])
                    if "provider" in recent_weather
                    and recent_weather["provider"].notna().any()
                    else "외부"
                )
                limitations.append(
                    f"{provider} 서울 일별 날씨를 POS 일자 기준으로 결합했습니다.",
                )

    historical_weather_files = list(AGENT_DATA.glob("03*2021.xls"))
    if not daily_weather_matched and len(historical_weather_files) == 1:
        historical_weather = pd.read_excel(historical_weather_files[0])
        if historical_weather.shape[1] >= 7:
            historical_weather = historical_weather.iloc[:, [2, 3, 4, 5, 6]].copy()
            historical_weather.columns = [
                "date",
                "temperature_c",
                "temperature_min_c",
                "precipitation_mm",
                "wind_speed_ms",
            ]
            historical_weather["date"] = pd.to_datetime(
                historical_weather["date"],
                errors="coerce",
            )
            for column in (
                "temperature_c",
                "temperature_min_c",
                "precipitation_mm",
                "wind_speed_ms",
            ):
                historical_weather[column] = pd.to_numeric(
                    historical_weather[column],
                    errors="coerce",
                )
            dates = dates.merge(
                historical_weather,
                on="date",
                how="left",
                validate="one_to_one",
            )
            historical_columns = [
                "temperature_c",
                "temperature_min_c",
                "precipitation_mm",
                "wind_speed_ms",
            ]
            if dates[historical_columns].notna().any().any():
                daily_weather_matched = True
                limitations.append(
                    "2021년 서울 실측 날씨를 POS 일자 기준으로 결합했습니다.",
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
        if weather_columns and dates[weather_columns].notna().any().any():
            limitations.append("서울 날씨는 월 단위 값을 POS 일자에 확장해 사용했습니다.")
        elif not daily_weather_matched:
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
    return dates[["date", *factor_columns]], limitations, is_delivery_sales


def _attach_cafe_delivery_weather_reference(
    result: dict[str, Any],
    factors: pd.DataFrame,
    is_delivery_sales: bool | None = None,
) -> None:
    """Attach the historical study as a hypothesis, never as a causal estimate."""
    root_cause = result["rootCauseAnalysis"]
    root_cause["possibleExternalDrivers"] = []
    current_temperature_drivers = [
        driver
        for driver in root_cause.get("externalDrivers", [])
        if driver.get("factor") in _TEMPERATURE_COLUMNS
    ]
    if current_temperature_drivers:
        root_cause["evidencePolicy"]["currentWeather"] = (
            "현재 POS 기간의 매출과 같은 기간 날씨를 통제 회귀로 분석"
        )
        return

    reference = {
        **CAFE_DELIVERY_TEMPERATURE_REFERENCE,
        "sourceFiles": list(CAFE_DELIVERY_TEMPERATURE_REFERENCE["sourceFiles"]),
        "confidenceInterval95": list(
            CAFE_DELIVERY_TEMPERATURE_REFERENCE["confidenceInterval95"],
        ),
    }
    root_cause["referenceEvidence"] = [reference]
    root_cause["evidencePolicy"]["historicalReference"] = (
        "과거 별도 데이터의 상관관계는 현재 기간의 변화 방향이 일치할 때만 "
        "가능한 원인 후보로 제시하며 기여액이나 인과효과로 환산하지 않음"
    )

    internal = result.get("internalAnalysis", {})
    period = internal.get("period", {})
    summary = internal.get("summary", {})
    revenue_change = float(summary.get("revenueChange") or 0)
    if revenue_change == 0 or "date" not in factors:
        return

    temperature_column = next(
        (
            column
            for column in _TEMPERATURE_COLUMNS
            if column in factors and factors[column].notna().any()
        ),
        None,
    )
    required_period_keys = {
        "baselineStart",
        "baselineEnd",
        "currentStart",
        "currentEnd",
    }
    if temperature_column is None or not required_period_keys.issubset(period):
        return

    dated_factors = factors.copy()
    dated_factors["date"] = pd.to_datetime(dated_factors["date"], errors="coerce")
    baseline = dated_factors.loc[
        dated_factors["date"].between(period["baselineStart"], period["baselineEnd"]),
        temperature_column,
    ]
    current = dated_factors.loc[
        dated_factors["date"].between(period["currentStart"], period["currentEnd"]),
        temperature_column,
    ]
    baseline_value = pd.to_numeric(baseline, errors="coerce").mean()
    current_value = pd.to_numeric(current, errors="coerce").mean()
    if pd.isna(baseline_value) or pd.isna(current_value):
        return

    temperature_change = float(current_value - baseline_value)
    aligned = (
        (revenue_change > 0 and temperature_change < 0)
        or (revenue_change < 0 and temperature_change > 0)
    )
    if not aligned:
        return

    revenue_direction = "증가" if revenue_change > 0 else "감소"
    temperature_direction = "낮아졌고" if temperature_change < 0 else "높아졌고"
    factor_label = _TEMPERATURE_COLUMNS[temperature_column]
    channel_interpretation = (
        "현재 업로드 매출에 배달 채널이 포함되어 기온 변화를 가능한 외부 "
        "원인 후보로 제시합니다."
        if is_delivery_sales
        else (
            "배달 주문이 포함된 매출이라면 기온 변화가 가능한 외부 원인일 수 "
            "있습니다."
        )
    )
    candidate = {
        "factor": "historical_cafe_delivery_temperature",
        "label": reference["label"],
        "evidenceType": "historical_reference_association",
        "confidence": "hypothesis",
        "direction": "positive" if revenue_change > 0 else "negative",
        "baselineValue": round(float(baseline_value), 3),
        "currentValue": round(float(current_value), 3),
        "changeValue": round(temperature_change, 3),
        "effectUnit": "°C",
        "matchedCurrentMetric": factor_label,
        "referenceId": reference["id"],
        "interpretation": (
            f"현재 {factor_label}과 매출의 변화 방향이 과거 카페 배달 자료에서 "
            f"관측된 방향과 일치합니다. {channel_interpretation}"
        ),
        "causal": False,
    }
    root_cause["possibleExternalDrivers"].append(candidate)
    channel_narrative = (
        "업로드 매출에 배달 채널이 포함된 것으로 확인됐으며,"
        if is_delivery_sales
        else "업로드 매출에 배달 주문이 포함되어 있다면,"
    )
    root_cause["narrative"] += (
        f" 현재 {factor_label}은 비교 기간보다 {abs(temperature_change):.1f}°C "
        f"{temperature_direction} 매출은 {revenue_direction}했습니다. "
        f"{channel_narrative} 2021년 카페 배달 자료에서 관측된 "
        "'낮은 기온-높은 배달 매출' 패턴과 방향이 같아 기온 변화가 가능한 외부 "
        "원인일 수 있습니다. 다만 이는 다른 기간 자료의 상관관계이므로 현재 "
        "매출의 직접적인 인과 원인으로 확정할 수 없습니다."
    )
    reference_limitation = (
        "카페 배달·기온 참고근거는 2021년 별도 자료의 상관관계이므로 현재 "
        "배달 매출의 인과효과로 확정할 수 없습니다."
        if is_delivery_sales
        else (
            "카페 배달·기온 참고근거는 2021년 별도 자료의 상관관계이며, 현재 "
            "업로드 데이터에는 배달 채널 구분이 없어 적용 여부를 추가 확인해야 합니다."
        )
    )
    root_cause["limitations"] = list(dict.fromkeys([
        *root_cause.get("limitations", []),
        reference_limitation,
    ]))


def analyze_uploaded_sales(file_bytes: bytes, trdar_cd: str) -> dict[str, Any]:
    factors, limitations, is_delivery_sales = _build_seoul_factors(
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
    root_cause = result["rootCauseAnalysis"]
    root_cause["limitations"] = list(dict.fromkeys(result["dataQuality"]["warnings"]))
    root_cause["narrative"] += " " + " ".join(limitations)
    _attach_cafe_delivery_weather_reference(
        result,
        factors,
        is_delivery_sales=is_delivery_sales,
    )
    return result
