# 대상 경로(AI 레포): app/sales_target/district_metrics.py
#
# 상권(TRDAR_CD) 단위 공공데이터(추정매출, 유동인구 등)를 분기별로 집계해서
# "최신 분기 대비 직전 분기 성장률"을 계산한다. scoring.growth_scores()에 넣을 입력을 만드는 단계다.
#
# 컬럼명을 함수 인자로 받는 이유: SalesEstimateCollector/FootTrafficCollector가 실제로 내려주는
# 필드명(매출 금액 컬럼명 등)을 아직 라이브 호출로 확인 못 했다. 이름을 하드코딩하면 확인 전까지
# 이 모듈을 테스트도 못 하고 잘못 짤 위험도 있어서, 집계 로직 자체는 컬럼명과 무관하게 동작하도록
# 분리했다. 실제 파이프라인(pipeline.py)에서 호출할 때만 검증된 실제 컬럼명을 넘긴다.
#
# 과제정의서 산식은 "최근 3개월 평균 매출 - 이전 3개월 평균 매출"이지만, 서울시 상권분석서비스는
# 월별이 아니라 분기 단위로만 제공된다. 분기가 곧 약 3개월이라 "최신 분기 합계 vs 직전 분기 합계"로
# 그대로 대응시켰다 — 데이터 granularity 제약에 따른 실용적 해석이다.

from __future__ import annotations

import pandas as pd


def aggregate_district_quarterly(
    df: pd.DataFrame,
    trdar_col: str,
    quarter_col: str,
    value_col: str,
) -> pd.DataFrame:
    """업종별로 흩어진 원본 데이터를 상권(trdar_col) x 분기(quarter_col) 단위로 합산한다.

    예: 추정매출 데이터는 TRDAR_CD x SVC_INDUTY_CD x STDR_YYQU_CD 단위인데, 업종 구분 없이
    상권 전체 매출 합계를 보고 싶을 때 이 함수로 SVC_INDUTY_CD를 없애고(합산) 상권 단위로 좁힌다.
    """
    if df.empty:
        return pd.DataFrame(columns=[trdar_col, quarter_col, value_col])

    result = df.copy()
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce").fillna(0)
    return (
        result.groupby([trdar_col, quarter_col], as_index=False)[value_col]
        .sum()
    )


def latest_vs_previous_growth(
    district_quarterly_df: pd.DataFrame,
    trdar_col: str,
    quarter_col: str,
    value_col: str,
) -> pd.DataFrame:
    """상권별로 가장 최근 두 분기를 비교해 성장률을 계산한다.

    분기 코드(quarter_col)는 "20261"처럼 문자열/숫자 정렬로 시간 순서가 그대로 맞는 형식이어야 한다
    (scripts/collection/collectors.py의 all_quarter_codes()가 만드는 형식과 동일).

    반환: DataFrame[trdar_col, "growth_rate"]. 상권마다 데이터가 2개 분기 미만이거나 직전 분기
    합계가 0이면 growth_rate는 NaN이다(비교 기준이 없다는 뜻 — 0%와 구분하기 위해 0이 아니라 NaN).
    """
    if district_quarterly_df.empty:
        return pd.DataFrame(columns=[trdar_col, "growth_rate"])

    sorted_df = district_quarterly_df.sort_values([trdar_col, quarter_col])

    def _growth(group: pd.DataFrame) -> float:
        if len(group) < 2:
            return float("nan")
        previous, latest = group[value_col].iloc[-2], group[value_col].iloc[-1]
        if previous == 0:
            return float("nan")
        return (latest - previous) / previous

    growth_series = sorted_df.groupby(trdar_col).apply(_growth, include_groups=False)
    return growth_series.reset_index(name="growth_rate")


def latest_quarter_values(
    district_quarterly_df: pd.DataFrame,
    trdar_col: str,
    quarter_col: str,
    value_col: str,
) -> pd.DataFrame:
    """상권별 가장 최근 분기의 값만 뽑는다(유동인구처럼 성장률이 아니라 현재값이 필요한 지표용)."""
    if district_quarterly_df.empty:
        return pd.DataFrame(columns=[trdar_col, value_col])

    sorted_df = district_quarterly_df.sort_values([trdar_col, quarter_col])
    return sorted_df.groupby(trdar_col, as_index=False).tail(1)[[trdar_col, value_col]]
