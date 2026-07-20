"""업로드된 신규 매출 원본 데이터를 기존 외부데이터와 합쳐 진단 가능한 패널로 만든다.

Diagnoser는 동종 상권 대비 z-score/percentile을 계산하므로 전체 과거 패널이 있어야만
진단이 성립한다(scripts/modeling/sales_analysis.py 참고) — 업로드 한 줄만으로는 진단 불가.
그래서 여기서는 업로드 행을 "그 자체로 완결된 데이터"가 아니라 기존 패널에 합류시켜야
하는 추가 행으로 취급한다.

merge 규칙은 scripts/processing/preprocessing.py와 동일(같은 키·suffix)하되, 그 스크립트는
data/ 아래 파일을 통째로 다시 읽어 디스크에 쓰는 배치 작업인 반면, 여기서는 업로드된 신규
행에 한해서만 같은 merge를 재현하고 결과를 메모리상으로만 합친다(디스크의
merged_sales_analysis.csv/trend_panel.csv는 건드리지 않음 — 동시성·데이터 오염 리스크 회피).
"""
from __future__ import annotations

import io

import pandas as pd

from app.core.config import (
    FOOT_TRAFFIC,
    MERGED_SALES_ANALYSIS,
    RESIDENT_POPULATION,
    STORE_STATS,
    WEATHER_QUARTERLY,
    WORKPLACE_POPULATION,
)

REQUIRED_COLUMNS = [
    "TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD",
    "THSMON_SELNG_AMT", "THSMON_SELNG_CO",
]
KEY_COLS = ["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"]


class IngestionSchemaError(Exception):
    """업로드 파일이 sales_estimate.csv 스키마(필수 컬럼)와 맞지 않을 때."""


_base_merged_cache: pd.DataFrame | None = None


def get_base_merged() -> pd.DataFrame:
    """기존 merged_sales_analysis.csv 전체(무거운 객체) — 프로세스당 한 번만 로드해 캐싱."""
    global _base_merged_cache
    if _base_merged_cache is None:
        _base_merged_cache = pd.read_csv(MERGED_SALES_ANALYSIS)
    return _base_merged_cache


def read_upload(file_bytes: bytes) -> pd.DataFrame:
    """업로드 파일(csv)을 sales_estimate.csv와 같은 스키마의 DataFrame으로 읽는다."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise IngestionSchemaError(f"CSV 파싱 실패: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionSchemaError(f"필수 컬럼 누락: {missing}")
    return df


def _merge_external(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """preprocessing.py와 동일한 merge 체인을 업로드 행에 한해서만 재현한다."""
    warnings: list[str] = []
    df = new_rows.copy()

    if STORE_STATS.exists():
        store = pd.read_csv(STORE_STATS)
        merged = df.merge(store, on=KEY_COLS, how="left", suffixes=("", "_store"))
        if merged["STOR_CO"].isna().any() if "STOR_CO" in merged.columns else True:
            warnings.append("일부 행에 대해 점포 통계(store_stats) 매칭 실패")
        df = merged
    else:
        warnings.append("store_stats.csv 없음 — 점포 통계 미반영")

    if FOOT_TRAFFIC.exists():
        foot = pd.read_csv(FOOT_TRAFFIC)
        merged = df.merge(foot, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_pop"))
        if merged["TOT_FLPOP_CO"].isna().any() if "TOT_FLPOP_CO" in merged.columns else True:
            warnings.append("일부 행에 대해 유동인구(foot_traffic) 매칭 실패")
        df = merged
    else:
        warnings.append("foot_traffic.csv 없음 — 유동인구 미반영")

    if RESIDENT_POPULATION.exists():
        repop = pd.read_csv(RESIDENT_POPULATION)
        df = df.merge(repop, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_repop"))

    if WORKPLACE_POPULATION.exists():
        workpop = pd.read_csv(WORKPLACE_POPULATION)
        df = df.merge(workpop, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left", suffixes=("", "_workpop"))

    if WEATHER_QUARTERLY.exists():
        weather = pd.read_csv(WEATHER_QUARTERLY)
        if "STDR_YYQU_CD" in weather.columns:
            weather_drop = [
                c for c in weather.columns
                if c.lower() in {"stn_id", "stn_ko", "stn_en", "info"} or weather[c].isna().all()
            ]
            weather = weather.drop(columns=weather_drop) if weather_drop else weather
            df = df.merge(weather, on="STDR_YYQU_CD", how="left", suffixes=("", "_weather"))

    return df, warnings


def build_combined_panel(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """업로드 행을 외부데이터와 합친 뒤, 기존 patched merged 패널에 in-memory concat한다.

    같은 (TRDAR_CD, SVC_INDUTY_CD, STDR_YYQU_CD) 조합이 base에 이미 있으면 업로드 행으로
    대체한다(재제출 시나리오 — 사용자가 최신 원본으로 다시 올린 것으로 취급).
    """
    merged_new, warnings = _merge_external(new_rows)
    base = get_base_merged()

    key_tuples = set(map(tuple, merged_new[KEY_COLS].itertuples(index=False, name=None)))
    base_keys = list(base[KEY_COLS].itertuples(index=False, name=None))
    keep_mask = [t not in key_tuples for t in base_keys]
    base_filtered = base.loc[keep_mask]

    combined = pd.concat([base_filtered, merged_new], ignore_index=True, sort=False)
    return combined, warnings
