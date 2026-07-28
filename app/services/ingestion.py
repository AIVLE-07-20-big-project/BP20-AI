# 업로드된 신규 매출 원본 데이터를 기존 외부데이터와 합쳐 진단 가능한 패널로 만든다
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


# 업로드 파일이 sales_estimate.csv 스키마(필수 컬럼)와 맞지 않을 때
class IngestionSchemaError(Exception):
    pass


_base_merged_cache: pd.DataFrame | None = None


# 기존 merged_sales_analysis.csv 전체(무거운 객체) — 프로세스당 한 번만 로드해 캐싱
def get_base_merged() -> pd.DataFrame:

    global _base_merged_cache
    if _base_merged_cache is None:
        _base_merged_cache = pd.read_csv(MERGED_SALES_ANALYSIS)
    return _base_merged_cache


# 업로드 파일(csv)을 sales_estimate.csv와 같은 스키마의 DataFrame으로 읽는다
def read_upload(file_bytes: bytes) -> pd.DataFrame:

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise IngestionSchemaError(f"CSV 파싱 실패: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IngestionSchemaError(f"필수 컬럼 누락: {missing}")
    return df


# preprocessing.py와 동일한 merge 체인을 업로드 행에 한해서만 재현한다
def _merge_external(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

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


# 업로드 행을 외부데이터와 합친 뒤, 기존 patched merged 패널에 in-memory concat한다
def build_combined_panel(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:





    merged_new, warnings = _merge_external(new_rows)
    base = get_base_merged()

    key_tuples = set(map(tuple, merged_new[KEY_COLS].itertuples(index=False, name=None)))
    base_keys = list(base[KEY_COLS].itertuples(index=False, name=None))
    keep_mask = [t not in key_tuples for t in base_keys]
    base_filtered = base.loc[keep_mask]

    combined = pd.concat([base_filtered, merged_new], ignore_index=True, sort=False)
    return combined, warnings
