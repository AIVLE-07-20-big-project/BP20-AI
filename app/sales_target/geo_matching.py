# 대상 경로(AI 레포): app/sales_target/geo_matching.py
#
# 후보 업장(상가업소 데이터, WGS84 위경도)을 상권코드(TRDAR_CD)에 매핑한다.
# 이게 필요한 이유: scoring.py의 상권 성장성/유동인구 점수는 "상권" 단위 공공데이터(추정매출,
# 길단위인구 등, TRDAR_CD 기준)에서 나오는데, 상가업소 데이터는 개별 업장의 위경도만 준다.
# 이 둘을 연결하려면 각 업장이 어느 상권에 속하는지부터 알아야 한다.
#
# 좌표계 검증(2026-07-29, 실제 API 호출로 확인):
#   상권영역 API(TbgisTrdarRelm)의 XCNTS_VALUE/YDNTS_VALUE는 EPSG:5181(Korea 2000 / 중부원점 2010),
#   단위는 미터다. 예) 황학동벼룩시장(TRDAR_CD=3110055) X=201642, Y=452260을 EPSG:5181 기준
#   WGS84로 변환하면 lon=127.01859, lat=37.56988가 나오는데, 실제 황학동 위치(약 lon 127.017,
#   lat 37.566)와 500m 이내로 일치한다. 대표좌표가 상권 "중심점"이라 랜드마크와 정확히 같은 지점은
#   아니지만, 같은 동네라는 건 확인됐다.
#
# 매칭 방식: 각 후보 업장에서 가장 가까운 상권 중심점을 하나 고른다(최근접 이웃). 상권마다 반경을
# 강제로 딱 자르지 않고, 대신 상권 면적(RELM_AR, m²)에서 근사 반경을 계산해 같이 내려준다 —
# 후보가 상권 반경보다 훨씬 멀리 있으면(예: 3배 이상) 그 매핑을 신뢰하지 않는 판단은 호출하는
# 쪽(pipeline)에서 하도록 열어뒀다.
#
# 성능: 상권이 서울 전체 기준 약 1,650개, 후보 업장은 수만 건 규모가 될 수 있어 순수 파이썬 이중
# 루프(O(n_후보 x n_상권))는 너무 느리다. scikit-learn의 BallTree(haversine)로 최근접 탐색을 한다
# (requirements.txt에 scikit-learn이 이미 있어 새 의존성 추가 없음).

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree

_TRDAR_CRS = "EPSG:5181"  # 상권영역 API 좌표계 (Korea 2000 / 중부원점 2010)
_WGS84_CRS = "EPSG:4326"
_EARTH_RADIUS_M = 6371000.0

_transformer = Transformer.from_crs(_TRDAR_CRS, _WGS84_CRS, always_xy=True)


def add_trdar_wgs84_columns(trdar_df: pd.DataFrame) -> pd.DataFrame:
    """TrdarBoundaryCollector(TbgisTrdarRelm) 결과에 lon/lat(WGS84), approx_radius_m 컬럼을 추가한다."""
    df = trdar_df.copy()
    if df.empty:
        df["lon"] = pd.Series(dtype=float)
        df["lat"] = pd.Series(dtype=float)
        df["approx_radius_m"] = pd.Series(dtype=float)
        return df

    x = df["XCNTS_VALUE"].astype(float).to_numpy()
    y = df["YDNTS_VALUE"].astype(float).to_numpy()
    lon, lat = _transformer.transform(x, y)
    df["lon"] = lon
    df["lat"] = lat
    # 원(circle)으로 근사한 반경. 실제 상권은 원이 아니지만, "이 거리 안이면 그럴듯하다"는
    # 대략적인 판단 기준으로만 쓴다.
    df["approx_radius_m"] = np.sqrt(df["RELM_AR"].astype(float) / math.pi)
    return df


def assign_trdar_cd(
    candidates: pd.DataFrame,
    trdar_df: pd.DataFrame,
    candidate_lon_col: str = "lon",
    candidate_lat_col: str = "lat",
) -> pd.DataFrame:
    """각 후보에 가장 가까운 상권코드(trdar_cd)와 그 상권 중심까지 거리(trdar_distance_m)를 붙인다.

    trdar_df는 add_trdar_wgs84_columns()를 거쳐 lon/lat/approx_radius_m 컬럼이 있어야 한다.
    좌표가 없는 후보는 trdar_cd/trdar_distance_m이 None으로 남는다.
    """
    result = candidates.copy()
    result["trdar_cd"] = None
    result["trdar_distance_m"] = np.nan
    result["trdar_approx_radius_m"] = np.nan

    trdar = trdar_df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    if trdar.empty or candidates.empty:
        return result

    cand_lonlat = candidates[[candidate_lon_col, candidate_lat_col]].apply(pd.to_numeric, errors="coerce")
    valid_mask = cand_lonlat.notna().all(axis=1).to_numpy()
    if not valid_mask.any():
        return result

    trdar_rad = np.radians(trdar[["lat", "lon"]].to_numpy(dtype=float))
    tree = BallTree(trdar_rad, metric="haversine")

    valid_rad = np.radians(cand_lonlat.loc[valid_mask, [candidate_lat_col, candidate_lon_col]].to_numpy(dtype=float))
    dist_rad, idx = tree.query(valid_rad, k=1)

    matched_cds = trdar["TRDAR_CD"].to_numpy()[idx[:, 0]]
    matched_radius = trdar["approx_radius_m"].to_numpy()[idx[:, 0]]
    matched_dist_m = dist_rad[:, 0] * _EARTH_RADIUS_M

    result.loc[valid_mask, "trdar_cd"] = matched_cds
    result.loc[valid_mask, "trdar_distance_m"] = matched_dist_m
    result.loc[valid_mask, "trdar_approx_radius_m"] = matched_radius

    return result
