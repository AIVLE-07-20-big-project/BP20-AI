"""날씨·문화행사 등 외부요인의 매출 연관성 분석.

분기 매출로 개별 행사의 인과효과를 확정하지 않는다. 상권별 행사 노출 변화와
매출 변화의 연관성을 추정하고, 데이터 품질·표본·위약 검정을 통과한 결과만
보고한다. 서울 단일 관측소 분기 날씨는 분기 효과와 분리되지 않아 자동 차단한다.

사용:
    python external_factor_analysis.py build-events
    python external_factor_analysis.py build-subway
    python external_factor_analysis.py fit
    python external_factor_analysis.py analyze 3120153 CS300008 [20261]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pyproj import Transformer
from scipy.spatial import cKDTree

from scripts.modeling.sales_analysis import AMT, shift_quarter


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"
PANEL = AGENT_DATA / "trend_panel.csv"
AREA_COORDS = SOURCE_DATA / "area_coords.csv"
EVENTS = SOURCE_DATA / "cultural_event.csv"
WEATHER = SOURCE_DATA / "weather_seoul_quarterly.csv"
EVENT_FEATURES = PROCESSED_DATA / "event_exposure_quarterly.csv"
ANCHOR_RAW = SOURCE_DATA / "big_store.csv"
ANCHOR_FEATURES = PROCESSED_DATA / "anchor_exposure_quarterly.csv"
SUBWAY_RAW_DIR = SOURCE_DATA / "subway_data"          # CARD_SUBWAY_MONTH_YYYYMM.csv (수동 확보, 일별 원본)
SUBWAY_STATIONS = SOURCE_DATA / "subway_stations.csv"  # 역 좌표(수집 단계 산출물)
SUBWAY_FEATURES = PROCESSED_DATA / "subway_exposure_quarterly.csv"
RESULT_PATH = MODEL / "external_factor_analysis.json"
RADIUS_M = 2_000.0
ANCHOR_RADIUS_M = 2_000.0  # 문화행사와 동일 반경 재사용 — 데이터 없이 임의로 다른 값을 고르지 않음
# "인접 상권 비교"(sales_analysis.NEIGHBOR_RADIUS_M) 때와 같은 판단 기준 재사용 —
# 500m는 역 0개 상권 38.1%로 너무 좁고 2,000m는 평균 11.4개로 흐려짐(실측).
SUBWAY_RADIUS_M = 1_000.0
MIN_PEERS = 20


def _quarter_code(ts: pd.Timestamp) -> int:
    return int(ts.year * 10 + (ts.month - 1) // 3 + 1)


def _quarter_bounds(code: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    year, quarter = divmod(int(code), 10)
    start = pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1)
    return start, start + pd.offsets.QuarterEnd(startingMonth=start.month + 2)


def _event_quarters(start: pd.Timestamp, end: pd.Timestamp):
    code = _quarter_code(start)
    end_code = _quarter_code(end)
    while code <= end_code:
        q_start, q_end = _quarter_bounds(code)
        overlap_start, overlap_end = max(start, q_start), min(end, q_end)
        if overlap_start <= overlap_end:
            yield code, int((overlap_end - overlap_start).days + 1)
        code = shift_quarter(code, 1)


def audit_weather(path=WEATHER) -> dict:
    if not Path(path).exists():
        return {"사용가능": False, "이유": "날씨 파일 없음"}
    weather = pd.read_csv(path)
    precip = pd.to_numeric(weather.get("precip_total_mm", weather.get("rn_day")), errors="coerce")
    negative_rain = int((precip < 0).sum())
    station_count = int(weather["stnid"].nunique()) if "stnid" in weather else 0
    # 단일 관측소도 서울 전체 시계열 요인으로는 사용할 수 있지만 상권별 차이는 설명하지 못한다.
    usable = negative_rain == 0 and len(weather) >= 12
    reasons = []
    if negative_rain:
        reasons.append(f"음수 강수량 {negative_rain}개")
    if station_count <= 1:
        reasons.append("서울 단일 관측소라 상권별 날씨 차이 식별 불가")
    if len(weather) < 12:
        reasons.append(f"분기 관측 {len(weather)}개로 시계열 부족")
    return {
        "사용가능": usable,
        "관측분기수": int(len(weather)),
        "관측소수": station_count,
        "음수강수량수": negative_rain,
        "이유": "; ".join(reasons) if reasons else "품질 검사 통과",
    }


def peer_exposure_percentile(trdar_cd, yyqu_cd, panel_path=PANEL, event_path=EVENT_FEATURES) -> dict:
    """동종 상권 대비 문화행사 노출도의 순수 서술적 비교(인과 주장 아님).

    행사는 업종과 무관하므로 비교 모집단은 같은 분기의 전체 상권이다.
    event_exposure_quarterly.csv 는 반경 내 행사가 있는 상권만 행이 있는 희소 파일이라,
    빠진 상권을 0으로 채우지 않으면 "노출이 있는 상권들끼리만" 비교하게 되어 백분위가
    왜곡된다 — 반드시 전체 상권 기준으로 0-fill 한 뒤 비교한다.
    """
    quarter = int(yyqu_cd)
    panel_cells = pd.read_csv(panel_path, usecols=["TRDAR_CD", "STDR_YYQU_CD"])
    peers = panel_cells.loc[panel_cells["STDR_YYQU_CD"] == quarter, ["TRDAR_CD"]].drop_duplicates()
    if peers.empty:
        return {"노출_백분위": None, "비교대상수": 0, "판정": "비교불가"}

    events = pd.read_csv(event_path)
    exposure = events.loc[events["STDR_YYQU_CD"] == quarter, ["TRDAR_CD", "event_exposure"]]
    merged = peers.merge(exposure, on="TRDAR_CD", how="left")
    merged["event_exposure"] = merged["event_exposure"].astype(float).fillna(0.0)

    n = len(merged)
    if n < MIN_PEERS:
        return {"노출_백분위": None, "비교대상수": n, "판정": "비교불가"}

    target_rows = merged.loc[merged["TRDAR_CD"] == int(trdar_cd), "event_exposure"]
    target_value = float(target_rows.iloc[0]) if not target_rows.empty else 0.0
    # 노출 0인 상권이 많아 동률이 큰 분포라, 단순 "<=" 비율로 백분위를 매기면 동률
    # 집단 전체가 부풀려진 높은 백분위를 받는다("<" 비율과 평균 내는 표준적인
    # 동률 보정 방식을 사용한다).
    values = merged["event_exposure"]
    less = float((values < target_value).mean())
    less_equal = float((values <= target_value).mean())
    percentile = (less + less_equal) / 2 * 100
    verdict = "낮음" if percentile <= 25 else ("높음" if percentile >= 75 else "보통")

    return {"노출_백분위": round(percentile, 1), "비교대상수": n, "판정": verdict}


def build_event_features(
    area_path=AREA_COORDS, event_path=EVENTS, out_path=EVENT_FEATURES,
) -> pd.DataFrame:
    area = pd.read_csv(area_path).dropna(subset=["TRDAR_CD", "XCNTS_VALUE", "YDNTS_VALUE"])
    events = pd.read_csv(event_path)
    for col in ["LAT", "LOT"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events["start"] = pd.to_datetime(events["STRTDATE"], errors="coerce").dt.normalize()
    events["end"] = pd.to_datetime(events["END_DATE"], errors="coerce").dt.normalize()
    events["end"] = events["end"].fillna(events["start"])
    events.loc[events["end"] < events["start"], "end"] = events["start"]
    events = events.dropna(subset=["LAT", "LOT", "start"])

    to_5181 = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    ex, ey = to_5181.transform(events["LOT"].to_numpy(), events["LAT"].to_numpy())
    tree = cKDTree(area[["XCNTS_VALUE", "YDNTS_VALUE"]].to_numpy(dtype=float))

    records = []
    for event_idx, event in events.reset_index(drop=True).iterrows():
        nearby = tree.query_ball_point([ex[event_idx], ey[event_idx]], r=RADIUS_M)
        if not nearby:
            continue
        points = area.iloc[nearby]
        distances = np.sqrt(
            (points["XCNTS_VALUE"].to_numpy() - ex[event_idx]) ** 2
            + (points["YDNTS_VALUE"].to_numpy() - ey[event_idx]) ** 2
        )
        for quarter, active_days in _event_quarters(event["start"], event["end"]):
            weights = active_days * np.exp(-distances / 1_000.0)
            for trdar_cd, distance, weight in zip(points["TRDAR_CD"], distances, weights):
                records.append((int(trdar_cd), quarter, 1, active_days, float(distance), float(weight)))

    columns = [
        "TRDAR_CD", "STDR_YYQU_CD", "event_count", "event_days",
        "nearest_event_m", "event_exposure",
    ]
    raw = pd.DataFrame(records, columns=columns)
    if raw.empty:
        result = pd.DataFrame(columns=columns)
    else:
        result = raw.groupby(["TRDAR_CD", "STDR_YYQU_CD"], as_index=False).agg(
            event_count=("event_count", "sum"),
            event_days=("event_days", "sum"),
            nearest_event_m=("nearest_event_m", "min"),
            event_exposure=("event_exposure", "sum"),
        )
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def build_anchor_events(
    raw_path=ANCHOR_RAW, area_path=AREA_COORDS, out_path=ANCHOR_FEATURES,
) -> pd.DataFrame:
    """대형 앵커시설(대규모점포) 개업/폐업 이벤트의 상권별 공간 노출.

    "준대규모점포"(GS THE FRESH 등 SSM 체인)는 유통산업발전법상 소상공인 경쟁자에
    가까워 앵커가 아니므로 제외한다. 이 함수는 앵커시설이 존재하는 상시 효과가 아니라
    "그 상권 반경 내에서 개업/폐업이 일어난 분기"만 포착한다(문화행사와 동일하게 단일
    시점 이벤트의 거리가중 노출로 다룸). X/Y 좌표는 area_coords.csv와 같은 좌표계라
    변환 없이 그대로 공간조인한다(실측 확인함).
    """
    raw = pd.read_csv(raw_path, dtype=str)
    anchors = raw[raw["JPSENM"].str.strip() == "대규모점포"].copy()
    anchors["X"] = pd.to_numeric(anchors["X"], errors="coerce")
    anchors["Y"] = pd.to_numeric(anchors["Y"], errors="coerce")
    anchors = anchors.dropna(subset=["X", "Y"])

    area = pd.read_csv(area_path).dropna(subset=["TRDAR_CD", "XCNTS_VALUE", "YDNTS_VALUE"])
    tree = cKDTree(area[["XCNTS_VALUE", "YDNTS_VALUE"]].to_numpy(dtype=float))

    def _events(subset: pd.DataFrame, date_col: str, kind: str) -> list[tuple]:
        dates = pd.to_datetime(subset[date_col].str.strip(), errors="coerce")
        rows = subset.loc[dates.notna()]
        dates = dates.dropna()
        records = []
        for (_, row), date in zip(rows.iterrows(), dates):
            nearby = tree.query_ball_point([row["X"], row["Y"]], r=ANCHOR_RADIUS_M)
            if not nearby:
                continue
            points = area.iloc[nearby]
            distances = np.sqrt(
                (points["XCNTS_VALUE"].to_numpy() - row["X"]) ** 2
                + (points["YDNTS_VALUE"].to_numpy() - row["Y"]) ** 2
            )
            quarter = _quarter_code(date)
            weights = np.exp(-distances / 1_000.0)
            for trdar_cd, distance, weight in zip(points["TRDAR_CD"], distances, weights):
                records.append((int(trdar_cd), quarter, kind, float(distance), float(weight)))
        return records

    # 휴업(재개업 가능)은 폐업으로 단정하지 않으므로 폐업 이벤트에서 제외한다.
    closed = anchors[anchors["TRDSTATENM"].str.strip().str.startswith("폐업")]
    records = _events(anchors, "APVPERMYMD", "open") + _events(closed, "DCBYMD", "close")

    columns = ["TRDAR_CD", "STDR_YYQU_CD", "kind", "dist_m", "weight"]
    out_columns = [
        "TRDAR_CD", "STDR_YYQU_CD",
        "anchor_open_count", "anchor_open_exposure", "anchor_open_nearest_m",
        "anchor_close_count", "anchor_close_exposure", "anchor_close_nearest_m",
    ]
    raw_df = pd.DataFrame(records, columns=columns)
    if raw_df.empty:
        result = pd.DataFrame(columns=out_columns)
    else:
        agg = raw_df.groupby(["TRDAR_CD", "STDR_YYQU_CD", "kind"], as_index=False).agg(
            count=("dist_m", "count"), exposure=("weight", "sum"), nearest_m=("dist_m", "min"),
        )
        wide = agg.pivot(index=["TRDAR_CD", "STDR_YYQU_CD"], columns="kind",
                          values=["count", "exposure", "nearest_m"])
        wide.columns = [f"anchor_{kind}_{metric}" for metric, kind in wide.columns]
        result = wide.reset_index()
        for col in ["anchor_open_count", "anchor_open_exposure", "anchor_close_count", "anchor_close_exposure"]:
            if col not in result.columns:
                result[col] = 0.0
            result[col] = result[col].fillna(0.0)
        for col in ["anchor_open_nearest_m", "anchor_close_nearest_m"]:
            if col not in result.columns:
                result[col] = np.nan
        result = result[out_columns]

    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def _load_subway_ridership(raw_dir=SUBWAY_RAW_DIR) -> pd.DataFrame:
    """CARD_SUBWAY_MONTH_YYYYMM.csv 전부 로드.

    ★ 이 파일들은 헤더가 6개 필드인데 데이터 행 끝에 빈 필드가 하나 더 있어(트레일링
    콤마), 기본 read_csv로 읽으면 pandas가 첫 컬럼을 인덱스로 착각해 전체 컬럼이
    밀린다(실측으로 확인함 — 역명 자리에 노선명이 들어가는 식). index_col=False로
    읽어야 정상이다.
    """
    frames = []
    for path in sorted(Path(raw_dir).glob("CARD_SUBWAY_MONTH_*.csv")):
        # 파일마다 인코딩이 다르다(실측: 대부분 utf-8이지만 일부는 cp949) — 순서대로 시도.
        for encoding in ("utf-8", "cp949", "utf-8-sig"):
            try:
                frames.append(pd.read_csv(path, index_col=False, encoding=encoding))
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"{path.name}: 지원하는 인코딩으로 읽지 못함")
    if not frames:
        return pd.DataFrame(columns=["사용일자", "노선명", "역명", "승차총승객수", "하차총승객수"])
    return pd.concat(frames, ignore_index=True)


def build_subway_exposure(raw_dir=SUBWAY_RAW_DIR, station_path=SUBWAY_STATIONS,
                           area_path=AREA_COORDS, out_path=SUBWAY_FEATURES) -> pd.DataFrame:
    """상권x분기별 지하철 승하차인원 노출(거리가중).

    라이브 승하차 API(CardSubwayStatsNew)는 최근 ~4개월치만 롤링 보관해 매출 패널
    (20241~20261)과 겹치지 않는다는 걸 실측으로 확인했다 — 대신 파일 탭의 월별
    벌크 CSV(일별 원본 그대로, 인증키 불필요)를 사용자가 수동으로 확보해
    data/subway_data/에 넣었다. 역명이 같으면 호선이 달라도 합산한다(환승역이
    호선별로 별도 행이라서). 좌표 매칭이 안 되는 역(실측 1.9%, 괄호 부기명 표기 차이
    등)은 강제로 보정하지 않고 노출 계산에서 자연히 빠진다.
    """
    out_columns = ["TRDAR_CD", "STDR_YYQU_CD", "subway_exposure",
                   "subway_station_count", "subway_nearest_m"]
    ridership = _load_subway_ridership(raw_dir)
    if ridership.empty or not Path(station_path).exists():
        result = pd.DataFrame(columns=out_columns)
        result.to_csv(out_path, index=False, encoding="utf-8-sig")
        return result

    ridership = ridership.copy()
    ridership["ridership"] = (
        pd.to_numeric(ridership["승차총승객수"], errors="coerce").fillna(0)
        + pd.to_numeric(ridership["하차총승객수"], errors="coerce").fillna(0)
    )
    dates = pd.to_datetime(ridership["사용일자"].astype(str), format="%Y%m%d", errors="coerce")
    ridership["STDR_YYQU_CD"] = [
        _quarter_code(d) if pd.notna(d) else None for d in dates
    ]
    ridership = ridership.dropna(subset=["STDR_YYQU_CD"])

    # 역명 기준(같은 이름 다른 호선은 합산) 분기별 총 승하차량
    station_quarter = ridership.groupby(["역명", "STDR_YYQU_CD"], as_index=False)["ridership"].sum()
    station_quarter = station_quarter.rename(columns={"역명": "stnKrNm"})

    stations = pd.read_csv(station_path)
    stations["convX"] = pd.to_numeric(stations["convX"], errors="coerce")
    stations["convY"] = pd.to_numeric(stations["convY"], errors="coerce")
    stations = stations.dropna(subset=["convX", "convY"]).drop_duplicates("stnKrNm")

    to_5181 = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
    sx, sy = to_5181.transform(stations["convX"].to_numpy(), stations["convY"].to_numpy())
    stations = stations.assign(sx=sx, sy=sy)

    merged = station_quarter.merge(stations[["stnKrNm", "sx", "sy"]], on="stnKrNm", how="inner")

    area = pd.read_csv(area_path).dropna(subset=["TRDAR_CD", "XCNTS_VALUE", "YDNTS_VALUE"])
    tree = cKDTree(area[["XCNTS_VALUE", "YDNTS_VALUE"]].to_numpy(dtype=float))

    records = []
    for _, row in merged.iterrows():
        nearby = tree.query_ball_point([row["sx"], row["sy"]], r=SUBWAY_RADIUS_M)
        if not nearby:
            continue
        points = area.iloc[nearby]
        distances = np.sqrt(
            (points["XCNTS_VALUE"].to_numpy() - row["sx"]) ** 2
            + (points["YDNTS_VALUE"].to_numpy() - row["sy"]) ** 2
        )
        weights = row["ridership"] * np.exp(-distances / 1_000.0)
        for trdar_cd, distance, weight in zip(points["TRDAR_CD"], distances, weights):
            records.append((int(trdar_cd), int(row["STDR_YYQU_CD"]), float(distance), float(weight)))

    raw_df = pd.DataFrame(records, columns=["TRDAR_CD", "STDR_YYQU_CD", "dist_m", "weight"])
    if raw_df.empty:
        result = pd.DataFrame(columns=out_columns)
    else:
        result = raw_df.groupby(["TRDAR_CD", "STDR_YYQU_CD"], as_index=False).agg(
            subway_exposure=("weight", "sum"),
            subway_station_count=("dist_m", "count"),
            subway_nearest_m=("dist_m", "min"),
        )
        result = result[out_columns]

    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def _regression_table(panel_path=PANEL, event_path=EVENT_FEATURES, anchor_path=ANCHOR_FEATURES,
                       subway_path=SUBWAY_FEATURES) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    events = pd.read_csv(event_path)
    df = panel.merge(events, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left")
    df["event_count"] = df["event_count"].fillna(0)
    df["event_days"] = df["event_days"].fillna(0)
    df["event_exposure"] = df["event_exposure"].fillna(0)

    if Path(anchor_path).exists():
        anchors = pd.read_csv(anchor_path)[
            ["TRDAR_CD", "STDR_YYQU_CD", "anchor_open_exposure", "anchor_close_exposure"]
        ]
    else:
        anchors = pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "anchor_open_exposure", "anchor_close_exposure"])
    df = df.merge(anchors, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left")
    df["anchor_open_exposure"] = df["anchor_open_exposure"].fillna(0)
    df["anchor_close_exposure"] = df["anchor_close_exposure"].fillna(0)

    if Path(subway_path).exists():
        subway = pd.read_csv(subway_path)[["TRDAR_CD", "STDR_YYQU_CD", "subway_exposure"]]
    else:
        subway = pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "subway_exposure"])
    df = df.merge(subway, on=["TRDAR_CD", "STDR_YYQU_CD"], how="left")
    df["subway_exposure"] = df["subway_exposure"].fillna(0)

    df = df.sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"])
    group = df.groupby(["TRDAR_CD", "SVC_INDUTY_CD"], sort=False)
    previous_q = group["STDR_YYQU_CD"].shift(1)
    exact = previous_q == df["STDR_YYQU_CD"].map(lambda q: shift_quarter(q, -1))

    exposure_sources = {"event_exposure", "anchor_open_exposure", "anchor_close_exposure", "subway_exposure"}
    for source, target in [
        (AMT, "d_log_sales"), ("STOR_CO", "d_log_stores"),
        ("TOT_FLPOP_CO", "d_log_traffic"), ("event_exposure", "d_event_exposure"),
        ("anchor_open_exposure", "d_anchor_open_exposure"),
        ("anchor_close_exposure", "d_anchor_close_exposure"),
        ("subway_exposure", "d_log_subway_exposure"),
    ]:
        previous = group[source].shift(1).where(exact)
        if source in exposure_sources:
            df[target] = np.log1p(df[source]) - np.log1p(previous)
        else:
            df[target] = np.log1p(df[source].clip(lower=0)) - np.log1p(previous.clip(lower=0))
    return df.dropna(subset=["d_log_sales", "d_event_exposure", "d_log_stores", "d_log_traffic"])


def _fit_ols_multi(data: pd.DataFrame, cols: list[str]):
    quarter_dummies = pd.get_dummies(
        data["STDR_YYQU_CD"].astype(str), prefix="q", drop_first=True, dtype=float
    )
    x = pd.concat([
        data[cols + ["d_log_stores", "d_log_traffic"]].astype(float).reset_index(drop=True),
        quarter_dummies.reset_index(drop=True),
    ], axis=1)
    x = sm.add_constant(x, has_constant="add")
    model = sm.OLS(data["d_log_sales"].to_numpy(), x).fit(
        cov_type="cluster", cov_kwds={"groups": data["TRDAR_CD"].to_numpy()}
    )
    return model


def _fit_ols(data: pd.DataFrame, event_col="d_event_exposure"):
    return _fit_ols_multi(data, [event_col])


def _fit_anchor(data: pd.DataFrame) -> dict:
    """개업/폐업 노출을 한 모델에 동시에 넣어 서로 교란 없이 각각의 계수를 추정한다.

    문화행사 회귀와 동일한 위약검정(미래로 1분기 shift)·부호안정성(분기별 leave-one-out)
    기준을 그대로 적용한다. 앵커시설의 상시 존재 효과(누적 재고형 신호)는 다루지
    않는다 — "개업/폐업이 일어난 그 분기"의 동반 변화만 검증한다.
    """
    cols = ["d_anchor_open_exposure", "d_anchor_close_exposure"]
    data = data.dropna(subset=cols).copy()
    label = {"d_anchor_open_exposure": "개업", "d_anchor_close_exposure": "폐업"}
    if len(data) < 1_000:
        return {
            label[c]: {"사용가능": False, "표본수": int(len(data)), "판정": "표본 부족으로 검증 불가"}
            for c in cols
        }

    model = _fit_ols_multi(data, cols)

    data = data.sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"]).copy()
    for c in cols:
        data[f"placebo_{c}"] = data.groupby(["TRDAR_CD", "SVC_INDUTY_CD"])[c].shift(-1)
    placebo_cols = [f"placebo_{c}" for c in cols]
    placebo_data = data.dropna(subset=placebo_cols)
    placebo_model = _fit_ols_multi(placebo_data, placebo_cols) if len(placebo_data) >= 1_000 else None

    signs = {c: [] for c in cols}
    for quarter in sorted(data["STDR_YYQU_CD"].unique()):
        subset = data[data["STDR_YYQU_CD"] != quarter]
        if len(subset) >= 1_000:
            m = _fit_ols_multi(subset, cols)
            for c in cols:
                signs[c].append(float(m.params[c]))

    result = {}
    for c in cols:
        coef = float(model.params[c])
        ci = model.conf_int().loc[c].tolist()
        p_value = float(model.pvalues[c])
        placebo_p = float(placebo_model.pvalues[f"placebo_{c}"]) if placebo_model is not None else None
        sign_stability = float(np.mean(np.sign(signs[c]) == np.sign(coef))) if signs[c] else 0.0
        usable = bool(p_value < 0.05 and (placebo_p is None or placebo_p >= 0.05) and sign_stability >= 0.7)
        result[label[c]] = {
            "사용가능": usable,
            "표본수": int(len(data)),
            "이벤트노출표본수": int((data[c] != 0).sum()),
            "노출변화계수": round(coef, 6),
            "95%신뢰구간": [round(float(v), 6) for v in ci],
            "p값": round(p_value, 6),
            "위약검정_p값": round(placebo_p, 6) if placebo_p is not None else None,
            "방향안정성": round(sign_stability, 3),
            "판정": (
                f"대형점포 {label[c]} 노출 변화와 매출 변화의 통계적 연관성 확인"
                if usable else "교란·불안정성 또는 유의성 기준 미통과"
            ),
        }
    return result


def _fit_subway(data: pd.DataFrame) -> dict:
    """지하철 승하차 노출 변화와 매출 변화의 연관성을 문화행사와 동일한 방식(위약검정
    + 부호안정성)으로 검증한다. 유동인구(TOT_FLPOP_CO)와 개념적으로 겹칠 수 있어
    d_log_traffic을 통제변수로 이미 포함한 상태에서 그 위에 남는 설명력만 본다.
    """
    col = "d_log_subway_exposure"
    data = data.dropna(subset=[col]).copy()
    if len(data) < 1_000:
        return {"사용가능": False, "표본수": int(len(data)), "판정": "표본 부족으로 검증 불가"}

    model = _fit_ols_multi(data, [col])
    coef = float(model.params[col])
    ci = model.conf_int().loc[col].tolist()
    p_value = float(model.pvalues[col])

    data = data.sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"]).copy()
    data["placebo_subway"] = data.groupby(["TRDAR_CD", "SVC_INDUTY_CD"])[col].shift(-1)
    placebo_data = data.dropna(subset=["placebo_subway"])
    placebo_model = _fit_ols_multi(placebo_data, ["placebo_subway"]) if len(placebo_data) >= 1_000 else None
    placebo_p = float(placebo_model.pvalues["placebo_subway"]) if placebo_model is not None else None

    signs = []
    for quarter in sorted(data["STDR_YYQU_CD"].unique()):
        subset = data[data["STDR_YYQU_CD"] != quarter]
        if len(subset) >= 1_000:
            signs.append(float(_fit_ols_multi(subset, [col]).params[col]))
    sign_stability = float(np.mean(np.sign(signs) == np.sign(coef))) if signs else 0.0
    exposed = int((data[col] != 0).sum())
    usable = bool(p_value < 0.05 and (placebo_p is None or placebo_p >= 0.05) and sign_stability >= 0.7)

    return {
        "사용가능": usable,
        "표본수": int(len(data)),
        "노출변화표본수": exposed,
        "노출변화계수": round(coef, 6),
        "95%신뢰구간": [round(float(v), 6) for v in ci],
        "p값": round(p_value, 6),
        "위약검정_p값": round(placebo_p, 6) if placebo_p is not None else None,
        "방향안정성": round(sign_stability, 3),
        "판정": (
            "지하철 승하차 노출 변화와 매출 변화의 통계적 연관성 확인(유동인구 통제 후)"
            if usable else "교란·불안정성 또는 유의성 기준 미통과"
        ),
    }


def fit(panel_path=PANEL, event_path=EVENT_FEATURES, anchor_path=ANCHOR_FEATURES,
        subway_path=SUBWAY_FEATURES, out_path=RESULT_PATH) -> dict:
    if not Path(event_path).exists():
        build_event_features(out_path=event_path)
    if not Path(anchor_path).exists():
        build_anchor_events(out_path=anchor_path)
    if not Path(subway_path).exists():
        build_subway_exposure(out_path=subway_path)
    data = _regression_table(panel_path, event_path, anchor_path, subway_path)
    model = _fit_ols(data)
    coef = float(model.params["d_event_exposure"])
    ci = model.conf_int().loc["d_event_exposure"].tolist()
    p_value = float(model.pvalues["d_event_exposure"])

    # 미래 행사 노출이 현재 매출과 연결되면 잔여 교란 가능성이 크다.
    data = data.sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"]).copy()
    data["placebo_event"] = data.groupby(["TRDAR_CD", "SVC_INDUTY_CD"])[
        "d_event_exposure"
    ].shift(-1)
    placebo_data = data.dropna(subset=["placebo_event"])
    placebo = _fit_ols(placebo_data, "placebo_event") if len(placebo_data) >= 1_000 else None
    placebo_p = float(placebo.pvalues["placebo_event"]) if placebo is not None else None

    signs = []
    for quarter in sorted(data["STDR_YYQU_CD"].unique()):
        subset = data[data["STDR_YYQU_CD"] != quarter]
        if len(subset) >= 1_000:
            signs.append(float(_fit_ols(subset).params["d_event_exposure"]))
    sign_stability = float(np.mean(np.sign(signs) == np.sign(coef))) if signs else 0.0
    exposed = int((data["event_exposure"] > 0).sum())
    usable = bool(p_value < 0.05 and (placebo_p is None or placebo_p >= 0.05) and sign_stability >= 0.7)

    result = {
        "데이터해상도": "분기",
        "인과추정": False,
        "문화행사": {
            "사용가능": usable,
            "표본수": int(len(data)),
            "행사노출표본수": exposed,
            "노출변화계수": round(coef, 6),
            "95%신뢰구간": [round(float(v), 6) for v in ci],
            "p값": round(p_value, 6),
            "위약검정_p값": round(placebo_p, 6) if placebo_p is not None else None,
            "방향안정성": round(sign_stability, 3),
            "판정": (
                "행사 노출 변화와 매출 변화의 통계적 연관성 확인"
                if usable else "교란·불안정성 또는 유의성 기준 미통과"
            ),
        },
        "대형점포": _fit_anchor(data),
        "지하철승하차": _fit_subway(data),
        "날씨": audit_weather(),
        "해석주의": (
            "분기 집계 관찰자료이므로 행사·날씨·대형점포 개폐업·지하철 승하차가 매출 변화의 "
            "원인이라고 단정할 수 없음. 대형점포 항목은 개업/폐업이 일어난 그 분기의 동반 "
            "변화만 검증하며, 이미 존재하는 앵커시설이 상시적으로 끌어오는 집객 효과는 다루지 "
            "않음. 지하철승하차는 유동인구와 개념적으로 겹칠 수 있어 유동인구를 통제한 뒤 "
            "남는 설명력만 봄"
        ),
    }
    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


class ExternalFactorAnalyzer:
    def __init__(self, result_path=RESULT_PATH, event_path=EVENT_FEATURES, anchor_path=ANCHOR_FEATURES,
                 subway_path=SUBWAY_FEATURES):
        self.result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        self.events = pd.read_csv(event_path)
        self.anchors = (
            pd.read_csv(anchor_path) if Path(anchor_path).exists()
            else pd.DataFrame(columns=[
                "TRDAR_CD", "STDR_YYQU_CD",
                "anchor_open_count", "anchor_open_exposure", "anchor_open_nearest_m",
                "anchor_close_count", "anchor_close_exposure", "anchor_close_nearest_m",
            ])
        )
        self.subway = (
            pd.read_csv(subway_path) if Path(subway_path).exists()
            else pd.DataFrame(columns=[
                "TRDAR_CD", "STDR_YYQU_CD", "subway_exposure",
                "subway_station_count", "subway_nearest_m",
            ])
        )

    def analyze(self, trdar_cd, yyqu_cd) -> dict:
        row = self.events[
            (self.events["TRDAR_CD"] == int(trdar_cd))
            & (self.events["STDR_YYQU_CD"] == int(yyqu_cd))
        ]
        exposure = {
            "주변행사수": int(row["event_count"].iloc[0]) if not row.empty else 0,
            "행사일수합": int(row["event_days"].iloc[0]) if not row.empty else 0,
            "최근접행사_m": round(float(row["nearest_event_m"].iloc[0]), 1) if not row.empty else None,
            "거리가중노출": round(float(row["event_exposure"].iloc[0]), 3) if not row.empty else 0.0,
        }
        peer = peer_exposure_percentile(trdar_cd, yyqu_cd)

        anchor_row = self.anchors[
            (self.anchors["TRDAR_CD"] == int(trdar_cd))
            & (self.anchors["STDR_YYQU_CD"] == int(yyqu_cd))
        ]
        anchor_exposure = {
            "인근개업수": int(anchor_row["anchor_open_count"].iloc[0]) if not anchor_row.empty else 0,
            "최근접개업_m": (
                round(float(anchor_row["anchor_open_nearest_m"].iloc[0]), 1)
                if not anchor_row.empty and pd.notna(anchor_row["anchor_open_nearest_m"].iloc[0]) else None
            ),
            "인근폐업수": int(anchor_row["anchor_close_count"].iloc[0]) if not anchor_row.empty else 0,
            "최근접폐업_m": (
                round(float(anchor_row["anchor_close_nearest_m"].iloc[0]), 1)
                if not anchor_row.empty and pd.notna(anchor_row["anchor_close_nearest_m"].iloc[0]) else None
            ),
        }
        subway_row = self.subway[
            (self.subway["TRDAR_CD"] == int(trdar_cd))
            & (self.subway["STDR_YYQU_CD"] == int(yyqu_cd))
        ]
        prev_row = self.subway[
            (self.subway["TRDAR_CD"] == int(trdar_cd))
            & (self.subway["STDR_YYQU_CD"] == shift_quarter(int(yyqu_cd), -1))
        ]
        change = None
        if not subway_row.empty and not prev_row.empty:
            current = float(subway_row["subway_exposure"].iloc[0])
            previous = float(prev_row["subway_exposure"].iloc[0])
            if previous:
                change = round((current - previous) / previous, 4)
        subway_exposure = {
            "인근역수": int(subway_row["subway_station_count"].iloc[0]) if not subway_row.empty else 0,
            "최근접역_m": (
                round(float(subway_row["subway_nearest_m"].iloc[0]), 1)
                if not subway_row.empty and pd.notna(subway_row["subway_nearest_m"].iloc[0]) else None
            ),
            "거리가중노출": (
                round(float(subway_row["subway_exposure"].iloc[0]), 1)
                if not subway_row.empty and pd.notna(subway_row["subway_exposure"].iloc[0]) else 0.0
            ),
            "노출_변화율": change,
        }
        return {
            **self.result,
            "대상분기_문화행사노출": exposure,
            "동종상권_대비_노출도": peer,
            "대상분기_대형점포_개폐업": anchor_exposure,
            "대상분기_지하철승하차노출": subway_exposure,
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args == ["build-events"]:
        print(build_event_features())
    elif args == ["build-anchor"]:
        print(build_anchor_events())
    elif args == ["build-subway"]:
        print(build_subway_exposure())
    elif args == ["fit"]:
        print(json.dumps(fit(), ensure_ascii=False, indent=2))
    elif len(args) in {3, 4} and args[0] == "analyze":
        quarter = int(args[3]) if len(args) == 4 else 20261
        print(json.dumps(ExternalFactorAnalyzer().analyze(args[1], quarter), ensure_ascii=False, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
