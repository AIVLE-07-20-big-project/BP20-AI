"""
매출 분석 — 상권 x 업종 진단

모델 예측에 의존하지 않는다. 전부 관측값 기반.
(LightGBM 잔차는 전체의 71%가 ±20% 밖으로 빗나가 판정 근거로 쓸 수 없었다)

의존성
    필수: pandas, numpy
    선택: lifelines  (없으면 위험도 블록만 빠지고 나머지는 그대로 동작)

사용
    python sales_analysis.py build                              # 패널 구축
    python sales_analysis.py build-neighbors                    # 인접상권 비교 피처 구축
    python sales_analysis.py fit-risk                           # Cox 위험도 적합
    python sales_analysis.py diagnose 3001491 CS100003 [20261]  # 진단 (기준분기 선택)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"
MERGED = PROCESSED_DATA / "merged_sales_analysis.csv"
PANEL = AGENT_DATA / "trend_panel.csv"
AREA_COORDS = SOURCE_DATA / "area_coords.csv"
NEIGHBOR_FEATURES = AGENT_DATA / "neighbor_sales_quarterly.csv"
COX = MODEL / "cox_risk.pkl"

AMT, CO = "THSMON_SELNG_AMT", "THSMON_SELNG_CO"
MIN_CO = 30
PERIOD = 4
MIN_Q = 6
MIN_PEERS = 20
SPECIAL_SCALE_RATIO = 50

STRUCT = {"STOR_CO": "점포수", "FRC_STOR_CO": "프랜차이즈수",
          "OPBIZ_RT": "개업률", "CLSBIZ_RT": "폐업률", "TOT_FLPOP_CO": "유동인구",
          "TOT_REPOP_CO": "상주인구", "TOT_WRC_POPLTN_CO": "직장인구"}












FLPOP_THRESHOLD = 0.12
WORKPOP_THRESHOLD = 0.15
REPOP_THRESHOLD = 0.05
REPOP_FRESHNESS_EPS = 1e-6





NEIGHBOR_RADIUS_M = 1_000.0
MIN_NEIGHBORS = 3
AREA_DECLINE_THRESHOLD = 0.10


AXES = {
    "시간대": {"TMZON_00_06_SELNG_AMT": "00-06시", "TMZON_06_11_SELNG_AMT": "06-11시",
               "TMZON_11_14_SELNG_AMT": "11-14시", "TMZON_14_17_SELNG_AMT": "14-17시",
               "TMZON_17_21_SELNG_AMT": "17-21시", "TMZON_21_24_SELNG_AMT": "21-24시"},
    "요일": {"MON_SELNG_AMT": "월", "TUES_SELNG_AMT": "화", "WED_SELNG_AMT": "수",
             "THUR_SELNG_AMT": "목", "FRI_SELNG_AMT": "금",
             "SAT_SELNG_AMT": "토", "SUN_SELNG_AMT": "일"},
    "성별": {"ML_SELNG_AMT": "남성", "FML_SELNG_AMT": "여성"},
    "연령대": {"AGRDE_10_SELNG_AMT": "10대", "AGRDE_20_SELNG_AMT": "20대",
               "AGRDE_30_SELNG_AMT": "30대", "AGRDE_40_SELNG_AMT": "40대",
               "AGRDE_50_SELNG_AMT": "50대", "AGRDE_60_ABOVE_SELNG_AMT": "60대이상"},
}
Z_STRONG, Z_WEAK = 1.0, -1.0


ORDERED = {
    "시간대": ["00-06시", "06-11시", "11-14시", "14-17시", "17-21시", "21-24시"],
    "요일": ["월", "화", "수", "목", "금", "토", "일"],
    "연령대": ["10대", "20대", "30대", "40대", "50대", "60대이상"],
}
RATES = {"개업률", "폐업률"}



COVS = ["매출_하락률", "폐업률", "개업률", "유동인구_변화", "log_점포수"]


def build_panel(merged=MERGED, out=PANEL) -> pd.DataFrame:
    """merged가 경로면 CSV로 읽고, 이미 로드된 DataFrame이면 그대로 쓴다(FastAPI에서
    업로드 신규 행을 메모리상으로 합친 DataFrame을 디스크에 쓰지 않고 바로 넘기기 위함).
    out=None이면 디스크에 쓰지 않고 DataFrame만 반환한다.
    """
    df = pd.read_csv(merged) if isinstance(merged, (str, Path)) else merged.copy()
    df = df[df[CO] >= MIN_CO].sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"])

    for keys, name in [(["TRDAR_CD", "STDR_YYQU_CD"], "TRDAR_AMT"),
                       (["SVC_INDUTY_CD", "STDR_YYQU_CD"], "INDUTY_AMT")]:
        agg = df.groupby(keys, as_index=False)[AMT].sum().rename(columns={AMT: name})
        df = df.merge(agg, on=keys, how="left")

    axis_cols = [c for m in AXES.values() for c in m]
    cols = (["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD", "TRDAR_CD_NM",
             "TRDAR_SE_CD_NM", "SVC_INDUTY_CD_NM", AMT, CO,
             "TRDAR_AMT", "INDUTY_AMT"] + list(STRUCT) + axis_cols)
    panel = df[[c for c in cols if c in df.columns]]

    n = panel.groupby(["TRDAR_CD", "SVC_INDUTY_CD"]).size()
    if out is not None:
        out.parent.mkdir(exist_ok=True)
        panel.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"패널: {panel.shape} → {out}")
        print(f"셀 {len(n):,}개 / {MIN_Q}분기 이상 {(n >= MIN_Q).sum():,}개")
    return panel


def build_neighbor_features(panel_path=PANEL, area_path=AREA_COORDS,
                             out_path=NEIGHBOR_FEATURES) -> pd.DataFrame:
    """상권x분기별 '상권 전체(업종 무관) 매출' 전분기 대비 변화율과, 반경
    NEIGHBOR_RADIUS_M 안 이웃 상권들의 같은 변화율 중앙값을 계산한다.

    동일 상권유형(TRDAR_SE_CD_NM) 중앙값 비교와 달리 지리적으로 실제 가까운 상권만
    본다 — 지금까지 없던 축. 대상 상권 자신은 이웃에서 제외한다.
    """
    panel = pd.read_csv(panel_path, usecols=["TRDAR_CD", "STDR_YYQU_CD", "TRDAR_AMT"])
    area_amt = panel.drop_duplicates(["TRDAR_CD", "STDR_YYQU_CD"]).sort_values(
        ["TRDAR_CD", "STDR_YYQU_CD"]).reset_index(drop=True)

    group = area_amt.groupby("TRDAR_CD", sort=False)
    previous_q = group["STDR_YYQU_CD"].shift(1)
    exact = previous_q == area_amt["STDR_YYQU_CD"].map(lambda q: shift_quarter(q, -1))
    previous_amt = group["TRDAR_AMT"].shift(1).where(exact)
    area_amt["area_change"] = (area_amt["TRDAR_AMT"] - previous_amt) / previous_amt.replace(0, np.nan)

    area = pd.read_csv(area_path).dropna(subset=["TRDAR_CD", "XCNTS_VALUE", "YDNTS_VALUE"])
    area = area.drop_duplicates("TRDAR_CD").reset_index(drop=True)
    tree = cKDTree(area[["XCNTS_VALUE", "YDNTS_VALUE"]].to_numpy(dtype=float))
    neighbor_idx = tree.query_ball_point(
        area[["XCNTS_VALUE", "YDNTS_VALUE"]].to_numpy(dtype=float), r=NEIGHBOR_RADIUS_M)
    trdar_list = area["TRDAR_CD"].to_numpy()
    neighbor_map = {
        trdar_list[i]: [trdar_list[j] for j in idxs if j != i]
        for i, idxs in enumerate(neighbor_idx)
    }

    records = []
    for quarter, group_q in area_amt.groupby("STDR_YYQU_CD"):
        change_by_trdar = dict(zip(group_q["TRDAR_CD"], group_q["area_change"]))
        for trdar_cd, target_change in change_by_trdar.items():
            neighbor_changes = [
                change_by_trdar[n] for n in neighbor_map.get(trdar_cd, [])
                if n in change_by_trdar and pd.notna(change_by_trdar[n])
            ]
            records.append((
                int(trdar_cd), int(quarter),
                target_change if pd.notna(target_change) else None,
                float(np.median(neighbor_changes)) if neighbor_changes else None,
                len(neighbor_changes),
            ))

    result = pd.DataFrame(records, columns=[
        "TRDAR_CD", "STDR_YYQU_CD", "target_change", "neighbor_change", "neighbor_count",
    ])
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def slope(y) -> float:
    """Theil–Sen 기울기 (로그 스케일) → 분기당 변화율.

    ★ 계절조정을 하지 않는다. 관측이 9분기(2.25주기)뿐이라 계절성과 노이즈를
      분리할 수 없다. 실제로 계절조정을 시도하면, 매끄럽게 하락하던 2024년에
      2025년 노이즈에서 뽑은 가짜 계절지수가 적용되어 없던 지그재그가 생긴다.
      대신 이상치에 강건한 Theil–Sen(쌍별 기울기의 중앙값)을 쓴다.
    """
    ly = np.log1p(np.asarray(y, dtype=float))
    n = len(ly)
    if n < 2:
        return 0.0
    sl = [(ly[j] - ly[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return float(np.expm1(np.median(sl)))


def decline_ratio(y) -> float:
    """분기 전환 중 하락한 비율. 연속 하락보다 노이즈에 강건하다.
    (이태원 일식: 8번 중 7번 하락 = 0.875)"""
    d = np.diff(np.asarray(y, dtype=float))
    return round(float((d < 0).mean()), 3) if len(d) else 0.0


def consec_decline(y) -> int:
    """말단부터 연속 하락 구간 수. 노이즈에 약하므로 보조 지표로만 쓴다."""
    y = np.asarray(y, dtype=float)
    n = 0
    for i in range(len(y) - 1, 0, -1):
        if y[i] < y[i - 1]:
            n += 1
        else:
            break
    return n


def shift_quarter(yyqu_cd: int, offset: int) -> int:
    """YYYYQ 형식의 분기 코드를 offset만큼 이동한다."""
    year, quarter = divmod(int(yyqu_cd), 10)
    serial = year * PERIOD + quarter - 1 + offset
    return (serial // PERIOD) * 10 + serial % PERIOD + 1


def pct_change(s: pd.Series, periods: int):
    """행 위치가 아니라 실제 분기 코드를 기준으로 증감률을 계산한다."""
    if s.empty:
        return None
    current_q = int(s.index[-1])
    previous_q = shift_quarter(current_q, -periods)
    if previous_q not in s.index:
        return None
    a, b = s.loc[previous_q], s.loc[current_q]
    return round(float((b - a) / a), 4) if a else None


def classify_market_state(sales_change, store_change, per_store_change) -> str:
    """총매출·점포수·점포당매출의 장기 변화를 함께 판정한다."""
    values = (sales_change, store_change, per_store_change)
    if any(pd.isna(v) for v in values):
        return "판정불가"
    if sales_change <= -0.1 and store_change <= -0.1:
        return "복합침체" if per_store_change <= -0.1 else "시장축소"
    if per_store_change <= -0.1 and store_change >= 0.1:
        return "경쟁심화"
    if per_store_change <= -0.1 and abs(store_change) < 0.1:
        return "수요이탈"
    if sales_change > -0.1 and per_store_change > -0.1:
        return "정상"
    return "관찰필요"





MARKET_STATE_EXPLANATION = {
    "정상": "관측 기간 전체(첫 분기 대비 현재)로 보면 총매출과 점포당 매출 모두 -10% 이상 "
            "급감하지는 않았다는 뜻입니다. 최근 1~2개 분기의 하락세와는 별개의 장기 판정이므로, "
            "전분기·전년동기 대비 수치가 계속 나쁘면 별도로 확인해야 합니다.",
    "시장축소": "총매출과 점포수가 함께 10% 이상 줄었습니다. 상권 자체가 축소되는 신호일 수 "
                "있어, 프로모션만으로는 해결되지 않을 가능성이 있습니다.",
    "복합침체": "총매출·점포수·점포당매출이 모두 10% 이상 줄었습니다. 가장 강한 하락 신호로, "
                "구조적 요인 확인이 우선입니다.",
    "경쟁심화": "점포 수는 늘었는데 점포당 매출은 줄었습니다. 매장 수 증가와 매출 감소 시점이 "
                "겹치는지 확인해, 같은 상권 안 경쟁이 심해졌는지 살펴봐야 합니다.",
    "수요이탈": "점포 수는 유지됐지만 점포당 매출만 줄었습니다. 점포가 줄어서가 아니라 손님 "
                "자체가 줄었을 가능성이 있습니다.",
    "관찰필요": "뚜렷한 패턴으로 분류되지 않았습니다. 다른 지표(축 분해, 거래건수 등)와 함께 "
                "확인이 필요합니다.",
    "판정불가": "비교에 필요한 점포수·매출 데이터가 부족해 판정할 수 없습니다.",
}


def classify_traffic_source(repop_change, flpop_change, workpop_change=None) -> str:
    """상주인구(거주자)·유동인구(방문객 전체)·직장인구(근무자) 변화를 비교해 매출 하락의
    성격을 구분한다.

    세 변수 조합을 전부 나열하지 않는다(8가지 다 판정하면 오히려 해석이 흐려진다) —
    실제로 서로 다른 대응이 필요한 경우만 우선순위대로 판정하고 나머지는 판정불가로
    남긴다. workpop_change 를 안 주면(기존 호출부 호환) 상주인구·유동인구 2축 판정만
    한다.

    상주인구는 갱신 주기가 길어(연 1회 미만) 변화율이 정확히 0에 가까우면 "안정적"이
    아니라 "이번 구간엔 갱신이 없었다"는 뜻이다 — 그런 경우 거주자가 안정적이라고
    확신하는 판정(외부_유입_감소류)에는 쓰지 않는다. 직장인구는 그런 신선도 문제가
    상대적으로 덜하고 이미 STRUCT 주석에 별도 설명이 있어 여기서는 그대로 취급한다.

    관측된 동반 변화를 서술할 뿐이며 인과관계를 의미하지 않는다 — 다른 판정 함수들과
    동일한 원칙.
    """
    if pd.isna(repop_change) or pd.isna(flpop_change):
        return "판정불가"

    def _down(v, threshold):
        return v is not None and pd.notna(v) and v <= -threshold

    def _stable(v, threshold):
        return v is not None and pd.notna(v) and abs(v) < threshold

    repop_fresh = abs(repop_change) > REPOP_FRESHNESS_EPS
    repop_down = repop_fresh and _down(repop_change, REPOP_THRESHOLD)
    repop_confirmed_stable = repop_fresh and _stable(repop_change, REPOP_THRESHOLD)
    flpop_down, flpop_stable = _down(flpop_change, FLPOP_THRESHOLD), _stable(flpop_change, FLPOP_THRESHOLD)

    if workpop_change is not None and pd.notna(workpop_change):
        workpop_down = _down(workpop_change, WORKPOP_THRESHOLD)
        workpop_stable = _stable(workpop_change, WORKPOP_THRESHOLD)
        if repop_down and flpop_down and workpop_down:
            return "상권_전방위_축소"
        if repop_confirmed_stable and flpop_down and workpop_down:
            return "직장인구_이탈형_외부유입감소"


        if flpop_stable and workpop_down:
            return "직장인구_감소_선행신호"
        if repop_down and flpop_stable and workpop_stable:
            return "거주자_이탈"
        if repop_confirmed_stable and workpop_stable and flpop_down:
            return "외부_유입_감소"
        if repop_down and flpop_down and workpop_stable:
            return "상권_자체_축소"
        return "판정불가"

    if repop_down and flpop_down:
        return "상권_자체_축소"
    if repop_confirmed_stable and flpop_down:
        return "외부_유입_감소"
    if flpop_stable and repop_down:
        return "거주자_이탈"
    return "판정불가"


def classify_regional_pattern(target_change, neighbor_change, neighbor_count) -> str:
    """상권 전체(업종 무관) 매출의 전분기 대비 변화가 인접 상권들과 동조화되는지 판정.

    동일 상권유형(TRDAR_SE_CD_NM) 중앙값 비교와 달리 지리적으로 실제 가까운 상권만
    본다. 관측된 동반 변화이며 인과관계를 의미하지 않는다 — 다른 판정 함수들과 동일한
    원칙.
    """
    if (neighbor_count < MIN_NEIGHBORS or pd.isna(target_change) or pd.isna(neighbor_change)):
        return "판정불가"

    target_down = target_change <= -AREA_DECLINE_THRESHOLD
    neighbor_down = neighbor_change <= -AREA_DECLINE_THRESHOLD

    if target_down and neighbor_down:
        return "지역_동반_하락"
    if target_down and not neighbor_down:
        return "상권_고립형_하락"
    if not target_down and neighbor_down:
        return "상권_역행_호조"
    if target_change > 0 and neighbor_change > 0:
        return "지역_동반_호조"
    return "판정불가"


def _covariates(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["cell"] = df["TRDAR_CD"].astype(str) + "_" + df["SVC_INDUTY_CD"]
    df = df.sort_values(["cell", "STDR_YYQU_CD"])
    g = df.groupby("cell")

    df["t"] = g.cumcount()
    df["매출_하락률"] = -((df[AMT] - g[AMT].cummax()) / g[AMT].cummax())
    df["점포_하락률"] = -((df["STOR_CO"] - g["STOR_CO"].cummax())
                        / g["STOR_CO"].cummax().replace(0, np.nan))
    df["폐업률"] = df["CLSBIZ_RT"]
    df["개업률"] = df["OPBIZ_RT"]
    df["유동인구_변화"] = g["TOT_FLPOP_CO"].pct_change(fill_method=None).clip(-1, 1)
    df["log_점포수"] = np.log1p(df["STOR_CO"])
    return df


def fit_risk(panel=PANEL, out=COX, thr: float = 0.20):
    """이벤트 = 점포수 최고점 대비 -20% (실제 폐업으로 시장이 축소).
    산출 = 각 위험인자의 위험비(HR). 손으로 정한 임계값을 데이터로 대체한다."""
    from lifelines import CoxTimeVaryingFitter

    df = _covariates(pd.read_csv(panel))
    df = df[df.groupby("cell")["cell"].transform("size") >= MIN_Q]

    df["ev"] = df["점포_하락률"] >= thr
    ev_t = df[df["ev"]].groupby("cell")["t"].min().rename("ev_t")
    df = df.merge(ev_t, on="cell", how="left")
    df = df[df["ev_t"].isna() | (df["t"] <= df["ev_t"])]

    df["start"], df["stop"] = df["t"], df["t"] + 1
    df["event"] = (df["ev_t"].notna() & (df["t"] == df["ev_t"])).astype(int)

    sf = (df[["cell", "start", "stop", "event"] + COVS]
          .replace([np.inf, -np.inf], np.nan).dropna())
    sf = sf[sf.groupby("cell")["cell"].transform("size") >= 2]

    n_cell, n_ev = sf["cell"].nunique(), int(sf["event"].sum())
    print(f"이벤트: 점포수 -{thr:.0%} (최고점 대비)")
    print(f"셀 {n_cell:,} / 구간 {len(sf):,} / 이벤트 {n_ev:,} ({n_ev/n_cell:.1%})\n")

    m = CoxTimeVaryingFitter(penalizer=0.1)
    m.fit(sf, id_col="cell", event_col="event", start_col="start", stop_col="stop")

    s = m.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    s.columns = ["HR", "하한", "상한", "p"]
    print(s.sort_values("HR", ascending=False).round(3).to_string())
    print("\nHR > 1 : 값이 클수록 위험 증가 / HR < 1 : 보호 요인")

    sf = sf.assign(risk=m.predict_partial_hazard(sf[COVS]).values)
    cr = sf.groupby("cell").agg(risk=("risk", "mean"), event=("event", "max"))
    cr["q"] = pd.qcut(cr["risk"], 5, labels=False, duplicates="drop")
    t = cr.groupby("q")["event"].agg(["size", "mean"])
    t["mean"] = (t["mean"] * 100).round(1)
    print(f"\n위험 5분위별 실제 붕괴율(%)\n{t.to_string()}")

    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"model": m, "covs": COVS}, f)
    print(f"\n저장: {out}")
    print("⚠ 2024Q1 이전부터 쇠퇴 중이던 셀은 식별 불가(좌측절단) → 위험 과소추정 가능")


def load_risk(panel: pd.DataFrame):
    """(모델, 셀별 위험도 테이블). 실패 시 (None, None) → 위험도 블록 생략."""
    try:
        with open(COX, "rb") as f:
            b = pickle.load(f)
    except (FileNotFoundError, ImportError):
        return None, None

    m, covs = b["model"], b["covs"]
    X = (_covariates(panel).groupby("cell").tail(1).set_index("cell")[covs]
         .replace([np.inf, -np.inf], np.nan).dropna())
    X["risk"] = m.predict_partial_hazard(X[covs]).values
    return m, X


class Diagnoser:
    def __init__(self, panel=PANEL, neighbor_path=NEIGHBOR_FEATURES):


        p = pd.read_csv(panel) if isinstance(panel, (str, Path)) else panel
        self.p = p.sort_values("STDR_YYQU_CD")
        self.p["cell"] = self.p["TRDAR_CD"].astype(str) + "_" + self.p["SVC_INDUTY_CD"]
        self.neighbor = pd.read_csv(neighbor_path) if Path(neighbor_path).exists() else None




        self.axis_cols, shares = {}, {}
        for ax, m in AXES.items():
            have = [c for c in m if c in self.p.columns]
            if len(have) < 2:
                continue
            tot = self.p[have].sum(axis=1).replace(0, np.nan)
            for c in have:
                shares[f"{c}__s"] = self.p[c] / tot
            self.axis_cols[ax] = {f"{c}__s": m[c] for c in have}

        if shares:
            self.p = pd.concat([self.p, pd.DataFrame(shares, index=self.p.index)],
                               axis=1)
        self.model, self.risk = load_risk(self.p)

    def _axes(self, row) -> dict:
        """동업종 분포 대비 z-score. 강점(살릴 것)과 약점(메울 것)을 가른다.

        상위 3개씩만 남긴다. 목록이 길면 처방이 흐려진다 —
        "약한 축 8개를 다 살리세요"는 아무것도 하지 말라는 말과 같다.
        """
        out = {}
        peers = self.p[
            (self.p["STDR_YYQU_CD"] == row["STDR_YYQU_CD"])
            & (self.p["SVC_INDUTY_CD"] == row["SVC_INDUTY_CD"])
        ]
        basis = "동일 분기·동일 업종"
        if "TRDAR_SE_CD_NM" in self.p and pd.notna(row.get("TRDAR_SE_CD_NM")):
            narrow = peers[peers["TRDAR_SE_CD_NM"] == row["TRDAR_SE_CD_NM"]]
            if len(narrow) >= MIN_PEERS:
                peers = narrow
                basis += "·동일 상권유형"

        for ax, cs in self.axis_cols.items():
            stat = peers[list(cs)].agg(["mean", "std"])
            z = {}
            for c, label in cs.items():
                m, sd, v = stat.loc["mean", c], stat.loc["std", c], row[c]

                if pd.isna(v) or pd.isna(sd) or sd < 1e-4 or (m and sd / m < 0.01):
                    continue
                z[label] = round(float((v - m) / sd), 2)
            if not z:
                continue

            srt = sorted(z.items(), key=lambda x: -x[1])
            strong = [k for k, v in srt[:3] if v >= Z_STRONG]
            weak = [k for k, v in srt[::-1][:3] if v <= Z_WEAK]

            out[ax] = {
                "강점": strong,
                "약점": weak,
                "z": z,
                "비교기준": basis,
                "비교대상수": int(len(peers)),
                "저신뢰경고": bool(len(peers) < MIN_PEERS),
            }
        return out

    def diagnose(self, trdar_cd, svc_induty_cd, yyqu_cd=None) -> dict:
        cell = f"{int(trdar_cd)}_{svc_induty_cd}"
        c = self.p[self.p["cell"] == cell].sort_values("STDR_YYQU_CD")
        if c.empty:
            return {"error": f"데이터 없음: {cell}"}
        if yyqu_cd is not None:
            target_q = int(yyqu_cd)
            if target_q not in c["STDR_YYQU_CD"].values:
                return {"error": f"기준분기 데이터 없음: {cell}, {target_q}"}
            c = c[c["STDR_YYQU_CD"] <= target_q]

        s = c.set_index("STDR_YYQU_CD")[AMT]
        last = c.iloc[-1]
        y = s.to_numpy()

        quality_warnings = []
        required = [CO, "STOR_CO"]
        missing = [col for col in required if col not in c or pd.isna(last.get(col))]
        if missing:
            quality_warnings.append("거래건수 또는 점포수 누락")
        quarters = c["STDR_YYQU_CD"].astype(int).tolist()
        if len(quarters) < MIN_Q:
            quality_warnings.append(f"과거 관측 분기 {MIN_Q}개 미만")
        if any(quarters[i] != shift_quarter(quarters[i - 1], 1)
               for i in range(1, len(quarters))):
            quality_warnings.append("분기 불연속")

        quality_peers = self.p[
            (self.p["STDR_YYQU_CD"] == last["STDR_YYQU_CD"])
            & (self.p["SVC_INDUTY_CD"] == last["SVC_INDUTY_CD"])
            & (self.p["TRDAR_CD"] != last["TRDAR_CD"])
        ]




        if "TRDAR_SE_CD_NM" in self.p and pd.notna(last.get("TRDAR_SE_CD_NM")):
            narrow_peers = quality_peers[quality_peers["TRDAR_SE_CD_NM"] == last["TRDAR_SE_CD_NM"]]
            if len(narrow_peers) >= MIN_PEERS:
                quality_peers = narrow_peers
        peer_median = float(quality_peers[AMT].median()) if not quality_peers.empty else np.nan
        scale_ratio = float(last[AMT] / peer_median) if peer_median > 0 else None
        if len(quality_peers) < MIN_PEERS:
            quality_warnings.append(f"동종 비교집단 {MIN_PEERS}곳 미만")
        if scale_ratio is not None and scale_ratio >= SPECIAL_SCALE_RATIO:
            quality_warnings.append("동종 중앙값 대비 50배 이상인 특수 상권 또는 집계 단위 이상")
        special_text = f"{last.get('TRDAR_CD_NM', '')} {last.get('TRDAR_SE_CD_NM', '')}"
        if any(token in special_text for token in ("전통시장", "도매시장", "관광특구", "수산시장")):
            quality_warnings.append("특수 상권 유형으로 일반 상권 비교 불가")
        analysis_usable = not quality_warnings


        sev = {
            "전분기_대비": pct_change(s, 1),
            "전년동기_대비": pct_change(s, PERIOD),
            "최고분기": int(s.idxmax()),
            "최고점_대비": round(float((y[-1] - y.max()) / y.max()), 4),
            "하락_분기_비율": decline_ratio(y),
            "연속_하락_분기수": consec_decline(y),
            "관측_분기수": len(s),
        }
        rank_peers = self.p[
            (self.p["STDR_YYQU_CD"] == last["STDR_YYQU_CD"])
            & (self.p["SVC_INDUTY_CD"] == last["SVC_INDUTY_CD"])
        ]
        if "TRDAR_SE_CD_NM" in self.p:
            narrow = rank_peers[rank_peers["TRDAR_SE_CD_NM"] == last["TRDAR_SE_CD_NM"]]
            if len(narrow) >= MIN_PEERS:
                rank_peers = narrow
        ranks = rank_peers[AMT].rank(ascending=False)
        target_rank = ranks.loc[rank_peers["cell"] == cell]
        if analysis_usable and not target_rank.empty:
            sev["동업종_순위"] = f"{len(rank_peers)}곳 중 {int(target_rank.iloc[0])}위"


        g_me = slope(y)
        g_tr = slope(c["TRDAR_AMT"].to_numpy())
        g_in = slope(c["INDUTY_AMT"].to_numpy())
        benchmark = (g_tr + g_in) / 2
        relative = round(g_me - benchmark, 4)

        if g_me >= -0.01:
            v = "하락 아님"
        elif g_tr < -0.01 and g_in < -0.01:
            v = "상권과 업종이 함께 하락하는 패턴"
        elif g_tr < -0.01:
            v = "상권 하락과 동반되는 패턴"
        elif g_in < -0.01:
            v = "업종 하락과 동반되는 패턴"
        elif relative < -0.01:
            v = "동일 상권·업종 기준보다 상대적으로 부진"
        else:
            v = "뚜렷한 동반 요인을 식별하기 어려움"

        cause = {"내_추세": round(g_me, 4), "상권_추세": round(g_tr, 4),
                 "업종_추세": round(g_in, 4), "비교기준_추세": round(benchmark, 4),
                 "상대_추세_차이": relative, "판정": v,
                 "해석주의": "관측 추세의 동반 여부이며 인과관계를 의미하지 않음"}


        st = {}
        for col, name in STRUCT.items():
            if col not in c:
                continue
            v0, v1 = float(c[col].iloc[0]), float(c[col].iloc[-1])
            st[name] = ({"처음": round(v0, 1), "현재": round(v1, 1),
                         "변화_pp": round(v1 - v0, 1)}
                        if name in RATES else
                        {"처음": round(v0, 1), "현재": round(v1, 1),
                         "변화율": round((v1 - v0) / v0, 4) if v0 else None})



        repop_change = st.get("상주인구", {}).get("변화율")
        flpop_change = st.get("유동인구", {}).get("변화율")
        workpop_change = st.get("직장인구", {}).get("변화율")
        st["유동인구_원인"] = classify_traffic_source(repop_change, flpop_change, workpop_change)



        neighbor_row = None
        if self.neighbor is not None:
            match = self.neighbor[
                (self.neighbor["TRDAR_CD"] == int(trdar_cd))
                & (self.neighbor["STDR_YYQU_CD"] == int(last["STDR_YYQU_CD"]))
            ]
            neighbor_row = match.iloc[0] if not match.empty else None
        target_change = float(neighbor_row["target_change"]) if neighbor_row is not None and pd.notna(neighbor_row["target_change"]) else None
        area_neighbor_change = float(neighbor_row["neighbor_change"]) if neighbor_row is not None and pd.notna(neighbor_row["neighbor_change"]) else None
        neighbor_count = int(neighbor_row["neighbor_count"]) if neighbor_row is not None else 0
        st["인접상권_비교"] = {
            "대상_상권_추세": target_change, "인접상권_추세": area_neighbor_change,
            "인접상권_수": neighbor_count,
            "판정": classify_regional_pattern(target_change, area_neighbor_change, neighbor_count),
            "해석주의": "관측된 동반 변화이며 인과관계를 의미하지 않음",
        }

        state = None
        if "STOR_CO" in c:
            stor = c["STOR_CO"].astype(float)
            ps = (s.to_numpy() / stor.replace(0, np.nan).to_numpy())
            if stor.iloc[0] and not np.isnan(ps[0]) and not np.isnan(ps[-1]):
                d_st = (stor.iloc[-1] - stor.iloc[0]) / stor.iloc[0]
                d_ps = (ps[-1] - ps[0]) / ps[0]
                st["점포당_매출"] = {"처음": int(ps[0]), "현재": int(ps[-1]),
                                    "변화율": round(float(d_ps), 4)}
                d_sales = (y[-1] - y[0]) / y[0] if y[0] else np.nan
                state = classify_market_state(d_sales, d_st, d_ps)
        st["시장_상태"] = state
        st["시장_상태_해설"] = MARKET_STATE_EXPLANATION.get(state)


        if CO in c and c[CO].notna().all():
            transactions = c[CO].astype(float)
            ticket = c[AMT].astype(float) / transactions.replace(0, np.nan)
            st["거래건수"] = {
                "처음": int(transactions.iloc[0]), "현재": int(transactions.iloc[-1]),
                "변화율": round(float((transactions.iloc[-1] - transactions.iloc[0]) /
                                      transactions.iloc[0]), 4) if transactions.iloc[0] else None,
            }
            st["거래당_매출"] = {
                "처음": int(ticket.iloc[0]) if pd.notna(ticket.iloc[0]) else None,
                "현재": int(ticket.iloc[-1]) if pd.notna(ticket.iloc[-1]) else None,
                "변화율": round(float((ticket.iloc[-1] - ticket.iloc[0]) / ticket.iloc[0]), 4)
                if pd.notna(ticket.iloc[0]) and ticket.iloc[0] else None,
            }
            st["거래건수_해설"] = (
                "매출액 = 거래건수 × 거래당 매출입니다. 거래건수만 보면 매출 변화가 방문객 "
                "감소 때문인지 객단가 변화 때문인지 알 수 없으니, 거래당 매출과 함께 확인하세요."
            )


        axes = self._axes(last) if self.axis_cols and analysis_usable else {}


        rx = self._prescribe(state, sev, st, axes) if analysis_usable else {
            "등급": "분석_차단",
            "긴급도": "확인 필요",
            "방향": "일반 상권 비교와 자동 처방을 제공하지 않습니다.",
            "확인과제": quality_warnings,
        }

        if (analysis_usable and self.risk is not None and cell in self.risk.index
                and int(last["STDR_YYQU_CD"]) == int(self.p["STDR_YYQU_CD"].max())):
            rx.update(self._risk_block(cell))

        return {
            "대상": {"상권명": str(last["TRDAR_CD_NM"]),
                     "업종명": str(last["SVC_INDUTY_CD_NM"]),
                     "기준분기": int(last["STDR_YYQU_CD"])},
            "1_심각도": sev,
            "2_원인_분해": cause,
            "3_구조_변화": st,
            "4_축_분해": axes,
            "5_처방": rx,
            "6_신뢰도": {"거래건수": int(last[CO]),
                         "저신뢰경고": bool(last[CO] < 100),
                         "분석사용가능": analysis_usable,
                         "판정": "사용 가능" if analysis_usable else "분석 차단",
                         "차단사유": quality_warnings,
                         "동종비교대상수": int(len(quality_peers)),
                         "동종중앙값_대비배수": round(scale_ratio, 2) if scale_ratio is not None else None,
                         "해석원칙": "관측된 동반 변화이며 인과관계를 의미하지 않음"},
            "매출_추이": {int(q): int(v) for q, v in s.items()},
        }

    @staticmethod
    def _prescribe(state, sev, st, axes) -> dict:
        """관측 패턴으로 효과를 단정하지 않고 다음 확인 과제만 제시한다."""
        ax = axes or {}

        def g(a, k):
            return ax.get(a, {}).get(k, [])

        strong = {a: g(a, "강점") for a in ax if g(a, "강점")}
        weak = {a: g(a, "약점") for a in ax if g(a, "약점")}

        def fmt(d):
            return ", ".join(f"{a} {'·'.join(v)}" for a, v in d.items()) or "없음"

        core = {
            "정체": f"강점 = {fmt(strong)}",
            "강점": {a: v for a, v in strong.items()},
            "약점": {a: v for a, v in weak.items()},
        }

        severity_score = 0
        qoq = sev.get("전분기_대비")
        yoy = sev.get("전년동기_대비")
        peak = sev.get("최고점_대비")
        if qoq is not None:
            severity_score += 2 if qoq <= -0.2 else int(qoq <= -0.1)
        if yoy is not None:
            severity_score += 2 if yoy <= -0.2 else int(yoy <= -0.1)
        if peak is not None:
            severity_score += 2 if peak <= -0.3 else int(peak <= -0.15)
        severity_score += int((sev.get("하락_분기_비율") or 0) >= 0.7)
        urgency = "높음" if severity_score >= 5 else ("중간" if severity_score >= 2 else "낮음")
        core["하락_심각도점수"] = severity_score

        rec = []
        if strong:
            rec.append(f"높은 구성비({fmt(strong)})가 실제 절대 매출 증가인지, "
                       "다른 구간 감소에 따른 상대적 상승인지 확인")

        if state in {"시장축소", "복합침체"}:
            rec.append("점포 감소가 총매출 감소를 얼마나 설명하는지 원천 집계 단위 확인")
            rec.append("거래건수와 거래당 매출 감소를 분리해 점검")
            return {**core, "등급": "구조_전환", "긴급도": "높음",
                    "방향": "매출과 점포 수가 함께 감소한 관측 패턴입니다. 효과적인 대응은 별도 검증이 필요합니다.",
                    "확인과제": rec}

        if state == "경쟁심화":
            rec.append("점포 증가와 점포당 매출 감소의 시점이 일치하는지 확인")
            return {**core, "등급": "차별화", "긴급도": urgency,
                    "방향": "점포 증가와 점포당 매출 감소가 함께 관측됐습니다. 경쟁 심화 여부는 추가 확인이 필요합니다.",
                    "확인과제": rec}

        if state == "수요이탈":
            rec.append(f"낮은 구성비({fmt(weak)})가 과거보다 실제 감소했는지 확인")
            rec.append("유동인구 대비 거래건수 변화로 구매 전환 약화 여부 확인")
            return {**core, "등급": "고객_회복", "긴급도": urgency,
                    "방향": "점포 수는 유지되지만 점포당 매출이 감소한 패턴입니다. 고객 이탈을 원인으로 단정할 수 없습니다.",
                    "확인과제": rec}

        if state == "정상" and severity_score < 2:
            rec.append("현재 구성비 패턴이 다음 기간에도 유지되는지 관찰")
            return {**core, "등급": "강점_확대", "긴급도": "낮음",
                    "방향": "뚜렷한 하락은 관측되지 않았습니다.",
                    "확인과제": rec}

        return {**core, "등급": "관찰", "긴급도": urgency,
                "방향": f"하락 신호를 점검해야 합니다(하락 분기 비율 "
                        f"{sev.get('하락_분기_비율')}). 원인은 확정하지 말고 강점을 유지하며 확인하십시오.",
                "확인과제": rec or ["추가 기간의 실제 매출과 거래건수 확인"]}

    def _risk_block(self, cell) -> dict:
        t, row = self.risk, self.risk.loc[cell]
        ph = float(row["risk"])
        safer = float((t["risk"] < ph).mean() * 100)
        pos = (f"위험 상위 {100-safer:.0f}%" if safer >= 50
               else f"위험 하위 {safer:.0f}%")

        coef = self.model.params_
        contrib = ((row[COVS] - t[COVS].mean()) * coef).sort_values(ascending=False)
        return {
            "위험도": round(ph, 3),
            "위험_순위": f"{pos} (전체 {len(t):,}개 셀)",
            "위험_기여": {k: {"현재값": round(float(row[k]), 3),
                              "HR": round(float(np.exp(coef[k])), 3)}
                          for k, v in contrib.head(3).items() if v > 0},
        }



def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    if cmd == "build":
        build_panel()
    elif cmd == "build-neighbors":
        build_neighbor_features()
    elif cmd == "fit-risk":
        fit_risk()
    elif cmd == "diagnose" and len(args) in {3, 4}:
        d = Diagnoser()
        yyqu_cd = args[3] if len(args) == 4 else None
        print(json.dumps(d.diagnose(args[1], args[2], yyqu_cd), ensure_ascii=False, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
