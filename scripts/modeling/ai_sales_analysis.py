"""입력된 현재 매출을 진단하는 비지도 ML 분석기.

매출을 예측하지 않는다. 현재 매출과 함께 관측된 점포수, 거래건수,
유동인구 및 동종 상권 분포를 사용해 이상성과 구조적 특징만 분석한다.

사용:
    python ai_sales_analysis.py train
    python ai_sales_analysis.py analyze 3120153 CS300008 [20261] [매출액]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from scripts.modeling.sales_analysis import AMT, CO, MIN_CO, shift_quarter


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data" / "trend_panel.csv"
MODEL_PATH = ROOT / "model" / "ai_sales_model.pkl"
MIN_PEERS = 20

STRUCTURE_FEATURES = [
    "log_sales", "log_sales_per_store", "log_sales_per_transaction",
    "log_sales_per_traffic", "area_sales_share", "industry_sales_share",
]
CHANGE_FEATURES = ["sales_qoq", "sales_yoy", "store_qoq", "traffic_qoq"]
FEATURES = STRUCTURE_FEATURES + CHANGE_FEATURES
FEATURE_LABELS = {
    "log_sales": "매출 규모",
    "log_sales_per_store": "점포당 매출",
    "log_sales_per_transaction": "거래당 매출",
    "log_sales_per_traffic": "유동인구 대비 매출",
    "area_sales_share": "상권 내 업종 매출 비중",
    "industry_sales_share": "서울 업종 매출 비중",
    "sales_qoq": "전분기 매출 증감",
    "sales_yoy": "전년동기 매출 증감",
    "store_qoq": "점포수 증감",
    "traffic_qoq": "유동인구 증감",
}


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _safe_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    return ((current - previous) / previous.replace(0, np.nan)).clip(-3, 3)


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """각 분기 관측값만 사용해 분석용 효율·변화 특성을 만든다."""
    df = panel.copy().sort_values(["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD"])
    df = df[df[CO] >= MIN_CO].copy()
    group = df.groupby(["TRDAR_CD", "SVC_INDUTY_CD"], sort=False)
    quarter = df["STDR_YYQU_CD"]

    prev_q = group["STDR_YYQU_CD"].shift(1)
    exact_prev = prev_q == quarter.map(lambda q: shift_quarter(q, -1))
    year_q = group["STDR_YYQU_CD"].shift(4)
    exact_year = year_q == quarter.map(lambda q: shift_quarter(q, -4))

    prev_sales = group[AMT].shift(1).where(exact_prev)
    year_sales = group[AMT].shift(4).where(exact_year)
    prev_stores = group["STOR_CO"].shift(1).where(exact_prev)
    prev_traffic = group["TOT_FLPOP_CO"].shift(1).where(exact_prev)

    df["log_sales"] = np.log1p(df[AMT].clip(lower=0))
    df["log_sales_per_store"] = np.log1p(_safe_ratio(df[AMT], df["STOR_CO"]).clip(lower=0))
    df["log_sales_per_transaction"] = np.log1p(_safe_ratio(df[AMT], df[CO]).clip(lower=0))
    df["log_sales_per_traffic"] = np.log1p(_safe_ratio(df[AMT], df["TOT_FLPOP_CO"]).clip(lower=0))
    df["area_sales_share"] = _safe_ratio(df[AMT], df["TRDAR_AMT"]).clip(0, 1)
    df["industry_sales_share"] = _safe_ratio(df[AMT], df["INDUTY_AMT"]).clip(0, 1)
    df["sales_qoq"] = _safe_change(df[AMT], prev_sales)
    df["sales_yoy"] = _safe_change(df[AMT], year_sales)
    df["store_qoq"] = _safe_change(df["STOR_CO"], prev_stores)
    df["traffic_qoq"] = _safe_change(df["TOT_FLPOP_CO"], prev_traffic)
    return df


def _prepare_matrix(rows: pd.DataFrame, features=FEATURES, medians: dict | None = None):
    x = rows[features].replace([np.inf, -np.inf], np.nan)
    fill = medians or {c: float(x[c].median()) for c in features}
    return x.fillna(fill), fill


def _fit_detector(table: pd.DataFrame, features: list[str]) -> dict:
    x, medians = _prepare_matrix(table, features)
    scaler = RobustScaler(quantile_range=(10, 90)).fit(x)
    model = IsolationForest(
        n_estimators=300, max_samples=8192, contamination=0.03,
        random_state=42, n_jobs=-1,
    ).fit(scaler.transform(x))
    scores = -model.score_samples(scaler.transform(x))
    return {
        "features": features,
        "model": model,
        "scaler": scaler,
        "medians": medians,
        "threshold": float(np.quantile(scores, 0.97)),
        "bounds": {
            c: (float(x[c].quantile(0.01)), float(x[c].quantile(0.99)))
            for c in features
        },
        "training_scores": scores,
    }


def _score_detector(rows: pd.DataFrame, detector: dict) -> dict:
    features = detector["features"]
    x, _ = _prepare_matrix(rows, features, detector["medians"])
    score = float(-detector["model"].score_samples(detector["scaler"].transform(x))[0])
    percentile = float((detector["training_scores"] <= score).mean() * 100)
    extremes = [
        c for c in features
        if float(x[c].iloc[0]) < detector["bounds"][c][0]
        or float(x[c].iloc[0]) > detector["bounds"][c][1]
    ]
    return {
        "score": score,
        "percentile": percentile,
        "extremes": extremes,
        "is_anomaly": bool(score >= detector["threshold"] or len(extremes) >= 2),
    }


def train(panel_path=PANEL, model_path=MODEL_PATH) -> dict:
    table = make_features(pd.read_csv(panel_path))
    structure = _fit_detector(table, STRUCTURE_FEATURES)
    change = _fit_detector(table, CHANGE_FEATURES)
    x, _ = _prepare_matrix(table, CHANGE_FEATURES, change["medians"])
    scores, threshold, bounds = change["training_scores"], change["threshold"], change["bounds"]
    training_extreme_count = sum(
        ((x[c] < bounds[c][0]) | (x[c] > bounds[c][1])).astype(int)
        for c in CHANGE_FEATURES
    )
    training_change_extreme = (
        (x["sales_qoq"] < bounds["sales_qoq"][0])
        | (x["sales_qoq"] > bounds["sales_qoq"][1])
        | (x["sales_yoy"] < bounds["sales_yoy"][0])
        | (x["sales_yoy"] > bounds["sales_yoy"][1])
    )
    training_hybrid = (
        (scores >= threshold) | (training_extreme_count >= 2) | training_change_extreme
    )
    quarter_rates = {}
    for quarter, idx in table.groupby("STDR_YYQU_CD").groups.items():
        quarter_rates[int(quarter)] = round(float(training_hybrid[idx].mean()), 4)

    # 명백한 매출 충격을 주입해 탐지 민감도를 검증한다.
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(table), size=min(5000, len(table)), replace=False)
    sample = table.iloc[sample_idx].copy()
    detected = {}
    for label, multiplier in [("80%_급락", 0.2), ("5배_급증", 5.0)]:
        shocked = sample.copy()
        shocked[AMT] *= multiplier
        shocked = _recalculate_sales_features(shocked)
        for change_col in ("sales_qoq", "sales_yoy"):
            known = shocked[change_col].notna()
            shocked.loc[known, change_col] = (
                multiplier * (1 + shocked.loc[known, change_col]) - 1
            ).clip(-3, 3)
        shock_x, _ = _prepare_matrix(shocked, CHANGE_FEATURES, change["medians"])
        shock_scores = -change["model"].score_samples(change["scaler"].transform(shock_x))
        extreme_count = sum(
            ((shock_x[c] < bounds[c][0]) | (shock_x[c] > bounds[c][1])).astype(int)
            for c in CHANGE_FEATURES
        )
        change_extreme = (
            (shock_x["sales_qoq"] < bounds["sales_qoq"][0])
            | (shock_x["sales_qoq"] > bounds["sales_qoq"][1])
            | (shock_x["sales_yoy"] < bounds["sales_yoy"][0])
            | (shock_x["sales_yoy"] > bounds["sales_yoy"][1])
        )
        hybrid = (shock_scores >= threshold) | (extreme_count >= 2) | change_extreme
        detected[label] = round(float(hybrid.mean()), 4)

    summary = {
        "학습방식": "구조·변화 분리 Isolation Forest",
        "예측사용": False,
        "학습건수": int(len(table)),
        "이상판정기준": "Isolation Forest 상위 3% 또는 변화율 극단 구간",
        "분기별_이상판정률": quarter_rates,
        "인위적충격_탐지율": detected,
        "사용가능": bool(min(detected.values()) >= 0.7),
    }
    bundle = {
        "structure_detector": structure,
        "change_detector": change,
        "summary": summary,
    }
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    return summary


def _recalculate_sales_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["log_sales"] = np.log1p(rows[AMT].clip(lower=0))
    rows["log_sales_per_store"] = np.log1p(_safe_ratio(rows[AMT], rows["STOR_CO"]).clip(lower=0))
    rows["log_sales_per_transaction"] = np.log1p(_safe_ratio(rows[AMT], rows[CO]).clip(lower=0))
    rows["log_sales_per_traffic"] = np.log1p(_safe_ratio(rows[AMT], rows["TOT_FLPOP_CO"]).clip(lower=0))
    rows["area_sales_share"] = _safe_ratio(rows[AMT], rows["TRDAR_AMT"]).clip(0, 1)
    rows["industry_sales_share"] = _safe_ratio(rows[AMT], rows["INDUTY_AMT"]).clip(0, 1)
    return rows


def _robust_peer_analysis(row: pd.Series, peers: pd.DataFrame) -> list[dict]:
    result = []
    for feature in FEATURES:
        values = peers[feature].replace([np.inf, -np.inf], np.nan).dropna()
        value = row[feature]
        if pd.isna(value) or len(values) < MIN_PEERS:
            continue
        median = float(values.median())
        mad = float((values - median).abs().median())
        raw_z = 0.6745 * (float(value) - median) / mad if mad > 1e-9 else 0.0
        robust_z = float(np.clip(raw_z, -10, 10))
        percentile = float((values <= value).mean() * 100)
        result.append({
            "지표": FEATURE_LABELS[feature],
            "백분위": round(percentile, 1),
            "강건_z": round(float(robust_z), 2),
            "판정": "매우 높음" if robust_z >= 2.5 else (
                "매우 낮음" if robust_z <= -2.5 else "일반 범위"
            ),
        })
    return sorted(result, key=lambda item: abs(item["강건_z"]), reverse=True)


class AISalesAnalyzer:
    def __init__(self, panel_path=PANEL, model_path=MODEL_PATH):
        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)
        self.table = make_features(pd.read_csv(panel_path))

    def analyze(self, trdar_cd, svc_induty_cd, yyqu_cd=None, sales_value=None) -> dict:
        cell = self.table[
            (self.table["TRDAR_CD"] == int(trdar_cd))
            & (self.table["SVC_INDUTY_CD"] == svc_induty_cd)
        ].sort_values("STDR_YYQU_CD")
        if cell.empty:
            return {"error": "대상 데이터가 없습니다."}
        target_q = int(yyqu_cd or cell["STDR_YYQU_CD"].iloc[-1])
        selected = cell[cell["STDR_YYQU_CD"] == target_q].copy()
        if selected.empty:
            return {"error": f"기준분기 데이터가 없습니다: {target_q}"}
        if sales_value is not None:
            original_sales = float(selected[AMT].iloc[0])
            delta = float(sales_value) - original_sales
            selected[AMT] = float(sales_value)
            selected["TRDAR_AMT"] += delta
            selected["INDUTY_AMT"] += delta
            selected = _recalculate_sales_features(selected)
            prev = cell[cell["STDR_YYQU_CD"] == shift_quarter(target_q, -1)]
            year = cell[cell["STDR_YYQU_CD"] == shift_quarter(target_q, -4)]
            selected["sales_qoq"] = (
                (float(sales_value) - float(prev[AMT].iloc[0])) / float(prev[AMT].iloc[0])
                if not prev.empty and float(prev[AMT].iloc[0]) else np.nan
            )
            selected["sales_yoy"] = (
                (float(sales_value) - float(year[AMT].iloc[0])) / float(year[AMT].iloc[0])
                if not year.empty and float(year[AMT].iloc[0]) else np.nan
            )
        row = selected.iloc[0]

        peers = self.table[
            (self.table["STDR_YYQU_CD"] == target_q)
            & (self.table["SVC_INDUTY_CD"] == svc_induty_cd)
            & (self.table["TRDAR_CD"] != int(trdar_cd))
        ]
        basis = "동일 분기·동일 업종"
        narrow = peers[peers["TRDAR_SE_CD_NM"] == row["TRDAR_SE_CD_NM"]]
        if len(narrow) >= MIN_PEERS:
            peers = narrow
            basis += "·동일 상권유형"

        structure = _score_detector(selected, self.bundle["structure_detector"])
        change = _score_detector(selected, self.bundle["change_detector"])
        # 매출 변화율 자체가 극단이면 단일 변화 지표만으로도 변화 이상이다.
        sales_change_extreme = bool({"sales_qoq", "sales_yoy"}.intersection(change["extremes"]))
        change["is_anomaly"] = bool(change["is_anomaly"] or sales_change_extreme)
        observed_changes = [
            float(row[c]) for c in ("sales_qoq", "sales_yoy") if pd.notna(row[c])
        ]
        average_change = float(np.mean(observed_changes)) if observed_changes else 0.0
        direction = "하락" if average_change < -0.03 else ("상승" if average_change > 0.03 else "혼합·보합")
        peer_analysis = _robust_peer_analysis(row, peers)

        warnings = []
        if float(row[AMT]) <= 0:
            warnings.append("매출이 0 이하")
        if float(row[CO]) <= 0:
            warnings.append("거래건수가 0 이하")
        if float(row["STOR_CO"]) <= 0:
            warnings.append("점포수가 0 이하")
        peer_median = float(peers[AMT].median()) if not peers.empty else np.nan
        scale_ratio = float(row[AMT] / peer_median) if peer_median > 0 else None
        if scale_ratio is not None and scale_ratio >= 50:
            warnings.append("동종 중앙값의 50배 이상: 특수 상권 또는 집계 단위 확인 필요")
        special_text = f"{row.get('TRDAR_CD_NM', '')} {row.get('TRDAR_SE_CD_NM', '')}"
        if any(token in special_text for token in ("전통시장", "도매시장", "관광특구", "수산시장")):
            warnings.append("특수 상권 유형으로 일반 상권 비교 불가")
        peer_confidence = "보통" if len(peers) >= MIN_PEERS else "낮음"
        comparison_usable = bool(len(peers) >= MIN_PEERS and not warnings)

        return {
            "대상": {
                "상권명": str(row["TRDAR_CD_NM"]),
                "업종명": str(row["SVC_INDUTY_CD_NM"]),
                "기준분기": target_q,
                "입력매출": int(row[AMT]),
            },
            "모델검증": self.bundle["summary"],
            "구조_특이성": {
                "이상도점수": round(structure["score"], 4),
                "이상도_백분위": round(structure["percentile"], 1),
                "판정": "구조적으로 매우 특이" if structure["is_anomaly"] else "일반 구조",
                "극단지표": [FEATURE_LABELS[c] for c in structure["extremes"]],
            },
            "변화_이상": {
                "이상도점수": round(change["score"], 4),
                "이상도_백분위": round(change["percentile"], 1),
                "판정": "비정상 변화" if change["is_anomaly"] else "일반 변화 범위",
                "방향": direction,
                "극단지표": [FEATURE_LABELS[c] for c in change["extremes"]],
            },
            "데이터_품질": {
                "판정": "확인 필요" if warnings else "기본 검사 통과",
                "경고": warnings,
                "동종중앙값_대비배수": round(scale_ratio, 2) if scale_ratio is not None else None,
                "일반비교사용가능": comparison_usable,
            },
            "동종비교": {
                "비교기준": basis,
                "비교대상수": int(len(peers)),
                "신뢰도": peer_confidence,
                "지표분석": peer_analysis if comparison_usable else [],
                "차단사유": warnings if not comparison_usable else [],
            },
            "해석주의": "이상 탐지는 이례적인 패턴을 찾으며 매출 변화의 원인이나 미래 매출을 예측하지 않음",
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "train":
        print(json.dumps(train(), ensure_ascii=False, indent=2))
    elif args[0] == "analyze" and len(args) in {3, 4, 5}:
        quarter = args[3] if len(args) >= 4 else None
        sales = args[4] if len(args) == 5 else None
        result = AISalesAnalyzer().analyze(args[1], args[2], quarter, sales)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
