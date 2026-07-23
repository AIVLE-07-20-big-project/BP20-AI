# Survival(Cox 시변모형) 평가 보강 — 위험 5분위별 Kaplan-Meier 생존곡선 + C-index
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.sales_analysis import COVS, COX, MIN_Q, PANEL, _covariates

ROOT = Path(__file__).resolve().parents[2]
REPORT_OUT = ROOT / "model" / "survival_eval_report.json"
EVENT_THRESHOLD = 0.20
N_QUANTILES = 5

PH_LIMITATION = (
    "이 모형(CoxTimeVaryingFitter)은 시변 공변량을 쓰기 때문에 lifelines의 Schoenfeld 잔차 "
    "기반 비례위험 가정 검증(check_assumptions)을 지원하지 않는다 — 그 기능은 정적 CoxPHFitter "
    "전용이다. 정적 스냅샷으로 근사 검증하려면 별도 모형을 새로 적합해야 하므로 이번 평가 "
    "범위(기존 모형의 평가지표 보강)에서는 생략했다. C-index/KM은 참고용이며, 실제 배포 전 "
    "실제 파일럿·최신 분기 데이터로 재검증이 필요하다."
)


# fit_risk()와 동일한 이벤트/구간 구성 — 새 정의를 만들지 않고 그대로 재사용
def _event_frame(panel: Path = PANEL, thr: float = EVENT_THRESHOLD) -> pd.DataFrame:

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
    return sf[sf.groupby("cell")["cell"].transform("size") >= 2]


def evaluate(panel: Path = PANEL) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.utils import concordance_index

    with open(COX, "rb") as f:
        bundle = pickle.load(f)
    model, covs = bundle["model"], bundle["covs"]

    sf = _event_frame(panel)
    sf = sf.assign(risk=model.predict_partial_hazard(sf[covs]).values)

    cell_summary = sf.groupby("cell").agg(
        risk=("risk", "mean"), event=("event", "max"), duration=("stop", "max"))
    cell_summary["분위"] = pd.qcut(cell_summary["risk"], N_QUANTILES, labels=False,
                                  duplicates="drop") + 1

    c_index = float(concordance_index(cell_summary["duration"], -cell_summary["risk"],
                                       cell_summary["event"]))

    km_by_quantile = {}
    for q, g in cell_summary.groupby("분위"):
        kmf = KaplanMeierFitter()
        kmf.fit(g["duration"], event_observed=g["event"], label=f"위험분위_{int(q)}")
        surv = kmf.survival_function_.iloc[:, 0]
        km_by_quantile[f"위험분위_{int(q)}"] = {
            "셀수": int(len(g)), "이벤트수": int(g["event"].sum()),
            "최종_생존율": round(float(surv.iloc[-1]), 4),
            "곡선": [{"t": round(float(t), 2), "생존율": round(float(s), 4)}
                     for t, s in surv.items()],
        }

    report = {
        "해석주의": PH_LIMITATION,
        "표본": {"셀수": int(len(cell_summary)), "이벤트수": int(cell_summary["event"].sum())},
        "C_index": round(c_index, 4),
        "위험분위별_KM": km_by_quantile,
    }
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(PH_LIMITATION, "\n")
    print(f"표본: 셀 {report['표본']['셀수']:,} / 이벤트 {report['표본']['이벤트수']:,}")
    print(f"C-index: {report['C_index']} (0.5=무작위 판별, 1.0=완벽 판별)")
    for label, v in km_by_quantile.items():
        print(f"  {label}: 셀 {v['셀수']}, 이벤트 {v['이벤트수']}, 최종 생존율 {v['최종_생존율']}")
    print(f"\n리포트 저장: {REPORT_OUT}")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 2 and sys.argv[1] == "evaluate":
        evaluate()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
