"""
Synthetic Control — 대응방안 효과의 사전/사후 검증

실제 랜덤화 실험 로그 없이, 실제 매출 패널(trend_panel.csv)만으로 반사실 매출 추이를
합성한다. 가짜 캠페인 데이터를 새로 만들지 않는다.

trend_panel.csv에는 "어떤 셀이 언제 어떤 대응방안을 적용했는지" 기록이 없다 — 그래서 두
함수로 나눈다(2026-07-18 결정, 배경: docs/response_recommendation_agent_plan.md §5.2):

    counterfactual_baseline()  대상 셀의 "개입 없을 시" 반사실 매출 추이. 패널 데이터만
                               있으면 항상 계산 가능.
    measured_effect()          실제 대응방안 시행 사례(data/campaign_logs.csv)가 있어야
                               계산됨. 사례가 없으면(현재 상태) 판정불가를 반환한다 —
                               anchor_facility usable=False와 동일한 정직성 원칙.

의존성
    필수: pandas, numpy, scipy (전부 requirements.txt에 이미 있음 — 새 의존성 없음)

사용
    python -m scripts.response_strategy.synthetic_control baseline 3001491 CS100003 [20261]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
# 문서에 남아 있는 직접 실행(`python scripts/response_strategy/synthetic_control.py ...`)
# 호환. 패키지 실행(-m)에서는 프로젝트 루트가 이미 import 경로에 있다.
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.modeling.sales_analysis import AMT, MIN_PEERS, MIN_Q, PANEL, shift_quarter

DATA = ROOT / "data"
CAMPAIGN_LOGS = DATA / "campaign_logs.csv"

MIN_DONORS = 5              # 도너풀 최소 크기 — 이보다 적으면 가중치 최적화가 불안정
MAX_DONORS = 30             # 도너 수가 관측 분기수(보통 6~9개)보다 훨씬 크면 볼록조합
                            # 최적화가 과적합(가중치가 수백~수천 곳에 0.001씩 흩어짐)으로
                            # 흐른다 — 실제로 3001491/CS100001을 도너 1,400곳 그대로 돌리면
                            # RMSE 0.0008까지 내려가지만 "괜찮은 대역체를 찾은 것"이 아니라
                            # 그냥 평균을 정교하게 흉내낸 것이라 판정 의미가 없어진다. 그래서
                            # pre-period 로그매출 궤적과의 상관계수 상위 MAX_DONORS만 추린 뒤
                            # 최적화한다.
FIT_RMSE_THRESHOLD = 0.15   # pre-period 적합도 임계값(로그스케일 RMSE).
                            # 초기값 — 실행 후 실측 분포로 재확인 예정(다른 임계값들과 동일한 관례)
AGE_COLS = {
    "10대": "AGRDE_10_SELNG_AMT", "20대": "AGRDE_20_SELNG_AMT",
    "30대": "AGRDE_30_SELNG_AMT", "40대": "AGRDE_40_SELNG_AMT",
    "50대": "AGRDE_50_SELNG_AMT", "60대이상": "AGRDE_60_ABOVE_SELNG_AMT",
}


def _cell_series(panel: pd.DataFrame, trdar_cd: int, svc_induty_cd: str, col: str = AMT) -> pd.Series:
    c = panel[(panel["TRDAR_CD"] == trdar_cd) & (panel["SVC_INDUTY_CD"] == svc_induty_cd)]
    return c.sort_values("STDR_YYQU_CD").set_index("STDR_YYQU_CD")[col]


def build_donor_pool(panel: pd.DataFrame, trdar_cd: int, svc_induty_cd: str,
                      as_of_quarter: int) -> list[int]:
    """같은 업종 x 대상 셀 제외 x 처치 전 기간(과거~as_of_quarter) 데이터를 가진 상권 목록.

    Diagnoser의 MIN_PEERS 패턴과 같은 취지로, 같은 상권유형(TRDAR_SE_CD_NM)으로 좁혀도
    MIN_PEERS 이상 남으면 좁힌다 — 발달상권과 골목상권을 섞으면 스케일이 달라 적합도가
    나빠진다.
    """
    target_rows = panel[(panel["TRDAR_CD"] == trdar_cd) & (panel["SVC_INDUTY_CD"] == svc_induty_cd)]
    pool = panel[
        (panel["SVC_INDUTY_CD"] == svc_induty_cd)
        & (panel["TRDAR_CD"] != trdar_cd)
        & (panel["STDR_YYQU_CD"] <= as_of_quarter)
    ]
    trdar_se = (target_rows["TRDAR_SE_CD_NM"].iloc[0]
                if not target_rows.empty and "TRDAR_SE_CD_NM" in target_rows else None)
    if trdar_se is not None and pd.notna(trdar_se):
        narrow = pool[pool["TRDAR_SE_CD_NM"] == trdar_se]
        if narrow["TRDAR_CD"].nunique() >= MIN_PEERS:
            pool = narrow

    counts = pool.groupby("TRDAR_CD")["STDR_YYQU_CD"].nunique()
    return counts[counts >= MIN_Q].index.tolist()


def _restrict_to_most_similar(target: pd.Series, donors: pd.DataFrame, max_donors: int) -> pd.DataFrame:
    """도너 수가 max_donors보다 많으면, pre-period 로그매출 궤적과의 상관계수 상위
    max_donors만 남긴다(과적합 방지 — MAX_DONORS 주석 참고)."""
    if donors.shape[1] <= max_donors:
        return donors
    log_target = np.log1p(target.to_numpy(dtype=float))
    log_donors = np.log1p(donors.to_numpy(dtype=float))
    corr = np.array([
        np.corrcoef(log_target, log_donors[:, j])[0, 1] if np.std(log_donors[:, j]) > 0 else -1.0
        for j in range(log_donors.shape[1])
    ])
    top_idx = np.argsort(-corr)[:max_donors]
    return donors.iloc[:, top_idx]


def fit_weights(target: pd.Series, donors: pd.DataFrame) -> tuple[np.ndarray, float]:
    """convex combination(합=1, 비음수) 가중치를 pre-period 로그 RMSE 최소화로 구한다.

    로그 스케일을 쓰는 이유는 sales_analysis.slope()와 동일 — 상권 규모 편차가 커서 절대
    오차가 아니라 비율 오차를 맞춰야 큰 도너 하나가 최적화를 지배하지 않는다.
    """
    y = np.log1p(target.to_numpy(dtype=float))
    X = np.log1p(donors.to_numpy(dtype=float))
    n = X.shape[1]

    def loss(w):
        return float(np.mean((y - X @ w) ** 2))

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else w0
    rmse = float(np.sqrt(loss(w)))
    return w, rmse


def counterfactual_baseline(trdar_cd, svc_induty_cd, yyqu_cd=None, panel=PANEL) -> dict:
    """대상 셀의 '개입 없을 시' 다음 분기 예상 매출. 패널 데이터만 있으면 항상 계산된다.

    특정 대응방안의 효과가 아니다 — 실측 효과는 measured_effect()가 담당한다.
    """
    p = pd.read_csv(panel) if isinstance(panel, (str, Path)) else panel.copy()
    trdar_cd = int(trdar_cd)

    target = _cell_series(p, trdar_cd, svc_induty_cd)
    if target.empty:
        return {"판정": "판정불가", "사유": f"데이터 없음: {trdar_cd}/{svc_induty_cd}"}

    as_of = int(yyqu_cd) if yyqu_cd is not None else int(target.index.max())
    target = target[target.index <= as_of]
    if len(target) < MIN_Q:
        return {"판정": "판정불가", "사유": f"관측 분기 {MIN_Q}개 미만 ({len(target)}개)"}

    donor_ids = build_donor_pool(p, trdar_cd, svc_induty_cd, as_of)
    if len(donor_ids) < MIN_DONORS:
        return {"판정": "판정불가",
                "사유": f"도너풀 부족 (같은 업종 {len(donor_ids)}곳, 최소 {MIN_DONORS}곳 필요)"}

    donor_wide = (
        p[(p["SVC_INDUTY_CD"] == svc_induty_cd) & (p["TRDAR_CD"].isin(donor_ids))
          & (p["STDR_YYQU_CD"] <= as_of)]
        .pivot_table(index="STDR_YYQU_CD", columns="TRDAR_CD", values=AMT)
    )
    common_q = target.index.intersection(donor_wide.index)
    donor_wide = donor_wide.loc[common_q].dropna(axis=1)
    target_aligned = target.loc[common_q]
    if donor_wide.shape[1] < MIN_DONORS or len(common_q) < MIN_Q:
        return {"판정": "판정불가", "사유": "도너풀과 관측 분기가 겹치는 구간 부족"}

    donor_wide = _restrict_to_most_similar(target_aligned, donor_wide, MAX_DONORS)
    w, rmse = fit_weights(target_aligned, donor_wide)
    fit_ok = rmse <= FIT_RMSE_THRESHOLD
    synthetic_pre = np.expm1(np.log1p(donor_wide.to_numpy(dtype=float)) @ w)

    next_q = shift_quarter(as_of, 1)
    donor_next = (
        p[(p["SVC_INDUTY_CD"] == svc_induty_cd) & (p["TRDAR_CD"].isin(donor_wide.columns))
          & (p["STDR_YYQU_CD"] == next_q)]
        .set_index("TRDAR_CD")[AMT].reindex(donor_wide.columns)
    )
    next_baseline = (
        float(np.expm1(np.log1p(donor_next.to_numpy(dtype=float)) @ w))
        if donor_next.notna().all() else None
    )

    return {
        "판정": "양호" if fit_ok else "적합도 미달",
        "도너풀_크기": int(donor_wide.shape[1]),
        "처치전_적합도_RMSE_로그스케일": round(rmse, 4),
        "처치전_실측": [int(v) for v in target_aligned.to_numpy()],
        "처치전_반사실": [int(v) for v in synthetic_pre],
        "다음분기": next_q,
        "다음분기_반사실_예상매출": int(next_baseline) if next_baseline is not None else None,
        "가중치": {int(k): round(float(v), 4) for k, v in zip(donor_wide.columns, w) if v > 1e-3},
        "해석주의": "관측 패널만으로 만든 반사실 베이스라인이며, 특정 대응방안의 실측 효과가 아님",
    }


def segment_baseline(trdar_cd, svc_induty_cd, yyqu_cd=None, panel=PANEL) -> dict:
    """연령대별 매출 비중에 counterfactual_baseline()과 같은 가중치를 적용한 참고치.

    별도로 세그먼트마다 가중치를 다시 최적화하지 않는다 — 표본이 얇아 세그먼트별로 따로
    최적화하면 과적합 위험이 크다. "세그먼트별로도 베이스라인이 다르다"는 참고 정보이지
    세그먼트별 실측 효과가 아니다(실측 효과는 campaign_logs가 쌓여야 measured_effect()로 가능).
    """
    base = counterfactual_baseline(trdar_cd, svc_induty_cd, yyqu_cd, panel)
    if base.get("판정") == "판정불가":
        return base

    p = pd.read_csv(panel) if isinstance(panel, (str, Path)) else panel.copy()
    trdar_cd = int(trdar_cd)
    as_of = int(yyqu_cd) if yyqu_cd is not None else int(_cell_series(p, trdar_cd, svc_induty_cd).index.max())
    donor_ids = list(base["가중치"].keys())
    w = np.array(list(base["가중치"].values()))
    w = w / w.sum()  # 표시용으로 반올림된 가중치를 재정규화

    out = {}
    for label, col in AGE_COLS.items():
        target_seg = _cell_series(p, trdar_cd, svc_induty_cd, col=col)
        target_seg = target_seg[target_seg.index <= as_of]
        donor_seg = (
            p[(p["SVC_INDUTY_CD"] == svc_induty_cd) & (p["TRDAR_CD"].isin(donor_ids))
              & (p["STDR_YYQU_CD"] <= as_of)]
            .pivot_table(index="STDR_YYQU_CD", columns="TRDAR_CD", values=col)
        )
        common_q = target_seg.index.intersection(donor_seg.index)
        donor_seg = donor_seg.loc[common_q, donor_ids]
        if donor_seg.isna().any().any() or len(common_q) < MIN_Q:
            continue
        synthetic = np.expm1(np.log1p(donor_seg.to_numpy(dtype=float)) @ w)
        actual = target_seg.loc[common_q].to_numpy(dtype=float)
        out[label] = {
            "실측_최근": int(actual[-1]) if len(actual) else None,
            "반사실_최근": int(synthetic[-1]) if len(synthetic) else None,
        }
    return {**base, "세그먼트별_베이스라인": out,
            "해석주의_세그먼트": "동일 가중치를 세그먼트 매출에 적용한 참고치이며, 세그먼트별 실측 효과가 아님"}


MEASURED_EFFECT_MIN_RELIABLE_CASES = 5   # 이 이상 반사실 계산 가능해야 "사용가능"
_LOG_REQUIRED_COLS = {
    "action_id", "trdar_cd", "svc_induty_cd", "treatment_yyqu_cd", "executed", "revenue_after",
}


def measured_effect(trdar_cd, svc_induty_cd, action_id, campaign_logs=CAMPAIGN_LOGS,
                     panel=PANEL, n_bootstrap: int = 500, seed: int = 0) -> dict:
    """campaign_logs.csv에서 이 방안(action_id)의 실제 적용 사례를 찾아 사후 효과를 추정한다.

    각 사례의 효과는 (실측 revenue_after - counterfactual_baseline 반사실 예상매출)로
    계산한다 — revenue_before와의 나이브 전후 비교가 아니다(시장 전체 추세를 방안의 효과로
    착각하는 걸 막기 위해 기존 SCM 도너풀 로직을 그대로 재사용). 같은 상권(trdar_cd) 사례가
    있으면 그것만 쓰고, 없으면 같은 업종(svc_induty_cd) 전체 사례로 넓혀서(프랜차이즈 여러
    지점의 실측을 일반화) "적용범위"에 그 사실을 명시한다.

    사례가 없거나(현재 상태) 반사실을 계산할 수 있는 사례가 하나도 없으면 억지로 수치를
    만들지 않고 판정불가를 반환한다 — anchor_facility usable=False와 동일한 정직성 원칙.
    """
    path = Path(campaign_logs)
    if not path.exists():
        return {"판정": "판정불가", "사유": "실제 시행 사례 없음 (campaign_logs.csv 없음)",
                "실측_사례수": 0}

    logs = pd.read_csv(path)
    if not _LOG_REQUIRED_COLS.issubset(logs.columns):
        return {"판정": "판정불가", "사유": "실제 시행 사례 없음", "실측_사례수": 0}

    matched = logs[
        (logs["action_id"] == action_id)
        & (logs["svc_induty_cd"] == svc_induty_cd)
        & (logs["executed"].astype(bool))
        & logs["revenue_after"].notna()
    ]
    if matched.empty:
        return {"판정": "판정불가", "사유": "실제 시행 사례 없음", "실측_사례수": 0}

    same_cell = matched[matched["trdar_cd"].astype(str) == str(trdar_cd)]
    pool = same_cell if not same_cell.empty else matched
    적용범위 = "동일 상권 실측" if not same_cell.empty else "동일 업종 전체 실측(해당 상권 사례 없음)"

    p = pd.read_csv(panel) if isinstance(panel, (str, Path)) else panel.copy()

    effects, fit_flags = [], []
    for _, row in pool.iterrows():
        treatment_q = int(row["treatment_yyqu_cd"])
        as_of = shift_quarter(treatment_q, -1)
        baseline = counterfactual_baseline(int(row["trdar_cd"]), row["svc_induty_cd"], yyqu_cd=as_of, panel=p)
        if baseline.get("판정") == "판정불가" or baseline.get("다음분기") != treatment_q:
            continue
        counterfactual = baseline.get("다음분기_반사실_예상매출")
        if counterfactual is None or counterfactual == 0:
            continue
        effects.append((float(row["revenue_after"]) - counterfactual) / counterfactual)
        fit_flags.append(baseline["판정"] == "양호")

    if not effects:
        return {"판정": "판정불가",
                "사유": "매칭 사례는 있으나 반사실 계산 가능한 사례 없음(도너풀/적합도 부족)",
                "실측_사례수": int(len(pool)), "적용범위": 적용범위}

    effects_arr = np.asarray(effects)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(effects_arr, size=len(effects_arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    판정 = "사용가능" if len(effects) >= MEASURED_EFFECT_MIN_RELIABLE_CASES else "탐색적"
    return {
        "판정": 판정,
        "적용범위": 적용범위,
        "실측_사례수": int(len(pool)),
        "반사실_계산가능_사례수": len(effects),
        "효과율_평균": round(float(effects_arr.mean()), 4),
        "효과율_95%CI": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "적합도_양호_비율": round(float(np.mean(fit_flags)), 2),
        "해석주의": "counterfactual_baseline과 동일한 SCM 반사실 대비 실측 효과이며, "
                  "표본이 적으면(5건 미만) 탐색적으로만 참고할 것",
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if len(args) not in {3, 4} or args[0] != "baseline":
        print(__doc__)
        return
    yyqu_cd = args[3] if len(args) == 4 else None
    result = segment_baseline(args[1], args[2], yyqu_cd)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
