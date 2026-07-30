# synthetic v2 로그 생성기(계획 §5단계) — BanditPolicy로 실제 샘플링해 v2 계약을 만족하는
# 합성 로그를 만든다. 데이터_출처="synthetic_v2"로 legacy 합성("synthetic")과 구분한다.
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.response import action_rules
from app.services.response.campaign_logs import CONTEXT_COLS, SCHEMA_COLUMNS
from app.services.response.context import CONTEXT_DIM, CONTEXT_SCHEMA_VERSION
from app.services.response.policy import BanditPolicy
from app.services.response.reward import (
    DEFAULT_REWARD_DENOMINATOR_FLOOR,
    REWARD_DEFINITION_VERSION,
    calculate_reward_v2,
)
from scripts.modeling.sales_analysis import AMT, MIN_Q, PANEL

from scripts.response_strategy.bandit import NeuralContextualBandit

# problem_type × action별 "참 효과"(reward v2 기대값) — 등급마다 최소 1개는 의도적으로 나쁜 arm
TRUE_EFFECTS: dict[str, dict[str, float]] = {
    "고객_회복": {
        "즉시할인": 0.08, "쿠폰발행": 0.05, "타임세일": 0.04,
        "세트메뉴 도입": 0.03, "사이드메뉴 추가": 0.02, "배달채널 확대": -0.02,
    },
    "차별화": {"매장 리뉴얼": -0.02, "신메뉴 출시": 0.04, "배달채널 확대": 0.02},
    "강점_확대": {"브랜드 SNS 캠페인": 0.03, "지역 제휴 마케팅": 0.02},
    "관찰": {"웰컴 프로모션": 0.02, "리뷰 관리 캠페인": 0.01},
}

REWARD_NOISE_STD = 0.05
CONTRIBUTION_BEFORE_MEAN = 5_000_000.0
CONTRIBUTION_BEFORE_STD = 1_500_000.0
VARIABLE_COST_RATIO = 0.4
CAMPAIGN_COST_RATIO = 0.02
MEASUREMENT_DAYS = 30
SAMPLES_PER_PROBLEM_TYPE = 400
EXPERIMENT_EPSILON = 0.30
CHAMPION_PRIOR_BIAS = 10.0
GENERATOR_VERSION = "synthetic-v2-generator-1"


def _sample_context(rng: np.random.Generator) -> np.ndarray:
    return np.array([
        np.clip(rng.normal(0, 0.15), -0.6, 0.6),
        np.clip(rng.normal(0, 0.15), -0.6, 0.6),
        np.clip(rng.normal(-0.2, 0.15), -0.8, 0.0),
        rng.uniform(0, 1),
        rng.uniform(0, 1),
        rng.choice([0.0, 0.5, 1.0]),
    ], dtype=np.float32)


# target reward를 만족하도록 net_sales/variable_cost/campaign_cost를 역산한다(variable cost는 매출의 고정 비율)
def _sample_costs(rng: np.random.Generator, target_reward: float,
                  baseline_sales: float | None = None) -> dict[str, float]:
    if baseline_sales is not None and baseline_sales > 0:
        # 합성 사후매출도 실제 분석 패널의 SCM 반사실 기준선에서 생성한다.
        net_sales_before = float(baseline_sales)
        contribution_before = net_sales_before * (1 - VARIABLE_COST_RATIO)
    else:
        contribution_before = abs(rng.normal(CONTRIBUTION_BEFORE_MEAN, CONTRIBUTION_BEFORE_STD))
        net_sales_before = contribution_before / (1 - VARIABLE_COST_RATIO)
    denominator = max(contribution_before, DEFAULT_REWARD_DENOMINATOR_FLOOR)
    contribution_after = contribution_before + target_reward * denominator

    variable_cost_before = net_sales_before * VARIABLE_COST_RATIO
    campaign_cost = net_sales_before * CAMPAIGN_COST_RATIO

    net_sales_after = (contribution_after + campaign_cost) / (1 - VARIABLE_COST_RATIO)
    variable_cost_after = net_sales_after * VARIABLE_COST_RATIO

    return {
        "net_sales_before": net_sales_before, "net_sales_after": net_sales_after,
        "variable_cost_before": variable_cost_before, "variable_cost_after": variable_cost_after,
        "campaign_cost": campaign_cost,
    }


def _load_target_cells() -> list[tuple[int, str, int, int]]:
    """실제 분석 패널과 조인 가능한 합성 캠페인 대상 셀을 만든다."""
    panel = pd.read_csv(PANEL, usecols=["TRDAR_CD", "SVC_INDUTY_CD", "STDR_YYQU_CD", AMT])
    cells: list[tuple[int, str, int, int]] = []
    for (trdar_cd, svc_induty_cd), group in panel.groupby(["TRDAR_CD", "SVC_INDUTY_CD"]):
        quarters = sorted(int(q) for q in group["STDR_YYQU_CD"].dropna().unique())
        # treatment 직전 기간에 SCM이 사용할 MIN_Q개 이상의 관측값이 있어야 한다.
        for index in range(MIN_Q, len(quarters)):
            treatment_q = quarters[index]
            as_of_q = quarters[index - 1]
            cells.append((int(trdar_cd), str(svc_induty_cd), as_of_q, treatment_q))
    if not cells:
        raise RuntimeError(f"실제 패널에서 합성 캠페인 대상 셀을 만들 수 없습니다: {PANEL}")
    return cells


def generate_rows(등급: str, arms: list[str], n: int, rng: np.random.Generator,
                  target_cells: list[tuple[int, str, int, int]],
                  coverage: list[tuple[tuple[int, str, int, int], str]],
                  panel: pd.DataFrame) -> list[dict]:
    effects = TRUE_EFFECTS[등급]
    bandit = NeuralContextualBandit(
        context_dim=CONTEXT_DIM, arms=arms, policy_version=GENERATOR_VERSION,
        model_version=GENERATOR_VERSION, context_schema_version=CONTEXT_SCHEMA_VERSION,
        reward_definition_version=REWARD_DEFINITION_VERSION,
    )
    policy = BanditPolicy(bandit=bandit, epsilon=EXPERIMENT_EPSILON)

    rows = []
    baseline_cache: dict[tuple[int, str, int], float | None] = {}
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        if i < len(coverage):
            (trdar_cd, svc_induty_cd, as_of_q, treatment_q), forced_action = coverage[i]
        else:
            trdar_cd, svc_induty_cd, as_of_q, treatment_q = target_cells[int(rng.integers(0, len(target_cells)))]
            forced_action = None
        context = _sample_context(rng)
        # 매 행마다 "믿는 최고 arm"을 랜덤하게 바꿔 action별 표본이 고르게 퍼지게 한다
        champion = rng.choice(arms)
        bandit._prior_bias = np.array([CHAMPION_PRIOR_BIAS if a == champion else 0.0 for a in arms])

        decision = policy.choose(context, arms, mode="experiment", rng=rng)
        action = forced_action or decision.selected_action
        # 업종×방안 커버리지 행은 균등 탐색 로그로 기록해 propensity와 action을 일치시킨다.
        propensity = 1.0 / len(arms) if forced_action else decision.propensity
        action_probabilities = (
            {arm: 1.0 / len(arms) for arm in arms}
            if forced_action else decision.action_probabilities
        )

        # 데모용 합성 로그에서는 모든 방안이 최소한의 양의 기준효과를 갖게 해
        # 추천 화면에 구조적으로 의미 없는 대폭 하락값이 나오지 않게 한다.
        target_reward = max(0.01, effects[action] + rng.normal(0, REWARD_NOISE_STD))
        cache_key = (trdar_cd, svc_induty_cd, as_of_q)
        if cache_key not in baseline_cache:
            next_rows = panel[
                (panel["TRDAR_CD"] == trdar_cd)
                & (panel["SVC_INDUTY_CD"] == svc_induty_cd)
                & (panel["STDR_YYQU_CD"] == treatment_q)
            ]
            baseline_cache[cache_key] = (
                float(next_rows.iloc[0][AMT]) if not next_rows.empty else None
            )
        costs = _sample_costs(rng, target_reward, baseline_cache[cache_key])
        reward_result = calculate_reward_v2(
            **costs, measurement_days_before=MEASUREMENT_DAYS, measurement_days_after=MEASUREMENT_DAYS,
        )
        if reward_result.status != "complete":
            raise RuntimeError(f"synthetic reward 계산 실패: {reward_result.reason}")

        logged_at = base_time + timedelta(hours=i)
        execution_end = logged_at + timedelta(days=MEASUREMENT_DAYS)
        measurement_start = logged_at.date()
        measurement_end = (logged_at + timedelta(days=MEASUREMENT_DAYS - 1)).date()
        baseline_start = (logged_at - timedelta(days=MEASUREMENT_DAYS)).date()
        baseline_end = (logged_at - timedelta(days=1)).date()
        row = {col: None for col in SCHEMA_COLUMNS}
        row.update({
            "decision_id": str(uuid.uuid4()),
            "store_id": f"synthetic-{trdar_cd}-{i % 20}",
            "trdar_cd": trdar_cd, "svc_induty_cd": svc_induty_cd,
            "yyqu_cd": as_of_q, "treatment_yyqu_cd": treatment_q,
            "problem_type": 등급, "context_schema_version": CONTEXT_SCHEMA_VERSION,
            **{col: float(context[j]) for j, col in enumerate(CONTEXT_COLS)},
            "candidate_actions": json.dumps(arms, ensure_ascii=False),
            "eligible_actions": json.dumps(arms, ensure_ascii=False),
            "unknown_actions": json.dumps([], ensure_ascii=False),
            "blocked_actions": json.dumps([], ensure_ascii=False),
            "selectable_actions": json.dumps(arms, ensure_ascii=False),
            "exploration_excluded_actions": json.dumps(
                [a for a in arms if a in action_rules.EXPLORATION_EXCLUDED_ACTIONS], ensure_ascii=False,
            ),
            "policy_mode": decision.mode,
            "action_probabilities": json.dumps(action_probabilities, ensure_ascii=False),
            "recommended_action": action,
            "policy_selected_action": action,
            "policy_selected_propensity": propensity,
            "approved_action": action, "executed_action": action,
            "behavior_propensity": propensity, "decision_source": "policy",
            "selection_source": "policy",
            "expected_rewards": json.dumps({}, ensure_ascii=False),
            "uncertainties": json.dumps({}, ensure_ascii=False),
            **costs, "measurement_days": MEASUREMENT_DAYS,
            "execution_started_at": logged_at.isoformat(),
            "execution_ended_at": execution_end.isoformat(),
            "measurement_started_at": measurement_start.isoformat(),
            "measurement_ended_at": measurement_end.isoformat(),
            "baseline_period_start": baseline_start.isoformat(),
            "baseline_period_end": baseline_end.isoformat(),
            "control_store_ids": json.dumps(
                [f"synthetic-control-{trdar_cd}"], ensure_ascii=False,
            ),
            "reward_definition_version": REWARD_DEFINITION_VERSION,
            "reward_denominator_floor": DEFAULT_REWARD_DENOMINATOR_FLOOR,
            "reward_status": "complete", "reward": reward_result.reward,
            "policy_version": bandit.policy_version, "model_version": bandit.model_version,
            "model_sha256": None, "ope_eligible": True, "training_eligible": True,
            "데이터_출처": "synthetic_v2", "logged_at": logged_at.isoformat(),
        })
        rows.append(row)
    return rows


def generate_all(seed: int = 0, n_per_grade: int = SAMPLES_PER_PROBLEM_TYPE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target_cells = _load_target_cells()
    panel = pd.read_csv(PANEL)
    cells_by_service: dict[str, list[tuple[int, str, int, int]]] = {}
    for cell in target_cells:
        cells_by_service.setdefault(cell[1], []).append(cell)
    all_rows = []
    for 등급, effects in TRUE_EFFECTS.items():
        arms = list(effects.keys())
        coverage = [
            (cells_by_service[svc][index % len(cells_by_service[svc])], action)
            for index, svc in enumerate(sorted(cells_by_service))
            for action in arms
        ]
        all_rows.extend(generate_rows(등급, arms, n_per_grade, rng, target_cells, coverage, panel))
    return pd.DataFrame(all_rows, columns=SCHEMA_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="synthetic v2 캠페인 로그 생성기")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-per-grade", type=int, default=SAMPLES_PER_PROBLEM_TYPE)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = generate_all(seed=args.seed, n_per_grade=args.n_per_grade)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"{len(df)}건 저장: {out_path}")
    else:
        print(df.head().to_string())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
