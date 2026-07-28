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
def _sample_costs(rng: np.random.Generator, target_reward: float) -> dict[str, float]:
    contribution_before = abs(rng.normal(CONTRIBUTION_BEFORE_MEAN, CONTRIBUTION_BEFORE_STD))
    denominator = max(contribution_before, DEFAULT_REWARD_DENOMINATOR_FLOOR)
    contribution_after = contribution_before + target_reward * denominator

    net_sales_before = contribution_before / (1 - VARIABLE_COST_RATIO)
    variable_cost_before = net_sales_before * VARIABLE_COST_RATIO
    campaign_cost = net_sales_before * CAMPAIGN_COST_RATIO

    net_sales_after = (contribution_after + campaign_cost) / (1 - VARIABLE_COST_RATIO)
    variable_cost_after = net_sales_after * VARIABLE_COST_RATIO

    return {
        "net_sales_before": net_sales_before, "net_sales_after": net_sales_after,
        "variable_cost_before": variable_cost_before, "variable_cost_after": variable_cost_after,
        "campaign_cost": campaign_cost,
    }


def generate_rows(등급: str, arms: list[str], n: int, rng: np.random.Generator) -> list[dict]:
    effects = TRUE_EFFECTS[등급]
    bandit = NeuralContextualBandit(
        context_dim=CONTEXT_DIM, arms=arms, policy_version=GENERATOR_VERSION,
        model_version=GENERATOR_VERSION, context_schema_version=CONTEXT_SCHEMA_VERSION,
        reward_definition_version=REWARD_DEFINITION_VERSION,
    )
    policy = BanditPolicy(bandit=bandit, epsilon=EXPERIMENT_EPSILON)

    rows = []
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        context = _sample_context(rng)
        # 매 행마다 "믿는 최고 arm"을 랜덤하게 바꿔 action별 표본이 고르게 퍼지게 한다
        champion = rng.choice(arms)
        bandit._prior_bias = np.array([CHAMPION_PRIOR_BIAS if a == champion else 0.0 for a in arms])

        decision = policy.choose(context, arms, mode="experiment", rng=rng)
        action = decision.selected_action

        target_reward = effects[action] + rng.normal(0, REWARD_NOISE_STD)
        costs = _sample_costs(rng, target_reward)
        reward_result = calculate_reward_v2(
            **costs, measurement_days_before=MEASUREMENT_DAYS, measurement_days_after=MEASUREMENT_DAYS,
        )
        if reward_result.status != "complete":
            raise RuntimeError(f"synthetic reward 계산 실패: {reward_result.reason}")

        logged_at = base_time + timedelta(hours=i)
        row = {col: None for col in SCHEMA_COLUMNS}
        row.update({
            "decision_id": str(uuid.uuid4()),
            "store_id": f"synthetic-{등급}-{i % 20}",
            "trdar_cd": 0, "svc_induty_cd": "SYNTHETIC",
            "yyqu_cd": 20261, "treatment_yyqu_cd": 20262,
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
            "action_probabilities": json.dumps(decision.action_probabilities, ensure_ascii=False),
            "recommended_action": decision.selected_action,
            "policy_selected_action": decision.selected_action,
            "policy_selected_propensity": decision.propensity,
            "approved_action": action, "executed_action": action,
            "behavior_propensity": decision.propensity, "decision_source": "policy",
            "selection_source": "policy",
            "expected_rewards": json.dumps({}, ensure_ascii=False),
            "uncertainties": json.dumps({}, ensure_ascii=False),
            **costs, "measurement_days": MEASUREMENT_DAYS,
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
    all_rows = []
    for 등급, effects in TRUE_EFFECTS.items():
        arms = list(effects.keys())
        all_rows.extend(generate_rows(등급, arms, n_per_grade, rng))
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
