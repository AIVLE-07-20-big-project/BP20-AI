"""현재 밴딧 모델의 OPE 및 Contextual Bandit 성능 지표를 계산한다.

예시:
    python scripts/response_strategy/evaluate_bandit_metrics.py
    python scripts/response_strategy/evaluate_bandit_metrics.py --log samples/campaign_logs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# 파일 경로로 직접 실행해도 프로젝트 루트를 import 경로에 포함한다.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.response_strategy.bandit import NeuralContextualBandit
from scripts.response_strategy.ope import (
    LoggedBatchV2,
    _arm_block_features,
    effective_sample_size,
    stochastic_doubly_robust,
    stochastic_importance_weights,
    stochastic_ips,
    stochastic_snips,
)


CONTEXT_COLS = [f"context_{i}" for i in range(1, 7)]


def _fast_dr_cluster_ci(batch, target_policy, baseline_policy, arms, n_bootstrap, seed=0):
    """DR 영향값을 한 번 계산한 뒤 점포 단위로 신뢰구간을 계산한다."""
    X_logged = _arm_block_features(batch.contexts, batch.actions, arms)
    model = Ridge(alpha=0.1, fit_intercept=False).fit(X_logged, batch.rewards)
    logged_prediction = model.predict(X_logged)
    def dr_influence(policy):
        prediction = np.zeros(len(batch.rewards))
        for arm in arms:
            probs = np.array([policy(context).get(arm, 0.0) for context in batch.contexts])
            X_arm = _arm_block_features(batch.contexts, [arm] * len(batch.contexts), arms)
            prediction += probs * model.predict(X_arm)
        weights = stochastic_importance_weights(batch, policy)
        return prediction + weights * (batch.rewards - logged_prediction)

    influence = dr_influence(target_policy) - dr_influence(baseline_policy)

    rng = np.random.default_rng(seed)
    stores = np.unique(batch.store_ids)
    values = []
    for _ in range(max(n_bootstrap, 1)):
        sampled = rng.choice(stores, size=len(stores), replace=True)
        idx = np.concatenate([np.where(batch.store_ids == store)[0] for store in sampled])
        values.append(float(influence[idx].mean()))
    return tuple(np.percentile(values, [2.5, 97.5]))


def _policy_fn(bandit: NeuralContextualBandit):
    """모델의 UCB 점수를 softmax propensity 분포로 변환한다."""

    def policy(context: np.ndarray) -> dict[str, float]:
        scores = np.asarray(
            [score.ucb_score for score in bandit.score_arms(context)], dtype=float
        )
        temperature = max(float(bandit.temperature), 1e-8)
        exp_scores = np.exp((scores - scores.max()) / temperature)
        probs = exp_scores / exp_scores.sum()
        return dict(zip(bandit.arms, probs))

    return policy


def _evaluate_group(group: pd.DataFrame, model_root: Path, n_bootstrap: int) -> dict[str, object]:
    problem_type = str(group["problem_type"].iloc[0])
    group = group[
        group["executed_action"].notna()
        & group["reward"].notna()
        & group["behavior_propensity"].between(0, 1, inclusive="neither")
    ].copy()
    observed_arms = set(group["executed_action"].unique().tolist())
    model_path = model_root / problem_type / "active.pt"

    if len(observed_arms) < 2:
        raise ValueError(f"{problem_type}: 비교 가능한 대응방안이 2개 미만입니다.")
    if not model_path.exists():
        raise FileNotFoundError(f"{problem_type}: 모델 파일이 없습니다: {model_path}")

    # 저장된 모델의 대응방안 순서를 그대로 사용한다. 순서를 정렬하면
    # 모델의 A/b 통계와 대응방안의 매핑이 달라질 수 있다.
    bandit = NeuralContextualBandit.load_any(model_path)
    if set(bandit.arms) != observed_arms:
        raise ValueError(f"{problem_type}: 로그와 모델의 대응방안 목록이 다릅니다.")
    arms = list(bandit.arms)
    target_policy = _policy_fn(bandit)
    batch = LoggedBatchV2(
        contexts=group[CONTEXT_COLS].to_numpy(dtype=float),
        actions=group["executed_action"].tolist(),
        propensities=group["behavior_propensity"].to_numpy(dtype=float),
        rewards=group["reward"].to_numpy(dtype=float),
        store_ids=group["store_id"].astype(str).to_numpy(),
        problem_type=problem_type,
        reward_definition_version=str(group["reward_definition_version"].iloc[0]),
        context_schema_version=str(group["context_schema_version"].iloc[0]),
    )

    uniform_policy = lambda _context: {arm: 1.0 / len(arms) for arm in arms}
    weights = stochastic_importance_weights(batch, target_policy)
    target_actions = [max(target_policy(context), key=target_policy(context).get)
                      for context in batch.contexts]
    policy_value = stochastic_doubly_robust(batch, target_policy, arms, seed=0)
    baseline_value = stochastic_doubly_robust(batch, uniform_policy, arms, seed=0)
    ci_low, ci_high = _fast_dr_cluster_ci(
        batch, target_policy, uniform_policy, arms, n_bootstrap=n_bootstrap, seed=0,
    )
    support = group["executed_action"].value_counts()
    min_support = int(min(support.get(arm, 0) for arm in arms))
    improvement = policy_value - baseline_value
    relative_improvement = improvement / abs(baseline_value) if baseline_value else 0.0
    ess = effective_sample_size(weights)

    return {
        "문제 유형": problem_type,
        "표본 수": len(group),
        "평균 실제 보상": float(batch.rewards.mean()),
        "IPS": stochastic_ips(batch, target_policy),
        "SNIPS": stochastic_snips(batch, target_policy),
        "DR 정책 가치": policy_value,
        "균등 기준 정책 가치": baseline_value,
        "기준 대비 개선폭": improvement,
        "상대 개선율": relative_improvement,
        "개선폭 95% CI 하한": ci_low,
        "개선폭 95% CI 상한": ci_high,
        "ESS": ess,
        "ESS 비율": ess / len(group),
        "최소 대응방안 관측 수": min_support,
        "선택 일치율": float(np.mean(np.asarray(target_actions) == np.asarray(batch.actions))),
        "대응방안 수": len(arms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Contextual Bandit OPE 성능지표 계산")
    parser.add_argument("--log", type=Path, default=Path("samples/campaign_logs.csv"))
    parser.add_argument("--model-root", type=Path, default=Path("model/bandit"))
    parser.add_argument("--bootstrap", type=int, default=30, help="신뢰구간 부트스트랩 반복 횟수")
    args = parser.parse_args()

    if not args.log.exists():
        raise FileNotFoundError(f"로그 파일이 없습니다: {args.log}")

    logs = pd.read_csv(args.log)
    required = {
        "problem_type", "executed_action", "behavior_propensity", "reward",
        "store_id", "reward_definition_version", "context_schema_version",
        *CONTEXT_COLS,
    }
    missing = required - set(logs.columns)
    if missing:
        raise ValueError(f"로그 필드가 없습니다: {sorted(missing)}")

    results = []
    for _, group in logs.groupby("problem_type", sort=True):
        try:
            results.append(_evaluate_group(group, args.model_root, args.bootstrap))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[제외] {exc}")

    if not results:
        raise RuntimeError("계산 가능한 문제 유형이 없습니다.")

    result_df = pd.DataFrame(results)
    numeric = result_df.select_dtypes(include=[np.number]).columns
    weighted = {
        column: np.average(result_df[column], weights=result_df["표본 수"])
        for column in numeric if column != "표본 수"
    }
    weighted["문제 유형"] = "전체(표본수 가중평균)"
    weighted["표본 수"] = int(result_df["표본 수"].sum())
    weighted["상대 개선율"] = (
        weighted["기준 대비 개선폭"] / abs(weighted["균등 기준 정책 가치"])
        if weighted["균등 기준 정책 가치"] else 0.0
    )
    result_df = pd.concat([result_df, pd.DataFrame([weighted])], ignore_index=True)

    print("\n[OPE·Contextual Bandit 성능지표]")
    print(result_df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\n실무 판단 기준:")
    print("- 개선폭 95% CI 하한이 0보다 크면 기준 정책 대비 개선 근거가 있습니다.")
    print("- ESS 비율이 낮거나 최소 대응방안 관측 수가 작으면 평가 결과를 보류합니다.")
    print("- 선택 일치율은 정확도가 아니라 로그 정책과의 행동 일치율입니다.")


if __name__ == "__main__":
    main()
