# OPE — 정책 사전검증 (Off-Policy Evaluation)
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CAMPAIGN_LOGS = DATA / "agent" / "campaign_logs.csv"


@dataclass
class LoggedBatch:
    contexts: np.ndarray
    actions: list[str]
    propensities: np.ndarray
    rewards: np.ndarray


# 중요도가중 추정량. 로깅 정책이 실제로 택한 행동이 타깃 정책과 같을 때만 보상을
def ips(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str]) -> float:


    target_actions = np.array([target_action_fn(c) for c in batch.contexts])
    match = (target_actions == np.asarray(batch.actions)).astype(float)
    weights = match / batch.propensities
    return float(np.mean(weights * batch.rewards))


# 팔마다 (절편, 컨텍스트) 블록을 따로 둔다 — 팔별로 보상계수가 다를 수 있다는
def _arm_block_features(contexts: np.ndarray, actions, arms: list[str]) -> np.ndarray:




    blocks = []
    actions = np.asarray(actions)
    for arm in arms:
        indicator = (actions == arm).astype(float).reshape(-1, 1)
        blocks.append(indicator)
        blocks.append(indicator * contexts)
    return np.hstack(blocks)


# 팔별 블록 특징에 릿지 회귀로 보상모형을 적합한 뒤, 타깃 정책이 골랐을 행동에
def direct_method(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str],
                   arms: list[str], alpha: float = 0.1) -> float:


    X = _arm_block_features(batch.contexts, batch.actions, arms)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, batch.rewards)

    target_actions = [target_action_fn(c) for c in batch.contexts]
    X_target = _arm_block_features(batch.contexts, target_actions, arms)
    return float(np.mean(model.predict(X_target)))


# DM 예측값 + (실제 보상 - DM 예측값)의 중요도가중 보정항
def doubly_robust(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str],
                   arms: list[str], alpha: float = 0.1) -> float:


    X = _arm_block_features(batch.contexts, batch.actions, arms)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, batch.rewards)

    target_actions = np.array([target_action_fn(c) for c in batch.contexts])
    X_target = _arm_block_features(batch.contexts, target_actions, arms)
    dm_pred_target = model.predict(X_target)
    dm_pred_logged = model.predict(X)

    match = (target_actions == np.asarray(batch.actions)).astype(float)
    residual = match / batch.propensities * (batch.rewards - dm_pred_logged)
    return float(np.mean(dm_pred_target + residual))


# 정답(참 정책가치)을 아는 합성환경으로 IPS/DM/DR 구현 자체를 검증한다
def selftest(n: int = 5000, seed: int = 0) -> dict:






    rng = np.random.default_rng(seed)
    arms = ["A", "B", "C"]
    d = 4
    true_theta = {"A": rng.normal(1.0, 0.1, size=d), "B": rng.normal(0.0, 0.1, size=d),
                  "C": rng.normal(-0.5, 0.1, size=d)}
    true_intercept = {"A": 1.0, "B": 0.5, "C": 0.0}

    def true_reward(context, action):
        return float(true_theta[action] @ context + true_intercept[action])

    def target_action_fn(_context):
        return "A"

    contexts = rng.normal(size=(n, d))
    propensity = 1.0 / len(arms)
    actions = rng.choice(arms, size=n)
    noise = rng.normal(0, 0.5, size=n)
    rewards = np.array([true_reward(contexts[i], actions[i]) for i in range(n)]) + noise
    batch = LoggedBatch(contexts=contexts, actions=list(actions),
                         propensities=np.full(n, propensity), rewards=rewards)

    true_value = float(np.mean([true_reward(c, "A") for c in contexts]))

    est_ips = ips(batch, target_action_fn)
    est_dm = direct_method(batch, target_action_fn, arms)
    est_dr = doubly_robust(batch, target_action_fn, arms)

    return {
        "참값": round(true_value, 4),
        "IPS": {"추정값": round(est_ips, 4), "오차": round(abs(est_ips - true_value), 4)},
        "DM": {"추정값": round(est_dm, 4), "오차": round(abs(est_dm - true_value), 4)},
        "DR": {"추정값": round(est_dr, 4), "오차": round(abs(est_dr - true_value), 4)},
        "표본수": n,
    }


_CONTEXT_COLS = [f"context_{i}" for i in range(1, 7)]
_LOG_REQUIRED_COLS = {"action_id", "svc_induty_cd", "propensity", "reward", "executed"} | set(_CONTEXT_COLS)
MIN_RELIABLE_SAMPLES = 20


def _importance_weights(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str]) -> np.ndarray:
    target_actions = np.array([target_action_fn(c) for c in batch.contexts])
    match = (target_actions == np.asarray(batch.actions)).astype(float)
    return match / batch.propensities


# 기준정책 = 같은 후보군에서 매번 무작위로 골랐을 때의 기대 정책가치
def uniform_random_baseline(batch: LoggedBatch, arms: list[str], alpha: float = 0.1) -> float:





    values = [direct_method(batch, target_action_fn=lambda _c, a=arm: a, arms=arms, alpha=alpha)
              for arm in arms]
    return float(np.mean(values))


def _bootstrap_ci(batch: LoggedBatch, target_action_fn, arms: list[str],
                   n_bootstrap: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(batch.rewards)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = LoggedBatch(
            contexts=batch.contexts[idx],
            actions=[batch.actions[i] for i in idx],
            propensities=batch.propensities[idx],
            rewards=batch.rewards[idx],
        )
        values.append(doubly_robust(resampled, target_action_fn, arms))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


# 실 로그(data/campaign_logs.csv)로 실제 정책가치를 평가한다
def evaluate_policy(trdar_cd, svc_induty_cd, target_action_fn, campaign_logs=CAMPAIGN_LOGS,
                     baseline_fn=uniform_random_baseline, min_reliable_samples: int = MIN_RELIABLE_SAMPLES,
                     n_bootstrap: int = 200, seed: int = 0) -> dict:











    path = Path(campaign_logs)
    if not path.exists():
        return {"판정": "판정불가", "사유": "실제 로그 없음 (campaign_logs.csv 없음)", "표본수": 0}

    logs = pd.read_csv(path)
    if not _LOG_REQUIRED_COLS.issubset(logs.columns):
        return {"판정": "판정불가", "사유": "실제 로그 없음 (스키마 불일치)", "표본수": 0}

    matched = logs[
        (logs["svc_induty_cd"] == svc_induty_cd)
        & logs["executed"].astype(bool)
        & logs["reward"].notna()
        & logs["propensity"].gt(0) & logs["propensity"].le(1)
    ]
    if matched.empty:
        return {"판정": "판정불가", "사유": "실제 로그 없음", "표본수": 0}

    arms = sorted(matched["action_id"].unique().tolist())
    if len(arms) < 2:
        return {"판정": "판정불가",
                "사유": "비교 가능한 방안이 1개뿐이라 정책가치 비교 불가",
                "표본수": int(len(matched))}

    batch = LoggedBatch(
        contexts=matched[_CONTEXT_COLS].to_numpy(dtype=float),
        actions=matched["action_id"].tolist(),
        propensities=matched["propensity"].to_numpy(dtype=float),
        rewards=matched["reward"].to_numpy(dtype=float),
    )

    weights = _importance_weights(batch, target_action_fn)
    ess = float(weights.sum() ** 2 / np.sum(weights ** 2)) if weights.sum() > 0 else 0.0

    est_ips = ips(batch, target_action_fn)
    est_dm = direct_method(batch, target_action_fn, arms)
    est_dr = doubly_robust(batch, target_action_fn, arms)
    ci_low, ci_high = _bootstrap_ci(batch, target_action_fn, arms, n_bootstrap, seed)
    baseline_value = baseline_fn(batch, arms)

    reliable = len(matched) >= min_reliable_samples and ess >= min_reliable_samples / 2
    판정 = "사용가능" if reliable else "탐색적"

    return {
        "판정": 판정,
        "표본수": int(len(matched)),
        "유효표본크기_ESS": round(ess, 2),
        "정책가치_IPS": round(est_ips, 4),
        "정책가치_DM": round(est_dm, 4),
        "정책가치_DR": round(est_dr, 4),
        "정책가치_DR_95%CI": [round(ci_low, 4), round(ci_high, 4)],
        "기준정책가치(균등랜덤)": round(baseline_value, 4),
        "기준정책_대비_차이": round(est_dr - baseline_value, 4),
        "비교_arm목록": arms,
        "해석주의": "표본이 적으면(기본 임계 20건 미만) 탐색적으로만 참고할 것",
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args and args[0] == "selftest":
        print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
