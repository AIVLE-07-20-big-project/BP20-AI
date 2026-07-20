"""
OPE — 정책 사전검증 (Off-Policy Evaluation)

Bandit이 고른 정책을 실제 배포 전에, 과거 로그로 기대값을 추정한다. IPS/DM/DR을
sklearn만으로 직접 구현한다(causalml/mabwiser/obp는 Windows 빌드 위험이 있어 배제 —
docs/response_recommendation_agent_plan.md §3 참고).

2단계 검증:
    selftest()  실 로그가 없는 지금, 정답(참 정책가치)을 아는 합성환경으로 추정량
                구현 자체가 정확한지 확인.
    evaluate_policy()  실 로그(data/campaign_logs.csv)가 쌓이면 그걸로 실제 정책가치
                평가(스키마 확정 전까지는 자리만 잡아둠 —
                synthetic_control.measured_effect()와 동일한 상황).

로그 스키마: 각 로그는 (context, action, propensity, reward).
target_action_fn(context) -> action 시그니처로 일반화해 합성환경과 실데이터 양쪽에
그대로 쓴다.

의존성
    필수: numpy, pandas, scikit-learn (전부 requirements.txt에 이미 있음 — 새 의존성 없음)

사용
    python ope.py selftest
"""
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
CAMPAIGN_LOGS = DATA / "campaign_logs.csv"


@dataclass
class LoggedBatch:
    contexts: np.ndarray       # (n, d)
    actions: list[str]         # (n,)
    propensities: np.ndarray   # (n,) — 로깅 정책이 그 행동을 택했을 확률
    rewards: np.ndarray        # (n,)


def ips(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str]) -> float:
    """중요도가중 추정량. 로깅 정책이 실제로 택한 행동이 타깃 정책과 같을 때만 보상을
    반영하고, propensity로 나눠 보정한다."""
    target_actions = np.array([target_action_fn(c) for c in batch.contexts])
    match = (target_actions == np.asarray(batch.actions)).astype(float)
    weights = match / batch.propensities
    return float(np.mean(weights * batch.rewards))


def _arm_block_features(contexts: np.ndarray, actions, arms: list[str]) -> np.ndarray:
    """팔마다 (절편, 컨텍스트) 블록을 따로 둔다 — 팔별로 보상계수가 다를 수 있다는
    가정을 회귀 특징에 그대로 반영해야, 보상모형(DM)이 구조적으로 틀린 값에 수렴하지
    않는다(모든 팔이 같은 계수를 공유한다고 잘못 가정하면 self-test에서 DM만 크게
    틀려 보여, 구현 오류와 모형 오설정을 구분할 수 없게 된다)."""
    blocks = []
    actions = np.asarray(actions)
    for arm in arms:
        indicator = (actions == arm).astype(float).reshape(-1, 1)
        blocks.append(indicator)
        blocks.append(indicator * contexts)
    return np.hstack(blocks)


def direct_method(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str],
                   arms: list[str], alpha: float = 0.1) -> float:
    """팔별 블록 특징에 릿지 회귀로 보상모형을 적합한 뒤, 타깃 정책이 골랐을 행동에
    대한 예측 보상의 평균을 낸다."""
    X = _arm_block_features(batch.contexts, batch.actions, arms)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, batch.rewards)

    target_actions = [target_action_fn(c) for c in batch.contexts]
    X_target = _arm_block_features(batch.contexts, target_actions, arms)
    return float(np.mean(model.predict(X_target)))


def doubly_robust(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str],
                   arms: list[str], alpha: float = 0.1) -> float:
    """DM 예측값 + (실제 보상 - DM 예측값)의 중요도가중 보정항.
    보상모형(DM)이나 중요도가중(IPS) 둘 중 하나만 맞아도 편향이 없다."""
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


def selftest(n: int = 5000, seed: int = 0) -> dict:
    """정답(참 정책가치)을 아는 합성환경으로 IPS/DM/DR 구현 자체를 검증한다.

    로깅 정책은 완전 무작위(팔마다 propensity = 1/팔수)로 로그를 만들고, 타깃 정책
    (항상 arm "A"를 고르는 정책)의 참값은 노이즈 평균이 0이라는 사실로 직접 계산해
    세 추정량과 비교한다.
    """
    rng = np.random.default_rng(seed)
    arms = ["A", "B", "C"]
    d = 4
    true_theta = {"A": rng.normal(1.0, 0.1, size=d), "B": rng.normal(0.0, 0.1, size=d),
                  "C": rng.normal(-0.5, 0.1, size=d)}
    true_intercept = {"A": 1.0, "B": 0.5, "C": 0.0}

    def true_reward(context, action):
        return float(true_theta[action] @ context + true_intercept[action])

    def target_action_fn(_context):
        return "A"  # 검증 대상 타깃 정책: 항상 A

    contexts = rng.normal(size=(n, d))
    propensity = 1.0 / len(arms)  # 완전 무작위 로깅 정책 — propensity가 상수라 검증이 쉬움
    actions = rng.choice(arms, size=n)
    noise = rng.normal(0, 0.5, size=n)
    rewards = np.array([true_reward(contexts[i], actions[i]) for i in range(n)]) + noise
    batch = LoggedBatch(contexts=contexts, actions=list(actions),
                         propensities=np.full(n, propensity), rewards=rewards)

    true_value = float(np.mean([true_reward(c, "A") for c in contexts]))  # 노이즈 평균 0

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
MIN_RELIABLE_SAMPLES = 20   # 이 이상 & ESS 충분해야 "사용가능"(조정 가능한 초기값)


def _importance_weights(batch: LoggedBatch, target_action_fn: Callable[[np.ndarray], str]) -> np.ndarray:
    target_actions = np.array([target_action_fn(c) for c in batch.contexts])
    match = (target_actions == np.asarray(batch.actions)).astype(float)
    return match / batch.propensities


def uniform_random_baseline(batch: LoggedBatch, arms: list[str], alpha: float = 0.1) -> float:
    """기준정책 = 같은 후보군에서 매번 무작위로 골랐을 때의 기대 정책가치.

    각 arm을 항상 고르는 정책의 DM 추정치를 arm마다 구해 평균낸다(균등 혼합 정책의 기대값).
    별도 추정량을 새로 만들지 않고 기존 direct_method()를 arm 수만큼 재사용한다.
    """
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


def evaluate_policy(trdar_cd, svc_induty_cd, target_action_fn, campaign_logs=CAMPAIGN_LOGS,
                     baseline_fn=uniform_random_baseline, min_reliable_samples: int = MIN_RELIABLE_SAMPLES,
                     n_bootstrap: int = 200, seed: int = 0) -> dict:
    """실 로그(data/campaign_logs.csv)로 실제 정책가치를 평가한다.

    같은 업종(svc_induty_cd) 전체의 실행된(executed) 로그를 모아 IPS/DM/DR을 계산한다
    (trdar_cd는 인자로 받지만 필터로 쓰지 않는다 — 단일 상권만으로는 정책 비교에 필요한
    표본이 거의 항상 부족하고, 이 프로젝트의 대상이 여러 지점을 운영하는 기업이라 업종
    전체로 일반화하는 게 맞다. synthetic_control.measured_effect()의 "동일 업종 전체 폴백"과
    같은 논리).

    사례가 없거나(현재 상태) 비교 가능한 방안이 1개뿐이면(비교할 대상 자체가 없음) 억지로
    수치를 만들지 않고 판정불가를 반환한다 — 억지로 수치를 만들지 않는다.
    """
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
