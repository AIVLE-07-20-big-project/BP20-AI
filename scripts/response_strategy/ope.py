# OPE — 정책 사전검증 (Off-Policy Evaluation)
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.core.config import CAMPAIGN_LOGS_V2

# 운영 기본 로그는 v2 계약을 사용한다. 명시적으로 다른 경로를 넘기는
# 기존 호출·테스트는 하위 호환을 위해 그대로 지원한다.
CAMPAIGN_LOGS = CAMPAIGN_LOGS_V2


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


# 기준정책 = 같은 후보군에서 매번 무작위로 골랐을 때의 기대 정책가치.
# 타깃 정책가치(evaluate_policy의 est_dr)와 같은 DR 추정량으로 맞춰, 두 값을 뺀
# '기준정책_대비_차이'가 DR-vs-DM 추정량 혼용으로 편향되지 않도록 한다.
def uniform_random_baseline(batch: LoggedBatch, arms: list[str], alpha: float = 0.1) -> float:





    values = [doubly_robust(batch, target_action_fn=lambda _c, a=arm: a, arms=arms, alpha=alpha)
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


# 실 로그(campaign_logs_v2.csv)로 실제 정책가치를 평가한다.
# 주의: trdar_cd는 호출부 시그니처 호환을 위해 받지만 필터에는 쓰지 않는다 —
# 상권 단위로 좁히면 표본이 급감해 대부분 '판정불가'가 되므로, 의도적으로 동일
# 업종(svc_induty_cd) 로그 전체를 풀링해 정책가치를 추정한다.
def evaluate_policy(trdar_cd, svc_induty_cd, target_action_fn, campaign_logs=CAMPAIGN_LOGS,
                     baseline_fn=uniform_random_baseline, min_reliable_samples: int = MIN_RELIABLE_SAMPLES,
                     n_bootstrap: int = 200, seed: int = 0) -> dict:











    path = Path(campaign_logs)
    if not path.exists():
        return {"판정": "판정불가", "사유": "실제 로그 없음 (campaign_logs_v2.csv 없음)", "표본수": 0}

    logs = pd.read_csv(path)
    # v2 로그를 legacy evaluate_policy 시그니처에서도 읽을 수 있게 정규화한다.
    # 신규 운영 경로는 evaluate_action_safety()를 사용하지만, 이 함수는 CLI·기존
    # 호출부 호환용으로 남아 있으므로 입력 파일만 v2로 통일한다.
    if {"executed_action", "behavior_propensity", "net_sales_after"}.issubset(logs.columns):
        logs = logs.rename(columns={
            "executed_action": "action_id",
            "behavior_propensity": "propensity",
        })
        logs["executed"] = logs["action_id"].notna() & logs["reward"].notna()
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


# 확률 target 기반 OPE — 위 legacy 함수는 graph.py의 결정론적 호출부 호환용으로 남기고,
# 여기서부터는 확률분포 target policy를 지원한다(계획 §3)

TargetPolicyFn = Callable[[np.ndarray], dict[str, float]]

CONTEXT_DIM = len(_CONTEXT_COLS)


@dataclass
class LoggedBatchV2:
    contexts: np.ndarray
    actions: list[str]
    propensities: np.ndarray
    rewards: np.ndarray
    store_ids: np.ndarray
    problem_type: str
    reward_definition_version: str
    context_schema_version: str


V2_REQUIRED_COLS = {
    "problem_type", "context_schema_version", "reward_definition_version",
    "policy_version", "model_version", "store_id", "decision_source",
    "executed_action", "behavior_propensity", "reward", "reward_status",
    "ope_eligible", "training_eligible", "데이터_출처", "logged_at",
} | set(_CONTEXT_COLS)


def deterministic_policy_fn(action: str) -> TargetPolicyFn:
    return lambda _ctx, _a=action: {_a: 1.0}


def uniform_policy_fn(arms: list[str]) -> TargetPolicyFn:
    probs = {a: 1.0 / len(arms) for a in arms}
    return lambda _ctx, _p=probs: dict(_p)


# 로그 v2 CSV를 problem_type·버전·기간(cutoff_before=누적, cutoff_after=holdout)으로 거른다
def load_v2_logged_batch(
    path: str | Path, *, problem_type: str, reward_definition_version: str,
    context_schema_version: str, cutoff_after: str | None = None,
    cutoff_before: str | None = None,
    allowed_data_sources: tuple[str, ...] = ("real", "synthetic_v2"),
) -> tuple["LoggedBatchV2 | None", dict]:
    path = Path(path)
    diag: dict = {"총행수": 0, "사용행수": 0, "제외행수": 0, "제외사유": {}}
    if not path.exists():
        diag["제외사유"]["로그없음"] = True
        return None, diag

    logs = pd.read_csv(path)
    diag["총행수"] = int(len(logs))
    missing_cols = V2_REQUIRED_COLS - set(logs.columns)
    if missing_cols:
        diag["제외사유"]["스키마 컬럼 누락"] = sorted(missing_cols)
        diag["제외행수"] = diag["총행수"]
        return None, diag

    mask = pd.Series(True, index=logs.index)

    def exclude(reason: str, bad_mask: pd.Series) -> None:
        nonlocal mask
        newly = bad_mask & mask
        count = int(newly.sum())
        if count:
            diag["제외사유"][reason] = diag["제외사유"].get(reason, 0) + count
        mask &= ~bad_mask

    exclude("problem_type 불일치", logs["problem_type"] != problem_type)
    exclude("reward_definition_version 불일치", logs["reward_definition_version"] != reward_definition_version)
    exclude("context_schema_version 불일치", logs["context_schema_version"] != context_schema_version)
    exclude("ope_eligible=False", ~logs["ope_eligible"].astype(bool))
    exclude("decision_source != policy", logs["decision_source"] != "policy")
    exclude("behavior_propensity 결측", logs["behavior_propensity"].isna())
    exclude("reward 결측", logs["reward"].isna())
    exclude("데이터_출처 허용 목록 밖", ~logs["데이터_출처"].isin(allowed_data_sources))
    # store_id가 없으면 store 단위 cross-fitting/bootstrap에서 NaN과 문자열이 섞여
    # np.unique가 깨진다(정렬 불가) — 사용 전에 걸러낸다.
    exclude("store_id 결측", logs["store_id"].isna())
    if cutoff_after is not None:
        exclude("safety_data_version 이전 기간 제외(holdout 분리)", logs["logged_at"] < cutoff_after)
    if cutoff_before is not None:
        exclude("safety_data_version 이후 기간 제외(누적 이력만 사용)", logs["logged_at"] >= cutoff_before)

    used = logs[mask]
    diag["사용행수"] = int(len(used))
    diag["제외행수"] = diag["총행수"] - diag["사용행수"]
    if used.empty:
        return None, diag

    batch = LoggedBatchV2(
        contexts=used[_CONTEXT_COLS].to_numpy(dtype=float),
        actions=used["executed_action"].tolist(),
        propensities=used["behavior_propensity"].to_numpy(dtype=float),
        rewards=used["reward"].to_numpy(dtype=float),
        store_ids=used["store_id"].astype(str).to_numpy(),
        problem_type=problem_type,
        reward_definition_version=reward_definition_version,
        context_schema_version=context_schema_version,
    )
    return batch, diag


@lru_cache(maxsize=64)
def _load_v2_logged_batch_cached(
    path_str: str, mtime_ns: int, problem_type: str, reward_definition_version: str,
    context_schema_version: str, cutoff_after: str | None, cutoff_before: str | None,
    allowed_data_sources: tuple[str, ...],
) -> tuple["LoggedBatchV2 | None", dict]:
    return load_v2_logged_batch(
        path_str, problem_type=problem_type, reward_definition_version=reward_definition_version,
        context_schema_version=context_schema_version, cutoff_after=cutoff_after,
        cutoff_before=cutoff_before, allowed_data_sources=allowed_data_sources,
    )


# load_v2_logged_batch()를 파일 mtime로 캐싱 — 로그가 바뀌면 자동 무효화된다
def load_v2_logged_batch_cached(
    path: str | Path, *, problem_type: str, reward_definition_version: str,
    context_schema_version: str, cutoff_after: str | None = None,
    cutoff_before: str | None = None,
    allowed_data_sources: tuple[str, ...] = ("real", "synthetic_v2"),
) -> tuple["LoggedBatchV2 | None", dict]:
    path = Path(path)
    mtime_ns = path.stat().st_mtime_ns if path.exists() else -1
    return _load_v2_logged_batch_cached(
        str(path), mtime_ns, problem_type, reward_definition_version, context_schema_version,
        cutoff_after, cutoff_before, allowed_data_sources,
    )


# w_i = π(a_i|x_i) / μ(a_i|x_i) — target이 deterministic이면 기존 match/propensity와 같다
def stochastic_importance_weights(batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn) -> np.ndarray:
    weights = np.empty(len(batch.actions))
    for i, (ctx, action) in enumerate(zip(batch.contexts, batch.actions)):
        probs = target_policy_fn(ctx)
        weights[i] = probs.get(action, 0.0) / batch.propensities[i]
    return weights


def stochastic_ips(batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn) -> float:
    weights = stochastic_importance_weights(batch, target_policy_fn)
    return float(np.mean(weights * batch.rewards))


# Self-normalized IPS — 분산이 크게 흔들리는 상황에서 IPS의 대안
def stochastic_snips(batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn) -> float:
    weights = stochastic_importance_weights(batch, target_policy_fn)
    denom = float(weights.sum())
    if denom <= 0:
        raise ValueError("SNIPS 분모(중요도가중치 합)가 0 이하입니다")
    return float(np.sum(weights * batch.rewards) / denom)


# (row, arm) 쌍을 한 번에 쌓아 model.predict()를 1회만 호출한다(행별 반복 호출은 느림)
def _dm_target_predictions(
    contexts: np.ndarray, target_policy_fn: TargetPolicyFn, arms: list[str], model: Ridge,
) -> np.ndarray:
    n = len(contexts)
    preds = np.zeros(n)
    row_idx: list[int] = []
    row_arms: list[str] = []
    row_weights: list[float] = []
    for i, ctx in enumerate(contexts):
        for arm, p in target_policy_fn(ctx).items():
            if p == 0 or arm not in arms:
                continue
            row_idx.append(i)
            row_arms.append(arm)
            row_weights.append(p)

    if not row_idx:
        return preds

    X = _arm_block_features(contexts[row_idx], row_arms, arms)
    weighted_preds = model.predict(X) * np.asarray(row_weights)
    np.add.at(preds, row_idx, weighted_preds)
    return preds


def stochastic_direct_method(
    batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn, arms: list[str], alpha: float = 0.1,
) -> float:
    X = _arm_block_features(batch.contexts, batch.actions, arms)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(X, batch.rewards)
    return float(np.mean(_dm_target_predictions(batch.contexts, target_policy_fn, arms, model)))


# store_id 그룹 기준 K-fold — 같은 매장의 행은 항상 같은 fold에 남긴다(계획 §3)
def _cross_fitted_logged_predictions(
    batch: LoggedBatchV2, arms: list[str], alpha: float = 0.1, n_splits: int = 5, seed: int = 0,
) -> np.ndarray:
    X = _arm_block_features(batch.contexts, batch.actions, arms)
    unique_stores = np.unique(batch.store_ids)
    effective_splits = min(n_splits, len(unique_stores))
    if effective_splits < 2:
        model = Ridge(alpha=alpha, fit_intercept=False).fit(X, batch.rewards)
        return model.predict(X)

    rng = np.random.default_rng(seed)
    shuffled = unique_stores.copy()
    rng.shuffle(shuffled)
    fold_of_store = {store: i % effective_splits for i, store in enumerate(shuffled)}
    fold_ids = np.array([fold_of_store[s] for s in batch.store_ids])

    preds = np.zeros(len(batch.rewards))
    for fold in range(effective_splits):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        model = Ridge(alpha=alpha, fit_intercept=False).fit(X[train_mask], batch.rewards[train_mask])
        preds[test_mask] = model.predict(X[test_mask])
    return preds


# DM(cross-fitted 잔차 보정) + IPS 보정항 — nuisance model 과적합 편향을 cross-fitting으로 줄임
def stochastic_doubly_robust(
    batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn, arms: list[str],
    alpha: float = 0.1, n_splits: int = 5, seed: int = 0,
) -> float:
    dm_pred_logged = _cross_fitted_logged_predictions(batch, arms, alpha=alpha, n_splits=n_splits, seed=seed)

    X_full = _arm_block_features(batch.contexts, batch.actions, arms)
    full_model = Ridge(alpha=alpha, fit_intercept=False).fit(X_full, batch.rewards)
    dm_target = _dm_target_predictions(batch.contexts, target_policy_fn, arms, full_model)

    weights = stochastic_importance_weights(batch, target_policy_fn)
    residual = weights * (batch.rewards - dm_pred_logged)
    return float(np.mean(dm_target + residual))


def effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float(total ** 2 / np.sum(weights ** 2))


def action_support(batch: LoggedBatchV2, arms: list[str]) -> dict[str, int]:
    counts = {a: 0 for a in arms}
    for action in batch.actions:
        if action in counts:
            counts[action] += 1
    return counts


# store_id 단위로 통째로 리샘플(cluster bootstrap)해 target-baseline 차이의 CI를 구한다
def paired_cluster_bootstrap_ci(
    batch: LoggedBatchV2, target_policy_fn: TargetPolicyFn, baseline_policy_fn: TargetPolicyFn,
    arms: list[str], n_bootstrap: int = 200, seed: int = 0, alpha: float = 0.1,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_stores = np.unique(batch.store_ids)
    diffs = []
    for _ in range(n_bootstrap):
        sampled_stores = rng.choice(unique_stores, size=len(unique_stores), replace=True)
        idx = np.concatenate([np.where(batch.store_ids == s)[0] for s in sampled_stores])
        resampled = LoggedBatchV2(
            contexts=batch.contexts[idx], actions=[batch.actions[i] for i in idx],
            propensities=batch.propensities[idx], rewards=batch.rewards[idx],
            store_ids=batch.store_ids[idx], problem_type=batch.problem_type,
            reward_definition_version=batch.reward_definition_version,
            context_schema_version=batch.context_schema_version,
        )
        target_v = stochastic_doubly_robust(resampled, target_policy_fn, arms, alpha=alpha, seed=seed)
        baseline_v = stochastic_doubly_robust(resampled, baseline_policy_fn, arms, alpha=alpha, seed=seed)
        diffs.append(target_v - baseline_v)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


MIN_HOLDOUT_SAMPLES = MIN_RELIABLE_SAMPLES


# 후보 action 하나를 사전 차단할지 판단한다 — 누적 이력 사용, 학습 데이터와 겹쳐도 됨(설계 §8)
def evaluate_action_safety(
    campaign_logs_v2: str | Path, *, problem_type: str, candidate_action: str, arms: list[str],
    reward_definition_version: str, context_schema_version: str,
    safety_data_version: str | None = None,
    min_reliable_samples: int = MIN_RELIABLE_SAMPLES, n_bootstrap: int = 200, seed: int = 0,
) -> dict:
    batch, diag = load_v2_logged_batch_cached(
        campaign_logs_v2, problem_type=problem_type,
        reward_definition_version=reward_definition_version,
        context_schema_version=context_schema_version,
        cutoff_before=safety_data_version,
    )
    if batch is None:
        return {"판정": "판정불가", "사유": "사용 가능한 v2 로그 없음", "로그_진단": diag}

    candidate_fn = deterministic_policy_fn(candidate_action)
    baseline_fn = uniform_policy_fn(arms)

    weights = stochastic_importance_weights(batch, candidate_fn)
    ess = effective_sample_size(weights)
    support = action_support(batch, arms)

    candidate_value = stochastic_doubly_robust(batch, candidate_fn, arms, seed=seed)
    baseline_value = stochastic_doubly_robust(batch, baseline_fn, arms, seed=seed)
    diff_lo, diff_hi = paired_cluster_bootstrap_ci(
        batch, candidate_fn, baseline_fn, arms, n_bootstrap=n_bootstrap, seed=seed,
    )

    enough_data = (
        diag["사용행수"] >= min_reliable_samples
        and ess >= min_reliable_samples / 2
        and support.get(candidate_action, 0) >= max(3, min_reliable_samples // 10)
    )
    if not enough_data:
        판정 = "판정불가(표본부족)"
    elif diff_hi < 0:
        판정 = "차단"
    else:
        판정 = "사용가능"

    return {
        "판정": 판정, "action": candidate_action,
        "표본수": diag["사용행수"], "유효표본크기_ESS": round(ess, 2),
        "action_support": support,
        "정책가치_DR": round(candidate_value, 4),
        "기준정책가치(균등랜덤)": round(baseline_value, 4),
        "기준정책_대비_차이_95%CI": [round(diff_lo, 4), round(diff_hi, 4)],
        "안전성_데이터_버전": safety_data_version,
        "로그_진단": diag,
    }


# 후보 모델 전체를 승격 전에 평가한다 — training_data_cutoff 이후 holdout만 사용(설계 §8)
def evaluate_policy_value(
    campaign_logs_v2: str | Path, *, problem_type: str, arms: list[str],
    target_policy_fn: TargetPolicyFn, target_policy_version: str, evaluated_policy_version: str,
    training_data_cutoff: str, reward_definition_version: str, context_schema_version: str,
    min_holdout_samples: int = MIN_HOLDOUT_SAMPLES, n_bootstrap: int = 200, seed: int = 0,
) -> dict:
    if target_policy_version != evaluated_policy_version:
        return {
            "판정": "판정불가",
            "사유": (
                f"target_policy_version({target_policy_version})과 "
                f"evaluated_policy_version({evaluated_policy_version})이 다름"
            ),
        }

    batch, diag = load_v2_logged_batch_cached(
        campaign_logs_v2, problem_type=problem_type,
        reward_definition_version=reward_definition_version,
        context_schema_version=context_schema_version,
        cutoff_after=training_data_cutoff,
    )
    if batch is None or diag["사용행수"] < min_holdout_samples:
        return {"판정": "판정불가", "사유": "holdout 표본 부족 — 승격 보류", "로그_진단": diag}

    baseline_fn = uniform_policy_fn(arms)
    weights = stochastic_importance_weights(batch, target_policy_fn)
    ess = effective_sample_size(weights)
    support = action_support(batch, arms)

    policy_value = stochastic_doubly_robust(batch, target_policy_fn, arms, seed=seed)
    baseline_value = stochastic_doubly_robust(batch, baseline_fn, arms, seed=seed)
    diff_lo, diff_hi = paired_cluster_bootstrap_ci(
        batch, target_policy_fn, baseline_fn, arms, n_bootstrap=n_bootstrap, seed=seed,
    )

    return {
        "판정": "평가완료", "policy_version": evaluated_policy_version,
        "표본수": diag["사용행수"], "유효표본크기_ESS": round(ess, 2),
        "action_support": support,
        "정책가치_DR": round(policy_value, 4),
        "기준정책가치(균등랜덤)": round(baseline_value, 4),
        "기준정책_대비_차이_95%CI": [round(diff_lo, 4), round(diff_hi, 4)],
        "training_data_cutoff": training_data_cutoff,
        "로그_진단": diag,
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
