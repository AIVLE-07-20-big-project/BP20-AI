# 실제(§0: real 또는 synthetic_v2) 표본이 coldstart→active 전환 기준을 만족하는지 판정(설계 §10)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.response.context import validate_context_vector
from scripts.response_strategy.ope import (
    action_support,
    effective_sample_size,
    load_v2_logged_batch_cached,
    stochastic_importance_weights,
    uniform_policy_fn,
)

DEFAULT_MINIMUM_REAL_SAMPLES = 100
DEFAULT_MINIMUM_SAMPLES_PER_ACTION = 20
DEFAULT_MINIMUM_OPE_ESS = 10.0
DEFAULT_MAXIMUM_CONTEXT_VIOLATION_RATE = 0.05


@dataclass(frozen=True)
class ActivationCriteria:
    minimum_real_samples: int = DEFAULT_MINIMUM_REAL_SAMPLES
    minimum_samples_per_action: int = DEFAULT_MINIMUM_SAMPLES_PER_ACTION
    minimum_ope_ess: float = DEFAULT_MINIMUM_OPE_ESS
    maximum_context_violation_rate: float = DEFAULT_MAXIMUM_CONTEXT_VIOLATION_RATE


@dataclass(frozen=True)
class ActivationResult:
    activated: bool
    reasons: tuple[str, ...]
    sample_count: int
    action_support: dict[str, int]
    ess: float
    context_violation_rate: float

    def as_dict(self) -> dict:
        return {
            "활성화": self.activated, "미충족_사유": list(self.reasons),
            "표본수": self.sample_count, "action_support": self.action_support,
            "유효표본크기_ESS": round(self.ess, 2),
            "context_위반비율": round(self.context_violation_rate, 4),
        }


def _context_violation_rate(contexts: np.ndarray) -> float:
    if len(contexts) == 0:
        return 0.0
    violations = 0
    for row in contexts:
        try:
            validate_context_vector(row)
        except ValueError:
            violations += 1
    return violations / len(contexts)


def check_activation(
    campaign_logs_v2: str | Path, *, problem_type: str, arms: list[str],
    reward_definition_version: str, context_schema_version: str,
    criteria: ActivationCriteria = ActivationCriteria(),
) -> ActivationResult:
    batch, _diag = load_v2_logged_batch_cached(
        campaign_logs_v2, problem_type=problem_type,
        reward_definition_version=reward_definition_version,
        context_schema_version=context_schema_version,
    )
    if batch is None:
        return ActivationResult(
            activated=False, reasons=("사용 가능한 v2 로그 없음",),
            sample_count=0, action_support={a: 0 for a in arms}, ess=0.0,
            context_violation_rate=0.0,
        )

    sample_count = len(batch.actions)
    support = action_support(batch, arms)
    weights = stochastic_importance_weights(batch, uniform_policy_fn(arms))
    ess = effective_sample_size(weights)
    violation_rate = _context_violation_rate(batch.contexts)

    reasons = []
    if sample_count < criteria.minimum_real_samples:
        reasons.append(f"표본수 부족({sample_count} < {criteria.minimum_real_samples})")
    under_supported = [a for a in arms if support.get(a, 0) < criteria.minimum_samples_per_action]
    if under_supported:
        reasons.append(f"action별 표본 부족: {under_supported}")
    if ess < criteria.minimum_ope_ess:
        reasons.append(f"ESS 부족({ess:.2f} < {criteria.minimum_ope_ess})")
    if violation_rate > criteria.maximum_context_violation_rate:
        reasons.append(f"context 위반비율 초과({violation_rate:.4f} > {criteria.maximum_context_violation_rate})")

    return ActivationResult(
        activated=not reasons, reasons=tuple(reasons),
        sample_count=sample_count, action_support=support, ess=ess,
        context_violation_rate=violation_rate,
    )
