# 문제유형별 실험 설정(epsilon·비용·기간·매장수)과 중단조건(계획 §6)
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import BANDIT_EXPERIMENT_EPSILON


@dataclass(frozen=True)
class ExperimentConfig:
    epsilon: float = BANDIT_EXPERIMENT_EPSILON
    max_cost_per_store: float = 200_000.0
    max_duration_days: int = 30
    max_stores: int = 20
    max_blocked_action_rate: float = 0.20


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# 고비용·비가역적 문제유형은 더 보수적인 실험 설정을 쓴다(비어 있으면 기본값)
EXPERIMENT_CONFIG_BY_PROBLEM_TYPE: dict[str, ExperimentConfig] = {
    "차별화": ExperimentConfig(epsilon=0.05, max_cost_per_store=500_000.0, max_stores=10),
}


def experiment_config_for(등급: str) -> ExperimentConfig:
    return EXPERIMENT_CONFIG_BY_PROBLEM_TYPE.get(등급, DEFAULT_EXPERIMENT_CONFIG)


def should_stop_experiment(
    등급: str, *, blocked_action_rate: float, elapsed_days: int, store_count: int,
) -> tuple[bool, str | None]:
    cfg = experiment_config_for(등급)
    if blocked_action_rate > cfg.max_blocked_action_rate:
        return True, f"차단 action 비율 초과({blocked_action_rate:.2f} > {cfg.max_blocked_action_rate})"
    if elapsed_days > cfg.max_duration_days:
        return True, f"실험 기간 초과({elapsed_days}일 > {cfg.max_duration_days}일)"
    if store_count > cfg.max_stores:
        return True, f"실험 매장 수 초과({store_count} > {cfg.max_stores})"
    return False, None
