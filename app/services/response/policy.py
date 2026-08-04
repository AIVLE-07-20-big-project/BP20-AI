# 선택과 OPE가 공유하는 단일 정책(계획 §2단계) — serve/experiment 확률·선택을 여기서만 만든다
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from scripts.response_strategy.bandit import ArmScore, NeuralContextualBandit

PolicyMode = Literal["serve", "experiment"]
DecisionSource = Literal["policy", "human_edit", "admin_override"]

EXPERIMENT_EPSILON_DEFAULT = 0.10


@dataclass(frozen=True)
class PolicyDecision:
    selected_action: str
    propensity: float
    action_probabilities: dict[str, float]
    requested_mode: PolicyMode
    mode: PolicyMode
    policy_version: str
    fallback_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "policy_selected_action": self.selected_action,
            "policy_selected_propensity": self.propensity,
            "action_probabilities": dict(self.action_probabilities),
            "requested_mode": self.requested_mode,
            "policy_mode": self.mode,
            "policy_version": self.policy_version,
            "fallback_reason": self.fallback_reason,
        }


# 사람이 정책 선택을 바꾸면(edit/override) 정책 propensity를 로그에 붙이지 않는다(설계 §7)
def behavior_propensity_for(decision_source: DecisionSource, decision: PolicyDecision) -> float | None:
    if decision_source == "policy":
        return decision.propensity
    return None


@dataclass
class BanditPolicy:
    bandit: NeuralContextualBandit
    exploration_excluded_actions: frozenset[str] = field(default_factory=frozenset)
    epsilon: float = EXPERIMENT_EPSILON_DEFAULT

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon은 [0, 1] 범위여야 합니다")

    @property
    def policy_version(self) -> str:
        return self.bandit.policy_version

    def score(self, context: np.ndarray) -> list[ArmScore]:
        return self.bandit.score_arms(context)

    def probabilities(
        self, context: np.ndarray, selectable_actions: list[str], mode: PolicyMode,
    ) -> dict[str, float]:
        if not selectable_actions:
            raise ValueError("selectable_actions가 비어 있습니다")

        scores = {s.action_id: s for s in self.score(context)}
        missing = [a for a in selectable_actions if a not in scores]
        if missing:
            raise ValueError(f"selectable_actions 중 점수가 없는 action이 있습니다: {missing}")

        if mode == "serve":
            best = max(selectable_actions, key=lambda a: scores[a].expected_reward)
            return {a: (1.0 if a == best else 0.0) for a in selectable_actions}

        if mode == "experiment":
            experiment_actions = self.experiment_actions(selectable_actions)
            if not experiment_actions:
                raise ValueError(
                    "experiment_actions가 비어 있습니다 — choose()가 serve로 fallback해야 합니다"
                )
            k = len(experiment_actions)
            best = max(experiment_actions, key=lambda a: scores[a].expected_reward)
            probs: dict[str, float] = {}
            for a in selectable_actions:
                if a in self.exploration_excluded_actions:
                    probs[a] = 0.0
                elif a == best:
                    probs[a] = 1.0 - self.epsilon + self.epsilon / k
                else:
                    probs[a] = self.epsilon / k
            return probs

        raise ValueError(f"알 수 없는 policy mode: {mode}")

    def experiment_actions(self, selectable_actions: list[str]) -> list[str]:
        return [a for a in selectable_actions if a not in self.exploration_excluded_actions]

    def choose(
        self, context: np.ndarray, selectable_actions: list[str], mode: PolicyMode,
        rng: np.random.Generator,
    ) -> PolicyDecision:
        actual_mode: PolicyMode = mode
        fallback_reason: str | None = None

        if mode == "experiment" and not self.experiment_actions(selectable_actions):
            actual_mode = "serve"
            fallback_reason = "실험 가능한 후보가 없어 serve로 대체함"

        probs = self.probabilities(context, selectable_actions, actual_mode)
        actions = list(probs.keys())
        weights = np.asarray([probs[a] for a in actions], dtype=np.float64)
        weights = weights / weights.sum()

        selected = actions[int(rng.choice(len(actions), p=weights))]
        return PolicyDecision(
            selected_action=selected,
            propensity=probs[selected],
            action_probabilities=probs,
            requested_mode=mode,
            mode=actual_mode,
            policy_version=self.policy_version,
            fallback_reason=fallback_reason,
        )
