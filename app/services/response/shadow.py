# v1 active와 v2 candidate를 같은 요청에서 비교하는 shadow 리포트(계획 §6)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.services.response import bandit_store
from app.services.response.policy import BanditPolicy
from scripts.response_strategy.bandit import NeuralContextualBandit


@dataclass(frozen=True)
class ShadowComparison:
    active_policy_version: str
    candidate_policy_version: str
    active_top_action: str
    candidate_top_action: str
    agree: bool
    active_used_fallback: bool
    active_scores: dict[str, dict]
    candidate_scores: dict[str, dict]

    def as_dict(self) -> dict:
        return {
            "active_policy_version": self.active_policy_version,
            "candidate_policy_version": self.candidate_policy_version,
            "active_선택": self.active_top_action,
            "candidate_선택": self.candidate_top_action,
            "일치": self.agree,
            "active_coldstart_fallback": self.active_used_fallback,
            "active_점수": self.active_scores,
            "candidate_점수": self.candidate_scores,
        }


# candidate가 없으면 None(비교 불가) — 실패가 실제 추천 흐름을 막으면 안 되므로 호출부는 예외를 흡수해야 한다
def compare_candidate_to_active(
    등급: str, candidate_version: str, context: np.ndarray, selectable_actions: list[str],
) -> ShadowComparison | None:
    candidate_file = bandit_store.candidate_path(등급, candidate_version)
    if not candidate_file.exists():
        return None

    active, active_loaded = bandit_store.load_or_coldstart(
        등급, context_dim=len(context), arms=selectable_actions,
    )
    candidate = NeuralContextualBandit.load_any(candidate_file)

    active_policy = BanditPolicy(bandit=active)
    candidate_policy = BanditPolicy(bandit=candidate)

    active_probs = active_policy.probabilities(context, selectable_actions, mode="serve")
    candidate_probs = candidate_policy.probabilities(context, selectable_actions, mode="serve")

    active_top = max(active_probs, key=active_probs.get)
    candidate_top = max(candidate_probs, key=candidate_probs.get)

    return ShadowComparison(
        active_policy_version=active.policy_version if active_loaded else "coldstart",
        candidate_policy_version=candidate.policy_version,
        active_top_action=active_top, candidate_top_action=candidate_top,
        agree=active_top == candidate_top,
        active_used_fallback=not active_loaded,
        active_scores={s.action_id: s.as_dict(4) for s in active_policy.score(context)},
        candidate_scores={s.action_id: s.as_dict(4) for s in candidate_policy.score(context)},
    )
