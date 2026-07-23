# agent-runs LangGraph 그래프의 공유 상태 (계획 §4)
from __future__ import annotations

from typing import Literal, TypedDict


class RecommendationState(TypedDict, total=False):
    analysis_id: str
    user_id: str | None
    store_id: str | None
    trdar_cd: str
    svc_induty_cd: str
    yyqu_cd: int | None

    diagnosis: dict | None
    문제유형: str | None

    candidate_actions: list[dict]
    selected_action: dict | None
    bandit_result: dict | None
    context_vector: list[float] | None
    policy_version: str | None

    scm_result: dict | None
    rag_evidence: dict | None
    ope_result: dict | None

    retry_count: int
    rejected_actions: list[str]
    approval_status: Literal["pending", "approved", "edited", "rejected"] | None
    final_report: dict | None

    status: str
    warnings: list[str]
