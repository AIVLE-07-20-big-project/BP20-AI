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
    report: dict | None
    detailed_analysis: dict | None
    문제유형: str | None

    candidate_actions: list[dict]
    candidate_scores: dict[str, dict] | None
    candidate_status: dict[str, str] | None
    candidate_safety: dict[str, dict] | None
    selectable_actions: list[str]
    exploration_excluded_actions: list[str]

    policy_decision: dict | None
    policy_selected_action: str | None
    selected_action: dict | None
    approved_action: str | None
    executed_action: str | None
    decision_source: Literal["policy", "human_edit", "admin_override"] | None
    selection_source: Literal["policy", "business_rule_fallback"] | None

    context_vector: list[float] | None
    policy_version: str | None
    model_version: str | None
    model_status: str | None
    model_sha256: str | None
    context_schema_version: str | None
    reward_definition_version: str | None

    scm_result: dict | None
    rag_evidence: dict | None
    ope_result: dict | None
    shadow_report: dict | None
    final_report: dict | None

    approval_status: Literal["pending", "approved", "edited", "rejected"] | None

    status: str
    warnings: list[str]
