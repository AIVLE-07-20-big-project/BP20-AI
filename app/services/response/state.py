"""agent-runs LangGraph 그래프의 공유 상태 (계획 §4).

§11(고객군 세그먼트 확장)은 아직 채택 전 검토 단계라, 지금은 단일 셀
(trdar_cd x svc_induty_cd x yyqu_cd) 기준의 원안 스키마로 구현한다.
"""
from __future__ import annotations

from typing import Literal, TypedDict


class RecommendationState(TypedDict, total=False):
    trdar_cd: str
    svc_induty_cd: str
    yyqu_cd: int | None

    diagnosis: dict | None
    문제유형: str | None  # = Diagnoser의 5_처방.등급 (docs/response_recommendation_agent_plan.md §6)

    candidate_actions: list[dict]
    selected_action: dict | None
    bandit_result: dict | None
    context_vector: list[float] | None  # build_context_vector() 출력 — campaign-logs가
                                          # thread_id 체크포인트에서 그대로 읽어감(계획 §0)
    policy_version: str | None

    scm_result: dict | None
    rag_evidence: dict | None
    ope_result: dict | None

    retry_count: int
    rejected_actions: list[str]
    approval_status: Literal["pending", "approved", "edited", "rejected"] | None
    final_report: dict | None

    status: str  # 진행 상태/종료 사유 요약(사람이 읽는 용도)
    warnings: list[str]
