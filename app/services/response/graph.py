"""agent-runs LangGraph 그래프 (계획 §4).

범위: diagnose -> recommend -> estimate/evidence(병렬) -> validate -> await_approval ->
(approve: generate_report / edit: estimate+evidence 재계산 후 다시 await_approval / reject: 종료).

`generate_report`의 수치 재대조·재생성(최대 1회)은 `rag.generator.generate_report()` 내부의
evidence_gate 재생성 루프로 구현한다(계획 §4 상태도의 `generate_report -> generate_report`
자기루프와 동일한 것 — 별도 외부 루프를 그래프 레벨에 중복으로 두지 않는다). 실패해도
그래프를 에러로 끝내지 않고 경고와 함께 종료한다(정직한 낮은 신뢰도 > 거짓 확신, 계획 §3).

`validate`(OPE)가 기준정책보다 낮은 정책가치를 감지하면 `recommend`로 되돌리는 재시도 로직은
`_route_after_validate` -> `reject_candidate` -> `recommend`로 구현돼 있다. campaign_logs.csv에
실 로그가 쌓여 `evaluate_policy()`가 "사용가능" 판정을 낼 때만 트리거되며(그 전까지는 "판정불가"/
"탐색적"이라 재시도하지 않음), 무한루프 방지를 위해 `retry_count < 2`로 재시도 횟수를 제한한다.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.core.config import AGENT_RUNS_DB
from app.services.response import action_rules, bandit_store
from app.services.response.context import CONTEXT_DIM, build_context_vector
from app.services.response.state import RecommendationState
from rag.generator import generate_report
from rag.retriever import get_index

from scripts.modeling.sales_analysis import Diagnoser
from scripts.response_strategy.ope import evaluate_policy
from scripts.response_strategy.synthetic_control import measured_effect, segment_baseline

# 운영 기본값은 충분한 안정성을 유지하고, 통합 테스트에서는 patch로 낮출 수 있다.
OPE_BOOTSTRAP_SAMPLES = 200
MEASURED_EFFECT_BOOTSTRAP_SAMPLES = 500

_APPROVAL_STATUS = {"approve": "approved", "edit": "edited", "reject": "rejected"}


def _diagnose(state: RecommendationState) -> dict:
    diag = Diagnoser().diagnose(state["trdar_cd"], state["svc_induty_cd"], state.get("yyqu_cd"))
    warnings = list(state.get("warnings", []))

    if "error" in diag:
        warnings.append(diag["error"])
        return {"diagnosis": diag, "status": f"종료: {diag['error']}", "warnings": warnings}

    usable = bool(diag.get("6_신뢰도", {}).get("분석사용가능", False))
    등급 = diag.get("5_처방", {}).get("등급") if usable else None
    if not usable:
        warnings.append("분석사용가능=False — 대응방안 추천을 진행하지 않음")

    return {
        "diagnosis": diag,
        "문제유형": 등급,
        "status": "진단 완료" if usable else "종료: 진단 신뢰도 부족",
        "warnings": warnings,
    }


def _route_after_diagnose(state: RecommendationState) -> str:
    diag = state.get("diagnosis") or {}
    if "error" in diag or not diag.get("6_신뢰도", {}).get("분석사용가능", False):
        return END
    return "recommend"


def _recommend(state: RecommendationState) -> dict:
    등급 = state.get("문제유형")
    candidates = action_rules.candidate_actions(등급)
    warnings = list(state.get("warnings", []))

    if not candidates:
        warnings.append(f"등급={등급} — 실행 가능한 방안 후보 없음(구조적 요인이거나 뚜렷한 하락 없음)")
        return {
            "candidate_actions": [], "selected_action": None,
            "status": "종료: 방안 후보 없음", "warnings": warnings,
        }

    context = build_context_vector(state["diagnosis"])
    arms = [c["방안"] for c in candidates]
    bandit, model_loaded = bandit_store.load_or_coldstart(등급, context_dim=CONTEXT_DIM, arms=arms)
    result = bandit.select_arm(context)
    rejected = set(state.get("rejected_actions", []))
    ranked_arms = sorted(
        arms, key=lambda arm: result["arm별_점수"].get(arm, float("-inf")), reverse=True,
    )
    selected_arm = next((arm for arm in ranked_arms if arm not in rejected), None)
    selected = next((c for c in candidates if c["방안"] == selected_arm), None)
    if selected is None:
        warnings.append("OPE 기준을 통과할 대체 방안이 없어 최상위 원추천을 유지함")
        selected = next(c for c in candidates if c["방안"] == result["선택된_arm"])
    result = dict(result)
    result["선택된_arm"] = selected["방안"]
    result["propensity"] = result["arm별_propensity"][selected["방안"]]

    status = ("방안 선택 완료(학습된 모델 기반)" if model_loaded else
              "방안 선택 완료(콜드스타트 — 실 로그 학습 전이라 arm별 점수는 참고용)")
    return {
        "candidate_actions": candidates,
        "selected_action": selected,
        "bandit_result": result,
        "context_vector": context.tolist(),
        "policy_version": result.get("policy_version"),
        "status": status,
        "warnings": warnings,
    }


def _route_after_recommend(state: RecommendationState):
    if not state.get("selected_action"):
        return END
    return ["estimate", "evidence"]


def _estimate(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    baseline = segment_baseline(state["trdar_cd"], state["svc_induty_cd"], state.get("yyqu_cd"))
    effect = measured_effect(
        state["trdar_cd"], state["svc_induty_cd"], action,
        n_bootstrap=MEASURED_EFFECT_BOOTSTRAP_SAMPLES,
    )
    return {"scm_result": {"베이스라인": baseline, "실측효과": effect}}


def _evidence(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    axis = action_rules.ACTION_TO_AXIS.get(action)
    evidence = get_index().build_evidence(action, axis=axis)
    return {"rag_evidence": evidence}


def _validate(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    result = evaluate_policy(
        state["trdar_cd"], state["svc_induty_cd"],
        target_action_fn=lambda _ctx, _action=action: _action,
        n_bootstrap=OPE_BOOTSTRAP_SAMPLES,
    )
    return {"ope_result": result, "status": "검증 완료 — 승인 대기"}


def _route_after_validate(state: RecommendationState) -> str:
    result = state.get("ope_result") or {}
    retry_count = state.get("retry_count", 0)
    is_reliably_worse = (
        state.get("approval_status") != "edited"
        and
        result.get("판정") == "사용가능"
        and result.get("기준정책_대비_차이", 0) < 0
        and retry_count < 2
    )
    return "reject_candidate" if is_reliably_worse else "await_approval"


def _reject_candidate(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    rejected = list(state.get("rejected_actions", []))
    if action not in rejected:
        rejected.append(action)
    warnings = list(state.get("warnings", []))
    warnings.append(f"OPE에서 기준정책보다 낮게 평가된 '{action}'을 제외하고 재추천함")
    return {
        "rejected_actions": rejected,
        "retry_count": state.get("retry_count", 0) + 1,
        "warnings": warnings,
        "status": "OPE 기준 미달 — 대체 방안 재추천",
    }


def _await_approval(state: RecommendationState) -> dict:
    decision = interrupt({
        "선택된_방안": state.get("selected_action"),
        "방안_후보": state.get("candidate_actions", []),
        "효과추정": state.get("scm_result"),
        "근거_문헌": state.get("rag_evidence"),
        "정책_사전검증": state.get("ope_result"),
        "주의사항": state.get("warnings", []),
    })
    결정 = decision.get("결정") if isinstance(decision, dict) else None
    warnings = list(state.get("warnings", []))

    if 결정 == "edit":
        방안명 = decision.get("수정_방안")
        candidate = next((c for c in state.get("candidate_actions", []) if c["방안"] == 방안명), None)
        if candidate is None:
            warnings.append(f"edit 방안 '{방안명}' — 후보 목록에 없어 반려 처리")
            return {"approval_status": "rejected", "warnings": warnings, "status": "종료: 잘못된 edit 요청으로 반려"}
        return {
            "approval_status": "edited", "selected_action": candidate, "warnings": warnings,
            "status": "방안 수정됨 — 재계산 후 다시 승인 대기",
        }

    status = "승인됨 — 리포트 생성 중" if 결정 == "approve" else "종료: 반려됨"
    return {"approval_status": _APPROVAL_STATUS.get(결정, "rejected"), "warnings": warnings, "status": status}


def _route_after_approval(state: RecommendationState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        return "generate_report"
    if status == "edited":
        return ["estimate", "evidence"]
    return END


def _generate_report(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    target = (state.get("diagnosis") or {}).get("대상", {})
    shop_context = ", ".join(v for v in (target.get("업종명"), target.get("상권명")) if v)

    out = generate_report(state.get("rag_evidence") or {}, action, shop_context)
    warnings = list(state.get("warnings", []))
    if not out["verified"]:
        warnings.append("리포트 수치 검증 실패(재생성 후에도 위반) — 방향성만 신뢰하고 수치는 재확인 필요")

    return {
        "final_report": out,
        "warnings": warnings,
        "status": "리포트 생성 완료" if out["verified"] else "리포트 생성 완료(수치 검증 경고)",
    }


@lru_cache(maxsize=1)
def _checkpointer() -> SqliteSaver:
    AGENT_RUNS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AGENT_RUNS_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(RecommendationState)
    g.add_node("diagnose", _diagnose)
    g.add_node("recommend", _recommend)
    g.add_node("estimate", _estimate)
    g.add_node("evidence", _evidence)
    g.add_node("validate", _validate)
    g.add_node("reject_candidate", _reject_candidate)
    g.add_node("await_approval", _await_approval)
    g.add_node("generate_report", _generate_report)

    g.set_entry_point("diagnose")
    g.add_conditional_edges("diagnose", _route_after_diagnose, {END: END, "recommend": "recommend"})
    g.add_conditional_edges(
        "recommend", _route_after_recommend,
        {END: END, "estimate": "estimate", "evidence": "evidence"},
    )
    g.add_edge("estimate", "validate")
    g.add_edge("evidence", "validate")
    g.add_conditional_edges(
        "validate", _route_after_validate,
        {"reject_candidate": "reject_candidate", "await_approval": "await_approval"},
    )
    g.add_edge("reject_candidate", "recommend")
    g.add_conditional_edges(
        "await_approval", _route_after_approval,
        {"generate_report": "generate_report", "estimate": "estimate", "evidence": "evidence", END: END},
    )
    g.add_edge("generate_report", END)

    return g.compile(checkpointer=_checkpointer())
