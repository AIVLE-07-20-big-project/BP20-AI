# agent-runs LangGraph 그래프 (계획 §4단계)
#
# diagnose → score_candidates → validate_candidates → select_action
#   → [estimate, evidence] → prepare_approval → await_approval → generate_report
from __future__ import annotations

import logging
import sqlite3
from functools import lru_cache

import numpy as np
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.core.config import AGENT_RUNS_DB, BANDIT_V2_SHADOW_ENABLED, CAMPAIGN_LOGS_V2
from app.services.response import action_rules, bandit_store, shadow
from app.services.response.context import CONTEXT_DIM, CONTEXT_SCHEMA_VERSION, build_context_vector
from app.services.response.policy import BanditPolicy
from app.services.response.state import RecommendationState
from app.services.sales_summary import build_diagnosis_facts
from rag.generator import generate_report
from rag.retriever import get_index

from scripts.modeling.sales_analysis import Diagnoser
from scripts.response_strategy.ope import evaluate_action_safety
from scripts.response_strategy.synthetic_control import measured_effect, segment_baseline

OPE_BOOTSTRAP_SAMPLES = 200
MEASURED_EFFECT_BOOTSTRAP_SAMPLES = 500

_APPROVAL_STATUS = {"approve": "approved", "edit": "edited", "reject": "rejected"}
_SAFETY_VERDICT_TO_STATUS = {"차단": "blocked", "사용가능": "eligible"}

_logger = logging.getLogger(__name__)


def _diagnose(state: RecommendationState) -> dict:
    diag = state.get("diagnosis")
    if diag is None:
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
    return "score_candidates"


# 후보별 기대보상·불확실성·UCB를 계산한다 — 선택은 하지 않는다(선택은 validate 이후)
def _score_candidates(state: RecommendationState) -> dict:
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
    scores = bandit.score_arms(context)

    return {
        "candidate_actions": candidates,
        "candidate_scores": {s.action_id: s.as_dict(ndigits=4) for s in scores},
        "context_vector": context.tolist(),
        "policy_version": bandit.policy_version,
        "model_version": bandit.model_version,
        "model_status": bandit_store.model_status(model_loaded),
        "model_sha256": bandit_store.model_sha256(등급),
        "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "reward_definition_version": bandit.reward_definition_version,
        "status": f"점수 계산 완료({'학습된 모델' if model_loaded else '콜드스타트'})",
        "warnings": warnings,
    }


def _route_after_score_candidates(state: RecommendationState) -> str:
    if not state.get("candidate_actions"):
        return END
    return "validate_candidates"


# 후보 전체를 ActionSafetyOPE로 검증해 eligible/unknown/blocked를 매긴다(blocked는 선택 제외)
def _validate_candidates(state: RecommendationState) -> dict:
    등급 = state.get("문제유형")
    candidates = state.get("candidate_actions") or []
    arms = [c["방안"] for c in candidates]
    warnings = list(state.get("warnings", []))

    candidate_status: dict[str, str] = {}
    candidate_safety: dict[str, dict] = {}
    for action in arms:
        result = evaluate_action_safety(
            CAMPAIGN_LOGS_V2, problem_type=등급, candidate_action=action, arms=arms,
            reward_definition_version=state.get("reward_definition_version"),
            context_schema_version=state.get("context_schema_version"),
        )
        candidate_safety[action] = result
        candidate_status[action] = _SAFETY_VERDICT_TO_STATUS.get(result.get("판정"), "unknown")

    selectable = [a for a in arms if candidate_status[a] != "blocked"]
    selectable_candidates = [
        c for c in candidates if c["방안"] in selectable
    ]
    exploration_excluded = [a for a in arms if a in action_rules.EXPLORATION_EXCLUDED_ACTIONS]

    if not selectable:
        warnings.append("모든 후보가 OPE 안전성 검사에서 차단되어 자동 추천을 중단함")
        return {
            "candidate_actions": [],
            "candidate_status": candidate_status, "candidate_safety": candidate_safety,
            "selectable_actions": [], "exploration_excluded_actions": exploration_excluded,
            "ope_result": candidate_safety, "selected_action": None,
            "status": "종료: 모든 후보 방안이 안전성 검사에서 차단됨",
            "warnings": warnings,
        }

    return {
        # 차단된 방안은 후보 목록·선택 UI에 노출하지 않는다.
        "candidate_actions": selectable_candidates,
        "candidate_status": candidate_status, "candidate_safety": candidate_safety,
        "selectable_actions": selectable, "exploration_excluded_actions": exploration_excluded,
        "ope_result": candidate_safety,
        "status": f"후보 검증 완료(선택가능 {len(selectable)}/{len(arms)})",
        "warnings": warnings,
    }


def _route_after_validate_candidates(state: RecommendationState) -> str:
    if not state.get("selectable_actions"):
        return END
    return "select_action"


# v1 active(또는 coldstart)와 최신 v2 candidate를 선택에 영향 없이 비교만 한다(실패해도 흐름은 유지)
def _build_shadow_report(등급: str, context: np.ndarray, selectable: list[str]) -> dict | None:
    if not BANDIT_V2_SHADOW_ENABLED:
        return None
    candidate_version = bandit_store.latest_candidate_version(등급)
    if candidate_version is None:
        return None
    try:
        comparison = shadow.compare_candidate_to_active(등급, candidate_version, context, selectable)
    except Exception:
        _logger.warning("shadow 비교 실패(등급=%s)", 등급, exc_info=True)
        return None
    return comparison.as_dict() if comparison else None


# coldstart(학습된 active 모델 없음)면 미학습 모델의 argmax 대신 비즈니스 우선순위
# 기본값으로 fallback한다(설계 §10). 학습된 모델이 있으면 BanditPolicy로 선택한다.
def _select_action(state: RecommendationState) -> dict:
    등급 = state.get("문제유형")
    candidates = state.get("candidate_actions") or []
    # candidate_actions는 validate_candidates가 이미 selectable로만 좁혀놨다 — 모델은
    # 등급 전체 arm 집합으로 저장돼 있으므로, 로딩은 항상 전체 arm으로 해야 한다
    # (아니면 arm 수가 달라 BanditLoadMismatch로 매번 콜드스타트된다).
    full_arms = [c["방안"] for c in action_rules.candidate_actions(등급)]
    selectable = state.get("selectable_actions") or []
    warnings = list(state.get("warnings", []))

    context = np.asarray(state["context_vector"], dtype=np.float32)
    bandit, model_loaded = bandit_store.load_or_coldstart(등급, context_dim=CONTEXT_DIM, arms=full_arms)
    shadow_report = _build_shadow_report(등급, context, selectable)

    if not model_loaded:
        action = action_rules.coldstart_default_action(등급, selectable)
        selected = next(c for c in candidates if c["방안"] == action)
        return {
            "policy_decision": {
                "policy_mode": "coldstart", "policy_selected_action": action,
                "policy_selected_propensity": None, "action_probabilities": {},
                "fallback_reason": "coldstart — 학습된 모델 없음, 비즈니스 우선순위 기본값 사용",
            },
            "policy_selected_action": action,
            "selected_action": selected,
            "decision_source": "policy",
            "selection_source": "business_rule_fallback",
            "shadow_report": shadow_report,
            "status": "방안 선택 완료(coldstart 기본값)",
            "warnings": warnings,
        }

    policy = BanditPolicy(
        bandit=bandit, exploration_excluded_actions=frozenset(action_rules.EXPLORATION_EXCLUDED_ACTIONS),
    )
    decision = policy.choose(context, selectable, mode="serve", rng=np.random.default_rng())
    selected = next(c for c in candidates if c["방안"] == decision.selected_action)

    return {
        "policy_decision": decision.as_dict(),
        "policy_selected_action": decision.selected_action,
        "selected_action": selected,
        "decision_source": "policy",
        "selection_source": "policy",
        "shadow_report": shadow_report,
        "status": f"방안 선택 완료(policy_mode={decision.mode})",
        "warnings": warnings,
    }


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
    evidence = get_index().build_evidence(action, axis=axis, action=action)
    return {"rag_evidence": evidence}


# estimate/evidence 완료를 기다렸다가 승인화면 상태 문구를 통일한다(최초 선택·edit 재계산 공용)
def _prepare_approval(state: RecommendationState) -> dict:
    return {"status": "검증 완료 — 승인 대기"}


def _await_approval(state: RecommendationState) -> dict:
    decision = interrupt({
        "선택된_방안": state.get("selected_action"),
        "방안_후보": state.get("candidate_actions", []),
        "선택_가능_방안": [
            c for c in state.get("candidate_actions", [])
            if (state.get("candidate_status") or {}).get(c.get("방안")) != "blocked"
        ],
        "후보별_점수": state.get("candidate_scores"),
        "후보별_상태": state.get("candidate_status"),
        "선택_확률분포": (state.get("policy_decision") or {}).get("action_probabilities"),
        "선택_이유": (state.get("policy_decision") or {}).get("fallback_reason") or "기대보상 최대 후보 선택",
        "효과추정": state.get("scm_result"),
        "근거_문헌": state.get("rag_evidence"),
        "정책_사전검증": state.get("ope_result"),
        "shadow_비교": state.get("shadow_report"),
        "주의사항": state.get("warnings", []),
    })
    결정 = decision.get("결정") if isinstance(decision, dict) else None
    선택방안 = decision.get("선택_방안") or decision.get("selected_action") if isinstance(decision, dict) else None
    warnings = list(state.get("warnings", []))

    if 결정 == "edit":
        방안명 = 선택방안 or decision.get("수정_방안")
        candidate = next((c for c in state.get("candidate_actions", []) if c["방안"] == 방안명), None)
        candidate_status = state.get("candidate_status") or {}
        if candidate is None:
            warnings.append(f"edit 방안 '{방안명}' — 후보 목록에 없어 반려 처리")
            return {"approval_status": "rejected", "warnings": warnings, "status": "종료: 잘못된 edit 요청으로 반려"}
        if candidate_status.get(방안명) == "blocked":
            warnings.append(f"edit 방안 '{방안명}' — OPE 안전성 검사에서 차단(blocked)되어 반려 처리")
            return {
                "approval_status": "rejected", "warnings": warnings,
                "status": "종료: 차단된 방안으로 edit 요청되어 반려",
            }
        return {
            "approval_status": "edited", "selected_action": candidate,
            "decision_source": "human_edit", "warnings": warnings,
            "status": "방안 수정됨 — 재계산 후 다시 승인 대기",
        }

    if 결정 == "approve":
        decision_source = state.get("decision_source") or "policy"
        selected_action_update = None
        if 선택방안 is not None:
            candidate = next(
                (c for c in state.get("candidate_actions", []) if c.get("방안") == 선택방안), None,
            )
            candidate_status = state.get("candidate_status") or {}
            if candidate is None:
                warnings.append(f"approve 방안 '{선택방안}' — 후보 목록에 없어 반려 처리")
                return {"approval_status": "rejected", "warnings": warnings,
                        "status": "종료: 잘못된 승인 방안으로 반려"}
            if candidate_status.get(선택방안) == "blocked":
                warnings.append(f"approve 방안 '{선택방안}' — OPE 안전성 검사에서 차단되어 반려 처리")
                return {"approval_status": "rejected", "warnings": warnings,
                        "status": "종료: 차단된 방안으로 승인 요청되어 반려 처리"}
            # 바로 승인한 경우에도 선택 방안이 최종 상태가 되도록 한다.
            selected_action_update = candidate
            decision_source = "human_edit"
        result = {
            "approval_status": "approved", "decision_source": decision_source,
            "executed_action": (selected_action_update or state.get("selected_action") or {}).get("방안"),
            "warnings": warnings, "status": "승인됨 — 리포트 생성 중",
        }
        if selected_action_update is not None:
            result["selected_action"] = selected_action_update
            # 한 번의 approve 요청으로 대안을 선택한 경우에도 이전 방안의
            # 검증·근거가 남지 않도록 선택 방안 기준으로 즉시 재계산한다.
            selected_state = dict(state)
            selected_state["selected_action"] = selected_action_update
            result.update(_estimate(selected_state))
            result.update(_evidence(selected_state))
        return result

    return {"approval_status": _APPROVAL_STATUS.get(결정, "rejected"), "warnings": warnings, "status": "종료: 반려됨"}


def _route_after_approval(state: RecommendationState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        return "generate_report"
    if status == "edited":
        return ["estimate", "evidence"]
    # 반려 이력도 같은 thread에서 다시 승인할 수 있도록 승인 대기를 유지한다.
    if status == "rejected":
        return "await_approval"
    return END


def _generate_report(state: RecommendationState) -> dict:
    action = state["selected_action"]["방안"]
    target = (state.get("diagnosis") or {}).get("대상", {})
    shop_context = ", ".join(v for v in (target.get("업종명"), target.get("상권명")) if v)
    diagnosis_facts = build_diagnosis_facts(
        state.get("report") or {}, state.get("detailed_analysis") or {},
    )

    out = generate_report(state.get("rag_evidence") or {}, action, diagnosis_facts, shop_context)
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
    g.add_node("score_candidates", _score_candidates)
    g.add_node("validate_candidates", _validate_candidates)
    g.add_node("select_action", _select_action)
    g.add_node("estimate", _estimate)
    g.add_node("evidence", _evidence)
    g.add_node("prepare_approval", _prepare_approval)
    g.add_node("await_approval", _await_approval)
    g.add_node("generate_report", _generate_report)

    g.set_entry_point("diagnose")
    g.add_conditional_edges(
        "diagnose", _route_after_diagnose, {END: END, "score_candidates": "score_candidates"},
    )
    g.add_conditional_edges(
        "score_candidates", _route_after_score_candidates,
        {END: END, "validate_candidates": "validate_candidates"},
    )
    g.add_conditional_edges(
        "validate_candidates", _route_after_validate_candidates,
        {END: END, "select_action": "select_action"},
    )
    g.add_edge("select_action", "estimate")
    g.add_edge("select_action", "evidence")
    g.add_edge("estimate", "prepare_approval")
    g.add_edge("evidence", "prepare_approval")
    g.add_edge("prepare_approval", "await_approval")
    g.add_conditional_edges(
        "await_approval", _route_after_approval,
        {
            "generate_report": "generate_report", "estimate": "estimate", "evidence": "evidence",
            # 반려(rejected) 시 _route_after_approval이 자기 자신으로 돌아가는데(재승인 대기),
            # 이 self-loop이 ends에 없으면 반려할 때마다 KeyError로 그래프가 죽는다.
            "await_approval": "await_approval", END: END,
        },
    )
    g.add_edge("generate_report", END)

    return g.compile(checkpointer=_checkpointer())
