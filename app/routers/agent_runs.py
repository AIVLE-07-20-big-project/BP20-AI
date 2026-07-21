"""POST /api/v1/agent-runs, GET /api/v1/agent-runs/{thread_id}, POST .../resume

계획 §7 — diagnose→recommend→estimate/evidence→validate까지 실행하고 await_approval에서
정지한다. resume은 approve(→generate_report)/edit(→estimate+evidence 재계산 후 다시
await_approval)/reject(→종료)로 재개한다(계획 §4 라우팅 규칙표).
"""
from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from app.schemas.agent_run import AgentRunRequest, AgentRunResumeRequest
from app.services.response.graph import get_graph

router = APIRouter()


def _to_response(thread_id: str, values: dict, interrupt_value: dict | None) -> dict:
    payload = dict(values)
    payload["상태"] = payload.pop("status", "알 수 없음")
    payload["thread_id"] = thread_id
    payload["대기중_승인"] = interrupt_value
    return payload


def start_agent_run(initial_state: dict) -> dict:
    """새 에이전트 실행을 시작하고 승인 대기 상태까지 진행한다."""
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(initial_state, config=config)
    interrupts = result.pop("__interrupt__", None)
    interrupt_value = interrupts[0].value if interrupts else None
    return _to_response(thread_id, result, interrupt_value)


def read_agent_run(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"agent-run을 찾을 수 없음: {thread_id}")

    interrupt_value = None
    for task in snapshot.tasks:
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            break
    return _to_response(thread_id, snapshot.values, interrupt_value)


def continue_agent_run(thread_id: str, decision: dict) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"agent-run을 찾을 수 없음: {thread_id}")
    if not any(task.interrupts for task in snapshot.tasks):
        raise HTTPException(status_code=409, detail="현재 승인 대기 상태가 아닙니다")

    result = get_graph().invoke(Command(resume=decision), config=config)
    interrupts = result.pop("__interrupt__", None)
    interrupt_value = interrupts[0].value if interrupts else None
    return _to_response(thread_id, result, interrupt_value)


@router.post("/agent-runs", deprecated=True)
def create_agent_run(payload: AgentRunRequest) -> dict:
    initial_state = {
        "trdar_cd": payload.trdar_cd,
        "svc_induty_cd": payload.svc_induty_cd,
        "yyqu_cd": payload.yyqu_cd,
        "warnings": [],
    }
    return start_agent_run(initial_state)


@router.get("/agent-runs/{thread_id}")
def get_agent_run(thread_id: str) -> dict:
    return read_agent_run(thread_id)


@router.post("/agent-runs/{thread_id}/resume")
def resume_agent_run(thread_id: str, payload: AgentRunResumeRequest) -> dict:
    resume_payload = {"결정": payload.decision}
    if payload.modification_plan is not None:
        resume_payload["수정_방안"] = payload.modification_plan

    return continue_agent_run(thread_id, resume_payload)
