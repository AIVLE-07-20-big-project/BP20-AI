# POST /api/v1/agent-runs, GET /api/v1/agent-runs/{thread_id}, POST .../resume
from __future__ import annotations

from datetime import date
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from langgraph.types import Command

from app.schemas.agent_run import AgentRunRequest, AgentRunResumeRequest
from app.services.response.graph import get_graph

router = APIRouter(tags=["전략 추천"])


# 상권·업종·기준분기(공공데이터 분기)가 같아도 업로드한 POS 실제 기간은 다를 수 있어
# before/after 실행을 구분하는 표시용 라벨을 실제 POS 기간에서 만든다.
def _format_analysis_period(detailed_analysis: dict | None) -> str | None:
    summary = (detailed_analysis or {}).get("dataSummary") or {}
    start, end = summary.get("startDate"), summary.get("endDate")
    if not start or not end:
        return None
    try:
        start_dt, end_dt = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        return None
    if (start_dt.year, start_dt.month) == (end_dt.year, end_dt.month):
        return f"{start_dt.year}년 {start_dt.month}월"
    if start_dt.year == end_dt.year:
        return f"{start_dt.year}년 {start_dt.month}월~{end_dt.month}월"
    return f"{start_dt.year}년 {start_dt.month}월~{end_dt.year}년 {end_dt.month}월"


# POS 분석 기간을 알 수 없을 때(detailed_analysis 없이 trdar_cd만으로 호출된 경우)의
# 최후 폴백 — 20261 같은 원시 코드 대신 "2026년 1분기"로 보여준다.
def _format_quarter_label(기준분기: int | None) -> str | None:
    if not 기준분기:
        return None
    year, quarter = divmod(int(기준분기), 10)
    return f"{year}년 {quarter}분기"


def _to_response(thread_id: str, values: dict, interrupt_value: dict | None) -> dict:
    payload = dict(values)
    payload["상태"] = payload.pop("status", "알 수 없음")
    payload["thread_id"] = thread_id
    payload["대기중_승인"] = interrupt_value
    target = (payload.get("diagnosis") or {}).get("대상", {})
    trdar_name = target.get("상권명") or payload.get("trdar_cd") or "상권 미지정"
    svc_name = target.get("업종명") or payload.get("svc_induty_cd") or "업종 미지정"
    store_id = payload.get("store_id")
    기준분기 = target.get("기준분기")
    분석기간 = _format_analysis_period(payload.get("detailed_analysis")) or _format_quarter_label(기준분기)
    표시명 = f"{trdar_name} / {svc_name}"
    if 분석기간:
        표시명 += f" · {분석기간}"
    표시명 += f" (매장 ID: {store_id or '미지정'})"
    payload["대상_매장"] = {
        "표시명": 표시명,
        "매장_ID": store_id,
        "상권코드": payload.get("trdar_cd"),
        "상권명": trdar_name,
        "업종코드": payload.get("svc_induty_cd"),
        "업종명": svc_name,
        "분석기간": 분석기간,
        "기준분기": 기준분기,
    }
    return payload


# 새 에이전트 실행을 시작하고 승인 대기 상태까지 진행한다
def start_agent_run(initial_state: dict) -> dict:

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


def _assert_owner(values: dict, user_id: str | None) -> None:
    owner = values.get("user_id")
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="해당 에이전트 실행에 접근할 권한이 없습니다")


@router.post("/agent-runs", deprecated=True)
def create_agent_run(payload: AgentRunRequest) -> dict:
    initial_state = {
        "user_id": payload.user_id,
        "store_id": payload.store_id,
        "trdar_cd": payload.trdar_cd,
        "svc_induty_cd": payload.svc_induty_cd,
        "yyqu_cd": payload.yyqu_cd,
        "warnings": [],
    }
    return start_agent_run(initial_state)


@router.get("/agent-runs/{thread_id}")
def get_agent_run(thread_id: str, x_user_id: str | None = Header(None, alias="X-User-Id")) -> dict:
    result = read_agent_run(thread_id)
    _assert_owner(result, x_user_id)
    return result


@router.post("/agent-runs/{thread_id}/resume")
def resume_agent_run(
    thread_id: str, payload: AgentRunResumeRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict:
    current = read_agent_run(thread_id)
    _assert_owner(current, x_user_id)
    resume_payload = {"결정": payload.decision}
    selected_action = payload.selected_action or payload.modification_plan
    if selected_action is not None:
        # edit뿐 아니라 approve와 함께 보내도 선택 방안을 먼저 적용한 뒤
        # 해당 방안의 검증 결과와 리포트를 생성한다.
        resume_payload["선택_방안"] = selected_action

    return continue_agent_run(thread_id, resume_payload)
