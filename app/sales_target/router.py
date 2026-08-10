"""신규 가맹점 영업 타겟 추천 — 그래프 실행 API.

POST /sales-targets/generate                실행 시작 (prepare_candidates까지 진행 후 대기)
GET  /sales-targets/jobs/{thread_id}         현재 상태 조회
POST /sales-targets/jobs/{thread_id}/approve 승인 -> generate_pitch -> finalize
POST /sales-targets/jobs/{thread_id}/reject  반려 -> discard

agent_runs.py(app/routers/agent_runs.py)의 start/read/continue 패턴을 그대로 따른다.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.sales_target.graph import get_graph, new_thread_id

router = APIRouter(prefix="/sales-targets", tags=["신규 가맹점 영업 타겟"])


def _to_response(thread_id: str, values: dict, interrupt_value: dict | None) -> dict:
    payload = dict(values)
    payload["상태"] = payload.pop("status", "알 수 없음")
    payload["thread_id"] = thread_id
    payload["대기중_승인"] = interrupt_value
    return payload


def start_sales_target_run(initial_state: dict) -> dict:
    thread_id = new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(initial_state, config=config)
    interrupts = result.pop("__interrupt__", None)
    interrupt_value = interrupts[0].value if interrupts else None
    return _to_response(thread_id, result, interrupt_value)


def read_sales_target_run(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"sales-target 실행을 찾을 수 없음: {thread_id}")

    interrupt_value = None
    for task in snapshot.tasks:
        if task.interrupts:
            interrupt_value = task.interrupts[0].value
            break
    return _to_response(thread_id, snapshot.values, interrupt_value)


def continue_sales_target_run(thread_id: str, approved: bool) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"sales-target 실행을 찾을 수 없음: {thread_id}")
    if not any(task.interrupts for task in snapshot.tasks):
        raise HTTPException(status_code=409, detail="현재 승인 대기 상태가 아닙니다")

    result = get_graph().invoke(Command(resume={"approved": approved}), config=config)
    interrupts = result.pop("__interrupt__", None)
    interrupt_value = interrupts[0].value if interrupts else None
    return _to_response(thread_id, result, interrupt_value)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise HTTPException(status_code=500, detail=f"환경변수 {name}가 설정되지 않았습니다.")
    return value


class GenerateRequest(BaseModel):
    top_n: int = 20


@router.post("/generate")
def generate(payload: GenerateRequest) -> dict:
    from scripts.collection.collectors import latest_quarter_codes

    initial_state = {
        "store_registry_api_key": _require_env("PUBLIC_DATA_STORE_API_KEY"),
        "seoul_api_key": _require_env("SEOUL_API_KEY"),
        "backend_base_url": os.environ.get("BACKEND_INTERNAL_BASE_URL", "http://localhost:8080"),
        "backend_internal_api_key": _require_env("INTERNAL_API_KEY"),
        # 6단계(리뷰 2단계 반영). 필수 아님 — 없으면 review_score는 채워지지 않고(NaN),
        # final_score는 성장률·유동인구·유사도 3개 지표만으로 계산된다.
        "google_places_api_key": os.environ.get("GOOGLE_PLACES_API_KEY", ""),
        # district_metrics는 상권별 최근 2개 분기만 쓰므로 21개 전체 대신 최근 4개(1년치, 여유분
        # 포함)만 받는다 — 배치 소요시간이 여기서 가장 크게 줄어든다.
        "quarter_codes": latest_quarter_codes(),
        "top_n": payload.top_n,
        "source_batch_id": f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}",
        "warnings": [],
    }
    return start_sales_target_run(initial_state)


@router.get("/jobs/{thread_id}")
def get_job(thread_id: str) -> dict:
    return read_sales_target_run(thread_id)


@router.post("/jobs/{thread_id}/approve")
def approve_job(thread_id: str) -> dict:
    return continue_sales_target_run(thread_id, approved=True)


@router.post("/jobs/{thread_id}/reject")
def reject_job(thread_id: str) -> dict:
    return continue_sales_target_run(thread_id, approved=False)
