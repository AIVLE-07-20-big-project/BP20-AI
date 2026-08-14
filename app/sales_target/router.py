"""신규 가맹점 영업 타겟 추천 — 그래프 실행 API.

POST /sales-targets/generate                실행 시작 (prepare_candidates까지 진행 후 대기)
GET  /sales-targets/jobs/{thread_id}         현재 상태 조회
POST /sales-targets/jobs/{thread_id}/approve 승인 -> generate_pitch -> finalize
POST /sales-targets/jobs/{thread_id}/reject  반려 -> discard

agent_runs.py(app/routers/agent_runs.py)의 start/read/continue 패턴을 그대로 따른다.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app.sales_target.graph import get_graph, new_thread_id

router = APIRouter(prefix="/sales-targets", tags=["신규 가맹점 영업 타겟"])
logger = logging.getLogger(__name__)

# /generate가 그래프 실행을 백그라운드로 넘긴 뒤 즉시 thread_id를 반환하는 방식으로 바뀌면서
# (CloudFront/ALB 타임아웃보다 파이프라인이 더 오래 걸려 FE에 조기 에러가 뜨던 문제 대응),
# "백그라운드로 넘어갔지만 아직 첫 체크포인트가 안 찍힌" 짧은 공백 구간을 메워주는 용도.
# 이 dict가 없으면 그 사이 GET /jobs/{thread_id}가 404를 내려서 FE 폴링이 곧바로 에러로 빠진다.
#
# 주의: 프로세스 로컬(in-memory) 상태다. bp20-prod-fastapi가 태스크 1개로 운영되는 동안에는
# 문제없지만(2026-08 기준 Cloud Map에 인스턴스 1개만 등록됨), 나중에 태스크를 2개 이상으로
# 늘리면 폴링 요청이 다른 인스턴스로 갈 수 있어 이 dict로는 부족해진다 — 그때는
# SALES_TARGET_CHECKPOINT_DB_URL(Postgres)처럼 공유 저장소로 옮겨야 한다.
_INFLIGHT: dict[str, str] = {}  # thread_id -> "처리중" | "오류"
_INFLIGHT_LOCK = threading.Lock()

_PROCESSING_STATUS_MESSAGE = (
    "처리 중 — 공공데이터 수집 및 스코어링이 진행 중입니다. 완료되면 자동으로 승인 대기로 바뀝니다."
)


def _to_response(thread_id: str, values: dict, interrupt_value: dict | None) -> dict:
    payload = dict(values)
    payload["상태"] = payload.pop("status", "알 수 없음")
    payload["thread_id"] = thread_id
    payload["대기중_승인"] = interrupt_value
    return payload


def start_sales_target_run(initial_state: dict, thread_id: str | None = None) -> dict:
    thread_id = thread_id or new_thread_id()
    config = {"configurable": {"thread_id": thread_id}}
    result = get_graph().invoke(initial_state, config=config)
    interrupts = result.pop("__interrupt__", None)
    interrupt_value = interrupts[0].value if interrupts else None
    return _to_response(thread_id, result, interrupt_value)


def _run_generate_in_background(thread_id: str, initial_state: dict) -> None:
    try:
        start_sales_target_run(initial_state, thread_id=thread_id)
    except Exception:
        logger.exception("영업 타겟 배치 백그라운드 실행 실패 (thread_id=%s)", thread_id)
        with _INFLIGHT_LOCK:
            _INFLIGHT[thread_id] = "오류"
        return
    with _INFLIGHT_LOCK:
        _INFLIGHT.pop(thread_id, None)


def read_sales_target_run(thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = get_graph().get_state(config)
    if not snapshot.values:
        with _INFLIGHT_LOCK:
            inflight_status = _INFLIGHT.get(thread_id)
        if inflight_status == "처리중":
            return {
                "thread_id": thread_id,
                "상태": _PROCESSING_STATUS_MESSAGE,
                "대기중_승인": None,
                "진행중": True,
            }
        if inflight_status == "오류":
            raise HTTPException(
                status_code=500,
                detail="영업 타겟 배치 실행 중 오류가 발생했습니다. 다시 시도해 주세요.",
            )
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
def generate(payload: GenerateRequest, background_tasks: BackgroundTasks) -> dict:
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

    # 실제 파이프라인(공공데이터 수집~스코어링)은 몇 분씩 걸릴 수 있어, 이 요청 안에서 끝까지
    # 기다리면 CloudFront/ALB 타임아웃에 걸려 FE에 "요청 처리 중 오류가 발생했습니다"가 먼저
    # 뜨는 문제가 있었다(백엔드는 실제로는 끝까지 계속 실행됨). thread_id만 먼저 만들어 즉시
    # 돌려주고, 실제 그래프 실행은 백그라운드로 넘긴다 — FE는 이 thread_id로 폴링한다.
    thread_id = new_thread_id()
    with _INFLIGHT_LOCK:
        _INFLIGHT[thread_id] = "처리중"
    background_tasks.add_task(_run_generate_in_background, thread_id, initial_state)
    return {
        "thread_id": thread_id,
        "상태": _PROCESSING_STATUS_MESSAGE,
        "대기중_승인": None,
        "진행중": True,
    }


@router.get("/jobs/{thread_id}")
def get_job(thread_id: str) -> dict:
    return read_sales_target_run(thread_id)


@router.post("/jobs/{thread_id}/approve")
def approve_job(thread_id: str) -> dict:
    return continue_sales_target_run(thread_id, approved=True)


@router.post("/jobs/{thread_id}/reject")
def reject_job(thread_id: str) -> dict:
    return continue_sales_target_run(thread_id, approved=False)
