"""신규 가맹점 영업 타겟 추천 파이프라인 — LangGraph StateGraph.

AI_Agent_전환_가이드라인.md 1단계 구현. 계산 로직(matching/geo_matching/district_metrics/
scoring/hero_store)은 전혀 안 바뀐다 — pipeline.py의 generate_sales_targets()를 그대로
호출한다. 바뀐 건 오케스트레이션 계층뿐이다.

노드 순서(고도화 설계 문서 5절 — review_matching_agent/critic_agent 2개 신규):

    prepare_candidates ──▶ review_matching_agent ──▶ critic_agent ──▶ review(interrupt)
                                                                              │
                                                              ┌───────────────┴───────────────┐
                                                              ▼                               ▼
                                                       generate_pitch ──▶ finalize ──▶ END   discard ──▶ END

가이드라인 문서는 collect / match_and_score / select_hero를 별도 노드 3개로 나눴었는데,
실제로 만들면서 하나(prepare_candidates)로 합쳤다. 이유:
  1) 이 세 단계 사이엔 사람이 개입할 지점이 없다 — interrupt는 스코어링이 다 끝난 뒤에만 필요하다.
  2) candidate_registry가 서울 전체 상가업소 기준 약 53만 행(2026-07-29 첫 실행 실측치)이다.
     노드마다 나눠서 state로 넘기면 LangGraph가 노드 전환마다 체크포인트에 그 무거운 DataFrame을
     통째로 직렬화하게 된다. 하나로 합치면 무거운 중간 데이터는 이 함수의 지역 변수로만 존재하고,
     체크포인트에는 top_n으로 이미 추려진 최종 랭킹(records, 작음)만 남는다.

review_matching_agent/critic_agent는 반대로 별도 노드로 나눴다 — 이 둘이 다루는 건 이미 top_n으로
추려진 작은 목록(보통 20~200행)이라 위 1)/2) 이유가 적용되지 않고, 오히려 나눠두면 에이전트가
지금 뭘 하고 있는지가 그래프 상태로 그대로 드러난다(신규가맹점_영업타겟추천_AI_Agent_고도화_설계.md
2절 참고).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from functools import lru_cache
from typing import Literal, TypedDict
from uuid import uuid4

import pandas as pd
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.core.config import SALES_TARGET_CHECKPOINT_DB_URL, SALES_TARGET_GRAPH_DB
from app.sales_target.backend_client import BackendStoreRegistryClient, SalesTargetIngestClient
from app.sales_target.critic_agent import review_batch
from app.sales_target.district_metrics import (
    aggregate_district_quarterly,
    latest_quarter_values,
    latest_vs_previous_growth,
)
from app.sales_target.google_places import fetch_review_stats
from app.sales_target.hero_store import select_hero_stores
from app.sales_target.pipeline import apply_review_scores, generate_sales_targets, to_bulk_upsert_items
from app.sales_target.pitch import generate_sales_pitch
from app.sales_target.review_matching_agent import collect_review_stats_agentic
from app.sales_target.scoring import foot_traffic_index

# pipeline.py의 collect_and_generate()와 동일 — 서울시 상권분석서비스 공공데이터 컬럼명.
_FIELD_NAMES = {
    "sales_trdar_col": "TRDAR_CD",
    "sales_quarter_col": "STDR_YYQU_CD",
    "sales_amount_col": "THSMON_SELNG_AMT",
    "traffic_trdar_col": "TRDAR_CD",
    "traffic_quarter_col": "STDR_YYQU_CD",
    "traffic_morning_col": "TMZON_06_11_FLPOP_CO",
    "traffic_lunch_col": "TMZON_11_14_FLPOP_CO",
    "traffic_afternoon_col": "TMZON_14_17_FLPOP_CO",
    "store_trdar_col": "TRDAR_CD",
    "store_quarter_col": "STDR_YYQU_CD",
    "store_count_col": "STOR_CO",
}


class SalesTargetState(TypedDict, total=False):
    # 입력 (start 시점에 채워짐)
    store_registry_api_key: str
    seoul_api_key: str
    backend_base_url: str
    backend_internal_api_key: str
    quarter_codes: list[str]
    top_n: int | None
    source_batch_id: str
    # 6단계(리뷰 2단계 반영): 미설정/빈 문자열이면 리뷰 활성도 점수(review_score)는 계산되지 않고
    # NaN으로 남으며, final_score는 growth/traffic/similarity 3축만으로 계산된다.
    google_places_api_key: str | None

    # prepare_candidates 결과. DataFrame이 아니라 list[dict](records)로 저장한다 — 체크포인트
    # 직렬화를 단순/가볍게 유지하고, 관리자 승인 화면에 그대로 내려줄 JSON 모양과 일치시키기 위함.
    ranked: list[dict]

    # prepare_candidates에서 함께 뽑아둔 우수 가맹점 목록(1단계 근사 기준 통과분). generate_pitch
    # 단계에서 세일즈 피칭 문구의 참고 자료로 쓴다. 자사 가맹점의 부분집합이라 후보 registry(53만
    # 행)에 비하면 훨씬 작아서 체크포인트에 넣어도 부담이 없다.
    hero_stores: list[dict]

    # critic_agent 결과(고도화 설계 4절). review(interrupt) payload에 실려 관리자 승인 화면에
    # 노출된다.
    agent_review_summary: str
    agent_flagged_candidates: list[dict]

    # review(interrupt) 결과
    approval_status: Literal["approved", "rejected"] | None

    # finalize 결과
    push_result: dict | None

    status: str
    warnings: list[str]


def _run_async(coro):
    """동기 노드 함수 안에서 비동기 콜렉터/BE 클라이언트를 호출하기 위한 헬퍼.

    response/graph.py의 기존 노드들도 전부 sync라 이 그래프도 동일한 전제를 따른다: 이미 실행
    중인 이벤트 루프 안(async 라우트 핸들러 등)에서 부르면 RuntimeError가 난다. FastAPI의 sync
    `def` 라우트(스레드풀에서 실행)나 Celery task에서 호출하는 걸 전제로 한다.
    """
    return asyncio.run(coro)


def _prepare_candidates(state: SalesTargetState) -> dict:
    # 순환 import를 피하려고 지역 임포트(pipeline.py의 collect_and_generate()와 동일한 이유).
    from scripts.collection.collectors import (
        FootTrafficCollector,
        SalesEstimateCollector,
        StoreStatsCollector,
        TrdarBoundaryCollector,
    )
    from scripts.collection.store_registry_collector import StoreRegistryCollector

    warnings = list(state.get("warnings", []))
    f = _FIELD_NAMES

    # 배치가 어느 단계에서 오래 걸리는지(외부 API 응답 지연 vs 대용량 데이터 처리) 콘솔에서
    # 바로 보이게 각 수집 단계마다 소요 시간을 찍는다 — apis.data.go.kr/openapi.seoul.go.kr
    # 둘 다 이 파이프라인이 의존하는 서로 다른 외부 공공데이터 게이트웨이라, 전체가 "그냥 오래
    # 걸림"으로만 보이면 어느 쪽이 원인인지 알 수 없다.
    async def _timed(label: str, coro):
        start = time.monotonic()
        result = await coro
        elapsed = time.monotonic() - start
        size = len(result) if hasattr(result, "__len__") else "?"
        print(f"[prepare_candidates] {label}: {elapsed:.1f}초 (결과 {size}건)")
        return result

    async def _collect_all():
        # 서로 데이터 의존관계가 없는 7개 수집을 동시에 돌린다(기존엔 순차라 각 단계 시간이
        # 그대로 더해졌음). _timed가 단계별 소요시간을 각자 찍어주므로 어느 게 오래 걸리는지는
        # 여전히 로그로 구분된다 — 다만 이제는 "합"이 아니라 "가장 느린 하나"에 수렴한다.
        backend_client = BackendStoreRegistryClient(state["backend_base_url"], state["backend_internal_api_key"])
        ingest_client = SalesTargetIngestClient(state["backend_base_url"], state["backend_internal_api_key"])
        (
            candidate_registry, our_stores, rejected_addresses,
            trdar_boundary, sales_raw, traffic_raw, store_stats_raw,
        ) = await asyncio.gather(
            _timed(
                "store_registry(공공데이터포털/로컬 CSV)",
                StoreRegistryCollector(state["store_registry_api_key"]).fetch_sigungus(),
            ),
            _timed("BE 자사 가맹점 조회", backend_client.fetch_our_stores()),
            # 5단계(피드백 루프): 영업팀이 이미 EXCLUDED 처리한 후보는 다음 배치에서 다시 추천하지 않는다.
            _timed("BE 제외 주소 조회", ingest_client.fetch_excluded_addresses()),
            _timed("상권 경계(서울 열린데이터광장)", TrdarBoundaryCollector(state["seoul_api_key"]).fetch_all()),
            _timed(
                "매출 추정(서울 열린데이터광장)",
                SalesEstimateCollector(state["seoul_api_key"]).fetch_quarters(state["quarter_codes"]),
            ),
            _timed(
                "유동인구(서울 열린데이터광장)",
                FootTrafficCollector(state["seoul_api_key"]).fetch_quarters(state["quarter_codes"]),
            ),
            _timed(
                "점포수(서울 열린데이터광장)",
                StoreStatsCollector(state["seoul_api_key"]).fetch_quarters(state["quarter_codes"]),
            ),
        )
        return (
            candidate_registry, our_stores, rejected_addresses,
            trdar_boundary, sales_raw, traffic_raw, store_stats_raw,
        )

    _collect_start = time.monotonic()
    (
        candidate_registry, our_stores, rejected_addresses,
        trdar_boundary, sales_raw, traffic_raw, store_stats_raw,
    ) = _run_async(_collect_all())
    print(f"[prepare_candidates] 데이터 수집 전체: {time.monotonic() - _collect_start:.1f}초")

    # pipeline.collect_and_generate()와 동일한 분기: BE는 2026-08-03 기준 이미 리뷰 통계
    # (salesGrowthRate/reviewAvgRating/reviewRatingStd)를 내려준다 — 정상 상황에선 이 분기를
    # 안 타고 실값으로 우수 가맹점이 선정된다. 그래도 컬럼이 빠진 응답(빈 registry 등)에 대비해
    # 방어적으로 남겨둔다 — 그 경우 hero_stores=None, similarity_score는 placeholder(50.0).
    hero_stores = None
    required_hero_cols = {"salesGrowthRate", "reviewAvgRating", "reviewRatingStd"}
    if required_hero_cols.issubset(our_stores.columns):
        selected = select_hero_stores(our_stores)
        hero_stores = selected if not selected.empty else None
    else:
        warnings.append(
            "BE 응답에 리뷰 통계 컬럼이 없어 우수 가맹점 선정을 건너뜀 — similarity_score는 placeholder(50.0)"
        )

    sales_district_quarterly = aggregate_district_quarterly(
        sales_raw, f["sales_trdar_col"], f["sales_quarter_col"], f["sales_amount_col"]
    )
    district_sales_growth = latest_vs_previous_growth(
        sales_district_quarterly, f["sales_trdar_col"], f["sales_quarter_col"], f["sales_amount_col"]
    ).rename(columns={f["sales_trdar_col"]: "trdar_cd"})

    traffic_raw = traffic_raw.copy()
    traffic_raw["_traffic_index"] = foot_traffic_index(
        traffic_raw[f["traffic_morning_col"]].astype(float),
        traffic_raw[f["traffic_lunch_col"]].astype(float),
        traffic_raw[f["traffic_afternoon_col"]].astype(float),
    )
    traffic_district_quarterly = aggregate_district_quarterly(
        traffic_raw, f["traffic_trdar_col"], f["traffic_quarter_col"], "_traffic_index"
    )
    district_foot_traffic_index = latest_quarter_values(
        traffic_district_quarterly, f["traffic_trdar_col"], f["traffic_quarter_col"], "_traffic_index"
    ).rename(columns={f["traffic_trdar_col"]: "trdar_cd", "_traffic_index": "traffic_index"})

    store_district_quarterly = aggregate_district_quarterly(
        store_stats_raw, f["store_trdar_col"], f["store_quarter_col"], f["store_count_col"]
    )
    district_store_count = latest_quarter_values(
        store_district_quarterly, f["store_trdar_col"], f["store_quarter_col"], f["store_count_col"]
    ).rename(columns={f["store_trdar_col"]: "trdar_cd", f["store_count_col"]: "store_count"})

    _scoring_start = time.monotonic()
    ranked = generate_sales_targets(
        candidate_registry=candidate_registry,
        our_stores=our_stores,
        trdar_boundary=trdar_boundary,
        district_sales_growth=district_sales_growth,
        district_foot_traffic_index=district_foot_traffic_index,
        district_store_count=district_store_count,
        hero_stores=hero_stores,
        top_n=state.get("top_n"),
        rejected_addresses=rejected_addresses,
    )
    print(
        f"[prepare_candidates] 매칭/지오매칭/스코어링(후보 {len(candidate_registry)}건 처리): "
        f"{time.monotonic() - _scoring_start:.1f}초"
    )

    if ranked.empty:
        warnings.append("후보가 없음(전량 자사 가맹점이거나 공공데이터가 비어있음)")

    return {
        "ranked": ranked.to_dict("records"),
        "hero_stores": hero_stores.to_dict("records") if hero_stores is not None else [],
        "warnings": warnings,
        "status": "1차 스코어링 완료" if not ranked.empty else "종료: 후보 없음",
    }


def _route_after_prepare(state: SalesTargetState) -> str:
    return "review_matching_agent" if state.get("ranked") else END


def _review_matching_agent(state: SalesTargetState) -> dict:
    """고도화 설계 3절 구현: growth/traffic/similarity로 이미 추려진 top_n 후보에 한해서만
    리뷰 데이터를 채운다. OPENAI_API_KEY가 있으면 review_matching_agent(도구 호출 재시도 루프)를
    쓰고, 없거나 실패하면 기존 결정론적 google_places.fetch_review_stats()(단발 시도)로
    폴백한다 — 새 기능이 실패해도 6단계까지 만든 동작은 항상 보장된다."""
    ranked = pd.DataFrame(state.get("ranked", []))
    warnings = list(state.get("warnings", []))
    google_places_api_key = state.get("google_places_api_key")

    if not google_places_api_key:
        warnings.append("GOOGLE_PLACES_API_KEY 미설정 — 리뷰 활성도 점수 없이 성장률·유동인구·유사도 3개 지표로만 최종 점수 산출됨")
        return {"ranked": ranked.to_dict("records"), "warnings": warnings}

    if os.environ.get("OPENAI_API_KEY"):
        try:
            review_stats = _run_async(collect_review_stats_agentic(ranked, google_places_api_key))
        except Exception as exc:
            warnings.append(f"리뷰 매칭 에이전트 실패({type(exc).__name__}) — 단발 검색으로 폴백")
            review_stats = _run_async(fetch_review_stats(ranked, google_places_api_key))
    else:
        warnings.append("OPENAI_API_KEY 미설정 — 리뷰 매칭 에이전트 대신 단발 검색으로 폴백")
        review_stats = _run_async(fetch_review_stats(ranked, google_places_api_key))

    before_count = len(ranked)
    ranked = apply_review_scores(ranked, review_stats)
    excluded = before_count - len(ranked)
    if excluded > 0:
        warnings.append(f"구글 플레이스에서 매칭 안 된(리뷰 데이터 없는) 후보 {excluded}건을 제외함")
    return {"ranked": ranked.to_dict("records"), "warnings": warnings, "status": "리뷰 매칭 완료"}


def _critic_agent(state: SalesTargetState) -> dict:
    """고도화 설계 4절 구현: 관리자 승인 화면에 올리기 전에 배치를 검수한다."""
    result = review_batch(state.get("ranked", []))
    return {
        "agent_review_summary": result.get("summary", ""),
        "agent_flagged_candidates": result.get("flagged", []),
        "status": "배치 검수 완료 — 관리자 승인 대기",
    }


def _review(state: SalesTargetState) -> dict:
    decision = interrupt({
        "후보_리스트": state.get("ranked", []),
        "후보_수": len(state.get("ranked", [])),
        "주의사항": state.get("warnings", []),
        "에이전트_요약": state.get("agent_review_summary", ""),
        "주목_후보": state.get("agent_flagged_candidates", []),
    })
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else False
    status = "승인됨 — 피칭 문구 생성 중" if approved else "종료: 반려됨"
    return {"approval_status": "approved" if approved else "rejected", "status": status}


def _route_after_review(state: SalesTargetState) -> str:
    return "generate_pitch" if state.get("approval_status") == "approved" else "discard"


def _generate_pitch(state: SalesTargetState) -> dict:
    # 2단계(AI_Agent_전환_가이드라인.md §7) 구현: app/sales_target/pitch.py의 generate_sales_pitch()로
    # 후보별 세일즈 피칭 문구를 실제로 생성한다. hero_stores는 배치 공통 참고 자료로 모든 후보에
    # 동일하게 넘긴다(후보별 정밀 매칭 정보가 파이프라인에 없는 현재 한계 — pitch.py 참고).
    ranked = state.get("ranked", [])
    hero_stores = state.get("hero_stores", [])
    for row in ranked:
        row["sales_pitch"] = generate_sales_pitch(row, hero_stores)
    return {"ranked": ranked, "status": "피칭 문구 생성 완료 — BE 반영 중"}


def _finalize(state: SalesTargetState) -> dict:
    ranked_df = pd.DataFrame(state.get("ranked", []))
    items = to_bulk_upsert_items(ranked_df)
    client = SalesTargetIngestClient(state["backend_base_url"], state["backend_internal_api_key"])
    result = _run_async(client.push_bulk(state["source_batch_id"], items))
    return {"push_result": result, "status": "완료 — BE 반영됨"}


def _discard(state: SalesTargetState) -> dict:
    return {"status": "종료: 반려되어 BE에 반영하지 않음"}


# PostgresSaver.from_conn_string()은 제너레이터 기반 컨텍스트매니저(@contextmanager)라서,
# 연결을 실제로 쥐고 있는 건 그 제너레이터 프레임이다. _checkpointer() 안에서 __enter__()만
# 부르고 conn_cm 지역변수를 버리면(반환 직후 스코프 종료) 참조가 없어져 GC가 제너레이터를
# 정리하면서 finally 블록(=Connection.connect()의 __exit__)이 실행돼 연결이 곧바로 닫혀버린다
# (SqliteSaver(conn)처럼 평범한 생성자가 아니라 실제로 실전 Postgres에 붙여봐야만 드러나는
# 문제였다 — 모듈 전역에 붙잡아 둬서 GC 대상에서 빼야 한다).
_pg_conn_cm = None


@lru_cache(maxsize=1)
def _checkpointer():
    global _pg_conn_cm
    if SALES_TARGET_CHECKPOINT_DB_URL:
        # Postgres. 배치는 Celery 워커에서 실행되고 승인은 별도 프로세스(BE→AI API 요청)에서
        # 들어오므로, 다중 워커 환경에서도 안전한 PostgresSaver를 프로덕션 기본값으로 쓴다
        # (AI_Agent_전환_가이드라인.md 2.3절 근거).
        from langgraph.checkpoint.postgres import PostgresSaver

        _pg_conn_cm = PostgresSaver.from_conn_string(SALES_TARGET_CHECKPOINT_DB_URL)
        saver = _pg_conn_cm.__enter__()
        saver.setup()
        return saver

    # SALES_TARGET_CHECKPOINT_DB_URL이 없으면(로컬 개발) sqlite로 폴백한다.
    from langgraph.checkpoint.sqlite import SqliteSaver

    SALES_TARGET_GRAPH_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SALES_TARGET_GRAPH_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(SalesTargetState)
    g.add_node("prepare_candidates", _prepare_candidates)
    g.add_node("review_matching_agent", _review_matching_agent)
    g.add_node("critic_agent", _critic_agent)
    g.add_node("review", _review)
    g.add_node("generate_pitch", _generate_pitch)
    g.add_node("finalize", _finalize)
    g.add_node("discard", _discard)

    g.set_entry_point("prepare_candidates")
    g.add_conditional_edges(
        "prepare_candidates", _route_after_prepare, {END: END, "review_matching_agent": "review_matching_agent"}
    )
    g.add_edge("review_matching_agent", "critic_agent")
    g.add_edge("critic_agent", "review")
    g.add_conditional_edges(
        "review", _route_after_review, {"generate_pitch": "generate_pitch", "discard": "discard"}
    )
    g.add_edge("generate_pitch", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("discard", END)

    return g.compile(checkpointer=_checkpointer())


def get_checkpointer():
    """`_checkpointer()`의 공개 래퍼. 4단계 체크포인트 정리 작업(app/tasks/sales_target.py,
    run_sales_target_checkpoint_cleanup.py)처럼 그래프가 아니라 체크포인터 자체가 필요한
    외부 모듈은 밑줄 붙은 내부 함수 대신 이 함수를 통해 접근한다."""
    return _checkpointer()


def new_thread_id() -> str:
    return str(uuid4())
