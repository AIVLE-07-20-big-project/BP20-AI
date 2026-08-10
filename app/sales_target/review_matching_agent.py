# 대상 경로(AI 레포): app/sales_target/review_matching_agent.py
#
# 리뷰 매칭 에이전트 — 고도화 설계 3절 구현(신규가맹점_영업타겟추천_AI_Agent_고도화_설계.md).
# google_places.fetch_review_stats()는 후보마다 "상호명+주소" 문자열 하나로 딱 한 번만 검색하고
# 실패하면 포기한다. 여기서는 LLM이 도구 호출(search_google_places/accept_match/give_up)로
# 검색어를 스스로 조정하며 최대 max_attempts번 재시도하는 ReAct 루프를 돈다.
#
# 반환 컬럼 구조는 fetch_review_stats()와 동일하게 맞춰서, pipeline.apply_review_scores()에
# 그대로 넘길 수 있다 — "리뷰 데이터를 어떻게 찾을지"만 에이전트로 바뀌고 "찾은 데이터를 점수로
# 바꾸는 계산"은 그대로 결정론으로 남는다(설계 문서 원칙 1).
#
# 한 후보의 매칭 루프 중 예외(OpenAI 오류, JSON 파싱 실패 등)가 나면 그 후보만 "매칭 실패"로
# 처리하고 계속 진행한다(pitch.py의 "LLM 실패가 파이프라인을 막으면 안 된다" 원칙과 동일).

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import httpx
import pandas as pd

from app.sales_target.google_places import fetch_place_details, search_places

_SYSTEM_PROMPT = (
    "너는 실제 업장을 Google Places에서 찾는 에이전트다. 검색 결과가 없거나 확신이 안 서면 "
    "건물 내 호수/층수를 빼거나 건물명 기준으로 바꾸는 등 검색어를 스스로 조정해 다시 검색해라. "
    "확신이 서면 accept_match로 확정하고, 최대 {max_attempts}번 검색해도 못 찾겠으면 give_up으로 "
    "이유를 남기고 마무리해라. 다른 설명 없이 도구만 호출해라."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_google_places",
            "description": "검색어로 Google Places에서 업체 후보 목록(place_id/이름/주소)을 찾는다. 못 찾으면 빈 목록을 반환한다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색할 상호명+주소 문자열"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accept_match",
            "description": "지금까지 검색된 결과 중 하나를 이 후보의 최종 매칭으로 확정한다.",
            "parameters": {
                "type": "object",
                "properties": {"place_id": {"type": "string"}},
                "required": ["place_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "더 시도해도 이 후보를 찾을 수 없다고 판단해 매칭 실패로 마무리한다.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


def _call_openai_agent(messages: list[dict], tools: list[dict]):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4.1", temperature=0.2, messages=messages, tools=tools, tool_choice="auto"
    )
    return resp.choices[0].message


def _tool_call_dicts(tool_calls) -> list[dict]:
    return [
        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
        for tc in tool_calls
    ]


def _parse_rfc3339(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _growth_trend(publish_times: list[float]) -> float | None:
    if len(publish_times) < 2:
        return None
    ordered = sorted(publish_times)
    gaps_days = [(b - a) / 86400 for a, b in zip(ordered, ordered[1:])]
    return -(sum(gaps_days) / len(gaps_days))


async def _match_one(
    client: httpx.AsyncClient,
    api_key: str,
    llm,
    business_name: str,
    address: str,
    max_attempts: int,
) -> str | None:
    """후보 1건에 대해 도구 호출 루프를 돌려 최종 place_id(없으면 None)를 반환한다.

    루프 상한은 max_attempts(검색 횟수) + 2(여유 턴 — 마지막 검색 결과를 보고 accept_match/give_up을
    부를 한 번, 그리고 LLM이 즉시 마무리하지 못했을 때를 대비한 한 번 더)로 둔다. 이 상한에 도달할
    때까지도 accept_match/give_up을 안 부르면 강제로 매칭 실패 처리한다 — LLM이 검색만 반복하며
    끝없이 도는 것을 막는 안전장치."""
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(max_attempts=max_attempts)},
        {"role": "user", "content": f"상호명: {business_name}\n주소: {address}"},
    ]
    candidates_seen: dict[str, dict] = {}
    searches_done = 0

    for _ in range(max_attempts + 2):
        message = llm(messages, TOOLS)
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return None  # 도구 호출 없이 끝냄 = 스스로 마무리하지 못한 것으로 간주, 매칭 실패

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": _tool_call_dicts(tool_calls),
        })

        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")

            if name == "accept_match":
                place_id = args.get("place_id")
                return place_id if place_id in candidates_seen else None

            if name == "give_up":
                return None

            if name == "search_google_places":
                searches_done += 1
                if searches_done > max_attempts:
                    result = {"places": [], "error": "최대 검색 횟수 초과 — accept_match 또는 give_up을 호출하세요"}
                else:
                    found = await search_places(client, api_key, args.get("query", ""))
                    candidates_seen.update({c["place_id"]: c for c in found})
                    result = {"places": found}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

    return None  # 루프 상한 도달 — 강제 매칭 실패


async def collect_review_stats_agentic(
    candidates: pd.DataFrame,
    api_key: str,
    llm=None,
    max_attempts: int = 2,
    business_name_col: str = "bizesNm",
    address_col: str = "rdnmAdr",
) -> pd.DataFrame:
    """리뷰 매칭 에이전트로 candidates(top_n)의 리뷰 데이터를 수집한다.

    반환 컬럼은 google_places.fetch_review_stats()와 동일(address_col/review_count/avg_rating/
    days_since_latest_review/review_growth_trend) — pipeline.apply_review_scores()에 그대로
    넘길 수 있다.
    """
    llm = llm or _call_openai_agent
    now = datetime.now(timezone.utc).timestamp()
    rows = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for _, row in candidates.iterrows():
            name, address = row.get(business_name_col), row.get(address_col)
            if pd.isna(name) or pd.isna(address) or not name or not address:
                continue
            try:
                place_id = await _match_one(client, api_key, llm, str(name), str(address), max_attempts)
                if not place_id:
                    continue
                details = await fetch_place_details(client, api_key, place_id)
            except Exception as exc:  # noqa: BLE001 — 한 후보 실패가 배치 전체를 막으면 안 됨
                print(f"[review_matching_agent] {name} 매칭 실패, 건너뜀: {type(exc).__name__}: {exc}")
                continue

            rating, review_count = details.get("rating"), details.get("userRatingCount")
            if rating is None or review_count is None:
                continue

            publish_times = [
                _parse_rfc3339(r["publishTime"]) for r in (details.get("reviews") or []) if r.get("publishTime")
            ]
            latest_time = max(publish_times, default=None)
            rows.append({
                address_col: address,
                "review_count": review_count,
                "avg_rating": rating,
                "days_since_latest_review": (now - latest_time) / 86400 if latest_time else None,
                "review_growth_trend": _growth_trend(publish_times),
            })

    return pd.DataFrame(rows)
