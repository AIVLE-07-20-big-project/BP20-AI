# 대상 경로(AI 레포): app/sales_target/critic_agent.py
#
# 배치 검수 에이전트 — 고도화 설계 4절 구현(신규가맹점_영업타겟추천_AI_Agent_고도화_설계.md).
# 스코어링·리뷰 매칭이 끝난 ranked 목록을 관리자 승인 화면에 올리기 전에 검수한다.
#
# 최초 설계(후보 전체 데이터를 프롬프트에 다 넣고 JSON 하나로 요약/플래그를 뽑는 단발 호출)는
# 자체 재검토에서 "generate_pitch와 구조적으로 동일한 단발 생성이라 에이전트가 아니다"로
# 불합격 판정을 받았다(설계 문서 9절). 그래서 후보 전체를 처음부터 안 주고, LLM이 의심 가는
# 후보만 get_score_breakdown으로 상세 조회해야 축별 점수를 볼 수 있게 바꿨다 — 어떤 후보를
# 더 볼지, 언제 끝낼지를 LLM이 도구 호출로 직접 정하는 구조다.
#
# review_matching_agent.py와 달리 순수 계산(pandas)만 하고 네트워크 호출이 없어 동기 함수로
# 둔다(pitch.py의 generate_sales_pitch()와 동일한 패턴 — graph.py에서 _run_async() 없이 직접 호출).

from __future__ import annotations

import json
import math
import os

import pandas as pd

_SCORE_COLS = ["growth_score", "traffic_score", "review_score", "similarity_score", "final_score"]

_SYSTEM_PROMPT = (
    "너는 신규 가맹점 영업 타겟 배치를 관리자에게 넘기기 전에 검수하는 에이전트다. "
    "이름/순위/종합점수만 보고 점수 구성이 의심스러운(예: 종합점수는 높은데 다른 지표는 안 그런) "
    "후보가 있으면 get_score_breakdown으로 상세를 확인해라. 근거가 확보된 후보만 "
    "flag_candidate로 표시해라(최대 5건). 다 봤으면 finish_review로 요약문을 제출해라. "
    "제공된 숫자 밖의 통계나 사실을 새로 만들어내지 마라."
)

_FALLBACK_SUMMARY = "자동 요약 생성 실패"
_MAX_LLM_TURNS = 15  # LLM 호출(턴) 상한 — 한 턴에 도구를 여러 개 부를 수 있어 실제 도구 호출 수와는 다르다
_MAX_FLAGS = 5

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_score_breakdown",
            "description": "특정 후보의 축별 원점수와 이번 배치 중앙값을 상세 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {"bizesNm": {"type": "string"}},
                "required": ["bizesNm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_candidate",
            "description": "이 후보를 관리자가 한 번 더 확인해야 할 후보로 표시한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bizesNm": {"type": "string"},
                    "reason": {"type": "string", "description": "get_score_breakdown으로 확인한 근거를 포함해야 함"},
                },
                "required": ["bizesNm", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_review",
            "description": "검수를 마치고 배치 요약문을 제출한다.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
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


def _safe_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _light_overview(ranked_df: pd.DataFrame) -> str:
    lines = [
        f"{i + 1}. {row.get('bizesNm')} (final_score={_safe_float(row.get('final_score'))})"
        for i, row in ranked_df.reset_index(drop=True).iterrows()
    ]
    return "\n".join(lines)


def _score_breakdown(ranked_df: pd.DataFrame, bizes_nm: str) -> dict:
    matched = ranked_df[ranked_df["bizesNm"] == bizes_nm]
    if matched.empty:
        return {"error": f"'{bizes_nm}' 후보를 찾을 수 없음"}
    row = matched.iloc[0]
    medians = ranked_df[_SCORE_COLS].median(numeric_only=True)
    return {
        "bizesNm": bizes_nm,
        "scores": {c: _safe_float(row.get(c)) for c in _SCORE_COLS},
        "batch_median": {c: _safe_float(medians.get(c)) for c in _SCORE_COLS},
    }


def review_batch(ranked: list[dict], llm=None) -> dict:
    """ranked(top_n 후보 목록)를 검수해 배치 요약문과 주목 후보 목록을 만든다.

    반환: {"summary": str, "flagged": [{"bizesNm": str, "reason": str}, ...]}

    LLM 실패(오류, 도구 호출 파싱 실패 등) 시 검수를 건너뛴 것으로 보고 폴백값을 반환한다 —
    검수 실패가 배치 승인 자체를 막으면 안 된다.
    """
    if not ranked:
        return {"summary": "후보가 없어 검수를 건너뜀", "flagged": []}

    llm = llm or _call_openai_agent
    ranked_df = pd.DataFrame(ranked)
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"[후보 목록 — 이름/순위/종합점수]\n{_light_overview(ranked_df)}"},
    ]
    flagged: list[dict] = []

    try:
        for _ in range(_MAX_LLM_TURNS):
            message = llm(messages, TOOLS)
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                break

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": _tool_call_dicts(tool_calls),
            })

            finished_summary = None
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")

                if name == "finish_review":
                    finished_summary = args.get("summary") or _FALLBACK_SUMMARY
                elif name == "flag_candidate":
                    if len(flagged) < _MAX_FLAGS:
                        flagged.append({"bizesNm": args.get("bizesNm"), "reason": args.get("reason")})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": "표시됨"})
                elif name == "get_score_breakdown":
                    result = _score_breakdown(ranked_df, args.get("bizesNm", ""))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            if finished_summary is not None:
                return {"summary": finished_summary, "flagged": flagged}

        return {"summary": _FALLBACK_SUMMARY, "flagged": flagged}
    except Exception as exc:
        print(f"[critic_agent] 배치 검수 실패, 폴백 사용: {type(exc).__name__}: {exc}")
        return {"summary": _FALLBACK_SUMMARY, "flagged": []}
