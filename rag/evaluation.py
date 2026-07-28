# RAG 리포트 품질 채점 모듈 — 검색(retriever.py)/생성(generator.py)과는 분리된 채점 전용 모듈.
# 검색 단계(근거 관련도)와 생성 단계(핵심정보·근거일치)를 각각 독립적으로 LLM에게 채점시킨다.
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

JudgeFn = Callable[[str], str]

_JUDGE_SYSTEM_PROMPT = (
    "당신은 대응방안 추천 시스템의 근거·리포트 품질을 채점하는 깐깐한 심사자다. "
    "반드시 지정된 JSON 형식으로만 답하고, 그 외 텍스트는 출력하지 않는다."
)


# OpenAI 호출(JSON 강제). 키는 환경변수 OPENAI_API_KEY 사용. TPM/RPM rate limit(429)은
# 지수 백오프로 재시도한다(배치 스크립트가 방안 12개를 연속 호출하면 낮은 티어 한도에 쉽게 걸림).
def _call_openai_judge(
    prompt: str, model: str = "gpt-4.1", temperature: float = 0.0, max_retries: int = 5,
) -> str:

    from openai import OpenAI, RateLimitError

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or "{}"
        except RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
    return "{}"  # 도달하지 않음(위에서 raise 또는 return)


# 특정 모델에 고정된 judge 콜러블을 만든다(배치 스크립트에서 --model 옵션 지원용)
def make_openai_judge(model: str = "gpt-4.1") -> JudgeFn:
    return lambda prompt: _call_openai_judge(prompt, model=model)


def _parse_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


@dataclass
class RetrievalJudgment:
    판정: str  # "관련" | "부분관련" | "무관"
    사유: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"판정": self.판정, "사유": self.사유}


@dataclass
class GenerationJudgment:
    핵심정보포함: str  # "충분" | "부분" | "누락"
    근거일치: str  # "일치" | "부분일치" | "불일치"
    사유: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"핵심정보포함": self.핵심정보포함, "근거일치": self.근거일치, "사유": self.사유}


def _format_evidence(evidence: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("[방향성 근거]")
    direction_refs = evidence.get("direction_refs") or []
    if not direction_refs:
        lines.append("(없음)")
    for r in direction_refs:
        lines.append(f"- ({r['doc_id']} p{r['page']}, {r['tier_label']}) {r['text'][:250]}")
    lines.append("")
    lines.append("[허용 수치]")
    allowed_numbers = evidence.get("allowed_numbers") or []
    if not allowed_numbers:
        lines.append("(없음)")
    for a in allowed_numbers:
        lines.append(f"- {a['value']} | {a['sentence'][:250]}")
    return "\n".join(lines)


# 검색된 근거(evidence) 자체가 이 방안과 관련 있는지 채점 — 생성 이전 단계
def score_retrieval_relevance(
    evidence: dict[str, Any], action_name: str, judge: JudgeFn | None = None,
) -> RetrievalJudgment:

    judge = judge or _call_openai_judge
    prompt = (
        "아래 [대응방안]에 대해 검색된 근거가 실제로 관련 있는지 판정하라.\n\n"
        f"[대응방안] {action_name}\n\n"
        f"{_format_evidence(evidence)}\n\n"
        "판정 기준:\n"
        "- '관련': 근거가 이 대응방안의 효과나 실행 방식을 직접 다룬다.\n"
        "- '부분관련': 근거가 방안과 인접한 주제이지만 직접적이지 않다"
        "(예: 다른 방안 효과, 일반적 매장 운영).\n"
        "- '무관': 근거가 이 방안과 실질적 연관이 없다. 근거가 전혀 없는 경우도 '무관'으로 판정한다.\n\n"
        "다음 JSON 형식으로만 답하라: "
        '{"판정": "관련" | "부분관련" | "무관", "사유": "1~2문장"}'
    )
    parsed = _parse_json(judge(prompt))
    return RetrievalJudgment(
        판정=str(parsed.get("판정", "무관")),
        사유=str(parsed.get("사유", "")),
        raw=parsed,
    )


# 생성된 리포트가 핵심정보를 담고 근거를 벗어나지 않았는지 채점 — 생성 이후 단계
def score_report_quality(
    report_text: str, evidence: dict[str, Any], action_name: str, judge: JudgeFn | None = None,
) -> GenerationJudgment:

    judge = judge or _call_openai_judge
    prompt = (
        "아래 [리포트]가 근거를 벗어나지 않고 핵심 정보를 담고 있는지 판정하라.\n\n"
        f"[대응방안] {action_name}\n\n"
        f"{_format_evidence(evidence)}\n\n"
        f"[리포트]\n{report_text}\n\n"
        "판정 기준:\n"
        "- 핵심정보포함: 대응방안 요약·매장 상황·권장 실행방안·적용 시 한계·결론이 "
        "실질적 내용으로 채워졌는가('충분'/'부분'/'누락').\n"
        "- 근거일치: 리포트의 주장·수치가 제공된 근거의 범위를 벗어나지 않는가"
        "(추론·과장·인과관계 왜곡 없음, '일치'/'부분일치'/'불일치').\n\n"
        "다음 JSON 형식으로만 답하라: "
        '{"핵심정보포함": "충분"|"부분"|"누락", "근거일치": "일치"|"부분일치"|"불일치", "사유": "1~2문장"}'
    )
    parsed = _parse_json(judge(prompt))
    return GenerationJudgment(
        핵심정보포함=str(parsed.get("핵심정보포함", "누락")),
        근거일치=str(parsed.get("근거일치", "불일치")),
        사유=str(parsed.get("사유", "")),
        raw=parsed,
    )
