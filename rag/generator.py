"""RAG 리포트 생성 모듈 (VSCode / Windows 앱용).

evidence_gate: 검색된 근거 밖의 수치가 리포트에 등장하지 못하게 막는다.
흐름은 build_prompt -> LLM 호출 -> verify_output -> (위반 시) 1회 재생성.

핵심 원칙(계획 §6):
  - vendor 수치는 '실측'이 아니라 문헌 참고값이므로 신뢰도 라벨을 낮춰 인용한다.
  - 근거가 없으면 숫자를 만들지 말고 방향성만 서술한다. 정직한 낮은 신뢰도 > 거짓 확신.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

DISCLAIMER = (
    "본 수치는 유사 사례 문헌에 근거한 참고값이며, 해당 매장의 실측 효과가 아닙니다. "
    "실제 결과는 상권·메뉴·시기에 따라 달라질 수 있습니다."
)

SYSTEM_PROMPT = "당신은 프랜차이즈·체인 기업의 지점/상권별 대응방안 리포트를 작성하는 분석가다. 주어진 근거 밖의 사실이나 수치를 결코 만들어내지 않는다."

_EXEMPT = re.compile(r"(19|20)\d{2}\s*년")
_NUM_RE = re.compile(
    r"[+\-]?\d[\d,]*(?:\.\d+)?\s?(?:%|퍼센트|원)"
    r"|[+\-]?\d[\d,]*(?:\.\d+)?배(?![달란])"
)


@dataclass
class VerifyResult:
    ok: bool
    violations: list[dict]
    disclaimer_present: bool
    numbers_found: list[str]

    def as_feedback(self) -> str:
        """재생성 시 LLM에 돌려줄 교정 지시문."""
        msgs = []
        if self.violations:
            bad = ", ".join(f"'{v['value']}'" for v in self.violations)
            msgs.append(
                f"다음 수치는 허용 목록에 없으므로 삭제하거나 허용된 수치로 교체하라: {bad}. "
                "추정·환산·반올림도 금지된다."
            )
        if not self.disclaimer_present:
            msgs.append(f"마지막에 다음 문구를 그대로 포함하라: {DISCLAIMER}")
        return " ".join(msgs)


def build_prompt(evidence: dict[str, Any], action_name: str, shop_context: str = "") -> str:
    """검색 결과(evidence)를 규칙이 박힌 프롬프트로 변환."""
    lines: list[str] = []
    lines.append("아래 규칙을 반드시 지켜 대응방안 리포트를 작성하라.\n")
    lines.append("[규칙]")
    lines.append("1. '허용 수치' 목록에 없는 숫자는 절대 생성하지 않는다. 추정·반올림·환산도 금지.")
    lines.append("2. 수치를 인용할 때 원문 맥락의 귀속(무엇의 효과인지)을 바꾸지 않는다.")
    lines.append("3. 각 수치 옆에 출처와 신뢰도 라벨을 병기한다.")
    lines.append("4. 허용 수치가 비어 있으면 숫자를 쓰지 말고 방향성만 서술한다.")
    lines.append(f"5. 마지막에 다음 문구를 그대로 넣는다: {DISCLAIMER}")
    lines.append("")
    lines.append(f"[대응방안] {action_name}")
    if shop_context:
        lines.append(f"[매장 상황] {shop_context}")
    lines.append("")

    lines.append("[방향성 근거 — 학술 실증]")
    if not evidence.get("direction_refs"):
        lines.append("- (없음)")
    for r in evidence.get("direction_refs", []):
        lines.append(f"- ({r['doc_id']} p{r['page']}, {r['tier_label']}) {r['text'][:250]}")
    lines.append("")

    lines.append("[허용 수치]")
    if not evidence.get("allowed_numbers"):
        lines.append("- (없음) → 수치 사용 금지, 방향성만 서술")
    for a in evidence.get("allowed_numbers", []):
        src = f"{a['doc_id']} p{a['page']}"
        lines.append(f"- {a['value']} | 출처: {src} ({a['tier_label']})")
        lines.append(f"  원문맥락: {a['sentence'][:250]}")
    return "\n".join(lines)


def verify_output(text: str, evidence: dict[str, Any]) -> VerifyResult:
    """생성문의 모든 수치가 허용 목록 안에 있는지, disclaimer가 있는지 검사."""
    allowed = {a["value"].replace(" ", "") for a in evidence.get("allowed_numbers", [])}
    found: list[str] = []
    violations: list[dict] = []

    for m in _NUM_RE.finditer(text):
        ctx = text[max(0, m.start() - 12) : m.end() + 4]
        if _EXEMPT.search(ctx):
            continue
        v = m.group(0).replace(" ", "")
        found.append(v)
        if v not in allowed:
            violations.append({"value": v, "context": ctx.strip()})

    has_disc = DISCLAIMER in text
    return VerifyResult(
        ok=(not violations) and has_disc,
        violations=violations,
        disclaimer_present=has_disc,
        numbers_found=found,
    )


def _call_openai(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2) -> str:
    """OpenAI 호출. 키는 환경변수 OPENAI_API_KEY 사용."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def generate_report(
    evidence: dict[str, Any],
    action_name: str,
    shop_context: str = "",
    llm=None,
    max_retry: int = 1,
) -> dict[str, Any]:
    """리포트 생성 + 검증 + 1회 재생성(Evaluator-Optimizer).

    llm: prompt(str) -> str 형태의 호출자. 미지정 시 OpenAI 사용.
         테스트에서는 가짜 함수를 주입할 수 있다.
    """
    llm = llm or _call_openai
    prompt = build_prompt(evidence, action_name, shop_context)

    try:
        text = llm(prompt)
    except Exception as exc:
        return {
            "report": "",
            "verified": False,
            "violations": [],
            "disclaimer_present": False,
            "attempts": 1,
            "evidence_refs": [],
            "has_magnitude": evidence.get("has_magnitude", False),
            "error": f"LLM 리포트 생성 실패: {type(exc).__name__}: {exc}",
        }
    result = verify_output(text, evidence)
    attempts = 1

    while not result.ok and attempts <= max_retry:
        retry_prompt = prompt + "\n\n[이전 출력의 문제점]\n" + result.as_feedback() + "\n\n위 문제를 고쳐 다시 작성하라."
        try:
            text = llm(retry_prompt)
        except Exception as exc:
            return {
                "report": text, "verified": False,
                "violations": result.violations,
                "disclaimer_present": result.disclaimer_present,
                "attempts": attempts + 1,
                "evidence_refs": [],
                "has_magnitude": evidence.get("has_magnitude", False),
                "error": f"LLM 리포트 재생성 실패: {type(exc).__name__}: {exc}",
            }
        result = verify_output(text, evidence)
        attempts += 1

    return {
        "report": text,
        "verified": result.ok,
        "violations": result.violations,
        "disclaimer_present": result.disclaimer_present,
        "attempts": attempts,
        "evidence_refs": [
            {
                "doc_id": a["doc_id"],
                "page": a["page"],
                "value": a["value"],
                "source_url": a.get("source_url"),
                "tier_label": a["tier_label"],
            }
            for a in evidence.get("allowed_numbers", [])
        ],
        "has_magnitude": evidence.get("has_magnitude", False),
    }


if __name__ == "__main__":
    import sys

    from rag.retriever import RagIndex
    from app.core.config import RAG_INDEX_EXPORT

    export = sys.argv[1] if len(sys.argv) > 1 else RAG_INDEX_EXPORT
    idx = RagIndex.load(export)
    ev = idx.build_evidence("세트메뉴가 객단가에 미치는 효과", axis="set_bundle")

    print(build_prompt(ev, "세트메뉴 도입", "치킨 전문점, 주말 객단가 정체")[:1200])
    if os.environ.get("OPENAI_API_KEY"):
        out = generate_report(ev, "세트메뉴 도입", "치킨 전문점, 주말 객단가 정체")
        print("\n=== 리포트 ===\n", out["report"])
        print("\n검증:", out["verified"], "| 시도:", out["attempts"], "| 위반:", out["violations"])
    else:
        print("\n(OPENAI_API_KEY 미설정 — 프롬프트만 출력했습니다)")
