# RAG 리포트 생성 모듈 (VSCode / Windows 앱용)
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
# build_prompt()의 [규칙] 3번이 요구하는 "(문서ID p페이지)" 인용 형식과 짝을 이룬다
_CITATION_RE = re.compile(r"\(([^\s()]+)\s+p(\d+)")

# build_prompt()의 [출력 형식]과 짝을 이루는 필수 섹션(소제목). "안내 문구"는 disclaimer_present로 별도 검사한다
REQUIRED_SECTIONS = ["대응방안 요약", "매장 상황", "권장 실행방안", "근거", "허용 수치", "적용 시 한계", "결론"]


@dataclass
class VerifyResult:
    ok: bool
    violations: list[dict]
    disclaimer_present: bool
    numbers_found: list[str]
    missing_sections: list[str]
    unauthorized_sources: list[str]

    # 재생성 시 LLM에 돌려줄 교정 지시문
    def as_feedback(self) -> str:

        msgs = []
        if self.violations:
            bad = ", ".join(f"'{v['value']}'" for v in self.violations)
            msgs.append(
                f"다음 수치는 허용 목록에 없으므로 삭제하거나 허용된 수치로 교체하라: {bad}. "
                "추정·환산·반올림도 금지된다."
            )
        if self.unauthorized_sources:
            bad_src = ", ".join(self.unauthorized_sources)
            msgs.append(
                f"다음 출처는 제공된 근거 목록에 없으므로 인용을 삭제하거나 제공된 출처로 교체하라: {bad_src}."
            )
        if self.missing_sections:
            msgs.append(
                "다음 섹션이 누락됐다. 지정된 순서와 소제목으로 모두 포함하라: "
                + ", ".join(self.missing_sections)
            )
        if not self.disclaimer_present:
            msgs.append(f"마지막에 다음 문구를 그대로 포함하라: {DISCLAIMER}")
        return " ".join(msgs)


# 검색 결과(evidence)를 규칙이 박힌 프롬프트로 변환
def build_prompt(evidence: dict[str, Any], action_name: str, shop_context: str = "") -> str:

    lines: list[str] = []
    lines.append("아래 규칙을 반드시 지켜 대응방안 리포트를 작성하라.\n")
    lines.append("[규칙]")
    lines.append("1. '허용 수치' 목록에 없는 숫자는 절대 생성하지 않는다. 추정·반올림·환산도 금지.")
    lines.append("2. 수치를 인용할 때 원문 맥락의 귀속(무엇의 효과인지)을 바꾸지 않는다.")
    lines.append("3. 출처를 인용할 때는 반드시 (문서ID p페이지) 형식으로 표기하고 신뢰도 라벨을 함께 병기한다.")
    lines.append("4. 허용 수치가 비어 있으면 숫자를 쓰지 말고 방향성만 서술한다.")
    lines.append(f"5. 마지막에 다음 문구를 그대로 넣는다: {DISCLAIMER}")
    lines.append("")
    lines.append("[근거 사용 제한]")
    lines.append("6. 제공된 방향성 근거와 허용 수치만 사용한다. 근거에 없는 원인·효과·고객 반응·매출 영향을 추론하거나 단정하지 않는다.")
    lines.append("7. 학술 근거는 방향성 설명에만 사용하고, 특정 매장의 효과로 일반화하지 않는다.")
    lines.append("8. 근거 문장의 의미를 확대하거나 인과관계로 바꾸지 않는다.")
    lines.append("9. 서로 다른 문헌의 내용을 조합해 근거에 없는 새로운 사실을 만들지 않는다.")
    lines.append("")
    lines.append("[수치 사용 규칙]")
    lines.append("10. 허용 수치를 인용할 때는 원문의 대상·지표·기간·조건을 함께 기술한다.")
    lines.append("11. 예산, 할인율, 기간, 고객 수, 매출액, 예상 효과 등 제공되지 않은 수치는 작성하지 않는다.")
    lines.append("")
    lines.append("[근거 부족 처리]")
    lines.append("12. 직접적인 근거가 부족하면 '직접적인 근거가 부족하다'고 명시한다.")
    lines.append("13. 현재 매장과 사례의 업종·상권·고객 조건이 다르면 적용 한계를 함께 설명한다.")
    lines.append("14. 출처가 불명확하거나 근거가 부족한 문장은 사용하지 않는다.")
    lines.append("")
    lines.append("[출력 형식]")
    lines.append(
        "다음 순서와 소제목으로 작성한다: 대응방안 요약 / 매장 상황 / 권장 실행방안 / "
        "근거(출처·신뢰도 포함) / 허용 수치 / 적용 시 한계 / 결론 / 안내 문구"
    )
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


# 생성문의 모든 수치가 허용 목록 안에 있는지, disclaimer가 있는지 검사
def verify_output(text: str, evidence: dict[str, Any]) -> VerifyResult:

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

    allowed_doc_ids = {r["doc_id"] for r in evidence.get("direction_refs", [])}
    allowed_doc_ids |= {a["doc_id"] for a in evidence.get("allowed_numbers", [])}
    unauthorized_sources = sorted({
        doc_id for doc_id, _page in _CITATION_RE.findall(text) if doc_id not in allowed_doc_ids
    })

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in text]

    has_disc = DISCLAIMER in text
    return VerifyResult(
        ok=(not violations) and has_disc and not missing_sections and not unauthorized_sources,
        violations=violations,
        disclaimer_present=has_disc,
        numbers_found=found,
        missing_sections=missing_sections,
        unauthorized_sources=unauthorized_sources,
    )


# OpenAI 호출. 키는 환경변수 OPENAI_API_KEY 사용
def _call_openai(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2) -> str:

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


# 리포트 생성 + 검증 + 1회 재생성(Evaluator-Optimizer)
def generate_report(
    evidence: dict[str, Any],
    action_name: str,
    shop_context: str = "",
    llm=None,
    max_retry: int = 1,
) -> dict[str, Any]:





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
            "missing_sections": [],
            "unauthorized_sources": [],
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
                "missing_sections": result.missing_sections,
                "unauthorized_sources": result.unauthorized_sources,
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
        "missing_sections": result.missing_sections,
        "unauthorized_sources": result.unauthorized_sources,
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
