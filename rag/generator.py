# RAG 리포트 생성 모듈 (VSCode / Windows 앱용)
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from rag.retriever import STAT_RE

# 허용 수치를 실제로 인용했을 때만 붙인다 — 수치를 안 썼는데 "본 수치는..."이라고
# 쓰면 모순이 되므로, 수치 미사용 시에는 DISCLAIMER_WITHOUT_NUMBERS를 대신 요구한다.
DISCLAIMER_WITH_NUMBERS = (
    "본 수치는 유사 사례 문헌에 근거한 참고값이며, 해당 매장의 실측 효과가 아닙니다. "
    "실제 결과는 상권·메뉴·시기에 따라 달라질 수 있습니다."
)
DISCLAIMER_WITHOUT_NUMBERS = (
    "본 대응방안은 매장 내부 매출 분석 결과를 바탕으로 도출한 방향성 제안이며, "
    "실행 후 효과는 매장 데이터를 통해 별도로 검증해야 합니다."
)
# 이전 버전과의 호환(다른 모듈이 import할 수 있음)
DISCLAIMER = DISCLAIMER_WITH_NUMBERS

SYSTEM_PROMPT = (
    "당신은 소상공인의 매출 분석 결과를 바탕으로 대응방안을 제안하는 분석가다. "
    "제안은 반드시 분석에서 확인된 문제와 연결해야 하며, 주어진 근거 밖의 사실이나 수치를 "
    "결코 만들어내지 않는다."
)

# 수치 검증은 retriever.STAT_RE(허용 수치 추출과 동일 정규식)를 그대로 재사용한다.
_CITATION_RE = re.compile(r"\(([^\s()]+)\s+p(\d+)")

# build_prompt()의 [출력 형식]과 짝을 이루는 필수 섹션(소제목)
REQUIRED_SECTIONS = [
    "대응방안 요약",
    "매장 현황 및 핵심 문제",
    "권장 대응방안",
    "권장 실행 순서",
    "분석 근거",
    "실행 후 확인할 지표",
    "적용 시 한계",
    "결론",
]

# action-특화가 아닌(axis-공용) 수치를 인용할 때 강제하는 한정 문구(설계 §10)
AXIS_QUALIFIER_PHRASE = "동일 유형"

# 근거 부족을 이렇게 쓰면 안 된다 — 대신 "매장 내부 매출 분석을 근거로..." 식으로 써야 한다.
_FORBIDDEN_VAGUE_EVIDENCE_PHRASE = "직접적인 근거가 부족하다"
# 유동인구 등 상권 특성을 데이터 없이 단정하는 표현(=diagnosis_facts.traffic이 비어 있을 때 금지)
_UNSUPPORTED_TRAFFIC_PHRASES = ("유동인구가 많은 지역", "유동인구가 많음을 고려", "유동인구가 많다는 점을 고려")


@dataclass
class VerifyResult:
    ok: bool
    violations: list[dict]
    disclaimer_present: bool
    numbers_found: list[str]
    missing_sections: list[str]
    unauthorized_sources: list[str]
    missing_axis_qualifier: bool = False
    vague_evidence_phrase_found: bool = False
    unsupported_traffic_claim: bool = False
    expected_disclaimer: str = DISCLAIMER_WITHOUT_NUMBERS
    diagnosis_disconnected: bool = False

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
        if self.missing_axis_qualifier:
            msgs.append(
                "인용한 수치 중 일부는 이 대응방안이 아니라 같은 유형(axis)의 다른 사례에서 "
                f"나온 것이다. 해당 수치를 인용할 때 '{AXIS_QUALIFIER_PHRASE} 사례'처럼 한정하는 표현을 포함하라."
            )
        if self.vague_evidence_phrase_found:
            msgs.append(
                f"'{_FORBIDDEN_VAGUE_EVIDENCE_PHRASE}' 같은 모호한 표현 대신, "
                "'매장 내부 매출 분석 결과를 바탕으로 도출한 대응방안이며, 동일 조건에서 이 방안의 효과를 "
                "검증한 외부 근거는 확보되지 않았습니다'처럼 구체적으로 서술하라."
            )
        if self.unsupported_traffic_claim:
            msgs.append(
                "제공된 분석 사실에 유동인구·요일·시간대 데이터가 없는데도 '유동인구가 많은 지역' 같은 "
                "상권 특성을 단정했다. 해당 데이터가 없으면 그 문장 자체를 삭제하라."
            )
        if self.diagnosis_disconnected:
            msgs.append(
                "'매장 현황 및 핵심 문제'에서 언급한 문제(거래 건수·채널·상품군 감소 등)와 "
                "'권장 대응방안'이 서로 연결되지 않는다. 각 실행방안이 어떤 문제를 해결하려는 것인지 "
                "명시적으로 연결해서 다시 작성하라."
            )
        if not self.disclaimer_present:
            msgs.append(f"마지막에 다음 문구를 그대로 포함하라: {self.expected_disclaimer}")
        return " ".join(msgs)


def _format_amount(value: float | int | None) -> str | None:
    if value is None:
        return None
    return f"{round(value):,}원"


def _diagnosis_facts_lines(diagnosis_facts: dict[str, Any] | None) -> list[str]:
    """build_diagnosis_facts(...) 결과를 프롬프트용 한국어 사실 목록으로 바꾼다."""

    facts = diagnosis_facts or {}
    lines: list[str] = []

    store_sales = facts.get("storeSales") or {}
    if store_sales.get("changePct") is not None:
        direction = "감소" if store_sales.get("direction") == "decrease" else "증가"
        lines.append(f"- 매장 매출 변화: {store_sales['changePct']}% {direction}")

    market_sales = facts.get("marketSales") or {}
    if market_sales.get("yearOverYear") is not None:
        lines.append(f"- 상권 전체 매출(전년동분기대비): {market_sales['yearOverYear']}")
    if market_sales.get("quarterOverQuarter") is not None:
        lines.append(f"- 상권 전체 매출(전분기대비): {market_sales['quarterOverQuarter']}")

    if facts.get("primaryInternalCause"):
        lines.append(f"- 매출 변화의 가장 큰 내부 원인: {facts['primaryInternalCause']}")

    for item in facts.get("detailedInternalCauses") or []:
        label = item.get("label")
        if not label:
            continue
        direction = "감소" if item.get("direction") == "negative" else "증가"
        amount = _format_amount(item.get("contributionAmount"))
        suffix = f" ({amount} 기여)" if amount else ""
        lines.append(f"- 세부 내부 원인({item.get('dimension') or '기타'}): {label} {direction}{suffix}")

    for item in facts.get("externalCauseCandidates") or []:
        label = item.get("label")
        if not label:
            continue
        confidence = item.get("confidence")
        suffix = f" (신뢰도 {confidence})" if confidence is not None else ""
        lines.append(f"- 외부 요인 후보(확정 아님): {label}{suffix}")

    traffic = facts.get("traffic") or {}
    if traffic.get("peakDay") or traffic.get("peakTime"):
        parts = [v for v in (traffic.get("peakDay"), traffic.get("peakTime")) if v]
        lines.append(f"- 유동인구가 많은 시점: {', '.join(parts)}")

    return lines


def _has_diagnosis_facts(diagnosis_facts: dict[str, Any] | None) -> bool:
    facts = diagnosis_facts or {}
    return bool(
        facts.get("primaryInternalCause")
        or facts.get("detailedInternalCauses")
        or facts.get("externalCauseCandidates")
    )


def _has_traffic_facts(diagnosis_facts: dict[str, Any] | None) -> bool:
    traffic = (diagnosis_facts or {}).get("traffic") or {}
    return bool(traffic.get("peakDay") or traffic.get("peakTime"))


# 검색 결과(evidence)와 매출 분석 사실(diagnosis_facts)을 규칙이 박힌 프롬프트로 변환
def build_prompt(
    evidence: dict[str, Any],
    action_name: str,
    diagnosis_facts: dict[str, Any] | None = None,
    shop_context: str = "",
) -> str:

    fact_lines = _diagnosis_facts_lines(diagnosis_facts)

    lines: list[str] = []
    lines.append("아래 규칙을 반드시 지켜 대응방안 리포트를 작성하라.\n")
    lines.append("[핵심 원칙]")
    lines.append(
        "이 리포트의 목적은 '이 업종·상권에는 이 방안이 일반적으로 좋다'가 아니라, "
        "'매출 분석에서 확인된 이 문제를 해결하기 위해 이 방안을 이렇게 실행한다'를 설명하는 것이다."
    )
    lines.append("")
    lines.append("[작성 순서 — 반드시 이 논리로 작성한다]")
    lines.append("1. 매출 분석에서 확인된 핵심 문제가 무엇인지 설명한다.")
    lines.append("2. 그 문제가 매출에 어떤 영향을 주었는지 설명한다.")
    lines.append("3. 추천 방안이 그 문제 중 어떤 것을 해결하기 위한 것인지 명시적으로 연결한다.")
    lines.append("4. 구체적인 실행 방법을 제시한다(추상적인 마케팅 문구 금지).")
    lines.append("5. 실행 후 확인해야 할 지표를 제시한다.")
    lines.append("6. 효과를 단정할 수 없는 이유를 설명한다.")
    lines.append(
        "각 권장 실행방안은 반드시 분석 결과와 연결한다. 예: 거래 건수 감소 → 신규 방문과 주문 유입 확대 / "
        "배달 매출 감소 → SNS에서 배달 주문 경로 강화 / 특정 상품 매출 감소 → 해당 상품 중심 콘텐츠 운영 / "
        "특정 시간대 매출 감소 → 해당 시간대 대상 게시 및 프로모션 검토."
    )
    lines.append("")
    lines.append("[구체성 규칙]")
    lines.append(
        "'브랜드 인지도를 높인다', '접점을 확대한다' 같은 추상적 표현만 쓰지 않는다. "
        "무엇을, 어떤 순서로, 어떤 기준으로 실행할지 실행 단위로 서술한다."
    )
    lines.append("")
    lines.append("[수치·근거 규칙]")
    lines.append("1. '허용 수치' 목록에 없는 숫자는 절대 생성하지 않는다. 추정·반올림·환산도 금지.")
    lines.append("2. 수치를 인용할 때 원문 맥락의 귀속(무엇의 효과인지)을 바꾸지 않는다.")
    lines.append("3. 출처를 인용할 때는 반드시 (문서ID p페이지) 형식으로 표기하고 신뢰도 라벨을 함께 병기한다.")
    lines.append("4. 허용 수치가 비어 있으면 숫자를 쓰지 말고 방향성만 서술한다.")
    lines.append(
        "5. [분석 사실]에 없는 상권 특성(예: 유동인구가 많다는 판단)은 임의로 단정하지 않는다. "
        "유동인구·요일·시간대 데이터가 [분석 사실]에 있을 때만 그 내용을 그대로 인용해서 쓴다."
    )
    lines.append(
        "6. 학술 근거는 방향성 설명에만 사용하고, 특정 매장의 효과로 일반화하지 않는다. "
        "근거에 없는 원인·효과·고객 반응·매출 영향을 추론하거나 단정하지 않는다."
    )
    lines.append("7. 서로 다른 문헌의 내용을 조합해 근거에 없는 새로운 사실을 만들지 않는다.")
    lines.append(
        "8. 외부 문헌 근거가 없으면 '직접적인 근거가 부족하다'라고만 쓰지 말고, "
        "'매장 내부 매출 분석 결과를 바탕으로 도출한 대응방안이며, 동일 조건에서 이 방안의 효과를 검증한 "
        "외부 근거는 확보되지 않았습니다'처럼 구체적으로 서술한다."
    )
    lines.append(
        "9. 허용 수치가 전혀 없으면 예상 증가율·목표 매출·기대 효과 수치를 절대 작성하지 않고, "
        f"마지막에 다음 문구를 그대로 넣는다: {DISCLAIMER_WITHOUT_NUMBERS}"
    )
    lines.append(
        "10. 허용 수치를 실제로 인용했다면(수치를 하나라도 썼다면) 마지막에 다음 문구를 그대로 넣는다: "
        f"{DISCLAIMER_WITH_NUMBERS}"
    )
    lines.append("11. 두 안내 문구를 동시에 쓰지 않는다 — 수치를 하나라도 썼는지 여부로 하나만 고른다.")
    lines.append("")
    lines.append("[출력 형식]")
    lines.append(
        "다음 순서와 소제목으로 작성한다: "
        + " / ".join(REQUIRED_SECTIONS)
        + ". '분석 근거'는 매장 데이터 근거·상권 데이터 근거·외부 근거로 구분해서 쓴다."
    )
    lines.append("")
    lines.append(f"[대응방안] {action_name}")
    if shop_context:
        lines.append(f"[매장 상황] {shop_context}")
    lines.append("")

    lines.append("[분석 사실 — 매출 분석에서 확인된 내용]")
    if fact_lines:
        lines.extend(fact_lines)
    else:
        lines.append("- (제공된 분석 사실 없음) → 매장 고유의 문제를 단정하지 말고, 방안의 일반적 실행 방법만 서술한다.")
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
        note = ""
        if a.get("action_specific") is False:
            note = (
                f" ※ 이 수치는 '{action_name}'이 아니라 같은 유형(axis)의 다른 방안 사례에서 나온 것이다 — "
                f"인용할 때 '{AXIS_QUALIFIER_PHRASE} 사례'처럼 한정해서 표현하라."
            )
        lines.append(f"- {a['value']} | 출처: {src} ({a['tier_label']}){note}")
        lines.append(f"  원문맥락: {a['sentence'][:250]}")
    return "\n".join(lines)


# 생성문의 모든 수치가 허용 목록 안에 있는지, 섹션·안내문구·근거 연결이 맞는지 검사
def verify_output(
    text: str,
    evidence: dict[str, Any],
    diagnosis_facts: dict[str, Any] | None = None,
) -> VerifyResult:

    allowed = {a["value"].replace(" ", "") for a in evidence.get("allowed_numbers", [])}
    found: list[str] = []
    violations: list[dict] = []

    for m in STAT_RE.finditer(text):
        ctx = text[max(0, m.start() - 12) : m.end() + 4]
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

    # action-특화가 아닌(axis-공용) 수치를 인용했으면 한정 문구를 강제한다(설계 §10)
    non_specific_values = {
        a["value"].replace(" ", "") for a in evidence.get("allowed_numbers", [])
        if a.get("action_specific") is False
    }
    cited_non_specific = bool(non_specific_values & set(found))
    missing_axis_qualifier = cited_non_specific and AXIS_QUALIFIER_PHRASE not in text

    # 수치를 하나라도 실제로 썼는지에 따라 요구하는 안내 문구가 달라진다(모순 방지).
    numbers_actually_used = bool(found)
    expected_disclaimer = DISCLAIMER_WITH_NUMBERS if numbers_actually_used else DISCLAIMER_WITHOUT_NUMBERS
    has_disc = expected_disclaimer in text

    vague_evidence_phrase_found = _FORBIDDEN_VAGUE_EVIDENCE_PHRASE in text
    unsupported_traffic_claim = (
        not _has_traffic_facts(diagnosis_facts)
        and any(phrase in text for phrase in _UNSUPPORTED_TRAFFIC_PHRASES)
    )
    # 분석 사실이 있는데 리포트 본문 어디에도 핵심 원인이 등장하지 않으면 진단과 방안이
    # 분리된 것으로 본다(예: "거래 건수" 원인인데 리포트에 "거래 건수"가 한 번도 안 나옴).
    primary_cause = (diagnosis_facts or {}).get("primaryInternalCause")
    diagnosis_disconnected = bool(primary_cause) and primary_cause not in text

    return VerifyResult(
        ok=(
            not violations and has_disc and not missing_sections and not unauthorized_sources
            and not missing_axis_qualifier and not vague_evidence_phrase_found
            and not unsupported_traffic_claim and not diagnosis_disconnected
        ),
        violations=violations,
        disclaimer_present=has_disc,
        numbers_found=found,
        missing_sections=missing_sections,
        unauthorized_sources=unauthorized_sources,
        missing_axis_qualifier=missing_axis_qualifier,
        vague_evidence_phrase_found=vague_evidence_phrase_found,
        unsupported_traffic_claim=unsupported_traffic_claim,
        expected_disclaimer=expected_disclaimer,
        diagnosis_disconnected=diagnosis_disconnected,
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
    diagnosis_facts: dict[str, Any] | None = None,
    shop_context: str = "",
    llm=None,
    max_retry: int = 1,
) -> dict[str, Any]:

    llm = llm or _call_openai
    prompt = build_prompt(evidence, action_name, diagnosis_facts, shop_context)
    # 근거 출처는 LLM 성공 여부와 무관하게 evidence만으로 결정되므로 모든 반환 경로에서 동일하게 채운다
    evidence_refs = [
        {
            "doc_id": a["doc_id"],
            "page": a["page"],
            "value": a["value"],
            "source_url": a.get("source_url"),
            "tier_label": a["tier_label"],
        }
        for a in evidence.get("allowed_numbers", [])
    ]

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
            "evidence_refs": evidence_refs,
            "has_magnitude": evidence.get("has_magnitude", False),
            "error": f"LLM 리포트 생성 실패: {type(exc).__name__}: {exc}",
        }
    result = verify_output(text, evidence, diagnosis_facts)
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
                "evidence_refs": evidence_refs,
                "has_magnitude": evidence.get("has_magnitude", False),
                "error": f"LLM 리포트 재생성 실패: {type(exc).__name__}: {exc}",
            }
        result = verify_output(text, evidence, diagnosis_facts)
        attempts += 1

    return {
        "report": text,
        "verified": result.ok,
        "violations": result.violations,
        "disclaimer_present": result.disclaimer_present,
        "missing_sections": result.missing_sections,
        "unauthorized_sources": result.unauthorized_sources,
        "attempts": attempts,
        "evidence_refs": evidence_refs,
        "has_magnitude": evidence.get("has_magnitude", False),
    }


if __name__ == "__main__":
    import sys

    from rag.retriever import RagIndex
    from app.core.config import RAG_INDEX_EXPORT

    export = sys.argv[1] if len(sys.argv) > 1 else RAG_INDEX_EXPORT
    idx = RagIndex.load(export)
    ev = idx.build_evidence("세트메뉴가 객단가에 미치는 효과", axis="set_bundle")
    sample_facts = {
        "storeSales": {"changePct": -12.4, "direction": "decrease"},
        "primaryInternalCause": "거래 건수",
        "detailedInternalCauses": [
            {"label": "배달", "direction": "negative", "contributionAmount": -800000, "dimension": "채널"},
        ],
        "externalCauseCandidates": [],
        "traffic": {},
    }

    print(build_prompt(ev, "세트메뉴 도입", sample_facts, "치킨 전문점, 주말 객단가 정체")[:1200])
    if os.environ.get("OPENAI_API_KEY"):
        out = generate_report(ev, "세트메뉴 도입", sample_facts, "치킨 전문점, 주말 객단가 정체")
        print("\n=== 리포트 ===\n", out["report"])
        print("\n검증:", out["verified"], "| 시도:", out["attempts"], "| 위반:", out["violations"])
    else:
        print("\n(OPENAI_API_KEY 미설정 — 프롬프트만 출력했습니다)")
