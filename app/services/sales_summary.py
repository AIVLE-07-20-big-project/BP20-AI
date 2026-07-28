from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


SYSTEM_PROMPT = """당신은 소상공인이 쉽게 이해할 수 있도록 매출분석 결과를 설명하는 분석가다.

규칙:
1. 제공된 사실과 수치만 사용하고 새로운 원인이나 수치를 만들지 않는다.
2. 상권 전체 매출과 사용자의 매장 매출을 명확히 구분한다.
3. 가장 큰 내부 원인을 먼저 설명한다.
4. 제공된 세부 원인과 외부 원인은 유의미한 항목이므로 빠뜨리지 않는다.
5. 외부 원인은 확정 원인이 아니라 관련 가능성이 있는 원인으로 표현한다.
6. 데이터 결합, 값 확장, 회귀, 다중검정, 기여 방향 같은 기술 용어를 사용하지 않는다.
7. 같은 의미를 반복하지 않고 평이한 한국어로 작성한다.
8. 분석 방법이나 데이터 처리 과정은 설명하지 않는다.
9. JSON 객체 하나만 반환한다."""


def _fact_payload(report: dict[str, Any], detailed: dict[str, Any]) -> dict[str, Any]:
    market = report.get("매출분석") or {}
    traffic = report.get("간단분석 정보요약") or {}
    root = detailed.get("rootCauseAnalysis") or {}
    internal = detailed.get("internalAnalysis") or {}
    summary = internal.get("summary") or {}
    period = internal.get("period") or {}
    direction = (root.get("change") or {}).get("direction")
    primary = next(
        (
            item for item in root.get("internalDrivers", [])
            if (float(item.get("contributionAmount") or 0) < 0)
            == (direction == "decrease")
        ),
        (root.get("internalDrivers") or [{}])[0],
    )

    internal_labels = {
        "transaction_count": "거래 건수",
        "average_order_value": "객단가",
    }
    detailed_causes = [
        {
            "label": item.get("label"),
        }
        for item in root.get("internalDetailedDrivers", [])
        if item.get("label")
    ]
    external_causes = [
        {
            "label": item.get("label") or item.get("factor"),
            "confidence": item.get("confidence"),
            "causal": False,
        }
        for item in root.get("externalDrivers", [])
        if item.get("label") or item.get("factor")
    ]
    external_causes.extend(
        {
            "label": item.get("label") or item.get("factor"),
            "confidence": item.get("confidence"),
            "causal": False,
        }
        for item in root.get("possibleExternalDrivers", [])
        if item.get("label") or item.get("factor")
    )

    return {
        "marketSales": {
            "yearOverYear": market.get("전년동분기대비"),
            "quarterOverQuarter": market.get("전분기대비"),
        },
        "storeSales": {
            "currentPeriod": [period.get("currentStart"), period.get("currentEnd")],
            "comparisonPeriod": [period.get("baselineStart"), period.get("baselineEnd")],
            "changePct": (
                round(float(summary["revenueChangePct"]), 1)
                if summary.get("revenueChangePct") is not None
                else None
            ),
            "direction": direction,
        },
        "primaryInternalCause": internal_labels.get(
            primary.get("factor"),
            primary.get("factor"),
        ),
        "detailedInternalCauses": detailed_causes,
        "externalCauseCandidates": external_causes,
        "traffic": {
            "peakDay": traffic.get("유동인구 많은 요일"),
            "peakTime": traffic.get("유동인구 많은 시간대"),
        },
    }


def build_prompt(report: dict[str, Any], detailed: dict[str, Any]) -> str:
    facts = _fact_payload(report, detailed)
    return (
        "다음 분석 사실을 사용자용 핵심 요약으로 작성하라.\n"
        "반환 형식:\n"
        '{"headline":"한 문장 핵심 원인","summary":"headline을 반복하지 않는 전체 요약"}\n\n'
        f"[분석 사실]\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def _format_change(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.endswith("%"):
        return text
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return text


def _directional_change(label: str, value: Any) -> str | None:
    if value is None:
        return None
    try:
        numeric = float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return f"{label} 대비 {_format_change(value)} 변동"
    if numeric == 0:
        return f"{label}와 차이 없음"
    direction = "증가" if numeric > 0 else "감소"
    return f"{label}보다 {abs(numeric):g}% {direction}"


def _fallback_summary(report: dict[str, Any], detailed: dict[str, Any]) -> dict[str, Any]:
    facts = _fact_payload(report, detailed)
    market = facts["marketSales"]
    store = facts["storeSales"]
    primary = facts.get("primaryInternalCause")
    details = [item["label"] for item in facts["detailedInternalCauses"]]
    externals = [item["label"] for item in facts["externalCauseCandidates"]]
    traffic = facts["traffic"]

    sentences = []
    market_changes = [
        change
        for change in (
            _directional_change("전년 동기", market.get("yearOverYear")),
            _directional_change("전분기", market.get("quarterOverQuarter")),
        )
        if change
    ]
    if market_changes:
        sentences.append(f"상권 매출: {', '.join(market_changes)}.")

    change = _format_change(store.get("changePct"))
    current = [value for value in store.get("currentPeriod", []) if value]
    if change:
        period_text = f"{current[0]}~{current[-1]} " if current else ""
        direction = "감소" if store.get("direction") == "decrease" else "증가"
        sentences.append(
            f"내 매장의 {period_text}매출은 비교 기간보다 {change.lstrip('-')} {direction}했습니다."
        )
    if details:
        sentences.append(f"주요 세부 원인: {', '.join(details)}.")
    if externals:
        sentences.append(f"외부 원인 후보: {', '.join(externals)}.")
    if traffic.get("peakDay") and traffic.get("peakTime"):
        sentences.append(
            f"유동인구는 {traffic['peakDay']}과 {traffic['peakTime']}에 가장 많습니다."
        )

    return {
        "headline": f"매출 변화의 가장 큰 내부 원인은 {primary} 변화입니다." if primary else "매출 변화를 분석했습니다.",
        "summary": " ".join(sentences),
        "source": "deterministic_fallback",
        "facts": facts,
    }


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("SALES_SUMMARY_MODEL", "gpt-4.1"),
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


def generate_sales_summary(
    report: dict[str, Any],
    detailed: dict[str, Any],
    *,
    llm: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    fallback = _fallback_summary(report, detailed)
    if llm is None and not os.getenv("OPENAI_API_KEY"):
        return fallback

    try:
        raw = (llm or _call_openai)(build_prompt(report, detailed))
        parsed = json.loads(raw)
        headline = str(parsed.get("headline") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        if not headline or not summary:
            return {**fallback, "fallbackReason": "LLM 응답에 필수 문장이 없습니다."}
        required_labels = [
            item["label"]
            for key in ("detailedInternalCauses", "externalCauseCandidates")
            for item in fallback["facts"][key]
        ]
        forbidden_terms = ("POS 일자", "확장해 사용", "다중검정", "결합률", "통제 회귀")
        normalized_summary = re.sub(r"\s+", "", summary)
        missing_labels = []
        for label in required_labels:
            normalized_label = re.sub(r"\s+", "", label)
            core_label = normalized_label
            for removable in ("카테고리", "시간대", "평균", "여부", "변화"):
                core_label = core_label.replace(removable, "")
            if normalized_label not in normalized_summary and core_label not in normalized_summary:
                missing_labels.append(label)
        if missing_labels:
            return {
                **fallback,
                "fallbackReason": f"LLM 응답에서 원인이 누락됐습니다: {', '.join(missing_labels)}",
            }
        if any(term in f"{headline} {summary}" for term in forbidden_terms):
            return {**fallback, "fallbackReason": "LLM 응답에 기술 용어가 포함됐습니다."}
        return {
            "headline": headline,
            "summary": summary,
            "source": "gpt-4.1",
            "facts": fallback["facts"],
        }
    except Exception as exc:
        return {
            **fallback,
            "fallbackReason": f"{type(exc).__name__}: {exc}",
        }


def attach_sales_summary(
    report: dict[str, Any],
    detailed: dict[str, Any],
    *,
    llm: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    facts = _fact_payload(report, detailed)
    has_facts = any([
        facts["marketSales"].get("yearOverYear"),
        facts["marketSales"].get("quarterOverQuarter"),
        facts["storeSales"].get("changePct") is not None,
        facts.get("primaryInternalCause"),
        facts["detailedInternalCauses"],
        facts["externalCauseCandidates"],
    ])
    if not has_facts:
        return None

    generated = generate_sales_summary(report, detailed, llm=llm)
    report["사장님_요약"] = generated["summary"]
    report["사장님_요약_구조화"] = generated
    root = detailed.setdefault("rootCauseAnalysis", {})
    report["기술_분석결과_해설"] = report.get("분석결과 해설")
    root["technicalHeadline"] = root.get("headline")
    root["technicalNarrative"] = root.get("narrative")
    report["분석결과 해설"] = None
    root["headline"] = generated["headline"]
    root["narrative"] = generated["summary"]
    root["userSummary"] = generated["summary"]
    root["userHeadline"] = generated["headline"]
    return generated
