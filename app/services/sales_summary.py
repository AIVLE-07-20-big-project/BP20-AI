from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


SYSTEM_PROMPT = """
당신은 소상공인을 위한 매출 증감 원인 분석가다.

사용자가 알고 싶은 것은 대응방안이나 미래 매출 예측이 아니라,
분석 기간의 매출이 왜 상승하거나 하락했는지에 대한 설명이다.

[분석 목적]
제공된 매장 데이터와 외부 환경 데이터를 바탕으로 다음 내용을 설명한다.

1. 매출이 얼마나 증가하거나 감소했는지
2. 거래 건수와 객단가 중 어느 요인이 더 크게 작용했는지
3. 어떤 판매채널, 주문유형, 상품군, 상품, 시간대가 매출 변화와 관련됐는지
4. 날씨, 행사, 주말, 유동인구 등 외부 환경이 매출 변화와 어떤 방향으로 함께 나타났는지
5. 확인된 내용을 종합했을 때 매출 변화의 주요 원인이 무엇인지

[사실 사용 원칙]
1. 제공된 사실과 수치만 사용한다.
2. 제공되지 않은 원인, 고객 행동, 지역 상황은 만들지 않는다.
3. 미래 매출을 예측하지 않는다.
4. 마케팅 전략이나 대응방안을 추천하지 않는다.
5. 분석 결과에 없는 날씨나 행사를 임의로 언급하지 않는다.
6. 상권 전체 매출과 사용자의 매장 매출을 명확히 구분한다.
7. 외부 요인은 직접적인 원인으로 단정하지 않고 매출 변화와 함께 나타난 관련 요인으로 설명한다.
8. 추정 영향률이 제공됐을 때만 퍼센트 효과를 작성한다.
9. 행사명, 행사 기간, 거리 정보가 제공됐을 때만 구체적인 행사를 언급한다.
10. 현재값과 비교값이 모두 있을 때만 외부 요인이 증가하거나 감소했다고 표현한다.

[내부 원인 작성 방법]
1. 매출 변화와 거래 건수·객단가의 관계를 먼저 설명한다.
2. 거래 건수가 핵심 원인이면 다음 의미를 자연스럽게 풀어 쓴다.
   '고객 한 명당 결제금액보다 실제 주문 건수의 변화가 매출에 더 크게 작용했습니다.'
3. 같은 현상을 나타내는 항목(판매채널·주문유형·배달 플랫폼 등)은 중복 나열하지 않고 묶는다.
4. 상위 상품군과 세부 상품은 계층적으로 설명한다(예: 카테고리 먼저, 세부 상품은 그 다음).
5. 연속된 시간대는 하나의 구간으로 묶는다.

[외부 환경 작성 방법]
각 외부 요인은 가능한 경우 다음 순서로 설명한다.
1. 비교 기간과 현재 기간의 값
2. 증가 또는 감소 방향
3. 매출 변화와 함께 나타난 방향
4. 제공된 경우 추정 영향률
5. 직접적인 원인으로 단정할 수 없다는 주의
행사명, 거리, 기간, 영향률이 없으면 해당 내용을 작성하지 않는다.
비교값·현재값이 없는 요인은 방향을 설명하지 말고, 이름만 다른 요인과 함께 "고려된 외부 요인"으로
짧게 언급한다(예: "이 외에 강수량도 외부 요인으로 함께 고려됐으나 구체적인 변화값은 확인되지 않았습니다").

[금지 사항]
다음과 같은 표현을 사용하지 않는다.
- '대응방안을 권장합니다', 'SNS 캠페인이 필요합니다', '향후 매출이 증가할 것입니다' 등 전략 제안·예측 문장
- '기여 폭이 큰 항목입니다', '유의미한 항목', '해당 요인', '긍정적인 영향을 미쳤습니다',
  '데이터상 확인됩니다', '분석 결과로 판단됩니다'
- '날씨 때문에 매출이 감소했습니다', '행사로 인해 매출이 증가했습니다'와 같은 단정적 인과 표현
- 회귀, 분해, 결합률, 다중검정, 통계적 유의성, 상관계수, 모델 기여도 같은 기술 용어
- 제공되지 않은 고객 행동이나 방문 이유를 추정하는 문장

[그 외 원칙]
- headline과 summary에서 같은 내용이나 숫자를 반복하지 않는다.
- JSON 객체 하나만 반환한다.

[출력 형식]
{
  "headline": "매출 증감의 가장 큰 원인을 설명하는 한 문장",
  "summary": "전체 원인 분석",
  "sections": {
    "salesChange": "매출 증감 요약",
    "primaryCause": "거래 건수와 객단가 중심의 핵심 원인",
    "internalDetails": "채널·상품·시간대별 세부 원인",
    "externalFactors": "날씨·행사·주말·유동인구 분석",
    "conclusion": "종합 원인 해석(상권·유동인구 포함)",
    "limitations": "직접적인 인과관계로 단정할 수 없는 사항"
  }
}
"""

EXTERNAL_FACTOR_META = {
    "temperature_c": {"label": "평균 기온", "unit": "℃", "changeExpression": "평균 기온이"},
    "wind_speed_ms": {"label": "풍속", "unit": "m/s", "changeExpression": "평균 풍속이"},
    "precipitation_mm": {"label": "강수량", "unit": "mm", "changeExpression": "강수량이"},
    "is_weekend": {"label": "주말 비중", "unit": "%p", "changeExpression": "주말 영업일 비중이"},
    "event_count_2km": {"label": "주변 행사 수", "unit": "건", "changeExpression": "매장 주변 행사 수가"},
    "event_exposure": {"label": "주변 행사 노출도", "unit": "", "changeExpression": "주변 행사 노출이"},
}


def _external_cause_payload(item: dict[str, Any]) -> dict[str, Any]:
    factor = item.get("factor")
    meta = EXTERNAL_FACTOR_META.get(factor, {})
    return {
        "factor": factor,
        "label": item.get("label") or meta.get("label") or factor,
        "unit": meta.get("unit"),
        "changeExpression": meta.get("changeExpression"),
        "baselineValue": item.get("baselineValue"),
        "currentValue": item.get("currentValue"),
        "changeValue": item.get("changeValue"),
        "estimatedEffectPct": item.get("estimatedEffectPct"),
        "estimatedContributionAmount": item.get("estimatedContributionAmount"),
        "direction": item.get("direction"),
        "confidence": item.get("confidence"),
        "effectUnit": item.get("effectUnit"),
        "evidenceType": item.get("evidenceType", "association"),
        "causal": False,
    }


def build_diagnosis_facts(report: dict[str, Any], detailed: dict[str, Any]) -> dict[str, Any]:
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
            # 카테고리·상품유형·채널은 서로 겹치므로 퍼센트를 사용자 문장에
            # 노출하지 않는다. 원본 상세 결과에는 dimensionSharePct가 남아 있다.
            "contributionPct": None,
            "contributionAmount": item.get("contributionAmount"),
            "direction": item.get("direction"),
            "dimension": item.get("dimension"),
        }
        for item in root.get("internalDetailedDrivers", [])
        if item.get("label")
    ]
    external_causes = [
        _external_cause_payload(item)
        for item in root.get("externalDrivers", [])
        if item.get("label") or item.get("factor")
    ]
    external_causes.extend(
        _external_cause_payload(item)
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
        "eventDrivers": root.get("eventDrivers", []),
        "traffic": {
            "peakDay": traffic.get("유동인구 많은 요일"),
            "peakTime": traffic.get("유동인구 많은 시간대"),
        },
    }


def build_prompt(report: dict[str, Any], detailed: dict[str, Any]) -> str:
    facts = build_diagnosis_facts(report, detailed)
    return (
        """다음 분석 사실을 바탕으로 소상공인이 읽기 쉬운 매출 원인 상세분석 보고서를 작성하라.

[작성 목표]
사용자는 대응방안이나 전략 제안이 아니라, 이번 매출의 증감이 왜 일어났는지 —
가장 큰 내부 원인(거래 건수 vs 객단가), 이를 주도한 채널·상품군·시간대,
외부 환경과의 관련성, 종합 해석과 한계를 알고 싶어 한다.

[작성 순서]
1. 내 매장의 매출 변화를 먼저 설명한다(salesChange).
2. 거래 건수와 객단가 중 어느 쪽이 더 크게 작용했는지 설명한다(primaryCause).
3. 세부 원인은 영향이 큰 항목부터, 채널·상품군·시간대를 묶어 설명한다(internalDetails).
4. 외부 환경 요인을 방향과 관련성 위주로 설명한다(externalFactors, 대응방안 제안 금지).
5. 상권 매출·유동인구를 포함해 종합 해석을 작성한다(conclusion). 날씨·행사를 직접적 원인으로
   단정하지 않는다.
6. 데이터 부족이나 인과관계로 단정할 수 없는 내용은 한계에서 안내한다(limitations).

[표현 규칙]
- 거래 건수가 핵심 원인이면 고객 한 명당 결제금액보다 주문 건수 변화가 더 큰 영향을 주었다는 의미를 설명한다.
- 외부 요인은 이름만 나열하지 않는다. 현재값과 비교값이 있으면 무엇이 얼마나 변했는지 설명한다.
- estimatedEffectPct가 있을 때만 추정 영향률을 사용한다.
- 외부 요인은 원인으로 단정하지 않고 매출 변화와 관련이 있거나 영향을 주었을 가능성이 있다고 표현한다.
- 행사명, 기간, 거리가 제공된 경우에만 구체적인 행사를 언급한다.
- eventDrivers에 행사 상세정보가 있으면 행사명, 기간, 매장과의 거리를 포함해 설명한다.
- 행사 상세정보가 없으면 행사명이나 날짜를 만들어내지 말고 행사 수·노출도만 설명한다.
- 행사 노출 매출과 비노출 매출의 비교값이 있을 때만 추정 영향률을 사용한다.
- 제공되지 않은 고객 특성, 방문 이유, 날씨, 행사, 마케팅, 가격 인상 등을 추정하지 않는다.
- 서로 다른 분석 기준의 기여도를 합산하지 않는다.
- '분기별 상권 분석'이나 '월별 상세 원인 분석'이라는 제목을 사용하지 않는다. 전체 제목은 '매출 원인 상세분석'으로 전제한다.

[분석 사실]
"""
        + json.dumps(facts, ensure_ascii=False, indent=2)
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


def _group_internal_details(items: list[dict[str, Any]], sales_direction: str | None) -> list[str]:
    direction_word = "감소" if sales_direction == "decrease" else "증가"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(item.get("dimension") or "other", []).append(item)
    result: list[str] = []
    delivery_related = [
        item for item in grouped.get("sales_channel", []) + grouped.get("order_type", [])
        if "배달" in str(item.get("label"))
    ]
    if delivery_related:
        sentence = f"판매 방식에서는 배달 매출 {direction_word}가 전체 흐름을 주도했습니다."
        platforms = grouped.get("delivery_platform", [])
        if platforms:
            names = [str(item["label"]).replace(" 배달 플랫폼", "") for item in platforms]
            sentence += f" 배달 플랫폼 중에서는 {', '.join(names)}의 매출 {direction_word}가 두드러졌습니다."
        result.append(sentence)
    categories = grouped.get("category", [])
    if categories:
        names = [str(item["label"]).replace(" 카테고리", "") for item in categories]
        sentence = f"상품군에서는 {', '.join(names)} 카테고리의 매출 {direction_word}가 두드러졌습니다."
        product_types = grouped.get("product_type", [])
        if product_types:
            product_names = [str(item["label"]).replace(" 상품 유형", "") for item in product_types]
            sentence += f" 세부 상품 중에서는 {', '.join(product_names)}의 매출 {direction_word}가 크게 나타났습니다."
        result.append(sentence)
    return result


def _format_external_driver(item: dict[str, Any]) -> str | None:
    label = item.get("label")
    before = item.get("baselineValue")
    current = item.get("currentValue")
    effect = item.get("estimatedEffectPct")
    if not label or before is None or current is None:
        return None
    try:
        before_number = float(before)
        current_number = float(current)
    except (TypeError, ValueError):
        return None
    if current_number == before_number:
        return None
    unit = item.get("unit") or ""
    change_direction = "증가" if current_number > before_number else "감소"
    sales_direction = {"positive": "상승", "negative": "하락"}.get(item.get("direction"))
    expression = item.get("changeExpression") or f"{label}이"
    sentence = f"{expression} 비교 기간의 {before_number:g}{unit}에서 {current_number:g}{unit}로 {change_direction}했습니다."
    if effect is not None and sales_direction:
        try:
            sentence += f" 이러한 변화는 매출 약 {abs(float(effect)):.1f}% {sales_direction}과 관련된 것으로 나타났습니다."
        except (TypeError, ValueError):
            pass
    else:
        sentence += f" 이 변화는 매출 {sales_direction or '변화'} 흐름과 함께 나타났지만 직접적인 원인으로 단정할 수는 없습니다."
    return sentence


def _candidate_only_label(item: dict[str, Any]) -> str | None:
    # baseline/current 값이 없어 방향을 설명할 수 없는 요인도, 후보로 고려됐다는 사실 자체는
    # 남긴다 — 이름조차 언급하지 않으면 실제로 검토된 요인이 조용히 사라진다.
    label = item.get("label")
    if not label:
        return None
    if item.get("baselineValue") is not None and item.get("currentValue") is not None:
        return None
    return str(label)


def _primary_cause_sentence(facts: dict[str, Any]) -> str | None:
    primary = facts.get("primaryInternalCause")
    if primary not in ("거래 건수", "객단가"):
        return None
    other = "객단가" if primary == "거래 건수" else "거래 건수"
    is_decrease = facts["storeSales"].get("direction") == "decrease"
    direction_word = "감소" if is_decrease else "증가"
    change_verb = "줄어든" if is_decrease else "늘어난"
    sentence = f"매출 {direction_word}에는 {other}보다 {primary} {direction_word}이 더 크게 작용한 것으로 나타났습니다."
    if primary == "거래 건수":
        sentence += f" 고객 한 명이 결제한 금액의 변화보다는 실제 주문과 방문 건수가 {change_verb} 영향이 컸다는 의미입니다."
    else:
        sentence += f" 실제 주문과 방문 건수보다는 고객 한 명이 결제한 금액이 {change_verb} 영향이 컸다는 의미입니다."
    return sentence


def _fallback_summary(report: dict[str, Any], detailed: dict[str, Any]) -> dict[str, Any]:
    facts = build_diagnosis_facts(report, detailed)
    market = facts["marketSales"]
    store = facts["storeSales"]
    primary = facts.get("primaryInternalCause")
    traffic = facts["traffic"]
    is_decrease = store.get("direction") == "decrease"
    direction_word = "감소" if is_decrease else "증가"

    sales_change_sentence = None
    change = _format_change(store.get("changePct"))
    current = [value for value in store.get("currentPeriod", []) if value]
    if change:
        period_text = f"{current[0]}~{current[-1]} " if current else ""
        sales_change_sentence = (
            f"내 매장의 {period_text}매출은 비교 기간보다 {change.lstrip('-')} {direction_word}했습니다."
        )

    primary_cause_sentence = _primary_cause_sentence(facts)

    internal_sentences = _group_internal_details(
        facts["detailedInternalCauses"],
        store.get("direction"),
    )
    external_sentences = [
        sentence
        for item in facts["externalCauseCandidates"]
        if (sentence := _format_external_driver(item))
    ]
    candidate_only_labels = [
        label
        for item in facts["externalCauseCandidates"]
        if (label := _candidate_only_label(item))
    ]
    external_parts = list(external_sentences)
    if candidate_only_labels:
        external_parts.append(
            f"{', '.join(candidate_only_labels)}도 외부 요인으로 함께 고려됐으나, "
            "구체적인 변화값이 없어 매출 변화와의 방향은 설명하지 않습니다."
        )
    external_intro = ("외부 환경을 함께 살펴보면 " + " ".join(external_parts)) if external_parts else ""

    market_changes = [
        change
        for change in (
            _directional_change("전년 동기", market.get("yearOverYear")),
            _directional_change("전분기", market.get("quarterOverQuarter")),
        )
        if change
    ]
    conclusion_sentences = []
    if market_changes:
        conclusion_sentences.append("상권 전체 매출은 " + ", ".join(market_changes) + "로 나타났습니다.")
    if traffic.get("peakDay") and traffic.get("peakTime"):
        conclusion_sentences.append(
            f"유동인구는 {traffic['peakDay']}과 {traffic['peakTime']}에 가장 많습니다."
        )
    if primary and internal_sentences:
        synthesis = f"종합하면 이번 매출 {direction_word}은 {primary} 변화를 중심으로 세부 원인이 함께 작용한 결과로 해석할 수 있습니다."
        if external_parts:
            synthesis += " 외부 환경도 매출 변화와 관련된 흐름을 보였을 수 있으나, 직접적인 원인으로 확정하려면 추가 비교가 필요합니다."
        conclusion_sentences.append(synthesis)

    sentences = [
        part for part in (
            sales_change_sentence,
            primary_cause_sentence,
            *internal_sentences,
            external_intro,
            *conclusion_sentences,
        ) if part
    ]

    return {
        "headline": f"매출 변화의 가장 큰 내부 원인은 {primary} 변화입니다." if primary else "매출 변화를 분석했습니다.",
        "summary": " ".join(sentences),
        "sections": {
            "salesChange": sales_change_sentence or "",
            "primaryCause": primary_cause_sentence or "",
            "internalDetails": " ".join(internal_sentences),
            "externalFactors": " ".join(external_parts),
            "conclusion": " ".join(conclusion_sentences),
            "limitations": "외부 요인은 매출 변화와 함께 나타난 관련성으로만 해석하며, 직접적인 원인으로 단정하지 않습니다.",
        },
        "source": "deterministic_fallback",
        "facts": facts,
    }


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    # 매출 분석 전체 응답을 늦추지 않도록 요약 전용 호출은 짧게 제한한다.
    # 환경변수로 기존 모델을 명시하면 운영자가 모델을 교체할 수 있다.
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=float(os.getenv("SALES_SUMMARY_TIMEOUT", "15")),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=os.getenv("SALES_SUMMARY_MODEL", "gpt-4.1-mini"),
        temperature=0.1,
        max_tokens=700,
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
            for item in fallback["facts"]["detailedInternalCauses"]
        ]
        forbidden_terms = (
            "POS 일자", "확장해 사용", "다중검정", "결합률", "통제 회귀",
            "대응방안을 권장", "캠페인이 필요", "전략을 추천", "권장합니다",
            "매출이 증가할 것", "매출이 감소할 것",
        )
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
        sections = parsed.get("sections")
        if not isinstance(sections, dict):
            sections = {}
        return {
            "headline": headline,
            "summary": summary,
            "sections": sections,
            # 기존 API 응답 계약의 source 값은 유지하고 실제 모델은 환경변수로 교체한다.
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
    facts = build_diagnosis_facts(report, detailed)
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
    report["AI분석"] = {
        "제목": "매출 원인 상세분석",
        "headline": generated["headline"],
        "summary": generated["summary"],
        "sections": generated.get("sections", {}),
    }
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
