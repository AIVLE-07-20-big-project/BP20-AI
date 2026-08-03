# 매출형 전략검증(적용전/적용후 지표 비교) 결과를 사람이 읽기 쉬운 보고서로 요약한다.
# app/services/sales_summary.py와 같은 패턴(LLM 호출 + 결정론적 fallback)을 따른다.
from __future__ import annotations

import json
import os
from typing import Any, Callable

from app.schemas.effect_verification_schema import MetricResult

STRATEGY_VALIDATION_PROMPT = """당신은 소상공인을 위해 매출 전략 실행 결과를 설명하는 분석가다.
전략 적용 전후 지표를 바탕으로, 사용자가 전략의 성과와 한계를 쉽게 이해할 수 있는 보고서를 작성한다.

[작성 목적]
단순히 지표를 나열하지 말고 다음을 설명한다.

1. 전략 실행 후 매출이 어떻게 변했는지
2. 어떤 지표가 매출 변화와 함께 개선됐는지
3. 어떤 지표는 악화됐는지
4. 전략이 어떤 측면에서 효과가 있었을 가능성이 있는지
5. 다음 전략에서 무엇을 보완해야 하는지
6. 현재 데이터만으로 전략 효과를 확정할 수 있는지

[핵심 규칙]
1. 제공된 수치만 사용하고 새로운 수치나 원인을 만들지 않는다.
2. 전략 실행 이후 지표가 변했다고 해서 전략의 직접적인 인과효과로 단정하지 않는다.
3. 다음 표현을 우선 사용한다.
   - 긍정적인 변화가 나타났습니다.
   - 전략과 관련된 변화로 볼 수 있습니다.
   - 도움이 되었을 가능성이 있습니다.
   - 현재 결과만으로 직접적인 효과를 단정하기 어렵습니다.
4. 매출 증가 원인을 설명할 때 방문 건수, 객단가, 신규 고객, 재방문율 변화를 함께 고려한다.
5. 증가한 지표와 감소한 지표를 모두 설명한다.
6. 동일한 의미의 수치를 반복하지 않는다.
7. '성공했습니다', '효과가 입증됐습니다', '전략으로 인해 증가했습니다'처럼 인과관계를 확정하는 표현을 사용하지 않는다.
8. 추천 대상 매출과 전체 매출의 증가율이 같거나 유사하면 전략의 추가 효과가 명확하지 않다고 설명한다.
9. 추천 대상 매출 증가율이 전체 매출 증가율보다 높을 때만 추가 개선 가능성을 언급한다.
10. 신규 고객이 증가했지만 객단가나 재방문율이 감소했다면 고객 유입에는 긍정적이지만 구매 전환과 유지에는 보완이 필요하다고 설명한다.
11. 각 지표를 한 줄씩 나열하지 말고 의미가 연결되는 지표끼리 묶어 설명한다.
12. 기술 용어 없이 자연스러운 한국어 보고서체로 작성한다.

[출력 구조]
{
  "headline": "전략 실행 결과를 요약하는 한 문장",
  "summary": "전체 평가",
  "sections": {
    "performance": "매출과 핵심 성과",
    "positiveChanges": "긍정적으로 변화한 지표",
    "negativeChanges": "악화되거나 보완이 필요한 지표",
    "interpretation": "전략 효과에 대한 신중한 해석",
    "nextAction": "다음 실행에서 보완할 방향"
  }
}

[작성 예시]
매출이 증가하고 방문 건수와 신규 고객 수가 늘어 고객 유입 측면에서는
긍정적인 변화가 나타났습니다. 반면 객단가와 재방문율은 감소해,
신규 방문을 실제 구매 확대와 재방문으로 연결하는 보완 전략이 필요합니다.

JSON 객체 하나만 반환한다."""

METRIC_LABELS: dict[str, str] = {
    "target_sales": "추천 대상 매출",
    "total_sales": "전체 매출",
    "visit_count": "방문 건수",
    "average_order_value": "객단가",
    "revisit_rate": "재방문율",
    "coupon_usage_rate": "쿠폰 사용률",
    "new_customer_count": "신규 고객 수",
    "dormant_customer_return_count": "장기 미방문 고객 복귀",
}

_MONEY_METRICS = {"target_sales", "total_sales", "average_order_value"}
_POINT_METRICS = {"revisit_rate", "coupon_usage_rate"}
_EXCLUDED_FROM_LISTS = {"target_sales", "total_sales", "visit_count"}

_FORBIDDEN_CAUSAL_TERMS = (
    "성공했습니다", "효과가 입증", "전략으로 인해 증가", "전략 덕분에 증가", "효과가 확인됐습니다",
)


def _metric_map(metric_results: list[MetricResult]) -> dict[str, MetricResult]:
    return {m.metric_name: m for m in metric_results}


def _eul_reul(word: str) -> str:
    """받침 유무에 따라 '을'/'를' 조사를 고른다."""

    if not word:
        return "를"
    code = ord(word[-1]) - 0xAC00
    if 0 <= code <= 11171:
        return "를" if code % 28 == 0 else "을"
    return "를"


def _format_change(metric: MetricResult) -> str:
    """지표 하나의 변화를 사람이 읽는 한국어 구절로 만든다(문장 종결 없이)."""

    label = METRIC_LABELS.get(metric.metric_name, metric.metric_name)
    direction_word = "증가" if (metric.change_value or 0) >= 0 else "감소"

    if metric.metric_name in _MONEY_METRICS and metric.change_value is not None:
        return f"{label} {abs(round(metric.change_value)):,.0f}원 {direction_word}"
    if metric.metric_name in _POINT_METRICS and metric.change_value is not None:
        point_word = "상승" if metric.change_value >= 0 else "하락"
        return f"{label} {abs(metric.change_value):.2f}%p {point_word}"
    if metric.change_rate is not None:
        return f"{label} {abs(metric.change_rate):.2f}% {direction_word}"
    if metric.change_value is not None:
        return f"{label} {abs(round(metric.change_value)):,.0f}명 {direction_word}"
    return f"{label} 변화 확인 불가"


def _fact_payload(metric_results: list[MetricResult]) -> dict[str, Any]:
    return {
        m.metric_name: {
            "label": METRIC_LABELS.get(m.metric_name, m.metric_name),
            "before": m.before_value,
            "after": m.after_value,
            "change_value": m.change_value,
            "change_rate": m.change_rate,
            "improved": m.improved,
        }
        for m in metric_results
    }


def build_prompt(metric_results: list[MetricResult]) -> str:
    facts = _fact_payload(metric_results)
    return (
        "다음은 매출형 전략검증의 적용전/적용후 지표 비교 결과다. 이 사실만 사용해 보고서를 작성하라.\n\n"
        f"[분석 사실]\n{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def _fallback_report(metric_results: list[MetricResult]) -> dict[str, Any]:
    m = _metric_map(metric_results)
    target = m.get("target_sales")
    total = m.get("total_sales")
    visit = m.get("visit_count")
    aov = m.get("average_order_value")
    revisit = m.get("revisit_rate")
    new_cust = m.get("new_customer_count")

    performance_parts = []
    if target is not None and target.change_value is not None:
        performance_parts.append(f"{_format_change(target)}했습니다")
    if visit is not None and visit.change_rate is not None:
        performance_parts.append(f"{_format_change(visit)}했습니다")
    performance = ". ".join(performance_parts) + "." if performance_parts else "매출 변화 데이터가 부족합니다."

    # change_value가 0에 가까우면(실질적으로 변화 없음) improved 플래그와 무관하게 목록에서 뺀다
    # — dormant_customer_return_count처럼 0→0인 지표를 "악화"로 잘못 서술하지 않기 위함이다.
    def _has_real_change(metric: MetricResult) -> bool:
        return metric.change_value is not None and abs(metric.change_value) > 1e-9

    positive_keys = [
        k for k, v in m.items() if v.improved and _has_real_change(v) and k not in _EXCLUDED_FROM_LISTS
    ]
    negative_keys = [
        k for k, v in m.items() if v.improved is False and _has_real_change(v) and k not in _EXCLUDED_FROM_LISTS
    ]

    positive_changes = (
        ", ".join(_format_change(m[k]) for k in positive_keys) + "했습니다."
        if positive_keys else "매출·방문 건수 외에 뚜렷하게 개선된 지표는 없습니다."
    )
    negative_changes = (
        ", ".join(_format_change(m[k]) for k in negative_keys) + "했습니다."
        if negative_keys else "뚜렷하게 악화된 지표는 없습니다."
    )

    interpretation_parts = []
    if target and total and target.change_rate is not None and total.change_rate is not None:
        diff = round(target.change_rate - total.change_rate, 2)
        if abs(diff) < 1:
            interpretation_parts.append(
                "추천 대상 매출과 전체 매출이 비슷한 비율로 변화해, 현재 데이터만으로는 전략이 만든 "
                "추가 효과를 별도로 구분하기 어렵습니다."
            )
        elif diff > 0:
            interpretation_parts.append(
                f"추천 대상 매출 증가율이 전체 매출 증가율보다 {diff}%p 높아, 전략이 대상 시간대에 "
                "도움이 되었을 가능성이 있습니다."
            )
        else:
            interpretation_parts.append(
                "추천 대상 매출 증가율이 전체 매출 증가율보다 낮아, 이 기간의 변화가 전략 외 다른 "
                "요인과 관련됐을 가능성이 있습니다."
            )
    if new_cust is not None and new_cust.improved and (
        (aov is not None and aov.improved is False) or (revisit is not None and revisit.improved is False)
    ):
        interpretation_parts.append(
            "신규 고객 유입에는 긍정적인 변화가 나타났지만, 구매 전환과 재방문 유지 측면은 보완이 "
            "필요해 보입니다."
        )
    interpretation = " ".join(interpretation_parts) if interpretation_parts else (
        "현재 결과만으로 직접적인 효과를 단정하기 어렵습니다."
    )

    next_action_targets = []
    if aov is not None and aov.improved is False:
        next_action_targets.append("세트 상품·추가 구매 유도 구성")
    if revisit is not None and revisit.improved is False:
        next_action_targets.append("재방문 혜택")
    if next_action_targets:
        joined_targets = ", ".join(next_action_targets)
        next_action = (
            f"다음 전략에서는 {joined_targets}{_eul_reul(next_action_targets[-1])} 함께 적용해 "
            "고객당 매출과 재방문율을 보완할 필요가 있습니다."
        )
    else:
        next_action = "현재 개선 흐름을 유지하면서 다음 기간에도 같은 지표로 다시 확인해 보는 것을 권장합니다."

    headline = (
        f"전략 적용 후 {_format_change(target)}했습니다." if target is not None
        else "전략 적용 결과를 확인했습니다."
    )
    summary = " ".join([performance, positive_changes, negative_changes, interpretation, next_action])

    return {
        "headline": headline,
        "summary": summary,
        "sections": {
            "performance": performance,
            "positiveChanges": positive_changes,
            "negativeChanges": negative_changes,
            "interpretation": interpretation,
            "nextAction": next_action,
        },
        "source": "deterministic_fallback",
    }


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        timeout=float(os.getenv("STRATEGY_REPORT_TIMEOUT", "15")),
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=os.getenv("STRATEGY_REPORT_MODEL", "gpt-4.1-mini"),
        temperature=0.1,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": STRATEGY_VALIDATION_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""


_REQUIRED_SECTION_KEYS = ("performance", "positiveChanges", "negativeChanges", "interpretation", "nextAction")


def generate_strategy_report(
    metric_results: list[MetricResult],
    *,
    llm: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """매출형 전략검증 지표를 요약한 구조화 보고서를 만든다. LLM 미설정/실패 시 결정론적 요약으로 대체한다."""

    fallback = _fallback_report(metric_results)
    if llm is None and not os.getenv("OPENAI_API_KEY"):
        return fallback

    try:
        raw = (llm or _call_openai)(build_prompt(metric_results))
        parsed = json.loads(raw)
        headline = str(parsed.get("headline") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        sections = parsed.get("sections") or {}
        if not headline or not summary or not all(
            str(sections.get(key) or "").strip() for key in _REQUIRED_SECTION_KEYS
        ):
            return {**fallback, "fallbackReason": "LLM 응답에 필수 항목이 없습니다."}

        full_text = f"{headline} {summary} " + " ".join(
            str(sections.get(key, "")) for key in _REQUIRED_SECTION_KEYS
        )
        if any(term in full_text for term in _FORBIDDEN_CAUSAL_TERMS):
            return {**fallback, "fallbackReason": "LLM 응답에 인과관계를 확정하는 표현이 포함됐습니다."}

        return {
            "headline": headline,
            "summary": summary,
            "sections": {key: str(sections.get(key, "")) for key in _REQUIRED_SECTION_KEYS},
            "source": "gpt-4.1",
        }
    except Exception as exc:  # noqa: BLE001 - LLM 실패는 fallback으로 흡수한다.
        return {**fallback, "fallbackReason": f"{type(exc).__name__}: {exc}"}
