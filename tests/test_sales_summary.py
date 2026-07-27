import json

from app.services.sales_summary import attach_sales_summary, generate_sales_summary
from scripts.modeling.detailed_sales_external_analysis import build_root_cause_analysis


def _report():
    return {
        "매출분석": {
            "전년동분기대비": "3.8%",
            "전분기대비": "-2.6%",
        },
        "간단분석 정보요약": {
            "유동인구 많은 요일": "금요일",
            "유동인구 많은 시간대": "17-21시",
        },
        "분석결과 해설": "기존 상권 설명",
    }


def _detailed():
    return {
        "internalAnalysis": {
            "period": {
                "currentStart": "2026-06-01",
                "currentEnd": "2026-06-30",
                "baselineStart": "2026-05-02",
                "baselineEnd": "2026-05-31",
            },
            "summary": {"revenueChangePct": -4.119},
        },
        "rootCauseAnalysis": {
            "change": {"direction": "decrease"},
            "headline": "기존 제목",
            "narrative": "기존 기술 설명",
            "internalDrivers": [
                {"factor": "transaction_count", "contributionAmount": -100},
                {"factor": "average_order_value", "contributionAmount": 20},
            ],
            "internalDetailedDrivers": [
                {"label": "12시 시간대"},
                {"label": "논커피음료 카테고리"},
            ],
            "externalDrivers": [
                {"label": "평균 기온", "confidence": "high"},
                {"label": "풍속", "confidence": "medium"},
            ],
            "possibleExternalDrivers": [],
        },
    }


def test_summary_keeps_all_meaningful_causes():
    output = {
        "headline": "가장 큰 내부 원인은 거래 건수 감소입니다.",
        "summary": (
            "12시 시간대와 논커피음료 카테고리 매출이 감소했습니다. "
            "평균 기온과 풍속 변화도 관련된 것으로 보입니다."
        ),
    }

    result = generate_sales_summary(
        _report(),
        _detailed(),
        llm=lambda _prompt: json.dumps(output, ensure_ascii=False),
    )

    assert result["source"] == "gpt-4.1"
    assert all(
        label in result["summary"]
        for label in ("12시 시간대", "논커피음료 카테고리", "평균 기온", "풍속")
    )


def test_summary_falls_back_when_llm_omits_a_cause():
    result = generate_sales_summary(
        _report(),
        _detailed(),
        llm=lambda _prompt: json.dumps({
            "headline": "거래 건수가 감소했습니다.",
            "summary": "12시 시간대 매출이 감소했습니다.",
        }, ensure_ascii=False),
    )

    assert result["source"] == "deterministic_fallback"
    assert "풍속" in result["summary"]


def test_attach_replaces_user_fields_and_preserves_technical_text():
    report = _report()
    detailed = _detailed()

    attach_sales_summary(
        report,
        detailed,
        llm=lambda _prompt: json.dumps({
            "headline": "거래 건수 감소가 가장 큰 원인입니다.",
            "summary": (
                "12시 시간대와 논커피음료 카테고리 매출이 감소했습니다. "
                "평균 기온과 풍속 변화도 관련된 것으로 보입니다."
            ),
        }, ensure_ascii=False),
    )

    root = detailed["rootCauseAnalysis"]
    assert report["분석결과 해설"] is None
    assert report["기술_분석결과_해설"] == "기존 상권 설명"
    assert root["technicalHeadline"] == "기존 제목"
    assert root["technicalNarrative"] == "기존 기술 설명"
    assert root["headline"] == root["userHeadline"]
    assert root["narrative"] == root["userSummary"]


def test_root_cause_keeps_material_details_and_deduplicates_temperature():
    internal = {
        "summary": {"revenueChange": -100, "revenueChangePct": -10},
        "revenueFormulaDrivers": [
            {"factor": "transaction_count", "contributionAmount": -80},
            {"factor": "average_order_value", "contributionAmount": -20},
        ],
        "dimensionDrivers": {
            "store": [
                {"value": "강남역점", "contributionAmount": -100},
            ],
            "hour": [
                {"value": "12", "contributionAmount": -50},
                {"value": "13", "contributionAmount": -30},
                {"value": "14", "contributionAmount": -3},
            ],
            "category": [
                {"value": "논커피음료", "contributionAmount": -70},
                {"value": "디저트", "contributionAmount": -20},
            ],
        },
    }
    external = [
        {
            "factor": "temperature_c", "label": "평균 기온",
            "estimatedContributionAmount": -30, "validationGainPct": 20,
            "confidence": "high", "direction": "negative", "joinRate": 1.0,
        },
        {
            "factor": "temperature_min_c", "label": "최저 기온",
            "estimatedContributionAmount": -20, "validationGainPct": 10,
            "confidence": "high", "direction": "negative", "joinRate": 1.0,
        },
        {
            "factor": "wind_speed_ms", "label": "풍속",
            "estimatedContributionAmount": -10, "validationGainPct": 5,
            "confidence": "medium", "direction": "negative", "joinRate": 1.0,
        },
    ]

    result = build_root_cause_analysis(internal, external, data_warnings=[])

    detail_labels = [item["label"] for item in result["internalDetailedDrivers"]]
    external_labels = [item["label"] for item in result["externalDrivers"]]
    assert detail_labels == [
        "논커피음료 카테고리",
        "12시 시간대",
        "13시 시간대",
        "디저트 카테고리",
    ]
    assert external_labels == ["평균 기온", "풍속"]
