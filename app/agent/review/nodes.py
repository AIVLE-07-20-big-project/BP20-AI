import json
from typing import Any, Dict
from openai import AsyncOpenAI

from app.agent.review.tools import fetch_existing_keywords, fetch_store_context
from app.schemas.review import ABSAClusterReport, ReviewAgentState, StoreImprovementReport
from app.services.absa_service import ABSAService

def node_roberta_classify(
    state: ReviewAgentState, absa_service: ABSAService
) -> Dict[str, Any]:
    print("RoBERTa Classify Node 시작")
    roberta_results = []
    for item in state.reviews:
        text = item.review_text.strip()
        if not text:
            continue

        results = absa_service.analyze(text)
        for res in results:
            roberta_results.append(
                {
                    "review_id": item.review_id,
                    "aspect": res.aspect,
                    "sentiment": res.sentiment,
                    "confidence": res.confidence,
                    "text": text,
                }
            )

    return {"roberta_results": roberta_results}

async def node_fetch_db_context(
    state: ReviewAgentState,
) -> Dict[str, Any]:
    # TODO: DB에서 Store 정보 조회 기능 추가 예정
    context = await fetch_store_context(state.store_id)
    existing_keywords = await fetch_existing_keywords(state.store_id)
    return {
        "store_context": context,
        "existing_keywords": existing_keywords,
    }

async def node_llm_cluster(
    state: ReviewAgentState, openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    print("LLM Cluster Node 시작")
    if not state.roberta_results:
        return {
            "final_report": ABSAClusterReport(
                overall_issue_summary="분석할 유효한 리뷰 데이터가 없습니다.",
                keyword_clusters=[],
            )
        }

    prompt = f"""
    너는 외식업 리뷰 데이터 정제 및 키워드 클러스터링 전문가야.
    아래 제공된 [1차 분석 데이터]를 바탕으로, 동일한 속성(aspect)과 감성(sentiment) 내에서 유사한 표현들을 하나의 대표 키워드로 묶어줘.

    "representative_keyword는 original_expressions의 의미를 가장 잘 대표하는 단어로 작성해줘. (예: 밍밍함/무맛인 경우 '음식 간이 심심함/무맛'으로 표기)"
    [매장 컨텍스트]
    {json.dumps(state.store_context, ensure_ascii=False)}

    [매장 기존 등록 대표 키워드]
    {json.dumps(state.existing_keywords, ensure_ascii=False)}
    - 가급적 위 [기존 등록 대표 키워드] 범주 내에서 일치하는 표현이 있다면 해당 키워드로 우선 통합해줘.
    - 기존 키워드로 표현하기 힘든 전혀 새로운 이슈일 때만 새로운 대표 키워드를 만들어줘.

    [1차 분석 데이터]
    {json.dumps(state.roberta_results, ensure_ascii=False)}
    """

    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "외식업 리뷰 키워드 추출 및 클러스터링 전문 에이전트입니다.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ABSAClusterReport,
        temperature=0.1,
    )

    return {"final_report": completion.choices[0].message.parsed}

async def node_generate_improvement_plan(
    state: ReviewAgentState, openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    print("Final Node: AI 개선 우선순위 및 액션플랜 도출 시작")

    if not state.final_report or not state.final_report.keyword_clusters:
        return {
            "improvement_report": StoreImprovementReport(
                executive_summary="분석할 리뷰 키워드 데이터가 충분하지 않습니다.",
                action_items=[],
            )
        }

    past_counts = state.past_keyword_counts or {}
    keyword_trends = []

    for cluster in state.final_report.keyword_clusters:
        keyword = cluster.representative_keyword
        curr_count = cluster.count
        prev_count = past_counts.get(keyword, 0)

        if prev_count == 0:
            trend_str = "이번 달 신규(NEW) 발생"
        else:
            rate = round(((curr_count - prev_count) / prev_count) * 100, 1)
            sign = "+" if rate > 0 else ""
            trend_str = f"전월({prev_count}건) 대비 {sign}{rate}%"

        keyword_trends.append({
            "aspect": cluster.aspect,
            "sentiment": cluster.sentiment,
            "keyword": keyword,
            "count": curr_count,
            "trend": trend_str,
            "sample_expressions": cluster.original_expressions[:3]
        })

    prompt = f"""
    너는 외식업 매장 수익성 개선 및 매장 운영 솔루션 컨설턴트야.
    제공된 [매장 정보], [키워드 트렌드 데이터]를 종합 분석하여 점주가 매장에서 즉시 실행할 수 있는 'AI 개선 우선순위 솔루션 리포트'를 작성해줘.

    [매장 정보]
    {json.dumps(state.store_context, ensure_ascii=False)}

    [키워드 트렌드 데이터 (이번 달 분석 & 전월 대비 변화)]
    {json.dumps(keyword_trends, ensure_ascii=False)}

    [작성 가이드라인]
    1. **우선순위 부여 기준**:
       - **HIGH (긴급)**: 부정 감성이면서 '신규 발생(NEW)'했거나 '전월 대비 급증(+100% 이상)'한 키워드, 혹은 고객 이탈에 직접적인 악영향을 주는 이슈.
       - **MEDIUM (보통/주의)**: 지속적으로 발생하는 부정 키워드 또는 관리가 필요한 항목.
       - **LOW (유지/권장)**: 긍정 키워드 유지 방안 또는 소폭의 개선 항목.
    2. **원인 추정 (problem_cause)**: sample_expressions(고객 실제 표현)를 분석하여 단순 추측이 아닌 현장 기반의 구체적 원인을 도출할 것.
    3. **액션 플랜 (action_plan)**: '직원 교육을 하세요' 같은 뻔한 소리 금지. 매장 업종과 컨텍스트에 맞춰 현장에서 바로 적용 가능한 구체적 가이드라인을 제시할 것. (예: '피크타임(12시~14시) 전담 서빙 체크리스트 도입', '파스타 간 맞춤용 소스 레시피 정량 용리 도구 배치' 등)
    """

    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "외식업 매장 컨설팅 분석 전문 에이전트입니다. 데이터 기반의 현실적이고 실행 가능한 솔루션을 제공합니다.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=StoreImprovementReport,
        temperature=0.2,
    )
    print(completion.choices[0].message.parsed)
    return {"improvement_report": completion.choices[0].message.parsed}