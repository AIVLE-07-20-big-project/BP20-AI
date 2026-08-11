import json
import os
from typing import Any, Dict, List
import httpx
from openai import AsyncOpenAI

from app.agent.review.tools import fetch_store_context
from app.schemas.review import (
    ABSAClusterReport,
    LLMRefinementResponse,
    ReviewAgentState,
    StoreImprovementReport,
)
from app.services.absa_service import ABSAService
from app.core.config import settings
from app.agent.review.internal_client import get_internal_headers

SPRINGBOOT_BASE_URL = os.getenv("SPRINGBOOT_BASE_URL", "http://springboot:8080")

def node_roberta_classify(
    state: ReviewAgentState, absa_service: ABSAService
) -> Dict[str, Any]:
    print("1RoBERTa Classify Node 시작")
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


async def node_refine_low_confidence_classifications(
    state: ReviewAgentState, openai_client: AsyncOpenAI
) -> Dict[str, Any]:
    """Use the LLM only to verify ABSA classifications below the confidence threshold."""
    threshold = settings.LLM_REFINEMENT_CONFIDENCE_THRESHOLD
    candidates = [
        item for item in state.roberta_results
        if item.get("confidence", 0) < threshold
    ]
    if not candidates:
        return {"roberta_results": state.roberta_results}

    allowed_sentiments = [
        label for label in settings.LABEL_MAP.values() if label != "none"
    ]
    prompt = f"""
    You are a conservative Korean restaurant-review ABSA verifier. Your job is to
    validate the original RoBERTa label, not to replace it by default. Review only
    the uncertain classifications below. Keep review_id and aspect unchanged.

    Set should_override to true only when explicit, aspect-specific wording in the
    review clearly contradicts the original sentiment. If the evidence is mixed,
    implicit, unrelated to the given aspect, or still ambiguous, preserve the
    original sentiment and set should_override to false. Do not infer unstated
    facts, and do not use general restaurant knowledge. Choose sentiment only from
    {allowed_sentiments}. Confidence must represent your confidence in the returned
    sentiment on a 0-100 scale.

    Uncertain classifications:
    {json.dumps(candidates, ensure_ascii=False)}
    """

    try:
        completion = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Verify uncertain aspect-based sentiment labels using only the supplied review text.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=LLMRefinementResponse,
            temperature=0,
        )
        refined = completion.choices[0].message.parsed
        if refined is None:
            return {"roberta_results": state.roberta_results}

        updates = {
            (item.review_id, item.aspect): item
            for item in refined.refinements
            if item.should_override and item.sentiment in allowed_sentiments
        }
        results: List[Dict[str, Any]] = []
        for item in state.roberta_results:
            update = updates.get((item["review_id"], item["aspect"]))
            if update is None:
                results.append(item)
                continue
            results.append({
                **item,
                "sentiment": update.sentiment,
                "confidence": update.confidence,
                "classification_source": "llm_refined",
            })

        return {"roberta_results": results}
    except Exception as e:
        # Preserve the RoBERTa prediction so a transient LLM failure does not block analysis.
        print(f"[node_refine_low_confidence_classifications] LLM refinement failed: {e}")
        return {"roberta_results": state.roberta_results}


async def node_fetch_mandatory_db(
    state: ReviewAgentState,
) -> Dict[str, Any]:
    print(" Mandatory DB Fetch Node: 키워드 및 히스토리 조회 시작")

    headers = get_internal_headers()
    url = f"{SPRINGBOOT_BASE_URL}/api/internal/stores/{state.store_id}/keywords"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            response.raise_for_status()

            res_body = response.json()
            raw_keywords = res_body.get("data", res_body) if isinstance(res_body, dict) else res_body

            if not isinstance(raw_keywords, list):
                raw_keywords = []

            existing_keywords = list({
                item["keyword"] for item in raw_keywords if item.get("keyword")
            })

            past_keyword_counts = {
                item["keyword"]: item.get("count", 0)
                for item in raw_keywords
                if item.get("keyword")
            }

            print(f"DB 조회 성공! 기존 키워드 {len(existing_keywords)}개 확보: {existing_keywords}")

            return {
                "existing_keywords": existing_keywords,
                "past_keyword_counts": past_keyword_counts,
            }

        except Exception as e:
            print(f"[node_fetch_mandatory_db] Spring Boot API 호출 실패: {e}")
            return {
                "existing_keywords": [],
                "past_keyword_counts": {},
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

    has_context = state.store_context is not None
    store_info_str = (
        json.dumps(state.store_context, ensure_ascii=False)
        if has_context
        else "미제공 (리뷰만으로 업종 파악이 모호한 경우에만 'fetch_store_context' 도구를 호출하세요)"
    )

    prompt = f"""
    너는 외식업 리뷰 데이터 정제 및 키워드 클러스터링 전문가야.
    아래 제공된 [1차 분석 데이터]를 바탕으로, 동일한 속성(aspect)과 감성(sentiment) 내에서 유사한 표현들을 하나의 대표 키워드로 묶어줘.

    representative_keyword는 original_expressions의 의미를 가장 잘 대표하는 단어로 작성해줘. (예: 밍밍함/무맛인 경우 '음식 간이 심심함/무맛'으로 표기)
    
    [매장 컨텍스트]
    {store_info_str}

    [매장 기존 등록 대표 키워드]
    {json.dumps(state.existing_keywords, ensure_ascii=False)}
    - 가급적 위 [기존 등록 대표 키워드] 범주 내에서 일치하는 표현이 있다면 해당 키워드로 우선 통합해줘.
    - 기존 키워드로 표현하기 힘든 전혀 새로운 이슈일 때만 새로운 대표 키워드를 만들어줘.

    [1차 분석 데이터]
    {json.dumps(state.roberta_results, ensure_ascii=False)}
    """

    if has_context:
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
        return {"final_report": completion.choices[0].message.parsed, "tool_calls": None}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fetch_store_context",
                "description": "리뷰 데이터만으로 가게가 어떤 메뉴/업종인지 판단하기 극도로 모호할 때 매장 기본 정보를 DB에서 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "store_id": {"type": "integer", "description": "조회할 매장 ID"}
                    },
                    "required": ["store_id"],
                },
            },
        }
    ]
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "외식업 리뷰 분석 에이전트입니다. 리뷰가 명확하면 도구를 부르지 말고 바로 클러스터링하세요."},
            {"role": "user", "content": prompt},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )

    message = response.choices[0].message

    if message.tool_calls:
        print("LLM 판단: 리뷰 맥락이 모호하여 store_context 조회가 필요합니다.")
        return {"tool_calls": message.tool_calls}

    print("LLM 판단: 리뷰만으로 맥락이 충분하여 Tool 호출 없이 클러스터링 진행합니다.")
    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "외식업 리뷰 키워드 추출 및 클러스터링 전문 에이전트입니다."},
            {"role": "user", "content": prompt},
        ],
        response_format=ABSAClusterReport,
        temperature=0.1,
    )
    return {"final_report": completion.choices[0].message.parsed, "tool_calls": None}

async def node_execute_store_tool(
    state: ReviewAgentState,
) -> Dict[str, Any]:
    print("🛠️ Tool Execution Node: LLM 요청으로 Spring Boot API(fetch_store_context) 실행 중...")
    try:
        context = await fetch_store_context.ainvoke({
            "store_id": state.store_id,
        })
        
    except Exception as e:
        print(f"[node_execute_store_tool] 실행 중 에러 발생: {e}")
        context = {"name": "알 수 없음", "category": "일반 매장", "address": ""}

    return {"store_context": context, "tool_calls": None}

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

    store_info_str = (
        json.dumps(state.store_context, ensure_ascii=False)
        if state.store_context
        else "일반 외식업 매장"
    )

    prompt = f"""
    너는 외식업 매장 수익성 개선 및 매장 운영 솔루션 컨설턴트야.
    제공된 [매장 정보], [키워드 트렌드 데이터]를 종합 분석하여 점주가 매장에서 즉시 실행할 수 있는 'AI 개선 우선순위 솔루션 리포트'를 작성해줘.

    [매장 정보]
    {store_info_str}

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
    return {"improvement_report": completion.choices[0].message.parsed}
