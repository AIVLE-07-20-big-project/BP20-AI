import json
from typing import Any, Dict
from openai import AsyncOpenAI

from app.agent.review.tools import fetch_existing_keywords, fetch_store_context
from app.schemas.review import ABSAClusterReport, ReviewAgentState
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