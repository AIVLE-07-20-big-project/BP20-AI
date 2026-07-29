from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from app.agent.review.nodes import (
    node_fetch_db_context,
    node_llm_cluster,
    node_roberta_classify,
)
from app.schemas.review import ReviewAgentState
from app.services.absa_service import ABSAService


def build_review_agent_graph(
    absa_service: ABSAService, openai_client: AsyncOpenAI
):
    """LangGraph 생성 및 노드 연결 파이프라인 빌더"""
    builder = StateGraph(ReviewAgentState)

    def _roberta_node(state: ReviewAgentState):
        return node_roberta_classify(state, absa_service)

    async def _llm_node(state: ReviewAgentState):
        return await node_llm_cluster(state, openai_client)

    builder.add_node("roberta_classify", _roberta_node)
    builder.add_node("fetch_db_context", node_fetch_db_context)
    builder.add_node("llm_cluster", _llm_node)

    builder.add_edge(START, "roberta_classify")
    builder.add_edge("roberta_classify", "fetch_db_context")
    builder.add_edge("fetch_db_context", "llm_cluster")
    builder.add_edge("llm_cluster", END)

    return builder.compile()