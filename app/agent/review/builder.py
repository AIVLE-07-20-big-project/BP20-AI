from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from app.agent.review.nodes import (
    node_execute_store_tool,
    node_fetch_mandatory_db,
    node_generate_improvement_plan,
    node_llm_cluster,
    node_refine_low_confidence_classifications,
    node_roberta_classify,
)
from app.core.config import settings
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

    async def _refine_node(state: ReviewAgentState):
        return await node_refine_low_confidence_classifications(state, openai_client)

    async def _node_improvement(state: ReviewAgentState):
        return await node_generate_improvement_plan(state, openai_client)

    builder.add_node("roberta_classify", _roberta_node)
    builder.add_node("refine_low_confidence", _refine_node)
    builder.add_node("fetch_mandatory_db", node_fetch_mandatory_db)
    builder.add_node("llm_cluster", _llm_node)
    builder.add_node("execute_store_tool", node_execute_store_tool) # 💡 Tool 실행 전용 노드
    builder.add_node("improvement", _node_improvement)

    def route_after_llm_cluster(state: ReviewAgentState) -> str:
        if state.tool_calls:
            return "execute_store_tool"
        return "improvement" if state.generate_improvement else "end"

    def route_after_start(state: ReviewAgentState) -> str:
        # Monthly reports reuse ABSA results saved by Spring Boot.
        return "fetch_mandatory_db" if state.roberta_results else "roberta_classify"

    def route_after_roberta(state: ReviewAgentState) -> str:
        threshold = settings.LLM_REFINEMENT_CONFIDENCE_THRESHOLD
        has_low_confidence = any(
            result.get("confidence", 0) < threshold
            for result in state.roberta_results
        )
        return "refine_low_confidence" if has_low_confidence else "fetch_mandatory_db"

    builder.add_conditional_edges(
        START,
        route_after_start,
        {
            "roberta_classify": "roberta_classify",
            "fetch_mandatory_db": "fetch_mandatory_db",
        },
    )
    builder.add_conditional_edges(
        "roberta_classify",
        route_after_roberta,
        {
            "refine_low_confidence": "refine_low_confidence",
            "fetch_mandatory_db": "fetch_mandatory_db",
        },
    )
    builder.add_edge("refine_low_confidence", "fetch_mandatory_db")
    builder.add_edge("fetch_mandatory_db", "llm_cluster")

    builder.add_conditional_edges(
        "llm_cluster",
        route_after_llm_cluster,
        {
            "execute_store_tool": "execute_store_tool",
            "improvement": "improvement",
            "end": END,
        },
    )

    builder.add_edge("execute_store_tool", "llm_cluster")
    builder.add_edge("improvement", END)

    return builder.compile()
