from openai import AsyncOpenAI
from collections import defaultdict

from app.agent.review.builder import build_review_agent_graph
from app.schemas.review import (
    ABSAClusterReport,
    ABSAAgentResponse,
    BatchReviewRequest,
    ReviewAgentState,
    AspectSentiment,
    ReviewResponse,
)
from app.services.absa_service import ABSAService

class ABSAGraphRunner:

    def __init__(self, absa_service: ABSAService, openai_api_key: str):
        self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        self.graph = build_review_agent_graph(absa_service, self.openai_client)

    async def run(self, payload: BatchReviewRequest) -> ABSAAgentResponse:
        initial_state = ReviewAgentState(
            store_id=payload.store_id, reviews=payload.reviews
        )

        final_state = await self.graph.ainvoke(initial_state)
        grouped_results = defaultdict(list)
        for item in final_state.get("roberta_results", []):
            grouped_results[item["review_id"]].append(
                AspectSentiment(
                    aspect=item["aspect"],
                    sentiment=item["sentiment"],
                    confidence=item.get("confidence", 1.0),
                )
            )

        reviews_analysis = [
            ReviewResponse(review_id=r_id, results=results)
            for r_id, results in grouped_results.items()
        ]
        report: ABSAClusterReport = final_state["final_report"]

        return ABSAAgentResponse(
            store_id=payload.store_id,
            summary=report.overall_issue_summary,
            total_reviews_analyzed=len(payload.reviews),
            reviews_analysis=reviews_analysis,
            clusters=report.keyword_clusters,
        )