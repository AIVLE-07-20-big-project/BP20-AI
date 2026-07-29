from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional

class ReviewRequest(BaseModel):
    review_id: int
    review_text: str

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "review_id": 1,
                "review_text": (
                    "파스타는 정말 맛있는데 직원이 너무 불친절하고 가격이 비싸요."
                )
            }]
        }
    }

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    confidence: float

class ReviewResponse(BaseModel):
    review_id: int
    results: List[AspectSentiment]

class ReviewItem(BaseModel):
    review_id: int
    review_text: str

class BatchReviewRequest(BaseModel):
    store_id: int
    reviews: List[ReviewItem]

class KeywordClusterItem(BaseModel):
    aspect: str = Field(
        description="속성 카테고리 (food, service, convenience, price, atmosphere)"
    )
    sentiment: str = Field(
        description="감성 (positive, negative, neutral)"
    )
    representative_keyword: str = Field(
        description="통합된 대표 키워드명 (예: '주차 공간 부족', '직원 불친절', '음식 간이 큼')"
    )
    count: int = Field(
        description="해당 키워드로 묶인 원본 리뷰 총 개수"
    )
    matched_review_ids: List[int] = Field(
        description="해당 키워드에 속하는 review_id 목록"
    )
    original_expressions: List[str] = Field(
        description="하나로 묶인 원본 표현 예시 (최대 3~5개)"
    )

class ABSAClusterReport(BaseModel):
    overall_issue_summary: str = Field(
        description="점주가 한눈에 파악할 수 있는 전체 종합 분석 및 주요 이슈 한줄 요약"
    )
    keyword_clusters: List[KeywordClusterItem]


class ABSAAgentResponse(BaseModel):
    store_id: int
    summary: str
    total_reviews_analyzed: int
    clusters: List[KeywordClusterItem]

class ReviewAgentState(BaseModel):
    store_id: int
    reviews: List[ReviewItem]
    roberta_results: List[Dict[str, Any]] = []
    store_context: Optional[Dict[str, Any]] = None
    existing_keywords: List[str] = []
    final_report: Optional[ABSAClusterReport] = None