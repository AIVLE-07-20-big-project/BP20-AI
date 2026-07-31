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

class BatchReviewRequest(BaseModel):
    store_id: int
    reviews: List[ReviewRequest]

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

class StoreImprovementReport(BaseModel):
    executive_summary: str = Field(
        description="점주를 위한 종합 총평 및 이번 달 매장 운영 핵심 한줄 가이드"
    )
    action_items: List[PriorityActionItem] = Field(
        description="우선순위 순으로 정렬된 개선 과제 리스트"
    )
    
class ABSAAgentResponse(BaseModel):
    store_id: int
    summary: str
    total_reviews_analyzed: int
    reviews_analysis: List[ReviewResponse]
    clusters: List[KeywordClusterItem]
    improvement_report: Optional[StoreImprovementReport] = None

class ReviewAgentState(BaseModel):
    store_id: int
    reviews: List[ReviewRequest]
    roberta_results: List[Dict[str, Any]] = []
    store_context: Optional[Dict[str, Any]] = None
    existing_keywords: List[str] = []
    final_report: Optional[ABSAClusterReport] = None

class PriorityActionItem(BaseModel):
    priority: str = Field(
        description="우선순위 (HIGH: 긴급, MEDIUM: 보통/주의, LOW: 유지/권장)"
    )
    aspect: str = Field(
        description="속성 카테고리 (food, service, convenience, price, atmosphere)"
    )
    keyword: str = Field(
        description="대상 대표 키워드 (예: '직원 불친절/응대 미흡')"
    )
    trend_summary: str = Field(
        description="키워드 수치 및 트렌드 요약 (예: '이번 달 3건 발생 (+200% 급증)', '신규 발생 3건')"
    )
    problem_cause: str = Field(
        description="리뷰 표현 기반 주요 원인 추정 (예: '피크타임 대기시간 길어짐에 따른 불친절 응대')"
    )
    action_plan: str = Field(
        description="점주가 현장에서 바로 실행 가능한 구체적 개선 행동 솔루션"
    )
    expected_outcome: str = Field(
        description="개선 실행 시 예상되는 효과"
    )

class ReviewAgentState(BaseModel):
    store_id: int
    reviews: List[ReviewRequest]
    roberta_results: List[Dict[str, Any]] = []
    store_context: Optional[Dict[str, Any]] = None
    existing_keywords: List[str] = []
    past_keyword_counts: Optional[Dict[str, int]] = {}
    final_report: Optional[ABSAClusterReport] = None
    improvement_report: Optional[StoreImprovementReport] = None