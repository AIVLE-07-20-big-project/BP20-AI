from pydantic import BaseModel, Field
from typing import List

class ReviewRequest(BaseModel):
    review_id: int = Field(..., description="리뷰 식별자") 
    human_prompt: str = Field(..., description="분석할 리뷰 원문")

class AspectSentimentResult(BaseModel):
    aspect: str = Field(..., description="추출된 속성 (예: 맛, 배송, 가격)")
    sentiment: str = Field(..., description="감정 (positive, negative, neutral)")
    confidence: float = Field(..., description="신뢰도 점수")

class ReviewAnalyzeResponse(BaseModel):
    review_id: int
    analysis_results: List[AspectSentimentResult]
    needs_rag: bool = Field(
        description="리뷰에 난해한 표현이 있거나 신조어로 추정되는 단어가 있다면 True"
    )