from pydantic import BaseModel
from typing import List, Dict, Any

class ReviewAnalysisRequest(BaseModel):
    reviews: List[str]

class KeywordCount(BaseModel):
    keyword: str
    count: int

class AnalysisSummary(BaseModel):
    total_input_reviews: int
    total_negative_aspects_found: int
    negative_aspects_breakdown: Dict[str, int]

class ReviewAnalysisResponse(BaseModel):
    summary: AnalysisSummary
    top5_negative_keywords: List[KeywordCount]
    details: List[Dict[str, Any]]