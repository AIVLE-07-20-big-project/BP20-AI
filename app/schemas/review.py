from pydantic import BaseModel
from typing import List

class ReviewRequest(BaseModel):
    review_text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"review_text": "파스타는 정말 맛있는데 직원이 너무 불친절하고 가격이 비싸요."}
            ]
        }
    }

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: str
    confidence: float

class ReviewResponse(BaseModel):
    review_text: str
    results: List[AspectSentiment]