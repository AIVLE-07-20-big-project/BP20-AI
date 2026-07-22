import torch
from collections import Counter
from kiwipiepy import Kiwi
from typing import List, Dict, Any

# Kiwi는 C++ 엔진이라 모듈 로드 시 1회 초기화
kiwi = Kiwi()
STOP_WORDS = {'생각', '사람', '정도', '때문', '보임', '가지', '얘기', '확인', '모습', '느낌', '방문', '하나', '부분'}

class DashboardService:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _predict_roberta(self, text: str) -> Dict[str, str]:
        """
        main.py에서 넘어온 RoBERTa 모델과 토크나이저로 추론을 수행하는 내부 메서드
        """
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        # ... (개발자님의 기존 predict 로직 연결) ...
        
        # 임시 예시 리턴 (실제 추론 결과 로직 적용)
        return {"food": "negative"} if "맛없" in text else {}

    def analyze_reviews_pipeline(self, reviews: List[str]) -> Dict[str, Any]:
        negative_aspect_counts = Counter()
        negative_review_texts = []
        classification_details = []

        # 1. RoBERTa 추론 및 부정 속성 집계
        for idx, text in enumerate(reviews):
            aspect_sentiments = self._predict_roberta(text)
            
            has_negative = False
            for aspect, sentiment in aspect_sentiments.items():
                if sentiment == "negative":
                    negative_aspect_counts[aspect] += 1
                    has_negative = True
            
            if has_negative:
                negative_review_texts.append(text)

            classification_details.append({
                "review_id": idx + 1,
                "text": text,
                "predictions": aspect_sentiments
            })

        # 2. Kiwi 형태소 분석으로 부정 TOP 5 키워드 추출
        keywords = []
        for text in negative_review_texts:
            tokens = kiwi.tokenize(text)
            for token in tokens:
                if token.tag in ['NNG', 'NNP', 'VA'] and len(token.form) > 1:
                    if token.form not in STOP_WORDS:
                        keywords.append(token.form)

        top5_keywords = [
            {"keyword": word, "count": count}
            for word, count in Counter(keywords).most_common(5)
        ]

        # 3. 최종 대시보드 응답 데이터 구성
        return {
            "summary": {
                "total_input_reviews": len(reviews),
                "total_negative_aspects_found": sum(negative_aspect_counts.values()),
                "negative_aspects_breakdown": dict(negative_aspect_counts),
            },
            "top5_negative_keywords": top5_keywords,
            "details": classification_details
        }