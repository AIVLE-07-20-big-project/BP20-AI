import torch
from typing import List, Any
from schemas.review import AspectSentiment
from core.config import settings

class ABSAService:
    def __init__(self, model: Any, tokenizer: Any, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, review_text: str) -> List[AspectSentiment]:
        inputs = self.tokenizer(
            [review_text] * len(settings.ASPECTS),
            settings.ASPECTS,
            truncation=True,
            padding="max_length",
            max_length=settings.MAX_LENGTH,
            return_token_type_ids=False,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_indices = torch.argmax(logits, dim=-1).cpu().tolist()
            probs_list = probs.cpu().tolist()

        results = []
        for aspect, pred_idx, prob_dist in zip(settings.ASPECTS, pred_indices, probs_list):
            sentiment_label = settings.LABEL_MAP.get(pred_idx, "none")

            if sentiment_label != "none":
                confidence = round(prob_dist[pred_idx] * 100, 2)
                results.append(
                    AspectSentiment(
                        aspect=aspect,
                        sentiment=sentiment_label,
                        confidence=confidence
                    )
                )

        return results