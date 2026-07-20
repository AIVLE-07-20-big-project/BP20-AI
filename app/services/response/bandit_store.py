"""등급별 Bandit 모델 저장소(model/bandit/{등급}/active.pt).

graph.py(_recommend, 콜드스타트/로드), campaign_logs.py(온라인 update 후 저장) 양쪽이
공유한다 — 경로 규칙이 두 곳에서 어긋나면 저장한 모델을 못 찾는 조용한 버그가 되므로
한 곳에만 둔다.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import BANDIT_MODEL_DIR
from scripts.response_strategy.bandit import BanditLoadMismatch, NeuralContextualBandit


def model_path(등급: str) -> Path:
    return BANDIT_MODEL_DIR / 등급 / "active.pt"


def load_or_coldstart(등급: str, context_dim: int, arms: list[str]) -> tuple[NeuralContextualBandit, bool]:
    """저장된 모델이 있으면 불러오고, 없거나 arm 집합이 바뀌었으면 콜드스타트한다.

    로드 실패는 서비스 중단 사유가 아니다 — 콜드스타트로 조용히 폴백한다(계획 §3
    "정직한 낮은 신뢰도 > 거짓 확신").
    """
    path = model_path(등급)
    if path.exists():
        try:
            return NeuralContextualBandit.load(path, context_dim=context_dim, arms=arms), True
        except BanditLoadMismatch:
            pass
    return NeuralContextualBandit(context_dim=context_dim, arms=arms), False


def save(등급: str, bandit: NeuralContextualBandit) -> None:
    bandit.save(model_path(등급))
