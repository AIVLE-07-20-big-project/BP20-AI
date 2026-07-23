# 등급별 Bandit 모델 저장소(model/bandit/{등급}/active.pt)
from __future__ import annotations

from pathlib import Path

from app.core.config import BANDIT_MODEL_DIR
from scripts.response_strategy.bandit import BanditLoadMismatch, NeuralContextualBandit


def model_path(등급: str) -> Path:
    return BANDIT_MODEL_DIR / 등급 / "active.pt"


# 저장된 모델이 있으면 불러오고, 없거나 arm 집합이 바뀌었으면 콜드스타트한다
def load_or_coldstart(등급: str, context_dim: int, arms: list[str]) -> tuple[NeuralContextualBandit, bool]:





    path = model_path(등급)
    if path.exists():
        try:
            return NeuralContextualBandit.load(path, context_dim=context_dim, arms=arms), True
        except BanditLoadMismatch:
            pass
    return NeuralContextualBandit(context_dim=context_dim, arms=arms), False


def save(등급: str, bandit: NeuralContextualBandit) -> None:
    bandit.save(model_path(등급))
