# 등급별 Bandit 모델 저장소(model/bandit/{등급}/active.pt)
from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.config import BANDIT_MODEL_DIR
from app.services.response.context import CONTEXT_SCHEMA_VERSION
from app.services.response.reward import REWARD_DEFINITION_VERSION
from scripts.response_strategy.bandit import BanditLoadMismatch, NeuralContextualBandit

# 신규 콜드스타트 모델용 model_version — bandit.py 생성자의 "legacy" 기본값은 구버전 파일용
CURRENT_MODEL_VERSION = "coldstart-v1"

# shadow 전용 모델 — 저장돼 있어도 운영 추천 근거로 쓰지 않고 coldstart로 취급
LEGACY_SHADOW_ONLY_POLICY_VERSIONS = frozenset({"backfill-v1"})


def model_path(등급: str) -> Path:
    return BANDIT_MODEL_DIR / 등급 / "active.pt"


# 로그 v2의 model_status 필드용(candidate/legacy 상태는 5단계에서 추가)
def model_status(loaded: bool) -> str:
    return "active" if loaded else "coldstart"


# 로그 v2의 model_sha256 — 어떤 모델 파일이 이 선택을 만들었는지 재현 가능하게 식별한다
def model_sha256(등급: str) -> str | None:
    path = model_path(등급)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 저장된 모델이 없거나, arm 집합이 바뀌었거나, shadow 전용 모델이면 콜드스타트한다
def load_or_coldstart(등급: str, context_dim: int, arms: list[str]) -> tuple[NeuralContextualBandit, bool]:
    path = model_path(등급)
    if path.exists():
        try:
            bandit = NeuralContextualBandit.load(path, context_dim=context_dim, arms=arms)
            if bandit.policy_version not in LEGACY_SHADOW_ONLY_POLICY_VERSIONS:
                return bandit, True
        except BanditLoadMismatch:
            pass
    return NeuralContextualBandit(
        context_dim=context_dim, arms=arms,
        model_version=CURRENT_MODEL_VERSION,
        context_schema_version=CONTEXT_SCHEMA_VERSION,
        reward_definition_version=REWARD_DEFINITION_VERSION,
    ), False


def save(등급: str, bandit: NeuralContextualBandit) -> None:
    bandit.save(model_path(등급))
