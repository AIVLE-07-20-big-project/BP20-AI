# 등급별 Bandit 모델 저장소(model/bandit/{등급}/active.pt)
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import BANDIT_MODEL_DIR, CAMPAIGN_LOGS_V2
from app.services.response import action_rules
from app.services.response.activation import ActivationCriteria, check_activation
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


def candidate_path(등급: str, version: str) -> Path:
    return BANDIT_MODEL_DIR / 등급 / f"candidate-{version}.pt"


def previous_active_path(등급: str) -> Path:
    return BANDIT_MODEL_DIR / 등급 / "previous-active.pt"


def save_candidate(등급: str, bandit: NeuralContextualBandit) -> Path:
    path = candidate_path(등급, bandit.policy_version)
    bandit.save(path)
    return path


# 가장 최근에 저장된 candidate의 버전(shadow 비교·승격 대상 조회용)
def latest_candidate_version(등급: str) -> str | None:
    grade_dir = BANDIT_MODEL_DIR / 등급
    if not grade_dir.exists():
        return None
    candidates = sorted(grade_dir.glob("candidate-*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0].stem[len("candidate-"):]


# check_activation() 통과 + 사람의 명시적 승인(approved_by)이 있어야만 승격한다(기존 active는 rollback용 백업)
def promote_candidate(
    등급: str, version: str, *, approved_by: str, criteria: ActivationCriteria = ActivationCriteria(),
) -> Path:
    if not approved_by:
        raise ValueError("approved_by가 필요합니다 — 승인 없이 승격할 수 없습니다")
    if version in LEGACY_SHADOW_ONLY_POLICY_VERSIONS:
        raise ValueError(f"'{version}'은 shadow 전용 모델이라 승격할 수 없습니다")

    src = candidate_path(등급, version)
    if not src.exists():
        raise FileNotFoundError(f"후보 모델을 찾을 수 없음: {src}")

    arms = action_rules.ACTION_RULES.get(등급, [])
    activation = check_activation(
        CAMPAIGN_LOGS_V2, problem_type=등급, arms=arms,
        reward_definition_version=REWARD_DEFINITION_VERSION,
        context_schema_version=CONTEXT_SCHEMA_VERSION,
        criteria=criteria,
    )
    if not activation.activated:
        raise ValueError(f"활성화 기준 미달로 승격 불가: {list(activation.reasons)}")

    bandit = NeuralContextualBandit.load_any(src)
    bandit.training_data_cutoff = datetime.now(timezone.utc).isoformat()

    active_path = model_path(등급)
    if active_path.exists():
        shutil.copy2(active_path, previous_active_path(등급))
    bandit.save(active_path)
    return active_path


# 직전 승격 이전의 active 모델로 즉시 되돌린다(계획 §6 "이전 정책으로 즉시 rollback")
def rollback(등급: str) -> Path:
    backup = previous_active_path(등급)
    if not backup.exists():
        raise FileNotFoundError(f"롤백할 이전 모델이 없음: {backup}")
    active_path = model_path(등급)
    shutil.copy2(backup, active_path)
    return active_path
