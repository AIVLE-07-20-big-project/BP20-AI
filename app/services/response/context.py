# Bandit 컨텍스트 벡터 구성
from __future__ import annotations

import numpy as np

CONTEXT_DIM = 6

# 모델의 context_schema_version과 대조해 정합성을 확인하는 값(설계 §3 정합성 계약)
CONTEXT_SCHEMA_VERSION = "response_context_v1"

_URGENCY_SCORE = {"낮음": 0.0, "중간": 0.5, "높음": 1.0}


def _num(value, default: float = 0.0) -> float:
    # None(직전 분기 데이터 없음 등)은 중립값으로 채우되, 0.0은 실제 값으로 보존한다
    return default if value is None else float(value)


# shape·NaN/Inf만 검사한다 — 값 범위 자체는 진단 로직의 책임이라 여기서 강제하지 않는다
def validate_context_vector(vector: np.ndarray) -> None:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (CONTEXT_DIM,):
        raise ValueError(f"context shape은 ({CONTEXT_DIM},)이어야 합니다: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("context에는 NaN 또는 Inf를 사용할 수 없습니다")


def build_context_vector(diagnosis: dict) -> np.ndarray:
    sev = diagnosis.get("1_심각도") or {}
    rx = diagnosis.get("5_처방") or {}
    features = [
        _num(sev.get("전분기_대비")),
        _num(sev.get("전년동기_대비")),
        _num(sev.get("최고점_대비")),
        _num(sev.get("하락_분기_비율")),
        _num(rx.get("하락_심각도점수")) / 6.0,
        _URGENCY_SCORE.get(rx.get("긴급도"), 0.0),
    ]
    vector = np.asarray(features, dtype=np.float32)
    validate_context_vector(vector)
    return vector
