# Bandit 컨텍스트 벡터 구성
from __future__ import annotations

import numpy as np

CONTEXT_DIM = 6

_URGENCY_SCORE = {"낮음": 0.0, "중간": 0.5, "높음": 1.0}


def _num(value, default: float = 0.0) -> float:
    # None(직전 분기 데이터 없음 등)은 중립값으로 채우되, 0.0은 실제 값으로 보존한다
    return default if value is None else float(value)


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
    return np.asarray(features, dtype=np.float32)
