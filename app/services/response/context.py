# Bandit 컨텍스트 벡터 구성
from __future__ import annotations

import numpy as np

CONTEXT_DIM = 6

_URGENCY_SCORE = {"낮음": 0.0, "중간": 0.5, "높음": 1.0}


def build_context_vector(diagnosis: dict) -> np.ndarray:
    sev = diagnosis.get("1_심각도") or {}
    rx = diagnosis.get("5_처방") or {}
    features = [
        sev.get("전분기_대비") or 0.0,
        sev.get("전년동기_대비") or 0.0,
        sev.get("최고점_대비") or 0.0,
        sev.get("하락_분기_비율") or 0.0,
        (rx.get("하락_심각도점수") or 0) / 6.0,
        _URGENCY_SCORE.get(rx.get("긴급도"), 0.0),
    ]
    return np.asarray(features, dtype=np.float32)
