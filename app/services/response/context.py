"""Bandit 컨텍스트 벡터 구성.

계획 §5.1은 "상권 특성·연령대 분포·채널선호·문제유형 조합"을 컨텍스트로 쓰자고
제안했지만, 지금은 그 중 Diagnoser가 실제로 내는 심각도 신호(`1_심각도`, `5_처방`)만으로
1차 버전을 만든다. 연령대 분포·채널선호는 §11(고객군 세그먼트) 채택 이후 실제 데이터가
쓰일 자리라, 지금 임의로 채우지 않는다 — 정직한 낮은 신뢰도 원칙(계획 §3).
"""
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
