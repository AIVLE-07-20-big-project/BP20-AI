# 대상 경로(AI 레포): app/sales_target/scoring.py
#
# 과제정의서 산식(19~23p) 그대로 구현:
#   최종 점수 = 상권 성장성×0.30 + 유동인구×0.25 + 리뷰 활성도×0.20 + 우수 가맹점 유사도×0.25
#
# 정규화 원칙: 각 서브 점수는 "이번 배치의 후보군 내 상대 순위(백분위)"를 기준으로 0~100으로 맞춘다.
# 절대값 기준으로 정규화하지 않는 이유는 설계 가이드에 적어둔 대로다 — 공공데이터 갱신 주기(분기)에
# 따라 절대 스케일이 흔들릴 수 있어서, 상대 순위가 배치마다 더 안정적인 기준이 된다.
#
# 리뷰 활성도 점수는 2단계(구글 플레이스 조회, google_places.py)를 거쳐야만 실값이 나온다.
# 상위 후보(top_n)를 추리는 1차 스코어링 시점엔 아직 이 데이터가 없으므로, 가짜 중간값을
# 채워 넣는 대신 review 축을 아예 빼고 나머지 세 축(growth/traffic/similarity)만으로
# preliminary_scores()를 계산한다(과거엔 review=50 placeholder를 넣었었는데, 상수라 순위엔
# 영향이 없어도 프론트에 "리뷰 활성도 50점"이라는 가짜 값이 그대로 노출되는 문제가 있었다).
# 2단계에서 review_activity_score()로 실값을 구한 후보만 final_scores()(4축)로 다시 계산한다.

from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {
    "growth": 0.30,
    "traffic": 0.25,
    "review": 0.20,
    "similarity": 0.25,
}


def growth_scores(growth_rates: pd.Series) -> pd.Series:
    """상권 성장성 점수. 과제정의서 5단계 버킷을 그대로 따른다:

    성장률 상위 10% -> 100점, 상위 30% -> 80점, 평균 이상 -> 60점, 평균 이하 -> 40점, 감소(<0) -> 20점.

    우선순위는 "상위 10/30% 버킷"이 "감소 여부"보다 먼저 적용된다 — 예를 들어 모든 후보의 매출이
    하락 중인 극단적 상황이라도, 그중 하락폭이 가장 작은(=상대적으로 성장률이 높은) 상위 10%는
    100점을 받는다. 상대 비교이지 절대적으로 "성장 중이어야만 고득점"은 아니라는 뜻이다.
    """
    if growth_rates.empty:
        return growth_rates.astype(float)

    p90 = growth_rates.quantile(0.90)
    p70 = growth_rates.quantile(0.70)
    mean = growth_rates.mean()

    def _bucket(rate: float) -> float:
        if pd.isna(rate):
            return 0.0
        if rate >= p90:
            return 100.0
        if rate >= p70:
            return 80.0
        if rate < 0:
            return 20.0
        if rate >= mean:
            return 60.0
        return 40.0

    return growth_rates.map(_bucket)


def foot_traffic_index(morning: pd.Series, lunch: pd.Series, afternoon: pd.Series) -> pd.Series:
    """과제정의서 원문 가중합: 오전×0.3 + 점심×0.3 + 오후×0.4. 아직 0~100으로 정규화되지 않은
    원시 지수다 — 최종 점수에 합산하려면 percentile_score()를 한 번 더 거쳐야 한다."""
    return morning * 0.3 + lunch * 0.3 + afternoon * 0.4


def percentile_score(values: pd.Series) -> pd.Series:
    """후보군 내 백분위 순위를 0~100 점수로 변환한다(값이 클수록 고득점)."""
    if values.empty:
        return values.astype(float)
    return values.rank(pct=True) * 100.0


def review_activity_score(
    review_count_pct: pd.Series,
    avg_rating_pct: pd.Series,
    recency_pct: pd.Series,
    growth_trend_pct: pd.Series,
) -> pd.Series:
    """리뷰 활성도 점수. 과제정의서 산식: 리뷰수×0.4 + 평균평점×0.3 + 최신리뷰×0.2 + 증가추세×0.1.
    네 입력 모두 이미 0~100으로 정규화된 백분위 점수여야 한다(percentile_score로 변환한 값을 넣는다).

    2단계(google_places.py로 리뷰 데이터를 수집한 후, pipeline.apply_review_scores())에서만
    호출된다 — 그 전까지(1차 스코어링)는 review 축 자체를 빼고 preliminary_scores()를 쓴다.
    """
    return (
        review_count_pct * 0.4
        + avg_rating_pct * 0.3
        + recency_pct * 0.2
        + growth_trend_pct * 0.1
    )


def cosine_similarity_scores(
    candidate_vectors: np.ndarray,
    hero_vectors: np.ndarray,
    top_k: int = 5,
) -> np.ndarray:
    """후보 벡터마다 우수 가맹점(Hero Store) 벡터들과의 코사인 유사도 상위 top_k 평균을 낸 뒤
    0~100 스케일로 변환한다. 코사인 유사도는 [-1, 1] 범위라 (sim + 1) / 2 * 100으로 맞춘다.

    벡터 구성은 설계 가이드 기준: [업종, 지역, 상권 성장률, 유동인구, 리뷰 수, 평점, 경쟁업소 수]
    등을 숫자로 인코딩한 것 — 인코딩 자체는 이 함수의 책임이 아니고 호출하는 쪽에서 준비해서 넘긴다.
    """
    n_candidates = candidate_vectors.shape[0] if candidate_vectors.ndim == 2 else len(candidate_vectors)
    if hero_vectors.size == 0 or candidate_vectors.size == 0:
        return np.zeros(n_candidates)

    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return matrix / norms

    cand_normalized = _normalize(candidate_vectors)
    hero_normalized = _normalize(hero_vectors)

    similarities = cand_normalized @ hero_normalized.T  # (n_candidates, n_heroes)
    k = min(top_k, similarities.shape[1])
    top_k_similarities = np.sort(similarities, axis=1)[:, -k:]
    avg_top_k = top_k_similarities.mean(axis=1)

    return (avg_top_k + 1.0) / 2.0 * 100.0


def final_scores(
    growth: pd.Series,
    traffic: pd.Series,
    review: pd.Series,
    similarity: pd.Series,
) -> pd.Series:
    """네 서브 점수(모두 0~100 스케일)를 과제정의서 가중치로 합산한다. review_score가 실값으로
    채워진 후보(2단계 완료)에만 쓴다."""
    return (
        growth * WEIGHTS["growth"]
        + traffic * WEIGHTS["traffic"]
        + review * WEIGHTS["review"]
        + similarity * WEIGHTS["similarity"]
    )


_PRELIMINARY_WEIGHT_SUM = WEIGHTS["growth"] + WEIGHTS["traffic"] + WEIGHTS["similarity"]  # 0.80


def preliminary_scores(
    growth: pd.Series,
    traffic: pd.Series,
    similarity: pd.Series,
) -> pd.Series:
    """1차 스코어링(리뷰 데이터 수집 전) 가중합. review 축을 빼고, 나머지 세 축의 원래 비율
    (growth 0.30 : traffic 0.25 : similarity 0.25)은 유지한 채 가중치 합이 1이 되도록
    재정규화한다(0.30/0.80, 0.25/0.80, 0.25/0.80). top_n 선정과, 2단계에서도 리뷰 매칭에
    실패한 후보의 final_score 계산에 그대로 쓰인다."""
    return (
        growth * (WEIGHTS["growth"] / _PRELIMINARY_WEIGHT_SUM)
        + traffic * (WEIGHTS["traffic"] / _PRELIMINARY_WEIGHT_SUM)
        + similarity * (WEIGHTS["similarity"] / _PRELIMINARY_WEIGHT_SUM)
    )
