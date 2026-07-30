# 대상 경로(AI 레포): app/sales_target/hero_store.py
#
# 우수 가맹점(Hero Store) 선정 + 후보 업장과의 유사도 벡터 계산.
# 설계가이드 결정 4(1단계 근사 기준)와 5.4/5.5 스펙을 구현한다.

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.sales_target.scoring import cosine_similarity_scores, percentile_score

# 1단계 근사 기준(설계가이드 결정 4): "매출 성장률 상위 20% + 리뷰 평점 안정적(표준편차 낮고
# 평균 일정 기준 이상)". 표준편차/평균 임계값의 구체적인 숫자는 원문에 없어서 5점 만점 평점 기준으로
# 합리적이라고 판단한 값을 넣었다 — 실제 리뷰 데이터 분포를 보고 조정이 필요할 수 있다.
GROWTH_PERCENTILE_THRESHOLD = 0.80  # 상위 20%
RATING_STD_THRESHOLD = 1.0
RATING_MEAN_THRESHOLD = 4.0

# 주소 문자열에서 "OO구"를 추출하는 패턴. 자사 매장은 좌표가 없어 상권코드 직접 매핑이 안 되므로
# 구 단위 근사가 필요하다(아래 sigungu_average_district_metrics 참고).
_SIGUNGU_PATTERN = re.compile(r"[가-힣]+구")


def select_hero_stores(stores_with_metrics: pd.DataFrame) -> pd.DataFrame:
    """자사 가맹점 중 우수 가맹점을 선정한다.

    필요 컬럼: salesGrowthRate, reviewAvgRating, reviewRatingStd.
    셋 중 하나라도 null인 매장은 후보에서 제외한다 — 데이터 부족을 "우수함"으로 착각하지 않기 위함.
    """
    required = {"salesGrowthRate", "reviewAvgRating", "reviewRatingStd"}
    missing = required - set(stores_with_metrics.columns)
    if missing:
        raise ValueError(f"select_hero_stores: 필요한 컬럼이 없습니다: {missing}")

    df = stores_with_metrics
    if df.empty:
        return df.copy()

    valid_growth = df["salesGrowthRate"].dropna()
    if valid_growth.empty:
        return df.iloc[0:0].copy()
    growth_cutoff = valid_growth.quantile(GROWTH_PERCENTILE_THRESHOLD)

    mask = (
        df["salesGrowthRate"].notna()
        & (df["salesGrowthRate"] >= growth_cutoff)
        & df["reviewRatingStd"].notna()
        & (df["reviewRatingStd"] <= RATING_STD_THRESHOLD)
        & df["reviewAvgRating"].notna()
        & (df["reviewAvgRating"] >= RATING_MEAN_THRESHOLD)
    )
    return df[mask].copy()


def extract_sigungu(address: str | float | None) -> str | None:
    """주소 문자열에서 '~구'를 추출한다. 못 찾으면 None."""
    if not isinstance(address, str):
        return None
    match = _SIGUNGU_PATTERN.search(address)
    return match.group() if match else None


def sigungu_average_district_metrics(
    candidates_with_district_metrics: pd.DataFrame,
    sigungu_col: str,
    growth_col: str,
    traffic_col: str,
) -> pd.DataFrame:
    """구별 평균 상권 성장률/유동인구 지수를 계산한다.

    자사 매장은 좌표가 없어 상권코드(trdar_cd)로 직접 매핑할 수 없다. 대신 이미 상권코드가 매핑된
    후보 업장 데이터(=candidate_registry 전체, 서울 상가 대부분을 커버)를 구 단위로 평균 내
    근사치로 쓴다. 반환 컬럼: district, district_growth_rate, district_traffic_index.
    """
    return (
        candidates_with_district_metrics
        .groupby(sigungu_col)[[growth_col, traffic_col]]
        .mean()
        .rename(columns={growth_col: "district_growth_rate", traffic_col: "district_traffic_index"})
        .reset_index()
        .rename(columns={sigungu_col: "district"})
    )


def build_similarity_vectors(
    candidates: pd.DataFrame,
    hero_stores: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """코사인 유사도용 특징 벡터를 만든다.

    candidates/hero_stores 공통 컬럼: industry, district, district_growth_rate,
    district_traffic_index, review_count, review_avg_rating.

    설계가이드 5.4의 벡터 스펙([업종, 지역, 상권 성장률, 유동인구, 리뷰 수, 평점, 경쟁업소 수]) 중
    "경쟁업소 수"는 이번 증분에서 뺐다 — StoreStatsCollector(점포/개폐업률) 연동이 아직
    파이프라인에 없어서다. 나머지 6개 항목으로 벡터를 구성한다.

    업종/지역은 원핫 인코딩한다. 자사 매장의 category(자유 입력 텍스트)와 공공데이터의 업종
    대분류명(indsLclsNm)이 표기가 다르면 같은 업종이어도 다른 차원으로 잡혀 유사도에 안 잡힐 수
    있다 — 업종 명칭 매핑 테이블 없이는 못 고치는 한계이고, 지금은 이 상태로 둔다.
    나머지 수치형 항목은 candidates+hero_stores를 합친 분포 기준 백분위(0~1)로 정규화해서
    원핫 차원(0/1)과 같은 스케일로 맞춘다.
    """
    n_cand = len(candidates)

    combined = pd.concat([candidates, hero_stores], ignore_index=True)

    industry_dummies = pd.get_dummies(combined["industry"].fillna("__unknown__"))
    district_dummies = pd.get_dummies(combined["district"].fillna("__unknown__"))

    numeric = pd.DataFrame({
        "growth": percentile_score(combined["district_growth_rate"].fillna(0)) / 100.0,
        "traffic": percentile_score(combined["district_traffic_index"].fillna(0)) / 100.0,
        "review_count": percentile_score(combined["review_count"].fillna(0)) / 100.0,
        "review_rating": percentile_score(combined["review_avg_rating"].fillna(0)) / 100.0,
    })

    full_matrix = pd.concat(
        [industry_dummies.reset_index(drop=True), district_dummies.reset_index(drop=True), numeric],
        axis=1,
    ).to_numpy(dtype=float)

    return full_matrix[:n_cand], full_matrix[n_cand:]


def hero_similarity_scores(
    candidates: pd.DataFrame,
    hero_stores: pd.DataFrame,
    top_k: int = 5,
) -> np.ndarray:
    """candidates 각 행에 대해 hero_stores와의 코사인 유사도(top-k 평균, 0~100)를 계산한다.

    hero_stores가 비어있으면 전부 0을 반환한다 — 우수 가맹점이 아직 없다는 뜻이므로 유사도
    신호를 줄 수 없다(placeholder 50.0과 달리 "판단 불가"를 0으로 명시하는 쪽을 택했다. 호출부인
    pipeline.py가 hero_stores가 없을 때는 아예 이 함수를 안 부르고 50.0 placeholder를 쓰도록
    분기하므로, 이 0 반환은 "hero_stores를 넘겼는데 우연히 비어있던" 경우에만 실제로 쓰인다).
    """
    if hero_stores.empty or candidates.empty:
        return np.zeros(len(candidates))
    cand_vectors, hero_vectors = build_similarity_vectors(candidates, hero_stores)
    return cosine_similarity_scores(cand_vectors, hero_vectors, top_k=top_k)
