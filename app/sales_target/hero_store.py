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

# 5단계(업종 명칭 매핑) — 자사 매장의 category(자유 입력 텍스트, 예: "카페·베이커리")와 공공데이터
# 상가업소 중분류명(indsMclsNm, 예: "커피점/카페")은 어차피 정확히 같은 문자열일 수가 없다.
# 완전일치 대신 키워드 포함 여부로 두 표기를 같은 대분류로 묶는다 — 정확한 공공데이터 표기를
# 몰라도(키만 있으면 실제 API로 확인 가능하지만 이 환경엔 키가 없어 검증 못 함) 어느 쪽이든
# 흔히 쓰는 단어가 겹치면 매칭되도록 관대하게 잡았다. 못 찾으면 원문 그대로 둔다(완전히 다른
# 업종끼리 같은 차원으로 묶이는 것보다, 매칭 안 되는 게 안전하다는 기존 설계 원칙과 동일).
_INDUSTRY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("카페/베이커리", ("카페", "커피", "베이커리", "제과", "디저트", "빵")),
    ("치킨", ("치킨", "통닭")),
    ("한식", ("한식",)),
    ("중식", ("중식", "중국")),
    ("일식", ("일식", "돈까스", "초밥", "스시", "라멘")),
    ("양식/피자", ("양식", "피자", "파스타", "스테이크")),
    ("분식", ("분식",)),
    ("주점", ("주점", "호프", "포차", "술집")),
    ("패스트푸드", ("패스트푸드", "버거", "샌드위치")),
    ("미용", ("미용", "네일", "헤어", "피부", "뷰티")),
    ("편의점/마트", ("편의점", "마트", "슈퍼")),
]


def normalize_industry(text: object) -> str:
    """업종 표기를 위 키워드 그룹 기준으로 정규화한다. 매칭 실패 시 원문(strip)을 그대로 쓴다.

    text가 NaN/None/문자열이 아닌 값이어도 안전하게 "__unknown__"을 반환한다(candidates/
    hero_stores 어느 쪽이든 industry 컬럼 자체가 없으면 pipeline.py가 None으로 채워서 넘긴다).
    """
    if not isinstance(text, str) or not text.strip():
        return "__unknown__"
    for canonical, keywords in _INDUSTRY_KEYWORDS:
        if any(k in text for k in keywords):
            return canonical
    return text.strip()


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
    store_count_col: str,
) -> pd.DataFrame:
    """구별 평균 상권 성장률/유동인구 지수/경쟁업소 수를 계산한다.

    자사 매장은 좌표가 없어 상권코드(trdar_cd)로 직접 매핑할 수 없다. 대신 이미 상권코드가 매핑된
    후보 업장 데이터(=candidate_registry 전체, 서울 상가 대부분을 커버)를 구 단위로 평균 내
    근사치로 쓴다. 반환 컬럼: district, district_growth_rate, district_traffic_index,
    district_store_count.
    """
    return (
        candidates_with_district_metrics
        .groupby(sigungu_col)[[growth_col, traffic_col, store_count_col]]
        .mean()
        .rename(columns={
            growth_col: "district_growth_rate",
            traffic_col: "district_traffic_index",
            store_count_col: "district_store_count",
        })
        .reset_index()
        .rename(columns={sigungu_col: "district"})
    )


def build_similarity_vectors(
    candidates: pd.DataFrame,
    hero_stores: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """코사인 유사도용 특징 벡터를 만든다.

    candidates/hero_stores 공통 컬럼: industry, district, district_growth_rate,
    district_traffic_index, review_count, review_avg_rating, competitor_count.

    설계가이드 5.4의 벡터 스펙([업종, 지역, 상권 성장률, 유동인구, 리뷰 수, 평점, 경쟁업소 수])을
    전부 구현한다. "경쟁업소 수"는 처음 증분(2026-07-29)엔 StoreStatsCollector(점포/개폐업률,
    VwsmTrdarStorQq) 연동이 파이프라인에 없어서 뺐었는데, 이번에 채워 넣었다 — 이미 다른 기능
    (scripts/modeling/sales_analysis.py의 STOR_CO 사용)에서 검증된 콜렉터라 그대로 재사용했다.

    업종/지역은 원핫 인코딩한다. 업종은 normalize_industry()로 먼저 정규화해서, 자사 매장의
    category(자유 입력 텍스트)와 공공데이터 업종명(indsMclsNm)의 표기가 달라도 흔한 키워드가
    겹치면 같은 차원으로 묶이게 한다(5단계, 이전엔 이 정규화가 없어서 표기가 다르면 유사도에
    전혀 안 잡혔다).
    나머지 수치형 항목은 candidates+hero_stores를 합친 분포 기준 백분위(0~1)로 정규화해서
    원핫 차원(0/1)과 같은 스케일로 맞춘다. competitor_count는 "적을수록 좋다/많을수록 좋다"를
    판단하는 점수가 아니라 유사도 축이라, growth/traffic과 동일하게 방향 반전 없이 백분위만
    적용한다(경쟁업소 수가 비슷한 정도를 비교하는 것이지, 적은 쪽에 가산점을 주는 게 아니다).
    """
    n_cand = len(candidates)

    combined = pd.concat([candidates, hero_stores], ignore_index=True)

    industry_dummies = pd.get_dummies(combined["industry"].map(normalize_industry))
    district_dummies = pd.get_dummies(combined["district"].fillna("__unknown__"))

    numeric = pd.DataFrame({
        "growth": percentile_score(combined["district_growth_rate"].fillna(0)) / 100.0,
        "traffic": percentile_score(combined["district_traffic_index"].fillna(0)) / 100.0,
        "review_count": percentile_score(combined["review_count"].fillna(0)) / 100.0,
        "review_rating": percentile_score(combined["review_avg_rating"].fillna(0)) / 100.0,
        "competitor_count": percentile_score(combined["competitor_count"].fillna(0)) / 100.0,
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
