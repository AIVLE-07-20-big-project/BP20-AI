# 대상 경로(AI 레포): app/sales_target/pipeline.py
#
# 지금까지 만든 조각(콜렉터/매칭/상권매핑/지표집계/스코어링)을 실제로 이어붙이는 오케스트레이션.
#
# 구조를 두 층으로 나눴다:
#   1) generate_sales_targets() — 순수 함수. 이미 받아온 DataFrame들만 입력으로 받고, 네트워크 호출도
#      없고 실제 공공데이터 컬럼명 추측값도 안 들어있다. 그래서 100% 유닛테스트 가능하고, 실제
#      컬럼명이 뭐든 상관없이(호출하는 쪽에서 이미 정리해서 넘기니까) 로직만 검증할 수 있다.
#   2) collect_and_generate() — 실제로 콜렉터/BE 클라이언트를 호출해서 데이터를 모으고 1)에 넘기는
#      비동기 함수. 여기에만 공공데이터 컬럼명이 들어있다(아래 _FIELD_NAMES 참고).

from __future__ import annotations

import math

import pandas as pd

from app.sales_target.backend_client import BackendStoreRegistryClient
from app.sales_target.district_metrics import (
    aggregate_district_quarterly,
    latest_quarter_values,
    latest_vs_previous_growth,
)
from app.sales_target.geo_matching import add_trdar_wgs84_columns, assign_trdar_cd
from app.sales_target.hero_store import (
    hero_similarity_scores,
    select_hero_stores,
    sigungu_average_district_metrics,
    extract_sigungu,
)
from app.sales_target.matching import exclude_existing_merchants
from app.sales_target.scoring import (
    final_scores,
    foot_traffic_index,
    growth_scores,
    percentile_score,
    review_activity_placeholder,
)

# 우수 가맹점 유사도(similarity_score) 계산에는 hero_stores(자사 우수 가맹점 목록, 리뷰
# 평점/표준편차 포함)가 필요하다. 아직 BE가 리뷰 통계를 내려주지 않는 상태에서 호출되면(또는
# 우수 가맹점이 0곳이면) 이 중간값으로 대체한다. review_activity_placeholder와 같은 이유:
# "판단 불가"를 억지로 0점 처리하면 오히려 순위를 왜곡하므로 중립값을 쓴다.
_SIMILARITY_PLACEHOLDER = 50.0


def generate_sales_targets(
    candidate_registry: pd.DataFrame,
    our_stores: pd.DataFrame,
    trdar_boundary: pd.DataFrame,
    district_sales_growth: pd.DataFrame,
    district_foot_traffic_index: pd.DataFrame,
    hero_stores: pd.DataFrame | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """신규 영업 타겟 후보 리스트를 만든다. 네트워크 호출 없는 순수 함수.

    인자:
      candidate_registry: store_registry_collector.py 결과. lon/lat(WGS84) 컬럼 필요.
      our_stores: backend_client.py 결과. address, salesGrowthRate 컬럼 필요(매칭/우수가맹점용).
      trdar_boundary: TrdarBoundaryCollector 결과(변환 전, XCNTS_VALUE/YDNTS_VALUE/RELM_AR 포함).
      district_sales_growth: district_metrics로 미리 계산한 DataFrame[trdar_cd, growth_rate].
      district_foot_traffic_index: district_metrics로 미리 계산한 DataFrame[trdar_cd, traffic_index]
        (foot_traffic_index()로 이미 가중합까지 끝낸, 0~100 정규화 전 원시 지수).
      hero_stores: hero_store.select_hero_stores() 결과(category, address, reviewCount,
        reviewAvgRating 컬럼 필요). None이거나 빈 DataFrame이면 similarity_score는
        _SIMILARITY_PLACEHOLDER(50.0)로 채운다 — BE가 아직 리뷰 통계를 안 내려주는 동안의
        임시 상태이며, 실제 컬럼이 갖춰지면 자동으로 실값 계산으로 전환된다.
      top_n: 상위 N개만 반환하고 싶을 때 지정. None이면 전체 반환.

    반환: final_score 내림차순으로 정렬된 DataFrame. growth_score/traffic_score/review_score/
      similarity_score/final_score 컬럼이 추가된다.
    """
    if candidate_registry.empty:
        return candidate_registry.copy()

    filtered = exclude_existing_merchants(candidate_registry, our_stores)
    if filtered.empty:
        return filtered

    trdar_wgs84 = add_trdar_wgs84_columns(trdar_boundary)
    mapped = assign_trdar_cd(filtered, trdar_wgs84)

    merged = mapped.merge(
        district_sales_growth.rename(columns={"growth_rate": "_district_growth_rate"}),
        on="trdar_cd",
        how="left",
    ).merge(
        district_foot_traffic_index.rename(columns={"traffic_index": "_district_traffic_index"}),
        on="trdar_cd",
        how="left",
    )

    merged["growth_score"] = growth_scores(merged["_district_growth_rate"])
    merged["traffic_score"] = percentile_score(merged["_district_traffic_index"])
    merged["review_score"] = review_activity_placeholder(len(merged))

    if hero_stores is not None and not hero_stores.empty:
        candidate_features = pd.DataFrame({
            "industry": merged["indsLclsNm"] if "indsLclsNm" in merged.columns else None,
            "district": merged["signguNm"] if "signguNm" in merged.columns else None,
            "district_growth_rate": merged["_district_growth_rate"],
            "district_traffic_index": merged["_district_traffic_index"],
            # 비가맹 업장 리뷰 수집기가 아직 없어 후보 쪽 리뷰 지표는 항상 0이다. hero_stores
            # 쪽에만 실제 리뷰 통계가 들어가므로, 유사도는 현재 [업종/지역/상권지표] 4개 축으로만
            # 사실상 갈리고 리뷰 2개 축은 "후보가 hero와 다르다"는 방향으로만 작용한다.
            "review_count": 0.0,
            "review_avg_rating": 0.0,
        })

        district_avg = sigungu_average_district_metrics(
            merged,
            sigungu_col="signguNm" if "signguNm" in merged.columns else merged.columns[0],
            growth_col="_district_growth_rate",
            traffic_col="_district_traffic_index",
        )
        hero_features = pd.DataFrame({
            "industry": hero_stores["category"] if "category" in hero_stores.columns else None,
            "district": hero_stores["address"].map(extract_sigungu) if "address" in hero_stores.columns else None,
        })
        hero_features = hero_features.merge(district_avg, on="district", how="left")
        hero_features["review_count"] = hero_stores["reviewCount"] if "reviewCount" in hero_stores.columns else 0.0
        hero_features["review_avg_rating"] = (
            hero_stores["reviewAvgRating"] if "reviewAvgRating" in hero_stores.columns else 0.0
        )

        merged["similarity_score"] = hero_similarity_scores(candidate_features, hero_features)
    else:
        merged["similarity_score"] = _SIMILARITY_PLACEHOLDER

    merged["final_score"] = final_scores(
        merged["growth_score"],
        merged["traffic_score"],
        merged["review_score"],
        merged["similarity_score"],
    )

    ranked = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked


# ── 서울시 상권분석서비스 공공데이터 컬럼명 ─────────────────────────────
# 검증 완료(2026-07-29, 실제 API 라이브 호출로 확인):
#   추정매출(VwsmTrdarSelngQq): THSMON_SELNG_AMT("당월 매출금액") 확인됨
#   길단위인구(VwsmTrdarFlpopQq): TMZON_06_11_FLPOP_CO(오전대), TMZON_11_14_FLPOP_CO(점심대),
#     TMZON_14_17_FLPOP_CO(오후대) 확인됨 — 과제정의서가 말하는 "오전/점심/오후" 유동인구와
#     정확히 대응된다.
_FIELD_NAMES = {
    "sales_trdar_col": "TRDAR_CD",
    "sales_quarter_col": "STDR_YYQU_CD",
    "sales_amount_col": "THSMON_SELNG_AMT",
    "traffic_trdar_col": "TRDAR_CD",
    "traffic_quarter_col": "STDR_YYQU_CD",
    "traffic_morning_col": "TMZON_06_11_FLPOP_CO",
    "traffic_lunch_col": "TMZON_11_14_FLPOP_CO",
    "traffic_afternoon_col": "TMZON_14_17_FLPOP_CO",
}


async def collect_and_generate(
    *,
    store_registry_api_key: str,
    seoul_api_key: str,
    backend_base_url: str,
    backend_internal_api_key: str,
    quarter_codes: list[str],
    top_n: int | None = None,
) -> pd.DataFrame:
    """실제 콜렉터/BE 클라이언트를 호출해서 데이터를 모으고 generate_sales_targets()에 넘긴다."""
    # 순환 import를 피하려고 여기서 지역 임포트한다(scripts.collection이 app을 참조하지 않지만,
    # 반대 방향 임포트를 파일 상단에 두면 이 모듈을 import하는 다른 코드가 항상 scripts 의존성까지
    # 끌고 오게 된다 — 이 함수를 실제로 쓸 때만 필요하도록 지연 임포트했다).
    from scripts.collection.collectors import (
        FootTrafficCollector,
        SalesEstimateCollector,
        TrdarBoundaryCollector,
    )
    from scripts.collection.store_registry_collector import StoreRegistryCollector

    f = _FIELD_NAMES

    candidate_registry = await StoreRegistryCollector(store_registry_api_key).fetch_sigungus()

    backend_client = BackendStoreRegistryClient(backend_base_url, backend_internal_api_key)
    our_stores = await backend_client.fetch_our_stores()

    # 우수 가맹점 선정에는 reviewAvgRating/reviewRatingStd가 필요한데, BE가 아직 이 필드를
    # 안 내려주면(salesGrowthRate까지만 있는 이전 버전) select_hero_stores()가 ValueError를
    # 던진다. 여기서는 그 경우 조용히 hero_stores=None으로 넘겨서 파이프라인이 계속 돌아가게
    # 하고, similarity_score는 placeholder로 채워진다 — BE가 필드를 추가하면 다음 실행부터
    # 자동으로 실값 계산으로 전환된다(이 함수는 수정할 필요 없음).
    hero_stores = None
    required_hero_cols = {"salesGrowthRate", "reviewAvgRating", "reviewRatingStd"}
    if required_hero_cols.issubset(our_stores.columns):
        selected = select_hero_stores(our_stores)
        hero_stores = selected if not selected.empty else None

    trdar_boundary = await TrdarBoundaryCollector(seoul_api_key).fetch_all()

    sales_raw = await SalesEstimateCollector(seoul_api_key).fetch_quarters(quarter_codes)
    sales_district_quarterly = aggregate_district_quarterly(
        sales_raw, f["sales_trdar_col"], f["sales_quarter_col"], f["sales_amount_col"]
    )
    district_sales_growth = latest_vs_previous_growth(
        sales_district_quarterly, f["sales_trdar_col"], f["sales_quarter_col"], f["sales_amount_col"]
    ).rename(columns={f["sales_trdar_col"]: "trdar_cd"})

    traffic_raw = await FootTrafficCollector(seoul_api_key).fetch_quarters(quarter_codes)
    traffic_raw = traffic_raw.copy()
    traffic_raw["_traffic_index"] = foot_traffic_index(
        traffic_raw[f["traffic_morning_col"]].astype(float),
        traffic_raw[f["traffic_lunch_col"]].astype(float),
        traffic_raw[f["traffic_afternoon_col"]].astype(float),
    )
    traffic_district_quarterly = aggregate_district_quarterly(
        traffic_raw, f["traffic_trdar_col"], f["traffic_quarter_col"], "_traffic_index"
    )
    district_foot_traffic_index = latest_quarter_values(
        traffic_district_quarterly, f["traffic_trdar_col"], f["traffic_quarter_col"], "_traffic_index"
    ).rename(columns={f["traffic_trdar_col"]: "trdar_cd", "_traffic_index": "traffic_index"})

    return generate_sales_targets(
        candidate_registry=candidate_registry,
        our_stores=our_stores,
        trdar_boundary=trdar_boundary,
        district_sales_growth=district_sales_growth,
        district_foot_traffic_index=district_foot_traffic_index,
        hero_stores=hero_stores,
        top_n=top_n,
    )


def _clean_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _clean_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def to_bulk_upsert_items(ranked: pd.DataFrame) -> list[dict]:
    """generate_sales_targets()/collect_and_generate() 결과를 BE의
    POST /api/internal/sales-targets/bulk 요청 바디(SalesTargetItemRequest 리스트)로 변환한다.

    businessName/address/totalScore가 없는(NaN) 행은 걸러낸다 — BE의 @Valid가 리스트 안 항목
    하나라도 검증에 실패하면 요청 전체를 거부하기 때문에, 여기서 미리 방어적으로 걸러서
    나머지 정상 항목까지 통째로 실패하는 걸 막는다.
    """
    items = []
    dropped = 0
    for _, row in ranked.iterrows():
        business_name = _clean_str(row.get("bizesNm"))
        address = _clean_str(row.get("rdnmAdr"))
        total_score = _clean_float(row.get("final_score"))
        if business_name is None or address is None or total_score is None:
            dropped += 1
            continue
        items.append({
            "businessName": business_name,
            "industry": _clean_str(row.get("indsLclsNm")),
            "address": address,
            "totalScore": total_score,
            "growthScore": _clean_float(row.get("growth_score")),
            "trafficScore": _clean_float(row.get("traffic_score")),
            "reviewScore": _clean_float(row.get("review_score")),
            "similarityScore": _clean_float(row.get("similarity_score")),
        })
    if dropped:
        print(f"[to_bulk_upsert_items] businessName/address/totalScore 누락으로 {dropped}건 제외")
    return items
