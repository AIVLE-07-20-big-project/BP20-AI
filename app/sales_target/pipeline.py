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

from app.sales_target.backend_client import BackendStoreRegistryClient, SalesTargetIngestClient
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
from app.sales_target.google_places import fetch_review_stats
from app.sales_target.matching import exclude_existing_merchants
from app.sales_target.scoring import (
    final_scores,
    foot_traffic_index,
    growth_scores,
    percentile_score,
    preliminary_scores,
    review_activity_score,
)

# 우수 가맹점 유사도(similarity_score) 계산에는 hero_stores(자사 우수 가맹점 목록, 리뷰
# 평점/표준편차 포함)가 필요하다. 아직 BE가 리뷰 통계를 내려주지 않는 상태에서 호출되면(또는
# 우수 가맹점이 0곳이면) 이 중간값으로 대체한다. "판단 불가"를 억지로 0점 처리하면 오히려
# 순위를 왜곡하므로 중립값을 쓴다(반면 review_score는 아예 축 자체를 빼는 방식을 쓴다 —
# scoring.preliminary_scores 참고. 두 축의 설계가 다른 이유는 similarity는 항상 계산 시도라도
# 가능하지만, review는 2단계 전엔 아예 데이터 자체가 없기 때문).
_SIMILARITY_PLACEHOLDER = 50.0


def generate_sales_targets(
    candidate_registry: pd.DataFrame,
    our_stores: pd.DataFrame,
    trdar_boundary: pd.DataFrame,
    district_sales_growth: pd.DataFrame,
    district_foot_traffic_index: pd.DataFrame,
    district_store_count: pd.DataFrame,
    hero_stores: pd.DataFrame | None = None,
    top_n: int | None = None,
    rejected_addresses: list[str] | None = None,
) -> pd.DataFrame:
    """신규 영업 타겟 후보 리스트를 만든다. 네트워크 호출 없는 순수 함수.

    인자:
      candidate_registry: store_registry_collector.py 결과. lon/lat(WGS84) 컬럼 필요.
      our_stores: backend_client.py 결과. address, salesGrowthRate 컬럼 필요(매칭/우수가맹점용).
      trdar_boundary: TrdarBoundaryCollector 결과(변환 전, XCNTS_VALUE/YDNTS_VALUE/RELM_AR 포함).
      district_sales_growth: district_metrics로 미리 계산한 DataFrame[trdar_cd, growth_rate].
      district_foot_traffic_index: district_metrics로 미리 계산한 DataFrame[trdar_cd, traffic_index]
        (foot_traffic_index()로 이미 가중합까지 끝낸, 0~100 정규화 전 원시 지수).
      district_store_count: district_metrics로 미리 계산한 DataFrame[trdar_cd, store_count]
        (StoreStatsCollector의 STOR_CO, 상권 내 점포 수 — 우수 가맹점 유사도 벡터의 "경쟁업소 수"
        축에만 쓰인다. final_score 자체에는 별도 가중치로 들어가지 않는다).
      hero_stores: hero_store.select_hero_stores() 결과(category, address, reviewCount,
        reviewAvgRating 컬럼 필요). None이거나 빈 DataFrame이면 similarity_score는
        _SIMILARITY_PLACEHOLDER(50.0)로 채운다 — BE가 아직 리뷰 통계를 안 내려주는 동안의
        임시 상태이며, 실제 컬럼이 갖춰지면 자동으로 실값 계산으로 전환된다.
      top_n: 상위 N개만 반환하고 싶을 때 지정. None이면 전체 반환.
      rejected_addresses: 5단계(피드백 루프). 영업팀이 EXCLUDED 처리한 후보 주소 목록
        (backend_client.SalesTargetIngestClient.fetch_excluded_addresses() 결과). 다음 배치에서
        같은 주소가 또 추천되지 않도록 our_stores와 동일하게 exclude_existing_merchants()로
        걸러낸다. None/빈 리스트면 아무것도 걸러내지 않는다.

    반환: final_score 내림차순으로 정렬된 DataFrame. growth_score/traffic_score/review_score/
      similarity_score/final_score 컬럼이 추가된다. review_score는 이 시점엔 항상 NaN이다 —
      리뷰 데이터는 아직 없으므로(2단계 apply_review_scores()가 채운다), final_score는
      growth/traffic/similarity 3축만으로 계산한다(preliminary_scores(), 리뷰 가중치 0.20은
      나머지 세 축에 비율대로 재분배됨).
    """
    if candidate_registry.empty:
        return candidate_registry.copy()

    filtered = exclude_existing_merchants(candidate_registry, our_stores)
    if rejected_addresses:
        filtered = exclude_existing_merchants(
            filtered, pd.DataFrame({"address": rejected_addresses})
        )
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
    ).merge(
        district_store_count.rename(columns={"store_count": "_district_store_count"}),
        on="trdar_cd",
        how="left",
    )

    merged["growth_score"] = growth_scores(merged["_district_growth_rate"])
    merged["traffic_score"] = percentile_score(merged["_district_traffic_index"])
    # 1차 스코어링엔 리뷰 데이터가 없다 — 가짜 중간값을 채우는 대신 NaN으로 비워두고 2단계
    # (apply_review_scores)가 실값을 채우게 한다.
    merged["review_score"] = float("nan")

    if hero_stores is not None and not hero_stores.empty:
        candidate_features = pd.DataFrame({
            # indsLclsNm(업종 대분류, 예: "음식")은 너무 뭉뚱그려져 있어(카페/치킨집/한식당이 전부
            # "음식"으로 같이 잡힘) 유사도 비교에 안 맞다. 자사 매장 category(자유 텍스트, "카페"
            # 같은 표기)와 granularity가 더 가까운 중분류(indsMclsNm, 예: "커피점/카페")를 쓴다.
            "industry": merged["indsMclsNm"] if "indsMclsNm" in merged.columns else None,
            "district": merged["signguNm"] if "signguNm" in merged.columns else None,
            "district_growth_rate": merged["_district_growth_rate"],
            "district_traffic_index": merged["_district_traffic_index"],
            # 비가맹 업장 리뷰 수집기가 아직 없어 후보 쪽 리뷰 지표는 항상 0이다. hero_stores
            # 쪽에만 실제 리뷰 통계가 들어가므로, 유사도는 현재 [업종/지역/상권지표/경쟁업소 수]
            # 5개 축으로 사실상 갈리고 리뷰 2개 축은 "후보가 hero와 다르다"는 방향으로만 작용한다.
            "review_count": 0.0,
            "review_avg_rating": 0.0,
            "competitor_count": merged["_district_store_count"],
        })

        district_avg = sigungu_average_district_metrics(
            merged,
            sigungu_col="signguNm" if "signguNm" in merged.columns else merged.columns[0],
            growth_col="_district_growth_rate",
            traffic_col="_district_traffic_index",
            store_count_col="_district_store_count",
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
        hero_features["competitor_count"] = hero_features["district_store_count"]

        merged["similarity_score"] = hero_similarity_scores(candidate_features, hero_features)
    else:
        merged["similarity_score"] = _SIMILARITY_PLACEHOLDER

    merged["final_score"] = preliminary_scores(
        merged["growth_score"],
        merged["traffic_score"],
        merged["similarity_score"],
    )

    ranked = merged.sort_values("final_score", ascending=False).reset_index(drop=True)
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked


def apply_review_scores(
    candidates: pd.DataFrame,
    review_stats: pd.DataFrame,
    address_col: str = "rdnmAdr",
) -> pd.DataFrame:
    """6단계(리뷰 2단계 반영): generate_sales_targets()가 growth/traffic/similarity 3축만으로
    이미 추려준 top_n 후보(review_score는 이 시점엔 전부 NaN)에, google_places.fetch_review_stats()
    결과를 합쳐 review_score/final_score를 갱신한다.

    구글 플레이스에서 실제로 매칭된 후보만 review_score가 채워지고, final_score도 4축
    가중합(final_scores())으로 다시 계산된다. 매칭 실패한(장소를 못 찾았거나 평점 데이터가 없는)
    후보는 결과에서 완전히 제외한다.

    이전엔 매칭 실패 후보를 "리뷰 축만 빼고 3축으로" 평가해서 남겨뒀는데, 실제 배치에서 이
    설계가 역효과를 냈다: 리뷰 데이터가 없다는 사실 자체가 오히려 순위를 끌어올리는 왜곡이
    생겨서(3축만 재가중합하면 종종 4축 평균보다 높게 나옴), critic_agent가 매 배치 상위권을
    "review_score가 null인데 점수가 비정상적으로 높다"고 반복해서 flag했다(2026-08-04). "판단
    불가"를 관대하게 봐주는 대신, 리뷰를 확인할 수 없는 후보는 영업 우선순위 후보에서 제외한다.

    review_stats 자체가 비어있으면(구글 플레이스 호출이 이번 배치 전체에서 실패한 경우 등)
    "이 후보들이 리뷰가 없다"가 아니라 "이번엔 리뷰 조회 자체가 안 됐다"는 다른 상황이라
    아무도 제외하지 않고 후보를 그대로 반환한다(3축 스코어 유지).
    """
    if review_stats.empty:
        return candidates

    merged = candidates.merge(review_stats, on=address_col, how="left")
    has_review = merged["review_count"].notna()

    max_days = merged["days_since_latest_review"].max()
    filled_days = merged["days_since_latest_review"].fillna(max_days if pd.notna(max_days) else 0.0)

    real_review_score = review_activity_score(
        review_count_pct=percentile_score(merged["review_count"].fillna(0)),
        avg_rating_pct=percentile_score(merged["avg_rating"].fillna(0)),
        recency_pct=percentile_score(-filled_days),
        growth_trend_pct=percentile_score(merged["review_growth_trend"].fillna(0)),
    )
    merged["review_score"] = real_review_score.where(has_review, merged["review_score"])
    merged["final_score"] = final_scores(
        merged["growth_score"], merged["traffic_score"], merged["review_score"], merged["similarity_score"]
    )

    matched = merged.loc[has_review].copy()
    return matched.sort_values("final_score", ascending=False).reset_index(drop=True)


# ── 서울시 상권분석서비스 공공데이터 컬럼명 ─────────────────────────────
# 검증 완료(2026-07-29, 실제 API 라이브 호출로 확인):
#   추정매출(VwsmTrdarSelngQq): THSMON_SELNG_AMT("당월 매출금액") 확인됨
#   길단위인구(VwsmTrdarFlpopQq): TMZON_06_11_FLPOP_CO(오전대), TMZON_11_14_FLPOP_CO(점심대),
#     TMZON_14_17_FLPOP_CO(오후대) 확인됨 — 과제정의서가 말하는 "오전/점심/오후" 유동인구와
#     정확히 대응된다.
#   점포(VwsmTrdarStorQq): STOR_CO("점포수") — scripts/modeling/sales_analysis.py의 매출
#     위험도 분석에서 이미 검증되어 쓰이고 있는 컬럼명을 그대로 재사용(경쟁업소 수 축).
_FIELD_NAMES = {
    "sales_trdar_col": "TRDAR_CD",
    "sales_quarter_col": "STDR_YYQU_CD",
    "sales_amount_col": "THSMON_SELNG_AMT",
    "traffic_trdar_col": "TRDAR_CD",
    "traffic_quarter_col": "STDR_YYQU_CD",
    "traffic_morning_col": "TMZON_06_11_FLPOP_CO",
    "traffic_lunch_col": "TMZON_11_14_FLPOP_CO",
    "traffic_afternoon_col": "TMZON_14_17_FLPOP_CO",
    "store_trdar_col": "TRDAR_CD",
    "store_quarter_col": "STDR_YYQU_CD",
    "store_count_col": "STOR_CO",
}


async def collect_and_generate(
    *,
    store_registry_api_key: str,
    seoul_api_key: str,
    backend_base_url: str,
    backend_internal_api_key: str,
    quarter_codes: list[str],
    top_n: int | None = None,
    google_places_api_key: str | None = None,
) -> pd.DataFrame:
    """실제 콜렉터/BE 클라이언트를 호출해서 데이터를 모으고 generate_sales_targets()에 넘긴다."""
    # 순환 import를 피하려고 여기서 지역 임포트한다(scripts.collection이 app을 참조하지 않지만,
    # 반대 방향 임포트를 파일 상단에 두면 이 모듈을 import하는 다른 코드가 항상 scripts 의존성까지
    # 끌고 오게 된다 — 이 함수를 실제로 쓸 때만 필요하도록 지연 임포트했다).
    from scripts.collection.collectors import (
        FootTrafficCollector,
        SalesEstimateCollector,
        StoreStatsCollector,
        TrdarBoundaryCollector,
    )
    from scripts.collection.store_registry_collector import StoreRegistryCollector

    f = _FIELD_NAMES

    candidate_registry = await StoreRegistryCollector(store_registry_api_key).fetch_sigungus()

    backend_client = BackendStoreRegistryClient(backend_base_url, backend_internal_api_key)
    our_stores = await backend_client.fetch_our_stores()

    # 5단계(피드백 루프): 영업팀이 이미 EXCLUDED 처리한 후보는 다음 배치에서 다시 추천하지 않는다.
    ingest_client = SalesTargetIngestClient(backend_base_url, backend_internal_api_key)
    rejected_addresses = await ingest_client.fetch_excluded_addresses()

    # 우수 가맹점 선정에는 reviewAvgRating/reviewRatingStd가 필요하다. BE(StoreRegistryEntryResponse)는
    # 2026-08-03 기준 이미 이 필드들을 내려준다(businessNumber/name/category/address/
    # salesGrowthRate/reviewCount/reviewAvgRating/reviewRatingStd) — select_hero_stores()가
    # 실제로 실값을 계산해서 similarity_score가 placeholder(50.0)가 아니게 된다. 그래도 방어적으로
    # 컬럼 체크는 남겨둔다 — BE 응답이 비어있거나(리뷰 통계 없는 매장만 있는 경우 등) 필드가
    # 빠진 경우엔 여전히 hero_stores=None으로 안전하게 넘어가야 한다.
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

    store_stats_raw = await StoreStatsCollector(seoul_api_key).fetch_quarters(quarter_codes)
    store_district_quarterly = aggregate_district_quarterly(
        store_stats_raw, f["store_trdar_col"], f["store_quarter_col"], f["store_count_col"]
    )
    district_store_count = latest_quarter_values(
        store_district_quarterly, f["store_trdar_col"], f["store_quarter_col"], f["store_count_col"]
    ).rename(columns={f["store_trdar_col"]: "trdar_cd", f["store_count_col"]: "store_count"})

    ranked = generate_sales_targets(
        candidate_registry=candidate_registry,
        our_stores=our_stores,
        trdar_boundary=trdar_boundary,
        district_sales_growth=district_sales_growth,
        district_foot_traffic_index=district_foot_traffic_index,
        district_store_count=district_store_count,
        hero_stores=hero_stores,
        top_n=top_n,
        rejected_addresses=rejected_addresses,
    )

    # 6단계(리뷰 2단계 반영): growth/traffic/similarity로 이미 추려진 top_n 후보에 한해서만
    # Google Places로 리뷰 통계를 가져온다 — 후보 전체(최대 약 53만 건)에 돌리는 건 비용/시간상
    # 불가능하다. 키가 없으면(GOOGLE_PLACES_API_KEY 미설정) 조용히 건너뛰고 review_score는
    # placeholder(50.0)로 유지된다.
    if google_places_api_key and not ranked.empty:
        review_stats = await fetch_review_stats(ranked, google_places_api_key)
        ranked = apply_review_scores(ranked, review_stats)

    return ranked


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

    salesPitch: 2단계(graph.py의 generate_pitch 노드, app/sales_target/pitch.py)에서 채워지는
    필드다. BE의 SalesTargetCandidate 엔티티(domain/SalesTargetCandidate.java)에는 이미
    salesPitch 컬럼이 있지만, 이 글을 쓰는 시점 기준 SalesTargetItemRequest/SalesTargetService의
    bulk upsert 경로는 이 필드를 아직 안 받는다 — 그때까진 이 키를 보내도 BE가 조용히 무시한다
    (Jackson 기본 동작). 3단계 진행 시 BE 쪽에 필드 추가가 필요하다.
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
            "industry": _clean_str(row.get("indsMclsNm")) or _clean_str(row.get("indsLclsNm")),
            "address": address,
            "totalScore": total_score,
            "growthScore": _clean_float(row.get("growth_score")),
            "trafficScore": _clean_float(row.get("traffic_score")),
            "reviewScore": _clean_float(row.get("review_score")),
            "similarityScore": _clean_float(row.get("similarity_score")),
            "salesPitch": _clean_str(row.get("sales_pitch")),
        })
    if dropped:
        print(f"[to_bulk_upsert_items] businessName/address/totalScore 누락으로 {dropped}건 제외")
    return items
