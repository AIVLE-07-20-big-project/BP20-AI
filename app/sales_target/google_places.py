# 대상 경로(AI 레포): app/sales_target/google_places.py
#
# 후보 업장 리뷰 활성도 점수(scoring.review_activity_score) 입력용 원시 리뷰 통계를 Google
# Places API에서 가져온다. 소상공인시장진흥공단 상가정보/서울시 상권분석서비스 어느 쪽도 리뷰
# 데이터를 안 줘서(scoring.preliminary_scores 주석 참고) 별도 소스가 필요했다.
#
# "Places API (New)"를 쓴다(구 Places API는 2026-08-03 기준 공식 문서에서 Legacy로 분류되고
# 신규 프로젝트엔 New 사용을 권장한다 — developers.google.com/maps/documentation/places/
# web-service/legacy/overview-legacy 확인). 구버전과 인증 방식(쿼리파라미터 key= 대신
# X-Goog-Api-Key 헤더)과 응답 스키마(place_id -> id, user_ratings_total -> userRatingCount,
# reviews[].time(유닉스초) -> reviews[].publishTime(RFC3339 문자열))가 다르다.
#
# 후보가 서울 전체 기준 최대 약 53만 건이라 전체에 이 API를 호출하는 건 비용/시간상 비현실적이다.
# pipeline.generate_sales_targets()가 review 축은 아예 빼고 growth/traffic/similarity 3축만으로
# 이미 top_n을 추려주므로(scoring.preliminary_scores 참고), 이 모듈은 그 top_n(가급적 200 이하 —
# 실제 과금이 발생하는 API라서)에 대해서만 호출한다.
#
# 정확도 한계: Place Details(New) 응답의 reviews 필드는 최대 5건의 "최신/관련도 높은" 리뷰만 준다.
# review_growth_trend는 이 5건 사이 평균 작성 간격의 역수로 근사한 값이지, 실제 장기 추세는 아니다
# (Places API가 리뷰 이력 전체나 시계열 통계를 제공하지 않기 때문).

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd

_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"


def _parse_rfc3339(value: str) -> float:
    """"2014-10-02T15:01:23Z" 같은 RFC3339 문자열을 유닉스 타임스탬프(초)로 변환한다."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


async def search_places(client: httpx.AsyncClient, api_key: str, query: str, limit: int = 3) -> list[dict]:
    """검색어로 후보 장소 목록(place_id/이름/주소)을 반환한다. 평점/리뷰수는 없다 — 필요하면
    fetch_place_details()로 별도 조회한다(과금 SKU가 다르다). review_matching_agent.py가
    이 함수를 도구 호출 결과로 그대로 LLM에 보여준다(여러 후보 중 하나를 고르거나, 다시
    검색어를 바꿀지 LLM이 판단할 수 있게 리스트로 반환)."""
    res = await client.post(
        _SEARCH_URL,
        json={"textQuery": query},
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress",
        },
    )
    res.raise_for_status()
    places = res.json().get("places", [])[:limit]
    return [
        {
            "place_id": p["id"],
            "name": (p.get("displayName") or {}).get("text"),
            "address": p.get("formattedAddress"),
        }
        for p in places
        if p.get("id")
    ]


async def find_place_id(client: httpx.AsyncClient, api_key: str, business_name: str, address: str) -> str | None:
    """단일 최상위 결과만 필요한 결정론적 경로(fetch_review_stats)용 — search_places(limit=1)의 얇은 래퍼."""
    results = await search_places(client, api_key, f"{business_name} {address}", limit=1)
    return results[0]["place_id"] if results else None


async def fetch_place_details(client: httpx.AsyncClient, api_key: str, place_id: str) -> dict:
    res = await client.get(
        _DETAILS_URL.format(place_id=place_id),
        headers={"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": "rating,userRatingCount,reviews"},
    )
    res.raise_for_status()
    return res.json()


def _growth_trend(publish_times: list[float]) -> float | None:
    """최근 리뷰(최대 5건) 작성 시각 사이 평균 간격(일)의 음수 근사치. 간격이 짧을수록(리뷰가
    자주 달릴수록) 값이 커진다 — scoring.percentile_score()에 그대로 넣으면 "활발함"이 고득점되는
    방향과 맞는다. 리뷰가 2건 미만이면 간격 자체를 계산할 수 없어 None을 반환한다."""
    if len(publish_times) < 2:
        return None
    ordered = sorted(publish_times)
    gaps_days = [(b - a) / 86400 for a, b in zip(ordered, ordered[1:])]
    return -(sum(gaps_days) / len(gaps_days))


async def fetch_review_stats(
    candidates: pd.DataFrame,
    api_key: str,
    business_name_col: str = "bizesNm",
    address_col: str = "rdnmAdr",
) -> pd.DataFrame:
    """candidates(generate_sales_targets()가 이미 추려준 top_n)의 상호명+주소로 Google Places를
    조회해 address_col 기준 review_count/avg_rating/days_since_latest_review/review_growth_trend를
    반환한다(전부 원시값 — percentile 정규화는 pipeline.apply_review_scores()가 담당).

    장소를 못 찾거나(매칭 실패) 평점 데이터가 아예 없는 업장은 결과 행 자체를 건너뛴다 —
    apply_review_scores()가 왼쪽 조인으로 처리하므로, 여기 없는 후보는 review_score가 기존
    placeholder(50.0)로 유지된다("판단 불가"를 벌점 처리하지 않는다는 기존 원칙과 동일).
    """
    now = datetime.now(timezone.utc).timestamp()
    rows = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for _, row in candidates.iterrows():
            name, address = row.get(business_name_col), row.get(address_col)
            if not name or not address:
                continue
            try:
                place_id = await find_place_id(client, api_key, str(name), str(address))
                if not place_id:
                    continue
                details = await fetch_place_details(client, api_key, place_id)
            except httpx.HTTPError:
                continue

            rating, review_count = details.get("rating"), details.get("userRatingCount")
            if rating is None or review_count is None:
                continue

            publish_times = [
                _parse_rfc3339(r["publishTime"]) for r in (details.get("reviews") or []) if r.get("publishTime")
            ]
            latest_time = max(publish_times, default=None)
            rows.append({
                address_col: address,
                "review_count": review_count,
                "avg_rating": rating,
                "days_since_latest_review": (now - latest_time) / 86400 if latest_time else None,
                "review_growth_trend": _growth_trend(publish_times),
            })

    return pd.DataFrame(rows)
