# 서울 열린데이터광장 OpenAPI collectors
import asyncio
import os
from pathlib import Path

import httpx
import pandas as pd

# 분기별 조회 결과 로컬 캐시 디렉터리. 설정돼 있으면(예: 로컬 개발용 .env) 과거 확정 분기는
# 파일에서 읽고 API를 다시 안 부른다 — 매 실행마다 21개 분기를 전부 재수집하던 걸 줄이기 위함
# (STORE_REGISTRY_LOCAL_CSV 우회와 같은 패턴). 운영 환경에서는 이 변수를 안 쓰므로 항상 실시간 API를 쓴다.
_QUARTER_CACHE_DIR = os.environ.get("SEOUL_API_QUARTER_CACHE_DIR")


def _quarter_cache_path(service_name: str, yyqu_cd: str) -> Path | None:
    return Path(_QUARTER_CACHE_DIR) / service_name / f"{yyqu_cd}.csv" if _QUARTER_CACHE_DIR else None


def _read_quarter_cache(service_name: str, yyqu_cd: str) -> pd.DataFrame | None:
    path = _quarter_cache_path(service_name, yyqu_cd)
    return pd.read_csv(path, dtype=str) if path and path.exists() else None


def _write_quarter_cache(service_name: str, yyqu_cd: str, df: pd.DataFrame) -> None:
    path = _quarter_cache_path(service_name, yyqu_cd)
    if path and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


# 공통 페이징/호출 로직
class SeoulOpenApiCollector:


    BASE_URL = "http://openapi.seoul.go.kr:8088"
    SERVICE_NAME: str = ""
    # 분기별 동시 호출 상한. 게이트웨이에 한꺼번에 너무 많은 요청이 몰리는 걸 막기 위한 안전판이다
    # (실측: openapi.seoul.go.kr 자체는 건당 0.55초 안팎으로 빠름 — 병목은 21개 분기 순차 대기였음).
    MAX_CONCURRENT_QUARTERS = 5

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _build_url(self, start: int, end: int, extra_path: str = "") -> str:
        url = f"{self.BASE_URL}/{self.api_key}/json/{self.SERVICE_NAME}/{start}/{end}/"
        if extra_path:
            url += f"{extra_path}/"
        return url

    async def fetch_page(
        self, client: httpx.AsyncClient, start: int, end: int, extra_path: str = ""
    ) -> list[dict]:
        url = self._build_url(start, end, extra_path)
        res = await client.get(url, timeout=30.0)
        res.raise_for_status()
        data = res.json()
        if self.SERVICE_NAME not in data:

            raise RuntimeError(f"API error: {data}")
        return data[self.SERVICE_NAME]["row"]

    async def fetch_all(self, page_size: int = 1000, extra_path: str = "") -> pd.DataFrame:
        rows, start = [], 1
        async with httpx.AsyncClient() as client:
            while True:
                page = await self.fetch_page(client, start, start + page_size - 1, extra_path)
                if not page:
                    break
                rows.extend(page)
                if len(page) < page_size:
                    break
                start += page_size
        return pd.DataFrame(rows)

    # 분기 파라미터를 지원하는 서비스(매출/유동인구/점포수)가 공용으로 쓰는 로직.
    # use_cache=False로 부르면(최신 분기) 캐시를 읽지도 쓰지도 않는다 — 최신 분기는 아직 데이터가
    # 정정될 수 있어서 항상 최신 API 응답을 써야 한다.
    async def fetch_quarter(
        self, client: httpx.AsyncClient, yyqu_cd: str, page_size: int = 1000, use_cache: bool = True
    ) -> pd.DataFrame:
        if use_cache:
            cached = _read_quarter_cache(self.SERVICE_NAME, yyqu_cd)
            if cached is not None:
                return cached

        rows, start = [], 1
        while True:
            page = await self.fetch_page(client, start, start + page_size - 1, extra_path=yyqu_cd)
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size

        df = pd.DataFrame(rows)
        if use_cache:
            _write_quarter_cache(self.SERVICE_NAME, yyqu_cd, df)
        return df

    # 예: ['20211','20212',...,'20261'] 21개 분기를 동시에(최대 MAX_CONCURRENT_QUARTERS개) 수집한다.
    # 기존엔 이 21개를 순차로(건당 ~0.5초) 돌아서 배치 전체가 몇 분씩 걸렸다.
    async def fetch_quarters(self, yyqu_codes: list[str]) -> pd.DataFrame:
        if not yyqu_codes:
            return pd.DataFrame()

        latest = yyqu_codes[-1]  # all_quarter_codes()는 오름차순이라 마지막이 최신 분기
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_QUARTERS)

        async def _bounded(client: httpx.AsyncClient, yyqu: str) -> pd.DataFrame:
            async with semaphore:
                return await self.fetch_quarter(client, yyqu, use_cache=yyqu != latest)

        async with httpx.AsyncClient() as client:
            frames = await asyncio.gather(*(_bounded(client, yyqu) for yyqu in yyqu_codes))
        return pd.concat(frames, ignore_index=True)


# 추정매출(상권) - 분기 파라미터 지원
class SalesEstimateCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarSelngQq"


# 점포(상권) - TRDAR_CD + SVC_INDUTY_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class StoreStatsCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarStorQq"


# 길단위인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class FootTrafficCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarFlpopQq"


# 상주인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class ResidentPopulationCollector(SeoulOpenApiCollector):









    SERVICE_NAME = "VwsmTrdarRepopQq"


# 직장인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class WorkplacePopulationCollector(SeoulOpenApiCollector):









    SERVICE_NAME = "VwsmTrdarWrcPopltnQq"


# 문화행사 정보 - 축제/행사 데이터
class CulturalEventCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "culturalEventInfo"


# 지하철역 좌표(서울교통빅데이터플랫폼 t-data.seoul.go.kr) - 대규모점포와 달리
class SubwayStationGeomCollector:







    BASE_URL = "https://t-data.seoul.go.kr/apig/apiman-gateway/tapi/TaimsKsccDvSubwayStationGeom/1.0"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def fetch_all(self) -> pd.DataFrame:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            res = await client.get(
                self.BASE_URL,
                params={"apikey": self.api_key, "startRow": 1, "rowCnt": 2000},
                timeout=30.0,
            )
            res.raise_for_status()
            return pd.DataFrame(res.json())


# 대규모점포(백화점/대형마트/쇼핑센터 등) 인허가 정보 - 지방행정 인허가데이터
class BigStoreCollector(SeoulOpenApiCollector):






    SERVICE_NAME = "LOCALDATA_082501"


def all_quarter_codes(start_year: int = 2021, end_year: int = 2026, end_quarter: int = 1) -> list[str]:
    codes = []
    for year in range(start_year, end_year + 1):
        max_q = end_quarter if year == end_year else 4
        for q in range(1, max_q + 1):
            codes.append(f"{year}{q}")

    # 개발 전용 단축 옵션. 설정 시 최신 분기 쪽 N개만 남긴다 — 로컬에서 배치를 빠르게 돌려보기
    # 위한 용도로, 성장률(latest_vs_previous_growth)은 최신 2개 분기만 있으면 계산되니 최소 2 이상.
    # 운영 배치에는 절대 쓰면 안 된다: 분기 범위가 좁아지면 지표 계산창이 달라져 실제 점수와
    # 어긋난다. 그래서 켜져 있으면 눈에 띄게 경고를 찍는다.
    dev_limit = os.environ.get("SALES_TARGET_DEV_QUARTER_LIMIT")
    if dev_limit:
        codes = codes[-int(dev_limit):]
        print(
            f"[all_quarter_codes] SALES_TARGET_DEV_QUARTER_LIMIT={dev_limit} 적용 — "
            f"분기 {len(codes)}개만 사용({codes[0]}~{codes[-1]}). 운영 배치에서는 이 환경변수를 절대 설정하지 말 것."
        )
    return codes


# 영업 타겟 배치 전용. district_metrics.latest_vs_previous_growth()/latest_quarter_values()는
# 상권별로 "실제 존재하는 가장 최근 2개 분기"만 쓴다 — 그런데도 지금까지 all_quarter_codes()로
# 2021년부터 21개 분기를 통째로 받아왔다(매출/유동인구/점포수 3개 지표 각각). 상권마다 데이터
# 공표 시점이 살짝 다를 수 있어서 정확히 2개만 받으면 일부 상권이 growth_rate=NaN이 될 위험이
# 있어, 여유를 두고 최근 lookback개(기본 4개, 1년치)를 받는다. 21 -> 4로 줄어드는 만큼 배치
# 소요시간도 그대로 줄어든다(내부 페이지네이션이 여전히 병목의 상당 부분이라 분기 수 자체를
# 줄이는 게 동시성 조정보다 효과가 크다).
def latest_quarter_codes(end_year: int = 2026, end_quarter: int = 1, lookback: int = 4) -> list[str]:
    codes = all_quarter_codes(start_year=max(2016, end_year - 2), end_year=end_year, end_quarter=end_quarter)
    return codes[-lookback:]


# 상권영역(경계) - 상권코드별 대표좌표(EPSG:5181)와 면적. 상가업소 데이터의 위경도를
# 상권코드에 매핑하는 데 쓴다(app/sales_target/geo_matching.py).
# 검증(2026-07-29): 좌표계가 EPSG:5181인 것을 실제 변환 결과로 확인함.
class TrdarBoundaryCollector(SeoulOpenApiCollector):

    SERVICE_NAME = "TbgisTrdarRelm"

