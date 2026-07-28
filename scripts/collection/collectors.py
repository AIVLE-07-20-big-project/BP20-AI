# 서울 열린데이터광장 OpenAPI collectors
import httpx
import pandas as pd


# 공통 페이징/호출 로직
class SeoulOpenApiCollector:


    BASE_URL = "http://openapi.seoul.go.kr:8088"
    SERVICE_NAME: str = ""

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


# 추정매출(상권) - 분기 파라미터 지원
class SalesEstimateCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarSelngQq"

    async def fetch_quarter(self, client: httpx.AsyncClient, yyqu_cd: str, page_size: int = 1000) -> pd.DataFrame:
        rows, start = [], 1
        while True:
            page = await self.fetch_page(client, start, start + page_size - 1, extra_path=yyqu_cd)
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        return pd.DataFrame(rows)

    # 예: ['20211','20212',...,'20261'] 21개 분기 전체 수집
    async def fetch_quarters(self, yyqu_codes: list[str]) -> pd.DataFrame:

        frames = []
        async with httpx.AsyncClient() as client:
            for yyqu in yyqu_codes:
                df = await self.fetch_quarter(client, yyqu)
                frames.append(df)
        return pd.concat(frames, ignore_index=True)


# 점포(상권) - TRDAR_CD + SVC_INDUTY_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class StoreStatsCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarStorQq"

    async def fetch_quarter(self, client: httpx.AsyncClient, yyqu_cd: str, page_size: int = 1000) -> pd.DataFrame:
        rows, start = [], 1
        while True:
            page = await self.fetch_page(client, start, start + page_size - 1, extra_path=yyqu_cd)
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        return pd.DataFrame(rows)

    async def fetch_quarters(self, yyqu_codes: list[str]) -> pd.DataFrame:
        frames = []
        async with httpx.AsyncClient() as client:
            for yyqu in yyqu_codes:
                df = await self.fetch_quarter(client, yyqu)
                frames.append(df)
        return pd.concat(frames, ignore_index=True)


# 길단위인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능
class FootTrafficCollector(SeoulOpenApiCollector):


    SERVICE_NAME = "VwsmTrdarFlpopQq"

    async def fetch_quarter(self, client: httpx.AsyncClient, yyqu_cd: str, page_size: int = 1000) -> pd.DataFrame:
        rows, start = [], 1
        while True:
            page = await self.fetch_page(client, start, start + page_size - 1, extra_path=yyqu_cd)
            if not page:
                break
            rows.extend(page)
            if len(page) < page_size:
                break
            start += page_size
        return pd.DataFrame(rows)

    async def fetch_quarters(self, yyqu_codes: list[str]) -> pd.DataFrame:
        frames = []
        async with httpx.AsyncClient() as client:
            for yyqu in yyqu_codes:
                df = await self.fetch_quarter(client, yyqu)
                frames.append(df)
        return pd.concat(frames, ignore_index=True)


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
    return codes

