"""
서울 열린데이터광장 OpenAPI collectors
- 추정매출(상권): VwsmTrdarSelngQq
- 점포(서울시): VwsmMegaStorW
- 길단위인구(자치구): VwsmSignguFlpopW
- 문화행사 정보: culturalEventInfo
"""
import httpx
import pandas as pd


class SeoulOpenApiCollector:
    """공통 페이징/호출 로직"""

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


class SalesEstimateCollector(SeoulOpenApiCollector):
    """추정매출(상권) - 분기 파라미터 지원"""

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

    async def fetch_quarters(self, yyqu_codes: list[str]) -> pd.DataFrame:
        """예: ['20211','20212',...,'20261'] 21개 분기 전체 수집"""
        frames = []
        async with httpx.AsyncClient() as client:
            for yyqu in yyqu_codes:
                df = await self.fetch_quarter(client, yyqu)
                frames.append(df)
        return pd.concat(frames, ignore_index=True)


class StoreStatsCollector(SeoulOpenApiCollector):
    """점포(상권) - TRDAR_CD + SVC_INDUTY_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능."""

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


class FootTrafficCollector(SeoulOpenApiCollector):
    """길단위인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능."""

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


class ResidentPopulationCollector(SeoulOpenApiCollector):
    """상주인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능.
    업종과 무관한 상권 단위 지표라 SVC_INDUTY_CD 없이 조인한다.

    ★ 이 API는 다른 상권분석서비스 API(매출/점포/유동인구)와 달리 분기 extra_path를
      무시하고 매 호출마다 전체 이력(2021~현재)을 통째로 반환한다. 그래서
      FootTrafficCollector처럼 분기별로 반복 호출하면 같은 데이터가 분기 수만큼
      중복된다 — CulturalEventCollector처럼 fetch_all()로 한 번만 호출해야 한다.
    """

    SERVICE_NAME = "VwsmTrdarRepopQq"


class WorkplacePopulationCollector(SeoulOpenApiCollector):
    """직장인구(상권) - TRDAR_CD + STDR_YYQU_CD 기준. 매출 데이터와 직접 join 가능.
    업종과 무관한 상권 단위 지표라 SVC_INDUTY_CD 없이 조인한다.

    ★ ResidentPopulationCollector와 동일하게 분기 extra_path를 무시하고 매 호출마다
      전체 이력을 통째로 반환하는 API라 fetch_all()로 한 번만 호출해야 한다(실측 확인함).
      또한 직장인구는 4분기에 한 번만 갱신되고 다음 해 1~3분기 값이 동일하게 채워지는
      원자료 특성이 있다 — 분기별 세밀한 변화 해석 시 주의.
    """

    SERVICE_NAME = "VwsmTrdarWrcPopltnQq"


class CulturalEventCollector(SeoulOpenApiCollector):
    """문화행사 정보 - 축제/행사 데이터"""

    SERVICE_NAME = "culturalEventInfo"


class SubwayStationGeomCollector:
    """지하철역 좌표(서울교통빅데이터플랫폼 t-data.seoul.go.kr) - 대규모점포와 달리
    openapi.seoul.go.kr 계열이 아니라 응답 구조가 다르다(SERVICE_NAME으로 감싸지 않은
    raw JSON 배열, 인증키를 URL 경로가 아니라 쿼리파라미터 apikey로 전달, HTTP는
    HTTPS로 리다이렉트됨) - SeoulOpenApiCollector를 상속하지 않는다.
    실측 확인: 784개 역, 좌표(convX=경도, convY=위도, WGS84) 전부 채워짐.
    """

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


class BigStoreCollector(SeoulOpenApiCollector):
    """대규모점포(백화점/대형마트/쇼핑센터 등) 인허가 정보 - 지방행정 인허가데이터.
    TRDAR_CD/STDR_YYQU_CD 기준이 아니라 개별 시설(사업장) 단위 + 좌표(X, Y)로 나오므로,
    상권과의 결합은 공간 조인(사업장 좌표 ↔ 상권 경계/중심점)으로 해야 한다.
    영업상태명(TRDSTATENM)·인허가일자(APVPERMYMD)·폐업일자(DCBYMD)로 개폐업 시점 판정.
    """

    SERVICE_NAME = "LOCALDATA_082501"


def all_quarter_codes(start_year: int = 2021, end_year: int = 2026, end_quarter: int = 1) -> list[str]:
    codes = []
    for year in range(start_year, end_year + 1):
        max_q = end_quarter if year == end_year else 4
        for q in range(1, max_q + 1):
            codes.append(f"{year}{q}")
    return codes

