# 대상 경로(AI 레포): scripts/collection/store_registry_collector.py
#
# 소상공인시장진흥공단 상가(상권)정보 오픈API 콜렉터 (공공데이터포털, 서비스 ID 15012005)
# https://www.data.go.kr/data/15012005/openapi.do
#
# 신규 가맹점 영업 타겟 추천 기능의 "전국(우선 서울) 업장 목록" 데이터 소스.
# 기존 scripts/collection/collectors.py의 서울 열린데이터광장 콜렉터들과 같은 방식(비동기 페이징)으로 맞췄다.
#
# 실제 서비스 키로 검증 완료(2026-07-29):
#   - GET http://apis.data.go.kr/B553077/api/open/sdsc2/storeListInDong
#     ?divId=signguCd&key=11680(강남구) -> resultCode=00, 정상 데이터 반환
#   - GET .../storeListInRadius?cx=...&cy=...&radius=... -> resultCode=00, 정상 데이터 반환
#   - divId=adongCd 로 10자리 코드(예: 1168064000)를 넣으면 NODATA_ERROR가 난다.
#     실제 응답의 adongCd 필드는 8자리(예: "11680750")라 자릿수가 다른 코드를 넣었던 게 원인이었다.
#     그래서 이 콜렉터는 구(시군구) 단위 조회를 기본 경로로 쓴다 — 서울 25개 구 코드만 있으면 되고,
#     행정동코드 목록을 별도 파일로 준비할 필요가 없어진다(기존 설계보다 단순해짐).
#
# 응답 구조(실제 확인됨): {"header": {"resultCode", "resultMsg", "description", "columns"}, "body": {"items": [...]}}
# item 필드(실제 확인됨): bizesId, bizesNm, brchNm, indsLclsCd, indsLclsNm, indsMclsCd, indsMclsNm,
#   indsSclsCd, indsSclsNm, ksicCd, ksicNm, ctprvnCd, ctprvnNm, signguCd, signguNm, adongCd, adongNm,
#   ldongCd, ldongNm, lnoAdr(지번주소), rdnmAdr(도로명주소), bldNm, lon, lat 등.

from __future__ import annotations

import httpx
import pandas as pd

# 표준 시군구코드(법정동코드 앞 5자리) — 서울 25개 자치구. 정부 표준코드로 사실상 고정값이다.
# 강남구(11680)는 이번 검증 호출에서 실제로 정상 응답을 확인했다.
SEOUL_SIGUNGU_CODES: dict[str, str] = {
    "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200",
    "광진구": "11215", "동대문구": "11230", "중랑구": "11260", "성북구": "11290",
    "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380",
    "서대문구": "11410", "마포구": "11440", "양천구": "11470", "강서구": "11500",
    "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590",
    "관악구": "11620", "서초구": "11650", "강남구": "11680", "송파구": "11710",
    "강동구": "11740",
}


class StoreRegistryCollector:
    """구/행정동 단위로 상가업소 목록을 조회한다. (서비스명: storeListInDong)

    기본 사용법은 fetch_sigungus() — 서울 25개 구를 한 번씩만 호출하면 전체를 커버한다.
    더 세분화하고 싶으면(예: 특정 구만 동 단위로 쪼개기) fetch_dong()/fetch_dongs()에
    실제 8자리 adongCd 값을 넣어 쓸 수 있다. 단, 10자리가 아니라 8자리라는 점을 주의.
    """

    BASE_URL = "http://apis.data.go.kr/B553077/api/open/sdsc2/storeListInDong"
    PAGE_SIZE = 1000

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _build_params(self, div_id: str, area_cd: str, page: int, inds_lcls_cd: str | None) -> dict:
        params = {
            "serviceKey": self.api_key,
            "type": "json",
            "divId": div_id,
            "key": area_cd,
            "numOfRows": self.PAGE_SIZE,
            "pageNo": page,
        }
        if inds_lcls_cd:
            # 업종대분류코드로 좁히고 싶을 때 사용 (예: 'Q' = 음식). 비워두면 전 업종.
            params["indsLclsCd"] = inds_lcls_cd
        return params

    async def _fetch_area(
        self,
        client: httpx.AsyncClient,
        div_id: str,
        area_cd: str,
        inds_lcls_cd: str | None = None,
    ) -> pd.DataFrame:
        rows, page = [], 1
        while True:
            params = self._build_params(div_id, area_cd, page, inds_lcls_cd)
            res = await client.get(self.BASE_URL, params=params, timeout=30.0)
            res.raise_for_status()
            payload = res.json()

            header = payload.get("header", {})
            if header.get("resultCode") not in (None, "00"):
                # NODATA_ERROR(03) 등은 예외가 아니라 "이 구역엔 데이터가 없다"는 정상 응답으로 취급한다.
                if header.get("resultCode") == "03":
                    break
                raise RuntimeError(f"{div_id}={area_cd} 호출 실패: {header}")

            items = self._extract_items(payload)
            if not items:
                break
            rows.extend(items)
            if len(items) < self.PAGE_SIZE:
                break
            page += 1

        df = pd.DataFrame(rows)
        if not df.empty:
            df["queried_div_id"] = div_id
            df["queried_area_cd"] = area_cd
        return df

    async def fetch_sigungu(
        self,
        client: httpx.AsyncClient,
        sigungu_cd: str,
        inds_lcls_cd: str | None = None,
    ) -> pd.DataFrame:
        return await self._fetch_area(client, "signguCd", sigungu_cd, inds_lcls_cd)

    async def fetch_sigungus(
        self,
        sigungu_codes: list[str] | None = None,
        inds_lcls_cd: str | None = None,
    ) -> pd.DataFrame:
        """구 단위로 순회 수집한다. 기본값은 서울 25개 구 전체."""
        codes = sigungu_codes or list(SEOUL_SIGUNGU_CODES.values())
        return await self._fetch_many(codes, "signguCd", inds_lcls_cd)

    async def fetch_dong(
        self,
        client: httpx.AsyncClient,
        adong_cd: str,
        inds_lcls_cd: str | None = None,
    ) -> pd.DataFrame:
        """행정동 단위 조회. adong_cd는 8자리(예: '11680750')여야 한다 — 10자리 코드를 넣으면 NODATA_ERROR."""
        return await self._fetch_area(client, "adongCd", adong_cd, inds_lcls_cd)

    async def fetch_dongs(
        self,
        dong_codes: list[str],
        inds_lcls_cd: str | None = None,
    ) -> pd.DataFrame:
        return await self._fetch_many(dong_codes, "adongCd", inds_lcls_cd)

    async def _fetch_many(
        self,
        area_codes: list[str],
        div_id: str,
        inds_lcls_cd: str | None,
    ) -> pd.DataFrame:
        frames = []
        async with httpx.AsyncClient() as client:
            for area_cd in area_codes:
                try:
                    df = await self._fetch_area(client, div_id, area_cd, inds_lcls_cd)
                except httpx.HTTPError as exc:
                    print(f"[store_registry] {div_id}={area_cd} 수집 실패, 건너뜀: {exc}")
                    continue
                if not df.empty:
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _extract_items(payload: dict) -> list[dict]:
        # 실제 응답 구조로 확인됨: {"header": {...}, "body": {"items": [...]}}
        # items가 {"item": [...]}로 한 번 더 감싸져 오는 경우도 있어 방어적으로 처리한다.
        body = payload.get("body", {})
        items = body.get("items", [])
        if isinstance(items, dict):
            items = items.get("item", [])
        return items or []


def load_dong_codes(path: str, code_column: str | None = None) -> list[str]:
    """(선택) 행정동 단위로 더 세분화하고 싶을 때만 쓴다. 서울 전체 수집은 SEOUL_SIGUNGU_CODES로 충분하다.

    행정표준코드관리시스템(https://www.code.go.kr)에서 내려받은 코드 CSV/TXT를 코드 리스트로 반환한다.
    이 API의 adongCd는 8자리이므로, 파일의 코드 자릿수가 다르면 앞 8자리만 잘라 쓰는 등 변환이 필요할 수 있다.
    """
    df = pd.read_csv(path, dtype=str)
    if code_column is None:
        code_column = next(
            (c for c in df.columns if "코드" in c or c.strip().lower() in {"adongcd", "code"}),
            df.columns[0],
        )
    return df[code_column].dropna().astype(str).unique().tolist()
