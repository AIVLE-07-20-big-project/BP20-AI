# 대상 경로(AI 레포): tests/test_store_registry_collector.py
#
# 실제 서비스 키 없이도 돌아가는 유닛 테스트. httpx.AsyncClient.get을 모킹해서
# 페이징 종료 조건, resultCode 처리(00=정상/03=데이터없음/그 외=예외), 응답 파싱 방어 로직을 검증한다.
# 저장소의 다른 테스트들과 동일하게 표준 unittest를 쓴다(IsolatedAsyncioTestCase로 async 지원,
# 별도 pytest-asyncio 의존성 불필요).
#
# 여기서 쓰는 header/body 스키마와 필드명(bizesNm 등)은 실제 서비스 키로 호출해서 확인한 응답 그대로다.

import unittest
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd

from scripts.collection.store_registry_collector import (
    SEOUL_SIGUNGU_CODES,
    StoreRegistryCollector,
    load_dong_codes,
    load_local_registry_csv,
)


def _fake_response(items, result_code="00", result_msg="NORMAL SERVICE"):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "header": {"resultCode": result_code, "resultMsg": result_msg},
        "body": {"items": items},
    }
    return response


class StoreRegistryCollectorTests(unittest.IsolatedAsyncioTestCase):

    async def test_fetch_sigungu_single_page(self):
        collector = StoreRegistryCollector(api_key="dummy-key")
        items = [
            {"bizesNm": "루트커피", "indsLclsNm": "음식", "rdnmAdr": "서울 마포구 ...", "lon": 126.9, "lat": 37.5},
        ]

        async def fake_get(*_args, **_kwargs):
            return _fake_response(items)

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            async with httpx.AsyncClient() as client:
                df = await collector.fetch_sigungu(client, sigungu_cd="11680")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["bizesNm"], "루트커피")
        self.assertEqual(df.iloc[0]["queried_div_id"], "signguCd")
        self.assertEqual(df.iloc[0]["queried_area_cd"], "11680")

    async def test_fetch_area_paginates_until_short_page(self):
        collector = StoreRegistryCollector(api_key="dummy-key")
        collector.PAGE_SIZE = 2
        pages = [
            [{"bizesNm": "A"}, {"bizesNm": "B"}],  # 꽉 찬 페이지 -> 다음 페이지 요청
            [{"bizesNm": "C"}],                     # PAGE_SIZE보다 작음 -> 종료
        ]

        async def fake_get(*_args, **_kwargs):
            return _fake_response(pages.pop(0))

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            async with httpx.AsyncClient() as client:
                df = await collector.fetch_sigungu(client, sigungu_cd="11680")

        self.assertEqual(len(df), 3)

    async def test_nodata_error_returns_empty_frame_not_exception(self):
        # 실제로 divId=adongCd에 잘못된 자릿수 코드를 넣었을 때 확인된 케이스: resultCode=03.
        # 예외로 죽이지 않고 "이 구역엔 매물 없음"으로 취급해 빈 DataFrame을 반환해야 한다.
        collector = StoreRegistryCollector(api_key="dummy-key")

        async def fake_get(*_args, **_kwargs):
            return _fake_response([], result_code="03", result_msg="NODATA_ERROR")

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            async with httpx.AsyncClient() as client:
                df = await collector.fetch_dong(client, adong_cd="99999999")

        self.assertTrue(df.empty)

    async def test_other_error_code_raises(self):
        collector = StoreRegistryCollector(api_key="bad-key")

        async def fake_get(*_args, **_kwargs):
            return _fake_response([], result_code="01", result_msg="SERVICE_KEY_IS_NOT_REGISTERED_ERROR")

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            async with httpx.AsyncClient() as client:
                with self.assertRaises(RuntimeError):
                    await collector.fetch_sigungu(client, sigungu_cd="11680")

    async def test_fetch_sigungus_defaults_to_all_seoul_gu(self):
        collector = StoreRegistryCollector(api_key="dummy-key")
        seen_codes = []

        async def fake_get(_self_or_url, *_args, params=None, **_kwargs):
            seen_codes.append(params["key"])
            return _fake_response([{"bizesNm": f"store-{params['key']}"}])

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await collector.fetch_sigungus()

        self.assertEqual(len(seen_codes), len(SEOUL_SIGUNGU_CODES))
        self.assertIn("11680", seen_codes)  # 강남구, 실제 검증 호출에서 정상 확인된 코드
        self.assertEqual(len(df), len(SEOUL_SIGUNGU_CODES))

    async def test_fetch_sigungus_uses_local_csv_and_skips_api_entirely_when_env_var_set(self):
        # STORE_REGISTRY_LOCAL_CSV가 설정돼 있으면 apis.data.go.kr가 정상이든 아니든 아예 호출
        # 하지 않는다 — 일부 구만 실패하는 부분 장애 상황에서도 매번 안정적으로 CSV 결과만 쓴다.
        # httpx.AsyncClient.get이 호출되면 바로 실패하도록 둬서 "API를 안 불렀다"를 확인한다.
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("상호명,도로명주소,경도,위도\n루트커피,서울 마포구 성지길 5,127.0,37.5\n")
            csv_path = f.name

        async def fail_if_called(*_args, **_kwargs):
            raise AssertionError("STORE_REGISTRY_LOCAL_CSV가 있으면 실제 API를 호출하면 안 된다")

        try:
            collector = StoreRegistryCollector(api_key="dummy-key")
            with patch.dict("os.environ", {"STORE_REGISTRY_LOCAL_CSV": csv_path}), \
                 patch.object(httpx.AsyncClient, "get", new=fail_if_called):
                df = await collector.fetch_sigungus()
        finally:
            import os as _os
            _os.remove(csv_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["bizesNm"], "루트커피")
        self.assertEqual(df.iloc[0]["rdnmAdr"], "서울 마포구 성지길 5")
        self.assertAlmostEqual(df.iloc[0]["lon"], 127.0)

    async def test_fetch_sigungus_returns_empty_when_api_fails_and_no_local_csv_configured(self):
        # STORE_REGISTRY_LOCAL_CSV가 아예 없으면(운영 환경 기본값) API 전체 실패 시 그냥 빈
        # DataFrame을 반환한다 — 기존 동작 그대로 유지.
        async def fake_get_all_timeout(*_args, **_kwargs):
            raise httpx.TimeoutException("timed out")

        collector = StoreRegistryCollector(api_key="dummy-key")
        with patch.dict("os.environ", {}, clear=False), \
             patch.object(httpx.AsyncClient, "get", new=fake_get_all_timeout):
            import os as _os
            _os.environ.pop("STORE_REGISTRY_LOCAL_CSV", None)
            df = await collector.fetch_sigungus()

        self.assertTrue(df.empty)

    async def test_fetch_dongs_skips_failed_dong_and_continues(self):
        collector = StoreRegistryCollector(api_key="dummy-key")

        async def fake_get(*_args, params=None, **_kwargs):
            if params["key"] == "bad-dong":
                raise httpx.HTTPError("boom")
            return _fake_response([{"bizesNm": f"store-{params['key']}"}])

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await collector.fetch_dongs(["good-dong-1", "bad-dong", "good-dong-2"])

        self.assertEqual(len(df), 2)
        self.assertListEqual(sorted(df["bizesNm"].tolist()), ["store-good-dong-1", "store-good-dong-2"])

    def test_extract_items_handles_item_wrapped_dict(self):
        payload = {"body": {"items": {"item": [{"bizesNm": "X"}]}}}
        items = StoreRegistryCollector._extract_items(payload)
        self.assertEqual(items, [{"bizesNm": "X"}])

    def test_extract_items_handles_flat_list(self):
        payload = {"body": {"items": [{"bizesNm": "Y"}]}}
        self.assertEqual(StoreRegistryCollector._extract_items(payload), [{"bizesNm": "Y"}])

    def test_extract_items_handles_missing_body(self):
        self.assertEqual(StoreRegistryCollector._extract_items({}), [])


class SeoulSigunguCodesTests(unittest.TestCase):

    def test_has_25_seoul_districts(self):
        self.assertEqual(len(SEOUL_SIGUNGU_CODES), 25)

    def test_gangnam_code_matches_verified_call(self):
        # 이번 검증 호출에서 실제로 resultCode=00을 받은 값.
        self.assertEqual(SEOUL_SIGUNGU_CODES["강남구"], "11680")


class LoadLocalRegistryCsvTests(unittest.TestCase):

    def _write_csv(self, text: str) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write(text)
            return f.name

    def test_renames_korean_headers_to_api_field_names(self):
        import os as _os

        path = self._write_csv(
            "상호명,상권업종대분류명,상권업종중분류명,시군구명,도로명주소,경도,위도\n"
            "루트커피,음식,커피점/카페,마포구,서울 마포구 성지길 5,127.0,37.5\n"
        )
        try:
            df = load_local_registry_csv(path)
        finally:
            _os.remove(path)

        self.assertListEqual(
            sorted(df.columns.tolist()),
            sorted(["bizesNm", "indsLclsNm", "indsMclsNm", "signguNm", "rdnmAdr", "lon", "lat"]),
        )
        self.assertEqual(df.iloc[0]["bizesNm"], "루트커피")
        self.assertEqual(df.iloc[0]["indsMclsNm"], "커피점/카페")

    def test_lon_lat_are_numeric(self):
        import os as _os

        path = self._write_csv("상호명,경도,위도\n루트커피,127.05,37.55\n")
        try:
            df = load_local_registry_csv(path)
        finally:
            _os.remove(path)

        self.assertTrue(pd.api.types.is_float_dtype(df["lon"]))
        self.assertTrue(pd.api.types.is_float_dtype(df["lat"]))

    def test_drops_columns_not_in_api_schema(self):
        # 지번코드/우편번호 등 이 패키지 어디서도 안 쓰는 CSV 전용 컬럼은 버려진다.
        import os as _os

        path = self._write_csv("상호명,구우편번호,신우편번호\n루트커피,12345,06789\n")
        try:
            df = load_local_registry_csv(path)
        finally:
            _os.remove(path)

        self.assertListEqual(df.columns.tolist(), ["bizesNm"])


class LoadDongCodesTests(unittest.TestCase):

    def test_autodetects_code_column_by_korean_name(self):
        import io
        csv_text = "행정동명,행정동코드\n역삼동,11680620\n삼성동,11680650\n"
        codes = load_dong_codes(io.StringIO(csv_text))
        self.assertEqual(codes, ["11680620", "11680650"])

    def test_explicit_code_column(self):
        import io
        csv_text = "name,code\n역삼동,11680620\n"
        codes = load_dong_codes(io.StringIO(csv_text), code_column="code")
        self.assertEqual(codes, ["11680620"])


if __name__ == "__main__":
    unittest.main()
