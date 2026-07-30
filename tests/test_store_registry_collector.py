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
