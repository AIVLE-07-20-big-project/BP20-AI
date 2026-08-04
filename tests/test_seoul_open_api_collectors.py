# 대상 경로(AI 레포): tests/test_seoul_open_api_collectors.py
#
# collectors.py의 SeoulOpenApiCollector.fetch_quarters() 개선사항(배치 속도 개선) 검증:
#   1) 분기별 호출이 동시에 나가는지(병렬화)
#   2) SEOUL_API_QUARTER_CACHE_DIR 설정 시 과거 분기는 캐시에서 읽고, 최신 분기는 항상 API를 다시 부르는지
#   3) all_quarter_codes()의 SALES_TARGET_DEV_QUARTER_LIMIT 단축 옵션
#
# 기존 test_store_registry_collector.py와 동일한 방식(unittest.IsolatedAsyncioTestCase,
# httpx.AsyncClient.get 모킹)을 쓴다.

import asyncio
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx

from scripts.collection import collectors as collectors_module
from scripts.collection.collectors import SalesEstimateCollector, all_quarter_codes


def _fake_response(rows):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {"VwsmTrdarSelngQq": {"row": rows}}
    return response


class FetchQuartersConcurrencyTests(unittest.IsolatedAsyncioTestCase):

    async def test_fetches_all_quarters_concurrently_not_sequentially(self):
        collector = SalesEstimateCollector(api_key="dummy-key")
        in_flight = 0
        max_in_flight = 0

        async def fake_get(*_args, **_kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)  # 실제 네트워크 대기를 흉내내서 겹치는지 확인
            in_flight -= 1
            return _fake_response([{"TRDAR_CD": "1"}])

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await collector.fetch_quarters(["20241", "20242", "20243"])

        self.assertEqual(len(df), 3)
        self.assertGreater(max_in_flight, 1, "분기 호출이 겹치지 않음 — 여전히 순차 호출 상태")

    async def test_respects_max_concurrent_quarters_limit(self):
        collector = SalesEstimateCollector(api_key="dummy-key")
        collector.MAX_CONCURRENT_QUARTERS = 2
        in_flight = 0
        max_in_flight = 0

        async def fake_get(*_args, **_kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _fake_response([{"TRDAR_CD": "1"}])

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            await collector.fetch_quarters(["20241", "20242", "20243", "20244", "20245"])

        self.assertLessEqual(max_in_flight, 2, "세마포어 상한을 넘겨서 동시 호출됨")


class FetchQuartersCacheTests(unittest.IsolatedAsyncioTestCase):

    async def test_past_quarter_is_cached_and_skips_api_on_second_call(self):
        call_count = 0

        async def fake_get(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _fake_response([{"TRDAR_CD": "1"}])

        with tempfile.TemporaryDirectory() as cache_dir:
            # 모듈을 reload하면 다른 테스트 파일이 이미 들고 있는 참조(sys.modules)까지 오염되므로,
            # 모듈 전역 변수 하나만 patch.object로 바꿔치기한다 — with 블록을 벗어나면 자동 원복.
            with patch.object(collectors_module, "_QUARTER_CACHE_DIR", cache_dir):
                collector = collectors_module.SalesEstimateCollector(api_key="dummy-key")
                with patch.object(httpx.AsyncClient, "get", new=fake_get):
                    await collector.fetch_quarters(["20241", "20242"])  # 20242가 최신
                    self.assertEqual(call_count, 2)
                    await collector.fetch_quarters(["20241", "20242"])  # 20241만 캐시 히트
                    self.assertEqual(call_count, 3, "과거 분기(20241)가 캐시를 안 타고 다시 API를 불렀음")


class AllQuarterCodesDevLimitTests(unittest.TestCase):

    def test_default_returns_full_range(self):
        codes = all_quarter_codes(start_year=2021, end_year=2026, end_quarter=1)
        self.assertEqual(len(codes), 21)

    def test_dev_limit_env_var_keeps_only_latest_n_quarters(self):
        full = all_quarter_codes(start_year=2021, end_year=2026, end_quarter=1)
        with patch.dict(os.environ, {"SALES_TARGET_DEV_QUARTER_LIMIT": "4"}):
            limited = all_quarter_codes(start_year=2021, end_year=2026, end_quarter=1)
        self.assertEqual(limited, full[-4:])
        self.assertEqual(len(limited), 4)


class LatestQuarterCodesTests(unittest.TestCase):

    def test_defaults_to_last_4_quarters(self):
        from scripts.collection.collectors import latest_quarter_codes

        codes = latest_quarter_codes(end_year=2026, end_quarter=1)
        self.assertEqual(len(codes), 4)
        self.assertEqual(codes, all_quarter_codes(start_year=2021, end_year=2026, end_quarter=1)[-4:])

    def test_lookback_is_configurable(self):
        from scripts.collection.collectors import latest_quarter_codes

        codes = latest_quarter_codes(end_year=2026, end_quarter=1, lookback=2)
        self.assertEqual(codes, ["20254", "20261"])

    def test_covers_the_two_quarters_growth_calc_actually_uses(self):
        # district_metrics.latest_vs_previous_growth()가 실제로 쓰는 건 마지막 2개뿐이라,
        # lookback=4의 결과에도 그 2개(직전 분기, 최신 분기)가 꼬리에 그대로 남아있어야 한다.
        from scripts.collection.collectors import latest_quarter_codes

        codes = latest_quarter_codes(end_year=2026, end_quarter=1)
        self.assertEqual(codes[-2:], ["20254", "20261"])


if __name__ == "__main__":
    unittest.main()
