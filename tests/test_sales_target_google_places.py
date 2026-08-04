# 대상 경로(AI 레포): tests/test_sales_target_google_places.py
#
# Places API (New)의 실제 요청/응답 형태(POST .../places:searchText, GET .../places/{id},
# X-Goog-Api-Key/X-Goog-FieldMask 헤더, reviews[].publishTime RFC3339 문자열)를 기준으로 모킹한다.

import unittest
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd

from app.sales_target.google_places import fetch_review_stats


def _fake_response(json_body):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = json_body
    return response


class FetchReviewStatsTests(unittest.IsolatedAsyncioTestCase):

    async def test_successful_lookup_returns_review_stats(self):
        candidates = pd.DataFrame({
            "bizesNm": ["루트커피"],
            "rdnmAdr": ["서울 마포구 성지길 5"],
        })
        captured = {}

        async def fake_post(_client, url, json=None, headers=None, **kwargs):
            captured["search_headers"] = headers
            captured["search_json"] = json
            return _fake_response({"places": [{"id": "PID1"}]})

        async def fake_get(_client, url, headers=None, **kwargs):
            captured["details_url"] = url
            captured["details_headers"] = headers
            return _fake_response({
                "rating": 4.5,
                "userRatingCount": 120,
                "reviews": [
                    {"publishTime": "2026-01-01T00:00:00Z", "rating": 5},
                    {"publishTime": "2026-01-13T00:00:00Z", "rating": 4},
                    {"publishTime": "2026-01-27T00:00:00Z", "rating": 5},
                ],
            })

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["rdnmAdr"], "서울 마포구 성지길 5")
        self.assertEqual(row["review_count"], 120)
        self.assertEqual(row["avg_rating"], 4.5)
        self.assertIsNotNone(row["days_since_latest_review"])
        self.assertLess(row["review_growth_trend"], 0)  # 간격의 음수 근사치
        self.assertEqual(captured["search_headers"]["X-Goog-Api-Key"], "test-key")
        self.assertEqual(captured["search_json"], {"textQuery": "루트커피 서울 마포구 성지길 5"})
        self.assertTrue(captured["details_url"].endswith("/v1/places/PID1"))

    async def test_place_not_found_is_skipped(self):
        candidates = pd.DataFrame({"bizesNm": ["없는가게"], "rdnmAdr": ["서울 어딘가 1"]})

        async def fake_post(_client, url, **kwargs):
            return _fake_response({"places": []})

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertTrue(result.empty)

    async def test_missing_rating_is_skipped(self):
        candidates = pd.DataFrame({"bizesNm": ["가게"], "rdnmAdr": ["서울 어딘가 1"]})

        async def fake_post(_client, url, **kwargs):
            return _fake_response({"places": [{"id": "PID1"}]})

        async def fake_get(_client, url, **kwargs):
            return _fake_response({})

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertTrue(result.empty)

    async def test_fewer_than_two_reviews_growth_trend_is_none(self):
        candidates = pd.DataFrame({"bizesNm": ["가게"], "rdnmAdr": ["서울 어딘가 1"]})

        async def fake_post(_client, url, **kwargs):
            return _fake_response({"places": [{"id": "PID1"}]})

        async def fake_get(_client, url, **kwargs):
            return _fake_response({"rating": 4.0, "userRatingCount": 5, "reviews": []})

        with patch.object(httpx.AsyncClient, "post", new=fake_post), \
             patch.object(httpx.AsyncClient, "get", new=fake_get):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertEqual(len(result), 1)
        self.assertIsNone(result.iloc[0]["review_growth_trend"])
        self.assertIsNone(result.iloc[0]["days_since_latest_review"])

    async def test_missing_name_or_address_skips_without_api_call(self):
        candidates = pd.DataFrame({"bizesNm": [None], "rdnmAdr": ["서울 어딘가 1"]})
        call_count = {"n": 0}

        async def fake_post(_client, url, **kwargs):
            call_count["n"] += 1
            return _fake_response({"places": []})

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertTrue(result.empty)
        self.assertEqual(call_count["n"], 0)

    async def test_http_error_is_skipped(self):
        candidates = pd.DataFrame({"bizesNm": ["가게"], "rdnmAdr": ["서울 어딘가 1"]})

        async def fake_post(_client, url, **kwargs):
            raise httpx.ConnectError("연결 실패")

        with patch.object(httpx.AsyncClient, "post", new=fake_post):
            result = await fetch_review_stats(candidates, api_key="test-key")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
