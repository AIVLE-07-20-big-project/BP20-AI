# 대상 경로(AI 레포): tests/test_sales_target_backend_client.py

import unittest
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd

from app.sales_target.backend_client import BackendStoreRegistryClient, SalesTargetIngestClient


def _fake_response(status_code=200, json_body=None, raise_exc=None):
    response = MagicMock()
    response.status_code = status_code
    if raise_exc:
        response.raise_for_status.side_effect = raise_exc
    else:
        response.raise_for_status = MagicMock()
    response.json.return_value = json_body
    return response


class BackendStoreRegistryClientTests(unittest.IsolatedAsyncioTestCase):

    async def test_fetch_our_stores_parses_api_response_envelope(self):
        client = BackendStoreRegistryClient(base_url="http://localhost:8080", api_key="test-key")
        envelope = {
            "status": 200,
            "success": True,
            "message": "가맹점 레지스트리를 조회했습니다.",
            "data": [
                {"businessNumber": "1234567890", "name": "테스트 카페", "category": "카페", "address": "서울 마포구 어딘가 1"},
            ],
        }
        captured = {}

        async def fake_get(_client, url, headers=None, **kwargs):
            # patch.object(AsyncClient, "get", new=fake_get)로 바꾸면 함수가 클래스 속성이 되어
            # client.get(...) 호출 시 client 인스턴스가 첫 인자로 바인딩된다.
            captured["url"] = url
            captured["headers"] = headers
            return _fake_response(json_body=envelope)

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await client.fetch_our_stores()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["businessNumber"], "1234567890")
        self.assertTrue(captured["url"].endswith("/api/internal/stores/registry"))
        self.assertEqual(captured["headers"]["X-Internal-Api-Key"], "test-key")

    async def test_fetch_our_stores_handles_flat_list_without_envelope(self):
        client = BackendStoreRegistryClient(base_url="http://localhost:8080", api_key="test-key")
        flat_body = [{"businessNumber": "111", "name": "A", "category": "카페", "address": "서울"}]

        async def fake_get(*_args, **_kwargs):
            return _fake_response(json_body=flat_body)

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await client.fetch_our_stores()

        self.assertEqual(len(df), 1)

    async def test_fetch_our_stores_empty_data_returns_empty_dataframe(self):
        client = BackendStoreRegistryClient(base_url="http://localhost:8080", api_key="test-key")
        envelope = {"status": 200, "success": True, "data": []}

        async def fake_get(*_args, **_kwargs):
            return _fake_response(json_body=envelope)

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            df = await client.fetch_our_stores()

        self.assertTrue(df.empty)

    async def test_fetch_our_stores_raises_on_http_error(self):
        client = BackendStoreRegistryClient(base_url="http://localhost:8080", api_key="wrong-key")

        async def fake_get(*_args, **_kwargs):
            return _fake_response(
                status_code=401,
                raise_exc=httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock()),
            )

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            with self.assertRaises(httpx.HTTPStatusError):
                await client.fetch_our_stores()

    def test_base_url_trailing_slash_is_stripped(self):
        client = BackendStoreRegistryClient(base_url="http://localhost:8080/", api_key="k")
        self.assertEqual(client.base_url, "http://localhost:8080")


class SalesTargetIngestClientExcludedAddressesTests(unittest.IsolatedAsyncioTestCase):
    """5단계(피드백 루프): GET /api/internal/sales-targets/excluded 클라이언트."""

    async def test_fetch_excluded_addresses_parses_api_response_envelope(self):
        client = SalesTargetIngestClient(base_url="http://localhost:8080", api_key="test-key")
        envelope = {"status": 200, "success": True, "data": ["서울 마포구 1", "서울 강남구 2"]}
        captured = {}

        async def fake_get(_client, url, headers=None, **kwargs):
            captured["url"] = url
            captured["headers"] = headers
            return _fake_response(json_body=envelope)

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            addresses = await client.fetch_excluded_addresses()

        self.assertEqual(addresses, ["서울 마포구 1", "서울 강남구 2"])
        self.assertTrue(captured["url"].endswith("/api/internal/sales-targets/excluded"))
        self.assertEqual(captured["headers"]["X-Internal-Api-Key"], "test-key")

    async def test_fetch_excluded_addresses_empty_data_returns_empty_list(self):
        client = SalesTargetIngestClient(base_url="http://localhost:8080", api_key="test-key")

        async def fake_get(*_args, **_kwargs):
            return _fake_response(json_body={"status": 200, "success": True, "data": []})

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            addresses = await client.fetch_excluded_addresses()

        self.assertEqual(addresses, [])

    async def test_fetch_excluded_addresses_raises_on_http_error(self):
        client = SalesTargetIngestClient(base_url="http://localhost:8080", api_key="wrong-key")

        async def fake_get(*_args, **_kwargs):
            return _fake_response(
                status_code=401,
                raise_exc=httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock()),
            )

        with patch.object(httpx.AsyncClient, "get", new=fake_get):
            with self.assertRaises(httpx.HTTPStatusError):
                await client.fetch_excluded_addresses()


if __name__ == "__main__":
    unittest.main()
