# 대상 경로(AI 레포): app/sales_target/backend_client.py
#
# BE(Spring Boot)와의 왕복 통신 클라이언트 2개.
#   BackendStoreRegistryClient — GET /api/internal/stores/registry (자사 가맹점 목록 조회)
#   SalesTargetIngestClient    — POST /api/internal/sales-targets/bulk (배치 결과 반영)
#                                 GET  /api/internal/sales-targets/excluded (5단계: 피드백 루프,
#                                 영업팀이 EXCLUDED 처리한 후보 주소 조회)

from __future__ import annotations

import httpx
import pandas as pd


class BackendStoreRegistryClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def fetch_our_stores(self) -> pd.DataFrame:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{self.base_url}/api/internal/stores/registry",
                headers={"X-Internal-Api-Key": self.api_key},
                timeout=30.0,
            )
        return self._parse(res)

    def _parse(self, res: httpx.Response) -> pd.DataFrame:
        res.raise_for_status()
        payload = res.json()
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        return pd.DataFrame(rows or [])


class SalesTargetIngestClient:
    """BE의 InternalSalesTargetIngestController(POST /api/internal/sales-targets/bulk)를 호출한다.
    같은 X-Internal-Api-Key 인증을 쓰므로 BackendStoreRegistryClient와 API 키/베이스 URL을
    공유해도 된다(같은 서버, 같은 필터가 보호하는 경로)."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def push_bulk(self, source_batch_id: str, items: list[dict]) -> dict:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{self.base_url}/api/internal/sales-targets/bulk",
                headers={"X-Internal-Api-Key": self.api_key},
                json={"sourceBatchId": source_batch_id, "items": items},
                timeout=60.0,
            )
        res.raise_for_status()
        payload = res.json()
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    async def fetch_excluded_addresses(self) -> list[str]:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{self.base_url}/api/internal/sales-targets/excluded",
                headers={"X-Internal-Api-Key": self.api_key},
                timeout=30.0,
            )
        res.raise_for_status()
        payload = res.json()
        return payload.get("data", payload) if isinstance(payload, dict) else payload
