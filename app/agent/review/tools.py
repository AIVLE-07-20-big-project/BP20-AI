import os
from typing import Any, Dict, Optional
from langchain_core.tools import tool
import httpx
from app.agent.review.internal_client import get_internal_headers

SPRINGBOOT_BASE_URL = os.getenv("SPRINGBOOT_BASE_URL", "http://springboot:8080")

@tool
async def fetch_store_context(
    store_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    [LLM Tool] 매장의 기본 맥락 정보(상호명, 업종/카테고리, 주소 등)를 조회합니다.

    ** LLM 도구 사용 가이드라인 (When to use this tool):
    - 리뷰 목록 전체를 확인했으나 맥락이 모호하여 "이 가게가 정확히 어떤 음식점/업종인가?" 판단하기 어려운 경우에만 호출하세요.
    - 리뷰 내용만으로도 매장 특성과 분석 목표가 명확하다면 이 도구를 호출하지 마세요.

    Args:
        store_id (int): 조회할 매장 ID

    Returns:
        Dict[str, Any]: 매장의 상호명, 카테고리, 주소 정보
    """
    print("LLM이 가게 정보 조회를 요청하였습니다.")
    headers = get_internal_headers()
    url = f"{SPRINGBOOT_BASE_URL}/api/internal/stores/{store_id}/context"


    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            response.raise_for_status()

            res_body = response.json()
            store_data = res_body.get("data", res_body)
            print(store_data)

            return {
                "store_id": store_data.get("id", store_id),
                "name": store_data.get("name", "알 수 없음"),
                "category": store_data.get("category", "일반 매장"),
                "address": store_data.get("address", ""),
            }
        except Exception as e:
            print(f"[fetch_store_context] Spring Boot API 호출 실패: {e}")
            return {
                "store_id": store_id,
                "name": "알 수 없음",
                "category": "일반 매장",
                "address": "",
            }