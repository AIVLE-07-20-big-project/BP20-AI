from typing import Any, Dict, List

async def fetch_store_context(store_id: int) -> Dict[str, Any]:
    """DB에서 매장 정보 조회"""
    # TODO: DB에서 실제로 데이터를 받아올 수 있도록 연동
    return {
        "store_id": store_id,
        "name": "랑트 연남",
        "category": "양식",
        "address": "서울특별시 마포구",
    }

async def fetch_existing_keywords(store_id: int) -> List[str]:
    """기존 매장에 등록된 이미 분류된 표준 키워드 목록 조회"""
    return ["대기시간이 길어요", "직원 불친절/응대 미흡", "음식 간이 셈/짜다"]