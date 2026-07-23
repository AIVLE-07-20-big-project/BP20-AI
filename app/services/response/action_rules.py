# 문제유형(등급) → 실행 가능 대응방안 후보 규칙
from __future__ import annotations

ACTIONS: dict[str, dict] = {
    "즉시할인": {"axis": "discount_coupon"},
    "쿠폰발행": {"axis": "discount_coupon"},
    "타임세일": {"axis": "discount_coupon"},
    "세트메뉴 도입": {"axis": "set_bundle"},
    "사이드메뉴 추가": {"axis": "set_bundle"},
    "배달채널 확대": {"axis": "delivery"},
    "매장 리뉴얼": {"axis": "store_menu_location"},
    "신메뉴 출시": {"axis": "store_menu_location"},
    "웰컴 프로모션": {"axis": "customer_acquisition"},
    "리뷰 관리 캠페인": {"axis": "customer_acquisition"},
    "브랜드 SNS 캠페인": {"axis": "customer_acquisition"},
    "지역 제휴 마케팅": {"axis": "customer_acquisition"},
}


ACTION_TO_AXIS: dict[str, str] = {name: v["axis"] for name, v in ACTIONS.items()}

ACTION_RULES: dict[str, list[str]] = {
    "고객_회복": ["즉시할인", "쿠폰발행", "타임세일", "세트메뉴 도입", "사이드메뉴 추가", "배달채널 확대"],
    "차별화": ["매장 리뉴얼", "신메뉴 출시", "배달채널 확대"],
    "구조_전환": [],
    "강점_확대": ["브랜드 SNS 캠페인", "지역 제휴 마케팅"],
    "관찰": ["웰컴 프로모션", "리뷰 관리 캠페인"],
}


# 등급(`5_처방.등급`) → Bandit arm 후보 목록
def candidate_actions(등급: str) -> list[dict]:




    names = ACTION_RULES.get(등급, [])
    return [{"방안": name, "axis": ACTIONS[name]["axis"]} for name in names]
