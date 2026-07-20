"""문제유형(등급) → 실행 가능 대응방안 후보 규칙.

Diagnoser(scripts/modeling/sales_analysis.py의 _prescribe())가 내는
`5_처방.등급` 값을 입력으로 받아, Neural Contextual Bandit(계획 §5.1)의 arm 후보와
RAG(rag/retriever.py)의 axis 필터를 함께 반환한다. 계획 문서의 "문제유형"은 이 `등급`
필드를 가리킨다.

등급별 후보 설계 근거(2026-07-18 확정, rag/HANDOFF.md §7.1 예시를 기반으로 함):
- `고객_회복`(수요이탈 — 점포수는 유지되고 손님만 감소): 즉각적 수요 자극(할인·세트메뉴)과
  배달 채널 확대로 이탈한 손님을 붙잡는 방안이 진단 문구("손님 자체가 줄었을 가능성")와 맞는다.
- `차별화`(경쟁심화 — 점포 증가 + 점포당매출 감소): 매장·메뉴 차별화와 배달 채널 확대로
  경쟁 상황에서 자기 매장을 구분 짓는 방안이 맞는다. 배달채널 확대는 고객_회복과 차별화
  둘 다에 걸친다 — 이탈 고객을 되찾는 수단이면서 동시에 경쟁 매장 대비 채널 차별화
  수단이기도 하기 때문(사용자 결정).
- `구조_전환`(시장축소·복합침체): 후보 없음. 진단 자체가 "프로모션만으로는 해결되지 않을
  가능성"이라고 명시해 방안을 추천하면 진단과 모순된다.

2026-07-18 추가 — 성장 방안(`customer_acquisition` 축, 2차 결정): 사용자가 "매출 상승이
목적이면 문제가 없어도 고객을 더 유치할 방법이 필요하다"고 지적해, `관찰`·`강점_확대`에도
후보를 추가했다. 기존 방안(즉시할인 등)은 "이탈 회복" 목적이라 "신규 유치" 목적과 근거
문헌이 다르므로 별도 축으로 분리했다(사용자 결정). 이 프로젝트의 대상은 개별 소상공인이
아니라 프랜차이즈 등 기업 — 상권×업종 단위로 집계된 진단([[response_recommendation_agent_replan]]
참고)에도 "본사가 여러 지점에 캠페인을 돌린다"는 프레임이 더 맞는다(rag/generator.py의
SYSTEM_PROMPT도 이에 맞춰 수정함).
- `관찰`(하락 신호는 있으나 확정적이지 않음): 웰컴 프로모션·리뷰 관리 캠페인 — 저비용으로
  먼저 신호를 검증하는 성격의 캠페인.
- `강점_확대`(이미 강점이 뚜렷함): 브랜드 SNS 캠페인·지역 제휴 마케팅 — 이미 있는 강점을
  적극적으로 확장하는 성격의 캠페인.
- `customer_acquisition` 축은 아직 RAG 코퍼스에 문헌이 없다(rag/HANDOFF.md의 6종 코퍼스는
  전부 "이탈 회복" 관점). 문헌이 없으면 `build_evidence()`가 빈 결과를 반환해 리포트에
  "근거없음"으로 정직하게 표시된다 — 숨기지 않고 라벨만 낮춘다(계획 §3).
"""
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

# rag/retriever.py의 build_evidence(axis=...) 호출에 그대로 쓰는 매핑
ACTION_TO_AXIS: dict[str, str] = {name: v["axis"] for name, v in ACTIONS.items()}

ACTION_RULES: dict[str, list[str]] = {
    "고객_회복": ["즉시할인", "쿠폰발행", "타임세일", "세트메뉴 도입", "사이드메뉴 추가", "배달채널 확대"],
    "차별화": ["매장 리뉴얼", "신메뉴 출시", "배달채널 확대"],
    "구조_전환": [],
    "강점_확대": ["브랜드 SNS 캠페인", "지역 제휴 마케팅"],
    "관찰": ["웰컴 프로모션", "리뷰 관리 캠페인"],
}


def candidate_actions(등급: str) -> list[dict]:
    """등급(`5_처방.등급`) → Bandit arm 후보 목록.

    미지의 등급 값이 들어오면 조용히 빈 목록을 반환한다 — 억지로 방안을 만들지 않는다.
    """
    names = ACTION_RULES.get(등급, [])
    return [{"방안": name, "axis": ACTIONS[name]["axis"]} for name in names]
