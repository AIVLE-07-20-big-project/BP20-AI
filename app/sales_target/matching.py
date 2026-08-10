# 대상 경로(AI 레포): app/sales_target/matching.py
#
# 상가업소 데이터(store_registry_collector.py 결과)에서 "이미 자사 가맹점인 업장"을 걸러낸다.
#
# 왜 주소 매칭이 주력인가: BE Store 엔티티엔 business_number(사업자등록번호)가 있지만,
# 공공데이터 상가업소 응답에는 사업자등록번호 필드가 없다(상가업소 관리번호 bizesId만 있음).
# 그래서 사업자번호로는 매칭할 수 없고, 주소 문자열을 정규화해서 대조하는 방식을 쓴다.
#
# 설계 결정: 애매하면 "제외하지 않는다" 쪽으로 기운다. 정규화된 주소가 완전히 일치하지 않으면
# 그냥 신규 후보로 남겨둔다 — 과제정의서의 관리자 화면(순위/점수/피칭 포인트)에서 사람이 최종
# 검수하는 구조이므로, 이미 가맹점인 곳이 어쩌다 후보에 한 번 더 뜨는 것보다 진짜 신규 후보를
# 놓치는 쪽이 영업팀 입장에서 더 나쁘다.

from __future__ import annotations

import re

import pandas as pd

_PROVINCE_ALIASES = {
    "서울특별시": "서울",
    "서울시": "서울",
}

# 상세 주소(동/층/호 등) 앞까지만 남긴다. 예: "서울 마포구 양화로 1 101동 202호" -> "서울 마포구 양화로 1"
_DETAIL_SUFFIX_PATTERN = re.compile(r"\d+동\b|\d+층\b|\d+호\b")


def normalize_address(address: str | None) -> str:
    if not address:
        return ""
    addr = address.strip()
    for full, short in _PROVINCE_ALIASES.items():
        addr = addr.replace(full, short)
    addr = re.sub(r"\s+", " ", addr).strip()

    match = _DETAIL_SUFFIX_PATTERN.search(addr)
    if match:
        addr = addr[: match.start()].strip()

    return addr


def exclude_existing_merchants(
    candidates: pd.DataFrame,
    our_stores: pd.DataFrame,
    candidate_address_col: str = "rdnmAdr",
    store_address_col: str = "address",
) -> pd.DataFrame:
    """candidates(상가업소)에서 our_stores(자사 가맹점)와 주소가 일치하는 행을 제외한다.

    candidate_address_col 기본값은 store_registry_collector.py 응답의 도로명주소 필드(rdnmAdr).
    지번주소(lnoAdr)만 있는 레코드가 있을 수 있어, 필요하면 호출 쪽에서 두 컬럼을 합쳐 넘겨도 된다.
    """
    if candidates.empty:
        return candidates

    our_normalized = {
        norm
        for norm in (normalize_address(a) for a in our_stores.get(store_address_col, []))
        if norm
    }

    normalized_col = candidates[candidate_address_col].map(normalize_address)
    is_existing = normalized_col.isin(our_normalized) & (normalized_col != "")

    return candidates.loc[~is_existing].reset_index(drop=True)
