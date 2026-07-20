"""분기 코드와 증감률을 다루는 공용 유틸리티."""
from __future__ import annotations


def prev_quarter_code(yyqu_cd: int) -> int | None:
    """YYYYQ 형식 분기 코드의 직전 분기를 반환한다."""
    year, quarter = divmod(yyqu_cd, 10)
    if quarter == 1:
        return (year - 1) * 10 + 4 if year > 0 else None
    return year * 10 + quarter - 1


def same_quarter_last_year_code(yyqu_cd: int) -> int | None:
    """YYYYQ 형식 분기 코드의 전년 동분기를 반환한다."""
    year, quarter = divmod(yyqu_cd, 10)
    return (year - 1) * 10 + quarter if year > 0 else None


def pct_change(curr: float | int | None, prev: float | int | None) -> float | None:
    """기준값이 없거나 0이면 증감률을 계산하지 않는다."""
    if curr is None or prev is None:
        return None
    try:
        curr_value, prev_value = float(curr), float(prev)
    except (TypeError, ValueError):
        return None
    if prev_value == 0:
        return None
    return (curr_value - prev_value) / prev_value
