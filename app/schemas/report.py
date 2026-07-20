"""응답 shape 문서화용 — 내부 필드는 기존 report/recommend dict 구조를 그대로 감싼다."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ReportResponse(BaseModel):
    관측_변화_분석: dict[str, Any]
    AI_분석: dict[str, Any] | None = None
    외부환경_참고: dict[str, Any] | None = None
    경고: list[str] = []

    model_config = {"extra": "allow"}
