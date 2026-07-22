"""Spring Boot가 MySQL에 저장한 분석 결과를 재전달하는 요청 계약."""
from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class RecommendationFromAnalysisRequest(BaseModel):
    analysis_id: str = Field(validation_alias=AliasChoices("analysis_id", "analysisId"))
    user_id: str | None = Field(
        default=None, validation_alias=AliasChoices("user_id", "userId"),
    )
    store_id: str | None = Field(
        default=None, validation_alias=AliasChoices("store_id", "storeId"),
    )
    trdar_cd: str = Field(validation_alias=AliasChoices("trdar_cd", "trdarCd"))
    svc_induty_cd: str = Field(
        validation_alias=AliasChoices("svc_induty_cd", "svcIndutyCd"),
    )
    yyqu_cd: int | None = Field(
        default=None, validation_alias=AliasChoices("yyqu_cd", "yyquCd"),
    )
    diagnosis: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)

