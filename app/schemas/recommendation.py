# Spring Boot가 MySQL에 저장한 분석 결과를 재전달하는 요청 계약
from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class RecommendationFromAnalysisRequest(BaseModel):
    analysis_id: str = Field(validation_alias=AliasChoices("analysisId", "analysis_id"))
    user_id: str | None = Field(
        default=None, validation_alias=AliasChoices("userId", "user_id"),
    )
    store_id: str | None = Field(
        default=None, validation_alias=AliasChoices("storeId", "store_id"),
    )
    trdar_cd: str = Field(validation_alias=AliasChoices("trdarCd", "trdar_cd"))
    svc_induty_cd: str = Field(
        validation_alias=AliasChoices("svcIndutyCd", "svc_induty_cd"),
    )
    yyqu_cd: int | None = Field(
        default=None, validation_alias=AliasChoices("yyquCd", "yyqu_cd"),
    )
    diagnosis: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
