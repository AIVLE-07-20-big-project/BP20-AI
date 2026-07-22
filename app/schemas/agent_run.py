from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field


class AgentRunRequest(BaseModel):
    user_id: Optional[str] = None
    store_id: Optional[str] = None
    trdar_cd: str
    svc_induty_cd: str
    yyqu_cd: Optional[int] = None


class AgentRunResumeRequest(BaseModel):
    decision: Literal["approve", "edit", "reject"] = Field(
        validation_alias=AliasChoices("decision", "결정"),
    )
    modification_plan: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("modificationPlan", "modification_plan", "수정_방안"),
    )


class AgentRunResponse(BaseModel):
    thread_id: str
    상태: str
    대기중_승인: dict[str, Any] | None = None

    model_config = {"extra": "allow"}
