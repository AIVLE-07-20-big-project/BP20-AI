from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel


class AgentRunRequest(BaseModel):
    trdar_cd: str
    svc_induty_cd: str
    yyqu_cd: Optional[int] = None


class AgentRunResumeRequest(BaseModel):
    결정: Literal["approve", "edit", "reject"]
    수정_방안: Optional[str] = None


class AgentRunResponse(BaseModel):
    thread_id: str
    상태: str
    대기중_승인: dict[str, Any] | None = None

    model_config = {"extra": "allow"}
