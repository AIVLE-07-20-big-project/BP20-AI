from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    detail: Any
    error: ErrorDetail


STATUS_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_SERVER_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}

OPENAI_QUOTA_ERROR_CODES = {
    "credit_balance_exhausted",
    "insufficient_quota",
    "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "project_spend_limit_exceeded",
}


def api_error(status_code: int, code: str, message: str, details: Any | None = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": details},
    )


def _payload(status_code: int, detail: Any) -> dict[str, Any]:
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        code = str(detail["code"])
        message = str(detail["message"])
        details = detail.get("details")
    else:
        code = STATUS_ERROR_CODES.get(status_code, "HTTP_ERROR")
        message = str(detail)
        details = None
    return {
        "detail": message,
        "error": {"code": code, "message": message, "details": details},
    }


def _openai_error_code(exc: OpenAIError) -> str | None:
    """OpenAI 응답의 provider error code만 추출한다.

    SDK 버전에 따라 body가 error 객체 자체이거나 ``{"error": ...}`` 형태일 수
    있으므로 두 구조를 모두 지원한다. 원문 message는 외부 응답에 노출하지 않는다.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    nested_error = body.get("error")
    if isinstance(nested_error, dict):
        body = nested_error
    code = body.get("code")
    return str(code) if code else None


def _classify_openai_error(exc: OpenAIError) -> tuple[int, str, str]:
    provider_code = _openai_error_code(exc)

    if provider_code == "model_not_found":
        return 503, "OPENAI_MODEL_UNAVAILABLE", "현재 AI 이미지 생성 모델을 사용할 수 없습니다."
    if provider_code in OPENAI_QUOTA_ERROR_CODES:
        return 503, "OPENAI_QUOTA_EXCEEDED", "AI 서비스 사용 한도가 초과되었습니다."
    if isinstance(exc, AuthenticationError):
        return 503, "OPENAI_AUTHENTICATION_FAILED", "AI 서비스 인증 설정을 확인할 수 없습니다."
    if isinstance(exc, PermissionDeniedError):
        return 503, "OPENAI_ACCESS_DENIED", "AI 서비스에 접근할 수 없습니다."
    if isinstance(exc, RateLimitError):
        return 429, "OPENAI_RATE_LIMITED", "AI 요청이 많습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, APITimeoutError):
        return 504, "OPENAI_TIMEOUT", "AI 이미지 생성 시간이 초과되었습니다. 다시 시도해 주세요."
    if isinstance(exc, APIConnectionError):
        return 503, "OPENAI_CONNECTION_FAILED", "AI 서비스에 일시적으로 연결할 수 없습니다."
    if isinstance(exc, (BadRequestError, UnprocessableEntityError)):
        return 422, "OPENAI_REQUEST_REJECTED", "AI가 이미지 생성 요청을 처리할 수 없습니다."
    if isinstance(exc, NotFoundError):
        return 503, "OPENAI_RESOURCE_NOT_FOUND", "AI 이미지 생성 리소스를 사용할 수 없습니다."
    if isinstance(exc, (InternalServerError, APIStatusError)):
        return 502, "OPENAI_UPSTREAM_ERROR", "AI 서비스 처리 중 일시적인 오류가 발생했습니다."
    return 503, "OPENAI_CONFIGURATION_ERROR", "AI 서비스 설정을 확인할 수 없습니다."


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OpenAIError)
    async def openai_exception_handler(_request: Request, exc: OpenAIError) -> JSONResponse:
        status_code, code, message = _classify_openai_error(exc)
        provider_code = _openai_error_code(exc)
        request_id = getattr(exc, "request_id", None)
        logger.warning(
            "OpenAI API request failed: exception=%s status=%s provider_code=%s request_id=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
            provider_code,
            request_id,
        )

        headers = None
        response = getattr(exc, "response", None)
        if status_code == 429 and response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                headers = {"Retry-After": retry_after}

        return JSONResponse(
            status_code=status_code,
            content=_payload(
                status_code,
                {"code": code, "message": message, "details": None},
            ),
            headers=headers,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.status_code, exc.detail),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        details = exc.errors()
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({
                "detail": details,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "요청값 검증에 실패했습니다",
                    "details": details,
                },
            }),
        )
