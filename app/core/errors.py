from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


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
    500: "INTERNAL_SERVER_ERROR",
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


def register_error_handlers(app: FastAPI) -> None:
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
