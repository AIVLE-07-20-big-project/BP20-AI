# 매출 분석을 저장하고, 선택한 분석에만 대응방안 추천을 실행한다
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from app.schemas.report import ReportResponse
from app.schemas.recommendation import RecommendationFromAnalysisRequest
from app.routers.agent_runs import start_agent_run
from app.core.uploads import (
    CSV_CONTENT_TYPES,
    CSV_EXTENSIONS,
    MAX_CSV_UPLOAD_BYTES,
    read_upload_limited,
    validate_upload_type,
)
from app.services import analyses, ingestion, pipeline

router = APIRouter(tags=["매출 분석"])


async def _ingest_and_diagnose(
    file: UploadFile, trdar_cd: str, svc_induty_cd: str, yyqu_cd: Optional[int],
) -> tuple[dict, dict, list[str]]:
    validate_upload_type(
        file,
        extensions=CSV_EXTENSIONS,
        content_types=CSV_CONTENT_TYPES,
        type_name="CSV",
    )
    raw_bytes = await read_upload_limited(file, MAX_CSV_UPLOAD_BYTES)
    try:
        new_rows = ingestion.read_upload(raw_bytes)
    except ingestion.IngestionSchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    combined_df, ingestion_warnings = ingestion.build_combined_panel(new_rows)
    try:
        report, raw_diag, pipeline_warnings = pipeline.run_pipeline(
            trdar_cd, svc_induty_cd, yyqu_cd, combined_df,
        )
    except pipeline.CellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return report, raw_diag, ingestion_warnings + pipeline_warnings


# CSV를 반영해 매출만 분석하고 후속 추천에 사용할 결과를 저장한다
@router.post("/analyses")
async def create_analysis(
    file: UploadFile = File(..., description="분석할 신규 매출 원본 CSV"),
    trdar_cd: str = Form(...),
    svc_induty_cd: str = Form(...),
    yyqu_cd: Optional[int] = Form(None),
    user_id: Optional[str] = Form(None),
    store_id: Optional[str] = Form(None),
) -> dict:

    report, raw_diag, warnings = await _ingest_and_diagnose(
        file, trdar_cd, svc_induty_cd, yyqu_cd,
    )
    return analyses.create_analysis(
        trdar_cd=trdar_cd,
        svc_induty_cd=svc_induty_cd,
        yyqu_cd=yyqu_cd,
        report=report,
        diagnosis=raw_diag,
        warnings=warnings,
        user_id=user_id,
        store_id=store_id,
    )


def _assert_owner(analysis: dict, user_id: str | None) -> None:
    owner = analysis.get("user_id")
    if owner is not None and owner != user_id:
        raise HTTPException(status_code=403, detail="해당 분석 결과에 접근할 권한이 없습니다")


@router.get("/analyses")
def list_user_analyses(
    x_user_id: str = Header(..., alias="X-User-Id"),
    store_id: Optional[str] = None,
) -> list[dict]:
    return analyses.list_analyses(x_user_id, store_id)


# Spring Boot/MySQL에서 재전달한 분석 결과로 추천 에이전트를 시작한다
@router.post("/recommendations", tags=["전략 추천"])
def create_recommendation_from_analysis(
    payload: RecommendationFromAnalysisRequest,
    x_user_id: str = Header(..., alias="X-User-Id"),
) -> dict:

    if payload.user_id is not None and payload.user_id != x_user_id:
        raise HTTPException(status_code=403, detail="요청 사용자와 분석 결과의 소유자가 다릅니다")
    return start_agent_run({
        "analysis_id": payload.analysis_id,
        "user_id": x_user_id,
        "store_id": payload.store_id,
        "trdar_cd": payload.trdar_cd,
        "svc_induty_cd": payload.svc_induty_cd,
        "yyqu_cd": payload.yyqu_cd,
        "diagnosis": payload.diagnosis,
        "warnings": payload.warnings,
    })


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> dict:
    analysis = analyses.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {analysis_id}")
    _assert_owner(analysis, x_user_id)
    return analysis


# 저장된 매출 분석 결과로 대응방안 추천·검증 에이전트를 실행한다
@router.post("/analyses/{analysis_id}/recommendations", tags=["전략 추천"])
def create_analysis_recommendation(
    analysis_id: str, x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
) -> dict:

    analysis = analyses.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {analysis_id}")
    _assert_owner(analysis, x_user_id)
    return start_agent_run({
        "analysis_id": analysis_id,
        "user_id": analysis.get("user_id"),
        "store_id": analysis.get("store_id"),
        "trdar_cd": analysis["trdar_cd"],
        "svc_induty_cd": analysis["svc_induty_cd"],
        "yyqu_cd": analysis["yyqu_cd"],
        "diagnosis": analysis["diagnosis"],
        "warnings": analysis["warnings"],
    })


@router.post("/reports", response_model=ReportResponse, deprecated=True)
async def create_report(
    file: UploadFile = File(..., description="신규 매출 원본 데이터(csv, sales_estimate.csv와 동일 스키마)"),
    trdar_cd: str = Form(...),
    svc_induty_cd: str = Form(...),
    yyqu_cd: Optional[int] = Form(None),
) -> dict:
    report, _raw_diag, warnings = await _ingest_and_diagnose(
        file, trdar_cd, svc_induty_cd, yyqu_cd,
    )
    report["경고"] = warnings
    return report
