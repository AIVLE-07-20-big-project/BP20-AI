"""POST /api/v1/reports — Spring Boot가 신규 매출 원본 파일을 업로드하면
매출 진단까지 한 번의 동기 호출로 처리해 JSON으로 돌려준다.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.report import ReportResponse
from app.services import ingestion, pipeline

router = APIRouter()


@router.post("/reports", response_model=ReportResponse)
async def create_report(
    file: UploadFile = File(..., description="신규 매출 원본 데이터(csv, sales_estimate.csv와 동일 스키마)"),
    trdar_cd: str = Form(...),
    svc_induty_cd: str = Form(...),
    yyqu_cd: Optional[int] = Form(None),
) -> dict:
    raw_bytes = await file.read()

    try:
        new_rows = ingestion.read_upload(raw_bytes)
    except ingestion.IngestionSchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    combined_df, ingestion_warnings = ingestion.build_combined_panel(new_rows)

    try:
        report, _raw_diag, pipeline_warnings = pipeline.run_pipeline(
            trdar_cd, svc_induty_cd, yyqu_cd, combined_df,
        )
    except pipeline.CellNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    report["경고"] = ingestion_warnings + pipeline_warnings
    return report
