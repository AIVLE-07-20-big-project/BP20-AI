"""POST /api/v1/reports — Spring Boot가 신규 매출 원본 파일을 업로드하면
매출 진단까지 한 번의 동기 호출로 처리해 JSON으로 돌려준다.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.report import ReportResponse
from app.routers.agent_runs import continue_agent_run, read_agent_run, start_agent_run
from app.services import ingestion, pipeline

router = APIRouter()


async def _ingest_and_diagnose(
    file: UploadFile, trdar_cd: str, svc_induty_cd: str, yyqu_cd: Optional[int],
) -> tuple[dict, dict, list[str]]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="현재는 CSV 파일만 지원합니다")

    raw_bytes = await file.read()
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


@router.post("/analyses")
async def create_analysis(
    file: UploadFile = File(..., description="분석할 신규 매출 원본 CSV"),
    trdar_cd: str = Form(...),
    svc_induty_cd: str = Form(...),
    yyqu_cd: Optional[int] = Form(None),
) -> dict:
    """파일을 반영해 진단·추천을 실행하고 리포트 생성 승인 전까지 진행한다."""
    _report, raw_diag, warnings = await _ingest_and_diagnose(
        file, trdar_cd, svc_induty_cd, yyqu_cd,
    )
    return start_agent_run({
        "trdar_cd": trdar_cd,
        "svc_induty_cd": svc_induty_cd,
        "yyqu_cd": yyqu_cd,
        "diagnosis": raw_diag,
        "warnings": warnings,
    })


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    return read_agent_run(analysis_id)


@router.post("/analyses/{analysis_id}/reports")
def create_analysis_report(analysis_id: str) -> dict:
    """사용자가 요청한 경우 승인 대기 중인 분석으로 최종 리포트를 생성한다."""
    return continue_agent_run(analysis_id, {"결정": "approve"})


@router.post("/reports", response_model=ReportResponse)
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
