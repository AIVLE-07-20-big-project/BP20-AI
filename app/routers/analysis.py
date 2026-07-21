"""매출 분석을 저장하고, 선택한 분석에만 대응방안 추천을 실행한다."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.report import ReportResponse
from app.routers.agent_runs import start_agent_run
from app.services import analyses, ingestion, pipeline

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
    """CSV를 반영해 매출만 분석하고 후속 추천에 사용할 결과를 저장한다."""
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
    )


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    analysis = analyses.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {analysis_id}")
    return analysis


@router.post("/analyses/{analysis_id}/recommendations")
def create_analysis_recommendation(analysis_id: str) -> dict:
    """저장된 매출 분석 결과로 대응방안 추천·검증 에이전트를 실행한다."""
    analysis = analyses.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"분석 결과를 찾을 수 없음: {analysis_id}")
    return start_agent_run({
        "analysis_id": analysis_id,
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
