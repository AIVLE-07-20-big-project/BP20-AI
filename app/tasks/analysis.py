"""매출 분석 비동기 태스크."""
from __future__ import annotations

from app.celery_app import celery_app


@celery_app.task(name="ping")
def ping() -> str:
    """브로커와 워커 연결을 확인한다."""
    return "pong"


@celery_app.task(bind=True, name="analyses.run")
def run_analysis_task(
    self,
    job_id: str,
    trdar_cd: str,
    svc_induty_cd: str,
    yyqu_cd: int | None = None,
    user_id: str | None = None,
    store_id: str | None = None,
) -> dict:
    """업로드 CSV를 분석하고 job_id를 analysis_id로 사용해 저장한다."""
    from app.core import uploads
    from app.services import analyses, detailed_sales, ingestion, jobs, pipeline
    from app.services.pipeline import CellNotFoundError
    from scripts.modeling.detailed_sales_external_analysis import DetailedSalesDataError

    if not jobs.mark_running(job_id):
        return {"skipped": True}

    try:
        raw_bytes = uploads.read_job_upload(job_id)
        detailed_analysis = detailed_sales.analyze_uploaded_sales(raw_bytes, trdar_cd)
        report, raw_diag, warnings = pipeline.run_pipeline(
            trdar_cd, svc_induty_cd, yyqu_cd, ingestion.get_base_merged(),
        )
        analysis = analyses.create_analysis(
            trdar_cd=trdar_cd,
            svc_induty_cd=svc_induty_cd,
            yyqu_cd=yyqu_cd,
            report=report,
            diagnosis=raw_diag,
            detailed_analysis=detailed_analysis,
            warnings=warnings,
            user_id=user_id,
            store_id=store_id,
            analysis_id=job_id,
        )
    except DetailedSalesDataError as exc:
        jobs.mark_failed(job_id, "INVALID_POS_DATA", str(exc))
        raise
    except CellNotFoundError as exc:
        jobs.mark_failed(job_id, "MARKET_CELL_NOT_FOUND", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001 - 그 밖의 오류는 내부 오류로 기록한다.
        jobs.mark_failed(job_id, "INTERNAL_ERROR", str(exc))
        raise

    jobs.mark_completed(job_id, analysis["analysis_id"])
    uploads.delete_job_upload(job_id)
    return {"analysis_id": analysis["analysis_id"]}
