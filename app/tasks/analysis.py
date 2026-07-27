"""매출 분석 비동기 태스크.

docs/speed/celery-async-development-plan.md의 1·3단계를 반영했다:
  - ping: 브로커·워커 연결을 실제로 검증할 수 있는 헬스 태스크.
  - run_analysis_task: 업로드 원본을 저장소에서 읽어 분석하고 결과 참조만 반환한다.

라우터 비동기화(라우터에서 .delay() 호출 + job_id 202 반환)는 아직 하지 않았다(6단계).
기존 동기 엔드포인트는 그대로 두고, 이 태스크가 워커에서 도는지부터 검증한 뒤 연결한다.

미구현(라우터 연결 전 필요):
  - 오류 → error_code 매핑 — DetailedSalesDataError/CellNotFoundError를 잡지 않아
    지금은 원인 구분이 불가능한 FAILURE가 된다(계획 문서 7단계).
  - 잡 상태 조건부 전이 — 결과 저장 직전 잡 상태를 확인해야 한다(계획 문서 4단계).
"""
from __future__ import annotations

from app.celery_app import celery_app


@celery_app.task(name="ping")
def ping() -> str:
    """브로커·워커·결과 백엔드 왕복 연결 확인용 헬스 태스크."""
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
    """저장된 업로드 CSV로 매출 분석을 실행하고 결과를 저장한다.

    라우터 create_analysis(app/routers/analysis.py)의 동기 처리를 워커로 옮긴 것이다.

    브로커에는 `job_id`만 싣고 원본은 공유 저장소에서 읽는다. 업로드 상한이 25MiB이고
    base64는 크기를 약 33% 늘리므로, 원본을 실으면 메시지 하나가 최대 약 33MiB가 된다.

    반환값은 `analysis_id` 참조뿐이다. 결과 dict 전체를 반환하면 result backend를 통해
    같은 내용이 SQLite와 Redis에 이중 저장된다.

    `job_id`를 `analysis_id`로 그대로 써서 멱등하게 저장한다(계획 문서 §2.4). 같은 잡이
    재배달 등으로 두 번 실행돼도 analyses 테이블엔 행이 하나만 남는다 — 두 번째 호출의
    분석 결과는 버려지고 첫 저장이 이긴다.

    성공한 잡의 업로드 원본은 삭제한다. 실패한 잡의 원본은 재현을 위해 남기고
    uploads.purge_expired_uploads()가 보존 기간 경과 후 회수한다.

    무거운 서비스 모듈은 함수 안에서 import해 워커 기동 시간을 줄인다.
    """
    from app.core import uploads
    from app.services import analyses, detailed_sales, ingestion, pipeline

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

    uploads.delete_job_upload(job_id)
    return {"analysis_id": analysis["analysis_id"]}
