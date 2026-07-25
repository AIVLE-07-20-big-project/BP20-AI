"""매출 분석 비동기 태스크 (골격).

docs/speed/celery-async-development-plan.md §5-3단계 대상.
현재는 골격만 제공한다:
  - ping: 브로커·워커 연결을 실제로 검증할 수 있는 헬스 태스크.
  - run_analysis_task: 라우터 create_analysis의 동기 로직을 워커로 옮기기 위한 자리.

라우터 비동기화(라우터에서 .delay() 호출 + job_id 202 반환)는 아직 하지 않았다.
기존 동기 엔드포인트는 그대로 두고, 이 태스크가 워커에서 도는지부터 검증한 뒤 연결한다.
"""
from __future__ import annotations

import base64

from app.celery_app import celery_app


@celery_app.task(name="ping")
def ping() -> str:
    """브로커·워커·결과 백엔드 왕복 연결 확인용 헬스 태스크."""
    return "pong"


@celery_app.task(bind=True, name="analyses.run")
def run_analysis_task(
    self,
    raw_bytes_b64: str,
    trdar_cd: str,
    svc_induty_cd: str,
    yyqu_cd: int | None = None,
    user_id: str | None = None,
    store_id: str | None = None,
) -> dict:
    """업로드 CSV 기반 매출 분석을 워커에서 실행하고 결과를 저장한다.

    라우터 create_analysis(app/routers/analysis.py)의 동기 처리를 그대로 옮긴 골격.

    !! 이 시그니처(base64 원본 전달)는 폐기 예정이다. 라우터와 연결하기 전에 계획 문서
    §3.4에 따라 "공유 저장소 저장 + 경로/key 전달"로 교체한다. 업로드 상한이 25MiB
    (MAX_POS_UPLOAD_BYTES)이고 base64는 크기를 약 33% 늘리므로, 이대로 두면 브로커
    메시지 하나가 최대 약 33MiB가 되어 Redis 메모리·네트워크를 낭비한다.

    !! 반환값도 교체 대상이다. 아래는 create_analysis()의 결과 dict 전체(report +
    diagnosis + detailed_analysis + warnings)를 반환하는데, result backend가 켜져 있어
    같은 내용이 SQLite와 Redis에 이중 저장된다("결과 저장 이중화 금지" 위반, 계획 문서
    §3.7). 참조만 반환하도록 바꾼다:
        analysis = analyses.create_analysis(...)
        return {"analysis_id": analysis["analysis_id"]}
    ignore_result=True는 task_track_started의 STARTED/SUCCESS 조회까지 없애므로 잡 테이블
    상태 전이를 완성한 뒤에 검토한다.

    미구현(라우터 연결 전 필요):
      - job_id 기반 멱등성 — 현재 analyses.create_analysis()는 호출마다 uuid4()로
        analysis_id를 만들어 재실행 시 결과가 중복 생성된다(계획 문서 §3.5).
      - 오류 → error_code 매핑 — DetailedSalesDataError(422)/CellNotFoundError(404)를
        잡지 않아 지금은 원인 구분이 불가능한 FAILURE가 된다(계획 문서 §3.6).
      - 잡 상태 조건부 전이 — 취소된 잡이 completed로 되돌아가지 않도록 결과 저장 직전
        취소 여부를 확인해야 한다(계획 문서 §3.1, §3.9).

    무거운 서비스 모듈은 함수 안에서 import해 워커 기동 시간을 줄인다.
    """
    from app.services import analyses, detailed_sales, ingestion, pipeline

    raw_bytes = base64.b64decode(raw_bytes_b64)

    detailed_analysis = detailed_sales.analyze_uploaded_sales(raw_bytes, trdar_cd)
    report, raw_diag, warnings = pipeline.run_pipeline(
        trdar_cd, svc_induty_cd, yyqu_cd, ingestion.get_base_merged(),
    )
    return analyses.create_analysis(
        trdar_cd=trdar_cd,
        svc_induty_cd=svc_induty_cd,
        yyqu_cd=yyqu_cd,
        report=report,
        diagnosis=raw_diag,
        detailed_analysis=detailed_analysis,
        warnings=warnings,
        user_id=user_id,
        store_id=store_id,
    )
