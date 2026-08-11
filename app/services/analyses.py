# 업로드한 매출 분석 결과를 후속 추천 실행에 재사용하기 위한 저장소
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, datetime, timezone
from uuid import uuid4

from fastapi.encoders import jsonable_encoder

from app.core.config import ANALYSES_DB
from app.core.database import connect, execute, fetchall, fetchone, using_mysql


# trdar_cd·svc_induty_cd·기준분기(공공데이터 분기)가 같아도 업로드한 POS 실제 기간은
# 다를 수 있어, "매출 분석 결과 선택" 목록에서 항목을 구분할 사람이 읽는 라벨을 만든다.
# agent_runs.py의 대상_매장 표시명과 같은 로직을 공유한다.
def format_analysis_period(detailed_analysis: dict | None) -> str | None:
    summary = (detailed_analysis or {}).get("dataSummary") or {}
    start, end = summary.get("startDate"), summary.get("endDate")
    if not start or not end:
        return None
    try:
        start_dt, end_dt = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        return None
    if (start_dt.year, start_dt.month) == (end_dt.year, end_dt.month):
        return f"{start_dt.year}년 {start_dt.month}월"
    if start_dt.year == end_dt.year:
        return f"{start_dt.year}년 {start_dt.month}월~{end_dt.month}월"
    return f"{start_dt.year}년 {start_dt.month}월~{end_dt.year}년 {end_dt.month}월"


# POS 실제 기간을 알 수 없을 때(구버전 데이터 등)의 최후 폴백 — 20261 같은 원시
# 코드 대신 "2026년 1분기"로 보여준다.
def format_quarter_label(기준분기: int | None) -> str | None:
    if not 기준분기:
        return None
    year, quarter = divmod(int(기준분기), 10)
    return f"{year}년 {quarter}분기"


@contextmanager
def _connection():
    ANALYSES_DB.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(ANALYSES_DB)) as connection:
        table = "ai_analyses" if using_mysql() else "analyses"
        if not using_mysql():
            execute(connection, "PRAGMA journal_mode=WAL")
            execute(connection, "PRAGMA busy_timeout=10000")
        execute(connection, f"""
            CREATE TABLE IF NOT EXISTS {table} (
                analysis_id VARCHAR(36) PRIMARY KEY,
                trdar_cd VARCHAR(64) NOT NULL,
                svc_induty_cd VARCHAR(64) NOT NULL,
                yyqu_cd INTEGER,
                result_json LONGTEXT NOT NULL,
                created_at DATETIME(6) NOT NULL,
                user_id BIGINT,
                store_id BIGINT,
                updated_at DATETIME(6) NOT NULL
            )
        """)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _dump(value: object) -> str:
    return json.dumps(jsonable_encoder(value), ensure_ascii=False)


def create_analysis(
    *,
    trdar_cd: str,
    svc_induty_cd: str,
    yyqu_cd: int | None,
    report: dict,
    diagnosis: dict,
    warnings: list[str],
    detailed_analysis: dict | None = None,
    user_id: str | None = None,
    store_id: str | None = None,
    analysis_id: str | None = None,
) -> dict:
    """지정한 analysis_id는 첫 결과만 저장하고, 없으면 새 ID를 생성한다."""
    analysis_id = analysis_id or str(uuid4())
    now = datetime.now(timezone.utc)
    created_at = now.replace(tzinfo=None) if using_mysql() else now.isoformat()
    with _connection() as connection:
        table = "ai_analyses" if using_mysql() else "analyses"
        insert = "INSERT IGNORE" if using_mysql() else "INSERT OR IGNORE"
        execute(connection, f"""
            {insert} INTO {table} (
                analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
                result_json, created_at, updated_at, user_id, store_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
            _dump({"report": report, "diagnosis": diagnosis, "warnings": warnings,
                   "detailed_analysis": detailed_analysis}),
            created_at, created_at, int(user_id) if user_id and str(user_id).isdigit() else None,
            int(store_id) if store_id and str(store_id).isdigit() else None,
        ))
    return get_analysis(analysis_id)


def _value(row, key: str):
    return row[key]


def _row_to_dict(row) -> dict:
    payload = json.loads(_value(row, "result_json"))
    detailed_analysis = payload.get("detailed_analysis")
    diagnosis = payload.get("diagnosis", {})
    기준분기 = (diagnosis.get("대상") or {}).get("기준분기")
    return {
        "analysis_id": _value(row, "analysis_id"),
        "user_id": _value(row, "user_id"),
        "store_id": _value(row, "store_id"),
        "trdar_cd": _value(row, "trdar_cd"),
        "svc_induty_cd": _value(row, "svc_induty_cd"),
        "yyqu_cd": _value(row, "yyqu_cd"),
        "report": payload.get("report", {}),
        "diagnosis": diagnosis,
        "detailed_analysis": detailed_analysis,
        "warnings": payload.get("warnings", []),
        "created_at": _value(row, "created_at"),
        # 목록 화면에서 같은 상권·업종의 여러 분석(예: 전략 전/후 비교)을 구분하기 위한
        # 사람이 읽는 기간 라벨 — 실제 POS 기간이 있으면 그걸 쓰고, 없으면 공공데이터
        # 기준분기로 폴백한다.
        "분석기간": format_analysis_period(detailed_analysis) or format_quarter_label(기준분기),
    }


def get_analysis(analysis_id: str) -> dict | None:
    with _connection() as connection:
        table = "ai_analyses" if using_mysql() else "analyses"
        row = fetchone(connection, f"SELECT * FROM {table} WHERE analysis_id = ?", (analysis_id,))
    return _row_to_dict(row) if row is not None else None


def list_analyses(user_id: str, store_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM analyses WHERE user_id = ?"
    table = "ai_analyses" if using_mysql() else "analyses"
    query = f"SELECT * FROM {table} WHERE user_id = ?"
    params: list[object] = [user_id]
    if store_id is not None:
        query += " AND store_id = ?"
        params.append(store_id)
    query += " ORDER BY created_at DESC"
    with _connection() as connection:
        rows = fetchall(connection, query, params)
    return [_row_to_dict(row) for row in rows]
