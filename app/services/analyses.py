# 업로드한 매출 분석 결과를 후속 추천 실행에 재사용하기 위한 저장소
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.encoders import jsonable_encoder

from app.core.config import ANALYSES_DB
from app.core.database import connect, execute, fetchall, fetchone, using_mysql


@contextmanager
def _connection():
    ANALYSES_DB.parent.mkdir(parents=True, exist_ok=True)
    with connect(str(ANALYSES_DB)) as connection:
        table = "ai_internal_analyses" if using_mysql() else "analyses"
        if not using_mysql():
            execute(connection, "PRAGMA journal_mode=WAL")
            execute(connection, "PRAGMA busy_timeout=10000")
        execute(connection, f"""
            CREATE TABLE IF NOT EXISTS {table} (
                analysis_id VARCHAR(64) PRIMARY KEY,
                trdar_cd VARCHAR(64) NOT NULL,
                svc_induty_cd VARCHAR(64) NOT NULL,
                yyqu_cd INTEGER,
                report_json LONGTEXT NOT NULL,
                diagnosis_json LONGTEXT NOT NULL,
                warnings_json LONGTEXT NOT NULL,
                created_at VARCHAR(40) NOT NULL,
                user_id VARCHAR(255),
                store_id VARCHAR(255),
                detailed_analysis_json LONGTEXT
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
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        table = "ai_internal_analyses" if using_mysql() else "analyses"
        insert = "INSERT IGNORE" if using_mysql() else "INSERT OR IGNORE"
        execute(connection, f"""
            {insert} INTO {table} (
                analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
                report_json, diagnosis_json, detailed_analysis_json,
                warnings_json, created_at, user_id, store_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
            _dump(report), _dump(diagnosis), _dump(detailed_analysis),
            _dump(warnings), created_at, user_id, store_id,
        ))
    return get_analysis(analysis_id)


def _value(row, key: str):
    return row[key]


def _row_to_dict(row) -> dict:
    detailed = _value(row, "detailed_analysis_json")
    return {
        "analysis_id": _value(row, "analysis_id"),
        "user_id": _value(row, "user_id"),
        "store_id": _value(row, "store_id"),
        "trdar_cd": _value(row, "trdar_cd"),
        "svc_induty_cd": _value(row, "svc_induty_cd"),
        "yyqu_cd": _value(row, "yyqu_cd"),
        "report": json.loads(_value(row, "report_json")),
        "diagnosis": json.loads(_value(row, "diagnosis_json")),
        "detailed_analysis": json.loads(detailed) if detailed is not None else None,
        "warnings": json.loads(_value(row, "warnings_json")),
        "created_at": _value(row, "created_at"),
    }


def get_analysis(analysis_id: str) -> dict | None:
    with _connection() as connection:
        table = "ai_internal_analyses" if using_mysql() else "analyses"
        row = fetchone(connection, f"SELECT * FROM {table} WHERE analysis_id = ?", (analysis_id,))
    return _row_to_dict(row) if row is not None else None


def list_analyses(user_id: str, store_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM analyses WHERE user_id = ?"
    table = "ai_internal_analyses" if using_mysql() else "analyses"
    query = f"SELECT * FROM {table} WHERE user_id = ?"
    params: list[object] = [user_id]
    if store_id is not None:
        query += " AND store_id = ?"
        params.append(store_id)
    query += " ORDER BY created_at DESC"
    with _connection() as connection:
        rows = fetchall(connection, query, params)
    return [_row_to_dict(row) for row in rows]
