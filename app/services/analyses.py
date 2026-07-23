# 업로드한 매출 분석 결과를 후속 추천 실행에 재사용하기 위한 저장소
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.encoders import jsonable_encoder

from app.core.config import ANALYSES_DB


def _connect() -> sqlite3.Connection:
    ANALYSES_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(ANALYSES_DB))
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            trdar_cd TEXT NOT NULL,
            svc_induty_cd TEXT NOT NULL,
            yyqu_cd INTEGER,
            report_json TEXT NOT NULL,
            diagnosis_json TEXT NOT NULL,
            warnings_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(analyses)")}
    if "user_id" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN user_id TEXT")
    if "store_id" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN store_id TEXT")
    if "detailed_analysis_json" not in columns:
        connection.execute("ALTER TABLE analyses ADD COLUMN detailed_analysis_json TEXT")
    return connection


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
) -> dict:
    analysis_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as connection, connection:
        connection.execute(
            """
            INSERT INTO analyses (
                analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
                report_json, diagnosis_json, detailed_analysis_json,
                warnings_json, created_at, user_id, store_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id, trdar_cd, svc_induty_cd, yyqu_cd,
                _dump(report), _dump(diagnosis), _dump(detailed_analysis),
                _dump(warnings), created_at, user_id, store_id,
            ),
        )
    return get_analysis(analysis_id)


def get_analysis(analysis_id: str) -> dict | None:
    with closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "analysis_id": row["analysis_id"],
        "user_id": row["user_id"],
        "store_id": row["store_id"],
        "trdar_cd": row["trdar_cd"],
        "svc_induty_cd": row["svc_induty_cd"],
        "yyqu_cd": row["yyqu_cd"],
        "report": json.loads(row["report_json"]),
        "diagnosis": json.loads(row["diagnosis_json"]),
        "detailed_analysis": (
            json.loads(row["detailed_analysis_json"])
            if row["detailed_analysis_json"] is not None
            else None
        ),
        "warnings": json.loads(row["warnings_json"]),
        "created_at": row["created_at"],
    }


# 사용자 소유 분석을 최신순으로 반환한다
def list_analyses(user_id: str, store_id: str | None = None) -> list[dict]:

    query = "SELECT analysis_id FROM analyses WHERE user_id = ?"
    params: list[object] = [user_id]
    if store_id is not None:
        query += " AND store_id = ?"
        params.append(store_id)
    query += " ORDER BY created_at DESC"
    with closing(_connect()) as connection:
        ids = [row["analysis_id"] for row in connection.execute(query, params).fetchall()]
    return [analysis for analysis_id in ids if (analysis := get_analysis(analysis_id)) is not None]
