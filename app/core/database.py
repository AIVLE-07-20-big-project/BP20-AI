"""AI 상태 저장소 연결 유틸리티.

개발 환경은 기존 SQLite를 그대로 사용할 수 있고, 운영 환경에서
AI_DB_HOST를 지정하면 개별 AI_DB_* 환경 변수로 MySQL을 사용한다.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.core.config import (
    AI_DB_HOST,
    AI_DB_NAME,
    AI_DB_PASSWORD,
    AI_DB_PORT,
    AI_DB_USERNAME,
)


def _mysql_connection_options() -> dict[str, object] | None:
    if not AI_DB_HOST:
        configured_without_host = [
            name
            for name, value in (
                ("AI_DB_NAME", AI_DB_NAME),
                ("AI_DB_USERNAME", AI_DB_USERNAME),
                ("AI_DB_PASSWORD", AI_DB_PASSWORD),
            )
            if value
        ]
        if configured_without_host:
            configured = ", ".join(configured_without_host)
            raise RuntimeError(f"{configured}가 설정되었지만 AI_DB_HOST가 비어 있습니다")
        return None

    missing = [
        name
        for name, value in (
            ("AI_DB_NAME", AI_DB_NAME),
            ("AI_DB_USERNAME", AI_DB_USERNAME),
            ("AI_DB_PASSWORD", AI_DB_PASSWORD),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"MySQL 연결 환경 변수가 누락되었습니다: {', '.join(missing)}")

    try:
        port = int(AI_DB_PORT)
    except ValueError as exc:
        raise RuntimeError("AI_DB_PORT는 숫자여야 합니다") from exc

    if not 1 <= port <= 65535:
        raise RuntimeError("AI_DB_PORT는 1~65535 범위여야 합니다")

    return {
        "host": AI_DB_HOST,
        "port": port,
        "user": AI_DB_USERNAME,
        "password": AI_DB_PASSWORD,
        "database": AI_DB_NAME,
    }


def using_mysql() -> bool:
    return bool(AI_DB_HOST)


@contextmanager
def connect(database_path: str) -> Iterator[object]:
    """SQLite 또는 MySQL 연결을 열고, 호출자가 commit/rollback을 수행한다."""
    mysql_options = _mysql_connection_options()
    if mysql_options is None:
        connection = sqlite3.connect(database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()
        return

    try:
        import pymysql
    except ImportError as exc:  # pragma: no cover - 운영 설정 오류 안내용
        raise RuntimeError("AI_DB_HOST가 설정되었지만 PyMySQL이 설치되지 않았습니다") from exc

    connection = pymysql.connect(
        **mysql_options,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        yield connection
    finally:
        connection.close()


def execute(connection: object, query: str, params: tuple | list = ()):
    """SQLite의 ? placeholder와 MySQL의 %s placeholder 차이를 숨긴다."""
    if using_mysql():
        query = query.replace("?", "%s")
    return connection.execute(query, params) if not using_mysql() else connection.cursor().execute(query, params)


def fetchone(connection: object, query: str, params: tuple | list = ()):
    if using_mysql():
        cursor = connection.cursor()
        cursor.execute(query.replace("?", "%s"), params)
        return cursor.fetchone()
    return connection.execute(query, params).fetchone()


def fetchall(connection: object, query: str, params: tuple | list = ()):
    if using_mysql():
        cursor = connection.cursor()
        cursor.execute(query.replace("?", "%s"), params)
        return cursor.fetchall()
    return connection.execute(query, params).fetchall()
