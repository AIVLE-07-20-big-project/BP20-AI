"""AI 상태 저장소 연결 유틸리티.

개발 환경은 기존 SQLite를 그대로 사용할 수 있고, 운영 환경에서
AI_DATABASE_URL을 지정하면 MySQL을 사용한다.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import unquote, urlparse


def database_url() -> str:
    return os.getenv("AI_DATABASE_URL", "").strip()


def using_mysql() -> bool:
    return database_url().lower().startswith(("mysql://", "mysql+pymysql://"))


@contextmanager
def connect(database_path: str) -> Iterator[object]:
    """SQLite 또는 MySQL 연결을 열고, 호출자가 commit/rollback을 수행한다."""
    if not using_mysql():
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
        raise RuntimeError("AI_DATABASE_URL이 MySQL인데 PyMySQL이 설치되지 않았습니다") from exc

    parsed = urlparse(database_url().replace("mysql+pymysql://", "mysql://", 1))
    connection = pymysql.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=parsed.path.lstrip("/"),
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
