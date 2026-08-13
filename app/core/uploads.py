from __future__ import annotations

import re
import time
from collections.abc import Collection
from pathlib import Path

from fastapi import UploadFile

from app.core.config import UPLOAD_DATA, UPLOAD_S3_PREFIX
from app.core.errors import api_error
from app.core.object_storage import (
    delete_object,
    download_bytes,
    enabled as s3_enabled,
    upload_bytes,
)


MIB = 1024 * 1024
MAX_CSV_UPLOAD_BYTES = 10 * MIB
MAX_POS_UPLOAD_BYTES = 25 * MIB
MAX_IMAGE_UPLOAD_BYTES = 10 * MIB

CSV_EXTENSIONS = {".csv"}
CSV_CONTENT_TYPES = {
    "application/csv",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "text/csv",
    "text/plain",
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_CONTENT_TYPES = {
    "application/octet-stream",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def validate_upload_type(
    file: UploadFile,
    *,
    extensions: Collection[str],
    content_types: Collection[str],
    type_name: str,
) -> None:
    filename = file.filename or ""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = (file.content_type or "").lower()
    if extension not in extensions or content_type not in content_types:
        raise api_error(
            415,
            "UNSUPPORTED_UPLOAD_TYPE",
            f"지원하지 않는 {type_name} 파일 형식입니다",
            {
                "filename": filename,
                "contentType": content_type,
                "allowedExtensions": sorted(extensions),
            },
        )


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    declared_size = getattr(file, "size", None)
    if declared_size is not None and declared_size > max_bytes:
        raise api_error(
            413,
            "UPLOAD_TOO_LARGE",
            "업로드 파일 크기가 제한을 초과했습니다",
            {"maxBytes": max_bytes, "actualBytes": declared_size},
        )

    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(MIB):
        total += len(chunk)
        if total > max_bytes:
            raise api_error(
                413,
                "UPLOAD_TOO_LARGE",
                "업로드 파일 크기가 제한을 초과했습니다",
                {"maxBytes": max_bytes, "actualBytes": total},
            )
        chunks.append(chunk)
    return b"".join(chunks)


# API가 저장한 원본을 워커가 job_id로 조회한다.
_JOB_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


class UploadNotFoundError(FileNotFoundError):
    """job_id에 해당하는 업로드 원본이 저장소에 없다."""


def _upload_path(job_id: str) -> Path:
    # 경로 이동 문자와 드라이브 문자를 허용하지 않는다.
    if not _JOB_ID_PATTERN.match(job_id):
        raise ValueError(f"허용되지 않는 job_id 형식입니다: {job_id!r}")
    return UPLOAD_DATA / f"{job_id}.csv"


def save_job_upload(job_id: str, raw_bytes: bytes) -> Path:
    """업로드 원본을 저장하고 경로를 반환한다(enqueue 전에 API가 호출)."""
    _upload_path(job_id)  # job_id 형식 검증
    if s3_enabled():
        upload_bytes(
            raw_bytes,
            prefix=UPLOAD_S3_PREFIX,
            name=f"{job_id}.csv",
            content_type="text/csv",
        )
        return _upload_path(job_id)
    path = _upload_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 임시 파일을 교체해 워커의 불완전한 파일 읽기를 막는다.
    tmp_path = path.with_suffix(".csv.part")
    tmp_path.write_bytes(raw_bytes)
    tmp_path.replace(path)
    return path


def read_job_upload(job_id: str) -> bytes:
    """저장된 업로드 원본을 읽는다(워커가 호출)."""
    _upload_path(job_id)  # job_id 형식 검증
    if s3_enabled():
        try:
            return download_bytes(prefix=UPLOAD_S3_PREFIX, name=f"{job_id}.csv")
        except Exception as exc:
            raise UploadNotFoundError(f"업로드 원본을 찾을 수 없습니다: {job_id}") from exc
    path = _upload_path(job_id)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise UploadNotFoundError(f"업로드 원본을 찾을 수 없습니다: {job_id}") from exc


def delete_job_upload(job_id: str) -> None:
    """업로드 원본을 삭제한다(성공한 잡의 정리용). 이미 없으면 조용히 넘어간다."""
    _upload_path(job_id)  # job_id 형식 검증
    if s3_enabled():
        delete_object(prefix=UPLOAD_S3_PREFIX, name=f"{job_id}.csv")
        return
    _upload_path(job_id).unlink(missing_ok=True)


def purge_expired_uploads(max_age_days: int = 7) -> int:
    """보존 기간이 지난 업로드 원본을 정리하고 삭제한 개수를 반환한다."""
    if s3_enabled():
        # S3에서는 Lifecycle 정책이 삭제를 담당한다. 애플리케이션이 목록을
        # 전부 스캔해 삭제하지 않도록 0을 반환한다.
        return 0
    if not UPLOAD_DATA.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for path in UPLOAD_DATA.glob("*.csv"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
