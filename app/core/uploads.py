from __future__ import annotations

import re
import time
from collections.abc import Collection
from pathlib import Path

from fastapi import UploadFile

from app.core.config import UPLOAD_DATA
from app.core.errors import api_error


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


# ------------------------------------------------------------------
# 비동기 분석용 업로드 저장소 — API가 저장하고 워커가 읽는다.
# 브로커에는 job_id만 싣는다(원본 25MiB를 base64로 실으면 메시지가 약 33MiB가 된다).
# ------------------------------------------------------------------
_JOB_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


class UploadNotFoundError(FileNotFoundError):
    """job_id에 해당하는 업로드 원본이 저장소에 없다."""


def _upload_path(job_id: str) -> Path:
    # job_id가 경로로 해석되지 않도록 형식을 제한한다(`..`, 슬래시, 드라이브 문자 차단).
    if not _JOB_ID_PATTERN.match(job_id):
        raise ValueError(f"허용되지 않는 job_id 형식입니다: {job_id!r}")
    return UPLOAD_DATA / f"{job_id}.csv"


def save_job_upload(job_id: str, raw_bytes: bytes) -> Path:
    """업로드 원본을 저장하고 경로를 반환한다(enqueue 전에 API가 호출)."""
    path = _upload_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 같은 디렉터리에 임시로 쓴 뒤 교체해, 워커가 절반만 쓰인 파일을 읽지 않도록 한다.
    tmp_path = path.with_suffix(".csv.part")
    tmp_path.write_bytes(raw_bytes)
    tmp_path.replace(path)
    return path


def read_job_upload(job_id: str) -> bytes:
    """저장된 업로드 원본을 읽는다(워커가 호출)."""
    path = _upload_path(job_id)
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise UploadNotFoundError(f"업로드 원본을 찾을 수 없습니다: {job_id}") from exc


def delete_job_upload(job_id: str) -> None:
    """업로드 원본을 삭제한다(성공한 잡의 정리용). 이미 없으면 조용히 넘어간다."""
    _upload_path(job_id).unlink(missing_ok=True)


def purge_expired_uploads(max_age_days: int = 7) -> int:
    """보존 기간이 지난 업로드 원본을 정리하고 삭제한 개수를 반환한다.

    실패한 잡의 파일은 재현을 위해 남겨두므로(계획 문서 §2), 이 정리가 유일한 회수 경로다.
    beat 청소 작업(4단계)에서 호출한다.
    """
    if not UPLOAD_DATA.exists():
        return 0
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for path in UPLOAD_DATA.glob("*.csv"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
