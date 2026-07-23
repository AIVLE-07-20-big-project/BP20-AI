from __future__ import annotations

from collections.abc import Collection

from fastapi import UploadFile

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
