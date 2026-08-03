"""S3 object storage abstraction.

운영에서는 S3를 사용하고, 개발 환경에서 버킷을 지정하지 않으면 호출자가
기존 로컬 파일 저장소를 계속 사용할 수 있도록 S3 연결을 선택적으로 제공한다.
"""
from __future__ import annotations

import os
from pathlib import Path


class S3ConfigurationError(RuntimeError):
    pass


def bucket_name() -> str:
    return os.getenv("S3_BUCKET_NAME", "").strip()


def enabled() -> bool:
    return bool(bucket_name())


def _client():
    if not enabled():
        raise S3ConfigurationError("S3_BUCKET_NAME이 설정되지 않았습니다")
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - 배포 이미지 설정 오류 안내용
        raise S3ConfigurationError("S3 사용을 위해 boto3가 설치되어야 합니다") from exc
    return boto3.client("s3", region_name=os.getenv("AWS_REGION") or None)


def _key(prefix: str, name: str) -> str:
    prefix = prefix.strip("/")
    name = name.replace("\\", "/").lstrip("/")
    if not name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("허용되지 않는 S3 object key입니다")
    return f"{prefix}/{name}" if prefix else name


def upload_file(local_path: Path, *, prefix: str, name: str | None = None) -> str:
    """로컬 파일을 S3에 업로드하고 object key를 반환한다."""
    key = _key(prefix, name or local_path.name)
    _client().upload_file(str(local_path), bucket_name(), key)
    return key


def download_file(*, prefix: str, name: str, local_path: Path) -> Path:
    """S3 object를 로컬 경로에 내려받는다."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket_name(), _key(prefix, name), str(local_path))
    return local_path


def upload_bytes(data: bytes, *, prefix: str, name: str, content_type: str | None = None) -> str:
    key = _key(prefix, name)
    extra_args = {"ContentType": content_type} if content_type else {}
    _client().put_object(Bucket=bucket_name(), Key=key, Body=data, **extra_args)
    return key


def download_bytes(*, prefix: str, name: str) -> bytes:
    response = _client().get_object(Bucket=bucket_name(), Key=_key(prefix, name))
    return response["Body"].read()


def delete_object(*, prefix: str, name: str) -> None:
    _client().delete_object(Bucket=bucket_name(), Key=_key(prefix, name))


def list_objects(*, prefix: str) -> list[dict]:
    response = _client().list_objects_v2(Bucket=bucket_name(), Prefix=prefix.strip("/") + "/")
    return response.get("Contents", [])
