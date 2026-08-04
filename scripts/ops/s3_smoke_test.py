"""S3 자산 존재 여부와 임시 CSV 업로드 lifecycle을 확인하는 smoke test.

실행 전 AWS_PROFILE 또는 AWS SSO 로그인 상태를 준비하고,
S3_BUCKET_NAME과 prefix 환경변수를 설정한다. 실제 업무 파일은 수정하지 않는다.
"""
from __future__ import annotations

import os
import sys
from uuid import uuid4

import boto3


def required_objects() -> list[str]:
    model = os.getenv("MODEL_S3_PREFIX", "models/v1").strip("/")
    rag = os.getenv("RAG_S3_PREFIX", "rag/v1").strip("/")
    data = os.getenv("DATA_S3_PREFIX", "data").strip("/")
    return [
        f"{model}/ai_sales_model.pkl",
        f"{model}/cox_risk.pkl",
        f"{rag}/export/embeddings.npy",
        f"{rag}/export/chunks.jsonl",
        f"{rag}/export/manifest.json",
        f"{data}/processed/merged_sales_analysis.csv",
        f"{data}/agent/trend_panel.csv",
        f"{data}/source/store_stats.csv",
    ]


def main() -> int:
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    region = os.getenv("AWS_REGION", "ap-northeast-2")
    if not bucket:
        print("S3_BUCKET_NAME이 설정되지 않았습니다", file=sys.stderr)
        return 2

    client = boto3.client("s3", region_name=region)
    missing: list[str] = []
    for key in required_objects():
        try:
            client.head_object(Bucket=bucket, Key=key)
            print(f"[OK] {key}")
        except client.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                missing.append(key)
                print(f"[MISSING] {key}")
            else:
                raise

    prefix = os.getenv("UPLOAD_S3_PREFIX", "uploads/v1").strip("/")
    key = f"{prefix}/smoke-{uuid4().hex}.csv"
    payload = b"smoke,checked\ntrue,yes\n"
    client.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="text/csv")
    downloaded = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    if downloaded != payload:
        raise RuntimeError("S3 업로드 후 다운로드한 내용이 일치하지 않습니다")
    client.delete_object(Bucket=bucket, Key=key)
    print(f"[OK] upload/download/delete: {key}")

    if missing:
        print(f"필수 자산 {len(missing)}개가 없어 smoke test를 실패 처리합니다", file=sys.stderr)
        return 1
    print("S3 smoke test 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
