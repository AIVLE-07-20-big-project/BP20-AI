# 대상 경로(AI 레포): run_sales_target_pipeline.py (레포 루트, 기존 파일 전체 교체)
#
# pipeline.py의 collect_and_generate()를 실제 서울 공공데이터 + 로컬 BE로 실행하고,
# 상위 N건을 BE의 POST /api/internal/sales-targets/bulk로 반영한다(신규).
# 전체 결과는 이전과 동일하게 CSV로도 저장한다(BE에는 상위 N건만 보낸다 — 53만 건을 통째로
# JSON으로 보내는 건 비현실적이라 판단).
#
# 사전 조건:
#   - BE 로컬 서버(./gradlew bootRun)가 떠 있어야 한다.
#   - 아래 4개 환경변수가 설정되어 있어야 한다.
#
# 실행 방법 (PowerShell):
#   $env:PUBLIC_DATA_STORE_API_KEY = "..."
#   $env:SEOUL_API_KEY = "..."
#   $env:BACKEND_INTERNAL_BASE_URL = "http://localhost:8080"
#   $env:INTERNAL_API_KEY = "..."
#   python run_sales_target_pipeline.py
#   python run_sales_target_pipeline.py --skip-backend-push   # BE에 반영 안 하고 CSV만 확인하고 싶을 때

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime

from app.sales_target.backend_client import SalesTargetIngestClient
from app.sales_target.pipeline import collect_and_generate, to_bulk_upsert_items


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"[오류] 환경변수 {name}가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)
    return value


async def main(top_n: int, output_path: str, skip_backend_push: bool) -> None:
    from scripts.collection.collectors import all_quarter_codes

    store_registry_api_key = _require_env("PUBLIC_DATA_STORE_API_KEY")
    seoul_api_key = _require_env("SEOUL_API_KEY")
    backend_base_url = os.environ.get("BACKEND_INTERNAL_BASE_URL", "http://localhost:8080")
    backend_internal_api_key = _require_env("INTERNAL_API_KEY")

    quarter_codes = all_quarter_codes()
    print(f"[1/2] 파이프라인 시작 (분기 코드: {quarter_codes})", flush=True)

    try:
        result = await collect_and_generate(
            store_registry_api_key=store_registry_api_key,
            seoul_api_key=seoul_api_key,
            backend_base_url=backend_base_url,
            backend_internal_api_key=backend_internal_api_key,
            quarter_codes=quarter_codes,
            top_n=None,
        )
    except Exception as exc:
        print(f"[오류] 파이프라인 실행 중 예외 발생: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print(f"\n총 후보 수: {len(result)}", flush=True)

    if len(result) == 0:
        print("결과가 0건입니다. 상가업소 수집 결과가 비어있거나 전부 자사 매장과 매칭돼 제외됐을 수 있습니다.")
        return

    if "trdar_cd" in result.columns:
        matched = result["trdar_cd"].notna().sum()
        print(f"상권코드 매칭 성공: {matched}/{len(result)}건 ({matched / len(result):.1%})")

    print("\nfinal_score 분포:")
    print(result["final_score"].describe())

    print(f"\n상위 {min(top_n, len(result))}건:")
    display_cols = [
        c
        for c in ["bizesNm", "rdnmAdr", "trdar_cd", "trdar_distance_m", "growth_score",
                  "traffic_score", "review_score", "similarity_score", "final_score"]
        if c in result.columns
    ]
    print(result[display_cols].head(top_n).to_string(index=False))

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n전체 결과 저장: {output_path}")

    if skip_backend_push:
        print("\n[2/2] --skip-backend-push 지정됨 — BE 반영 생략")
        return

    print(f"\n[2/2] 상위 {top_n}건 BE로 반영 중...", flush=True)
    top_result = result.head(top_n)
    items = to_bulk_upsert_items(top_result)
    if not items:
        print("BE로 보낼 유효한 항목이 없습니다(필수 필드 누락). 반영을 건너뜁니다.")
        return

    source_batch_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    ingest_client = SalesTargetIngestClient(backend_base_url, backend_internal_api_key)
    try:
        ingest_result = await ingest_client.push_bulk(source_batch_id, items)
    except Exception as exc:
        print(f"[오류] BE 반영 중 예외 발생: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print(f"BE 반영 완료 (batch={source_batch_id}): {ingest_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=20, help="화면 표시 + BE 반영 건수")
    parser.add_argument("--output", type=str, default="sales_targets_result.csv")
    parser.add_argument("--skip-backend-push", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.top_n, args.output, args.skip_backend_push))
