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
#   레포 루트 .env에 PUBLIC_DATA_STORE_API_KEY / SEOUL_API_KEY / INTERNAL_API_KEY가 있으면
#   자동으로 로드된다(아래 load_dotenv 참고). 필요하면 아래처럼 직접 셋업해서 .env 값을 덮어써도 된다.
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
from pathlib import Path

from dotenv import load_dotenv

# 이 스크립트는 app.core.config를 import하지 않는 경로라 .env가 자동으로 안 실려서(버그였음),
# 여기서 직접 로드한다. override=False라 이미 셋업된 쉘 환경변수($env:...)가 있으면 그게 우선한다.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from app.sales_target.backend_client import SalesTargetIngestClient
from app.sales_target.pipeline import collect_and_generate, to_bulk_upsert_items


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"[오류] 환경변수 {name}가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)
    return value


async def main(top_n: int, output_path: str, skip_backend_push: bool) -> None:
    from scripts.collection.collectors import latest_quarter_codes

    store_registry_api_key = _require_env("PUBLIC_DATA_STORE_API_KEY")
    seoul_api_key = _require_env("SEOUL_API_KEY")
    backend_base_url = os.environ.get("BACKEND_INTERNAL_BASE_URL", "http://localhost:8080")
    backend_internal_api_key = _require_env("INTERNAL_API_KEY")
    # 6단계(리뷰 반영)는 선택 사항이라 _require_env가 아니라 optional로 읽는다. 키가 없으면
    # collect_and_generate()가 조용히 6단계를 건너뛰고 review_score는 전부 NaN으로 남는다
    # (pipeline.py의 collect_and_generate 참고). 아래에서 없으면 사용자에게 안내만 출력한다.
    google_places_api_key = os.environ.get("GOOGLE_PLACES_API_KEY") or None
    if not google_places_api_key:
        print(
            "[안내] GOOGLE_PLACES_API_KEY가 없어 6단계(리뷰 반영)를 건너뜁니다 — "
            "review_score는 전부 NaN으로 남고 final_score는 3축(성장/유동인구/유사도)만으로 계산됩니다.",
            flush=True,
        )

    # district_metrics는 상권별 최근 2개 분기만 쓰므로 21개 전체 대신 최근 4개(1년치, 여유분
    # 포함)만 받는다 — 배치 소요시간이 여기서 가장 크게 줄어든다.
    quarter_codes = latest_quarter_codes()
    print(f"[1/2] 파이프라인 시작 (분기 코드: {quarter_codes})", flush=True)

    try:
        result = await collect_and_generate(
            store_registry_api_key=store_registry_api_key,
            seoul_api_key=seoul_api_key,
            backend_base_url=backend_base_url,
            backend_internal_api_key=backend_internal_api_key,
            quarter_codes=quarter_codes,
            top_n=None,
            google_places_api_key=google_places_api_key,
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
