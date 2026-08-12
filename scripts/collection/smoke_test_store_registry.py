# 대상 경로(AI 레포): scripts/collection/smoke_test_store_registry.py
#
# STORE_REGISTRY_LOCAL_CSV를 다시 API 방식으로 전환한 뒤, 신규 가맹점 영업 타겟 추천
# 파이프라인이 실제로 쓰는 StoreRegistryCollector.fetch_sigungus()를 구 1곳만 대상으로
# 빠르게 호출해서 API 키/네트워크가 정상인지 확인하는 스모크 테스트.
#
# verify_store_registry_key.py와 다른 점: 그건 raw httpx.get으로 엔드포인트 자체를 찍어보는
# 저수준 점검이고, 이건 실제 파이프라인이 호출하는 StoreRegistryCollector 클래스를 그대로
# 써서 "코드 경로 전체"가 API 응답 -> DataFrame 변환까지 문제없이 도는지 확인한다.
#
# 실행 (레포 루트에서, .env에 PUBLIC_DATA_STORE_API_KEY가 있으면 app.core.config가 자동 로드):
#   python -m scripts.collection.smoke_test_store_registry
#   python -m scripts.collection.smoke_test_store_registry --sigungu 11680 11440   # 강남구+마포구 등 복수 지정
#
# 반드시 로컬 PC(직접 개발 환경)에서 실행할 것 — 샌드박스/격리된 실행 환경에서는
# apis.data.go.kr가 네트워크 allowlist에 없어 매번 403으로 막힌다(코드 문제 아님).
#
# 성공 시: STORE_REGISTRY_LOCAL_CSV가 .env에 없는(또는 주석 처리된) 상태에서 API를 호출해
# 정상적으로 상가업소 목록을 받아왔다는 뜻이므로, 그대로 run_sales_target_pipeline.py를 돌리면 된다.

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import app.core.config  # noqa: F401  — import 시점에 .env를 로드한다(app/core/config.py 참고)
from scripts.collection.store_registry_collector import SEOUL_SIGUNGU_CODES, StoreRegistryCollector


async def main(sigungu_codes: list[str]) -> None:
    local_csv = os.environ.get("STORE_REGISTRY_LOCAL_CSV")
    if local_csv:
        print(f"[경고] STORE_REGISTRY_LOCAL_CSV가 아직 설정되어 있습니다: {local_csv}")
        print("       .env에서 이 줄을 주석 처리(또는 삭제)해야 실제 API를 호출합니다.")
        sys.exit(1)

    api_key = os.environ.get("PUBLIC_DATA_STORE_API_KEY")
    if not api_key:
        print("[오류] PUBLIC_DATA_STORE_API_KEY 환경변수가 없습니다.")
        sys.exit(1)

    label = ", ".join(
        f"{name}({code})" for name, code in SEOUL_SIGUNGU_CODES.items() if code in sigungu_codes
    )
    print(f"[1/1] StoreRegistryCollector.fetch_sigungus({sigungu_codes}) 호출 중... ({label})")

    collector = StoreRegistryCollector(api_key=api_key)
    df = await collector.fetch_sigungus(sigungu_codes=sigungu_codes)

    if df.empty:
        print("\n결과가 비어 있습니다 — API 응답이 없거나(NODATA) 전부 실패했을 가능성이 있습니다.")
        print("verify_store_registry_key.py를 먼저 돌려서 resultCode/에러 메시지를 확인해보세요.")
        sys.exit(1)

    print(f"\n성공: {len(df)}건 수집됨")
    print(f"컬럼: {list(df.columns)}")
    print("\n샘플 5건:")
    cols = [c for c in ["bizesNm", "indsMclsNm", "rdnmAdr", "lon", "lat"] if c in df.columns]
    print(df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sigungu",
        nargs="+",
        default=["11680"],  # 강남구 — verify 스크립트에서 정상 응답 확인된 코드
        help="테스트할 시군구코드(5자리, 여러 개 가능). 기본값: 강남구(11680)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.sigungu))
