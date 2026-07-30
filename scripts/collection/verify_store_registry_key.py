# 대상 경로(AI 레포): scripts/collection/verify_store_registry_key.py
#
# 1차 시도에서 divId=adongCd, key=1168064000(역삼동, 블로그 예제 값)으로 호출했더니
# resultCode=03(NODATA_ERROR)이 나왔다. 파라미터 이름 자체는 서버가 정상 인식했다(에러 응답에도
# 전체 컬럼 스키마가 내려옴). 원인 후보 두 가지를 같이 검증한다:
#   A) adongCd 10자리 값이 최신 코드가 아닐 가능성 -> divId=signguCd(구 단위, 5자리)로 재시도
#   B) storeListInDong 자체가 아닌 반경 검색(storeListInRadius)으로 좌표 기반 호출이 되는지 확인
#      (좌표 기반이 되면 최소한 키/엔드포인트/서버 자체는 정상이라는 뜻 -> 문제는 동코드 값으로 좁혀짐)
#
# 실행:
#   $env:PUBLIC_DATA_STORE_API_KEY = "<디코딩키>"
#   python scripts/collection/verify_store_registry_key.py

import asyncio
import json
import os
import sys

import httpx

BASE = "http://apis.data.go.kr/B553077/api/open/sdsc2"


async def call(client: httpx.AsyncClient, label: str, path: str, params: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"GET {path} params={ {k: v for k, v in params.items() if k != 'serviceKey'} }")
    res = await client.get(f"{BASE}/{path}", params=params, timeout=30.0)
    print(f"HTTP status: {res.status_code}")

    if res.status_code != 200:
        print(res.text[:500])
        return

    try:
        raw = res.json()
    except Exception:
        print("JSON 아님(에러 XML로 추정):")
        print(res.text[:500])
        return

    header = raw.get("header", {})
    body = raw.get("body", {})
    print(f"resultCode={header.get('resultCode')} resultMsg={header.get('resultMsg')}")

    items = body.get("items")
    if isinstance(items, dict):
        items = items.get("item")
    if items:
        print(f"item 개수: {len(items)}")
        print("첫 번째 item 필드:")
        print(json.dumps(items[0], ensure_ascii=False, indent=2)[:1500])
    else:
        print("items 없음")


async def main() -> None:
    api_key = os.environ.get("PUBLIC_DATA_STORE_API_KEY")
    if not api_key:
        print("PUBLIC_DATA_STORE_API_KEY 환경변수가 없습니다.")
        sys.exit(1)

    async with httpx.AsyncClient() as client:
        # A안: 시군구 단위(강남구, 5자리)로 같은 storeListInDong 호출
        await call(
            client,
            "A) storeListInDong, divId=signguCd, key=11680(강남구)",
            "storeListInDong",
            {
                "serviceKey": api_key, "type": "json",
                "divId": "signguCd", "key": "11680",
                "numOfRows": 5, "pageNo": 1,
            },
        )

        # A-2안: 행정동 단위 그대로, 인근 다른 동 코드로 재시도(강남구 삼성동, 좀 더 최근 자료에서도 자주 등장하는 코드)
        await call(
            client,
            "A-2) storeListInDong, divId=adongCd, key=1168065000(삼성동)",
            "storeListInDong",
            {
                "serviceKey": api_key, "type": "json",
                "divId": "adongCd", "key": "1168065000",
                "numOfRows": 5, "pageNo": 1,
            },
        )

        # B안: 좌표 기반 반경 검색 (강남역 인근). 동코드와 무관하게 서버/키 자체가 정상인지 확인.
        await call(
            client,
            "B) storeListInRadius, 강남역 반경 500m",
            "storeListInRadius",
            {
                "serviceKey": api_key, "type": "json",
                "cx": "127.027619", "cy": "37.497952", "radius": "500",
                "numOfRows": 5, "pageNo": 1,
            },
        )


if __name__ == "__main__":
    asyncio.run(main())
