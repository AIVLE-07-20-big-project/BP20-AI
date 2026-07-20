"""
실행:
    python run_collect.py --test   # 소량 샘플만 호출해서 필드명/키 확인
    python run_collect.py --full   # 전체 수집 (2021Q1~2026Q1)

사전 설치:
    pip install httpx pandas
"""
import argparse
import asyncio
import os
from pathlib import Path

import httpx
import pandas as pd

from scripts.collection.collectors import (
    SalesEstimateCollector,
    StoreStatsCollector,
    FootTrafficCollector,
    ResidentPopulationCollector,
    WorkplacePopulationCollector,
    CulturalEventCollector,
    BigStoreCollector,
    SubwayStationGeomCollector,
    all_quarter_codes,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def get_setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value.strip()

    for candidate in (Path(__file__).parent / ".env", Path.cwd() / ".env"):
        if not candidate.exists():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == name:
                return value.strip().strip("'\"")
    return None


def get_api_key(*names: str) -> str:
    for name in names:
        value = get_setting(name)
        if value:
            return value
    raise SystemExit("API key not found. Set one of: " + ", ".join(names) + " in .env or environment variables.")


def load_keys() -> tuple[str, str, str, str, str, str, str, str]:
    return (
        get_api_key("SEOUL_API_KEY_SALES", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_STORE", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_FOOT_TRAFFIC", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_RESIDENT_POP", "SEOUL_API_KEY_FOOT_TRAFFIC", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_WORK", "SEOUL_API_KEY_FOOT_TRAFFIC", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_EVENT", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_BIG_STORE", "SEOUL_API_KEY"),
        get_api_key("SEOUL_API_KEY_BIGDATA_FLATFORM_SUBWAY"),
    )

async def run_test():
    """분기 파라미터를 쓰는 4개 API를 최신 분기(20254) 소량으로 테스트."""
    (api_key_sales, api_key_store, api_key_foot_traffic, api_key_resident_pop,
     api_key_work, api_key_event, api_key_big_store, api_key_subway) = load_keys()
    async with httpx.AsyncClient() as client:
        sales_df = pd.DataFrame(await SalesEstimateCollector(api_key_sales).fetch_page(
            client, 1, 100, extra_path="20254"
        ))
        print("=== 추정매출-상권 (VwsmTrdarSelngQq) ===")
        print(sales_df.shape, sales_df.columns.tolist())
        print(sales_df.head(2), "\n")

        store_df = pd.DataFrame(await StoreStatsCollector(api_key_store).fetch_page(
            client, 1, 100, extra_path="20254"
        ))
        print("=== 점포-상권 (VwsmTrdarStorQq) ===")
        print(store_df.shape, store_df.columns.tolist())
        print(store_df.head(2), "\n")

        pop_df = pd.DataFrame(await FootTrafficCollector(api_key_foot_traffic).fetch_page(
            client, 1, 100, extra_path="20254"
        ))
        print("=== 길단위인구-상권 (VwsmTrdarFlpopQq) ===")
        print(pop_df.shape, pop_df.columns.tolist())
        print(pop_df.head(2), "\n")

        repop_df = pd.DataFrame(await ResidentPopulationCollector(api_key_resident_pop).fetch_page(
            client, 1, 100
        ))
        print("=== 상주인구-상권 (VwsmTrdarRepopQq, 분기 파라미터 없음) ===")
        print(repop_df.shape, repop_df.columns.tolist())
        print(repop_df.head(2), "\n")

        work_df = pd.DataFrame(await WorkplacePopulationCollector(api_key_work).fetch_page(
            client, 1, 100
        ))
        print("=== 직장인구-상권 (VwsmTrdarWrcPopltnQq, 분기 파라미터 없음) ===")
        print(work_df.shape, work_df.columns.tolist())
        print(work_df.head(2), "\n")

    # 문화행사는 분기 파라미터가 없어서 소량 페이지만 확인
    event_collector = CulturalEventCollector(api_key_event)
    async with httpx.AsyncClient() as client:
        event_df = await event_collector.fetch_page(client, 1, 100)
        event_df = pd.DataFrame(event_df)
    print("=== 문화행사 정보 (culturalEventInfo) ===")
    print(event_df.shape, event_df.columns.tolist())
    print(event_df.head(2))

    big_store_collector = BigStoreCollector(api_key_big_store)
    async with httpx.AsyncClient() as client:
        big_store_df = pd.DataFrame(await big_store_collector.fetch_page(client, 1, 100))
    print("=== 대규모점포 인허가 (LOCALDATA_082501, 분기 파라미터 없음) ===")
    print(big_store_df.shape, big_store_df.columns.tolist())
    print(big_store_df.head(2))

    subway_df = await SubwayStationGeomCollector(api_key_subway).fetch_all()
    print("=== 지하철역 좌표 (TaimsKsccDvSubwayStationGeom, t-data.seoul.go.kr) ===")
    print(subway_df.shape, subway_df.columns.tolist())
    print(subway_df.head(2))


async def run_full(start_year=2024, end_year=2026, end_quarter=1, only="all"):
    (api_key_sales, api_key_store, api_key_foot_traffic, api_key_resident_pop,
     api_key_work, api_key_event, api_key_big_store, api_key_subway) = load_keys()
    quarters = all_quarter_codes(start_year=start_year, end_year=end_year, end_quarter=end_quarter)
    DATA_DIR.mkdir(exist_ok=True)
    print(f"수집 대상 분기: {quarters[0]} ~ {quarters[-1]} ({len(quarters)}개)\n")

    if only in {"all", "sales"}:
        print("[1/8] 추정매출-상권 수집 중...", flush=True)
        sales_df = await SalesEstimateCollector(api_key_sales).fetch_quarters(quarters)
        sales_df.to_csv(DATA_DIR / "sales_estimate.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(sales_df)}행 -> sales_estimate.csv\n", flush=True)

    if only in {"all", "store"}:
        print("[2/8] 점포-상권 수집 중...", flush=True)
        store_df = await StoreStatsCollector(api_key_store).fetch_quarters(quarters)
        store_df.to_csv(DATA_DIR / "store_stats.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(store_df)}행 -> store_stats.csv\n", flush=True)

    if only in {"all", "traffic"}:
        print("[3/8] 길단위인구-상권 수집 중...", flush=True)
        pop_df = await FootTrafficCollector(api_key_foot_traffic).fetch_quarters(quarters)
        pop_df.to_csv(DATA_DIR / "foot_traffic.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(pop_df)}행 -> foot_traffic.csv\n", flush=True)

    if only in {"all", "resident_pop"}:
        print("[4/8] 상주인구-상권 수집 중... (분기 파라미터 없음, 전체 한 번에)", flush=True)
        repop_df = await ResidentPopulationCollector(api_key_resident_pop).fetch_all()
        repop_df.to_csv(DATA_DIR / "resident_population.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(repop_df)}행 -> resident_population.csv\n", flush=True)

    if only in {"all", "work_pop"}:
        print("[5/8] 직장인구-상권 수집 중... (분기 파라미터 없음, 전체 한 번에)", flush=True)
        work_df = await WorkplacePopulationCollector(api_key_work).fetch_all()
        work_df.to_csv(DATA_DIR / "workplace_population.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(work_df)}행 -> workplace_population.csv\n", flush=True)

    if only in {"all", "event"}:
        print("[6/8] 문화행사 정보 수집 중... (분기 파라미터 없음, 전체 한 번에)", flush=True)
        event_df = await CulturalEventCollector(api_key_event).fetch_all()
        event_df.to_csv(DATA_DIR / "cultural_event.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(event_df)}행 -> cultural_event.csv\n", flush=True)

    if only in {"all", "big_store"}:
        print("[7/8] 대규모점포 인허가 정보 수집 중... (분기 파라미터 없음, 전체 한 번에)", flush=True)
        big_store_df = await BigStoreCollector(api_key_big_store).fetch_all()
        big_store_df.to_csv(DATA_DIR / "big_store.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(big_store_df)}행 -> big_store.csv\n", flush=True)

    if only in {"all", "subway"}:
        print("[8/8] 지하철역 좌표 수집 중... (t-data.seoul.go.kr, 전체 한 번에)", flush=True)
        subway_df = await SubwayStationGeomCollector(api_key_subway).fetch_all()
        subway_df.to_csv(DATA_DIR / "subway_stations.csv", index=False, encoding="utf-8-sig")
        print(f"  -> {len(subway_df)}행 -> subway_stations.csv\n", flush=True)
        print("  (승하차인원은 data/subway_data/CARD_SUBWAY_MONTH_YYYYMM.csv 수동 확보 파일 사용)")

    print("전체 수집 완료.")
    print("join key: TRDAR_CD(상권코드) + SVC_INDUTY_CD(업종코드, 매출/점포만) + STDR_YYQU_CD(분기)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-quarter", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--only", choices=["all", "sales", "store", "traffic", "resident_pop", "work_pop", "event", "big_store", "subway"], default="all")
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_test())
    elif args.full:
        asyncio.run(run_full(args.start_year, args.end_year, args.end_quarter, args.only))
    else:
        print("사용법: python run_collect.py --test  또는  python run_collect.py --full")
