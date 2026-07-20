"""
기상청 날씨 수집기

서울 관측소 기준 월별 기상 요약을 수집하고, 분기 단위로 집계한다.

사용
    python weather_collect.py --station 108 --start-year 2024 --start-month 1 --end-year 2026 --end-month 1

산출물
    data/weather_seoul_monthly_raw.csv
    data/weather_seoul_quarterly.csv
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "source"

BASE_URL = "https://apihub.kma.go.kr/api/typ02/openApi/SfcMtlyInfoService/getMmSumry2"


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


def load_api_key() -> str:
    for key_name in ("KMA_API_KEY", "KMA_WEATHER_API_KEY", "KMA_API"):
        value = get_setting(key_name)
        if value:
            return value
    raise SystemExit("KMA API key not found. Add KMA_API_KEY to .env")


def month_iter(start_year: int, start_month: int, end_year: int, end_month: int) -> Iterable[tuple[int, int]]:
    cur = start_year * 12 + (start_month - 1)
    end = end_year * 12 + (end_month - 1)
    while cur <= end:
        year = cur // 12
        month = cur % 12 + 1
        yield year, month
        cur += 1


def quarter_code(year: int, month: int) -> int:
    quarter = (month - 1) // 3 + 1
    return year * 10 + quarter


def _xml_rows(text: str) -> list[dict]:
    root = ET.fromstring(text)
    candidates = root.findall(".//info")
    if not candidates:
        candidates = root.findall(".//item")
    if not candidates:
        candidates = [child for child in root if list(child)]

    rows: list[dict] = []
    for node in candidates:
        row = {}
        for child in list(node):
            key = child.tag.split("}")[-1]
            value = (child.text or "").strip()
            row[key] = value
        if row:
            rows.append(row)
    return rows


def _json_rows(obj) -> list[dict]:
    if isinstance(obj, list):
        rows = [x for x in obj if isinstance(x, dict)]
        if rows:
            return rows
        for item in obj:
            rows = _json_rows(item)
            if rows:
                return rows
        return []
    if isinstance(obj, dict):
        for key in ("item", "row", "rows", "data", "list"):
            if key in obj:
                rows = _json_rows(obj[key])
                if rows:
                    return rows

        if obj and all(not isinstance(v, (list, dict)) for v in obj.values()):
            return [obj]
        for value in obj.values():
            rows = _json_rows(value)
            if rows:
                return rows
    return []


def parse_rows(response: httpx.Response) -> list[dict]:
    text = response.text.strip()
    if not text:
        return []

    if text.startswith("<"):
        return _xml_rows(text)

    try:
        payload = response.json()
    except Exception:
        return []
    return _json_rows(payload)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    rename_map = {
        "stn_id": "stnid",
        "stn_ko": "stnko",
        "sumssday": "ss_day",
        "sum_ss_day": "ss_day",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns and k != v})
    text_cols = {"stnko", "stn_en", "wd_max", "resultMsg", "dataType"}
    for c in out.columns:
        if c in text_cols:
            continue
        out[c] = pd.to_numeric(out[c].replace({"null": np.nan, "NULL": np.nan, "": np.nan}), errors="coerce")
    drop_cols = [c for c in out.columns if out[c].notna().sum() == 0]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


async def fetch_monthly_summary(client: httpx.AsyncClient, api_key: str, year: int, month: int) -> pd.DataFrame:
    params = {
        "pageNo": 1,
        "numOfRows": 500,
        "dataType": "XML",
        "year": year,
        "month": f"{month:02d}",
        "authKey": api_key,
    }
    res = await client.get(BASE_URL, params=params, timeout=60.0)
    res.raise_for_status()
    rows = parse_rows(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return normalize_frame(df)


def pick_seoul_station(df: pd.DataFrame, station: int) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ("stnid", "Stn_id", "STN", "station", "stn_id"):
        if col in df.columns:
            return df[df[col].astype(str) == str(station)].copy()
    return df.copy()


def build_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    if "year" not in out.columns or "month" not in out.columns:
        return out
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out["STDR_YYQU_CD"] = [quarter_code(y, m) for y, m in zip(out["year"], out["month"])]

    if "rn_day" in out.columns:
        out["precip_total_mm"] = out["rn_day"]
    if "rn" in out.columns:
        out["precip_anomaly_mm"] = out["rn"]

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    keep_cols = ["year", "month", "STDR_YYQU_CD"]
    for c in ("stnid", "stnko", "Stn_id", "Stn_ko"):
        if c in out.columns:
            keep_cols.append(c)
    weather_cols = [c for c in numeric_cols if c not in {"year", "month", "STDR_YYQU_CD"}]
    meta = {}
    for c in ("stnid", "stnko", "Stn_id", "Stn_ko"):
        if c in out.columns:
            meta[c] = out[c].iloc[0]

    agg_map: dict[str, str] = {}
    mean_cols = {
        "taavg", "avgtamax", "avgtamin", "ta", "ta_max", "ta_min", "avghm", "ws", "ws_max",
        "ps", "pa", "avgcatot", "daydur", "ev_s", "avgte05", "tamin",
    }
    sum_cols = {
        "rn_day", "rn", "rain", "rn_day_cnt1", "rn_day_cnt2", "rn_day_cnt3", "rn_day_cnt4",
        "sd_new", "sd_max", "sd_day", "rn_60m_max", "rn_10m_max", "rn_pow_max",
        "rn_dur", "ss_day", "sumssday", "ss", "si_day", "si", "ws_max_tm", "ta_max_tm", "ta_min_tm",
        "precip_total_mm", "precip_anomaly_mm",
    }
    for col in weather_cols:
        if col.lower() in sum_cols or col in sum_cols:
            agg_map[col] = "sum"
        elif col.lower() in mean_cols or col in mean_cols:
            agg_map[col] = "mean"
        else:
            agg_map[col] = "mean"

    grouped = out.groupby("STDR_YYQU_CD", as_index=False).agg(agg_map)
    for k, v in meta.items():
        grouped[k] = v
    grouped["year"] = grouped["STDR_YYQU_CD"] // 10
    grouped["quarter"] = grouped["STDR_YYQU_CD"] % 10

    quarter_days = {
        1: lambda y: calendar.monthrange(y, 1)[1] + calendar.monthrange(y, 2)[1] + calendar.monthrange(y, 3)[1],
        2: lambda y: calendar.monthrange(y, 4)[1] + calendar.monthrange(y, 5)[1] + calendar.monthrange(y, 6)[1],
        3: lambda y: calendar.monthrange(y, 7)[1] + calendar.monthrange(y, 8)[1] + calendar.monthrange(y, 9)[1],
        4: lambda y: calendar.monthrange(y, 10)[1] + calendar.monthrange(y, 11)[1] + calendar.monthrange(y, 12)[1],
    }
    grouped["quarter_days"] = grouped.apply(
        lambda r: quarter_days[int(r["quarter"])](int(r["year"])),
        axis=1,
    )

    for col in ("rn_day", "rn", "max_rn_day", "rn_day_cnt1", "rn_day_cnt2", "rn_day_cnt3", "rn_day_cnt4", "cnt1", "cnt2", "cnt3", "cnt4", "cnt5", "cnt6", "cnt7", "cnt8", "cnt9"):
        if col in grouped.columns:
            grouped[f"{col}_per_day"] = grouped[col] / grouped["quarter_days"]

    if "ws" in grouped.columns and "ws_max" in grouped.columns:
        grouped["wind_gust_spread"] = grouped["ws_max"] - grouped["ws"]
        grouped["wind_gust_ratio"] = grouped["ws_max"] / grouped["ws"].replace(0, np.nan)

    drop_cols = [c for c in grouped.columns if grouped[c].notna().sum() == 0]
    if drop_cols:
        grouped = grouped.drop(columns=drop_cols)
    return grouped


async def run(
    station: int,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> None:
    api_key = load_api_key()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    monthly_frames = []
    async with httpx.AsyncClient() as client:
        for year, month in month_iter(start_year, start_month, end_year, end_month):
            print(f"[{year}-{month:02d}] 수집 중...", flush=True)
            df = await fetch_monthly_summary(client, api_key, year, month)
            df = pick_seoul_station(df, station)
            if df.empty:
                print("  -> 데이터 없음", flush=True)
                continue
            df = df.copy()
            df["year"] = year
            df["month"] = month
            monthly_frames.append(df)
            print(f"  -> {len(df)}행", flush=True)

    if not monthly_frames:
        raise SystemExit("수집된 날씨 데이터가 없습니다.")

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = normalize_frame(monthly)
    monthly_out = DATA_DIR / "weather_seoul_monthly_raw.csv"
    monthly.to_csv(monthly_out, index=False, encoding="utf-8-sig")

    quarterly = build_quarterly(monthly)
    quarterly_out = DATA_DIR / "weather_seoul_quarterly.csv"
    quarterly.to_csv(quarterly_out, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {monthly_out}")
    print(f"저장 완료: {quarterly_out}")
    print(f"행 수: monthly={len(monthly):,}, quarterly={len(quarterly):,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station", type=int, default=108, help="서울 관측소 번호 (기본 108)")
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--end-month", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(
        run(
            station=args.station,
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
        )
    )


if __name__ == "__main__":
    main()
