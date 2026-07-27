# 분석 요청 중 조회 API의 지연을 측정해 Celery 프로세스 격리 효과를 확인한다.
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "samples" / "sample_cafe_pos_omnichannel_weather_2026Q2.csv"
DEFAULT_OUT = ROOT / "docs" / "speed" / "api_isolation_result.json"
FORM = {"trdar_cd": "3120189", "svc_induty_cd": "CS100010"}

# 조회 데이터 증가가 측정에 영향을 주지 않도록 부하용과 조회용 사용자를 분리한다.
LOAD_USER = "bench-load"
PROBE_USER = "bench-probe"


def heavy_call(client: httpx.Client, base: str, csv_path: Path, timeout: float,
               user: str = LOAD_USER) -> None:
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        try:
            client.post(f"{base}/analyses", data={**FORM, "user_id": user},
                        files=files, timeout=timeout)
        except Exception:  # noqa: BLE001
            pass


def light_call(client: httpx.Client, base: str, timeout: float) -> tuple[float, int]:
    t0 = time.perf_counter()
    try:
        r = client.get(f"{base}/analyses", headers={"X-User-Id": PROBE_USER}, timeout=timeout)
        status = r.status_code
    except Exception:  # noqa: BLE001
        status = 0
    return (time.perf_counter() - t0) * 1000, status


def percentile(sorted_vals: list[float], p: float) -> float:
    k = (len(sorted_vals) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def sample_light(client: httpx.Client, base: str, n: int, gap: float, timeout: float) -> dict:
    lat, errors = [], 0
    for _ in range(n):
        e, s = light_call(client, base, timeout)
        lat.append(e)
        if s != 200:
            errors += 1
        time.sleep(gap)
    ordered = sorted(lat)
    return {
        "표본": len(lat),
        "p50": round(percentile(ordered, 50), 2),
        "p95": round(percentile(ordered, 95), 2),
        "max": round(max(lat), 2),
        "mean": round(statistics.mean(lat), 2),
        "오류": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="API 프로세스 격리 측정(조회 vs 분석 부하)")
    ap.add_argument("--base", default="http://127.0.0.1:8000/api/v1")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--load", type=int, default=8, help="동시에 던질 무거운 분석 요청 수")
    ap.add_argument("--samples", type=int, default=30, help="조회 요청 표본 수")
    ap.add_argument("--gap", type=float, default=0.2, help="조회 요청 간 간격(초)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    csv_path = Path(args.csv)

    with httpx.Client() as client:
        # 조회 데이터 한 건을 미리 만든다.
        print("[워밍업] 분석 1회 — 콜드 로드 제외 및 조회용 데이터 1건 생성", flush=True)
        heavy_call(client, args.base, csv_path, args.timeout, user=PROBE_USER)

        print(f"\n[1] 무부하 상태에서 GET /analyses {args.samples}회", flush=True)
        idle = sample_light(client, args.base, args.samples, args.gap, args.timeout)
        print(f"    p50={idle['p50']}ms  p95={idle['p95']}ms  max={idle['max']}ms", flush=True)

        print(f"\n[2] 분석 {args.load}건 동시 실행 중 GET /analyses {args.samples}회", flush=True)
        threads = [
            threading.Thread(target=heavy_call, args=(client, args.base, csv_path, args.timeout))
            for _ in range(args.load)
        ]
        for t in threads:
            t.start()
        time.sleep(1.0)  # 부하 시작 대기
        loaded = sample_light(client, args.base, args.samples, args.gap, args.timeout)
        for t in threads:
            t.join()
        print(f"    p50={loaded['p50']}ms  p95={loaded['p95']}ms  max={loaded['max']}ms", flush=True)

    ratio_p50 = round(loaded["p50"] / idle["p50"], 1) if idle["p50"] else None
    ratio_p95 = round(loaded["p95"] / idle["p95"], 1) if idle["p95"] else None

    result = {
        "endpoint_light": "GET /api/v1/analyses",
        "endpoint_heavy": "POST /api/v1/analyses",
        "조건": {"동시_분석_요청": args.load, "조회_표본": args.samples},
        "무부하": idle,
        "부하중": loaded,
        "악화배수": {"p50": ratio_p50, "p95": ratio_p95},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 결과: 가벼운 조회 요청 ===")
    print(f"{'상태':<12} {'p50(ms)':>10} {'p95(ms)':>10} {'max(ms)':>10}")
    print(f"{'무부하':<12} {idle['p50']:>10.1f} {idle['p95']:>10.1f} {idle['max']:>10.1f}")
    print(f"{'분석 ' + str(args.load) + '건 중':<12} {loaded['p50']:>10.1f} "
          f"{loaded['p95']:>10.1f} {loaded['max']:>10.1f}")
    print(f"\n악화 배수: p50 {ratio_p50}배 / p95 {ratio_p95}배")
    print(out_path)


if __name__ == "__main__":
    main()
