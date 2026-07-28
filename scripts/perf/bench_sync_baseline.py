# POST /api/v1/analyses 지연을 반복 측정해 p50/p95/p99·오류율·처리량을 집계한다.
# 성공 판정은 200/202이며, 동기 경로는 URL에 ?sync=true를 붙인다.
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "samples" / "sample_cafe_pos_omnichannel_weather_2026Q2.csv"
DEFAULT_OUT = ROOT / "docs" / "speed" / "sync_baseline_result.json"
FORM = {"trdar_cd": "3120189", "svc_induty_cd": "CS100010"}


def one_call(client: httpx.Client, url: str, csv_path: Path, timeout: float) -> tuple[float, int]:
    with open(csv_path, "rb") as f:
        files = {"file": (csv_path.name, f, "text/csv")}
        t0 = time.perf_counter()
        try:
            r = client.post(url, data=FORM, files=files, timeout=timeout)
            status = r.status_code
        except Exception:  # noqa: BLE001 — 타임아웃·연결 오류도 실패로 집계한다
            status = 0
        elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, status


def percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(latencies: list[float], errors: int, wall_s: float, concurrency: int) -> dict:
    ordered = sorted(latencies)
    n = len(latencies)
    return {
        "동시성": concurrency,
        "요청수": n,
        "소요_s": round(wall_s, 2),
        # 전체 배치의 벽시계 시간으로 처리량을 계산한다.
        "처리량_req_per_min": round(n / wall_s * 60, 1) if wall_s > 0 else None,
        "지연_ms": {
            "min": round(min(latencies), 2),
            "p50": round(percentile(ordered, 50), 2),
            "p95": round(percentile(ordered, 95), 2),
            "p99": round(percentile(ordered, 99), 2),
            "max": round(max(latencies), 2),
            "mean": round(statistics.mean(latencies), 2),
            "stdev": round(statistics.stdev(latencies), 2) if n > 1 else 0,
        },
        "오류율": round(errors / n, 4) if n else None,
        "raw_ms": [round(x, 2) for x in latencies],
    }


# runs개 요청을 최대 concurrency개씩 동시에 보낸다.
def run_level(
    client: httpx.Client, url: str, csv_path: Path, runs: int, concurrency: int, timeout: float,
) -> dict:
    latencies: list[float] = []
    errors = 0

    t0 = time.perf_counter()
    if concurrency <= 1:
        for i in range(runs):
            e, s = one_call(client, url, csv_path, timeout)
            latencies.append(e)
            if s not in (200, 202):
                errors += 1
            print(f"    run {i + 1}/{runs}: {e:.1f}ms status={s}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one_call, client, url, csv_path, timeout) for _ in range(runs)]
            for i, fut in enumerate(futures):
                e, s = fut.result()
                latencies.append(e)
                if s not in (200, 202):
                    errors += 1
                print(f"    run {i + 1}/{runs}: {e:.1f}ms status={s}", flush=True)
    wall_s = time.perf_counter() - t0

    return summarize(latencies, errors, wall_s, concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description="동기 baseline 측정 (POST /api/v1/analyses)")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/analyses")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--warmup", type=int, default=3, help="측정 제외 워밍업 횟수(순차)")
    parser.add_argument("--runs", type=int, default=10, help="동시성 수준당 측정 요청 수")
    parser.add_argument(
        "--concurrency",
        default="1",
        help='동시 요청 수. 쉼표로 여러 수준 지정 가능(예: "1,4,8,16")',
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="요청 타임아웃(초). 동시성이 높으면 지연이 커지므로 넉넉히 준다",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    csv_path = Path(args.csv)

    with httpx.Client() as client:
        # 최초 데이터 로딩 시간을 제외하기 위해 한 번 워밍업한다.
        print(f"[워밍업 {args.warmup}회]", flush=True)
        for i in range(args.warmup):
            e, s = one_call(client, args.url, csv_path, args.timeout)
            print(f"  warmup {i + 1}: {e:.1f}ms status={s}", flush=True)

        results = []
        for level in levels:
            print(f"\n[동시성 {level} — {args.runs}회]", flush=True)
            summary = run_level(client, args.url, csv_path, args.runs, level, args.timeout)
            lat = summary["지연_ms"]
            print(
                f"  => p50={lat['p50']}ms  p95={lat['p95']}ms  "
                f"처리량={summary['처리량_req_per_min']}req/min  오류율={summary['오류율']}",
                flush=True,
            )
            results.append(summary)

    result = {
        "endpoint": "POST /api/v1/analyses",
        "조건": {"워밍업": args.warmup, "수준당_요청수": args.runs, "동시성_수준": levels},
        "수준별": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 요약 ===")
    print(f"{'동시성':>6} {'p50(ms)':>10} {'p95(ms)':>10} {'처리량(req/min)':>16} {'오류율':>8}")
    for r in results:
        lat = r["지연_ms"]
        print(f"{r['동시성']:>6} {lat['p50']:>10.1f} {lat['p95']:>10.1f} "
              f"{r['처리량_req_per_min']:>16.1f} {r['오류율']:>8}")
    print(f"\n{out_path}")


if __name__ == "__main__":
    main()
