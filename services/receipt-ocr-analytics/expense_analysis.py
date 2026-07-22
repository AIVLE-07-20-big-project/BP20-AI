"""
AI 가계부 - 이상 지출 탐지 & 예산 초과 확인

에이전트나 LLM 없이, 순수 집계/통계 로직으로 동작한다.
(문장으로 다듬는 건 이후 단계에서 필요하면 LLM 한 번 호출로 추가하면 됨)

입력: cafe_expense_receipts.csv, cafe_budget.csv

사용법:
    python expense_analysis.py                          # 기본값: ./cafe_synthetic_data 폴더에서 찾음
    python expense_analysis.py --data-dir ./내폴더경로
"""

import argparse
import os

import pandas as pd


def load_data(receipts_path: str, budget_path: str):
    receipts = pd.read_csv(receipts_path, parse_dates=["TransactionDate"])
    budget = pd.read_csv(budget_path)
    return receipts, budget


# ------------------------------------------------------------------
# 1) 이상 지출 탐지 (Z-score 기반)
# ------------------------------------------------------------------
def detect_expense_anomalies(receipts: pd.DataFrame, z_thresh: float = 1.3) -> pd.DataFrame:
    """
    카테고리별로 "주(week) 단위 지출 합계"를 구하고,
    그 카테고리의 평균/표준편차 대비 얼마나 벗어났는지(Z-score)를 계산해서
    |Z| >= z_thresh 인 주만 이상 지출로 표시한다.

    Z-score = (해당 주 지출 - 그 카테고리 평균) / 표준편차
    표준편차가 0인 경우(매주 지출액이 완전히 똑같은 카테고리)는 비교 대상에서 제외.
    """
    df = receipts.copy()
    df["week"] = df["TransactionDate"].dt.to_period("W").apply(lambda p: p.start_time.date())

    weekly = df.groupby(["Category", "week"])["TotalAmount"].sum().reset_index()

    results = []
    for category, group in weekly.groupby("Category"):
        mean = group["TotalAmount"].mean()
        std = group["TotalAmount"].std(ddof=0)
        if std == 0 or pd.isna(std):
            continue
        for _, row in group.iterrows():
            z = (row["TotalAmount"] - mean) / std
            if abs(z) >= z_thresh:
                results.append({
                    "category": category,
                    "week": row["week"],
                    "weeklyAmount": int(row["TotalAmount"]),
                    "categoryAvg": round(mean),
                    "zScore": round(z, 2),
                    "direction": "급증" if z > 0 else "급감",
                })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("zScore", key=lambda s: s.abs(), ascending=False)
    return result_df


# ------------------------------------------------------------------
# 2) 예산 초과 확인
# ------------------------------------------------------------------
def check_budget_overage(receipts: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    """
    월(YearMonth) x 카테고리 단위로 실제 지출과 예산 목표치를 비교해서
    초과분(overAmount)과 초과율(overPct)을 계산한다.
    """
    df = receipts.copy()
    df["YearMonth"] = df["TransactionDate"].dt.to_period("M").astype(str)

    actual = df.groupby(["YearMonth", "Category"])["TotalAmount"].sum().reset_index()
    actual = actual.rename(columns={"TotalAmount": "actualAmount"})

    merged = actual.merge(
        budget[["YearMonth", "Category", "BudgetAmount"]],
        on=["YearMonth", "Category"], how="left"
    )
    merged["BudgetAmount"] = merged["BudgetAmount"].fillna(0)
    merged["overAmount"] = merged["actualAmount"] - merged["BudgetAmount"]
    merged["overPct"] = merged.apply(
        lambda r: round(r["overAmount"] / r["BudgetAmount"] * 100, 1) if r["BudgetAmount"] > 0 else None,
        axis=1,
    )

    over_budget = merged[merged["overAmount"] > 0].sort_values("overPct", ascending=False)
    return over_budget.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="AI 가계부 - 이상 지출 탐지 & 예산 초과 확인")
    parser.add_argument(
        "--data-dir", default="cafe_synthetic_data",
        help="CSV 파일들이 들어있는 폴더 경로 (기본값: ./cafe_synthetic_data)"
    )
    args = parser.parse_args()

    receipts_path = os.path.join(args.data_dir, "cafe_expense_receipts.csv")
    budget_path = os.path.join(args.data_dir, "cafe_budget.csv")

    for p in (receipts_path, budget_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {p}\n"
                f"--data-dir 옵션으로 CSV가 들어있는 폴더를 정확히 지정해주세요."
            )

    receipts, budget = load_data(receipts_path, budget_path)

    print("=" * 60)
    print("📉 이상 지출 탐지 (카테고리별 주간 지출, Z-score 기준)")
    print("=" * 60)
    anomalies = detect_expense_anomalies(receipts, z_thresh=1.3)
    if anomalies.empty:
        print("이상 지출로 탐지된 항목 없음.")
    else:
        for _, row in anomalies.iterrows():
            print(
                f"  [{row['week']}] {row['category']}: {row['weeklyAmount']:,}원 "
                f"(평균 {row['categoryAvg']:,}원 대비 Z={row['zScore']}, {row['direction']})"
            )

    print("\n" + "=" * 60)
    print("💸 예산 초과 확인 (월별 x 카테고리)")
    print("=" * 60)
    overage = check_budget_overage(receipts, budget)
    if overage.empty:
        print("예산 초과 항목 없음.")
    else:
        for _, row in overage.iterrows():
            print(
                f"  [{row['YearMonth']}] {row['Category']}: "
                f"실제 {int(row['actualAmount']):,}원 / 예산 {int(row['BudgetAmount']):,}원 "
                f"(+{int(row['overAmount']):,}원, {row['overPct']}% 초과)"
            )


if __name__ == "__main__":
    main()
