"""
원가·수익성 분석 - 매입 단가 변화 추적 & 원가율 계산

에이전트/LLM 없이 순수 계산으로 동작한다.

입력:
    cafe_expense_receipts.csv, cafe_expense_items.csv  (매입 데이터)
    cafe_products.csv                                   (판매 메뉴/가격)

⚠️ 원가율 계산의 데이터 한계:
    지금 매입 기록엔 "원두/우유/바닐라시럽"만 있고, 아이스티·레몬에이드에 들어가는
    찻잎/레몬이나 디저트류(크루아상 등) 원재료 매입 기록은 없다.
    그래서 커피 계열 음료(원두/우유/시럽으로 만드는 것)만 원가율을 계산하고,
    나머지는 "계산 불가"로 명시적으로 표시한다.
    (실제 서비스에서는 이 부분에 레시피/매입 데이터가 추가로 필요함)
"""

import argparse
import os

import pandas as pd

# 메뉴별 레시피(1잔당 원재료 사용량). 매입 데이터에 있는 원재료만 반영 가능.
# 단위는 "레시피 사용량 기준 단위"(g, ml)로 통일.
RECIPE = {
    "아메리카노":   [("원두(생두)", 18, "g")],
    "카페라떼":     [("원두(생두)", 18, "g"), ("우유", 200, "ml")],
    "카푸치노":     [("원두(생두)", 18, "g"), ("우유", 150, "ml")],
    "바닐라라떼":   [("원두(생두)", 18, "g"), ("우유", 200, "ml"), ("바닐라시럽", 15, "ml")],
    # 카라멜시럽 매입 기록이 없어 바닐라시럽 단가로 근사 (실제 서비스에선 캐러멜시럽 별도 매입 필요)
    "카라멜마키아토": [("원두(생두)", 18, "g"), ("우유", 200, "ml"), ("바닐라시럽", 15, "ml")],
}

# 매입 단위(kg/L/병) -> 레시피 기준 단위(g/ml) 환산 배수
# 1kg=1000g, 1L=1000ml, 시럽 1병=750ml 가정
PURCHASE_UNIT_TO_BASE = {"kg": 1000, "L": 1000, "병": 750}


def load_data(data_dir: str):
    receipts = pd.read_csv(os.path.join(data_dir, "cafe_expense_receipts.csv"), parse_dates=["TransactionDate"])
    items = pd.read_csv(os.path.join(data_dir, "cafe_expense_items.csv"))
    products = pd.read_csv(os.path.join(data_dir, "cafe_products.csv"))
    return receipts, items, products


# ------------------------------------------------------------------
# 1) 매입 단가 변화 추적
# ------------------------------------------------------------------
def detect_price_changes(receipts: pd.DataFrame, items: pd.DataFrame,
                          window_weeks: int = 4, min_change_pct: float = 10.0,
                          end_date=None) -> pd.DataFrame:
    """
    품목별로 "최근 window_weeks주 평균 단가" vs "그 이전 window_weeks주 평균 단가"를 비교.
    기준일(end_date)을 지정하지 않으면 데이터 상 가장 최근 거래일을 사용한다.
    (월간/연간 보고서처럼 "그 시점까지의 데이터만" 봐야 할 때 end_date를 지정)

    광고비/공과금/기기점검 같은 "단위(Unit) 없는 1회성 서비스 지출"은
    원재료가 아니므로 비교 대상에서 제외한다 (Unit이 있는 품목만 추적).
    """
    merged = items.merge(receipts[["ReceiptID", "TransactionDate"]], on="ReceiptID")
    merged = merged[merged["Unit"].notna()]  # 원재료(단위 있는 품목)만 대상으로 필터링

    if end_date is not None:
        merged = merged[merged["TransactionDate"] <= pd.Timestamp(end_date)]

    base_date = merged["TransactionDate"].max()
    recent_start = base_date - pd.Timedelta(weeks=window_weeks)
    prev_start = base_date - pd.Timedelta(weeks=window_weeks * 2)

    recent = merged[merged["TransactionDate"] > recent_start]
    prev = merged[(merged["TransactionDate"] > prev_start) & (merged["TransactionDate"] <= recent_start)]

    recent_avg = recent.groupby("ItemName")["UnitPrice"].mean()
    prev_avg = prev.groupby("ItemName")["UnitPrice"].mean()

    results = []
    for item_name in recent_avg.index:
        if item_name not in prev_avg.index:
            continue
        r_avg, p_avg = recent_avg[item_name], prev_avg[item_name]
        if p_avg == 0:
            continue
        change_pct = (r_avg - p_avg) / p_avg * 100
        if abs(change_pct) >= min_change_pct:
            results.append({
                "itemName": item_name,
                "previousAvgPrice": round(p_avg),
                "recentAvgPrice": round(r_avg),
                "changePct": round(change_pct, 1),
            })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("changePct", key=lambda s: s.abs(), ascending=False)
    return result_df


# ------------------------------------------------------------------
# 2) 원가율 계산
# ------------------------------------------------------------------
def latest_unit_prices(receipts: pd.DataFrame, items: pd.DataFrame, end_date=None) -> dict:
    """
    품목별 "가장 최근 매입 단가"와 그 매입 단위를 반환.
    end_date를 지정하면 그 시점 이전 데이터만으로 "최근"을 판단한다.
    """
    merged = items.merge(receipts[["ReceiptID", "TransactionDate"]], on="ReceiptID")
    if end_date is not None:
        merged = merged[merged["TransactionDate"] <= pd.Timestamp(end_date)]
    merged = merged.sort_values("TransactionDate")
    latest = merged.groupby("ItemName").tail(1)
    return {
        row["ItemName"]: (row["UnitPrice"], row["Unit"])
        for _, row in latest.iterrows()
    }


def calculate_cost_rates(receipts: pd.DataFrame, items: pd.DataFrame, products: pd.DataFrame,
                          end_date=None) -> pd.DataFrame:
    prices = latest_unit_prices(receipts, items, end_date=end_date)
    results = []

    for _, product in products.iterrows():
        name = product["ProductName"]
        sale_price = product["Price"]

        if name not in RECIPE:
            results.append({
                "productName": name, "salePrice": sale_price,
                "costPerServing": None, "costRatePct": None,
                "note": "원재료 매입 데이터 없음 - 계산 불가",
            })
            continue

        total_cost = 0
        missing_ingredient = None
        for ingredient_name, qty, unit in RECIPE[name]:
            if ingredient_name not in prices:
                missing_ingredient = ingredient_name
                break
            purchase_price, purchase_unit = prices[ingredient_name]
            base_multiplier = PURCHASE_UNIT_TO_BASE.get(purchase_unit)
            if base_multiplier is None:
                missing_ingredient = ingredient_name
                break
            price_per_base_unit = purchase_price / base_multiplier
            total_cost += price_per_base_unit * qty

        if missing_ingredient:
            results.append({
                "productName": name, "salePrice": sale_price,
                "costPerServing": None, "costRatePct": None,
                "note": f"'{missing_ingredient}' 매입 단가 정보 없음 - 계산 불가",
            })
            continue

        cost_rate = round(total_cost / sale_price * 100, 1)
        results.append({
            "productName": name, "salePrice": sale_price,
            "costPerServing": round(total_cost),
            "costRatePct": cost_rate,
            "note": "",
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="원가·수익성 분석 - 매입단가 추적 & 원가율 계산")
    parser.add_argument("--data-dir", default="cafe_synthetic_data", help="CSV 파일 폴더 경로")
    args = parser.parse_args()

    receipts, items, products = load_data(args.data_dir)

    print("=" * 60)
    print("📈 매입 단가 변화 추적 (최근 4주 vs 이전 4주, 10% 이상 변동)")
    print("=" * 60)
    changes = detect_price_changes(receipts, items)
    if changes.empty:
        print("변동 감지된 품목 없음.")
    else:
        for _, row in changes.iterrows():
            direction = "상승" if row["changePct"] > 0 else "하락"
            print(
                f"  {row['itemName']}: {row['previousAvgPrice']:,}원 -> {row['recentAvgPrice']:,}원 "
                f"({row['changePct']:+.1f}%, {direction})"
            )

    print("\n" + "=" * 60)
    print("📊 메뉴별 원가율 (최근 매입 단가 기준)")
    print("=" * 60)
    cost_rates = calculate_cost_rates(receipts, items, products)
    for _, row in cost_rates.iterrows():
        if pd.isna(row["costRatePct"]):
            print(f"  {row['productName']} (판매가 {row['salePrice']:,}원): {row['note']}")
        else:
            print(
                f"  {row['productName']} (판매가 {row['salePrice']:,}원): "
                f"원가 {row['costPerServing']:,.0f}원 -> 원가율 {row['costRatePct']}%"
            )


if __name__ == "__main__":
    main()
