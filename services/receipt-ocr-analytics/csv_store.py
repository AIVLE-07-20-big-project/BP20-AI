"""
OCR로 처리한 영수증 1건을, expense_analysis.py / cost_analysis.py가 바로 읽을 수 있는
CSV 스키마(cafe_expense_receipts.csv / cafe_expense_items.csv)에 그대로 누적 저장한다.

SQLite(receipt_db.py)와 별개의 경량 저장 방식 - 파일 하나로 관리되고,
2번(AI 가계부)/5번(원가분석) 스크립트가 그대로 읽어 쓸 수 있다는 게 장점.
"""

import os

import pandas as pd

RECEIPTS_COLUMNS = [
    "ReceiptID", "StoreID", "VendorName", "TransactionDate", "TransactionTime",
    "PaymentMethod", "Category", "SupplyAmount", "Vat", "TaxFreeAmount", "TotalAmount",
]
ITEMS_COLUMNS = [
    "ReceiptItemID", "ReceiptID", "ItemName", "Quantity", "Unit", "UnitPrice", "TotalPrice",
]


def _next_id(csv_path: str, id_column: str) -> int:
    if not os.path.exists(csv_path):
        return 1
    df = pd.read_csv(csv_path)
    if df.empty:
        return 1
    return int(df[id_column].max()) + 1


def _build_dedupe_key(structured_data: dict) -> str:
    vendor = (structured_data.get("storeName") or "").replace(" ", "")
    date = structured_data.get("transactionDate") or ""
    time_ = structured_data.get("transactionTime") or ""
    total = structured_data.get("totalAmount") or 0
    return f"{vendor}_{date}_{time_}_{total}"


def find_possible_duplicate(structured_data: dict, receipts_csv_path: str):
    """같은 (상호명, 날짜, 시각, 총액) 조합이 이미 CSV에 있으면 그 행을 반환, 없으면 None."""
    if not os.path.exists(receipts_csv_path):
        return None
    df = pd.read_csv(receipts_csv_path)
    if df.empty:
        return None

    target_vendor = (structured_data.get("storeName") or "").replace(" ", "")
    target_date = structured_data.get("transactionDate") or ""
    target_time = structured_data.get("transactionTime") or ""
    target_total = structured_data.get("totalAmount") or 0

    df["_vendor_norm"] = df["VendorName"].fillna("").astype(str).str.replace(" ", "")
    match = df[
        (df["_vendor_norm"] == target_vendor)
        & (df["TransactionDate"].astype(str) == str(target_date))
        & (df["TransactionTime"].astype(str) == str(target_time))
        & (df["TotalAmount"] == target_total)
    ]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def append_receipt_to_csv(
    structured_data: dict,
    data_dir: str,
    store_id: int = 1,
    receipts_filename: str = "cafe_expense_receipts.csv",
    items_filename: str = "cafe_expense_items.csv",
) -> int:
    """
    OCR 파이프라인의 최종 결과(structured_data)를 CSV 두 개(영수증/품목)에 이어서 저장.
    반환값: 새로 부여된 ReceiptID
    """
    os.makedirs(data_dir, exist_ok=True)
    receipts_path = os.path.join(data_dir, receipts_filename)
    items_path = os.path.join(data_dir, items_filename)

    receipt_id = _next_id(receipts_path, "ReceiptID")
    item_id_start = _next_id(items_path, "ReceiptItemID")

    receipt_row = {
        "ReceiptID": receipt_id,
        "StoreID": store_id,
        "VendorName": structured_data.get("storeName"),
        "TransactionDate": structured_data.get("transactionDate"),
        "TransactionTime": structured_data.get("transactionTime"),
        "PaymentMethod": structured_data.get("paymentMethod") or "현금",
        "Category": structured_data.get("category") or "기타운영비",
        "SupplyAmount": structured_data.get("supplyAmount"),
        "Vat": structured_data.get("vat"),
        "TaxFreeAmount": structured_data.get("taxFreeAmount") or 0,
        "TotalAmount": structured_data.get("totalAmount") or 0,
    }

    item_rows = []
    for i, item in enumerate(structured_data.get("items") or []):
        item_rows.append({
            "ReceiptItemID": item_id_start + i,
            "ReceiptID": receipt_id,
            "ItemName": item.get("itemName"),
            "Quantity": item.get("quantity") or 1,
            "Unit": item.get("unit"),
            "UnitPrice": item.get("unitPrice"),
            "TotalPrice": item.get("totalPrice") or 0,
        })

    # 영수증 CSV에 한 줄 추가
    receipt_df = pd.DataFrame([receipt_row], columns=RECEIPTS_COLUMNS)
    write_header = not os.path.exists(receipts_path)
    receipt_df.to_csv(receipts_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")

    # 품목 CSV에 여러 줄 추가 (품목이 있는 경우에만)
    if item_rows:
        items_df = pd.DataFrame(item_rows, columns=ITEMS_COLUMNS)
        write_header = not os.path.exists(items_path)
        items_df.to_csv(items_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")

    return receipt_id
