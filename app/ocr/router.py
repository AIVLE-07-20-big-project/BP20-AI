# 영수증 OCR, 비용 분석, 리포트 API

import os
import shutil
import tempfile
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.ocr.expense_analysis import check_budget_overage, detect_expense_anomalies
from app.ocr.cost_analysis import calculate_cost_rates, detect_price_changes
from app.ocr.report import build_html_from_frames

router = APIRouter(tags=["OCR"])


# ------------------------------------------------------------------
# 공통 변환 헬퍼: Java가 보내는 camelCase JSON <-> 기존 분석 코드의 PascalCase DataFrame
# ------------------------------------------------------------------
def _receipts_to_df(receipts: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(receipts)
    if df.empty:
        empty_df = pd.DataFrame(columns=[
            "ReceiptID", "StoreID", "VendorName", "TransactionDate", "TransactionTime",
            "PaymentMethod", "Category", "SupplyAmount", "Vat", "TaxFreeAmount", "TotalAmount",
        ])
        # 비어있어도 datetime 타입을 유지해야 이후 .dt 접근/날짜 비교 연산이 깨지지 않는다.
        empty_df["TransactionDate"] = pd.to_datetime(empty_df["TransactionDate"])
        return empty_df
    df = df.rename(columns={
        "receiptId": "ReceiptID", "storeId": "StoreID", "vendorName": "VendorName",
        "transactionDate": "TransactionDate", "transactionTime": "TransactionTime",
        "paymentMethod": "PaymentMethod", "category": "Category",
        "supplyAmount": "SupplyAmount", "vat": "Vat", "taxFreeAmount": "TaxFreeAmount",
        "totalAmount": "TotalAmount",
    })
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    return df


def _items_to_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=[
            "ReceiptItemID", "ReceiptID", "ItemName", "Quantity", "Unit", "UnitPrice", "TotalPrice",
        ])
    return df.rename(columns={
        "receiptItemId": "ReceiptItemID", "receiptId": "ReceiptID", "itemName": "ItemName",
        "quantity": "Quantity", "unit": "Unit", "unitPrice": "UnitPrice", "totalPrice": "TotalPrice",
    })


def _budgets_to_df(budgets: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(budgets)
    if df.empty:
        return pd.DataFrame(columns=["YearMonth", "Category", "BudgetAmount", "StoreID"])
    return df.rename(columns={
        "yearMonth": "YearMonth", "category": "Category",
        "budgetAmount": "BudgetAmount", "storeId": "StoreID",
    })


def _products_to_df(products: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(products)
    if df.empty:
        return pd.DataFrame(columns=["ProductID", "StoreID", "ProductName", "Category", "Price", "DiscountRate"])
    return df.rename(columns={
        "productId": "ProductID", "storeId": "StoreID", "productName": "ProductName",
        "category": "Category", "price": "Price", "discountRate": "DiscountRate",
    })


def _orders_to_df(orders: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(orders)
    if df.empty:
        empty_df = pd.DataFrame(columns=[
            "OrderID", "StoreID", "OrderType", "TotalAmount", "DiscountAmount",
            "PaymentMethod", "OrderedDate", "OrderedTime",
        ])
        # 비어있어도 datetime 타입을 유지해야 이후 resample/날짜 비교 연산이 깨지지 않는다.
        empty_df["OrderedDate"] = pd.to_datetime(empty_df["OrderedDate"])
        return empty_df
    df = df.rename(columns={
        "orderId": "OrderID", "storeId": "StoreID", "orderType": "OrderType",
        "totalAmount": "TotalAmount", "discountAmount": "DiscountAmount",
        "paymentMethod": "PaymentMethod", "orderedDate": "OrderedDate", "orderedTime": "OrderedTime",
    })
    df["OrderedDate"] = pd.to_datetime(df["OrderedDate"])
    return df


# ------------------------------------------------------------------
# 1) 영수증 OCR
# ------------------------------------------------------------------
@router.post("/api/v1/receipts/parse")
async def parse_receipt(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 영수증 이미지를 업로드하면 OCR + 좌표기반 추출 + 검증까지 마친
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 무거운 OCR 의존성은 영수증 처리 요청이 들어올 때만 로드한다.
        from app.ocr.pipeline import (
            classify_document_type,
            extract_items,
            extract_store_name,
            native_nlp_parser,
            preprocess_and_ocr,
            validate_and_reflect,
        )

        ocr_texts, ocr_results, image_height = preprocess_and_ocr(tmp_path)
        document_type = classify_document_type(ocr_texts)
        structured = native_nlp_parser(ocr_texts, document_type=document_type)

        store_name = extract_store_name(ocr_results, image_height)
        if store_name:
            structured["storeName"] = store_name

        structured["items"] = extract_items(ocr_results)
        final_result = validate_and_reflect(structured, ocr_texts)

        return {
            "ocrText": ocr_texts,
            "result": final_result,
        }
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"영수증 인식 처리 중 오류: {e}") from e
    finally:
        os.remove(tmp_path)


# ------------------------------------------------------------------
# 2) AI 가계부: 이상 지출 탐지 / 예산 초과 확인
# ------------------------------------------------------------------
class ExpenseAnomalyRequest(BaseModel):
    receipts: List[Dict[str, Any]]
    zThreshold: float = 1.3


@router.post("/api/v1/analytics/expense-anomalies")
async def expense_anomalies(body: ExpenseAnomalyRequest) -> List[Dict[str, Any]]:
    receipts_df = _receipts_to_df(body.receipts)
    result = detect_expense_anomalies(receipts_df, z_thresh=body.zThreshold)
    if result.empty:
        return []
    result["week"] = result["week"].astype(str)
    return result.to_dict(orient="records")


class BudgetOverageRequest(BaseModel):
    receipts: List[Dict[str, Any]]
    budgets: List[Dict[str, Any]]


@router.post("/api/v1/analytics/budget-overage")
async def budget_overage(body: BudgetOverageRequest) -> List[Dict[str, Any]]:
    receipts_df = _receipts_to_df(body.receipts)
    budget_df = _budgets_to_df(body.budgets)
    result = check_budget_overage(receipts_df, budget_df)
    if result.empty:
        return []
    result = result.rename(columns={
        "YearMonth": "yearMonth", "Category": "category", "BudgetAmount": "budgetAmount",
    })
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")


# ------------------------------------------------------------------
# 3) 원가·수익성 분석: 매입단가 추적 / 원가율
# ------------------------------------------------------------------
class PriceChangeRequest(BaseModel):
    receipts: List[Dict[str, Any]]
    items: List[Dict[str, Any]]
    endDate: Optional[date] = None
    windowWeeks: int = 4
    minChangePct: float = 10.0


@router.post("/api/v1/analytics/price-changes")
async def price_changes(body: PriceChangeRequest) -> List[Dict[str, Any]]:
    receipts_df = _receipts_to_df(body.receipts)
    items_df = _items_to_df(body.items)
    result = detect_price_changes(
        receipts_df, items_df,
        window_weeks=body.windowWeeks, min_change_pct=body.minChangePct, end_date=body.endDate,
    )
    if result.empty:
        return []
    return result.to_dict(orient="records")


class CostRateRequest(BaseModel):
    receipts: List[Dict[str, Any]]
    items: List[Dict[str, Any]]
    products: List[Dict[str, Any]]
    endDate: Optional[date] = None


@router.post("/api/v1/analytics/cost-rates")
async def cost_rates(body: CostRateRequest) -> List[Dict[str, Any]]:
    receipts_df = _receipts_to_df(body.receipts)
    items_df = _items_to_df(body.items)
    products_df = _products_to_df(body.products)
    result = calculate_cost_rates(receipts_df, items_df, products_df, end_date=body.endDate)
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")


# ------------------------------------------------------------------
# 4) 통합 HTML 리포트
# ------------------------------------------------------------------
class ReportRequest(BaseModel):
    receipts: List[Dict[str, Any]]
    budgets: List[Dict[str, Any]]
    items: List[Dict[str, Any]]
    products: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    storeName: str = "매장"
    reportType: str = "full"  # monthly | yearly | full
    year: Optional[int] = None
    month: Optional[int] = None


@router.post("/api/v1/analytics/report", response_class=HTMLResponse)
async def analytics_report(body: ReportRequest) -> str:
    print(
        f"[report] 요청 수신 - receipts={len(body.receipts)}건, budgets={len(body.budgets)}건, "
        f"items={len(body.items)}건, products={len(body.products)}건, orders={len(body.orders)}건"
    )
    receipts_df = _receipts_to_df(body.receipts)
    budget_df = _budgets_to_df(body.budgets)
    items_df = _items_to_df(body.items)
    products_df = _products_to_df(body.products)
    orders_df = _orders_to_df(body.orders)

    try:
        html = build_html_from_frames(
            receipts_df, budget_df, items_df, products_df, orders_df,
            store_name=body.storeName, report_type=body.reportType,
            year=body.year, month=body.month,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return html


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
