"""
영수증 OCR + AI 가계부 + 원가분석 마이크로서비스 (FastAPI).

Java 백엔드(Spring Boot)는 실제 데이터(Receipt/ReceiptItem/Product/Budget/Order)를
MySQL에 저장/관리하고, 이 서비스는 "계산 전용" 서버로 동작한다:

    1) 영수증 이미지를 보내면 OCR로 구조화된 데이터를 뽑아서 돌려주고
       (Java가 이 결과를 검토/보정한 뒤 자기 DB에 저장)
    2) Java가 자기 DB에서 조회한 데이터(JSON)를 보내면,
       이상지출/예산초과/매입단가/원가율/HTML리포트를 계산해서 돌려준다.

이 서비스 자체는 상태를 갖지 않는다(stateless) - 모든 데이터는 매 요청마다 함께 전달받는다.

실행:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Java 쪽에서는 application.yml에 이 서비스 base url을 설정해두고
WebClient/RestClient로 호출하면 된다 (예: ocr-service.base-url=http://localhost:8000).
"""

import os
import shutil
import tempfile
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from receipt_pipeline import (
    _get_ocr_engine,
    classify_document_type,
    extract_items,
    extract_store_name,
    native_nlp_parser,
    preprocess_and_ocr,
    validate_and_reflect,
)
from expense_analysis import check_budget_overage, detect_expense_anomalies
from cost_analysis import calculate_cost_rates, detect_price_changes
from build_report import build_html_from_frames

app = FastAPI(
    title="온기카페 OCR/가계부/원가분석 서비스",
    description="Java 백엔드에서 호출하는 계산 전용 마이크로서비스",
    version="1.0.0",
)


@app.on_event("startup")
async def warm_up_ocr_engine() -> None:
    """
    PaddleOCR 모델 로딩은 비용이 커서(수십 초~분 단위), 첫 사용자 요청 때 지연되지 않도록
    서버가 뜰 때 미리 한 번 로드해둔다. (이후 모든 요청은 이 로드된 엔진을 재사용)
    """
    print("서버 시작 - PaddleOCR 모델 예열 중...")
    _get_ocr_engine()
    print("PaddleOCR 모델 예열 완료.")


# ------------------------------------------------------------------
# 공통 변환 헬퍼: Java가 보내는 camelCase JSON <-> 기존 분석 코드의 PascalCase DataFrame
# ------------------------------------------------------------------
def _receipts_to_df(receipts: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(receipts)
    if df.empty:
        return pd.DataFrame(columns=[
            "ReceiptID", "StoreID", "VendorName", "TransactionDate", "TransactionTime",
            "PaymentMethod", "Category", "SupplyAmount", "Vat", "TaxFreeAmount", "TotalAmount",
        ])
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
@app.post("/api/v1/receipts/parse")
async def parse_receipt(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    영수증 이미지를 업로드하면 OCR + 좌표기반 추출 + 검증까지 마친
    구조화된 데이터를 반환한다. (DB 저장은 하지 않음 - Java가 검토 후 저장)
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
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


@app.post("/api/v1/analytics/expense-anomalies")
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


@app.post("/api/v1/analytics/budget-overage")
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


@app.post("/api/v1/analytics/price-changes")
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


@app.post("/api/v1/analytics/cost-rates")
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


@app.post("/api/v1/analytics/report", response_class=HTMLResponse)
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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}