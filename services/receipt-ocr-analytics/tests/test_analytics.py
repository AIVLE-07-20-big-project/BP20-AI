from fastapi.testclient import TestClient

from main import app


def test_expense_anomalies_basic():
    client = TestClient(app)
    payload = {
        "receipts": [
            {"receiptId": 1, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-06-01",
             "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
             "supplyAmount": 200000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 200000},
            {"receiptId": 2, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-06-08",
             "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
             "supplyAmount": 205000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 205000},
            {"receiptId": 3, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-06-15",
             "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
             "supplyAmount": 210000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 210000},
            {"receiptId": 4, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-06-22",
             "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
             "supplyAmount": 500000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 500000},
        ],
        "zThreshold": 1.0,
    }

    response = client.post("/api/v1/analytics/expense-anomalies", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    # 급격히 튀는 마지막 주(50만원)가 이상 지출로 탐지되어야 한다.
    assert any(row["direction"] == "급증" for row in body)


def test_expense_anomalies_empty_receipts_returns_empty_list():
    client = TestClient(app)
    response = client.post("/api/v1/analytics/expense-anomalies", json={"receipts": [], "zThreshold": 1.3})

    assert response.status_code == 200
    assert response.json() == []


def test_budget_overage_basic():
    client = TestClient(app)
    payload = {
        "receipts": [
            {"receiptId": 1, "storeId": 1, "vendorName": "카페기기수리센터", "transactionDate": "2026-06-20",
             "transactionTime": "15:30", "paymentMethod": "카드", "category": "유지비",
             "supplyAmount": 220000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 220000},
        ],
        "budgets": [
            {"yearMonth": "2026-06", "category": "유지비", "budgetAmount": 100000, "storeId": 1},
        ],
    }

    response = client.post("/api/v1/analytics/budget-overage", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["overAmount"] == 120000
    assert body[0]["overPct"] == 120.0


def test_cost_rates_basic():
    client = TestClient(app)
    payload = {
        "receipts": [
            {"receiptId": 1, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-06-01",
             "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
             "supplyAmount": 240000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 240000},
        ],
        "items": [
            {"receiptItemId": 1, "receiptId": 1, "itemName": "원두(생두)",
             "quantity": 10, "unit": "kg", "unitPrice": 24000, "totalPrice": 240000},
        ],
        "products": [
            {"productId": 1, "storeId": 1, "productName": "아메리카노", "category": "음료",
             "price": 4000, "discountRate": 0},
        ],
    }

    response = client.post("/api/v1/analytics/cost-rates", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["productName"] == "아메리카노"
    assert body[0]["costRatePct"] == 10.8
