# 영수증 OCR · AI 가계부 · 원가분석 서비스

> 이 폴더(`services/receipt-ocr-analytics/`)는 BP20-AI 레포 안의 여러 독립 FastAPI 서비스 중 하나입니다.
> 서비스마다 폴더가 분리되어 있고, 각자 자기만의 `requirements.txt`/`Dockerfile`/`tests`를 가집니다
> (다른 서비스와 의존성이 섞이지 않도록). 실행/테스트는 전부 **이 폴더 안에서** 하시면 됩니다.

영수증 이미지를 OCR로 읽어 구조화하고, 지출/매출 데이터를 바탕으로 AI 가계부(이상지출 탐지·예산 초과 확인)와
원가·수익성 분석(매입단가 추적·원가율 계산)을 수행하는 FastAPI 기반 계산 전용 마이크로서비스입니다.


## 이 서비스가 하는 일 / 하지 않는 일

- **한다**: 이미지 인식(OCR), 통계 계산(이상탐지·예산비교·단가추적·원가율), HTML 리포트 생성
- **안 한다**: 데이터베이스 저장, 인증/권한, 사용자 관리 — 이 서비스는 상태를 갖지 않습니다(stateless). 매 요청마다 필요한 데이터를 전부 함께 전달받아 계산만 하고 돌려줍니다.

**실제 데이터 저장과 서비스 로직은 Java 백엔드(BP20-BE)가 담당하고, 이 서비스는 Java가 호출하는 "계산 전용 서버"입니다.**

```
Spring Boot(Java, MySQL) ──HTTP──▶ FastAPI(Python, 이 레포)
   실제 데이터 저장/인증/권한          계산 전용, 상태 없음(stateless)
```

---

## 핵심 기능 4가지

### 1. 영수증 OCR 자동 정산
영수증 이미지를 업로드하면 다음 과정을 거쳐 구조화된 데이터를 반환합니다.

1. 이미지 전처리 (업스케일, 대비 향상, 노이즈 제거 — OpenCV)
2. PaddleOCR로 텍스트 인식 (한글 인식 모델, 서버 시작 시 1회만 로드해 재사용)
3. 문서 유형 자동 판별 (영수증/세금계산서/거래명세서 등 키워드 매칭)
4. 좌표(bbox) 기반으로 상호명·품목·수량·단가·금액 추출 (행 단위 클러스터링)
5. 검증: 수량×단가=금액, 공급가액+부가세+면세금액=총액 등 사칙연산으로 정합성 확인
6. 지출 카테고리 자동 분류 (키워드 매핑: 식재료비/포장재비/공과금 등)

> AI 모델은 PaddleOCR 하나만 사용합니다. 나머지(문서분류/추출/검증/분류)는 전부 규칙 기반 알고리즘이며, 별도의 파인튜닝이나 하이퍼파라미터 튜닝은 하지 않았습니다 (사전학습 모델을 그대로 사용).

### 2. AI 가계부
- **이상 지출 탐지**: 카테고리별 주간 지출을 Z-score로 계산해, 평소 평균 대비 통계적으로 튀는 주를 탐지
- **예산 초과 확인**: 월별·카테고리별 실제 지출과 예산 목표치를 비교

### 3. 원가·수익성 분석
- **매입 단가 변화 추적**: 최근 4주 평균 단가를 이전 4주와 비교 (기준일을 특정 시점으로 지정 가능 — 월간/연간 리포트에서 미래 데이터가 섞이지 않도록)
- **메뉴별 원가율**: 레시피(재료 사용량) × 최근 매입단가로 원가를 계산해 원가율 산출. 레시피/매입 데이터가 없는 메뉴는 "계산 불가"로 정직하게 표시 (임의로 채우지 않음)

### 4. 통합 HTML 리포트
위 세 기능의 결과를 하나의 HTML로 조립합니다. 기간을 3가지 중 선택:

| 유형 | 포함 내용 |
|---|---|
| 월간 | 이상지출/예산초과/매입단가/원가율 4개 섹션만 |
| 연간 | 위 4개 + 월별 매출·지출 그래프 + 월별 매입단가 지수 그래프 |
| 총기간 | 연간 항목 전부 + 분기별 매출·지출 그래프 |

그래프는 외부 라이브러리 없이 순수 SVG로 직접 그립니다 (매출·지출 막대+꺾은선 콤보 차트, 여러 원재료 품목의 단가 변동을 한 그래프에서 비교하는 지수 차트).

---

## API 엔드포인트

베이스 URL: `http://localhost:8001` (기본값)

| Method | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| POST | `/api/v1/receipts/parse` | 영수증 이미지 업로드 → OCR 구조화 결과 반환 (multipart/form-data, `file` 필드) |
| POST | `/api/v1/analytics/expense-anomalies` | 이상 지출 탐지 |
| POST | `/api/v1/analytics/budget-overage` | 예산 초과 확인 |
| POST | `/api/v1/analytics/price-changes` | 매입 단가 변화 추적 |
| POST | `/api/v1/analytics/cost-rates` | 메뉴별 원가율 계산 |
| POST | `/api/v1/analytics/report` | 통합 HTML 리포트 생성 (월간/연간/총기간) |

`/docs`에 접속하면 Swagger UI에서 전체 스펙을 확인하고 바로 테스트해볼 수 있습니다.

### 요청/응답 예시 — 매입 단가 변화 추적

```
POST /api/v1/analytics/price-changes
Content-Type: application/json

{
  "receipts": [
    {"receiptId": 1, "storeId": 1, "vendorName": "가온원두상회", "transactionDate": "2026-05-01",
     "transactionTime": "09:15", "paymentMethod": "카드", "category": "식재료비",
     "supplyAmount": 200000, "vat": 0, "taxFreeAmount": 0, "totalAmount": 200000}
  ],
  "items": [
    {"receiptItemId": 1, "receiptId": 1, "itemName": "원두(생두)",
     "quantity": 10, "unit": "kg", "unitPrice": 20000, "totalPrice": 200000}
  ],
  "endDate": "2026-06-01"
}
```
```json
[
  {"itemName": "원두(생두)", "previousAvgPrice": 20000, "recentAvgPrice": 24000, "changePct": 20.0}
]
```

모든 분석 엔드포인트는 이런 식으로 **Java가 DB에서 조회한 데이터를 그대로 실어 보내면, 계산 결과만 돌려주는 구조**입니다. Java 쪽 필드명(camelCase)과 이 서비스의 요청/응답 필드명이 1:1로 대응합니다.

---

## 기술 스택

| 용도 | 사용 기술 |
|---|---|
| 웹 프레임워크 | FastAPI + uvicorn |
| OCR | PaddleOCR (`PP-OCRv5_mobile_det` + `korean_PP-OCRv5_mobile_rec`, 사전학습 모델) |
| 이미지 전처리 | OpenCV |
| 데이터 계산 | pandas |
| 리포트 시각화 | 순수 SVG (외부 차트 라이브러리 미사용) |

---

## 실행 방법

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

서버가 시작되면 콘솔에 `PaddleOCR 모델 예열 중...` → `PaddleOCR 모델 예열 완료.` 가 뜹니다 (최초 1회, 이후 요청은 이 로드된 모델을 재사용하므로 빠릅니다).

`http://localhost:8001/health` 에서 `{"status":"ok"}` 확인되면 정상입니다.

### Docker로 실행

```bash
docker build -t receipt-ocr-service .
docker run -p 8000:8000 receipt-ocr-service
```

---

## Java 백엔드(BP20-BE)와의 연동

BP20-BE의 `application.yml`에 아래처럼 이 서비스의 주소를 지정해두면, `OcrServiceClient`가 이 서비스를 호출합니다.

```yaml
ocr-service:
  base-url: http://localhost:8001   # 이 서비스를 띄운 주소
```

로컬에서 함께 개발할 때는 이 서비스와 BP20-BE(Spring Boot)를 각각 별도 프로세스로 띄워야 합니다.

---

## 알려진 한계

- 원가율 계산에 필요한 "레시피(메뉴 하나에 원재료가 얼마나 들어가는지)" 데이터가 지금은 코드에 하드코딩되어 있습니다. 실제 서비스라면 별도 입력 데이터로 분리되어야 합니다.
- PaddleOCR은 사전학습 모델을 그대로 사용하며, 별도의 성능평가·파인튜닝은 진행하지 않았습니다.
- 이미지가 심하게 기울어지거나(원근 왜곡) 그림자가 심한 경우 인식률이 떨어질 수 있습니다 (자동 기울기 보정·자동 크롭 등은 아직 미구현).

---

## 폴더 구조

```
.
├── main.py                FastAPI 앱 - API 엔드포인트 정의
├── receipt_pipeline.py    OCR 전처리 + PaddleOCR 호출 + 추출/검증 로직
├── expense_analysis.py    이상지출 탐지 / 예산초과 확인 계산 로직
├── cost_analysis.py       매입단가 추적 / 원가율 계산 로직
├── build_report.py        HTML 통합 리포트 생성 (월간/연간/총기간 + SVG 그래프)
├── csv_store.py           로컬 CLI 테스트용 CSV 저장 유틸 (서비스 자체에는 미사용)
├── requirements.txt       Python 의존성
└── Dockerfile             컨테이너 빌드 설정
```
