# 영수증 OCR · AI 가계부 · 원가분석 모듈

> 이 폴더(`app/ocr/`)는 BP20-AI FastAPI 앱에 포함되는 OCR 전용 모듈입니다.
> 고객 대응 기능과 코드는 분리하지만 `app.main:app`에서 함께 실행합니다.

영수증 이미지를 OCR로 읽어 구조화하고, 지출/매출 데이터를 바탕으로 AI 가계부(이상지출 탐지·예산 초과 확인)와
원가·수익성 분석(매입단가 추적·원가율 계산)을 수행하는 계산 모듈입니다.


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
2. PaddleOCR로 텍스트 인식 (한글 인식 모델, 최초 OCR 요청 시 로드해 재사용)
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

베이스 URL: `http://localhost:8000`

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
pip install -r requirements.txt -r requirements-ocr.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

PaddleOCR 모델은 최초 OCR 요청에서 한 번 로드하며 이후 요청에서 재사용합니다.

`http://localhost:8000/health`에서 `{"status":"ok"}`가 반환되면 정상입니다.

---

## Java 백엔드(BP20-BE)와의 연동

BP20-BE의 `application.yml`에 아래처럼 이 서비스의 주소를 지정해두면, `OcrServiceClient`가 이 서비스를 호출합니다.

```yaml
ocr-service:
  base-url: http://localhost:8000
```

로컬에서는 BP20-AI 통합 앱과 BP20-BE(Spring Boot)를 각각 별도 프로세스로 실행합니다.

---

## 모듈 구성 원칙

- OCR 구현은 `app/ocr/`, 테스트는 `tests/ocr/`에서 관리합니다.
- 실행 앱과 포트는 공유하지만 OCR 전용 의존성은 루트 `requirements-ocr.txt`로 구분합니다.
- OCR Router는 `app/main.py`에서 등록합니다.

---

## 변경 및 고도화 내역

기존 단일 Python 스크립트 기반 OCR 기능을 Spring Boot 백엔드와 연동 가능한 FastAPI 모듈로 변경했습니다.

| 구분 | 기존 구조 | 현재 구조 |
|---|---|---|
| 실행 방식 | `python receipt_pipeline.py` | `uvicorn app.main:app --port 8000` |
| 서비스 포트 | 개별 실행 포트 | 통합 FastAPI 포트 8000 |
| 프로젝트 구조 | 실험용 단일 폴더 | `app/ocr/` 전용 모듈 |
| API 통신 | CLI 및 로컬 이미지 입력 | `multipart/form-data` 요청과 JSON 응답 |
| 상태 확인 | 별도 기능 없음 | `GET /health` 제공 |

### OCR 파이프라인 개선

- 한국어 영수증 인식을 위해 PaddleOCR PP-OCRv5 모델을 적용했습니다.
- 최초 OCR 요청 시 모델을 지연 로드하고 이후 요청에서 재사용합니다.
- 원본 이미지와 점 연결 보정 이미지를 이용한 2단계 텍스트 추출을 적용했습니다.
- 공급가액·부가세·면세금액과 총액의 정합성을 검증합니다.
- 품목별 합계와 총 결제 금액을 비교하고 지출 카테고리를 자동 분류합니다.

### 백엔드 연동

- Spring Boot의 `OCR_SERVICE_BASE_URL`을 `http://localhost:8000`으로 설정합니다.
- Swagger에서 JWT가 전달되도록 백엔드 컨트롤러에 Bearer 인증 설정이 필요합니다.
- `/health` 엔드포인트를 이용해 백엔드에서 OCR 서비스 연결 상태를 확인할 수 있습니다.
- OCR 서비스는 인증을 직접 처리하지 않고 백엔드가 인증한 요청의 계산만 담당합니다.

### API 검증 흐름

1. `POST /api/store-owner/receipts/parse`로 영수증 이미지 파싱 결과를 확인합니다.
2. `/api/store-owner/analytics/expense-anomalies`와 `/api/store-owner/analytics/budget-overage`로 지출 분석 결과를 확인합니다.
3. `/api/store-owner/analytics/report`로 통합 HTML 리포트를 확인합니다.

> 위 경로는 Spring Boot가 외부에 제공하는 API 경로입니다. OCR 서비스 내부 엔드포인트는 이 문서의 `API 엔드포인트` 절을 따릅니다.

---

## 알려진 한계

- 원가율 계산에 필요한 "레시피(메뉴 하나에 원재료가 얼마나 들어가는지)" 데이터가 지금은 코드에 하드코딩되어 있습니다. 실제 서비스라면 별도 입력 데이터로 분리되어야 합니다.
- PaddleOCR은 사전학습 모델을 그대로 사용하며, 별도의 성능평가·파인튜닝은 진행하지 않았습니다.
- 이미지가 심하게 기울어지거나(원근 왜곡) 그림자가 심한 경우 인식률이 떨어질 수 있습니다 (자동 기울기 보정·자동 크롭 등은 아직 미구현).

---

## 폴더 구조

```
app/ocr/
├── router.py              OCR 및 비용 분석 API
├── pipeline.py            OCR 전처리·호출·추출·검증
├── expense_analysis.py    이상 지출 및 예산 분석
├── cost_analysis.py       매입 단가 및 원가율 분석
├── report.py              HTML 통합 리포트 생성
└── csv_store.py           로컬 CSV 저장 유틸

tests/ocr/                 OCR 전용 테스트와 샘플 이미지
requirements-ocr.txt       OCR 런타임 의존성
requirements-ocr-dev.txt   OCR 개발·테스트 의존성
```
