# BP20-AI

## FastAPI 서버 실행

프로젝트 폴더에서 가상환경을 생성하고 패키지를 설치한다.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env.example`을 복사한 뒤 `.env`에 실제 OpenAI API 키를 입력한다.

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=실제_API_키
```

에이전트 실행에 필요한 데이터와 모델이 다음 위치에 있어야 한다.

```text
data/agent/
├─ trend_panel.csv
├─ campaign_logs.csv
└─ neighbor_sales_quarterly.csv

model/
├─ cox_risk.pkl
├─ bandit/
└─ rag_index/export/
```

FastAPI 서버를 실행한다.

```powershell
python -m uvicorn app.main:app --reload
```

서버 주소:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## 에이전트 실행

서버를 실행한 상태에서 새로운 PowerShell 창을 연다.

### 1. 에이전트 시작

아래 예시는 강남역 상권의 커피-음료 업종을 2026년 1분기 기준으로 분석한다.

```powershell
$body = @{
    trdar_cd = "3120189"
    svc_induty_cd = "CS100010"
    yyqu_cd = 20261
} | ConvertTo-Json

$run = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/agent-runs" `
    -ContentType "application/json" `
    -Body $body

$run
```

실행 결과에서 `thread_id`와 추천 방안을 확인한다.

```powershell
$run.thread_id
$run.selected_action
$run.대기중_승인
```

### 2. 실행 상태 조회

```powershell
$threadId = $run.thread_id

Invoke-RestMethod `
    -Method Get `
    -Uri "http://127.0.0.1:8000/api/v1/agent-runs/$threadId"
```

### 3. 추천 방안 승인 및 최종 보고서 생성

```powershell
$approval = @{
    결정 = "approve"
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/agent-runs/$threadId/resume" `
    -ContentType "application/json" `
    -Body $approval

$result
```

최종 보고서와 검증 결과를 확인한다.

```powershell
$result.final_report.report
$result.final_report.verified
```

### 4. 추천 방안 반려

```powershell
$rejection = @{
    결정 = "reject"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/agent-runs/$threadId/resume" `
    -ContentType "application/json" `
    -Body $rejection
```
