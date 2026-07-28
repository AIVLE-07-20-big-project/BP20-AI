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
