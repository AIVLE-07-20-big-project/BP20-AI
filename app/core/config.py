# 경로 상수 — scripts/modeling/sales_analysis.py의 ROOT/DATA/MODEL 정의와 동일하게 유지
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"
# 비동기 분석에서 API와 워커가 공유하는 업로드 원본 저장소
# (브로커에 원본 바이트를 싣지 않기 위한 것. docs/speed/celery-async-development-plan.md §2)
UPLOAD_DATA = DATA / "uploads"

AGENT_RUNS_DB = MODEL / "agent_runs.sqlite3"
ANALYSES_DB = MODEL / "analyses.sqlite3"
# 비동기 분석 잡 상태의 단일 소스(Celery result backend는 안 씀 — 1시간 후 만료되고
# 소유권을 표현 못 함). docs/speed/celery-async-development-plan.md §2.2
JOBS_DB = MODEL / "analysis_jobs.sqlite3"
BANDIT_MODEL_DIR = MODEL / "bandit"
CAMPAIGN_LOGS = AGENT_DATA / "campaign_logs.csv"
RAG_INDEX_EXPORT = MODEL / "rag_index" / "export"

SALES_ESTIMATE = SOURCE_DATA / "sales_estimate.csv"
STORE_STATS = SOURCE_DATA / "store_stats.csv"
FOOT_TRAFFIC = SOURCE_DATA / "foot_traffic.csv"
RESIDENT_POPULATION = SOURCE_DATA / "resident_population.csv"
WORKPLACE_POPULATION = SOURCE_DATA / "workplace_population.csv"
WEATHER_QUARTERLY = SOURCE_DATA / "weather_seoul_quarterly.csv"
MERGED_SALES_ANALYSIS = PROCESSED_DATA / "merged_sales_analysis.csv"

SEOUL_WEATHER_MONTHLY = SOURCE_DATA / "weather_seoul_monthly_raw.csv"
SEOUL_EVENT_EXPOSURE = PROCESSED_DATA / "event_exposure_quarterly.csv"
SEOUL_SUBWAY_EXPOSURE = PROCESSED_DATA / "subway_exposure_quarterly.csv"
