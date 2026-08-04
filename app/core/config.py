import os

# 경로 상수 — scripts/modeling/sales_analysis.py의 ROOT/DATA/MODEL 정의와 동일하게 유지
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"
UPLOAD_DATA = DATA / "uploads"  # API와 워커가 공유하는 업로드 저장소

AGENT_RUNS_DB = MODEL / "agent_runs.sqlite3"
# 신규 가맹점 영업 타겟 추천 그래프(app/sales_target/graph.py) 체크포인트.
# SALES_TARGET_CHECKPOINT_DB_URL이 설정돼 있으면 PostgresSaver, 없으면 이 sqlite 파일로 폴백한다
# (AI_Agent_전환_가이드라인.md 2.3절 — 로컬 개발은 sqlite, Celery 다중 워커 운영은 Postgres 권장).
SALES_TARGET_GRAPH_DB = MODEL / "sales_target_graph.sqlite3"
SALES_TARGET_CHECKPOINT_DB_URL = os.getenv("SALES_TARGET_CHECKPOINT_DB_URL", "")
# 4단계(운영 정리 정책) — 완료/반려된 지 이만큼 지난 thread의 체크포인트는 지운다.
# app/sales_target/cleanup.py, app/tasks/sales_target.py에서 사용.
SALES_TARGET_CHECKPOINT_RETENTION_DAYS = int(os.getenv("SALES_TARGET_CHECKPOINT_RETENTION_DAYS", "30"))
ANALYSES_DB = MODEL / "analyses.sqlite3"
JOBS_DB = MODEL / "analysis_jobs.sqlite3"  # 비동기 작업 상태 저장소
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
SEOUL_WEATHER_DAILY = SOURCE_DATA / "weather_seoul_daily_recent.csv"
SEOUL_EVENT_EXPOSURE = PROCESSED_DATA / "event_exposure_quarterly.csv"
SEOUL_SUBWAY_EXPOSURE = PROCESSED_DATA / "subway_exposure_quarterly.csv"

# 신규 가맹점 영업 타겟 추천
STORE_REGISTRY = SOURCE_DATA / "store_registry.csv"
ADONG_CODES = SOURCE_DATA / "adong_codes.csv"  # 행정표준코드관리시스템(code.go.kr)에서 내려받아 수동 배치

class ABSASettings:    
    MODEL_PATH: str = os.getenv("MODEL_PATH", "thadus2/roberta-absa-best-4class")

    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    ASPECTS: list = ["food", "service", "convenience", "price", "atmosphere"]
    LABEL_MAP: dict = {0: "부정", 1: "중립", 2: "긍정", 3: "none"}
    MAX_LENGTH: int = 128

settings = ABSASettings()