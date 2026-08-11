import os

# 경로 상수 — scripts/modeling/sales_analysis.py의 ROOT/DATA/MODEL 정의와 동일하게 유지
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"
UPLOAD_DATA = DATA / "uploads"  # API와 워커가 공유하는 업로드 저장소
# 운영에서는 MySQL URL을 주입하고, 비워두면 개발용 SQLite를 사용한다.
AI_DATABASE_URL = os.getenv("AI_DATABASE_URL", "")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "")
MODEL_S3_PREFIX = os.getenv("MODEL_S3_PREFIX", "models")
RAG_S3_PREFIX = os.getenv("RAG_S3_PREFIX", "rag")
DATA_S3_PREFIX = os.getenv("DATA_S3_PREFIX", "data")
FEEDBACK_S3_PREFIX = os.getenv("FEEDBACK_S3_PREFIX", "feedback/v1")
BANDIT_S3_PREFIX = os.getenv("BANDIT_S3_PREFIX", "bandit/v1")
UPLOAD_S3_PREFIX = os.getenv("UPLOAD_S3_PREFIX", "uploads/v1")
CAMPAIGN_LOGS_S3_PREFIX = os.getenv("CAMPAIGN_LOGS_S3_PREFIX", "logs/v2")

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
# v2 캠페인 로그: 추천 실행 결과·OPE·Bandit 학습의 단일 운영 로그
CAMPAIGN_LOGS_V2 = AGENT_DATA / "campaign_logs_v2.csv"
# RAG 산출물은 Hugging Face 모델 저장소를 /app/model에 동기화할 때
# 함께 내려받을 수 있도록 model 아래를 canonical 경로로 사용한다.
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
SEOUL_EVENT_DETAILS = PROCESSED_DATA / "event_details_quarterly.csv"
SEOUL_SUBWAY_EXPOSURE = PROCESSED_DATA / "subway_exposure_quarterly.csv"

# 신규 가맹점 영업 타겟 추천
STORE_REGISTRY = SOURCE_DATA / "store_registry.csv"
ADONG_CODES = SOURCE_DATA / "adong_codes.csv"  # 행정표준코드관리시스템(code.go.kr)에서 내려받아 수동 배치


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# Bandit v2 feature flag(계획 §6) — 매장 rollout은 범위 밖이라 기본값은 전부 보수적으로 둔다
BANDIT_POLICY_V2_ENABLED = _bool_env("BANDIT_POLICY_V2_ENABLED", False)
BANDIT_V2_SHADOW_ENABLED = _bool_env("BANDIT_V2_SHADOW_ENABLED", True)
BANDIT_EXPERIMENT_ENABLED = _bool_env("BANDIT_EXPERIMENT_ENABLED", False)
BANDIT_EXPERIMENT_EPSILON = float(os.environ.get("BANDIT_EXPERIMENT_EPSILON", "0.10"))
class ABSASettings:
    ROBERTA_MODEL_PATH: str = os.getenv("ROBERTA_MODEL_PATH", "thadus2/roberta-absa-best-4class")
    ROBERTA_S3_DIRECTORY: str = os.getenv("ROBERTA_S3_DIRECTORY", "roberta_absa").strip("/")

    ROBERTA_HF_TOKEN: str = os.getenv("ROBERTA_HF_TOKEN", "")
    HF_MODEL_REPO_ID: str = os.getenv("HF_MODEL_REPO_ID", "")
    HF_DATASET_REPO_ID: str = os.getenv("HF_DATASET_REPO_ID", "")
    HF_ASSET_REVISION: str = os.getenv("HF_ASSET_REVISION", "main")
    HF_AUTO_DOWNLOAD_ASSETS: bool = _bool_env("HF_AUTO_DOWNLOAD_ASSETS", True)

    ASPECTS: list = ["food", "service", "convenience", "price", "atmosphere"]
    LABEL_MAP: dict = {0: "부정", 1: "중립", 2: "긍정", 3: "none"}
    MAX_LENGTH: int = 128

settings = ABSASettings()
