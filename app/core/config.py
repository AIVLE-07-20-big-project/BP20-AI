"""경로 상수 — scripts/modeling/sales_analysis.py의 ROOT/DATA/MODEL 정의와 동일하게 유지."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"

AGENT_RUNS_DB = MODEL / "agent_runs.sqlite3"
BANDIT_MODEL_DIR = MODEL / "bandit"  # 등급별 하위폴더(model/bandit/{등급}/active.pt)
CAMPAIGN_LOGS = DATA / "campaign_logs.csv"
RAG_INDEX_EXPORT = MODEL / "rag_index" / "export"

SALES_ESTIMATE = DATA / "sales_estimate.csv"
STORE_STATS = DATA / "store_stats.csv"
FOOT_TRAFFIC = DATA / "foot_traffic.csv"
RESIDENT_POPULATION = DATA / "resident_population.csv"
WORKPLACE_POPULATION = DATA / "workplace_population.csv"
WEATHER_QUARTERLY = DATA / "weather_seoul_quarterly.csv"
MERGED_SALES_ANALYSIS = DATA / "merged_sales_analysis.csv"
