import os

# 경로 상수 — scripts/modeling/sales_analysis.py의 ROOT/DATA/MODEL 정의와 동일하게 유지
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODEL = ROOT / "model"
AGENT_DATA = DATA / "agent"
PROCESSED_DATA = DATA / "processed"
SOURCE_DATA = DATA / "source"

AGENT_RUNS_DB = MODEL / "agent_runs.sqlite3"
ANALYSES_DB = MODEL / "analyses.sqlite3"
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

class ABSASettings:    
    MODEL_PATH: str = str(ROOT / "roberta_absa_best_4class")
    ASPECTS: list = ["food", "service", "convenience", "price", "atmosphere"]
    LABEL_MAP: dict = {0: "부정", 1: "중립", 2: "긍정", 3: "none"}
    MAX_LENGTH: int = 128

settings = ABSASettings()