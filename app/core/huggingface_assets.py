"""Hugging Face Hub의 프로젝트 모델·데이터 아티팩트 동기화."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from app.core.config import DATA, MODEL, RAG_INDEX_EXPORT, settings


def _missing_model_assets() -> bool:
    required = (
        MODEL / "ai_sales_model.pkl",
        MODEL / "cox_risk.pkl",
        RAG_INDEX_EXPORT / "embeddings.npy",
        RAG_INDEX_EXPORT / "chunks.jsonl",
        RAG_INDEX_EXPORT / "manifest.json",
    )
    return any(not path.exists() for path in required)


def _missing_data_assets() -> bool:
    required = (
        DATA / "processed" / "merged_sales_analysis.csv",
        DATA / "agent" / "trend_panel.csv",
        DATA / "source" / "store_stats.csv",
    )
    return any(not path.exists() for path in required)


def sync_huggingface_assets(*, force: bool = False) -> dict[str, str]:
    """필요한 아티팩트만 HF에서 내려받아 기존 로컬 경로에 배치한다."""
    if not settings.HF_AUTO_DOWNLOAD_ASSETS:
        return {"status": "disabled"}

    token = settings.ROBERTA_HF_TOKEN or None
    synced: dict[str, str] = {}

    if settings.HF_MODEL_REPO_ID and (force or _missing_model_assets()):
        MODEL.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=settings.HF_MODEL_REPO_ID,
            repo_type="model",
            revision=settings.HF_ASSET_REVISION,
            token=token,
            local_dir=MODEL,
        )
        synced["model"] = settings.HF_MODEL_REPO_ID

    if settings.HF_DATASET_REPO_ID and (force or _missing_data_assets()):
        DATA.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=settings.HF_DATASET_REPO_ID,
            repo_type="dataset",
            revision=settings.HF_ASSET_REVISION,
            token=token,
            local_dir=DATA,
            ignore_patterns=["uploads/*"],
        )
        synced["dataset"] = settings.HF_DATASET_REPO_ID

    return {"status": "synced" if synced else "ready", **synced}
