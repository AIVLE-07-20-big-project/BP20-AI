"""Hugging Face Hub의 프로젝트 모델·데이터 아티팩트 동기화."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from app.core.config import (
    DATA,
    DATA_S3_PREFIX,
    MODEL,
    MODEL_S3_PREFIX,
    RAG_INDEX_EXPORT,
    RAG_S3_PREFIX,
    settings,
)
from app.core.object_storage import download_file, enabled as s3_enabled


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


def _sync_s3_assets(*, force: bool) -> dict[str, str]:
    """S3를 운영 자산 저장소로 사용할 때 필요한 파일만 내려받는다."""
    model_assets = {
        MODEL / "ai_sales_model.pkl": (MODEL_S3_PREFIX, "ai_sales_model.pkl"),
        MODEL / "cox_risk.pkl": (MODEL_S3_PREFIX, "cox_risk.pkl"),
        RAG_INDEX_EXPORT / "embeddings.npy": (RAG_S3_PREFIX, "export/embeddings.npy"),
        RAG_INDEX_EXPORT / "chunks.jsonl": (RAG_S3_PREFIX, "export/chunks.jsonl"),
        RAG_INDEX_EXPORT / "manifest.json": (RAG_S3_PREFIX, "export/manifest.json"),
    }
    data_assets = {
        DATA / "processed" / "merged_sales_analysis.csv": (DATA_S3_PREFIX, "processed/merged_sales_analysis.csv"),
        DATA / "agent" / "trend_panel.csv": (DATA_S3_PREFIX, "agent/trend_panel.csv"),
        DATA / "source" / "store_stats.csv": (DATA_S3_PREFIX, "source/store_stats.csv"),
    }
    synced: dict[str, str] = {}
    for local_path, (prefix, name) in {**model_assets, **data_assets}.items():
        if force or not local_path.exists():
            download_file(prefix=prefix, name=name, local_path=local_path)
            synced[str(local_path)] = f"s3://{prefix}/{name}"
    return synced


def sync_huggingface_assets(*, force: bool = False) -> dict[str, str]:
    """필요한 아티팩트만 HF에서 내려받아 기존 로컬 경로에 배치한다."""
    if s3_enabled():
        synced = _sync_s3_assets(force=force)
        return {"status": "synced" if synced else "ready", "source": "s3", **synced}
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
