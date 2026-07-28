# RAG 검색 모듈 (VSCode / Windows 앱용)
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import RAG_INDEX_EXPORT

TIER_LABEL = {
    "vendor": "플랫폼 자체데이터(실측 아님, 참고용)",
    "academic": "학술 실증",
    "official": "공공통계",
}


# 임베딩 행렬 + 청크 메타를 담은 검색 인덱스
@dataclass
class RagIndex:


    embeddings: np.ndarray
    chunks: list[dict[str, Any]]
    manifest: dict[str, Any]
    _model: Any = field(default=None, repr=False)
    _model_error: str | None = field(default=None, repr=False)

    @classmethod
    def load(cls, export_dir: str | Path) -> "RagIndex":
        d = Path(export_dir)
        emb = np.load(d / "embeddings.npy").astype("float32")
        chunks = [json.loads(line) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))

        if emb.shape[0] != len(chunks):
            raise ValueError(f"임베딩 행 수({emb.shape[0]})와 청크 수({len(chunks)})가 다릅니다. 빌드 산출물을 확인하세요.")
        return cls(embeddings=emb, chunks=chunks, manifest=manifest)

    # 임베딩 모델 지연 로딩(최초 호출 시 1회)
    @property
    def model(self):

        if self._model_error is not None:
            raise RuntimeError(self._model_error)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            allow_download = os.environ.get("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}
            try:
                self._model = SentenceTransformer(
                    self.manifest["model"], device="cpu", local_files_only=not allow_download,
                )
            except Exception as exc:
                self._model_error = f"질의 임베딩 모델을 로드하지 못함: {type(exc).__name__}: {exc}"
                raise RuntimeError(self._model_error) from exc
        return self._model

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True).astype("float32")[0]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[가-힣A-Za-z0-9]+", text.lower()) if len(t) >= 2}

    # Offline fail-safe used when the query embedding model is unavailable
    def _lexical_search(self, query: str, idxs: np.ndarray, k: int) -> list[tuple[float, dict]]:

        query_tokens = self._tokens(query)
        scored = []
        for i in idxs:
            chunk_tokens = self._tokens(self.chunks[i].get("text", ""))
            overlap = len(query_tokens & chunk_tokens)
            score = overlap / max(len(query_tokens), 1)
            scored.append((score, self.chunks[i]))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:k]

    def _mask(self, tier: str | None, axis: str | None, require_stat: bool) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if tier:
            mask &= np.array([c["tier"] == tier for c in self.chunks])
        if axis:
            mask &= np.array([c["axis"] == axis or c.get("axis_extra") == axis for c in self.chunks])
        if require_stat:
            mask &= np.array([c["contains_stat"] for c in self.chunks])
        return mask

    # 필터를 먼저 적용한 뒤 상위 k개를 반환(정확 검색)
    def search(
        self,
        query: str,
        k: int = 5,
        tier: str | None = None,
        axis: str | None = None,
        require_stat: bool = False,
    ) -> list[tuple[float, dict]]:

        idxs = np.where(self._mask(tier, axis, require_stat))[0]
        if idxs.size == 0:
            return []
        try:
            q = self.encode(query)
        except (RuntimeError, OSError):
            return self._lexical_search(query, idxs, k)
        sims = self.embeddings @ q
        order = idxs[np.argsort(-sims[idxs])][:k]
        return [(float(sims[i]), self.chunks[i]) for i in order]

    # 방향성(academic) + 수치(vendor)를 분리 수집
    def build_evidence(
        self,
        query: str,
        axis: str | None = None,
        n_direction: int = 3,
        n_magnitude: int = 3,
        window: int = 1,
    ) -> dict[str, Any]:








        direction = self.search(query, k=n_direction, tier="academic", axis=axis)
        magnitude = self.search(query, k=n_magnitude, tier="vendor", axis=axis, require_stat=True)

        allowed: list[dict[str, Any]] = []
        for _, c in magnitude:
            for sc in extract_stat_contexts(c, window=window):
                allowed.append(
                    {
                        "value": sc["value"],
                        "sentence": sc["sentence"],
                        "focus": sc["focus"],
                        "source_url": c.get("source_url"),
                        "doc_id": c["doc_id"],
                        "page": c["page_start"],
                        "tier_label": TIER_LABEL["vendor"],
                    }
                )
        allowed = dedup_numbers(allowed)

        return {
            "query": query,
            "axis": axis,
            "direction_refs": [
                {
                    "doc_id": c["doc_id"],
                    "page": c["page_start"],
                    "tier_label": TIER_LABEL[c["tier"]],
                    "score": round(s, 3),
                    "text": c["text"][:400],
                }
                for s, c in direction
            ],
            "allowed_numbers": allowed,
            "has_magnitude": bool(allowed),
        }


STAT_RE = re.compile(
    r"[+\-]?\d[\d,]*(?:\.\d+)?\s?(?:%|퍼센트|원)"
    r"|[+\-]?\d[\d,]*(?:\.\d+)?배(?![달란])"
)


def split_sentences(text: str) -> list[str]:
    t = re.sub(r"\s*\n\s*", " ", text)
    parts = re.split(r"(?<=[.!?])\s+|(?<=요\.)\s+|(?<=다\.)\s+", t)
    return [p.strip() for p in parts if p.strip()]


# 숫자에 앞뒤 window개 문장을 묶어 반환(귀속 보존)
def extract_stat_contexts(chunk: dict, window: int = 1) -> list[dict]:

    sents = split_sentences(chunk["text"])
    out, seen = [], set()
    for i, s in enumerate(sents):
        for v in STAT_RE.findall(s):
            if (v, i) in seen:
                continue
            seen.add((v, i))
            lo, hi = max(0, i - window), min(len(sents), i + window + 1)
            out.append({"value": v, "sentence": " ".join(sents[lo:hi]), "focus": s})
    return out


# 같은 값이 여러 청크에서 나오면 가장 긴 맥락 하나만 남긴다
def dedup_numbers(allowed: list[dict]) -> list[dict]:

    best: dict[str, dict] = {}
    for a in allowed:
        cur = best.get(a["value"])
        if cur is None or len(a["sentence"]) > len(cur["sentence"]):
            best[a["value"]] = a
    return list(best.values())


# 앱 전역에서 재사용할 인덱스 싱글턴
@lru_cache(maxsize=1)
def get_index(export_dir: str | Path = RAG_INDEX_EXPORT) -> RagIndex:

    return RagIndex.load(export_dir)


if __name__ == "__main__":
    import sys

    idx = RagIndex.load(sys.argv[1] if len(sys.argv) > 1 else RAG_INDEX_EXPORT)
    print("manifest:", json.dumps(idx.manifest, ensure_ascii=False)[:300])
    ev = idx.build_evidence("세트메뉴가 객단가에 미치는 효과", axis="set_bundle")
    for r in ev["direction_refs"]:
        print(f"  [dir] {r['score']} {r['doc_id']} p{r['page']}")
    for a in ev["allowed_numbers"]:
        print(f"  [num] {a['value']} ← {a['sentence'][:70]}")
