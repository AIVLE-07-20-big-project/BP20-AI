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

# action 태깅용 문자열 후보 — doc_id/source_url 문자열 매치만으로 action-특화 여부를 판정한다.
# 매핑이 없는 action은 매칭하지 않는다("즉시할인"은 URL 카테고리명과 우연히 겹칠 뿐이라 제외)
ACTION_NAME_SYNONYMS: dict[str, tuple[str, ...]] = {
    "쿠폰발행": ("쿠폰발행", "쿠폰"),
    "타임세일": ("타임세일",),
    "세트메뉴 도입": ("세트메뉴",),
    "사이드메뉴 추가": ("사이드메뉴",),
    "배달채널 확대": ("배달채널", "배달"),
    "매장 리뉴얼": ("매장 리뉴얼", "리뉴얼"),
    "신메뉴 출시": ("신메뉴",),
    "웰컴 프로모션": ("웰컴",),
    "리뷰 관리 캠페인": ("리뷰",),
    "브랜드 SNS 캠페인": ("SNS",),
    "지역 제휴 마케팅": ("제휴",),
}


def _action_tag_matches(chunk: dict, action: str) -> bool:
    candidates = ACTION_NAME_SYNONYMS.get(action, ())
    if not candidates:
        return False
    haystack = f"{chunk.get('source_url') or ''} {chunk.get('doc_id') or ''}"
    return any(name in haystack for name in candidates)


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
            mask &= np.array([
                c["axis"] == axis or c.get("axis_extra") == axis
                or axis in (c.get("axis_extra") or [])
                for c in self.chunks
            ])
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

    # 방향성(academic) + 수치(vendor)를 분리 수집. action이 주어지면 action-특화 자료를 우선한다
    def build_evidence(
        self,
        query: str,
        axis: str | None = None,
        action: str | None = None,
        n_direction: int = 3,
        n_magnitude: int = 3,
        window: int = 1,
    ) -> dict[str, Any]:
        direction = self.search(query, k=n_direction, tier="academic", axis=axis)
        pool = self.search(query, k=max(n_magnitude * 4, 20), tier="vendor", axis=axis, require_stat=True)

        if action:
            specific = [(s, c) for s, c in pool if _action_tag_matches(c, action)]
            generic = [(s, c) for s, c in pool if not _action_tag_matches(c, action)]
            magnitude = specific[:n_magnitude]
            if len(magnitude) < n_magnitude:
                magnitude += generic[: n_magnitude - len(magnitude)]
            specific_ids = {id(c) for _, c in specific[:n_magnitude]}
            action_specific_available = bool(specific)
        else:
            magnitude = pool[:n_magnitude]
            specific_ids = {id(c) for _, c in magnitude}
            action_specific_available = None

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
                        "action_specific": id(c) in specific_ids,
                    }
                )
        allowed = dedup_numbers(allowed)

        return {
            "query": query,
            "axis": axis,
            "action": action,
            "action_specific_available": action_specific_available,
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

    # axis별 vendor(수치) 자료 중 action-특화 태깅이 전혀 안 되는 축을 찾는다
    # (우선 보강 대상 파악용, 설계 §10)
    def axis_coverage_report(self) -> dict[str, dict]:
        axes = {c["axis"] for c in self.chunks if c["tier"] == "vendor"}
        report = {}
        for axis in axes:
            vendor_chunks = [
                c for c in self.chunks if c["tier"] == "vendor" and c["axis"] == axis and c["contains_stat"]
            ]
            specific_count = sum(
                1 for c in vendor_chunks
                if any(_action_tag_matches(c, action) for action in ACTION_NAME_SYNONYMS)
            )
            report[axis] = {
                "총_수치자료": len(vendor_chunks),
                "action_특화_수치자료": specific_count,
                "전부_axis_공용": specific_count == 0 and len(vendor_chunks) > 0,
            }
        return report


# 허용 수치 추출(build_evidence)과 생성문 검증(generator.verify_output)이 공유하는
# 단일 정규식 — 비율·통화·배수에 더해 자주 날조되는 수량 단위(명·개월·개·건·주·일·회)까지
# 포착한다. 두 쪽이 갈라지면 "추출은 되는데 검증 못 하는" 구멍이 생기므로 여기서만 정의한다.
STAT_RE = re.compile(
    r"[+\-]?\d[\d,]*(?:\.\d+)?\s?(?:%|퍼센트|원|명|개월|개|건|주|일|회)"
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


# 같은 값이 여러 청크에서 나오면 action-특화를 우선하고, 동률이면 가장 긴 맥락을 남긴다
def dedup_numbers(allowed: list[dict]) -> list[dict]:

    best: dict[str, dict] = {}
    for a in allowed:
        cur = best.get(a["value"])
        if cur is None:
            best[a["value"]] = a
            continue
        a_specific = a.get("action_specific", True)
        cur_specific = cur.get("action_specific", True)
        if a_specific and not cur_specific:
            best[a["value"]] = a
        elif a_specific == cur_specific and len(a["sentence"]) > len(cur["sentence"]):
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
