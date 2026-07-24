# RAG 검색 품질 지표(순수 함수) — Recall@k, Hit@k, MRR, NDCG@k.
# 정답 라벨(gold)은 등급 관련도(0 무관 / 1 부분 / 2 직접)로 주고, NDCG는 등급을,
# Recall/Hit/MRR은 relevant_threshold 이상을 '관련'으로 본다.
from __future__ import annotations

import math
from typing import Mapping, Sequence

# 정답에 등급이 하나도 없으면(코퍼스 공백) 지표를 계산하지 않고 None을 반환한다
NO_RELEVANT = None


def _relevant_ids(gold: Mapping[str, int], threshold: int) -> set[str]:
    return {cid for cid, grade in gold.items() if grade >= threshold}


def recall_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int, threshold: int = 1) -> float | None:
    relevant = _relevant_ids(gold, threshold)
    if not relevant:
        return NO_RELEVANT
    hit = sum(1 for cid in ranked[:k] if cid in relevant)
    return round(hit / len(relevant), 4)


def hit_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int, threshold: int = 1) -> float | None:
    relevant = _relevant_ids(gold, threshold)
    if not relevant:
        return NO_RELEVANT
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], gold: Mapping[str, int], threshold: int = 1) -> float | None:
    relevant = _relevant_ids(gold, threshold)
    if not relevant:
        return NO_RELEVANT
    for rank, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float | None:
    if not any(grade > 0 for grade in gold.values()):
        return NO_RELEVANT
    gains = [gold.get(cid, 0) for cid in ranked[:k]]
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return NO_RELEVANT
    return round(_dcg(gains) / idcg, 4)


# 한 질의의 랭킹에 대해 모든 지표를 한 번에 계산한다
def score_ranking(
    ranked: Sequence[str], gold: Mapping[str, int], ks: Sequence[int] = (3, 5, 10), threshold: int = 1,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {"MRR": mrr(ranked, gold, threshold)}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(ranked, gold, k, threshold)
        out[f"Hit@{k}"] = hit_at_k(ranked, gold, k, threshold)
        out[f"NDCG@{k}"] = ndcg_at_k(ranked, gold, k)
    return out


# 여러 질의의 지표를 평균낸다(None=코퍼스 공백 질의는 해당 지표 평균에서 제외)
def aggregate(per_query: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    summary: dict[str, float | None] = {}
    for key in keys:
        vals = [q[key] for q in per_query if q.get(key) is not None]
        summary[key] = round(sum(vals) / len(vals), 4) if vals else None
    return summary
