# RAG 검색 품질 배치 평가 — 정답 근거 라벨셋(rag/eval/gold_evidence.jsonl)에 대해
# Recall@k·Hit@k·MRR·NDCG@k를 계산한다.
#
# 핵심: 같은 임베딩 검색을 (1) 프로덕션처럼 axis 사전필터를 걸고, (2) 오라클처럼 필터 없이
# 두 번 돌려 두 recall의 격차 = 'axis 매핑이 놓치는 근거'를 정량화한다.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

from app.core import bootstrap  # noqa: E402,F401
from app.core.config import RAG_INDEX_EXPORT  # noqa: E402
from rag.retrieval_metrics import aggregate, score_ranking  # noqa: E402
from rag.retriever import RagIndex  # noqa: E402

DEFAULT_GOLD = ROOT / "rag" / "eval" / "gold_evidence.jsonl"
DEFAULT_OUT = ROOT / "reports" / "rag_retrieval_quality_eval.json"
KS = (3, 5, 10)


def _ranked_ids(idx: RagIndex, query: str, axis: str | None, k: int) -> list[str]:
    return [chunk["chunk_id"] for _score, chunk in idx.search(query, k=k, tier=None, axis=axis)]


def evaluate_query(idx: RagIndex, case: dict[str, Any], top_k: int) -> dict[str, Any]:
    gold: dict[str, int] = case["labels"]
    prod_ranked = _ranked_ids(idx, case["query"], case["axis"], top_k)
    oracle_ranked = _ranked_ids(idx, case["query"], None, top_k)

    prod_metrics = score_ranking(prod_ranked, gold, ks=KS)
    oracle_metrics = score_ranking(oracle_ranked, gold, ks=KS)

    # axis 필터가 놓친 recall = 오라클 - 프로덕션 (양수일수록 축 매핑 손실이 큼)
    recall_gap = {}
    for k in KS:
        key = f"Recall@{k}"
        p, o = prod_metrics[key], oracle_metrics[key]
        recall_gap[key] = round(o - p, 4) if (p is not None and o is not None) else None

    return {
        "query_id": case["query_id"],
        "axis": case["axis"],
        "정답_근거_수": sum(1 for g in gold.values() if g >= 1),
        "note": case.get("note", ""),
        "프로덕션_axis필터": prod_metrics,
        "오라클_필터없음": oracle_metrics,
        "axis매핑_recall손실": recall_gap,
        "프로덕션_검색된_정답": [cid for cid in prod_ranked if gold.get(cid, 0) >= 1],
        "오라클_검색된_정답": [cid for cid in oracle_ranked if gold.get(cid, 0) >= 1],
    }


def load_gold(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# 정답 라벨셋 확장용 — 한 질의의 검색 후보를 청크 본문과 함께 덤프해 사람이 등급을 매기게 한다
def dump_candidates(idx: RagIndex, query: str, axis: str | None, k: int) -> None:
    print(f"# query={query} axis={axis} (axis=None이면 필터 없음)")
    for rank, (score, chunk) in enumerate(idx.search(query, k=k, tier=None, axis=axis), start=1):
        print(f"[{rank}] {chunk['chunk_id']} score={score:.3f} axis={chunk['axis']} tier={chunk['tier']} "
              f"doc={chunk['doc_id'][:40]} p{chunk['page_start']}")
        print(f"    {chunk['text'][:160].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 검색 품질 평가(정답 근거 라벨셋 기반)")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--rag-export", default=str(RAG_INDEX_EXPORT))
    parser.add_argument("--top-k", type=int, default=10, help="검색 상위 몇 개까지 볼지")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dump-candidates", default=None, help="라벨셋 확장용: 이 질의의 검색 후보만 출력")
    parser.add_argument("--dump-axis", default=None, help="--dump-candidates와 함께 쓸 axis(생략 시 필터 없음)")
    args = parser.parse_args()

    idx = RagIndex.load(args.rag_export)

    if args.dump_candidates:
        dump_candidates(idx, args.dump_candidates, args.dump_axis, args.top_k)
        return

    cases = load_gold(args.gold)
    per_query = [evaluate_query(idx, case, args.top_k) for case in cases]

    result = {
        "cases": per_query,
        "summary": {
            "질의_수": len(per_query),
            "프로덕션_axis필터": aggregate([q["프로덕션_axis필터"] for q in per_query]),
            "오라클_필터없음": aggregate([q["오라클_필터없음"] for q in per_query]),
            "axis매핑_recall손실": aggregate([q["axis매핑_recall손실"] for q in per_query]),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
