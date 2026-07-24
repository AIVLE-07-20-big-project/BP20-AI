# RAG 리포트(rag/generator.py) 품질 배치 평가.
# 검색(retrieval) 단계와 생성(generation) 단계를 각각 독립적으로 채점하고(rag/evaluation.py),
# 점수가 낮은 케이스만 사람이 검토하도록 human_judgment 필드를 남긴다.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]

from app.core import bootstrap  # noqa: E402,F401  .env를 os.environ에 적재(OPENAI_API_KEY 등)
from app.core.config import RAG_INDEX_EXPORT  # noqa: E402
from app.services.response.action_rules import ACTION_TO_AXIS  # noqa: E402
from rag.evaluation import make_openai_judge, score_report_quality, score_retrieval_relevance  # noqa: E402
from rag.generator import generate_report  # noqa: E402
from rag.retriever import RagIndex  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "rag_report_quality_eval.json"
ReportGenerator = Callable[[dict[str, Any], str], dict[str, Any]]


# 방안 하나에 대해 검색→검색채점→생성→생성채점을 실행하고 사람검토 필요 여부까지 판단한다
def evaluate_action(
    action_name: str,
    axis: str,
    idx: RagIndex,
    *,
    max_retry: int = 1,
    retrieval_judge=None,
    generation_judge=None,
    report_generator: ReportGenerator | None = None,
) -> dict[str, Any]:

    evidence = idx.build_evidence(action_name, axis=axis)
    evidence_empty = not evidence.get("direction_refs") and not evidence.get("allowed_numbers")

    retrieval = score_retrieval_relevance(evidence, action_name, judge=retrieval_judge)

    generate = report_generator or (lambda ev, name: generate_report(ev, name, max_retry=max_retry))
    report_out = generate(evidence, action_name)

    generation = score_report_quality(report_out["report"], evidence, action_name, judge=generation_judge)

    needs_human_review = (
        retrieval.판정 != "관련"
        or generation.핵심정보포함 != "충분"
        or generation.근거일치 != "일치"
        or not report_out["verified"]
    )

    return {
        "action": action_name,
        "axis": axis,
        "evidence_empty": evidence_empty,
        "retrieval_judgment": retrieval.to_dict(),
        "generation_judgment": generation.to_dict(),
        "verified": report_out["verified"],
        "missing_sections": report_out.get("missing_sections", []),
        "unauthorized_sources": report_out.get("unauthorized_sources", []),
        "needs_human_review": needs_human_review,
        "human_judgment": None,
        "report": report_out["report"],
    }


# 케이스 목록에서 요약 지표를 계산하는 순수 함수(단위테스트 대상)
#
# 근거가 애초에 없는(evidence_empty) 케이스는 "무관/누락"이 나오는 게 당연해서 관련도·품질
# 비율에 섞으면 "코퍼스 공백"과 "채점 결과 나쁨"이 뒤섞인다. 그래서 이 두 지표군은
# evidence_empty가 아닌 케이스만으로 계산하고, 공백 케이스는 별도 카운트로만 보고한다.
def compute_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:

    n = len(cases)
    if n == 0:
        return {"케이스_수": 0}

    def rate(subset: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool]) -> float | None:
        return round(sum(1 for c in subset if pred(c)) / len(subset), 4) if subset else None

    scored = [c for c in cases if not c["evidence_empty"]]
    gap_cases = [c for c in cases if c["evidence_empty"]]

    summary: dict[str, Any] = {
        "케이스_수": n,
        "근거없음_방안": [c["action"] for c in gap_cases],
        "근거없음_제외_수": len(gap_cases),
        "평가대상_케이스_수": len(scored),
        "검색_관련_비율": rate(scored, lambda c: c["retrieval_judgment"]["판정"] == "관련"),
        "핵심정보_충분_비율": rate(scored, lambda c: c["generation_judgment"]["핵심정보포함"] == "충분"),
        "근거_일치_비율": rate(scored, lambda c: c["generation_judgment"]["근거일치"] == "일치"),
        "환각없음_비율": rate(scored, lambda c: c["verified"]),
        "사람검토_필요_비율": rate(cases, lambda c: c["needs_human_review"]),
    }

    reviewed = [c for c in cases if c.get("human_judgment")]
    if reviewed:
        summary["사람검토_완료_수"] = len(reviewed)
        summary["판정_일치율"] = round(
            sum(1 for c in reviewed if c["human_judgment"].get("판정") == c["retrieval_judgment"]["판정"])
            / len(reviewed), 4,
        )
        summary["핵심정보포함_일치율"] = round(
            sum(
                1 for c in reviewed
                if c["human_judgment"].get("핵심정보포함") == c["generation_judgment"]["핵심정보포함"]
            ) / len(reviewed), 4,
        )
        summary["근거일치_일치율"] = round(
            sum(1 for c in reviewed if c["human_judgment"].get("근거일치") == c["generation_judgment"]["근거일치"])
            / len(reviewed), 4,
        )

    return summary


# 이전에 사람이 human_judgment를 채워 넣은 출력 파일을 읽어 action명 기준으로 병합한다
def load_human_scores(path: str | Path) -> dict[str, dict[str, Any]]:

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        c["action"]: c["human_judgment"]
        for c in data.get("cases", [])
        if c.get("human_judgment")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 리포트 품질 배치 평가")
    parser.add_argument(
        "--actions", default=None,
        help="콤마로 구분한 방안 이름 목록(기본: action_rules.ACTION_TO_AXIS 전체)",
    )
    parser.add_argument("--rag-export", default=str(RAG_INDEX_EXPORT))
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--model", default="gpt-4.1", help="채점(judge)에 사용할 OpenAI 모델")
    parser.add_argument("--human-scores", default=None, help="사람이 채워 넣은 이전 출력 파일 경로")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    action_names = (
        [name.strip() for name in args.actions.split(",") if name.strip()]
        if args.actions
        else list(ACTION_TO_AXIS)
    )
    unknown = [name for name in action_names if name not in ACTION_TO_AXIS]
    if unknown:
        raise SystemExit(f"action_rules.ACTION_TO_AXIS에 없는 방안: {unknown}")

    idx = RagIndex.load(args.rag_export)
    judge = make_openai_judge(model=args.model)

    cases = [
        evaluate_action(
            name, ACTION_TO_AXIS[name], idx,
            max_retry=args.max_retry,
            retrieval_judge=judge,
            generation_judge=judge,
        )
        for name in action_names
    ]

    if args.human_scores:
        human_scores = load_human_scores(args.human_scores)
        for case in cases:
            if case["action"] in human_scores:
                case["human_judgment"] = human_scores[case["action"]]

    result = {"cases": cases, "summary": compute_summary(cases)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
