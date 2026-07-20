from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from scripts.modeling.ai_sales_analysis import AISalesAnalyzer, MODEL_PATH as AI_MODEL_PATH
from scripts.modeling.external_factor_analysis import (
    ExternalFactorAnalyzer,
    RESULT_PATH as EXTERNAL_RESULT_PATH,
)
from scripts.modeling.quarters import pct_change, prev_quarter_code, same_quarter_last_year_code
from scripts.modeling.sales_analysis import Diagnoser
from scripts.modeling.sales_report_renderer import build_simple_report, render_html_report


DEFAULT_CASE = {
    "trdar_cd": "3001491",
    "svc_induty_cd": "CS100003",
    "yyqu_cd": "20261",
}


def _axis_scores(axis_block: dict) -> dict:
    scores: dict[str, dict[str, float | str | None]] = {}
    for axis_name, payload in axis_block.items():
        z_map = payload.get("z", {}) if isinstance(payload, dict) else {}
        if not z_map:
            scores[axis_name] = {
                "강도": 0.0,
                "최대변화_구간": None,
                "최대변화량": 0.0,
            }
            continue

        max_label, max_value = max(z_map.items(), key=lambda kv: abs(float(kv[1])))
        avg_abs = sum(abs(float(v)) for v in z_map.values()) / len(z_map)
        scores[axis_name] = {
            "강도": round(float(avg_abs), 4),
            "최대변화_구간": max_label,
            "최대변화량": round(float(max_value), 4),
        }
    return scores


def _impact_payload(overall: float, axis_scores: dict) -> dict:
    return {
        "전체": round(float(overall), 4),
        "축별": axis_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trdar_cd", default=DEFAULT_CASE["trdar_cd"])
    parser.add_argument("--svc_induty_cd", default=DEFAULT_CASE["svc_induty_cd"])
    parser.add_argument("--yyqu_cd", default=DEFAULT_CASE["yyqu_cd"])
    parser.add_argument(
        "--out",
        default=str(ROOT / "sales_report.html"),
        help="HTML 리포트 저장 경로",
    )
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "sales_report.json"),
        help="진단 JSON 저장 경로",
    )
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "data" / "merged_sales_analysis.csv")
    target_q = int(args.yyqu_cd)
    row = df[
        (df["TRDAR_CD"] == int(args.trdar_cd))
        & (df["SVC_INDUTY_CD"] == args.svc_induty_cd)
        & (df["STDR_YYQU_CD"] == target_q)
    ].copy()
    if row.empty:
        raise SystemExit(f"대상 기준분기 데이터가 없습니다: {target_q}")
    row = row.sort_values("STDR_YYQU_CD").iloc[[-1]]

    cell_hist = df[
        (df["TRDAR_CD"] == int(args.trdar_cd))
        & (df["SVC_INDUTY_CD"] == args.svc_induty_cd)
    ].sort_values("STDR_YYQU_CD")
    current_q = int(row["STDR_YYQU_CD"].iloc[0])
    prev_q = prev_quarter_code(current_q)
    yoy_q = same_quarter_last_year_code(current_q)
    current_amt = row["THSMON_SELNG_AMT"].iloc[0] if "THSMON_SELNG_AMT" in row.columns else None
    prev_amt = cell_hist.loc[cell_hist["STDR_YYQU_CD"] == prev_q, "THSMON_SELNG_AMT"].iloc[0] if prev_q is not None and not cell_hist.loc[cell_hist["STDR_YYQU_CD"] == prev_q].empty else None
    yoy_amt = cell_hist.loc[cell_hist["STDR_YYQU_CD"] == yoy_q, "THSMON_SELNG_AMT"].iloc[0] if yoy_q is not None and not cell_hist.loc[cell_hist["STDR_YYQU_CD"] == yoy_q].empty else None
    row = row.copy()
    row["sales_qoq"] = pct_change(current_amt, prev_amt)
    row["sales_yoy"] = pct_change(current_amt, yoy_amt)

    raw_diag = Diagnoser().diagnose(args.trdar_cd, args.svc_induty_cd, current_q)
    if "error" in raw_diag:
        raise SystemExit(raw_diag["error"])
    report = build_simple_report(row, df, {})
    report["관측_변화_분석"] = {
        "심각도": raw_diag.get("1_심각도", {}),
        "동반_변화": raw_diag.get("2_원인_분해", {}),
        "구조_변화": raw_diag.get("3_구조_변화", {}),
        "축_분해": raw_diag.get("4_축_분해", {}),
        "확인과제": raw_diag.get("5_처방", {}),
        "분석_신뢰도": raw_diag.get("6_신뢰도", {}),
    }
    if AI_MODEL_PATH.exists():
        ai_result = AISalesAnalyzer().analyze(args.trdar_cd, args.svc_induty_cd, current_q)
        report["AI_분석"] = ai_result
    if EXTERNAL_RESULT_PATH.exists():
        external = ExternalFactorAnalyzer().analyze(args.trdar_cd, current_q)
        report["외부환경_참고"] = {
            "데이터해상도": external.get("데이터해상도"),
            "인과추정": False,
            "문화행사": {
                "사용가능": False,
                "판정": "일별 매출이 없어 효과 분석에 사용하지 않음",
            },
            "날씨": {
                "사용가능": False,
                "이유": (external.get("날씨") or {}).get("이유"),
            },
            "대상분기_문화행사노출": external.get("대상분기_문화행사노출", {}),
            "동종상권_대비_노출도": external.get("동종상권_대비_노출도", {}),
            "대상분기_대형점포_개폐업": external.get("대상분기_대형점포_개폐업", {}),
            "대형점포": external.get("대형점포", {}),
            "대상분기_지하철승하차노출": external.get("대상분기_지하철승하차노출", {}),
            "지하철승하차": external.get("지하철승하차", {}),
            "해석주의": (
                "행사·날씨는 참고 정보이며 매출 영향이나 원인으로 사용하지 않음. "
                "대형점포 개업/폐업, 지하철 승하차는 통계적 연관성이 확인된 경우에만"
                "(대형점포.사용가능, 지하철승하차.사용가능) 대응방안 추천에 반영되며, "
                "그렇지 않으면 참고 사실로만 표시됨"
            ),
        }
    html = render_html_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_path)
    print(json_path)


if __name__ == "__main__":
    main()
