# Diagnoser -> AISalesAnalyzer(선택) -> ExternalFactorAnalyzer(선택) 순서로 기존
from __future__ import annotations

import pandas as pd

from app.core.config import MODEL
from scripts.modeling.quarters import pct_change, prev_quarter_code, same_quarter_last_year_code
from scripts.modeling.sales_analysis import Diagnoser, build_panel
from scripts.modeling.sales_report_renderer import build_simple_report

AI_MODEL_PATH = MODEL / "ai_sales_model.pkl"
EXTERNAL_RESULT_PATH = MODEL / "external_factor_analysis.json"


# 대상 상권x업종x분기 조합을 패널에서 찾을 수 없을 때
class CellNotFoundError(Exception):
    pass


# 반환: (report, raw_diag, 경고목록)
def run_pipeline(trdar_cd: str, svc_induty_cd: str, yyqu_cd: int | None,
                  combined_df: pd.DataFrame) -> tuple[dict, dict, list[str]]:

    warnings: list[str] = []
    trdar_int = int(trdar_cd)

    cell_hist = combined_df[
        (combined_df["TRDAR_CD"] == trdar_int) & (combined_df["SVC_INDUTY_CD"] == svc_induty_cd)
    ].sort_values("STDR_YYQU_CD")
    if cell_hist.empty:
        raise CellNotFoundError(f"{trdar_cd}/{svc_induty_cd} 데이터 없음")

    target_q = int(yyqu_cd) if yyqu_cd is not None else int(cell_hist["STDR_YYQU_CD"].iloc[-1])
    row = cell_hist[cell_hist["STDR_YYQU_CD"] == target_q]
    if row.empty:
        raise CellNotFoundError(f"{trdar_cd}/{svc_induty_cd} 기준분기 {target_q} 없음")
    row = row.sort_values("STDR_YYQU_CD").iloc[[-1]].copy()

    prev_q = prev_quarter_code(target_q)
    yoy_q = same_quarter_last_year_code(target_q)
    current_amt = row["THSMON_SELNG_AMT"].iloc[0]
    prev_rows = cell_hist.loc[cell_hist["STDR_YYQU_CD"] == prev_q] if prev_q is not None else cell_hist.iloc[0:0]
    yoy_rows = cell_hist.loc[cell_hist["STDR_YYQU_CD"] == yoy_q] if yoy_q is not None else cell_hist.iloc[0:0]
    prev_amt = prev_rows["THSMON_SELNG_AMT"].iloc[0] if not prev_rows.empty else None
    yoy_amt = yoy_rows["THSMON_SELNG_AMT"].iloc[0] if not yoy_rows.empty else None
    row["sales_qoq"] = pct_change(current_amt, prev_amt)
    row["sales_yoy"] = pct_change(current_amt, yoy_amt)

    diagnoser = Diagnoser(panel=build_panel(combined_df, out=None))
    raw_diag = diagnoser.diagnose(trdar_cd, svc_induty_cd, target_q)
    if "error" in raw_diag:
        raise CellNotFoundError(raw_diag["error"])

    report = build_simple_report(row, combined_df, {})
    report["관측_변화_분석"] = {
        "심각도": raw_diag.get("1_심각도", {}),
        "동반_변화": raw_diag.get("2_원인_분해", {}),
        "구조_변화": raw_diag.get("3_구조_변화", {}),
        "축_분해": raw_diag.get("4_축_분해", {}),
        "확인과제": raw_diag.get("5_처방", {}),
        "분석_신뢰도": raw_diag.get("6_신뢰도", {}),
    }

    ai_result = None
    if AI_MODEL_PATH.exists():
        try:
            from scripts.modeling.ai_sales_analysis import AISalesAnalyzer
            candidate = AISalesAnalyzer().analyze(trdar_cd, svc_induty_cd, target_q)
            if "error" not in candidate:
                ai_result = candidate
                report["AI_분석"] = ai_result
            else:
                warnings.append(f"AI 분석 사용 불가: {candidate['error']}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"AI 분석 실패: {exc}")

    external_block = None
    if EXTERNAL_RESULT_PATH.exists():
        try:
            from scripts.modeling.external_factor_analysis import ExternalFactorAnalyzer
            external = ExternalFactorAnalyzer().analyze(trdar_cd, target_q)
            external_block = {
                "데이터해상도": external.get("데이터해상도"),
                "인과추정": False,
                "문화행사": {"사용가능": False, "판정": "일별 매출이 없어 효과 분석에 사용하지 않음"},
                "날씨": {"사용가능": False, "이유": (external.get("날씨") or {}).get("이유")},
                "대상분기_문화행사노출": external.get("대상분기_문화행사노출", {}),
                "동종상권_대비_노출도": external.get("동종상권_대비_노출도", {}),
                "대상분기_대형점포_개폐업": external.get("대상분기_대형점포_개폐업", {}),
                "대형점포": external.get("대형점포", {}),
                "대상분기_지하철승하차노출": external.get("대상분기_지하철승하차노출", {}),
                "지하철승하차": external.get("지하철승하차", {}),
            }
            report["외부환경_참고"] = external_block
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"외부환경 분석 실패: {exc}")

    return report, raw_diag, warnings
