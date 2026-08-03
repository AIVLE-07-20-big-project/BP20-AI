# Diagnoser -> AISalesAnalyzer(선택) -> ExternalFactorAnalyzer(선택) 순서로 기존
from __future__ import annotations

import threading

import pandas as pd

from app.core.config import MODEL
from scripts.modeling.quarters import pct_change, prev_quarter_code, same_quarter_last_year_code
from scripts.modeling.sales_analysis import Diagnoser, build_panel
from scripts.modeling.sales_report_renderer import build_simple_report

AI_MODEL_PATH = MODEL / "ai_sales_model.pkl"
EXTERNAL_RESULT_PATH = MODEL / "external_factor_analysis.json"

# AISalesAnalyzer 생성은 모델 pickle 로드 + 패널 CSV 읽기로 약 1초가 걸린다.
# 생성 후에는 읽기 전용이라 프로세스당 한 번만 만들어 재사용한다(ingestion.get_base_merged와 같은 방식).
_ai_analyzer = None
_ai_analyzer_lock = threading.Lock()


def get_ai_analyzer():
    global _ai_analyzer
    if _ai_analyzer is None:
        from scripts.modeling.ai_sales_analysis import AISalesAnalyzer

        # 동시 요청이 몰린 상태로 처음 호출되면 스레드마다 패널을 로드해 메모리가 급증한다.
        with _ai_analyzer_lock:
            if _ai_analyzer is None:  # 락을 기다리는 사이 다른 스레드가 만들었을 수 있다
                _ai_analyzer = AISalesAnalyzer()
    return _ai_analyzer


# ExternalFactorAnalyzer도 생성자가 고정 경로 json·csv 3개를 읽기만 해서 요청 파라미터에
# 의존하지 않는다 — get_ai_analyzer와 같은 이유로 프로세스당 한 번만 만들어 재사용한다.
_external_factor_analyzer = None
_external_factor_analyzer_lock = threading.Lock()


def get_external_factor_analyzer():
    global _external_factor_analyzer
    if _external_factor_analyzer is None:
        from scripts.modeling.external_factor_analysis import ExternalFactorAnalyzer

        with _external_factor_analyzer_lock:
            if _external_factor_analyzer is None:
                _external_factor_analyzer = ExternalFactorAnalyzer()
    return _external_factor_analyzer


# Diagnoser(build_panel(combined_df))는 combined_df 전체에 대한 groupby·share·위험모델
# 스코어링이라 요청마다 다시 만들면 비싸다. combined_df가 매 요청 동일 객체(ingestion.
# get_base_merged() 캐시)면 재사용하고, /reports처럼 매번 새 DataFrame이 오면 identity가
# 달라져 자동으로 다시 만든다(정확성 유지).
_diagnoser_cache: tuple[int, Diagnoser] | None = None
_diagnoser_lock = threading.Lock()


def get_diagnoser(combined_df: pd.DataFrame) -> Diagnoser:
    global _diagnoser_cache
    key = id(combined_df)
    cached = _diagnoser_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    with _diagnoser_lock:
        cached = _diagnoser_cache
        if cached is not None and cached[0] == key:  # 락 대기 중 다른 스레드가 만들었을 수 있다
            return cached[1]
        diagnoser = Diagnoser(panel=build_panel(combined_df, out=None))
        _diagnoser_cache = (key, diagnoser)
        return diagnoser


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

    diagnoser = get_diagnoser(combined_df)
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
            candidate = get_ai_analyzer().analyze(trdar_cd, svc_induty_cd, target_q)
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
            external = get_external_factor_analyzer().analyze(trdar_cd, target_q)
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
