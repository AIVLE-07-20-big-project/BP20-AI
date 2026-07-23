import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch


from scripts.modeling.sales_analysis import (
    Diagnoser,
    classify_market_state,
    classify_regional_pattern,
    classify_traffic_source,
    pct_change,
    shift_quarter,
)
from scripts.modeling.ai_sales_analysis import make_features
from scripts.modeling.external_factor_analysis import (
    _event_quarters,
    audit_weather,
    peer_exposure_percentile,
)
from scripts.modeling.sales_report_renderer import build_simple_report


class QuarterComparisonTests(unittest.TestCase):
    def test_shift_quarter_crosses_year(self):
        self.assertEqual(shift_quarter(20261, -1), 20254)
        self.assertEqual(shift_quarter(20261, -4), 20251)
        self.assertEqual(shift_quarter(20254, 1), 20261)

    def test_pct_change_uses_exact_quarter(self):
        values = pd.Series([100, 120], index=[20253, 20261])
        self.assertIsNone(pct_change(values, 1))
        self.assertIsNone(pct_change(values, 4))

    def test_pct_change_returns_expected_value(self):
        values = pd.Series([100, 110, 125], index=[20251, 20254, 20261])
        self.assertEqual(pct_change(values, 1), round((125 - 110) / 110, 4))
        self.assertEqual(pct_change(values, 4), 0.25)


class MarketStateTests(unittest.TestCase):
    def test_market_contraction_when_stores_and_sales_fall(self):
        self.assertEqual(classify_market_state(-0.18, -0.18, 0.01), "시장축소")

    def test_compound_decline(self):
        self.assertEqual(classify_market_state(-0.25, -0.15, -0.12), "복합침체")

    def test_competition_and_demand_loss(self):
        self.assertEqual(classify_market_state(-0.05, 0.15, -0.15), "경쟁심화")
        self.assertEqual(classify_market_state(-0.15, 0.02, -0.16), "수요이탈")

    def test_normal_requires_sales_and_per_store_sales_to_hold(self):
        self.assertEqual(classify_market_state(-0.05, -0.05, 0.0), "정상")


class TrafficSourceTests(unittest.TestCase):
    def test_both_declining_is_structural_contraction(self):
        self.assertEqual(classify_traffic_source(-0.15, -0.15), "상권_자체_축소")

    def test_stable_residents_with_declining_traffic_is_external_inflow_loss(self):
        self.assertEqual(classify_traffic_source(0.02, -0.15), "외부_유입_감소")

    def test_declining_residents_with_stable_traffic_is_resident_exodus(self):
        self.assertEqual(classify_traffic_source(-0.15, 0.01), "거주자_이탈")

    def test_missing_values_are_unclassifiable(self):
        self.assertEqual(classify_traffic_source(None, -0.15), "판정불가")

    def test_all_three_declining_is_citywide_contraction(self):
        self.assertEqual(classify_traffic_source(-0.15, -0.15, -0.15), "상권_전방위_축소")

    def test_workplace_and_traffic_declining_with_stable_residents(self):
        self.assertEqual(classify_traffic_source(0.01, -0.15, -0.15), "직장인구_이탈형_외부유입감소")

    def test_only_workplace_declining_is_leading_signal(self):
        self.assertEqual(classify_traffic_source(0.01, 0.01, -0.15), "직장인구_감소_선행신호")

    def test_resident_exodus_requires_stable_workplace_too(self):
        self.assertEqual(classify_traffic_source(-0.15, 0.01, 0.01), "거주자_이탈")

    def test_external_inflow_loss_requires_stable_workplace(self):
        self.assertEqual(classify_traffic_source(0.01, -0.15, 0.01), "외부_유입_감소")

    def test_structural_contraction_requires_stable_workplace(self):
        self.assertEqual(classify_traffic_source(-0.15, -0.15, 0.01), "상권_자체_축소")

    def test_missing_workplace_change_falls_back_to_two_axis_logic(self):
        self.assertEqual(classify_traffic_source(-0.15, -0.15, None), "상권_자체_축소")
        self.assertEqual(classify_traffic_source(-0.15, -0.15, float("nan")), "상권_자체_축소")

    def test_near_zero_repop_change_is_not_confirmed_stable(self):


        self.assertEqual(classify_traffic_source(0.0, -0.15), "판정불가")

    def test_near_zero_repop_does_not_block_workplace_leading_signal(self):


        self.assertEqual(classify_traffic_source(0.0, 0.01, -0.2), "직장인구_감소_선행신호")

    def test_near_zero_repop_blocks_external_inflow_claim_with_workplace(self):
        self.assertEqual(classify_traffic_source(0.0, -0.15, 0.01), "판정불가")
        self.assertEqual(classify_traffic_source(float("nan"), -0.15), "판정불가")


class RegionalPatternTests(unittest.TestCase):
    def test_both_declining_is_regional_co_decline(self):
        self.assertEqual(classify_regional_pattern(-0.15, -0.15, 5), "지역_동반_하락")

    def test_target_declining_alone_is_isolated_decline(self):
        self.assertEqual(classify_regional_pattern(-0.15, 0.01, 5), "상권_고립형_하락")

    def test_neighbor_declining_alone_is_counter_trend_strength(self):
        self.assertEqual(classify_regional_pattern(0.01, -0.15, 5), "상권_역행_호조")

    def test_both_rising_is_regional_co_growth(self):
        self.assertEqual(classify_regional_pattern(0.15, 0.15, 5), "지역_동반_호조")

    def test_too_few_neighbors_is_unclassifiable(self):
        self.assertEqual(classify_regional_pattern(-0.15, -0.15, 2), "판정불가")

    def test_missing_values_are_unclassifiable(self):
        self.assertEqual(classify_regional_pattern(None, -0.15, 5), "판정불가")
        self.assertEqual(classify_regional_pattern(-0.15, float("nan"), 5), "판정불가")


class PrescriptionTests(unittest.TestCase):
    def test_short_term_decline_is_not_called_normal(self):
        severity = {
            "전분기_대비": -0.17,
            "전년동기_대비": 0.0,
            "최고점_대비": -0.17,
            "하락_분기_비율": 0.75,
        }
        result = Diagnoser._prescribe("정상", severity, {}, {})
        self.assertEqual(result["등급"], "관찰")
        self.assertEqual(result["긴급도"], "중간")


class AnalysisSafetyTests(unittest.TestCase):
    def test_extreme_special_market_blocks_comparison_and_prescription(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "발달상권", "THSMON_SELNG_CO": 100,
            "STOR_CO": 10, "FRC_STOR_CO": 0, "OPBIZ_RT": 1,
            "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000,
        }
        for i, q in enumerate([20244, 20251, 20252, 20253, 20254, 20261]):
            rows.append({**base, "TRDAR_CD": 1, "TRDAR_CD_NM": "특수 대상",
                         "STDR_YYQU_CD": q, "THSMON_SELNG_AMT": 100_000,
                         "TRDAR_AMT": 100_000, "INDUTY_AMT": 100_000})
        for peer in range(2, 23):
            rows.append({**base, "TRDAR_CD": peer, "TRDAR_CD_NM": f"비교 {peer}",
                         "STDR_YYQU_CD": 20261, "THSMON_SELNG_AMT": 1_000,
                         "TRDAR_AMT": 1_000, "INDUTY_AMT": 1_000})
        path = Path(__file__).parent / "_safety_panel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(path).diagnose(1, "A", 20261)
        finally:
            path.unlink(missing_ok=True)
        self.assertFalse(result["6_신뢰도"]["분석사용가능"])
        self.assertEqual(result["4_축_분해"], {})
        self.assertEqual(result["5_처방"]["등급"], "분석_차단")
        self.assertTrue(any("50배" in x for x in result["6_신뢰도"]["차단사유"]))

    def test_large_district_passes_when_peer_group_narrowed_by_district_type(self):
        """전국 동업종 피어(대부분 소규모 골목상권)와 비교하면 50배 넘어 차단되지만,
        같은 상권유형(발달상권) 피어 20곳 이상과 비교하면 정상 범위 — 실제 강남역 카페
        재현: 전국 기준 107배(차단) vs 발달상권 기준 14배(정상)에서 발견된 버그의 회귀 테스트."""
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "THSMON_SELNG_CO": 100, "STOR_CO": 10, "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000,
        }
        for i, q in enumerate([20244, 20251, 20252, 20253, 20254, 20261]):
            rows.append({**base, "TRDAR_CD": 1, "TRDAR_CD_NM": "대형 상권",
                         "TRDAR_SE_CD_NM": "발달상권", "STDR_YYQU_CD": q,
                         "THSMON_SELNG_AMT": 100_000, "TRDAR_AMT": 100_000, "INDUTY_AMT": 100_000})
        for peer in range(2, 22):
            rows.append({**base, "TRDAR_CD": peer, "TRDAR_CD_NM": f"발달상권 {peer}",
                         "TRDAR_SE_CD_NM": "발달상권", "STDR_YYQU_CD": 20261,
                         "THSMON_SELNG_AMT": 100_000, "TRDAR_AMT": 100_000, "INDUTY_AMT": 100_000})
        for peer in range(22, 122):
            rows.append({**base, "TRDAR_CD": peer, "TRDAR_CD_NM": f"골목상권 {peer}",
                         "TRDAR_SE_CD_NM": "골목상권", "STDR_YYQU_CD": 20261,
                         "THSMON_SELNG_AMT": 1_000, "TRDAR_AMT": 1_000, "INDUTY_AMT": 1_000})
        path = Path(__file__).parent / "_narrowing_panel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(path).diagnose(1, "A", 20261)
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(result["6_신뢰도"]["분석사용가능"])
        self.assertLess(result["6_신뢰도"]["동종중앙값_대비배수"], 2.0)
        self.assertEqual(result["6_신뢰도"]["동종비교대상수"], 20)

    def test_sales_is_decomposed_into_transactions_and_ticket(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000,
            "STOR_CO": 10,
        }
        quarters = [20244, 20251, 20252, 20253, 20254, 20261]
        for market in range(1, 22):
            for i, q in enumerate(quarters):
                rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                             "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                             "THSMON_SELNG_AMT": (100 + i) * 1000,
                             "TRDAR_AMT": (100 + i) * 1000,
                             "INDUTY_AMT": (100 + i) * 1000})
        path = Path(__file__).parent / "_components_panel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(path).diagnose(1, "A", 20261)
        finally:
            path.unlink(missing_ok=True)
        self.assertTrue(result["6_신뢰도"]["분석사용가능"])
        self.assertIn("거래건수", result["3_구조_변화"])
        self.assertEqual(result["3_구조_변화"]["거래당_매출"]["현재"], 1000)

    def test_workplace_population_flows_into_structure_change(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000, "STOR_CO": 10,
        }
        quarters = [20244, 20251, 20252, 20253, 20254, 20261]
        for market in range(1, 22):
            for i, q in enumerate(quarters):
                rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                             "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                             "THSMON_SELNG_AMT": (100 + i) * 1000,
                             "TRDAR_AMT": (100 + i) * 1000, "INDUTY_AMT": (100 + i) * 1000,
                             "TOT_WRC_POPLTN_CO": 800 if market != 1 else 800 - i * 10})
        path = Path(__file__).parent / "_workpop_panel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(path).diagnose(1, "A", 20261)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("직장인구", result["3_구조_변화"])
        self.assertLess(result["3_구조_변화"]["직장인구"]["변화율"], 0)

    def test_resident_population_decline_flags_structural_contraction(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "STOR_CO": 10,
        }
        quarters = [20244, 20251, 20252, 20253, 20254, 20261]
        for market in range(1, 22):
            for i, q in enumerate(quarters):
                if market == 1:
                    flpop, repop = 1000 - i * 30, 500 - i * 20
                else:
                    flpop, repop = 1000, 500
                rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                             "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                             "THSMON_SELNG_AMT": (100 + i) * 1000,
                             "TRDAR_AMT": (100 + i) * 1000, "INDUTY_AMT": (100 + i) * 1000,
                             "TOT_FLPOP_CO": flpop, "TOT_REPOP_CO": repop})
        path = Path(__file__).parent / "_traffic_source_panel.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(path).diagnose(1, "A", 20261)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("상주인구", result["3_구조_변화"])
        self.assertEqual(result["3_구조_변화"]["유동인구_원인"], "상권_자체_축소")

    def test_neighbor_comparison_flows_into_structure_change(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000, "STOR_CO": 10,
        }
        quarters = [20244, 20251, 20252, 20253, 20254, 20261]
        for market in range(1, 22):
            for i, q in enumerate(quarters):
                rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                             "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                             "THSMON_SELNG_AMT": (100 + i) * 1000,
                             "TRDAR_AMT": (100 + i) * 1000, "INDUTY_AMT": (100 + i) * 1000})
        panel_path = Path(__file__).parent / "_neighbor_panel.csv"
        neighbor_path = Path(__file__).parent / "_neighbor_features.csv"
        pd.DataFrame(rows).to_csv(panel_path, index=False)
        pd.DataFrame([
            {"TRDAR_CD": 1, "STDR_YYQU_CD": 20261,
             "target_change": -0.20, "neighbor_change": -0.18, "neighbor_count": 5},
        ]).to_csv(neighbor_path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(panel_path, neighbor_path).diagnose(1, "A", 20261)
        finally:
            panel_path.unlink(missing_ok=True)
            neighbor_path.unlink(missing_ok=True)
        self.assertEqual(result["3_구조_변화"]["인접상권_비교"]["판정"], "지역_동반_하락")
        self.assertEqual(result["3_구조_변화"]["인접상권_비교"]["인접상권_수"], 5)

    def test_missing_neighbor_file_leaves_comparison_unclassifiable(self):
        rows = []
        base = {
            "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
            "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000, "STOR_CO": 10,
        }
        quarters = [20244, 20251, 20252, 20253, 20254, 20261]
        for market in range(1, 22):
            for i, q in enumerate(quarters):
                rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                             "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                             "THSMON_SELNG_AMT": (100 + i) * 1000,
                             "TRDAR_AMT": (100 + i) * 1000, "INDUTY_AMT": (100 + i) * 1000})
        panel_path = Path(__file__).parent / "_neighbor_missing_panel.csv"
        pd.DataFrame(rows).to_csv(panel_path, index=False)
        try:
            with patch("scripts.modeling.sales_analysis.load_risk", return_value=(None, None)):
                result = Diagnoser(panel_path, Path("no_such_file.csv")).diagnose(1, "A", 20261)
        finally:
            panel_path.unlink(missing_ok=True)
        self.assertEqual(result["3_구조_변화"]["인접상권_비교"]["판정"], "판정불가")

    def test_area_median_only_uses_same_industry(self):
        data = pd.DataFrame([
            {"TRDAR_CD": 1, "STDR_YYQU_CD": 20261, "SVC_INDUTY_CD": "A",
             "TRDAR_SE_CD": "R", "THSMON_SELNG_AMT": 100_000_000,
             "STOR_CO": 2, "TOT_FLPOP_CO": 100},
            {"TRDAR_CD": 2, "STDR_YYQU_CD": 20261, "SVC_INDUTY_CD": "A",
             "TRDAR_SE_CD": "R", "THSMON_SELNG_AMT": 300_000_000,
             "STOR_CO": 4, "TOT_FLPOP_CO": 100},
            {"TRDAR_CD": 3, "STDR_YYQU_CD": 20261, "SVC_INDUTY_CD": "B",
             "TRDAR_SE_CD": "R", "THSMON_SELNG_AMT": 10_000_000_000,
             "STOR_CO": 100, "TOT_FLPOP_CO": 100},
        ])
        row = data.iloc[[0]].copy()
        report = build_simple_report(row, data, {})
        self.assertEqual(report["매출분석"]["동일업종·지역유형 중앙값"], "20,000 만원")
        self.assertEqual(report["업종분석"]["동일업종·지역유형 중앙 업소수"], 3.0)

    def test_report_includes_gender_age_sales_mix(self):
        data = pd.DataFrame([{
            "TRDAR_CD": 1, "STDR_YYQU_CD": 20261, "SVC_INDUTY_CD": "A",
            "TRDAR_SE_CD": "R", "THSMON_SELNG_AMT": 1000, "STOR_CO": 2,
            "TOT_FLPOP_CO": 100, "ML_SELNG_AMT": 700, "FML_SELNG_AMT": 300,
            "AGRDE_10_SELNG_AMT": 100, "AGRDE_20_SELNG_AMT": 200,
            "AGRDE_30_SELNG_AMT": 300, "AGRDE_40_SELNG_AMT": 200,
            "AGRDE_50_SELNG_AMT": 150, "AGRDE_60_ABOVE_SELNG_AMT": 50,
        }])
        report = build_simple_report(data, data, {})
        gender = report["고객군별_매출비중"]["성별"]
        self.assertEqual(gender["labels"], ["남성", "여성"])
        self.assertAlmostEqual(gender["지역별"]["대상 상권"][0], 0.7)
        age = report["고객군별_매출비중"]["연령대"]
        self.assertIn("60대이상", age["labels"])


class AISupervisedDataTests(unittest.TestCase):
    def test_change_requires_exact_previous_quarter(self):
        base = {
            "TRDAR_CD": 1, "SVC_INDUTY_CD": "A", "TRDAR_CD_NM": "상권",
            "TRDAR_SE_CD_NM": "골목상권", "SVC_INDUTY_CD_NM": "업종",
            "THSMON_SELNG_CO": 100, "STOR_CO": 2, "FRC_STOR_CO": 0,
            "OPBIZ_RT": 1.0, "CLSBIZ_RT": 1.0, "TOT_FLPOP_CO": 1000,
            "TRDAR_AMT": 1000, "INDUTY_AMT": 1000,
        }
        rows = []
        for quarter, sales in [(20251, 100), (20252, 110), (20254, 130)]:
            rows.append({**base, "STDR_YYQU_CD": quarter, "THSMON_SELNG_AMT": sales})
        table = make_features(pd.DataFrame(rows))
        q1 = table.loc[table["STDR_YYQU_CD"] == 20251].iloc[0]
        q2 = table.loc[table["STDR_YYQU_CD"] == 20252].iloc[0]
        q4 = table.loc[table["STDR_YYQU_CD"] == 20254].iloc[0]
        self.assertTrue(pd.isna(q1["sales_qoq"]))
        self.assertEqual(round(q2["sales_qoq"], 4), 0.1)
        self.assertTrue(pd.isna(q4["sales_qoq"]))


class ExternalFactorTests(unittest.TestCase):
    def test_event_days_are_split_across_quarters(self):
        start = pd.Timestamp("2025-03-30")
        end = pd.Timestamp("2025-04-02")
        self.assertEqual(list(_event_quarters(start, end)), [(20251, 2), (20252, 2)])

    def test_short_quarterly_weather_history_is_blocked(self):
        result = audit_weather()
        self.assertFalse(result["사용가능"])
        self.assertEqual(result["음수강수량수"], 0)
        self.assertLess(result["관측분기수"], 12)

    def test_peer_exposure_percentile_fills_missing_rows_with_zero(self):
        panel_path = Path(__file__).parent / "_peer_panel.csv"
        event_path = Path(__file__).parent / "_peer_events.csv"
        pd.DataFrame([{"TRDAR_CD": i, "STDR_YYQU_CD": 20261} for i in range(1, 26)]
                     ).to_csv(panel_path, index=False)
        pd.DataFrame([{"TRDAR_CD": 1, "STDR_YYQU_CD": 20261, "event_count": 1,
                       "event_days": 5, "nearest_event_m": 100.0, "event_exposure": 50.0}]
                     ).to_csv(event_path, index=False)
        try:
            zero_cell = peer_exposure_percentile(2, 20261, panel_path=panel_path, event_path=event_path)
            top_cell = peer_exposure_percentile(1, 20261, panel_path=panel_path, event_path=event_path)
        finally:
            panel_path.unlink(missing_ok=True)
            event_path.unlink(missing_ok=True)
        self.assertEqual(zero_cell["비교대상수"], 25)


        self.assertLess(zero_cell["노출_백분위"], 60)
        self.assertEqual(top_cell["판정"], "높음")

    def test_peer_exposure_percentile_blocks_when_too_few_peers(self):
        panel_path = Path(__file__).parent / "_small_panel.csv"
        event_path = Path(__file__).parent / "_small_events.csv"
        pd.DataFrame([{"TRDAR_CD": i, "STDR_YYQU_CD": 20261} for i in range(1, 5)]
                     ).to_csv(panel_path, index=False)
        pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "event_count",
                               "event_days", "nearest_event_m", "event_exposure"]
                     ).to_csv(event_path, index=False)
        try:
            result = peer_exposure_percentile(1, 20261, panel_path=panel_path, event_path=event_path)
        finally:
            panel_path.unlink(missing_ok=True)
            event_path.unlink(missing_ok=True)
        self.assertEqual(result["판정"], "비교불가")


if __name__ == "__main__":
    unittest.main()
