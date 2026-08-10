# 대상 경로(AI 레포): tests/test_sales_target_district_metrics.py

import unittest

import pandas as pd

from app.sales_target.district_metrics import (
    aggregate_district_quarterly,
    latest_quarter_values,
    latest_vs_previous_growth,
)


class AggregateDistrictQuarterlyTests(unittest.TestCase):

    def test_sums_multiple_industries_into_one_district_total(self):
        df = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1", "A1"],
            "SVC_INDUTY_CD": ["CS100", "CS200", "CS300"],
            "STDR_YYQU_CD": ["20261", "20261", "20261"],
            "AMT": [100, 200, 300],
        })
        result = aggregate_district_quarterly(df, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["AMT"], 600)

    def test_keeps_districts_and_quarters_separate(self):
        df = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1", "A2"],
            "STDR_YYQU_CD": ["20254", "20261", "20261"],
            "AMT": [100, 150, 999],
        })
        result = aggregate_district_quarterly(df, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertEqual(len(result), 3)

    def test_non_numeric_values_treated_as_zero(self):
        df = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1"],
            "STDR_YYQU_CD": ["20261", "20261"],
            "AMT": [100, "이상값"],
        })
        result = aggregate_district_quarterly(df, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertEqual(result.iloc[0]["AMT"], 100)

    def test_empty_input_returns_empty_with_columns(self):
        df = pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "AMT"])
        result = aggregate_district_quarterly(df, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertTrue(result.empty)
        self.assertListEqual(list(result.columns), ["TRDAR_CD", "STDR_YYQU_CD", "AMT"])


class LatestVsPreviousGrowthTests(unittest.TestCase):

    def test_positive_growth(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1"],
            "STDR_YYQU_CD": ["20254", "20261"],
            "AMT": [100, 150],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertAlmostEqual(result.iloc[0]["growth_rate"], 0.5)

    def test_negative_growth(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1"],
            "STDR_YYQU_CD": ["20254", "20261"],
            "AMT": [200, 100],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertAlmostEqual(result.iloc[0]["growth_rate"], -0.5)

    def test_out_of_order_input_is_sorted_by_quarter_first(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1"],
            "STDR_YYQU_CD": ["20261", "20254"],  # 최신 분기가 먼저 나오는 순서로 입력
            "AMT": [150, 100],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertAlmostEqual(result.iloc[0]["growth_rate"], 0.5)

    def test_single_quarter_yields_nan(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1"],
            "STDR_YYQU_CD": ["20261"],
            "AMT": [100],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertTrue(pd.isna(result.iloc[0]["growth_rate"]))

    def test_zero_previous_yields_nan_not_error(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1"],
            "STDR_YYQU_CD": ["20254", "20261"],
            "AMT": [0, 100],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertTrue(pd.isna(result.iloc[0]["growth_rate"]))

    def test_multiple_districts_computed_independently(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1", "A2", "A2"],
            "STDR_YYQU_CD": ["20254", "20261", "20254", "20261"],
            "AMT": [100, 150, 100, 50],
        })
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        a1 = result[result["TRDAR_CD"] == "A1"].iloc[0]["growth_rate"]
        a2 = result[result["TRDAR_CD"] == "A2"].iloc[0]["growth_rate"]
        self.assertAlmostEqual(a1, 0.5)
        self.assertAlmostEqual(a2, -0.5)

    def test_empty_input_returns_empty(self):
        district_quarterly = pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "AMT"])
        result = latest_vs_previous_growth(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "AMT")
        self.assertTrue(result.empty)


class LatestQuarterValuesTests(unittest.TestCase):

    def test_picks_most_recent_quarter_per_district(self):
        district_quarterly = pd.DataFrame({
            "TRDAR_CD": ["A1", "A1", "A2"],
            "STDR_YYQU_CD": ["20254", "20261", "20261"],
            "POP": [1000, 1200, 500],
        })
        result = latest_quarter_values(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "POP")
        a1_value = result[result["TRDAR_CD"] == "A1"].iloc[0]["POP"]
        self.assertEqual(a1_value, 1200)

    def test_empty_input_returns_empty(self):
        district_quarterly = pd.DataFrame(columns=["TRDAR_CD", "STDR_YYQU_CD", "POP"])
        result = latest_quarter_values(district_quarterly, "TRDAR_CD", "STDR_YYQU_CD", "POP")
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
