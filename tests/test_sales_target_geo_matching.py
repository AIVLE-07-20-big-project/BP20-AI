# 대상 경로(AI 레포): tests/test_sales_target_geo_matching.py

import unittest

import pandas as pd

from app.sales_target.geo_matching import add_trdar_wgs84_columns, assign_trdar_cd


class AddTrdarWgs84ColumnsTests(unittest.TestCase):

    def test_matches_verified_hwanghakdong_example(self):
        # 2026-07-29 실제 API 호출로 확인한 값: 황학동벼룩시장(TRDAR_CD=3110055)
        trdar_df = pd.DataFrame({
            "TRDAR_CD": ["3110055"],
            "XCNTS_VALUE": ["201642"],
            "YDNTS_VALUE": ["452260"],
            "RELM_AR": ["27575"],
        })

        result = add_trdar_wgs84_columns(trdar_df)

        self.assertAlmostEqual(result.iloc[0]["lon"], 127.01859, places=3)
        self.assertAlmostEqual(result.iloc[0]["lat"], 37.56988, places=3)
        self.assertAlmostEqual(result.iloc[0]["approx_radius_m"], 93.69, places=1)

    def test_empty_dataframe_returns_empty_with_columns(self):
        trdar_df = pd.DataFrame(columns=["TRDAR_CD", "XCNTS_VALUE", "YDNTS_VALUE", "RELM_AR"])
        result = add_trdar_wgs84_columns(trdar_df)
        self.assertTrue(result.empty)
        for col in ("lon", "lat", "approx_radius_m"):
            self.assertIn(col, result.columns)


class AssignTrdarCdTests(unittest.TestCase):

    def setUp(self):
        # 이미 WGS84로 변환됐다고 가정한 상권 2곳 — 서로 충분히 멀리 떨어져 있다(약 8.8km 차이).
        self.trdar_df = pd.DataFrame({
            "TRDAR_CD": ["A1", "A2"],
            "lon": [127.0, 127.1],
            "lat": [37.5, 37.5],
            "approx_radius_m": [100.0, 150.0],
        })

    def test_picks_nearest_trdar(self):
        candidates = pd.DataFrame({
            "name": ["가까운후보"],
            "lon": [127.0005],
            "lat": [37.5001],
        })

        result = assign_trdar_cd(candidates, self.trdar_df)

        self.assertEqual(result.iloc[0]["trdar_cd"], "A1")
        self.assertLess(result.iloc[0]["trdar_distance_m"], 200)
        self.assertEqual(result.iloc[0]["trdar_approx_radius_m"], 100.0)

    def test_picks_second_trdar_when_closer(self):
        candidates = pd.DataFrame({
            "name": ["다른후보"],
            "lon": [127.099],
            "lat": [37.5],
        })

        result = assign_trdar_cd(candidates, self.trdar_df)

        self.assertEqual(result.iloc[0]["trdar_cd"], "A2")

    def test_missing_candidate_coords_yields_none(self):
        candidates = pd.DataFrame({"name": ["좌표없음"], "lon": [None], "lat": [None]})

        result = assign_trdar_cd(candidates, self.trdar_df)

        self.assertIsNone(result.iloc[0]["trdar_cd"])
        self.assertTrue(pd.isna(result.iloc[0]["trdar_distance_m"]))

    def test_empty_trdar_df_returns_none_for_all(self):
        candidates = pd.DataFrame({"name": ["A", "B"], "lon": [127.0, 127.1], "lat": [37.5, 37.5]})
        empty_trdar = pd.DataFrame(columns=["TRDAR_CD", "lon", "lat", "approx_radius_m"])

        result = assign_trdar_cd(candidates, empty_trdar)

        self.assertTrue(result["trdar_cd"].isna().all())

    def test_empty_candidates_returns_empty(self):
        candidates = pd.DataFrame(columns=["name", "lon", "lat"])
        result = assign_trdar_cd(candidates, self.trdar_df)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
