# 대상 경로(AI 레포): tests/test_sales_target_hero_store.py

import unittest

import pandas as pd

from app.sales_target.hero_store import (
    build_similarity_vectors,
    extract_sigungu,
    hero_similarity_scores,
    select_hero_stores,
    sigungu_average_district_metrics,
)


class SelectHeroStoresTests(unittest.TestCase):

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"salesGrowthRate": [0.1], "reviewAvgRating": [4.5]})
        with self.assertRaises(ValueError):
            select_hero_stores(df)

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["salesGrowthRate", "reviewAvgRating", "reviewRatingStd"])
        result = select_hero_stores(df)
        self.assertTrue(result.empty)

    def test_all_growth_nan_returns_empty(self):
        df = pd.DataFrame({
            "salesGrowthRate": [None, None],
            "reviewAvgRating": [4.5, 4.8],
            "reviewRatingStd": [0.5, 0.3],
        })
        result = select_hero_stores(df)
        self.assertTrue(result.empty)

    def test_selects_high_growth_stable_rating_stores_only(self):
        df = pd.DataFrame({
            "name": ["A(우수)", "B(성장률낮음)", "C(평점불안정)", "D(평점낮음)", "E(데이터없음)"],
            "salesGrowthRate": [0.9, 0.05, 0.85, 0.80, None],
            "reviewAvgRating": [4.8, 4.9, 4.7, 3.5, 4.9],
            "reviewRatingStd": [0.3, 0.2, 2.5, 0.2, 0.2],
        })
        result = select_hero_stores(df)
        self.assertEqual(result["name"].tolist(), ["A(우수)"])


class ExtractSigunguTests(unittest.TestCase):

    def test_finds_gu_in_address(self):
        self.assertEqual(extract_sigungu("서울특별시 강남구 테스트로 1"), "강남구")

    def test_returns_none_when_no_match(self):
        self.assertIsNone(extract_sigungu("주소없음"))

    def test_returns_none_for_non_string(self):
        self.assertIsNone(extract_sigungu(None))
        self.assertIsNone(extract_sigungu(float("nan")))


class SigunguAverageDistrictMetricsTests(unittest.TestCase):

    def test_averages_grouped_by_sigungu(self):
        df = pd.DataFrame({
            "signguNm": ["강남구", "강남구", "마포구"],
            "_district_growth_rate": [0.1, 0.3, 0.5],
            "_district_traffic_index": [100.0, 200.0, 50.0],
        })
        result = sigungu_average_district_metrics(
            df, sigungu_col="signguNm", growth_col="_district_growth_rate", traffic_col="_district_traffic_index"
        )
        gangnam = result[result["district"] == "강남구"].iloc[0]
        self.assertAlmostEqual(gangnam["district_growth_rate"], 0.2)
        self.assertAlmostEqual(gangnam["district_traffic_index"], 150.0)


class BuildSimilarityVectorsTests(unittest.TestCase):

    def test_returns_correct_split_sizes(self):
        candidates = pd.DataFrame({
            "industry": ["카페", "미용실"],
            "district": ["강남구", "마포구"],
            "district_growth_rate": [0.1, 0.2],
            "district_traffic_index": [100.0, 50.0],
            "review_count": [0.0, 0.0],
            "review_avg_rating": [0.0, 0.0],
        })
        heroes = pd.DataFrame({
            "industry": ["카페"],
            "district": ["강남구"],
            "district_growth_rate": [0.15],
            "district_traffic_index": [90.0],
            "review_count": [30.0],
            "review_avg_rating": [4.7],
        })
        cand_vectors, hero_vectors = build_similarity_vectors(candidates, heroes)
        self.assertEqual(cand_vectors.shape[0], 2)
        self.assertEqual(hero_vectors.shape[0], 1)
        self.assertEqual(cand_vectors.shape[1], hero_vectors.shape[1])


class HeroSimilarityScoresTests(unittest.TestCase):

    def test_empty_hero_stores_returns_zeros(self):
        candidates = pd.DataFrame({
            "industry": ["카페"],
            "district": ["강남구"],
            "district_growth_rate": [0.1],
            "district_traffic_index": [100.0],
            "review_count": [0.0],
            "review_avg_rating": [0.0],
        })
        heroes = pd.DataFrame(columns=candidates.columns)
        scores = hero_similarity_scores(candidates, heroes)
        self.assertEqual(list(scores), [0.0])

    def test_same_industry_and_district_scores_higher_than_different(self):
        candidates = pd.DataFrame({
            "industry": ["카페", "정육점"],
            "district": ["강남구", "노원구"],
            "district_growth_rate": [0.15, 0.15],
            "district_traffic_index": [90.0, 90.0],
            "review_count": [0.0, 0.0],
            "review_avg_rating": [0.0, 0.0],
        })
        heroes = pd.DataFrame({
            "industry": ["카페"],
            "district": ["강남구"],
            "district_growth_rate": [0.15],
            "district_traffic_index": [90.0],
            "review_count": [30.0],
            "review_avg_rating": [4.7],
        })
        scores = hero_similarity_scores(candidates, heroes)
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
