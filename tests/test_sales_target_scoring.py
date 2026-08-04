# 대상 경로(AI 레포): tests/test_sales_target_scoring.py

import unittest

import numpy as np
import pandas as pd

from app.sales_target.scoring import (
    WEIGHTS,
    cosine_similarity_scores,
    final_scores,
    foot_traffic_index,
    growth_scores,
    percentile_score,
    preliminary_scores,
    review_activity_score,
)


class GrowthScoresTests(unittest.TestCase):

    def test_empty_series_returns_empty(self):
        result = growth_scores(pd.Series([], dtype=float))
        self.assertTrue(result.empty)

    def test_top_10_percent_gets_100(self):
        # 10개 값, 명확히 가장 큰 값 하나가 상위 10%에 해당해야 한다.
        rates = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.50])
        result = growth_scores(rates)
        self.assertEqual(result.iloc[-1], 100.0)

    def test_declining_store_gets_20_when_not_in_top_bucket(self):
        rates = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01, -0.05, -0.10, -0.20])
        result = growth_scores(rates)
        # 가장 크게 하락한 마지막 값은 상위 10/30%에 못 들어가므로 20점.
        self.assertEqual(result.iloc[-1], 20.0)

    def test_nan_gets_zero(self):
        rates = pd.Series([0.1, 0.2, float("nan")])
        result = growth_scores(rates)
        self.assertEqual(result.iloc[-1], 0.0)


class FootTrafficIndexTests(unittest.TestCase):

    def test_weighted_sum_matches_pdf_formula(self):
        result = foot_traffic_index(pd.Series([100]), pd.Series([200]), pd.Series([300]))
        expected = 100 * 0.3 + 200 * 0.3 + 300 * 0.4
        self.assertAlmostEqual(result.iloc[0], expected)


class PercentileScoreTests(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertTrue(percentile_score(pd.Series([], dtype=float)).empty)

    def test_highest_value_gets_100(self):
        values = pd.Series([10, 20, 30, 40, 50])
        result = percentile_score(values)
        self.assertEqual(result.iloc[-1], 100.0)

    def test_monotonic_with_input_order(self):
        values = pd.Series([5, 1, 3])
        result = percentile_score(values)
        # 값이 클수록 점수도 커야 한다.
        self.assertGreater(result.iloc[0], result.iloc[2])
        self.assertGreater(result.iloc[2], result.iloc[1])


class ReviewActivityScoreTests(unittest.TestCase):

    def test_weighted_sum_matches_pdf_formula(self):
        result = review_activity_score(
            pd.Series([100.0]), pd.Series([100.0]), pd.Series([100.0]), pd.Series([100.0])
        )
        self.assertAlmostEqual(result.iloc[0], 100.0)

        result_zero = review_activity_score(
            pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.0])
        )
        self.assertAlmostEqual(result_zero.iloc[0], 0.0)



class CosineSimilarityScoresTests(unittest.TestCase):

    def test_identical_vectors_score_100(self):
        candidates = np.array([[1.0, 2.0, 3.0]])
        heroes = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        result = cosine_similarity_scores(candidates, heroes, top_k=2)
        self.assertAlmostEqual(result[0], 100.0, places=5)

    def test_orthogonal_vectors_score_50(self):
        candidates = np.array([[1.0, 0.0]])
        heroes = np.array([[0.0, 1.0]])
        result = cosine_similarity_scores(candidates, heroes, top_k=1)
        self.assertAlmostEqual(result[0], 50.0, places=5)

    def test_empty_hero_vectors_returns_zeros(self):
        candidates = np.array([[1.0, 2.0], [3.0, 4.0]])
        heroes = np.array([]).reshape(0, 2)
        result = cosine_similarity_scores(candidates, heroes)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_top_k_larger_than_available_heroes_does_not_error(self):
        candidates = np.array([[1.0, 0.0]])
        heroes = np.array([[1.0, 0.0]])
        result = cosine_similarity_scores(candidates, heroes, top_k=5)
        self.assertAlmostEqual(result[0], 100.0, places=5)


class FinalScoresTests(unittest.TestCase):

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_all_100_gives_100(self):
        hundred = pd.Series([100.0])
        result = final_scores(hundred, hundred, hundred, hundred)
        self.assertAlmostEqual(result.iloc[0], 100.0)

    def test_matches_pdf_weighting_with_distinct_inputs(self):
        result = final_scores(
            growth=pd.Series([100.0]),
            traffic=pd.Series([0.0]),
            review=pd.Series([0.0]),
            similarity=pd.Series([0.0]),
        )
        self.assertAlmostEqual(result.iloc[0], 30.0)  # growth 가중치 0.30만 반영


class PreliminaryScoresTests(unittest.TestCase):
    """1차 스코어링(리뷰 제외 3축) — scoring.preliminary_scores()."""

    def test_all_100_gives_100(self):
        hundred = pd.Series([100.0])
        result = preliminary_scores(hundred, hundred, hundred)
        self.assertAlmostEqual(result.iloc[0], 100.0)

    def test_weights_renormalized_without_review(self):
        # growth=100, traffic=0, similarity=0 -> growth 가중치(0.30)를 0.80으로 나눈 비율만 반영.
        result = preliminary_scores(pd.Series([100.0]), pd.Series([0.0]), pd.Series([0.0]))
        self.assertAlmostEqual(result.iloc[0], 100.0 * (0.30 / 0.80))

    def test_matches_final_scores_when_review_is_neutral_midpoint(self):
        # review=50(중간값)을 final_scores에 넣은 값과는 다르다는 것을 확인 — preliminary_scores는
        # review 축 자체가 없는 것처럼 재정규화하므로, review=50을 끼워 넣는 것과 결과가 같지 않다.
        growth, traffic, similarity = pd.Series([80.0]), pd.Series([60.0]), pd.Series([40.0])
        via_final_with_neutral_review = final_scores(growth, traffic, pd.Series([50.0]), similarity)
        via_preliminary = preliminary_scores(growth, traffic, similarity)
        self.assertNotAlmostEqual(via_preliminary.iloc[0], via_final_with_neutral_review.iloc[0])


if __name__ == "__main__":
    unittest.main()
