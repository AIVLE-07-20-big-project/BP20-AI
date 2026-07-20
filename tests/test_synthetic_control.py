import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.response_strategy.synthetic_control import (
    MAX_DONORS,
    build_donor_pool,
    counterfactual_baseline,
    fit_weights,
    measured_effect,
)

QUARTERS = [20241, 20242, 20243, 20244, 20251, 20252, 20253, 20254]


def _row(trdar_cd, svc, q, amt, se="발달상권"):
    return {
        "TRDAR_CD": trdar_cd, "SVC_INDUTY_CD": svc, "STDR_YYQU_CD": q,
        "TRDAR_CD_NM": f"상권{trdar_cd}", "TRDAR_SE_CD_NM": se,
        "SVC_INDUTY_CD_NM": "업종", "THSMON_SELNG_AMT": amt,
    }


class FitWeightsTests(unittest.TestCase):
    def test_recovers_exact_convex_combination(self):
        rng = np.random.default_rng(0)
        d1 = rng.uniform(500, 1500, size=8)
        d2 = rng.uniform(500, 1500, size=8)
        donors = pd.DataFrame({"A": d1, "B": d2})
        true_w = np.array([0.3, 0.7])
        target = pd.Series(np.expm1(true_w[0] * np.log1p(d1) + true_w[1] * np.log1p(d2)))

        w, rmse = fit_weights(target, donors)

        self.assertAlmostEqual(float(w.sum()), 1.0, places=3)
        self.assertLess(rmse, 1e-3)
        np.testing.assert_allclose(w, true_w, atol=0.05)


class BuildDonorPoolTests(unittest.TestCase):
    def test_excludes_target_industry_mismatch_and_short_history(self):
        rows = []
        for q in QUARTERS:
            rows.append(_row(1, "A", q, 1000))
            rows.append(_row(2, "A", q, 1000))
        for q in QUARTERS[:3]:
            rows.append(_row(3, "A", q, 1000))
        for q in QUARTERS:
            rows.append(_row(4, "B", q, 1000))
        panel = pd.DataFrame(rows)

        donors = build_donor_pool(panel, trdar_cd=1, svc_induty_cd="A", as_of_quarter=20254)

        self.assertNotIn(1, donors)
        self.assertIn(2, donors)
        self.assertNotIn(3, donors)
        self.assertNotIn(4, donors)

    def test_narrows_to_same_district_type_when_enough_peers(self):
        rows = [_row(1, "A", q, 1000, se="발달상권") for q in QUARTERS]
        for peer in range(2, 22):
            rows += [_row(peer, "A", q, 1000, se="발달상권") for q in QUARTERS]
        for peer in range(22, 27):
            rows += [_row(peer, "A", q, 1000, se="골목상권") for q in QUARTERS]
        panel = pd.DataFrame(rows)

        donors = build_donor_pool(panel, trdar_cd=1, svc_induty_cd="A", as_of_quarter=20254)

        self.assertTrue(all(22 > d for d in donors if d != 1))
        self.assertFalse(any(d in donors for d in range(22, 27)))


class CounterfactualBaselineTests(unittest.TestCase):
    def test_no_data_is_unjudgeable(self):
        panel = pd.DataFrame([_row(1, "A", 20254, 1000)])
        result = counterfactual_baseline(99, "A", 20254, panel=panel)
        self.assertEqual(result["판정"], "판정불가")

    def test_insufficient_history_is_unjudgeable(self):
        rows = [_row(1, "A", q, 1000) for q in QUARTERS[:3]]
        panel = pd.DataFrame(rows)
        result = counterfactual_baseline(1, "A", 20243, panel=panel)
        self.assertEqual(result["판정"], "판정불가")

    def test_insufficient_donor_pool_is_unjudgeable(self):
        rows = [_row(1, "A", q, 1000) for q in QUARTERS]
        rows += [_row(2, "A", q, 1000) for q in QUARTERS]
        panel = pd.DataFrame(rows)
        result = counterfactual_baseline(1, "A", 20254, panel=panel)
        self.assertEqual(result["판정"], "판정불가")

    def test_well_fit_donor_pool_produces_baseline_and_next_quarter(self):
        rows = []

        d2 = [1000, 1050, 1100, 1080, 1150, 1200, 1180, 1250]
        d3 = [800, 820, 850, 830, 900, 950, 920, 980]
        target = [round(np.expm1(0.5 * np.log1p(a) + 0.5 * np.log1p(b)))
                  for a, b in zip(d2, d3)]
        for i, q in enumerate(QUARTERS):
            rows.append(_row(1, "A", q, target[i]))
            rows.append(_row(2, "A", q, d2[i]))
            rows.append(_row(3, "A", q, d3[i]))
        for peer in range(4, 7):
            rows += [_row(peer, "A", q, 1000 + peer * 10) for q in QUARTERS]

        next_q = 20261
        rows.append(_row(2, "A", next_q, 1300))
        rows.append(_row(3, "A", next_q, 1000))
        for peer in range(4, 7):
            rows.append(_row(peer, "A", next_q, 1000 + peer * 10))
        panel = pd.DataFrame(rows)

        result = counterfactual_baseline(1, "A", 20254, panel=panel)

        self.assertIn(result["판정"], {"양호", "적합도 미달"})
        self.assertGreaterEqual(result["도너풀_크기"], 5)
        self.assertEqual(result["다음분기"], next_q)
        self.assertIsNotNone(result["다음분기_반사실_예상매출"])
        self.assertAlmostEqual(sum(result["가중치"].values()), 1.0, delta=0.05)

        self.assertGreater(result["가중치"].get(2, 0) + result["가중치"].get(3, 0), 0.8)
        self.assertLess(result["처치전_적합도_RMSE_로그스케일"], 0.01)


class DonorPoolSizeCapTests(unittest.TestCase):
    def test_large_donor_pool_is_capped_and_weights_stay_meaningful(self):
        """도너가 관측 분기수보다 훨씬 많으면(예: 1,400곳) 볼록조합 최적화가 과적합돼
        가중치가 흩어진다 — 실제 데이터(3001491/CS100001, 도너 1,400곳)에서 발견된 문제의
        회귀 테스트. 대상 셀과 거의 동일한 도너 2곳 + 무관한 도너 다수를 섞어, 유사도 상위
        MAX_DONORS로 추리면 그 2곳이 큰 가중치를 가져가는지 확인한다."""
        rng = np.random.default_rng(1)
        target_amt = [1000, 1050, 1100, 1080, 1150, 1200, 1180, 1250]
        rows = [_row(1, "A", q, target_amt[i]) for i, q in enumerate(QUARTERS)]

        for peer in (2, 3):
            rows += [_row(peer, "A", q, target_amt[i] + rng.integers(-5, 5))
                     for i, q in enumerate(QUARTERS)]

        for peer in range(4, 204):
            unrelated = [3000 - 50 * i + int(rng.integers(-20, 20)) for i in range(len(QUARTERS))]
            rows += [_row(peer, "A", q, unrelated[i]) for i, q in enumerate(QUARTERS)]
        panel = pd.DataFrame(rows)

        result = counterfactual_baseline(1, "A", 20254, panel=panel)

        self.assertLessEqual(result["도너풀_크기"], MAX_DONORS)
        self.assertGreater(result["가중치"].get(2, 0) + result["가중치"].get(3, 0), 0.5)


class MeasuredEffectTests(unittest.TestCase):
    def test_no_campaign_logs_file_is_unjudgeable(self):
        result = measured_effect(1, "A", "action_x", campaign_logs=Path("__no_such_file__.csv"))
        self.assertEqual(result["판정"], "판정불가")
        self.assertEqual(result["실측_사례수"], 0)

    def _panel(self):
        d2 = [1000, 1050, 1100, 1080, 1150, 1200, 1180, 1250]
        d3 = [800, 820, 850, 830, 900, 950, 920, 980]
        target = [round(np.expm1(0.5 * np.log1p(a) + 0.5 * np.log1p(b)))
                  for a, b in zip(d2, d3)]
        rows = []
        for i, q in enumerate(QUARTERS):
            rows.append(_row(1, "A", q, target[i]))
            rows.append(_row(2, "A", q, d2[i]))
            rows.append(_row(3, "A", q, d3[i]))
        for peer in range(4, 7):
            rows += [_row(peer, "A", q, 1000 + peer * 10) for q in QUARTERS]
        next_q = 20261
        rows.append(_row(1, "A", next_q, 1150))
        rows.append(_row(2, "A", next_q, 1300))
        rows.append(_row(3, "A", next_q, 1000))
        for peer in range(4, 7):
            rows.append(_row(peer, "A", next_q, 1000 + peer * 10))
        return pd.DataFrame(rows)

    def _write_logs(self, tmp, rows):
        path = Path(tmp) / "campaign_logs.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def test_recovers_positive_effect_from_logged_case(self):
        panel = self._panel()
        counterfactual = counterfactual_baseline(1, "A", 20254, panel=panel)["다음분기_반사실_예상매출"]

        with tempfile.TemporaryDirectory() as tmp:
            logs_path = self._write_logs(tmp, [{
                "decision_id": "d1", "trdar_cd": 1, "svc_induty_cd": "A",
                "yyqu_cd": 20253, "treatment_yyqu_cd": 20261, "action_id": "즉시할인",
                "executed": True, "revenue_after": counterfactual * 1.20,
            }])
            result = measured_effect(1, "A", "즉시할인", campaign_logs=logs_path, panel=panel)

        self.assertEqual(result["적용범위"], "동일 상권 실측")
        self.assertEqual(result["반사실_계산가능_사례수"], 1)
        self.assertAlmostEqual(result["효과율_평균"], 0.20, delta=0.02)

    def test_no_matching_action_is_unjudgeable(self):
        panel = self._panel()
        with tempfile.TemporaryDirectory() as tmp:
            logs_path = self._write_logs(tmp, [{
                "decision_id": "d1", "trdar_cd": 1, "svc_induty_cd": "A",
                "yyqu_cd": 20253, "treatment_yyqu_cd": 20261, "action_id": "다른방안",
                "executed": True, "revenue_after": 1000,
            }])
            result = measured_effect(1, "A", "즉시할인", campaign_logs=logs_path, panel=panel)
        self.assertEqual(result["판정"], "판정불가")

    def test_unexecuted_case_is_excluded(self):
        panel = self._panel()
        counterfactual = counterfactual_baseline(1, "A", 20254, panel=panel)["다음분기_반사실_예상매출"]
        with tempfile.TemporaryDirectory() as tmp:
            logs_path = self._write_logs(tmp, [{
                "decision_id": "d1", "trdar_cd": 1, "svc_induty_cd": "A",
                "yyqu_cd": 20253, "treatment_yyqu_cd": 20261, "action_id": "즉시할인",
                "executed": False, "revenue_after": counterfactual * 1.20,
            }])
            result = measured_effect(1, "A", "즉시할인", campaign_logs=logs_path, panel=panel)
        self.assertEqual(result["판정"], "판정불가")

    def test_falls_back_to_same_industry_pool_when_no_same_cell_case(self):
        panel = self._panel()
        with tempfile.TemporaryDirectory() as tmp:
            logs_path = self._write_logs(tmp, [{
                "decision_id": "d1", "trdar_cd": 2, "svc_induty_cd": "A",
                "yyqu_cd": 20253, "treatment_yyqu_cd": 20261, "action_id": "즉시할인",
                "executed": True, "revenue_after": 1400,
            }])

            result = measured_effect(1, "A", "즉시할인", campaign_logs=logs_path, panel=panel)
        self.assertIn("동일 업종", result["적용범위"])
        self.assertNotEqual(result["판정"], "판정불가")


if __name__ == "__main__":
    unittest.main()
