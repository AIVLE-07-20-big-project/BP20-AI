import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from scripts.response_strategy.bandit import BanditLoadMismatch, NeuralContextualBandit, retrain_cli

ARMS = ["쿠폰_20%", "이벤트_주말", "SNS_홍보"]


class ColdStartTests(unittest.TestCase):
    def test_no_data_falls_back_to_prior_bias(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS,
                                         prior_bias={"이벤트_주말": 5.0})
        ctx = np.zeros(4)
        result = bandit.select_arm(ctx)
        self.assertEqual(result["선택된_arm"], "이벤트_주말")
        self.assertEqual(result["표본수"], 0)

    def test_unseen_arm_has_nonzero_exploration_width(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=1.0)
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        result = bandit.select_arm(ctx)
        self.assertGreater(result["불확실성_폭"], 0.0)


class UpdateTests(unittest.TestCase):
    def test_update_reduces_uncertainty_for_that_arm(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=1.0)
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        width_before = bandit.select_arm(ctx)["불확실성_폭"]

        for _ in range(20):
            bandit.update(ctx, arm_index=0, reward=1.0)

        z = bandit._encode(ctx)
        a_inv = np.linalg.inv(bandit.A[0])
        width_after = bandit.alpha * float(np.sqrt(z @ a_inv @ z))
        self.assertLess(width_after, width_before)

    def test_high_reward_arm_eventually_wins_over_prior(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=0.1,
                                         prior_bias={"이벤트_주말": 5.0})
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        for _ in range(50):
            bandit.update(ctx, arm_index=0, reward=10.0)
        result = bandit.select_arm(ctx)
        self.assertEqual(result["선택된_arm"], "쿠폰_20%")

    def test_buffer_accumulates(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        bandit.update(ctx, arm_index=1, reward=2.0)
        bandit.update(ctx, arm_index=2, reward=1.0)
        self.assertEqual(len(bandit.buffer), 2)
        self.assertEqual(bandit.select_arm(ctx)["표본수"], 2)


class RetrainTests(unittest.TestCase):
    def test_raises_when_buffer_too_small(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        bandit.update(np.zeros(4), arm_index=0, reward=1.0)
        with self.assertRaises(ValueError):
            bandit.retrain_encoder(min_samples=10)

    def test_retrain_runs_and_rebuilds_linear_heads(self):
        rng = np.random.default_rng(0)
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        for _ in range(30):
            ctx = rng.normal(size=4)
            arm = int(rng.integers(0, len(ARMS)))
            bandit.update(ctx, arm_index=arm, reward=float(rng.normal()))

        loss = bandit.retrain_encoder(epochs=5, min_samples=10)

        self.assertFalse(np.isnan(loss))
        self.assertEqual(bandit.A.shape, (len(ARMS), bandit.encoding_dim, bandit.encoding_dim))
        self.assertEqual(len(bandit.buffer), 30)


class PropensityTests(unittest.TestCase):
    def test_propensities_sum_to_one_and_match_argmax_choice(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=1.0)
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        result = bandit.select_arm(ctx)

        self.assertAlmostEqual(sum(result["arm별_propensity"].values()), 1.0, places=4)
        self.assertEqual(result["propensity"], result["arm별_propensity"][result["선택된_arm"]])
        self.assertGreater(result["propensity"], 0.0)

    def test_top_arm_choice_unaffected_by_temperature(self):
        """propensity 계산(softmax)이 top-1 선택 로직(argmax) 자체를 바꾸면 안 된다."""
        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        low_temp = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=1.0, temperature=0.1)
        high_temp = NeuralContextualBandit(context_dim=4, arms=ARMS, alpha=1.0, temperature=10.0)
        self.assertEqual(low_temp.select_arm(ctx)["선택된_arm"], high_temp.select_arm(ctx)["선택된_arm"])


class SaveLoadTests(unittest.TestCase):
    def test_round_trip_preserves_selection_and_sample_count(self):
        rng = np.random.default_rng(0)
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, policy_version="v1")
        for _ in range(15):
            ctx = rng.normal(size=4)
            bandit.update(ctx, arm_index=int(rng.integers(0, len(ARMS))), reward=float(rng.normal()))

        ctx = np.array([0.1, 0.2, -0.1, 0.3])
        before = bandit.select_arm(ctx)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bandit.pt"
            bandit.save(path)
            restored = NeuralContextualBandit.load(path, context_dim=4, arms=ARMS)

        after = restored.select_arm(ctx)
        self.assertEqual(before["선택된_arm"], after["선택된_arm"])
        self.assertEqual(before["표본수"], after["표본수"])
        self.assertEqual(restored.policy_version, "v1")

    def test_mismatched_arms_raises_load_mismatch(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bandit.pt"
            bandit.save(path)
            with self.assertRaises(BanditLoadMismatch):
                NeuralContextualBandit.load(path, context_dim=4, arms=["다른_arm"])

    def test_mismatched_context_dim_raises_load_mismatch(self):
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bandit.pt"
            bandit.save(path)
            with self.assertRaises(BanditLoadMismatch):
                NeuralContextualBandit.load(path, context_dim=6, arms=ARMS)


class RetrainCliTests(unittest.TestCase):
    def test_missing_active_model_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch("scripts.response_strategy.bandit.BANDIT_MODEL_DIR", Path(tmp)):
                result = retrain_cli("없는등급")
        self.assertEqual(result["상태"], "실패")

    def test_insufficient_buffer_fails_gracefully(self):
        rng = np.random.default_rng(0)
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS)
        bandit.update(rng.normal(size=4), arm_index=0, reward=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            active_path = Path(tmp) / "등급A" / "active.pt"
            bandit.save(active_path)
            with unittest.mock.patch("scripts.response_strategy.bandit.BANDIT_MODEL_DIR", Path(tmp)):
                result = retrain_cli("등급A", min_samples=10)
        self.assertEqual(result["상태"], "실패")

    def test_enough_buffer_produces_candidate_version_without_overwriting_active(self):
        rng = np.random.default_rng(0)
        bandit = NeuralContextualBandit(context_dim=4, arms=ARMS, policy_version="coldstart")
        for _ in range(20):
            bandit.update(rng.normal(size=4), arm_index=int(rng.integers(0, len(ARMS))),
                          reward=float(rng.normal()))

        with tempfile.TemporaryDirectory() as tmp:
            active_path = Path(tmp) / "등급A" / "active.pt"
            bandit.save(active_path)
            with unittest.mock.patch("scripts.response_strategy.bandit.BANDIT_MODEL_DIR", Path(tmp)):
                result = retrain_cli("등급A", min_samples=10, epochs=3)

            self.assertEqual(result["상태"], "완료(수동 검토 필요)")
            self.assertEqual(result["표본수"], 20)
            self.assertTrue(Path(result["후보_경로"]).exists())

            restored_active = NeuralContextualBandit.load_any(active_path)
            self.assertEqual(restored_active.policy_version, "coldstart")


if __name__ == "__main__":
    unittest.main()
