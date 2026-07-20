import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.response_strategy.ope import (
    LoggedBatch,
    direct_method,
    doubly_robust,
    evaluate_policy,
    ips,
    selftest,
)

ARMS = ["A", "B"]


def _write_campaign_logs(tmp, n, seed=0, base_reward=None):
    base_reward = base_reward or {"A": 5.0, "B": 1.0}
    rng = np.random.default_rng(seed)
    actions = rng.choice(ARMS, size=n)
    rewards = np.array([base_reward[a] for a in actions]) + rng.normal(0, 0.5, size=n)
    contexts = rng.normal(size=(n, 6))
    rows = []
    for i in range(n):
        row = {
            "decision_id": f"d{i}", "svc_induty_cd": "A", "action_id": actions[i],
            "propensity": 0.5, "reward": rewards[i], "executed": True,
        }
        row.update({f"context_{j+1}": contexts[i, j] for j in range(6)})
        rows.append(row)
    path = Path(tmp) / "campaign_logs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class IpsTests(unittest.TestCase):
    def test_reduces_to_matched_reward_over_propensity(self):
        batch = LoggedBatch(
            contexts=np.zeros((3, 1)),
            actions=["A", "B", "A"],
            propensities=np.array([0.5, 0.5, 0.5]),
            rewards=np.array([1.0, 10.0, 3.0]),
        )
        result = ips(batch, target_action_fn=lambda c: "A")

        expected = (1.0 / 0.5 + 0.0 + 3.0 / 0.5) / 3
        self.assertAlmostEqual(result, expected, places=6)


class DirectMethodTests(unittest.TestCase):
    def test_recovers_arm_specific_linear_reward(self):
        rng = np.random.default_rng(0)
        n, d = 2000, 2
        contexts = rng.normal(size=(n, d))
        actions = rng.choice(ARMS, size=n)
        theta = {"A": np.array([1.0, -1.0]), "B": np.array([0.5, 0.5])}
        rewards = np.array([theta[a] @ c for a, c in zip(actions, contexts)])
        batch = LoggedBatch(contexts=contexts, actions=list(actions),
                             propensities=np.full(n, 0.5), rewards=rewards)

        est = direct_method(batch, target_action_fn=lambda c: "A", arms=ARMS, alpha=1e-6)
        true_value = float(np.mean(contexts @ theta["A"]))
        self.assertAlmostEqual(est, true_value, delta=0.05)


class DoublyRobustTests(unittest.TestCase):
    def test_matches_direct_method_when_model_is_exact(self):
        rng = np.random.default_rng(1)
        n, d = 2000, 2
        contexts = rng.normal(size=(n, d))
        actions = rng.choice(ARMS, size=n)
        theta = {"A": np.array([1.0, -1.0]), "B": np.array([0.5, 0.5])}
        rewards = np.array([theta[a] @ c for a, c in zip(actions, contexts)])
        batch = LoggedBatch(contexts=contexts, actions=list(actions),
                             propensities=np.full(n, 0.5), rewards=rewards)

        est_dm = direct_method(batch, target_action_fn=lambda c: "A", arms=ARMS, alpha=1e-6)
        est_dr = doubly_robust(batch, target_action_fn=lambda c: "A", arms=ARMS, alpha=1e-6)

        self.assertAlmostEqual(est_dr, est_dm, delta=0.01)


class SelftestTests(unittest.TestCase):
    def test_all_estimators_close_to_true_value(self):
        result = selftest(n=5000, seed=0)
        self.assertLess(result["IPS"]["오차"], 0.15)
        self.assertLess(result["DM"]["오차"], 0.05)
        self.assertLess(result["DR"]["오차"], 0.05)


class EvaluatePolicyTests(unittest.TestCase):
    def test_no_campaign_logs_file_is_unjudgeable(self):
        result = evaluate_policy(1, "A", target_action_fn=lambda c: "A",
                                  campaign_logs=Path("__no_such_file__.csv"))
        self.assertEqual(result["판정"], "판정불가")
        self.assertEqual(result["표본수"], 0)

    def test_single_arm_only_is_unjudgeable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_campaign_logs(tmp, n=10, base_reward={"A": 5.0, "B": 5.0})
            df = pd.read_csv(path)
            df = df[df["action_id"] == "A"]
            df.to_csv(path, index=False)
            result = evaluate_policy(1, "A", target_action_fn=lambda c: "A", campaign_logs=path)
        self.assertEqual(result["판정"], "판정불가")

    def test_better_policy_beats_uniform_random_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_campaign_logs(tmp, n=40, seed=1)
            result = evaluate_policy(1, "A", target_action_fn=lambda c: "A", campaign_logs=path)

        self.assertIn(result["판정"], {"탐색적", "사용가능"})
        self.assertEqual(result["표본수"], 40)
        self.assertGreater(result["정책가치_DR"], result["기준정책가치(균등랜덤)"])
        self.assertGreater(result["기준정책_대비_차이"], 0.0)
        self.assertEqual(sorted(result["비교_arm목록"]), ARMS)

    def test_enough_samples_and_ess_reach_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_campaign_logs(tmp, n=40, seed=2)
            result = evaluate_policy(1, "A", target_action_fn=lambda c: "A", campaign_logs=path)
        self.assertEqual(result["판정"], "사용가능")

    def test_too_few_samples_stays_exploratory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_campaign_logs(tmp, n=10, seed=3)
            result = evaluate_policy(1, "A", target_action_fn=lambda c: "A", campaign_logs=path)
        self.assertEqual(result["판정"], "탐색적")


if __name__ == "__main__":
    unittest.main()
