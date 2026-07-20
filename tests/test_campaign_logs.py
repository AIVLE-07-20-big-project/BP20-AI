import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.response.campaign_logs import SCHEMA_COLUMNS, validate_logs
from rag.generator import DISCLAIMER
from scripts.modeling.sales_analysis import shift_quarter

CUSTOMER_RECOVERY_CELL = {"trdar_cd": "3110835", "svc_induty_cd": "CS100009"}


def _fake_llm(prompt: str) -> str:
    return f"이 방안은 참고 문헌상 효과가 있는 것으로 보입니다. {DISCLAIMER}"


def _generate_report_with_fake_llm(evidence, action_name, shop_context="", llm=None, max_retry=1):
    from rag.generator import generate_report as real_generate_report
    return real_generate_report(evidence, action_name, shop_context, llm=_fake_llm, max_retry=max_retry)


@pytest.mark.integration
class CampaignLogsApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._tmp = tempfile.TemporaryDirectory()
        self.logs_path = Path(self._tmp.name) / "campaign_logs.csv"
        self.bandit_dir = Path(self._tmp.name) / "bandit"
        self.patchers = [
            patch("app.services.response.campaign_logs.CAMPAIGN_LOGS", self.logs_path),
            patch("app.services.response.bandit_store.BANDIT_MODEL_DIR", self.bandit_dir),
            patch("app.services.response.graph.OPE_BOOTSTRAP_SAMPLES", 5),
            patch("app.services.response.graph.MEASURED_EFFECT_BOOTSTRAP_SAMPLES", 5),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self._tmp.cleanup()

    def _approve(self):
        with patch("app.services.response.graph.generate_report", _generate_report_with_fake_llm):
            started = self.client.post("/api/v1/agent-runs", json=CUSTOMER_RECOVERY_CELL).json()
            resumed = self.client.post(
                f"/api/v1/agent-runs/{started['thread_id']}/resume", json={"결정": "approve"},
            ).json()
        return started, resumed

    def test_append_log_records_row_and_pulls_decision_time_values(self):
        started, resumed = self._approve()
        yyqu_cd = resumed["yyqu_cd"] or resumed["diagnosis"]["대상"]["기준분기"]
        treatment_q = shift_quarter(int(yyqu_cd), 1)

        response = self.client.post("/api/v1/campaign-logs", json={
            "thread_id": started["thread_id"], "executed": True,
            "treatment_yyqu_cd": treatment_q, "revenue_after": 123456789.0,
        })
        self.assertEqual(response.status_code, 200, response.text)
        row = response.json()

        self.assertEqual(row["action_id"], resumed["selected_action"]["방안"])
        self.assertEqual(row["데이터_출처"], "real")
        self.assertIsNotNone(row["propensity"])
        self.assertIsNotNone(row["revenue_before"])
        self.assertIsNotNone(row["reward"])
        self.assertTrue(self.logs_path.exists())

        saved = pd.read_csv(self.logs_path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(list(saved.columns), SCHEMA_COLUMNS)

        quality = self.client.get("/api/v1/campaign-logs/quality").json()
        self.assertEqual(quality["총행수"], 1)
        self.assertEqual(quality["유효행수"], 1)
        self.assertEqual(quality["실제_행수"], 1)


        self.assertTrue((self.bandit_dir / "고객_회복" / "active.pt").exists())

    def test_unexecuted_case_has_no_reward(self):
        started, _ = self._approve()
        response = self.client.post("/api/v1/campaign-logs", json={
            "thread_id": started["thread_id"], "executed": False,
            "treatment_yyqu_cd": 20999, "revenue_after": None,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["reward"])
        self.assertFalse(response.json()["executed"])

    def test_unknown_thread_id_returns_404(self):
        response = self.client.post("/api/v1/campaign-logs", json={
            "thread_id": "does-not-exist", "executed": True,
            "treatment_yyqu_cd": 20999, "revenue_after": 1.0,
        })
        self.assertEqual(response.status_code, 404)

    def test_not_yet_approved_returns_409(self):
        started = self.client.post("/api/v1/agent-runs", json=CUSTOMER_RECOVERY_CELL).json()
        response = self.client.post("/api/v1/campaign-logs", json={
            "thread_id": started["thread_id"], "executed": True,
            "treatment_yyqu_cd": 20999, "revenue_after": 1.0,
        })
        self.assertEqual(response.status_code, 409)


class ValidateLogsTests(unittest.TestCase):
    def _base_row(self, **overrides):
        row = {
            "decision_id": "d1", "trdar_cd": 1, "svc_induty_cd": "CS100009",
            "yyqu_cd": 20251, "treatment_yyqu_cd": 20252, "action_id": "즉시할인",
            **{f"context_{i}": 0.1 for i in range(1, 7)},
            "propensity": 0.5, "policy_version": "coldstart", "executed": True,
            "revenue_before": 1000.0, "revenue_after": 1100.0, "reward": 0.1,
            "데이터_출처": "real",
        }
        row.update(overrides)
        return row

    def _write(self, tmp, rows):
        path = Path(tmp) / "campaign_logs.csv"
        pd.DataFrame(rows, columns=SCHEMA_COLUMNS).to_csv(path, index=False)
        return path

    def test_missing_file_returns_zero_counts(self):
        result = validate_logs(Path("__no_such_file__.csv"))
        self.assertEqual(result, {"총행수": 0, "유효행수": 0, "제외행수": 0, "제외사유": {}})

    def test_all_valid_rows_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(), self._base_row(decision_id="d2")])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 2)
        self.assertEqual(result["제외행수"], 0)

    def test_duplicate_decision_id_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(), self._base_row()])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 1)
        self.assertEqual(result["제외사유"]["decision_id 중복"], 1)

    def test_unknown_action_id_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(action_id="존재하지_않는_방안")])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 0)
        self.assertEqual(result["제외사유"]["알 수 없는 action_id"], 1)

    def test_propensity_out_of_range_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(propensity=0.0), self._base_row(decision_id="d2", propensity=1.5)])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 0)
        self.assertEqual(result["제외사유"]["propensity 범위(0,1] 벗어남"], 2)

    def test_treatment_before_yyqu_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(treatment_yyqu_cd=20251)])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 0)
        self.assertEqual(result["제외사유"]["treatment_yyqu_cd가 yyqu_cd 이후가 아님"], 1)

    def test_executed_without_reward_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(reward=None)])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 0)
        self.assertEqual(result["제외사유"]["executed=True인데 reward 없음"], 1)

    def test_reward_mismatch_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(reward=0.9)])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 0)
        self.assertEqual(result["제외사유"]["reward 재계산 불일치"], 1)

    def test_unexecuted_row_without_reward_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [self._base_row(executed=False, reward=None, revenue_after=None)])
            result = validate_logs(path)
        self.assertEqual(result["유효행수"], 1)


if __name__ == "__main__":
    unittest.main()
