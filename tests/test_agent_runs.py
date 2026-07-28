import unittest
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from rag.generator import DISCLAIMER



CUSTOMER_RECOVERY_CELL = {"trdar_cd": "3110835", "svc_induty_cd": "CS100009"}
NO_CANDIDATE_CELL = {"trdar_cd": "3120139", "svc_induty_cd": "CS100010"}
INSUFFICIENT_DATA_CELL = {"trdar_cd": "3130326", "svc_induty_cd": "CS300011"}


@pytest.mark.integration
class AgentRunsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._sampling_patches = [
            patch("app.services.response.graph.OPE_BOOTSTRAP_SAMPLES", 5),
            patch("app.services.response.graph.MEASURED_EFFECT_BOOTSTRAP_SAMPLES", 5),
        ]
        for sample_patch in self._sampling_patches:
            sample_patch.start()

    def tearDown(self):
        for sample_patch in reversed(self._sampling_patches):
            sample_patch.stop()

    def test_full_pipeline_reaches_await_approval(self):
        response = self.client.post("/api/v1/agent-runs", json=CUSTOMER_RECOVERY_CELL)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(body["문제유형"], "고객_회복")
        self.assertIsNotNone(body["selected_action"])
        self.assertIn(body["selected_action"]["방안"], action_rules_names())
        self.assertIsNotNone(body["scm_result"])
        self.assertIsNotNone(body["rag_evidence"])
        self.assertIsNotNone(body["ope_result"])
        self.assertIsNotNone(body["대기중_승인"])
        self.assertEqual(body["상태"], "검증 완료 — 승인 대기")


        get_response = self.client.get(f"/api/v1/agent-runs/{body['thread_id']}")
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["대기중_승인"]["선택된_방안"], body["selected_action"])

    def test_grade_without_candidates_ends_early_without_approval(self):
        response = self.client.post("/api/v1/agent-runs", json=NO_CANDIDATE_CELL)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertEqual(body["문제유형"], "구조_전환")
        self.assertEqual(body["candidate_actions"], [])
        self.assertIsNone(body["selected_action"])
        self.assertIsNone(body["대기중_승인"])
        self.assertEqual(body["상태"], "종료: 방안 후보 없음")

    def test_insufficient_diagnosis_confidence_ends_early(self):
        response = self.client.post("/api/v1/agent-runs", json=INSUFFICIENT_DATA_CELL)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()

        self.assertIsNone(body["문제유형"])
        self.assertIsNone(body["대기중_승인"])
        self.assertEqual(body["상태"], "종료: 진단 신뢰도 부족")
        self.assertIn("분석사용가능=False — 대응방안 추천을 진행하지 않음", body["warnings"])

    def test_unknown_thread_id_returns_404(self):
        response = self.client.get("/api/v1/agent-runs/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_reliable_negative_ope_routes_to_alternative(self):
        from app.services.response.graph import _reject_candidate, _route_after_validate

        state = {
            "selected_action": {"방안": "쿠폰발행"}, "retry_count": 0,
            "ope_result": {"판정": "사용가능", "기준정책_대비_차이": -0.1}, "warnings": [],
        }
        self.assertEqual(_route_after_validate(state), "reject_candidate")
        update = _reject_candidate(state)
        self.assertEqual(update["rejected_actions"], ["쿠폰발행"])
        self.assertEqual(update["retry_count"], 1)

    def test_human_edit_is_not_overridden_by_ope(self):
        from app.services.response.graph import _route_after_validate

        state = {
            "approval_status": "edited", "retry_count": 0,
            "ope_result": {"판정": "사용가능", "기준정책_대비_차이": -0.1},
        }
        self.assertEqual(_route_after_validate(state), "await_approval")


def action_rules_names() -> set[str]:
    from app.services.response.action_rules import ACTIONS
    return set(ACTIONS)


def _fake_llm(prompt: str) -> str:
    return f"이 방안은 참고 문헌상 효과가 있는 것으로 보입니다. {DISCLAIMER}"


def _generate_report_with_fake_llm(evidence, action_name, shop_context="", llm=None, max_retry=1):

    from rag.generator import generate_report as real_generate_report
    return real_generate_report(evidence, action_name, shop_context, llm=_fake_llm, max_retry=max_retry)


@pytest.mark.integration
class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self._sampling_patches = [
            patch("app.services.response.graph.OPE_BOOTSTRAP_SAMPLES", 5),
            patch("app.services.response.graph.MEASURED_EFFECT_BOOTSTRAP_SAMPLES", 5),
        ]
        for sample_patch in self._sampling_patches:
            sample_patch.start()

    def tearDown(self):
        for sample_patch in reversed(self._sampling_patches):
            sample_patch.stop()

    def _start(self, cell):
        response = self.client.post("/api/v1/agent-runs", json=cell)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_approve_reaches_generate_report_and_ends(self):
        with patch("app.services.response.graph.generate_report", _generate_report_with_fake_llm):
            body = self._start(CUSTOMER_RECOVERY_CELL)
            resume = self.client.post(
                f"/api/v1/agent-runs/{body['thread_id']}/resume", json={"결정": "approve"},
            )
        self.assertEqual(resume.status_code, 200, resume.text)
        result = resume.json()
        self.assertIsNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "리포트 생성 완료")
        self.assertTrue(result["final_report"]["verified"])
        self.assertIn("report", result["final_report"])

    def test_edit_switches_action_and_returns_to_await_approval(self):
        body = self._start(CUSTOMER_RECOVERY_CELL)
        original = body["selected_action"]["방안"]
        alt = next(c["방안"] for c in body["candidate_actions"] if c["방안"] != original)

        resume = self.client.post(
            f"/api/v1/agent-runs/{body['thread_id']}/resume",
            json={"결정": "edit", "수정_방안": alt},
        )
        self.assertEqual(resume.status_code, 200, resume.text)
        result = resume.json()
        self.assertEqual(result["selected_action"]["방안"], alt)
        self.assertIsNotNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "검증 완료 — 승인 대기")

    def test_edit_with_unknown_action_is_rejected_safely(self):
        body = self._start(CUSTOMER_RECOVERY_CELL)
        resume = self.client.post(
            f"/api/v1/agent-runs/{body['thread_id']}/resume",
            json={"결정": "edit", "수정_방안": "존재하지_않는_방안"},
        )
        self.assertEqual(resume.status_code, 200, resume.text)
        result = resume.json()
        self.assertIsNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "종료: 잘못된 edit 요청으로 반려")
        self.assertIn("edit 방안 '존재하지_않는_방안' — 후보 목록에 없어 반려 처리", result["warnings"])

    def test_reject_ends_without_report(self):
        body = self._start(CUSTOMER_RECOVERY_CELL)
        resume = self.client.post(
            f"/api/v1/agent-runs/{body['thread_id']}/resume", json={"결정": "reject"},
        )
        self.assertEqual(resume.status_code, 200, resume.text)
        result = resume.json()
        self.assertIsNone(result["대기중_승인"])
        self.assertEqual(result["상태"], "종료: 반려됨")
        self.assertIsNone(result.get("final_report"))

    def test_resume_on_unknown_thread_returns_404(self):
        resume = self.client.post("/api/v1/agent-runs/does-not-exist/resume", json={"결정": "approve"})
        self.assertEqual(resume.status_code, 404)

    def test_resume_when_not_awaiting_approval_returns_409(self):
        body = self._start(NO_CANDIDATE_CELL)
        resume = self.client.post(
            f"/api/v1/agent-runs/{body['thread_id']}/resume", json={"결정": "approve"},
        )
        self.assertEqual(resume.status_code, 409)


if __name__ == "__main__":
    unittest.main()
