import unittest
from rag.generator import DISCLAIMER, build_prompt, generate_report, verify_output

EVIDENCE = {
    "query": "세트메뉴가 객단가에 미치는 효과",
    "axis": "set_bundle",
    "direction_refs": [
        {"doc_id": "KCI_TEST", "page": 2, "tier_label": "학술 실증", "score": 0.5, "text": "세트메뉴는 객단가를 높이는 경향이 있다."},
    ],
    "allowed_numbers": [
        {
            "value": "7.4%",
            "sentence": "세트메뉴는 일반 메뉴 대비 약 7.4% 객단가가 더 높았다.",
            "focus": "약 7.4% 객단가가 더 높았다.",
            "source_url": "https://example.com",
            "doc_id": "테스트문서",
            "page": 7,
            "tier_label": "플랫폼 자체데이터(실측 아님, 참고용)",
        },
    ],
    "has_magnitude": True,
}

EMPTY_EVIDENCE = {"query": "배달채널 확대 효과", "axis": "delivery", "direction_refs": [], "allowed_numbers": [], "has_magnitude": False}


class BuildPromptTests(unittest.TestCase):
    def test_includes_allowed_number_and_source(self):
        prompt = build_prompt(EVIDENCE, "세트메뉴 도입", "치킨 전문점")
        self.assertIn("7.4%", prompt)
        self.assertIn("테스트문서", prompt)
        self.assertIn(DISCLAIMER, prompt)

    def test_empty_evidence_instructs_no_numbers(self):
        prompt = build_prompt(EMPTY_EVIDENCE, "배달채널 확대")
        self.assertIn("수치 사용 금지", prompt)


class VerifyOutputTests(unittest.TestCase):
    def test_passes_when_only_allowed_numbers_and_disclaimer_present(self):
        text = f"세트메뉴 도입 시 약 7.4% 객단가 상승 사례가 있습니다. {DISCLAIMER}"
        result = verify_output(text, EVIDENCE)
        self.assertTrue(result.ok)
        self.assertEqual(result.violations, [])

    def test_flags_number_not_in_allowed_list(self):
        text = f"세트메뉴 도입 시 매출이 999% 상승합니다. {DISCLAIMER}"
        result = verify_output(text, EVIDENCE)
        self.assertFalse(result.ok)
        self.assertEqual(result.violations[0]["value"], "999%")

    def test_flags_missing_disclaimer(self):
        text = "세트메뉴 도입 시 약 7.4% 객단가 상승 사례가 있습니다."
        result = verify_output(text, EVIDENCE)
        self.assertFalse(result.ok)
        self.assertFalse(result.disclaimer_present)

    def test_empty_evidence_rejects_any_number(self):
        text = f"배달 채널 확대 시 매출이 30% 증가합니다. {DISCLAIMER}"
        result = verify_output(text, EMPTY_EVIDENCE)
        self.assertFalse(result.ok)
        self.assertEqual(result.violations[0]["value"], "30%")


class GenerateReportTests(unittest.TestCase):
    def test_llm_failure_is_returned_as_structured_unverified_result(self):
        def unavailable(_prompt):
            raise RuntimeError("service unavailable")

        out = generate_report(EVIDENCE, "세트메뉴 도입", llm=unavailable)
        self.assertFalse(out["verified"])
        self.assertIn("LLM 리포트 생성 실패", out["error"])

    def test_passes_on_first_attempt_when_llm_behaves(self):
        def good_llm(prompt):
            return f"세트메뉴 도입 시 약 7.4% 객단가 상승 사례가 있습니다. {DISCLAIMER}"

        out = generate_report(EVIDENCE, "세트메뉴 도입", "치킨 전문점", llm=good_llm)
        self.assertTrue(out["verified"])
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["evidence_refs"][0]["value"], "7.4%")

    def test_retries_once_and_recovers(self):
        calls = {"n": 0}

        def bad_then_good(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "세트메뉴 도입 시 매출이 999% 상승합니다."
            return f"세트메뉴 도입 시 약 7.4% 객단가 상승 사례가 있습니다. {DISCLAIMER}"

        out = generate_report(EVIDENCE, "세트메뉴 도입", llm=bad_then_good)
        self.assertEqual(out["attempts"], 2)
        self.assertTrue(out["verified"])

    def test_exhausts_retries_and_reports_failure_honestly(self):
        def always_bad(prompt):
            return "매출이 12345% 폭증합니다."

        out = generate_report(EVIDENCE, "세트메뉴 도입", llm=always_bad, max_retry=1)
        self.assertEqual(out["attempts"], 2)
        self.assertFalse(out["verified"])
        self.assertTrue(out["violations"])

    def test_hallucinated_number_on_empty_evidence_is_caught(self):
        def hallucinating_llm(prompt):
            return f"배달 채널 확대 시 매출이 30% 증가합니다. {DISCLAIMER}"

        out = generate_report(EMPTY_EVIDENCE, "배달채널 확대", llm=hallucinating_llm, max_retry=0)
        self.assertFalse(out["verified"])
        self.assertFalse(out["has_magnitude"])


if __name__ == "__main__":
    unittest.main()
