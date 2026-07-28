import unittest
from app.services import validation


class ValidateTests(unittest.TestCase):
    def test_passes_when_recommendation_matches_diagnosis(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": True}}
        recommendation = {
            "문제유형": [{"유형": "최근_매출감소", "근거": "전분기 대비 -0.2"}],
            "대응방안_추천": [
                {"방안": "재방문 쿠폰 발급", "대상_문제": "최근_매출감소", "근거": "전분기 대비 -0.2",
                 "진단기반_우선순위점수": 0.8, "점수구성": {"문제영향도": 0.8}},
            ],
        }
        result = validation.validate(raw_diag, recommendation)
        self.assertTrue(result["검증_통과"])
        self.assertEqual(result["발견사항"], [])

    def test_flags_recommendation_referencing_unknown_problem(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": True}}
        recommendation = {
            "문제유형": [],
            "대응방안_추천": [
                {"방안": "재방문 쿠폰 발급", "대상_문제": "최근_매출감소", "근거": "전분기 대비 -0.2"},
            ],
        }
        result = validation.validate(raw_diag, recommendation)
        self.assertFalse(result["검증_통과"])
        self.assertIn("검증경고", recommendation["대응방안_추천"][0])

    def test_flags_mismatched_reason_text(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": True}}
        recommendation = {
            "문제유형": [{"유형": "최근_매출감소", "근거": "원본 근거"}],
            "대응방안_추천": [
                {"방안": "재방문 쿠폰 발급", "대상_문제": "최근_매출감소", "근거": "다른 문구"},
            ],
        }
        result = validation.validate(raw_diag, recommendation)
        self.assertFalse(result["검증_통과"])

    def test_flags_gating_inconsistency(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": False}}
        recommendation = {
            "문제유형": [{"유형": "최근_매출감소", "근거": "x"}],
            "대응방안_추천": [{"방안": "재방문 쿠폰 발급", "대상_문제": "최근_매출감소", "근거": "x"}],
        }
        result = validation.validate(raw_diag, recommendation)
        self.assertFalse(result["검증_통과"])
        self.assertTrue(any("신뢰도 게이팅" in f for f in result["발견사항"]))

    def test_empty_recommendation_passes(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": False}}
        recommendation = {"문제유형": [], "대응방안_추천": []}
        result = validation.validate(raw_diag, recommendation)
        self.assertTrue(result["검증_통과"])

    def test_flags_low_business_impact(self):
        raw_diag = {"6_신뢰도": {"분석사용가능": True}}
        recommendation = {
            "문제유형": [{"유형": "요일_약세", "근거": "일요일 약세", "매출비중": 0.01}],
            "대응방안_추천": [{"방안": "요일 프로모션", "대상_문제": "요일_약세", "근거": "일요일 약세",
                           "진단기반_우선순위점수": 0.1, "점수구성": {"문제영향도": 0.01}}],
        }
        result = validation.validate(raw_diag, recommendation)
        self.assertFalse(result["검증_통과"])
        self.assertEqual(result["과정검증"]["사업영향도"], "실패")


if __name__ == "__main__":
    unittest.main()
