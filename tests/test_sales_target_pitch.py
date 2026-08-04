# 대상 경로(AI 레포): tests/test_sales_target_pitch.py
#
# app/sales_target/pitch.py 단위 테스트. generate_sales_pitch()가 (a) 프롬프트에 final_score를
# 정확히 반영하고 다른 숫자를 넣지 않는지, (b) LLM 호출을 llm= 인자로 주입할 수 있는지,
# (c) LLM이 실패해도 예외 없이 fallback 문구를 반환하는지 검증한다. 실제 OpenAI 호출은 안 한다.

import unittest

from app.sales_target.pitch import _FALLBACK_TEXT, _build_prompt, generate_sales_pitch


def _candidate(final_score=76.4, **overrides) -> dict:
    base = {
        "bizesNm": "루트커피",
        "indsLclsNm": "카페",
        "rdnmAdr": "서울 마포구 성지길 5",
        "growth_score": 100.0,
        "traffic_score": 80.0,
        "similarity_score": 60.0,
        "final_score": final_score,
    }
    base.update(overrides)
    return base


class BuildPromptTests(unittest.TestCase):

    def test_final_score_rounded_and_included(self):
        prompt, score = _build_prompt(_candidate(final_score=76.6), None)
        self.assertEqual(score, 77)
        self.assertIn("77", prompt)
        self.assertIn("전환 가능성이 77%로 높습니다", prompt)

    def test_score_clamped_to_0_100_range(self):
        _prompt, score = _build_prompt(_candidate(final_score=150.0), None)
        self.assertEqual(score, 100)

    def test_missing_final_score_defaults_to_zero(self):
        candidate = _candidate()
        del candidate["final_score"]
        _prompt, score = _build_prompt(candidate, None)
        self.assertEqual(score, 0)

    def test_no_hero_stores_uses_placeholder_line(self):
        prompt, _score = _build_prompt(_candidate(), None)
        self.assertIn("참고할 우수 가맹점 데이터가 아직 없으므로", prompt)

    def test_hero_stores_included_in_prompt(self):
        hero_stores = [{"category": "카페", "reviewAvgRating": 4.7}]
        prompt, _score = _build_prompt(_candidate(), hero_stores)
        self.assertIn("카페 우수 가맹점 사례", prompt)
        self.assertIn("4.7", prompt)

    def test_hero_store_from_different_industry_is_excluded(self):
        # 2026-08-04 실제 배치에서 확인된 문제: 노래방 후보에게 "과학·기술 분야 우수 가맹점"이
        # 근거로 언급됨 — 후보 업종(indsLclsNm)과 hero_stores의 category가 다르면 제외해야 한다.
        candidate = _candidate(indsLclsNm="예술·스포츠")
        hero_stores = [{"category": "과학·기술", "reviewAvgRating": 4.8}]
        prompt, _score = _build_prompt(candidate, hero_stores)
        self.assertNotIn("과학·기술 우수 가맹점 사례", prompt)
        self.assertIn("후보 업종과 일치하는 우수 가맹점 사례가 없으므로", prompt)

    def test_hero_store_matching_industry_is_kept_others_dropped(self):
        candidate = _candidate(indsLclsNm="카페")
        hero_stores = [
            {"category": "과학·기술", "reviewAvgRating": 4.8},
            {"category": "카페", "reviewAvgRating": 4.5},
        ]
        prompt, _score = _build_prompt(candidate, hero_stores)
        self.assertIn("카페 우수 가맹점 사례", prompt)
        self.assertNotIn("과학·기술 우수 가맹점 사례", prompt)


class GenerateSalesPitchTests(unittest.TestCase):

    def test_uses_injected_llm_and_returns_its_text(self):
        calls = []

        def fake_llm(prompt: str) -> str:
            calls.append(prompt)
            return "루트커피를 신규 영업 우선 후보로 추천합니다. 전환 가능성이 77%로 높습니다."

        result = generate_sales_pitch(_candidate(final_score=76.6), None, llm=fake_llm)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            result, "루트커피를 신규 영업 우선 후보로 추천합니다. 전환 가능성이 77%로 높습니다."
        )

    def test_llm_exception_falls_back_without_raising(self):
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("OpenAI API 오류(테스트용)")

        result = generate_sales_pitch(_candidate(), None, llm=failing_llm)

        self.assertEqual(result, _FALLBACK_TEXT)

    def test_blank_llm_response_falls_back(self):
        result = generate_sales_pitch(_candidate(), None, llm=lambda prompt: "   ")
        self.assertEqual(result, _FALLBACK_TEXT)


if __name__ == "__main__":
    unittest.main()
