import unittest
from pathlib import Path

from app.services.response import action_rules

ROOT = Path(__file__).resolve().parents[1]


class ActionRulesTests(unittest.TestCase):
    def test_customer_recovery_returns_discount_bundle_delivery_actions(self):
        result = action_rules.candidate_actions("고객_회복")
        axes = {item["axis"] for item in result}
        self.assertEqual(axes, {"discount_coupon", "set_bundle", "delivery"})

    def test_differentiation_returns_menu_location_and_delivery_actions(self):
        result = action_rules.candidate_actions("차별화")
        axes = {item["axis"] for item in result}
        self.assertEqual(axes, {"store_menu_location", "delivery"})

    def test_structural_transition_has_no_candidates(self):

        self.assertEqual(action_rules.candidate_actions("구조_전환"), [])

    def test_observation_returns_welcome_and_review_campaign(self):
        result = action_rules.candidate_actions("관찰")
        names = {item["방안"] for item in result}
        self.assertEqual(names, {"웰컴 프로모션", "리뷰 관리 캠페인"})
        axes = {item["axis"] for item in result}
        self.assertEqual(axes, {"customer_acquisition"})

    def test_strength_expansion_returns_sns_and_partnership_campaign(self):
        result = action_rules.candidate_actions("강점_확대")
        names = {item["방안"] for item in result}
        self.assertEqual(names, {"브랜드 SNS 캠페인", "지역 제휴 마케팅"})
        axes = {item["axis"] for item in result}
        self.assertEqual(axes, {"customer_acquisition"})

    def test_unknown_grade_returns_empty_list_silently(self):
        self.assertEqual(action_rules.candidate_actions("존재하지_않는_등급"), [])

    def test_every_rag_axis_has_at_least_one_action(self):



        rag_axes = {"discount_coupon", "set_bundle", "delivery", "store_menu_location"}
        covered = set(action_rules.ACTION_TO_AXIS.values())
        self.assertTrue(rag_axes.issubset(covered))

    def test_customer_acquisition_axis_has_no_rag_corpus_yet(self):


        manifest_path = ROOT / "model" / "rag_index" / "export" / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("RAG 인덱스 산출물 없음")
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertNotIn("customer_acquisition", manifest.get("axis_counts", {}))

    def test_action_to_axis_matches_actions(self):
        self.assertEqual(
            action_rules.ACTION_TO_AXIS,
            {name: v["axis"] for name, v in action_rules.ACTIONS.items()},
        )


if __name__ == "__main__":
    unittest.main()
