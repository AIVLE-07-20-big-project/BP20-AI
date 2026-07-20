import unittest
from pathlib import Path

import numpy as np

from app.core.config import RAG_INDEX_EXPORT
from rag.retriever import (
    RagIndex,
    dedup_numbers,
    extract_stat_contexts,
    split_sentences,
)

EXPORT_DIR = RAG_INDEX_EXPORT


class ExtractStatContextsTests(unittest.TestCase):
    """모델 로딩 없이 동작하는 순수 함수 테스트."""

    def test_extracts_percentage_with_surrounding_sentence(self):
        chunk = {"text": "이전 문장입니다. 세트메뉴가 7.4% 객단가를 올렸습니다. 다음 문장입니다."}
        out = extract_stat_contexts(chunk, window=1)
        values = [o["value"] for o in out]
        self.assertIn("7.4%", values)
        hit = next(o for o in out if o["value"] == "7.4%")
        self.assertIn("이전 문장입니다", hit["sentence"])
        self.assertIn("다음 문장입니다", hit["sentence"])

    def test_dedup_numbers_keeps_longest_context(self):
        allowed = [
            {"value": "10%", "sentence": "짧음"},
            {"value": "10%", "sentence": "이건 훨씬 더 긴 맥락 문장입니다"},
        ]
        result = dedup_numbers(allowed)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sentence"], "이건 훨씬 더 긴 맥락 문장입니다")

    def test_split_sentences_handles_newlines(self):
        sents = split_sentences("첫 문장입니다.\n둘째 문장이에요.")
        self.assertEqual(len(sents), 2)

    def test_search_falls_back_to_lexical_when_model_is_unavailable(self):
        chunks = [
            {"text": "쿠폰 할인 효과", "tier": "vendor", "axis": "discount_coupon", "contains_stat": False},
            {"text": "배달 채널 확대", "tier": "vendor", "axis": "discount_coupon", "contains_stat": False},
        ]
        idx = RagIndex(np.zeros((2, 2), dtype="float32"), chunks, {"model": "missing"})
        idx._model_error = "offline"
        results = idx.search("쿠폰 할인", k=1, tier="vendor", axis="discount_coupon")
        self.assertEqual(results[0][1]["text"], "쿠폰 할인 효과")


@unittest.skipUnless(EXPORT_DIR.exists(), "model/rag_index/export 산출물이 없음")
class RagIndexIntegrationTests(unittest.TestCase):
    """Colab에서 빌드한 실제 인덱스(embeddings.npy/chunks.jsonl/manifest.json)를 로드해
    검증한다. sentence-transformers로 BGE-M3를 로드하므로 최초 1회는 느릴 수 있다."""

    @classmethod
    def setUpClass(cls):
        cls.idx = RagIndex.load(EXPORT_DIR)

    def test_load_matches_manifest_chunk_count(self):
        self.assertEqual(len(self.idx.chunks), self.idx.manifest["n_chunks"])

    def test_embeddings_shape_matches_chunk_count_and_dim(self):
        self.assertEqual(self.idx.embeddings.shape, (len(self.idx.chunks), self.idx.manifest["dim"]))

    def test_search_tier_filter_returns_only_matching_tier(self):
        results = self.idx.search("할인 효과", k=5, tier="vendor", axis="discount_coupon")
        self.assertTrue(results)
        self.assertTrue(all(c["tier"] == "vendor" for _, c in results))

    def test_search_require_stat_filters_non_stat_chunks(self):
        results = self.idx.search("세트메뉴 효과", k=10, tier="vendor", require_stat=True)
        self.assertTrue(all(c["contains_stat"] for _, c in results))

    def test_build_evidence_set_bundle_axis_matches_documented_numbers(self):
        ev = self.idx.build_evidence("세트메뉴가 객단가에 미치는 효과", axis="set_bundle")
        values = {a["value"] for a in ev["allowed_numbers"]}

        self.assertTrue({"26.9%", "7.4%"} & values)
        self.assertTrue(ev["has_magnitude"])

    def test_build_evidence_delivery_axis_has_no_magnitude(self):

        ev = self.idx.build_evidence("배달채널 확대 효과", axis="delivery")
        self.assertFalse(ev["has_magnitude"])
        self.assertEqual(ev["allowed_numbers"], [])

    def test_build_evidence_never_uses_official_tier(self):

        for axis in ("discount_coupon", "set_bundle", "delivery", "store_menu_location", None):
            ev = self.idx.build_evidence("효과", axis=axis)
            for r in ev["direction_refs"]:
                self.assertNotEqual(r["tier_label"], "공공통계")

    def test_allowed_numbers_carry_source_and_context(self):
        ev = self.idx.build_evidence("세트메뉴가 객단가에 미치는 효과", axis="set_bundle")
        for a in ev["allowed_numbers"]:
            self.assertIn("sentence", a)
            self.assertIn("doc_id", a)
            self.assertEqual(a["tier_label"], "플랫폼 자체데이터(실측 아님, 참고용)")


if __name__ == "__main__":
    unittest.main()
