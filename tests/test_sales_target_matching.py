# 대상 경로(AI 레포): tests/test_sales_target_matching.py

import unittest

import pandas as pd

from app.sales_target.matching import exclude_existing_merchants, normalize_address


class NormalizeAddressTests(unittest.TestCase):

    def test_strips_province_alias(self):
        self.assertEqual(normalize_address("서울특별시 마포구 양화로 1"), "서울 마포구 양화로 1")
        self.assertEqual(normalize_address("서울시 마포구 양화로 1"), "서울 마포구 양화로 1")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_address("서울   마포구    양화로  1"), "서울 마포구 양화로 1")

    def test_cuts_at_detail_suffix(self):
        self.assertEqual(
            normalize_address("서울 마포구 양화로 1 101동 202호"),
            "서울 마포구 양화로 1",
        )
        self.assertEqual(
            normalize_address("서울 마포구 양화로 1 3층"),
            "서울 마포구 양화로 1",
        )

    def test_handles_none_and_empty(self):
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address("   "), "")


class ExcludeExistingMerchantsTests(unittest.TestCase):

    def test_removes_exact_normalized_match(self):
        candidates = pd.DataFrame({
            "bizesNm": ["루트커피", "오븐글로우"],
            "rdnmAdr": ["서울특별시 마포구 양화로 1 101동", "서울 마포구 성지길 5"],
        })
        our_stores = pd.DataFrame({"address": ["서울 마포구 양화로 1"]})

        result = exclude_existing_merchants(candidates, our_stores)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["bizesNm"], "오븐글로우")

    def test_keeps_non_matching_candidates(self):
        candidates = pd.DataFrame({
            "bizesNm": ["A", "B"],
            "rdnmAdr": ["서울 마포구 성지길 5", "서울 강남구 테헤란로 1"],
        })
        our_stores = pd.DataFrame({"address": ["서울 서초구 반포대로 1"]})

        result = exclude_existing_merchants(candidates, our_stores)

        self.assertEqual(len(result), 2)

    def test_does_not_falsely_exclude_on_blank_addresses(self):
        # 후보/자사 매장 둘 다 주소가 비어있는 경우, "빈 문자열끼리 일치"로 잘못 걸러지면 안 된다.
        candidates = pd.DataFrame({"bizesNm": ["A"], "rdnmAdr": [None]})
        our_stores = pd.DataFrame({"address": [""]})

        result = exclude_existing_merchants(candidates, our_stores)

        self.assertEqual(len(result), 1)

    def test_empty_candidates_returns_empty(self):
        candidates = pd.DataFrame(columns=["bizesNm", "rdnmAdr"])
        our_stores = pd.DataFrame({"address": ["서울 마포구 양화로 1"]})

        result = exclude_existing_merchants(candidates, our_stores)

        self.assertTrue(result.empty)

    def test_empty_our_stores_keeps_all_candidates(self):
        candidates = pd.DataFrame({
            "bizesNm": ["A", "B"],
            "rdnmAdr": ["서울 마포구 성지길 5", "서울 강남구 테헤란로 1"],
        })
        our_stores = pd.DataFrame({"address": []})

        result = exclude_existing_merchants(candidates, our_stores)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
