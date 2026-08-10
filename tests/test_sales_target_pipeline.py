# 대상 경로(AI 레포): tests/test_sales_target_pipeline.py
#
# generate_sales_targets()만 테스트한다(순수 함수, 네트워크 없음). collect_and_generate()는
# 실제 콜렉터를 호출하는 얇은 글루 코드라 여기선 테스트하지 않는다 — 컬럼명 검증이 끝난 뒤
# 실제 서버 대상으로 한 번 돌려보는 방식으로 검증하는 게 더 실질적이다(store_registry 때와 동일).
#
# trdar_boundary의 XCNTS_VALUE/YDNTS_VALUE는 pyproj로 역산해서 lon=127.0,lat=37.5(T1)와
# lon=127.1,lat=37.5(T2)로 정확히 변환되도록 만든 값이다(EPSG:4326 -> EPSG:5181 역변환).

import unittest

import numpy as np
import pandas as pd

from app.sales_target.hero_store import select_hero_stores
from app.sales_target.pipeline import apply_review_scores, generate_sales_targets, to_bulk_upsert_items


def _trdar_boundary_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "TRDAR_CD": ["T1", "T2"],
        "XCNTS_VALUE": ["200000.0", "208842.54558833793"],
        "YDNTS_VALUE": ["444504.123485596", "444508.8210425943"],
        "RELM_AR": ["10000", "10000"],
    })


class GenerateSalesTargetsTests(unittest.TestCase):

    def test_excludes_existing_merchant_maps_to_district_and_ranks_by_score(self):
        candidates = pd.DataFrame({
            "bizesNm": ["이미가맹점카페", "루트커피(T1)", "오븐글로우(T2)"],
            "rdnmAdr": ["서울 강남구 테스트로 1", "서울 마포구 성지길 5", "서울 종로구 아무데나 9"],
            "lon": [127.1, 127.0, 127.1],
            "lat": [37.5, 37.5, 37.5],
        })
        our_stores = pd.DataFrame({"address": ["서울 강남구 테스트로 1"]})
        trdar_boundary = _trdar_boundary_fixture()
        # T1은 성장률이 높고 유동인구도 많음, T2는 성장률이 낮고 유동인구도 적음
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1", "T2"], "growth_rate": [0.5, -0.5]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1", "T2"], "traffic_index": [1000.0, 10.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1", "T2"], "store_count": [50.0, 50.0]})

        result = generate_sales_targets(
            candidate_registry=candidates,
            our_stores=our_stores,
            trdar_boundary=trdar_boundary,
            district_sales_growth=district_sales_growth,
            district_foot_traffic_index=district_foot_traffic_index,
            district_store_count=district_store_count,
        )

        # 1) 이미 가맹점인 업장은 결과에서 빠져야 한다.
        self.assertNotIn("이미가맹점카페", result["bizesNm"].tolist())
        self.assertEqual(len(result), 2)

        # 2) 각 후보가 올바른 상권에 매핑됐는지.
        root_coffee = result[result["bizesNm"] == "루트커피(T1)"].iloc[0]
        oven_glow = result[result["bizesNm"] == "오븐글로우(T2)"].iloc[0]
        self.assertEqual(root_coffee["trdar_cd"], "T1")
        self.assertEqual(oven_glow["trdar_cd"], "T2")

        # 3) 필요한 점수 컬럼이 다 채워졌는지. review_score는 1차 스코어링 시점엔 리뷰 데이터가
        #    없으므로 컬럼은 존재하되 전부 NaN이어야 한다(2단계 apply_review_scores가 채운다).
        for col in ("growth_score", "traffic_score", "review_score", "similarity_score", "final_score"):
            self.assertIn(col, result.columns)
        self.assertTrue(result["review_score"].isna().all())

        # 4) T1 후보(성장률/유동인구 둘 다 높음)가 T2 후보보다 최종 점수가 높아야 하고,
        #    결과가 최종 점수 내림차순으로 정렬돼 있어야 한다.
        self.assertGreater(root_coffee["final_score"], oven_glow["final_score"])
        self.assertEqual(result.iloc[0]["bizesNm"], "루트커피(T1)")

        # 5) hero_stores를 안 넘기면(기본값 None) similarity_score는 placeholder 50.0이어야 한다
        #    (하위 호환성 확인 — 우수 가맹점 로직 추가 전 동작과 동일해야 함).
        self.assertTrue((result["similarity_score"] == 50.0).all())

    def test_top_n_truncates_result(self):
        candidates = pd.DataFrame({
            "bizesNm": ["A", "B", "C"],
            "rdnmAdr": ["addr-A", "addr-B", "addr-C"],
            "lon": [127.0, 127.0, 127.1],
            "lat": [37.5, 37.5, 37.5],
        })
        our_stores = pd.DataFrame({"address": []})
        trdar_boundary = _trdar_boundary_fixture()
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1", "T2"], "growth_rate": [0.1, 0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1", "T2"], "traffic_index": [100.0, 100.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1", "T2"], "store_count": [50.0, 50.0]})

        result = generate_sales_targets(
            candidates, our_stores, trdar_boundary,
            district_sales_growth, district_foot_traffic_index, district_store_count, top_n=1,
        )

        self.assertEqual(len(result), 1)

    def test_empty_candidate_registry_returns_empty(self):
        empty = pd.DataFrame(columns=["bizesNm", "rdnmAdr", "lon", "lat"])
        result = generate_sales_targets(
            empty, pd.DataFrame({"address": []}), _trdar_boundary_fixture(),
            pd.DataFrame(columns=["trdar_cd", "growth_rate"]),
            pd.DataFrame(columns=["trdar_cd", "traffic_index"]),
            pd.DataFrame(columns=["trdar_cd", "store_count"]),
        )
        self.assertTrue(result.empty)

    def test_all_candidates_excluded_returns_empty(self):
        candidates = pd.DataFrame({
            "bizesNm": ["가맹점A"],
            "rdnmAdr": ["서울 강남구 테스트로 1"],
            "lon": [127.1],
            "lat": [37.5],
        })
        our_stores = pd.DataFrame({"address": ["서울 강남구 테스트로 1"]})

        result = generate_sales_targets(
            candidates, our_stores, _trdar_boundary_fixture(),
            pd.DataFrame(columns=["trdar_cd", "growth_rate"]),
            pd.DataFrame(columns=["trdar_cd", "traffic_index"]),
            pd.DataFrame(columns=["trdar_cd", "store_count"]),
        )

        self.assertTrue(result.empty)

    def test_rejected_addresses_are_excluded_like_existing_merchants(self):
        # 5단계(피드백 루프): 영업팀이 EXCLUDED 처리한 주소는 our_stores와 별개로 후보에서 빠져야 한다.
        candidates = pd.DataFrame({
            "bizesNm": ["루트커피(T1)", "오븐글로우(T2)"],
            "rdnmAdr": ["서울 마포구 성지길 5", "서울 종로구 아무데나 9"],
            "lon": [127.0, 127.1],
            "lat": [37.5, 37.5],
        })
        our_stores = pd.DataFrame({"address": []})
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1", "T2"], "growth_rate": [0.1, 0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1", "T2"], "traffic_index": [100.0, 100.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1", "T2"], "store_count": [50.0, 50.0]})

        result = generate_sales_targets(
            candidates, our_stores, _trdar_boundary_fixture(),
            district_sales_growth, district_foot_traffic_index, district_store_count,
            rejected_addresses=["서울 마포구 성지길 5"],
        )

        self.assertEqual(result["bizesNm"].tolist(), ["오븐글로우(T2)"])

    def test_no_rejected_addresses_keeps_all_candidates(self):
        # rejected_addresses가 None/빈 리스트면 기존 동작(하위 호환)과 동일해야 한다.
        candidates = pd.DataFrame({
            "bizesNm": ["루트커피(T1)"],
            "rdnmAdr": ["서울 마포구 성지길 5"],
            "lon": [127.0],
            "lat": [37.5],
        })
        our_stores = pd.DataFrame({"address": []})
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1", "T2"], "growth_rate": [0.1, 0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1", "T2"], "traffic_index": [100.0, 100.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1", "T2"], "store_count": [50.0, 50.0]})

        result = generate_sales_targets(
            candidates, our_stores, _trdar_boundary_fixture(),
            district_sales_growth, district_foot_traffic_index, district_store_count,
            rejected_addresses=None,
        )

        self.assertEqual(len(result), 1)

    def test_candidate_with_no_matching_district_data_still_gets_a_row(self):
        # trdar_cd는 매핑되지만 district_sales_growth/traffic에 그 trdar_cd 데이터가 없는 경우
        # (아직 공공데이터가 안 갱신됐거나 하는 경우) -> NaN 점수 처리(growth_scores가 0점 처리)로
        # 죽지 않고 결과에는 남아야 한다.
        candidates = pd.DataFrame({
            "bizesNm": ["데이터없는후보"],
            "rdnmAdr": ["아무주소"],
            "lon": [127.0],
            "lat": [37.5],
        })
        our_stores = pd.DataFrame({"address": []})

        result = generate_sales_targets(
            candidates, our_stores, _trdar_boundary_fixture(),
            pd.DataFrame(columns=["trdar_cd", "growth_rate"]),
            pd.DataFrame(columns=["trdar_cd", "traffic_index"]),
            pd.DataFrame(columns=["trdar_cd", "store_count"]),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["growth_score"], 0.0)

    def test_hero_stores_produces_differentiated_similarity_score(self):
        # T1 후보 두 곳(indsMclsNm="카페"/"정육점")과 signguNm이 fixture 주소에서 추출되는 구와
        # 일치하도록 구성. hero 매장 하나(category="카페", 강남구 소재)를 넘기면, 업종/지역이
        # 겹치는 카페 후보의 similarity_score가 정육점 후보보다 높아야 한다.
        candidates = pd.DataFrame({
            "bizesNm": ["카페후보(T1)", "정육점후보(T1)"],
            "indsMclsNm": ["카페", "정육점"],
            "signguNm": ["강남구", "강남구"],
            "rdnmAdr": ["서울 강남구 아무로 1", "서울 강남구 아무로 2"],
            "lon": [127.0, 127.0],
            "lat": [37.5, 37.5],
        })
        our_stores = pd.DataFrame({"address": []})
        trdar_boundary = _trdar_boundary_fixture()
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1"], "growth_rate": [0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1"], "traffic_index": [100.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1"], "store_count": [80.0]})
        hero_stores = pd.DataFrame({
            "category": ["카페"],
            "address": ["서울 강남구 어딘가 3"],
            "reviewCount": [50.0],
            "reviewAvgRating": [4.8],
        })

        result = generate_sales_targets(
            candidates, our_stores, trdar_boundary,
            district_sales_growth, district_foot_traffic_index, district_store_count,
            hero_stores=hero_stores,
        )

        cafe_row = result[result["bizesNm"] == "카페후보(T1)"].iloc[0]
        butcher_row = result[result["bizesNm"] == "정육점후보(T1)"].iloc[0]
        self.assertGreater(cafe_row["similarity_score"], butcher_row["similarity_score"])
        # placeholder(50.0)를 그대로 쓰지 않고 실제로 계산됐는지 확인
        self.assertFalse((result["similarity_score"] == 50.0).all())

    def test_be_registry_shaped_our_stores_produce_real_hero_selection_end_to_end(self):
        # 5단계 착수 전 재검증: "리뷰 수집기가 없어서 우수 가맹점 선정이 placeholder로만 돈다"는
        # 로드맵 문서의 가정이 맞는지 확인한다. BE(StoreRegistryEntryResponse)가 실제로 내려주는
        # 필드명(businessNumber/name/category/address/salesGrowthRate/reviewCount/
        # reviewAvgRating/reviewRatingStd) 그대로 our_stores를 만들고, graph.py/pipeline.py가
        # 하는 것과 동일하게 select_hero_stores()에 그대로 넘겨 나온 결과를
        # generate_sales_targets()의 hero_stores로 써도 KeyError 없이 실제 유사도 점수가
        # 나오는지 끝까지 확인한다.
        our_stores = pd.DataFrame({
            "businessNumber": ["111", "222", "333"],
            "name": ["우수카페", "저성장카페", "평점불안정카페"],
            "category": ["카페", "카페", "카페"],
            "address": ["서울 강남구 어딘가 1", "서울 강남구 어딘가 2", "서울 강남구 어딘가 3"],
            "salesGrowthRate": [0.9, 0.05, 0.9],
            "reviewCount": [40, 40, 12],
            "reviewAvgRating": [4.8, 4.8, 4.8],
            "reviewRatingStd": [0.3, 0.3, 1.5],
        })
        hero_stores = select_hero_stores(our_stores)
        self.assertEqual(hero_stores["name"].tolist(), ["우수카페"])  # 성장률 낮은 것/평점 불안정한 것 제외 확인

        candidates = pd.DataFrame({
            "bizesNm": ["카페후보(T1)", "정육점후보(T1)"],
            "indsMclsNm": ["카페", "정육점"],
            "signguNm": ["강남구", "강남구"],
            "rdnmAdr": ["서울 강남구 아무로 1", "서울 강남구 아무로 2"],
            "lon": [127.0, 127.0],
            "lat": [37.5, 37.5],
        })
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1"], "growth_rate": [0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1"], "traffic_index": [100.0]})
        district_store_count = pd.DataFrame({"trdar_cd": ["T1"], "store_count": [80.0]})

        result = generate_sales_targets(
            candidates, pd.DataFrame({"address": []}), _trdar_boundary_fixture(),
            district_sales_growth, district_foot_traffic_index, district_store_count,
            hero_stores=hero_stores,
        )

        cafe_row = result[result["bizesNm"] == "카페후보(T1)"].iloc[0]
        butcher_row = result[result["bizesNm"] == "정육점후보(T1)"].iloc[0]
        self.assertGreater(cafe_row["similarity_score"], butcher_row["similarity_score"])
        self.assertFalse((result["similarity_score"] == 50.0).all())


class ToBulkUpsertItemsTests(unittest.TestCase):

    def test_converts_columns_to_expected_keys(self):
        ranked = pd.DataFrame({
            "bizesNm": ["카페A"],
            "indsLclsNm": ["카페"],
            "rdnmAdr": ["서울 강남구 1"],
            "final_score": [76.12],
            "growth_score": [100.0],
            "traffic_score": [94.49],
            "review_score": [50.0],
            "similarity_score": [50.0],
        })
        items = to_bulk_upsert_items(ranked)
        self.assertEqual(items, [{
            "businessName": "카페A",
            "industry": "카페",
            "address": "서울 강남구 1",
            "totalScore": 76.12,
            "growthScore": 100.0,
            "trafficScore": 94.49,
            "reviewScore": 50.0,
            "similarityScore": 50.0,
            "salesPitch": None,
        }])

    def test_sales_pitch_column_passes_through_when_present(self):
        # sales_pitch는 graph.py의 generate_pitch 노드(2단계)가 채워주는 컬럼이다.
        ranked = pd.DataFrame({
            "bizesNm": ["카페A"],
            "indsLclsNm": ["카페"],
            "rdnmAdr": ["서울 강남구 1"],
            "final_score": [76.12],
            "growth_score": [100.0],
            "traffic_score": [94.49],
            "review_score": [50.0],
            "similarity_score": [50.0],
            "sales_pitch": ["전환 가능성이 76%로 높습니다."],
        })
        items = to_bulk_upsert_items(ranked)
        self.assertEqual(items[0]["salesPitch"], "전환 가능성이 76%로 높습니다.")

    def test_drops_rows_missing_required_fields(self):
        ranked = pd.DataFrame({
            "bizesNm": ["카페A", np.nan],
            "indsLclsNm": ["카페", "정육점"],
            "rdnmAdr": ["서울 강남구 1", "서울 마포구 2"],
            "final_score": [76.12, 50.0],
            "growth_score": [100.0, 60.0],
            "traffic_score": [94.49, 40.0],
            "review_score": [50.0, 50.0],
            "similarity_score": [50.0, 50.0],
        })
        items = to_bulk_upsert_items(ranked)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["businessName"], "카페A")

    def test_empty_dataframe_returns_empty_list(self):
        ranked = pd.DataFrame(columns=["bizesNm", "rdnmAdr", "final_score"])
        self.assertEqual(to_bulk_upsert_items(ranked), [])

    def test_nan_review_score_becomes_none(self):
        # 1차 스코어링 직후(리뷰 미수집)에는 review_score가 NaN이다 — BE에는 null로 전달돼야 한다.
        ranked = pd.DataFrame({
            "bizesNm": ["카페A"],
            "indsLclsNm": ["카페"],
            "rdnmAdr": ["서울 강남구 1"],
            "final_score": [70.625],
            "growth_score": [80.0],
            "traffic_score": [70.0],
            "review_score": [float("nan")],
            "similarity_score": [60.0],
        })
        items = to_bulk_upsert_items(ranked)
        self.assertIsNone(items[0]["reviewScore"])


class ApplyReviewScoresTests(unittest.TestCase):
    """6단계(리뷰 2단계 반영): apply_review_scores()."""

    def _candidates(self) -> pd.DataFrame:
        # growth/traffic/similarity는 이미 계산이 끝난 상태, review_score는 1차 스코어링 직후라
        # NaN(리뷰 데이터 없음), final_score는 preliminary_scores(3축)로 계산된 상태로 가정한다.
        return pd.DataFrame({
            "bizesNm": ["리뷰있는카페", "리뷰없는카페"],
            "rdnmAdr": ["서울 마포구 성지길 5", "서울 강남구 테스트로 1"],
            "growth_score": [80.0, 80.0],
            "traffic_score": [70.0, 70.0],
            "review_score": [float("nan"), float("nan")],
            "similarity_score": [60.0, 60.0],
            "final_score": [70.625, 70.625],  # preliminary_scores(80, 70, 60)
        })

    def test_matched_candidate_gets_real_review_score_and_unmatched_is_excluded(self):
        review_stats = pd.DataFrame({
            "rdnmAdr": ["서울 마포구 성지길 5"],
            "review_count": [120],
            "avg_rating": [4.5],
            "days_since_latest_review": [3.0],
            "review_growth_trend": [-1.5],
        })

        result = apply_review_scores(self._candidates(), review_stats)

        # 매칭 안 된(구글 플레이스에서 못 찾은) 후보는 결과에서 아예 빠진다 — 리뷰 축을 빼고
        # 3축만으로 재가중합하면 오히려 순위가 부당하게 올라가는 왜곡이 있었다(2026-08-04
        # 실제 배치에서 critic_agent가 반복 flag).
        self.assertEqual(len(result), 1)
        matched = result.iloc[0]
        self.assertEqual(matched["bizesNm"], "리뷰있는카페")

        # percentile은 매칭 여부와 무관하게 두 후보(2건) 전체 기준으로 계산된 뒤 매칭된 쪽만
        # 남긴다. review_count/avg_rating은 매칭 후보가 더 커서 100%씩(0.4+0.3), recency는
        # 결측치가 matched와 동일한 값(3.0일)으로 채워져 동점 처리되어 75%(0.2), growth_trend는
        # unmatched(0으로 채움)보다 낮아 50%(0.1) -> 100*.4+100*.3+75*.2+50*.1 = 90.0.
        # final_score도 4축 가중합으로 재계산돼 3축만 반영한 이전 값(70.625)과 달라진다.
        self.assertAlmostEqual(matched["review_score"], 90.0)
        self.assertNotAlmostEqual(matched["final_score"], 70.625)

    def test_empty_review_stats_returns_candidates_unchanged(self):
        result = apply_review_scores(self._candidates(), pd.DataFrame())
        self.assertTrue(result["review_score"].isna().all())

    def test_result_is_sorted_by_updated_final_score(self):
        candidates = pd.DataFrame({
            "bizesNm": ["A", "B"],
            "rdnmAdr": ["주소A", "주소B"],
            "growth_score": [50.0, 50.0],
            "traffic_score": [50.0, 50.0],
            "review_score": [float("nan"), float("nan")],
            "similarity_score": [50.0, 50.0],
            "final_score": [50.0, 50.0],
        })
        # B만 리뷰가 압도적으로 좋아서 review_score가 A보다 높아져야 한다.
        review_stats = pd.DataFrame({
            "rdnmAdr": ["주소A", "주소B"],
            "review_count": [1, 500],
            "avg_rating": [3.0, 5.0],
            "days_since_latest_review": [365.0, 1.0],
            "review_growth_trend": [-30.0, -0.1],
        })

        result = apply_review_scores(candidates, review_stats)

        self.assertEqual(result.iloc[0]["bizesNm"], "B")
        self.assertGreater(result.iloc[0]["final_score"], result.iloc[1]["final_score"])


if __name__ == "__main__":
    unittest.main()
