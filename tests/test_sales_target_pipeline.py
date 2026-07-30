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

from app.sales_target.pipeline import generate_sales_targets, to_bulk_upsert_items


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

        result = generate_sales_targets(
            candidate_registry=candidates,
            our_stores=our_stores,
            trdar_boundary=trdar_boundary,
            district_sales_growth=district_sales_growth,
            district_foot_traffic_index=district_foot_traffic_index,
        )

        # 1) 이미 가맹점인 업장은 결과에서 빠져야 한다.
        self.assertNotIn("이미가맹점카페", result["bizesNm"].tolist())
        self.assertEqual(len(result), 2)

        # 2) 각 후보가 올바른 상권에 매핑됐는지.
        root_coffee = result[result["bizesNm"] == "루트커피(T1)"].iloc[0]
        oven_glow = result[result["bizesNm"] == "오븐글로우(T2)"].iloc[0]
        self.assertEqual(root_coffee["trdar_cd"], "T1")
        self.assertEqual(oven_glow["trdar_cd"], "T2")

        # 3) 필요한 점수 컬럼이 다 채워졌는지.
        for col in ("growth_score", "traffic_score", "review_score", "similarity_score", "final_score"):
            self.assertIn(col, result.columns)

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

        result = generate_sales_targets(
            candidates, our_stores, trdar_boundary,
            district_sales_growth, district_foot_traffic_index, top_n=1,
        )

        self.assertEqual(len(result), 1)

    def test_empty_candidate_registry_returns_empty(self):
        empty = pd.DataFrame(columns=["bizesNm", "rdnmAdr", "lon", "lat"])
        result = generate_sales_targets(
            empty, pd.DataFrame({"address": []}), _trdar_boundary_fixture(),
            pd.DataFrame(columns=["trdar_cd", "growth_rate"]),
            pd.DataFrame(columns=["trdar_cd", "traffic_index"]),
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
        )

        self.assertTrue(result.empty)

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
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["growth_score"], 0.0)

    def test_hero_stores_produces_differentiated_similarity_score(self):
        # T1 후보 두 곳(indsLclsNm="카페"/"정육점")과 signguNm이 fixture 주소에서 추출되는 구와
        # 일치하도록 구성. hero 매장 하나(category="카페", 강남구 소재)를 넘기면, 업종/지역이
        # 겹치는 카페 후보의 similarity_score가 정육점 후보보다 높아야 한다.
        candidates = pd.DataFrame({
            "bizesNm": ["카페후보(T1)", "정육점후보(T1)"],
            "indsLclsNm": ["카페", "정육점"],
            "signguNm": ["강남구", "강남구"],
            "rdnmAdr": ["서울 강남구 아무로 1", "서울 강남구 아무로 2"],
            "lon": [127.0, 127.0],
            "lat": [37.5, 37.5],
        })
        our_stores = pd.DataFrame({"address": []})
        trdar_boundary = _trdar_boundary_fixture()
        district_sales_growth = pd.DataFrame({"trdar_cd": ["T1"], "growth_rate": [0.2]})
        district_foot_traffic_index = pd.DataFrame({"trdar_cd": ["T1"], "traffic_index": [100.0]})
        hero_stores = pd.DataFrame({
            "category": ["카페"],
            "address": ["서울 강남구 어딘가 3"],
            "reviewCount": [50.0],
            "reviewAvgRating": [4.8],
        })

        result = generate_sales_targets(
            candidates, our_stores, trdar_boundary,
            district_sales_growth, district_foot_traffic_index,
            hero_stores=hero_stores,
        )

        cafe_row = result[result["bizesNm"] == "카페후보(T1)"].iloc[0]
        butcher_row = result[result["bizesNm"] == "정육점후보(T1)"].iloc[0]
        self.assertGreater(cafe_row["similarity_score"], butcher_row["similarity_score"])
        # placeholder(50.0)를 그대로 쓰지 않고 실제로 계산됐는지 확인
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
        }])

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


if __name__ == "__main__":
    unittest.main()
