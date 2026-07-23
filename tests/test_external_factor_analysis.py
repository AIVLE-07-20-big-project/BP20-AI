import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pyproj import Transformer

from scripts.modeling.external_factor_analysis import build_anchor_events, build_subway_exposure


def _write_area_coords(path):
    pd.DataFrame({
        "TRDAR_CD": [1, 2],
        "XCNTS_VALUE": [1000.0, 10000.0],
        "YDNTS_VALUE": [1000.0, 10000.0],
    }).to_csv(path, index=False)


def _write_anchors(path):
    rows = [

        dict(JPSENM="대규모점포", X="1000", Y="1500",
             APVPERMYMD="2024-05-15", DCBYMD="", TRDSTATENM="영업/정상"),

        dict(JPSENM="준대규모점포", X="1000", Y="1000",
             APVPERMYMD="2024-05-15", DCBYMD="", TRDSTATENM="영업/정상"),

        dict(JPSENM="대규모점포", X="1000", Y="1000",
             APVPERMYMD="2010-01-01", DCBYMD="2024-08-20", TRDSTATENM="폐업"),

        dict(JPSENM="대규모점포", X="1000", Y="1000",
             APVPERMYMD="2024-05-15", DCBYMD="", TRDSTATENM="휴업"),

        dict(JPSENM="대규모점포", X="1000", Y="1000",
             APVPERMYMD="2015-01-01", DCBYMD="2024-09-01", TRDSTATENM="휴업"),

        dict(JPSENM="대규모점포", X="10000", Y="10000",
             APVPERMYMD="2024-05-15", DCBYMD="", TRDSTATENM="영업/정상"),
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class BuildAnchorEventsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        self.area_path = tmp_path / "area_coords.csv"
        self.anchor_path = tmp_path / "big_store.csv"
        self.out_path = tmp_path / "anchor_exposure_quarterly.csv"
        _write_area_coords(self.area_path)
        _write_anchors(self.anchor_path)
        self.result = build_anchor_events(
            raw_path=self.anchor_path, area_path=self.area_path, out_path=self.out_path,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, trdar_cd, quarter):
        match = self.result[
            (self.result["TRDAR_CD"] == trdar_cd) & (self.result["STDR_YYQU_CD"] == quarter)
        ]
        return match.iloc[0] if not match.empty else None

    def test_quasi_large_store_excluded_from_open_count(self):
        row = self._row(1, 20242)
        self.assertIsNotNone(row)

        self.assertEqual(row["anchor_open_count"], 2)

    def test_actual_closure_counted_in_correct_quarter(self):
        row = self._row(1, 20243)
        self.assertIsNotNone(row)
        self.assertEqual(row["anchor_close_count"], 1)

    def test_suspended_status_never_creates_close_event(self):

        close_total = self.result["anchor_close_count"].sum()
        self.assertEqual(close_total, 1)

    def test_out_of_radius_area_not_matched(self):
        row = self._row(2, 20242)
        self.assertIsNotNone(row)
        self.assertEqual(row["anchor_open_count"], 1)

    def test_nearest_distance_picks_closest_event(self):
        row = self._row(1, 20242)
        self.assertAlmostEqual(row["anchor_open_nearest_m"], 0.0, places=3)


class BuildSubwayExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmp.name)
        self.raw_dir = tmp_path / "subway_data"
        self.raw_dir.mkdir()
        self.station_path = tmp_path / "subway_stations.csv"
        self.area_path = tmp_path / "area_coords.csv"
        self.out_path = tmp_path / "subway_exposure_quarterly.csv"




        to_5181 = Transformer.from_crs("EPSG:4326", "EPSG:5181", always_xy=True)
        self.near_x, self.near_y = to_5181.transform(127.0, 37.5)




        raw_csv = (
            '"사용일자","노선명","역명","승차총승객수","하차총승객수","등록일자"\n'
            '"20240101","1호선","테스트역","1000","800","20240104",""\n'
            '"20240102","2호선","테스트역","500","400","20240105",""\n'
            '"20240401","1호선","테스트역","2000","1800","20240404",""\n'
        )
        (self.raw_dir / "CARD_SUBWAY_MONTH_202401.csv").write_text(raw_csv, encoding="utf-8")

        pd.DataFrame({
            "lineNm": ["1호선"],
            "stnKrNm": ["테스트역"],
            "convX": [127.0],
            "convY": [37.5],
        }).to_csv(self.station_path, index=False)

        pd.DataFrame({
            "TRDAR_CD": [1, 2],
            "XCNTS_VALUE": [self.near_x, self.near_x + 50_000],
            "YDNTS_VALUE": [self.near_y, self.near_y + 50_000],
        }).to_csv(self.area_path, index=False)

        self.result = build_subway_exposure(
            raw_dir=self.raw_dir, station_path=self.station_path,
            area_path=self.area_path, out_path=self.out_path,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_station_name_different_lines_are_summed(self):

        row = self.result[(self.result["TRDAR_CD"] == 1) & (self.result["STDR_YYQU_CD"] == 20241)]
        self.assertFalse(row.empty)

        self.assertAlmostEqual(row["subway_exposure"].iloc[0], 2700.0, places=1)

    def test_dates_map_to_correct_quarter(self):
        q1 = self.result[(self.result["TRDAR_CD"] == 1) & (self.result["STDR_YYQU_CD"] == 20241)]
        q2 = self.result[(self.result["TRDAR_CD"] == 1) & (self.result["STDR_YYQU_CD"] == 20242)]
        self.assertFalse(q1.empty)
        self.assertFalse(q2.empty)
        self.assertAlmostEqual(q2["subway_exposure"].iloc[0], 2000.0 + 1800.0, places=1)

    def test_out_of_radius_area_excluded(self):

        row = self.result[(self.result["TRDAR_CD"] == 2) & (self.result["STDR_YYQU_CD"] == 20241)]
        self.assertTrue(row.empty)

    def test_nearest_distance_is_near_zero_for_colocated_area(self):
        row = self.result[(self.result["TRDAR_CD"] == 1) & (self.result["STDR_YYQU_CD"] == 20241)]
        self.assertAlmostEqual(row["subway_nearest_m"].iloc[0], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
