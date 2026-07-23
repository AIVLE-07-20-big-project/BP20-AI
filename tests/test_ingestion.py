import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.services import ingestion


def _write_csv(df: pd.DataFrame) -> Path:
    path = Path(tempfile.mktemp(suffix=".csv"))
    df.to_csv(path, index=False)
    return path


class ReadUploadTests(unittest.TestCase):
    def test_missing_required_columns_raises(self):
        csv_bytes = b"TRDAR_CD,SVC_INDUTY_CD\n1,A\n"
        with self.assertRaises(ingestion.IngestionSchemaError):
            ingestion.read_upload(csv_bytes)

    def test_invalid_csv_raises_schema_error(self):
        with self.assertRaises(ingestion.IngestionSchemaError):
            ingestion.read_upload(b"\x00\x01not a csv")

    def test_valid_upload_parses(self):
        csv_bytes = (
            b"TRDAR_CD,SVC_INDUTY_CD,STDR_YYQU_CD,THSMON_SELNG_AMT,THSMON_SELNG_CO\n"
            b"1,A,20261,1000000,100\n"
        )
        df = ingestion.read_upload(csv_bytes)
        self.assertEqual(df.shape[0], 1)


class MergeExternalTests(unittest.TestCase):
    def setUp(self):
        self.new_rows = pd.DataFrame([
            {"TRDAR_CD": 1, "SVC_INDUTY_CD": "A", "STDR_YYQU_CD": 20261,
             "THSMON_SELNG_AMT": 1_000_000, "THSMON_SELNG_CO": 100},
        ])
        store_stats = pd.DataFrame([
            {"TRDAR_CD": 1, "SVC_INDUTY_CD": "A", "STDR_YYQU_CD": 20261, "STOR_CO": 5},
        ])
        foot_traffic = pd.DataFrame([
            {"TRDAR_CD": 1, "STDR_YYQU_CD": 20261, "TOT_FLPOP_CO": 1000},
        ])
        self.store_path = _write_csv(store_stats)
        self.foot_path = _write_csv(foot_traffic)
        self._orig = {
            "STORE_STATS": ingestion.STORE_STATS,
            "FOOT_TRAFFIC": ingestion.FOOT_TRAFFIC,
            "RESIDENT_POPULATION": ingestion.RESIDENT_POPULATION,
            "WORKPLACE_POPULATION": ingestion.WORKPLACE_POPULATION,
            "WEATHER_QUARTERLY": ingestion.WEATHER_QUARTERLY,
        }
        ingestion.STORE_STATS = self.store_path
        ingestion.FOOT_TRAFFIC = self.foot_path
        ingestion.RESIDENT_POPULATION = Path("no_such_repop.csv")
        ingestion.WORKPLACE_POPULATION = Path("no_such_workpop.csv")
        ingestion.WEATHER_QUARTERLY = Path("no_such_weather.csv")

    def tearDown(self):
        self.store_path.unlink(missing_ok=True)
        self.foot_path.unlink(missing_ok=True)
        for attr, value in self._orig.items():
            setattr(ingestion, attr, value)

    def test_merge_joins_store_and_foot_traffic(self):
        merged, warnings = ingestion._merge_external(self.new_rows)
        self.assertEqual(merged["STOR_CO"].iloc[0], 5)
        self.assertEqual(merged["TOT_FLPOP_CO"].iloc[0], 1000)
        self.assertEqual(warnings, [])

    def test_build_combined_panel_replaces_existing_key(self):
        base = pd.DataFrame([
            {"TRDAR_CD": 1, "SVC_INDUTY_CD": "A", "STDR_YYQU_CD": 20261,
             "THSMON_SELNG_AMT": 500_000, "THSMON_SELNG_CO": 50},
            {"TRDAR_CD": 2, "SVC_INDUTY_CD": "B", "STDR_YYQU_CD": 20261,
             "THSMON_SELNG_AMT": 900_000, "THSMON_SELNG_CO": 90},
        ])
        ingestion._base_merged_cache = base
        try:
            combined, _ = ingestion.build_combined_panel(self.new_rows)
        finally:
            ingestion._base_merged_cache = None

        cell = combined[(combined["TRDAR_CD"] == 1) & (combined["SVC_INDUTY_CD"] == "A")]
        self.assertEqual(len(cell), 1)
        self.assertEqual(cell["THSMON_SELNG_AMT"].iloc[0], 1_000_000)
        self.assertEqual(len(combined), 2)

    def test_missing_external_files_warn_instead_of_raising(self):
        ingestion.STORE_STATS = Path("no_such_store.csv")
        ingestion.FOOT_TRAFFIC = Path("no_such_foot.csv")
        merged, warnings = ingestion._merge_external(self.new_rows)
        self.assertEqual(len(merged), 1)
        self.assertTrue(any("store_stats" in w for w in warnings))
        self.assertTrue(any("foot_traffic" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
