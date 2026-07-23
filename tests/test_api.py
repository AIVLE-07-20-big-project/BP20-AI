import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import ingestion, pipeline


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    base = {
        "SVC_INDUTY_CD": "A", "SVC_INDUTY_CD_NM": "업종",
        "TRDAR_SE_CD": "R", "TRDAR_SE_CD_NM": "골목상권", "FRC_STOR_CO": 0,
        "OPBIZ_RT": 1, "CLSBIZ_RT": 1, "TOT_FLPOP_CO": 1000, "STOR_CO": 10,
    }
    quarters = [20244, 20251, 20252, 20253, 20254, 20261]
    for market in range(1, 22):
        for i, q in enumerate(quarters):
            rows.append({**base, "TRDAR_CD": market, "TRDAR_CD_NM": f"상권 {market}",
                         "STDR_YYQU_CD": q, "THSMON_SELNG_CO": 100 + i,
                         "THSMON_SELNG_AMT": (100 + i) * 1000})
    return pd.DataFrame(rows)


class ReportEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        full = _synthetic_panel()
        is_upload_cell = (full["TRDAR_CD"] == 1) & (full["STDR_YYQU_CD"] == 20261)


        self.upload_row = full[is_upload_cell].copy()
        ingestion._base_merged_cache = full[~is_upload_cell].copy()

        self._orig_paths = {
            "STORE_STATS": ingestion.STORE_STATS,
            "FOOT_TRAFFIC": ingestion.FOOT_TRAFFIC,
            "RESIDENT_POPULATION": ingestion.RESIDENT_POPULATION,
            "WORKPLACE_POPULATION": ingestion.WORKPLACE_POPULATION,
            "WEATHER_QUARTERLY": ingestion.WEATHER_QUARTERLY,
        }
        for attr in self._orig_paths:
            setattr(ingestion, attr, Path("no_such_file.csv"))

        self._orig_ai_path = pipeline.AI_MODEL_PATH
        self._orig_external_path = pipeline.EXTERNAL_RESULT_PATH
        pipeline.AI_MODEL_PATH = Path("no_such_model.pkl")
        pipeline.EXTERNAL_RESULT_PATH = Path("no_such_result.json")

    def tearDown(self):
        ingestion._base_merged_cache = None
        for attr, value in self._orig_paths.items():
            setattr(ingestion, attr, value)
        pipeline.AI_MODEL_PATH = self._orig_ai_path
        pipeline.EXTERNAL_RESULT_PATH = self._orig_external_path

    def test_upload_produces_diagnosis_report(self):
        csv_bytes = self.upload_row.to_csv(index=False).encode("utf-8")
        response = self.client.post(
            "/api/v1/reports",
            files={"file": ("upload.csv", csv_bytes, "text/csv")},
            data={"trdar_cd": "1", "svc_induty_cd": "A", "yyqu_cd": "20261"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("관측_변화_분석", body)
        self.assertIn("경고", body)

    def test_missing_required_column_returns_422(self):
        response = self.client.post(
            "/api/v1/reports",
            files={"file": ("bad.csv", b"TRDAR_CD\n1\n", "text/csv")},
            data={"trdar_cd": "1", "svc_induty_cd": "A"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_cell_returns_404(self):
        csv_bytes = self.upload_row.to_csv(index=False).encode("utf-8")
        response = self.client.post(
            "/api/v1/reports",
            files={"file": ("upload.csv", csv_bytes, "text/csv")},
            data={"trdar_cd": "999", "svc_induty_cd": "ZZZ", "yyqu_cd": "20261"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
