import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.modeling import survival_eval
from scripts.modeling.survival_eval import _event_frame, evaluate


def _synthetic_survival_panel(path: Path, n_cells: int = 40, n_quarters: int = 8, seed: int = 0):
    """절반은 점포수가 꾸준히 줄어(이벤트), 절반은 유지되는(중도절단) 합성 패널."""
    rng = np.random.default_rng(seed)
    quarters = list(range(20241, 20241 + n_quarters))
    rows = []
    for cell in range(n_cells):
        declining = cell < n_cells // 2
        stor = 20
        for q in quarters:
            stor = max(1, stor - rng.integers(1, 4)) if declining else stor + rng.integers(0, 2)
            rows.append({
                "TRDAR_CD": cell, "SVC_INDUTY_CD": "A", "STDR_YYQU_CD": q,
                "THSMON_SELNG_AMT": 1_000_000 + rng.normal(0, 10_000),
                "STOR_CO": stor, "CLSBIZ_RT": rng.uniform(0, 5),
                "OPBIZ_RT": rng.uniform(0, 5), "TOT_FLPOP_CO": 10_000 + rng.normal(0, 100),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


class EventFrameTests(unittest.TestCase):
    def test_declining_cells_produce_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel_path = Path(tmp) / "panel.csv"
            _synthetic_survival_panel(panel_path)
            sf = _event_frame(panel_path)
        self.assertGreater(len(sf), 0)
        self.assertGreater(sf["event"].sum(), 0)


class EvaluateSmokeTests(unittest.TestCase):
    def test_evaluate_returns_valid_c_index_and_km_curves(self):
        from lifelines import CoxTimeVaryingFitter

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            panel_path = tmp / "panel.csv"
            _synthetic_survival_panel(panel_path)

            sf = _event_frame(panel_path)
            model = CoxTimeVaryingFitter(penalizer=0.1)
            model.fit(sf, id_col="cell", event_col="event", start_col="start", stop_col="stop")

            cox_path = tmp / "cox.pkl"
            with open(cox_path, "wb") as f:
                pickle.dump({"model": model, "covs": survival_eval.COVS}, f)
            report_path = tmp / "report.json"

            with patch.object(survival_eval, "COX", cox_path), \
                 patch.object(survival_eval, "REPORT_OUT", report_path):
                report = evaluate(panel=panel_path)

            self.assertTrue(report_path.exists())

        self.assertGreaterEqual(report["C_index"], 0.0)
        self.assertLessEqual(report["C_index"], 1.0)
        self.assertGreater(len(report["위험분위별_KM"]), 0)
        for v in report["위험분위별_KM"].values():
            self.assertIn("최종_생존율", v)
            self.assertGreater(len(v["곡선"]), 0)


if __name__ == "__main__":
    unittest.main()
