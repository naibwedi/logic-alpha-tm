import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from logic_alpha_tm import logicvol as lv
from logic_alpha_tm.data import synthetic_prices

SPEC = Path(__file__).resolve().parents[1] / "experiments" / "logicvol-v0.1.json"


def write_prices(path: Path, prices: pd.DataFrame) -> Path:
    prices.rename_axis("date").to_csv(path)
    return path


class TargetAndFeatures(unittest.TestCase):
    def setUp(self):
        self.prices = synthetic_prices(n=400, seed=3)
        self.r = lv.log_returns(self.prices)

    def test_target_matches_definition(self):
        y = lv.target(self.r["SPY"], 5)
        t = 200
        future = self.r["SPY"].iloc[t + 1:t + 6].to_numpy()
        expected = 0.5 * math.log(252 * np.mean(future ** 2))
        self.assertAlmostEqual(y.iloc[t], expected, places=10)
        self.assertTrue(np.isnan(y.iloc[-1]))

    def test_target_uses_only_the_next_five_sessions(self):
        t = 200
        base = lv.target(self.r["SPY"], 5).iloc[t]
        bumped = self.prices.copy()
        bumped.iloc[t + 7:, bumped.columns.get_loc("SPY")] *= 1.5  # changes return t+7 only
        self.assertAlmostEqual(lv.target(lv.log_returns(bumped)["SPY"], 5).iloc[t], base, places=12)
        bumped = self.prices.copy()
        bumped.iloc[t + 5:, bumped.columns.get_loc("SPY")] *= 1.5  # changes return t+5
        self.assertNotAlmostEqual(lv.target(lv.log_returns(bumped)["SPY"], 5).iloc[t], base, places=6)

    def test_features_never_use_the_future(self):
        t = 250
        base = lv.build_features(self.prices, "SPY").iloc[t]
        changed = self.prices.copy()
        changed.iloc[t + 1:] *= 0.7
        after = lv.build_features(changed, "SPY").iloc[t]
        pd.testing.assert_series_equal(base, after)

    def test_ewma_is_causal(self):
        t = 250
        base = lv.ewma_log_vol(self.r["SPY"]).iloc[t]
        changed = self.r["SPY"].copy()
        changed.iloc[t + 1:] = 0.2
        self.assertEqual(lv.ewma_log_vol(changed).iloc[t], base)


class FoldsAndMetrics(unittest.TestCase):
    def test_folds_are_purged_ordered_and_disjoint(self):
        dates = pd.bdate_range("2005-01-03", periods=2000)
        fl = lv.folds(dates, "2008-01-01", 756, 126, 5)
        seen = set()
        for train, test in fl:
            self.assertGreaterEqual(len(train), 756)
            self.assertEqual(test[0] - train[-1] - 1, 5)
            self.assertTrue(seen.isdisjoint(test))
            seen.update(test.tolist())
        self.assertGreaterEqual(dates[fl[0][1][0]], pd.Timestamp("2008-01-01"))

    def test_qlike_is_zero_only_for_a_perfect_forecast(self):
        y = np.array([-2.0, -1.5, -1.0])
        self.assertAlmostEqual(lv.qlike(y, y), 0.0)
        self.assertGreater(lv.qlike(y, y - 0.3), 0)
        self.assertGreater(lv.qlike(y, y - 0.3), lv.qlike(y, y + 0.3))  # under-forecasting costs more

    def test_lognormal_shift_is_out_of_sample_residual_variance(self):
        y, yhat = np.array([1.0, 2.0, 3.0]), np.array([1.5, 1.5, 2.5])
        self.assertAlmostEqual(lv.lognormal_shift(y, yhat), float(np.var(y - yhat)))

    def test_decision_rule(self):
        def m(q):
            return {"qlike": q}
        base = {"historical_22": m(0.64), "ewma": m(0.55), "har_style": m(0.52)}
        passing = {**base, "tm_seed7": m(0.53), "tm_seed19": m(0.54), "tm_seed41": m(0.56)}
        self.assertEqual(lv.decision(passing, [7, 19, 41])["status"], "pass")
        one_bad_seed = {**passing, "tm_seed41": m(0.60)}  # > 1.10 * 0.52
        result = lv.decision(one_bad_seed, [7, 19, 41])
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["checks"]["every_seed_within_10pct_of_har"])
        worse_than_ewma = {**base, "tm_seed7": m(0.56), "tm_seed19": m(0.56), "tm_seed41": m(0.56)}
        self.assertFalse(lv.decision(worse_than_ewma, [7, 19, 41])["checks"]["beats_historical_and_ewma"])


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.prices = synthetic_prices(n=1800, seed=7)  # 2017 .. 2023, so it crosses the development end
        self.csv = write_prices(self.dir / "prices.csv", self.prices)

    def tearDown(self):
        self.tmp.cleanup()

    def run_baselines(self, csv, name):
        return lv.run(csv, SPEC, self.dir / name, assets=["SPY"], skip_tm=True)

    def test_baselines_run_and_report(self):
        result = self.run_baselines(self.csv, "a")
        self.assertEqual(result["decision"]["status"], "not_evaluated")
        for model in lv.BASELINES:
            self.assertTrue(math.isfinite(result["metrics"]["SPY"][model]["qlike"]))
        manifest = json.loads((self.dir / "a" / "run-manifest.json").read_text())
        self.assertLessEqual(manifest["last_price_used"], "2020-12-31")
        self.assertGreater(manifest["rows_dropped_after_development"], 0)
        self.assertIn("SPY", (self.dir / "a" / "REPORT.md").read_text())

    def test_prices_after_the_development_end_cannot_change_anything(self):
        self.run_baselines(self.csv, "a")
        altered = self.prices.copy()
        altered.loc[altered.index > "2020-12-31"] *= 3.0
        csv2 = write_prices(self.dir / "altered.csv", altered)
        self.run_baselines(csv2, "b")
        a = pd.read_csv(self.dir / "a" / "predictions-SPY.csv")
        b = pd.read_csv(self.dir / "b" / "predictions-SPY.csv")
        pd.testing.assert_frame_equal(a, b)

    def test_existing_output_needs_resume_with_identical_inputs(self):
        self.run_baselines(self.csv, "a")
        with self.assertRaises(ValueError):
            self.run_baselines(self.csv, "a")
        altered = self.prices.copy()
        altered.iloc[100] *= 1.01
        csv2 = write_prices(self.dir / "altered.csv", altered)
        with self.assertRaises(ValueError):
            lv.run(csv2, SPEC, self.dir / "a", assets=["SPY"], skip_tm=True, resume=True)

    def test_cut_to_development(self):
        info = lv.cut_to_development(self.csv, SPEC, self.dir / "dev.csv")
        self.assertLessEqual(info["last"], "2020-12-31")
        self.assertLessEqual(pd.read_csv(self.dir / "dev.csv")["date"].max(), "2020-12-31")

    def test_quick_tm_run_when_tmu_is_installed(self):
        try:
            import tmu  # noqa: F401
        except ImportError:
            self.skipTest("tmu not installed")
        result = lv.run(self.csv, SPEC, self.dir / "q", assets=["SPY"], quick=True)
        self.assertEqual(result["decision"]["status"], "not_evaluated")
        pred = pd.read_csv(self.dir / "q" / "predictions-SPY.csv")
        self.assertIn("tm_seed7", pred.columns)
        self.assertTrue(np.isfinite(pred["tm_seed7"]).all())
        self.assertIn("SMOKE TEST", (self.dir / "q" / "REPORT.md").read_text())
        ck = sorted((self.dir / "q" / "checkpoints").glob("*.json"))
        self.assertTrue(ck)
        again = lv.run(self.csv, SPEC, self.dir / "q", assets=["SPY"], quick=True, resume=True)
        self.assertEqual(again["metrics"], result["metrics"])


if __name__ == "__main__":
    unittest.main()
