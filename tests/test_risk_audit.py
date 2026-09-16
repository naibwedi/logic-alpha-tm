import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from logic_alpha_tm.data import synthetic_prices
from logic_alpha_tm.risk_audit import (
    AuditConfig, account, audit_metrics, drawdown, predictions_for,
    risk_labels, run_audit, target_weights,
)


class RiskAuditTests(unittest.TestCase):
    def test_after_close_signal_cannot_capture_next_overnight_return(self):
        prices = pd.DataFrame({"SPY": [100, 200, 220, 220]}, index=pd.bdate_range("2020-01-01", periods=4))
        targets = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
        ledger = account(prices, targets, AuditConfig(cost_bps=0))
        np.testing.assert_allclose(ledger["return"], [0, 0, .1, 0])

    def test_costs_charge_asset_trades_and_initial_entry(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(100.0, index=dates, columns=["SPY", "TLT"])
        weights = pd.DataFrame([[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]], index=dates, columns=prices.columns)
        ledger = account(prices, weights, AuditConfig(cost_bps=10))
        np.testing.assert_allclose(ledger.turnover, [0, 0, 1, 0, 2])
        np.testing.assert_allclose(ledger["return"], [0, 0, -.001, 0, -.002])

    def test_turnover_accounts_for_weight_drift(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        prices = pd.DataFrame({"SPY": [100, 100, 200, 200]}, index=dates)
        weights = pd.DataFrame(.5, index=dates, columns=prices.columns)
        ledger = account(prices, weights, AuditConfig(cost_bps=0))
        self.assertAlmostEqual(ledger.turnover.iloc[3], 2/3 - .5)

    def test_standard_sharpe_and_first_loss_drawdown(self):
        r = pd.Series([-.1, .02, .03])
        measured = audit_metrics(r)
        self.assertAlmostEqual(measured["sharpe"], r.mean()/r.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(drawdown(np.array([-.1, .02])), -.1)

    def test_ambiguous_labels_keep_blend(self):
        stream = pd.Series([0.0] * 10)
        labels = risk_labels(stream, stream, AuditConfig(horizon=3))
        self.assertEqual(set(labels.dropna()), {"blend"})
        self.assertTrue(labels.iloc[-4:].isna().all())

    def test_loss_labels_reduce_and_future_window_starts_after_execution(self):
        blend = pd.Series([-.9, -.9, -.1, -.1, 0.0, 0.0])
        reduced = pd.Series([0.0, 0.0, -.05, -.05, 0.0, 0.0])
        config = AuditConfig(horizon=2, lambda_vol=0, lambda_drawdown=0)
        self.assertEqual(risk_labels(blend, reduced, config).iloc[0], "reduced")
        changed = blend.copy()
        changed.iloc[:2] = .9
        pd.testing.assert_series_equal(risk_labels(blend, reduced, config), risk_labels(changed, reduced, config))

    def test_targets_do_not_depend_on_future_prices(self):
        prices = synthetic_prices(220)
        changed = prices.copy()
        changed.iloc[160:] *= 2
        for name, weights in target_weights(prices).items():
            pd.testing.assert_frame_equal(weights.iloc[:160], target_weights(changed)[name].iloc[:160])

    def test_purge_and_resume_with_prediction_tail(self):
        dates = pd.bdate_range("2020-01-01", periods=50)
        features = pd.DataFrame({"x": np.arange(50), "regime": "test"}, index=dates)
        labels = pd.Series(["blend", "reduced"] * 25, index=dates)
        labels.iloc[-4:] = None
        config = AuditConfig(min_train=20, horizon=3, test_size=10)
        trained = []

        class Fake:
            def fit(self, x, y):
                trained.append(x.index[-1])
                return self
            def predict_with_margin(self, x):
                return np.full(len(x), "blend"), np.zeros(len(x))

        with tempfile.TemporaryDirectory() as temp:
            with patch("logic_alpha_tm.risk_audit.BernoulliSelector", Fake):
                first = predictions_for(features, labels, str(dates[20].date()), "bernoulli", config, Path(temp), "id", False)
            self.assertEqual(trained[0], dates[15])  # label resolves at 19, strictly before test 20
            self.assertEqual(first.index[-1], dates[-1])  # missing tail labels don't truncate returns
            with patch("logic_alpha_tm.risk_audit.BernoulliSelector", side_effect=AssertionError("should restore")):
                restored = predictions_for(features, labels, str(dates[20].date()), "bernoulli", config, Path(temp), "id", True)
            pd.testing.assert_frame_equal(first, restored)
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                predictions_for(features, labels, str(dates[20].date()), "bernoulli", config, Path(temp), "different", True)

    def test_audit_holdout_exclusion_and_safe_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prices = synthetic_prices(400)
            csv = root / "prices.csv"
            prices.to_csv(csv, index_label="date")
            availability = pd.DataFrame({"observation_at": prices.index,
                "available_at": prices.index.tz_localize("America/New_York") + pd.Timedelta(hours=20),
                "source": "synthetic", "revision": "test"})
            availability.to_csv(root / "prices.available-at.csv", index=False)
            spec = {"engine": "risk-audit-v0.3", "development": {"start": str(prices.index[220].date()), "end": str(prices.index[349].date())},
                    "holdout": {"start": str(prices.index[350].date())}, "models": ["bernoulli"], "seeds": [7],
                    "config": {"min_train": 100, "horizon": 10}, "evaluation_cost_bps": [2, 5]}
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            output = root / "result"
            result = run_audit(str(csv), str(spec_path), str(output))
            self.assertTrue((output / "completed.json").exists())
            self.assertEqual(result.end.unique().tolist(), [spec["development"]["end"]])
            resumed = run_audit(str(csv), str(spec_path), str(output), True)
            pd.testing.assert_frame_equal(result, resumed)
            with self.assertRaisesRegex(ValueError, "Existing output"):
                run_audit(str(csv), str(spec_path), str(output))
            prices.iloc[350:] *= 10
            prices.to_csv(csv, index_label="date")
            with self.assertRaisesRegex(ValueError, "Existing output"):
                run_audit(str(csv), str(spec_path), str(output), True)
            rerun = run_audit(str(csv), str(spec_path), str(root / "changed-future"))
            pd.testing.assert_frame_equal(result, rerun)

    def test_invalid_execution_lag_is_rejected(self):
        with self.assertRaises(ValueError):
            replace(AuditConfig(), execution_lag=1)


if __name__ == "__main__":
    unittest.main()
