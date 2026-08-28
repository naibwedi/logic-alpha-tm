import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from logic_alpha_tm.backtest import expanding_folds
from logic_alpha_tm.config import ResearchConfig
from logic_alpha_tm.data import synthetic_prices, validate_prices
from logic_alpha_tm.features import QuantileBooleanEncoder, build_features
from logic_alpha_tm.pipeline import run_research
from logic_alpha_tm.strategies import forward_utilities, strategy_labels, strategy_returns


class LogicAlphaTests(unittest.TestCase):
    def test_validation_rejects_missing_assets(self):
        with self.assertRaises(ValueError):
            validate_prices(pd.DataFrame({"SPY": [1]}, index=pd.to_datetime(["2020-01-01"])))

    def test_encoder_uses_training_thresholds(self):
        train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        test = pd.DataFrame({"x": [100.0]})
        encoder = QuantileBooleanEncoder((0.5,)).fit(train)
        self.assertEqual(float(encoder.thresholds_.loc["x", 0.5]), 1.5)
        self.assertEqual(int(encoder.transform(test).iloc[0, 0]), 1)

    def test_forward_label_starts_next_session(self):
        returns = pd.DataFrame({"a": [0.5, 0.1, 0.0], "cash": [0, 0, 0]})
        utility = forward_utilities(returns, 1, 0, 0)
        self.assertAlmostEqual(utility.loc[0, "a"], 0.1)

    def test_dead_zone_returns_cash(self):
        utility = pd.DataFrame({"a": [0.002], "b": [0.001], "cash": [0.0]})
        self.assertEqual(strategy_labels(utility, 0.003).iloc[0], "cash")

    def test_fold_purges_label_horizon(self):
        train, test = next(expanding_folds(100, 50, 10, 5))
        self.assertEqual(train[-1], 44)
        self.assertEqual(test[0], 50)

    def test_strategy_signal_is_lagged(self):
        prices = synthetic_prices(250)
        streams = strategy_returns(prices)
        self.assertTrue(np.isfinite(streams.to_numpy()).all())
        self.assertEqual(streams.index.tolist(), prices.index.tolist())

    def test_end_to_end_writes_report(self):
        config = ResearchConfig(min_train=300, test_size=100, horizon=10)
        with tempfile.TemporaryDirectory() as directory:
            summary = run_research(synthetic_prices(800), directory, config, data_kind="test synthetic")
            self.assertGreater(summary["observations"], 0)
            for name in ("summary.json", "predictions.csv", "rules.csv", "report.svg", "REPORT.md"):
                self.assertTrue((Path(directory) / name).exists())


if __name__ == "__main__":
    unittest.main()

