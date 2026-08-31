import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from logic_alpha_tm.backtest import expanding_folds
from logic_alpha_tm.config import ResearchConfig
from logic_alpha_tm.data import synthetic_prices, validate_prices
from logic_alpha_tm.experiments import load_spec, validate_availability
from logic_alpha_tm.features import QuantileBooleanEncoder, build_features
from logic_alpha_tm.models import BoostedTreeSelector, LogisticSelector, TMUSelector
from logic_alpha_tm.pipeline import run_research
from logic_alpha_tm.providers import (
    download_tiingo_prices,
    parse_massive_daily_bars,
    parse_tiingo_daily_prices,
)
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

    def test_massive_parser_uses_unadjusted_close_field(self):
        payload = {
            "status": "OK",
            "results": [
                {"t": 1704153600000, "c": 472.65, "o": 470.0},
                {"t": 1704240000000, "c": 475.10, "o": 473.0},
            ],
        }
        bars = parse_massive_daily_bars(payload, "SPY")
        self.assertEqual(bars.name, "SPY")
        self.assertEqual(bars.index[0], pd.Timestamp("2024-01-02"))
        self.assertAlmostEqual(bars.iloc[1], 475.10)

    def test_tiingo_parser_preserves_raw_adjusted_and_actions(self):
        payload = [
            {
                "date": "2024-01-02T00:00:00.000Z",
                "close": 100.0,
                "adjClose": 98.0,
                "divCash": 0.5,
                "splitFactor": 1.0,
            },
            {
                "date": "2024-01-03T00:00:00.000Z",
                "close": 51.0,
                "adjClose": 100.0,
                "divCash": 0.0,
                "splitFactor": 2.0,
            },
        ]
        parsed = parse_tiingo_daily_prices(payload, "SPY")
        self.assertEqual(parsed.index[0], pd.Timestamp("2024-01-02"))
        self.assertEqual(parsed.loc["2024-01-03", "split_factor"], 2.0)
        self.assertEqual(parsed.loc["2024-01-02", "dividend_cash"], 0.5)
        self.assertEqual(parsed.loc["2024-01-02", "adjusted_close"], 98.0)

    def test_tiingo_download_uses_token_and_conservative_availability(self):
        payload = [
            {
                "date": f"2024-01-0{day}T00:00:00.000Z",
                "close": 100.0 + day,
                "adjClose": 99.0 + day,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
            for day in (2, 3, 4)
        ]

        def opener(request, timeout):
            self.assertEqual(request.get_header("Authorization"), "Token secret-test-token")
            self.assertEqual(timeout, 30)
            return io.BytesIO(json.dumps(payload).encode("utf-8"))

        adjusted, raw, actions, availability = download_tiingo_prices(
            "2024-01-02", "2024-01-04", "secret-test-token", opener
        )
        self.assertEqual(adjusted.shape, (3, 4))
        self.assertEqual(raw.shape, (3, 4))
        self.assertTrue(actions.empty)
        first_available = pd.Timestamp(availability.iloc[0].available_at)
        self.assertEqual(first_available.hour, 20)

    def test_frozen_experiment_spec_has_disjoint_periods(self):
        for path in ("experiments/real-market-v0.2.json", "experiments/tiingo-v0.2.json"):
            with self.subTest(path=path):
                spec = load_spec(path)
                self.assertLess(
                    pd.Timestamp(spec["development"]["end"]),
                    pd.Timestamp(spec["holdout"]["start"]),
                )
                self.assertIn("tmu", spec["models"])

    def test_availability_must_precede_next_session(self):
        prices = synthetic_prices(5)
        availability = pd.DataFrame({
            "observation_at": prices.index,
            "available_at": prices.index.tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15),
            "source": "test",
            "revision": "test",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.available-at.csv"
            availability.to_csv(path, index=False)
            validated = validate_availability(prices, path)
            self.assertEqual(len(validated), len(prices))

    def test_comparison_models_return_predictions_and_margins(self):
        x = pd.DataFrame({
            "a": [0, 0, 1, 1, 0, 1, 0, 1, 0] * 4,
            "b": [0, 1, 0, 1, 1, 0, 0, 1, 1] * 4,
        })
        y = pd.Series(["cash", "trend", "momentum"] * 12)
        for selector in (LogisticSelector(), BoostedTreeSelector()):
            prediction, margin = selector.fit(x, y).predict_with_margin(x.iloc[:4])
            self.assertEqual(len(prediction), 4)
            self.assertTrue(np.isfinite(margin).all())

    def test_tmu_selector_records_requested_platform(self):
        self.assertEqual(TMUSelector().platform, "CPU")
        self.assertEqual(TMUSelector(platform="CUDA").platform, "CUDA")


if __name__ == "__main__":
    unittest.main()
