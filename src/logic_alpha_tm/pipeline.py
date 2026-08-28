from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import metrics, run_walk_forward, selector_returns
from .config import ResearchConfig
from .features import build_features
from .reporting import write_report, write_svg
from .strategies import forward_utilities, strategy_labels, strategy_returns


def run_research(prices: pd.DataFrame, output: str | Path, config: ResearchConfig, model: str = "bernoulli", data_kind: str = "user CSV"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = build_features(prices)
    streams = strategy_returns(prices, config.strategy_cost_bps)
    utilities = forward_utilities(streams, config.horizon, config.lambda_vol, config.lambda_drawdown)
    labels = strategy_labels(utilities, config.label_dead_zone)
    predictions, rules = run_walk_forward(features, labels, config, model)
    selected, held = selector_returns(predictions, streams, config)
    predictions["held_strategy"] = held
    predictions["actual_label"] = labels.reindex(predictions.index)
    predictions["regime"] = features.regime.reindex(predictions.index)

    finite_margin = predictions.dropna(subset=["margin"]).copy()
    if not finite_margin.empty:
        finite_margin["margin_bucket"] = pd.qcut(
            finite_margin.margin.rank(method="first"), q=min(5, len(finite_margin)), duplicates="drop"
        ).astype(str)
        finite_margin["correct"] = finite_margin.prediction == finite_margin.actual_label
        calibration = finite_margin.groupby("margin_bucket", observed=True).agg(
            observations=("correct", "size"), mean_margin=("margin", "mean"), accuracy=("correct", "mean")
        ).reset_index()
    else:
        calibration = pd.DataFrame(columns=["margin_bucket", "observations", "mean_margin", "accuracy"])

    if not rules.empty:
        stability = rules.groupby(["strategy", "literal"], as_index=False).agg(
            folds_present=("fold", "nunique"), mean_weight=("weight", "mean")
        )
        stability["fold_stability"] = stability.folds_present / predictions.fold.nunique()
        stability = stability.sort_values(["fold_stability", "mean_weight"], ascending=False)
    else:
        stability = pd.DataFrame(columns=["strategy", "literal", "folds_present", "mean_weight", "fold_stability"])

    aligned = streams.reindex(predictions.index)
    returns = pd.DataFrame({
        "selector": selected,
        "equal_weight": aligned[["trend", "momentum", "defensive"]].mean(axis=1),
        "SPY": prices.SPY.pct_change().reindex(predictions.index).fillna(0.0),
        "always_trend": aligned.trend,
        "always_momentum": aligned.momentum,
        "always_defensive": aligned.defensive,
    })
    equity = (1 + returns).cumprod()
    summary = {
        "data_kind": data_kind,
        "model": model,
        "observations": len(predictions),
        "folds": int(predictions.fold.nunique()),
        "accuracy": float((predictions.prediction == predictions.actual_label).mean()),
        "metrics": {column: metrics(returns[column]) for column in returns},
        "config": config.__dict__,
        "warning": "Research software; synthetic results are not evidence of tradable alpha.",
    }
    equity.to_csv(output / "equity.csv", index_label="date")
    returns.to_csv(output / "returns.csv", index_label="date")
    predictions.to_csv(output / "predictions.csv", index_label="date")
    rules.to_csv(output / "rules.csv", index=False)
    calibration.to_csv(output / "margin_calibration.csv", index=False)
    stability.to_csv(output / "rule_stability.csv", index=False)
    write_svg(equity, predictions, output / "report.svg")
    write_report(output, summary, predictions, rules)
    return summary
