from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .features import QuantileBooleanEncoder
from .models import BernoulliSelector, BoostedTreeSelector, LogisticSelector, TMUSelector


def expanding_folds(n: int, min_train: int, test_size: int, embargo: int, first_test: int | None = None):
    start = max(min_train, first_test or min_train)
    while start < n:
        stop = min(start + test_size, n)
        train_stop = start - embargo
        if train_stop > 0:
            yield np.arange(train_stop), np.arange(start, stop)
        start = stop


def _selector(model_name: str, config: ResearchConfig):
    if model_name == "tmu":
        return TMUSelector(platform=config.tmu_platform)
    if model_name == "logistic":
        return LogisticSelector(config.seed)
    if model_name == "boosted_tree":
        return BoostedTreeSelector(config.seed)
    if model_name == "bernoulli":
        return BernoulliSelector(config.smoothing)
    raise ValueError(f"Unknown model: {model_name}")


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    config: ResearchConfig,
    model_name: str = "bernoulli",
    evaluation_start: str | None = None,
    evaluation_end: str | None = None,
):
    usable = features.drop(columns=["regime"]).dropna()
    common = usable.index.intersection(labels.dropna().index)
    x = usable.loc[common]
    y = labels.loc[common]
    if evaluation_end:
        keep = x.index <= pd.Timestamp(evaluation_end)
        x, y = x.loc[keep], y.loc[keep]
    first_test = int(x.index.searchsorted(pd.Timestamp(evaluation_start))) if evaluation_start else None
    predictions = []
    rules = []
    for fold, (train_i, test_i) in enumerate(
        expanding_folds(len(x), config.min_train, config.test_size, config.horizon, first_test)
    ):
        train_x, test_x = x.iloc[train_i], x.iloc[test_i]
        train_y = y.iloc[train_i]
        encoder = QuantileBooleanEncoder(config.quantiles).fit(train_x)
        bx_train, bx_test = encoder.transform(train_x), encoder.transform(test_x)
        model = _selector(model_name, config).fit(bx_train, train_y)
        predicted, margin = model.predict_with_margin(bx_test)
        block = pd.DataFrame({"prediction": predicted, "margin": margin, "fold": fold}, index=test_x.index)
        predictions.append(block)
        fold_rules = model.rules()
        fold_rules["fold"] = fold
        rules.append(fold_rules)
    if not predictions:
        raise ValueError("Not enough observations for one walk-forward fold")
    return pd.concat(predictions), pd.concat(rules, ignore_index=True)


def selector_returns(predictions: pd.DataFrame, strategy_returns: pd.DataFrame, config: ResearchConfig):
    decisions = predictions.prediction.copy()
    # Only update every Nth test observation and carry the decision forward.
    mask = np.arange(len(decisions)) % config.rebalance_every == 0
    held = decisions.where(mask).ffill()
    executed = held.shift(1)  # decision at t earns from t+1
    result = pd.Series(0.0, index=decisions.index, name="selector")
    for strategy in strategy_returns.columns:
        result = result.where(executed != strategy, strategy_returns[strategy].reindex(result.index))
    switches = executed.ne(executed.shift()).astype(float)
    switches.iloc[0] = 0.0
    result -= switches * config.selector_switch_cost_bps / 10_000
    return result.fillna(0.0), held


def metrics(returns: pd.Series) -> dict[str, float]:
    clean = returns.fillna(0.0)
    wealth = (1 + clean).cumprod()
    years = max(len(clean) / 252, 1 / 252)
    total = float(wealth.iloc[-1] - 1)
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1)
    vol = float(clean.std(ddof=0) * np.sqrt(252))
    downside = float(clean.clip(upper=0).std(ddof=0) * np.sqrt(252))
    drawdown = wealth / wealth.cummax() - 1
    mdd = float(drawdown.min())
    return {
        "total_return": total,
        "cagr": cagr,
        "annual_volatility": vol,
        "sharpe": cagr / vol if vol else 0.0,
        "sortino": cagr / downside if downside else 0.0,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd else 0.0,
    }
