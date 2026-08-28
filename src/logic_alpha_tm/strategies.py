import numpy as np
import pandas as pd


def _charge_changes(gross: pd.Series, holding: pd.Series, cost_bps: float) -> pd.Series:
    changes = holding.ne(holding.shift()).astype(float)
    changes.iloc[0] = 0.0
    return gross - changes * cost_bps / 10_000


def strategy_returns(prices: pd.DataFrame, cost_bps: float = 2.0) -> pd.DataFrame:
    returns = prices.pct_change().fillna(0.0)
    ma20 = prices.SPY.rolling(20).mean()
    ma100 = prices.SPY.rolling(100).mean()

    trend_signal = (ma20 > ma100).shift(1).fillna(False)
    trend_holding = trend_signal.map({True: "SPY", False: "CASH"})
    trend_gross = returns.SPY.where(trend_signal, 0.0)

    momentum_scores = prices[["SPY", "QQQ", "IWM"]].pct_change(60)
    has_history = momentum_scores.notna().any(axis=1)
    leaders = momentum_scores.fillna(-np.inf).idxmax(axis=1).where(has_history, "CASH")
    momentum_holding = leaders.shift(1).fillna("CASH")
    momentum_gross = pd.Series(0.0, index=prices.index)
    for asset in ("SPY", "QQQ", "IWM"):
        momentum_gross = momentum_gross.where(momentum_holding != asset, returns[asset])

    risk_on = (prices.SPY > ma100).shift(1).fillna(False)
    defensive_holding = risk_on.map({True: "SPY", False: "TLT"})
    defensive_gross = returns.SPY.where(risk_on, returns.TLT)

    return pd.DataFrame({
        "trend": _charge_changes(trend_gross, trend_holding, cost_bps),
        "momentum": _charge_changes(momentum_gross, momentum_holding, cost_bps),
        "defensive": _charge_changes(defensive_gross, defensive_holding, cost_bps),
        "cash": np.zeros(len(prices)),
    }, index=prices.index)


def _max_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def forward_utilities(
    returns: pd.DataFrame, horizon: int, lambda_vol: float, lambda_drawdown: float
) -> pd.DataFrame:
    values = returns.to_numpy()
    result = np.full_like(values, np.nan, dtype=float)
    for i in range(len(returns) - horizon):
        future = values[i + 1:i + 1 + horizon]
        compounded = np.prod(1.0 + future, axis=0) - 1.0
        vol = np.std(future, axis=0, ddof=0) * np.sqrt(252)
        drawdown = np.array([abs(_max_drawdown(future[:, j])) for j in range(future.shape[1])])
        result[i] = compounded - lambda_vol * vol - lambda_drawdown * drawdown
    return pd.DataFrame(result, index=returns.index, columns=returns.columns)


def strategy_labels(utilities: pd.DataFrame, dead_zone: float) -> pd.Series:
    labels = []
    for _, row in utilities.iterrows():
        if row.isna().any():
            labels.append(None)
            continue
        ranked = row.sort_values(ascending=False)
        winner = ranked.index[0]
        advantage = ranked.iloc[0] - ranked.iloc[1]
        labels.append(winner if winner != "cash" and ranked.iloc[0] > dead_zone and advantage > dead_zone else "cash")
    return pd.Series(labels, index=utilities.index, name="label", dtype="object")
