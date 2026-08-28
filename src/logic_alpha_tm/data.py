from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ("SPY", "QQQ", "IWM", "TLT")


def validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED) - set(prices.columns))
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Dates must be unique and increasing")
    out = prices.loc[:, REQUIRED].astype(float)
    if (out <= 0).any().any() or out.isna().any().any():
        raise ValueError("Prices must be positive and complete")
    return out


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return validate_prices(frame)


def synthetic_prices(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    """Deterministic data for software validation, never for alpha claims."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n)
    regime_len = 150
    regimes = (np.arange(n) // regime_len) % 4
    drifts = np.array([0.0006, -0.0005, 0.0001, 0.0008])[regimes]
    vols = np.array([0.006, 0.018, 0.005, 0.012])[regimes]
    market = drifts + vols * rng.normal(size=n)
    qqq = 1.18 * market + 0.004 * rng.normal(size=n)
    iwm = 0.92 * market + 0.006 * rng.normal(size=n)
    # Bonds tend to help in risk-off blocks but are not a perfect hedge.
    tlt = np.where(regimes == 1, 0.00035, 0.00005) - 0.22 * market + 0.004 * rng.normal(size=n)
    returns = np.column_stack([market, qqq, iwm, tlt])
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=REQUIRED)

