from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_SETS = ("v1", "v2")


def build_features(prices: pd.DataFrame, feature_set: str = "v1") -> pd.DataFrame:
    """v1 is the frozen original set. v2 adds three predeclared groups (trend
    strength, volatility change, stock-bond correlation) and nothing else."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    r = prices.pct_change()
    out: dict[str, pd.Series] = {}
    for asset in prices.columns:
        for window in (5, 20, 60, 100):
            out[f"{asset}_ret_{window}"] = prices[asset].pct_change(window)
        out[f"{asset}_vol_20"] = r[asset].rolling(20).std() * np.sqrt(252)
    out["SPY_ma_gap_20_100"] = prices.SPY.rolling(20).mean() / prices.SPY.rolling(100).mean() - 1
    out["SPY_drawdown_100"] = prices.SPY / prices.SPY.rolling(100).max() - 1
    out["QQQ_vs_SPY_60"] = out["QQQ_ret_60"] - out["SPY_ret_60"]
    out["IWM_vs_SPY_60"] = out["IWM_ret_60"] - out["SPY_ret_60"]
    out["TLT_vs_SPY_20"] = out["TLT_ret_20"] - out["SPY_ret_20"]
    if feature_set == "v2":
        # trend strength
        out["SPY_ma_gap_50_200"] = prices.SPY.rolling(50).mean() / prices.SPY.rolling(200).mean() - 1
        out["SPY_vs_ma_200"] = prices.SPY / prices.SPY.rolling(200).mean() - 1
        # volatility change
        out["SPY_vol_ratio_5_60"] = r.SPY.rolling(5).std() / r.SPY.rolling(60).std() - 1
        out["SPY_vol_ratio_20_100"] = r.SPY.rolling(20).std() / r.SPY.rolling(100).std() - 1
        # stock-bond correlation
        out["SPY_TLT_corr_60"] = r.SPY.rolling(60).corr(r.TLT)
    frame = pd.DataFrame(out, index=prices.index)
    frame["regime"] = describe_regime(frame)
    return frame


def describe_regime(features: pd.DataFrame) -> pd.Series:
    trend = np.select(
        [features["SPY_ret_60"] > 0.05, features["SPY_ret_60"] < -0.05],
        ["UP", "DOWN"], default="SIDEWAYS"
    )
    vol = np.select(
        [features["SPY_vol_20"] < 0.12, features["SPY_vol_20"] > 0.22],
        ["LOW", "HIGH"], default="NORMAL"
    )
    return pd.Series(np.char.add(np.char.add(trend.astype(str), "_"), vol.astype(str)), index=features.index)


@dataclass
class QuantileBooleanEncoder:
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    thresholds_: pd.DataFrame | None = None

    def fit(self, x: pd.DataFrame) -> "QuantileBooleanEncoder":
        numeric = x.select_dtypes(include=[np.number])
        self.thresholds_ = numeric.quantile(self.quantiles).T
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.thresholds_ is None:
            raise RuntimeError("Encoder must be fitted")
        encoded: dict[str, pd.Series] = {}
        for column, row in self.thresholds_.iterrows():
            for q, threshold in row.items():
                encoded[f"{column}>q{int(float(q)*100):02d}"] = (x[column] > threshold).astype(np.uint8)
        return pd.DataFrame(encoded, index=x.index)

