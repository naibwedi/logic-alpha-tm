from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
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

