"""Licensed market-data adapters with explicit point-in-time metadata."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import REQUIRED, validate_prices


MASSIVE_BASE_URL = "https://api.massive.com"
TIINGO_BASE_URL = "https://api.tiingo.com"


def parse_massive_daily_bars(payload: dict, ticker: str) -> pd.Series:
    """Parse Massive aggregate bars without applying future-known adjustments."""
    if payload.get("status") not in ("OK", "DELAYED"):
        message = payload.get("error") or payload.get("message") or payload.get("status", "unknown error")
        raise ValueError(f"Massive request failed for {ticker}: {message}")
    rows = payload.get("results") or []
    if not rows:
        raise ValueError(f"Massive returned no daily bars for {ticker}")
    frame = pd.DataFrame(rows)
    if not {"t", "c"}.issubset(frame.columns):
        raise ValueError(f"Massive response for {ticker} is missing timestamp or close")
    dates = pd.to_datetime(frame["t"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    series = pd.Series(frame["c"].astype(float).to_numpy(), index=dates, name=ticker)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _massive_daily_bars(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    opener: Callable = urlopen,
) -> pd.Series:
    query = urlencode({"adjusted": "false", "sort": "asc", "limit": 50000})
    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?{query}"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "logic-alpha-tm/0.1"},
    )
    with opener(request, timeout=30) as response:
        return parse_massive_daily_bars(json.load(response), ticker)


def download_massive_prices(
    start: str,
    end: str,
    api_key: str,
    opener: Callable = urlopen,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unadjusted closes and a conservative point-in-time availability table."""
    columns = [_massive_daily_bars(ticker, start, end, api_key, opener) for ticker in REQUIRED]
    prices = validate_prices(pd.concat(columns, axis=1).dropna())

    # Record a conservative post-close timestamp for the unadjusted daily bar.
    # Users must verify latency against their licensed Massive plan before a study.
    local_dates = prices.index.tz_localize("America/New_York")
    available_at = local_dates + pd.Timedelta(hours=16, minutes=15)
    availability = pd.DataFrame(
        {
            "available_at": available_at,
            "source": "massive-us-stocks-daily-aggregates",
            "revision": "downloaded",
        },
        index=prices.index,
    )
    return prices, availability


def parse_tiingo_daily_prices(payload: list[dict], ticker: str) -> pd.DataFrame:
    """Parse Tiingo EOD rows with raw, adjusted, and corporate-action fields."""
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Tiingo returned no daily prices for {ticker}")
    frame = pd.DataFrame(payload)
    required = {"date", "close", "adjClose", "divCash", "splitFactor"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Tiingo response for {ticker} is missing: {missing}")
    dates = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None).dt.normalize()
    parsed = pd.DataFrame(
        {
            "raw_close": frame["close"].astype(float).to_numpy(),
            "adjusted_close": frame["adjClose"].astype(float).to_numpy(),
            "dividend_cash": frame["divCash"].fillna(0.0).astype(float).to_numpy(),
            "split_factor": frame["splitFactor"].fillna(1.0).astype(float).to_numpy(),
        },
        index=dates,
    )
    parsed.index.name = "date"
    return parsed[~parsed.index.duplicated(keep="last")].sort_index()


def _tiingo_daily_prices(
    ticker: str,
    start: str,
    end: str,
    api_token: str,
    opener: Callable = urlopen,
) -> pd.DataFrame:
    query = urlencode({"startDate": start, "endDate": end, "format": "json", "resampleFreq": "daily"})
    url = f"{TIINGO_BASE_URL}/tiingo/daily/{ticker}/prices?{query}"
    request = Request(
        url,
        headers={"Authorization": f"Token {api_token}", "User-Agent": "logic-alpha-tm/0.1"},
    )
    with opener(request, timeout=30) as response:
        return parse_tiingo_daily_prices(json.load(response), ticker)


def download_tiingo_prices(
    start: str,
    end: str,
    api_token: str,
    opener: Callable = urlopen,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return adjusted/raw closes, corporate actions, and conservative availability."""
    ticker_frames = {
        ticker: _tiingo_daily_prices(ticker, start, end, api_token, opener)
        for ticker in REQUIRED
    }
    common_index = ticker_frames[REQUIRED[0]].index
    for frame in ticker_frames.values():
        common_index = common_index.intersection(frame.index)
    adjusted = validate_prices(pd.DataFrame({
        ticker: frame.loc[common_index, "adjusted_close"] for ticker, frame in ticker_frames.items()
    }))
    raw = validate_prices(pd.DataFrame({
        ticker: frame.loc[common_index, "raw_close"] for ticker, frame in ticker_frames.items()
    }))
    action_rows = []
    for ticker, frame in ticker_frames.items():
        changed = frame[(frame["dividend_cash"] != 0) | (frame["split_factor"] != 1)]
        for date, row in changed.iterrows():
            action_rows.append({
                "date": date,
                "ticker": ticker,
                "dividend_cash": row["dividend_cash"],
                "split_factor": row["split_factor"],
            })
    actions = pd.DataFrame(
        action_rows, columns=["date", "ticker", "dividend_cash", "split_factor"]
    )

    # Tiingo says US EOD data normally arrive around 17:30 ET, with exchange
    # corrections possible through 20:00 ET. The strategy executes next session.
    local_dates = adjusted.index.tz_localize("America/New_York")
    availability = pd.DataFrame(
        {
            "available_at": local_dates + pd.Timedelta(hours=20),
            "source": "tiingo-end-of-day",
            "revision": "current-vintage-corporate-action-adjusted",
        },
        index=adjusted.index,
    )
    return adjusted, raw, actions, availability
