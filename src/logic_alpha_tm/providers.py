"""Licensed market-data adapters with explicit point-in-time metadata."""

from __future__ import annotations

import json
from collections.abc import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import REQUIRED, validate_prices


MASSIVE_BASE_URL = "https://api.massive.com"


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
