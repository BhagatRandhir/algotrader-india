"""
utils/yf_helpers.py  –  Shared helpers for working with yfinance output.

yfinance 1.4.x changed its default behaviour: even SINGLE-ticker downloads
now return MultiIndex columns shaped like ('Close', 'RELIANCE.NS') instead
of flat columns like 'Close'. Code that blindly does
    df.columns = [c.lower() for c in df.columns]
ends up turning each column into the STRING REPR of a tuple, e.g.
    "('close', 'reliance.ns')"
which then breaks every df["close"] lookup downstream (KeyError, or in some
pandas versions a confusing "tuple" error). This file provides one safe,
shared helper so every module flattens columns the same correct way.

Also: yfinance sometimes returns None instead of an empty DataFrame for
failed/unknown tickers (especially futures like ES=F, NQ=F, BZ=F and index
tickers like ^NSEI). It also prints its own TypeError/Failed download messages
to the console even when the caller handles the error gracefully.
safe_download() wraps yf.download to:
  1. Suppress yfinance's own noisy console output
  2. Return an empty DataFrame instead of None on failure
  3. Provide a single entry point so the fix only lives in one place
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import pandas as pd
import yfinance as yf


def safe_download(
    ticker: str,
    period: str = "5d",
    interval: str = "1d",
    **kwargs,
) -> pd.DataFrame:
    """
    Drop-in replacement for yf.download() that:
      - Silences yfinance's own stderr/logging noise (the
        "Failed download: TypeError NoneType..." messages)
      - Returns an empty DataFrame instead of None on failure
      - Accepts all the same kwargs as yf.download

    Always call flatten_yf_columns() on the result before using columns.
    """
    try:
        # Silence yfinance's own loggers and Python warnings during the call
        yf_logger = logging.getLogger("yfinance")
        original_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                **kwargs,
            )

        yf_logger.setLevel(original_level)

        # yfinance sometimes returns None instead of empty DataFrame
        if result is None:
            return pd.DataFrame()
        return result

    except Exception:
        return pd.DataFrame()


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely normalise a yfinance OHLCV DataFrame to flat lowercase columns
    ('open', 'high', 'low', 'close', 'volume'), regardless of whether
    yfinance returned flat columns or MultiIndex (Field, Ticker) columns.
    """
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        # Keep only the field name (level 0): 'Close', 'Open', etc.
        # Works regardless of whether Ticker is level 0 or level 1.
        new_cols = []
        for col in df.columns:
            # col is a tuple like ('Close', 'RELIANCE.NS') or ('RELIANCE.NS', 'Close')
            field = next((part for part in col
                         if str(part) in ("Open", "High", "Low", "Close",
                                          "Volume", "Adj Close")), col[0])
            new_cols.append(str(field).lower())
        df.columns = new_cols
    else:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

    return df
