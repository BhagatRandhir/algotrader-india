"""
core/higher_tf.py  –  Higher Timeframe (15m) Trend Filter

Phase 1 addition. Fetches 15-minute bars and determines:
  - Trend direction: BULLISH / BEARISH / SIDEWAYS
  - EMA alignment on 15m
  - Whether a 5m signal is allowed given 15m context

Rule:
  LONG  allowed only when 15m is BULLISH
  SHORT allowed only when 15m is BEARISH
  SIDEWAYS → NO_TRADE

Cached per symbol for 15 minutes (one 15m bar).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class HTFTrend(Enum):
    BULLISH  = "BULLISH"
    BEARISH  = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN  = "UNKNOWN"


@dataclass
class HTFResult:
    trend:        HTFTrend
    allow_long:   bool
    allow_short:  bool
    ema20_15m:    float
    ema50_15m:    float
    close_15m:    float
    adx_15m:      float
    reason:       str
    size_mult:    float = 1.0   # reduce size when trend is weak


class HigherTimeframeFilter:
    """
    Fetches 15-minute bars and classifies the higher-timeframe trend.
    Used as a hard directional filter — 5m signals that conflict
    with the 15m trend are rejected.
    """

    _cache: dict[str, tuple[float, HTFResult]] = {}
    CACHE_TTL = 900   # 15 min — one 15m bar

    def analyse(self, symbol: str, broker) -> HTFResult:
        now = time.time()
        if symbol in self._cache:
            cached_time, cached = self._cache[symbol]
            if now - cached_time < self.CACHE_TTL:
                return cached
        result = self._fetch_and_analyse(symbol, broker)
        self._cache[symbol] = (now, result)
        return result

    def _fetch_and_analyse(self, symbol: str, broker) -> HTFResult:
        UNKNOWN = HTFResult(
            trend=HTFTrend.UNKNOWN, allow_long=True, allow_short=False,
            ema20_15m=0, ema50_15m=0, close_15m=0, adx_15m=0,
            reason="15m data unavailable — allowing long only", size_mult=0.75,
        )
        try:
            df = broker.get_bars(symbol, interval="15m", period="5d")
            if df is None or len(df) < 55:
                logger.debug(f"HTF {symbol}: insufficient 15m bars ({len(df) if df is not None else 0})")
                return UNKNOWN

            close  = df["close"]
            ema20  = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50  = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            price  = float(close.iloc[-1])

            # EMA slopes (compare to 3 bars ago)
            ema20_ser  = close.ewm(span=20, adjust=False).mean()
            ema50_ser  = close.ewm(span=50, adjust=False).mean()
            ema20_slope = float(ema20_ser.iloc[-1] - ema20_ser.iloc[-4]) if len(ema20_ser) > 4 else 0
            ema50_slope = float(ema50_ser.iloc[-1] - ema50_ser.iloc[-4]) if len(ema50_ser) > 4 else 0

            # ADX
            adx = self._adx(df)

            # EMA compression check
            ema_gap_pct = abs(ema20 - ema50) / ema50 * 100 if ema50 > 0 else 0

            # ── Classify trend ────────────────────────────────────
            bullish_points = sum([
                price > ema20,
                ema20 > ema50,
                ema20_slope > 0,
                ema50_slope > 0,
                adx >= 18,
            ])
            bearish_points = sum([
                price < ema20,
                ema20 < ema50,
                ema20_slope < 0,
                ema50_slope < 0,
                adx >= 18,
            ])

            # Sideways: EMAs compressed + low ADX
            sideways = ema_gap_pct < 0.15 and adx < 18

            if sideways:
                trend = HTFTrend.SIDEWAYS
                reason = (f"15m SIDEWAYS — EMA gap={ema_gap_pct:.2f}% ADX={adx:.1f} "
                          f"(compressed EMAs, weak trend)")
                return HTFResult(trend, False, False, ema20, ema50, price, adx,
                                 reason, size_mult=0.0)

            if bullish_points >= 4:
                trend   = HTFTrend.BULLISH
                s_mult  = 1.0 if bullish_points == 5 else 0.75
                reason  = (f"15m BULLISH {bullish_points}/5 — "
                           f"P={price:.0f} EMA20={ema20:.0f} EMA50={ema50:.0f} ADX={adx:.1f}")
                return HTFResult(trend, True, False, ema20, ema50, price, adx,
                                 reason, size_mult=s_mult)

            if bearish_points >= 4:
                trend   = HTFTrend.BEARISH
                reason  = (f"15m BEARISH {bearish_points}/5 — "
                           f"P={price:.0f} EMA20={ema20:.0f} EMA50={ema50:.0f} ADX={adx:.1f}")
                return HTFResult(trend, False, True, ema20, ema50, price, adx,
                                 reason, size_mult=0.0)  # bot is long-only

            # Mixed — allow with reduced size
            trend  = HTFTrend.SIDEWAYS
            reason = (f"15m MIXED — bull={bullish_points} bear={bearish_points} "
                      f"ADX={adx:.1f} — no clear trend")
            return HTFResult(trend, False, False, ema20, ema50, price, adx,
                             reason, size_mult=0.0)

        except Exception as exc:
            logger.warning(f"HTF {symbol}: {exc}")
            return UNKNOWN

    def _adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index."""
        try:
            high = df["high"]; low = df["low"]; close = df["close"]
            tr   = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)

            plus_dm  = (high.diff()).clip(lower=0)
            minus_dm = (-low.diff()).clip(lower=0)
            # Prefer plus over minus on same bar
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm <= plus_dm] = 0

            atr   = tr.ewm(alpha=1/period, adjust=False).mean()
            plus  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
            minus = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-9)
            dx    = 100 * (plus - minus).abs() / (plus + minus + 1e-9)
            adx   = dx.ewm(alpha=1/period, adjust=False).mean()
            return round(float(adx.iloc[-1]), 1)
        except Exception:
            return 0.0
