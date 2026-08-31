"""
strategies/pattern_recognition.py  –  4th Signal Source: Chart Patterns

Honest framing: this is RULE-BASED GEOMETRIC PATTERN DETECTION, not a
neural network or deep learning model. Many "AI pattern recognition"
products are dressed-up versions of exactly this. We detect patterns
using explicit price/swing-point geometry — transparent and debuggable,
not a black box.

Patterns detected:
  1. SUPPORT/RESISTANCE BOUNCE  — price approaches a prior swing level
     and reverses (tested 2+ times = stronger level)
  2. TRENDLINE BREAKOUT         — price breaks above a falling trendline
     (or below a rising one) connecting 2+ swing points
  3. DOUBLE BOTTOM / DOUBLE TOP — classic reversal pattern, two troughs/
     peaks at similar price with a bounce between them
  4. BULLISH/BEARISH ENGULFING  — candlestick reversal pattern at a
     swing low/high
  5. VOLUME BREAKOUT CONFIRMATION — breakout candle backed by volume
     surge (filters out fake breakouts)

This integrates as a 4th strategy in the existing ensemble aggregator —
same StrategyResult interface as MA_Crossover, RSI_Momentum, Mean_Reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger

from .base import BaseStrategy, Signal, StrategyResult


@dataclass
class SwingPoint:
    index:    int
    price:    float
    kind:     str   # "high" or "low"


class PatternRecognitionStrategy(BaseStrategy):
    """
    Detects classic chart patterns using swing-point geometry.
    Returns BUY/SELL/HOLD like every other strategy in the ensemble.
    """

    name = "Pattern_Recognition"

    def __init__(
        self,
        swing_lookback:      int   = 5,     # bars on each side to confirm a swing point
        level_tolerance_pct: float = 0.5,   # % tolerance for "same" S/R level
        min_swing_points:    int   = 3,     # minimum swings needed before pattern search
        volume_surge_mult:   float = 1.8,   # breakout volume vs 20-bar avg
    ):
        self.swing_lookback      = swing_lookback
        self.level_tolerance_pct = level_tolerance_pct
        self.min_swing_points    = min_swing_points
        self.volume_surge_mult   = volume_surge_mult

    # ── Swing point detection ───────────────────────────────────────

    def _find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        """Find local highs/lows using a simple N-bar lookback/lookahead window.
        Requires the window to have genuine price variation — on perfectly flat
        or near-flat data, every bar trivially equals the window max/min, which
        would otherwise produce false swing points and fake S/R levels."""
        highs = df["high"].values
        lows  = df["low"].values
        n     = len(df)
        lb    = self.swing_lookback
        swings: list[SwingPoint] = []

        # Guard against flat/near-flat series: need meaningful range to call
        # anything a "swing" at all.
        price_range = float(np.nanmax(highs) - np.nanmin(lows))
        avg_price   = float(np.nanmean(df["close"].values))
        if avg_price <= 0 or price_range / avg_price < 0.002:   # <0.2% total range
            return []

        for i in range(lb, n - lb):
            window_high = highs[i - lb:i + lb + 1]
            window_low  = lows[i - lb:i + lb + 1]

            # Require a STRICT, UNIQUE max/min — not a tie across the window —
            # so a flat run of identical prices never registers as a swing.
            is_unique_high = highs[i] == window_high.max() and np.sum(window_high == window_high.max()) == 1
            is_unique_low  = lows[i]  == window_low.min()  and np.sum(window_low  == window_low.min())  == 1

            if is_unique_high:
                swings.append(SwingPoint(i, highs[i], "high"))
            if is_unique_low:
                swings.append(SwingPoint(i, lows[i], "low"))

        return swings

    # ── Pattern 1: Support / Resistance bounce ──────────────────────

    def _check_sr_bounce(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> tuple[float, str] | None:
        """
        If current price is near a level tested 2+ times before,
        and price is bouncing off it, return (strength, reason).
        """
        price = df["close"].iloc[-1]
        lows  = [s for s in swings if s.kind == "low"]
        highs = [s for s in swings if s.kind == "high"]

        # Group lows into levels within tolerance
        def cluster_levels(points: list[SwingPoint]) -> list[tuple[float, int]]:
            levels: list[tuple[float, int]] = []
            for p in points:
                matched = False
                for i, (lvl, count) in enumerate(levels):
                    if abs(p.price - lvl) / lvl * 100 <= self.level_tolerance_pct:
                        levels[i] = ((lvl * count + p.price) / (count + 1), count + 1)
                        matched = True
                        break
                if not matched:
                    levels.append((p.price, 1))
            return levels

        support_levels    = cluster_levels(lows)
        resistance_levels = cluster_levels(highs)

        # Support bounce — price near a tested support, multiple touches = stronger
        for lvl, touches in support_levels:
            if touches >= 2 and abs(price - lvl) / lvl * 100 <= self.level_tolerance_pct * 1.5:
                if price >= lvl:   # bouncing UP off support
                    strength = min(0.4 + touches * 0.15, 1.0)
                    return strength, (
                        f"Price ₹{price:.2f} bouncing off support ₹{lvl:.2f} "
                        f"(tested {touches}x)"
                    )

        # Resistance rejection — price near tested resistance, rejecting down
        for lvl, touches in resistance_levels:
            if touches >= 2 and abs(price - lvl) / lvl * 100 <= self.level_tolerance_pct * 1.5:
                if price <= lvl:
                    strength = min(0.4 + touches * 0.15, 1.0)
                    return -strength, (
                        f"Price ₹{price:.2f} rejected at resistance ₹{lvl:.2f} "
                        f"(tested {touches}x)"
                    )

        return None

    # ── Pattern 2: Trendline breakout ───────────────────────────────

    def _check_trendline_breakout(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> tuple[float, str] | None:
        """
        Connect the last 2 swing highs (falling trendline) or last 2 swing
        lows (rising trendline) and check if price just broke through.
        """
        highs = sorted([s for s in swings if s.kind == "high"], key=lambda s: s.index)
        lows  = sorted([s for s in swings if s.kind == "low"],  key=lambda s: s.index)
        price = df["close"].iloc[-1]
        last_idx = len(df) - 1

        # Falling trendline (resistance) from last 2 swing highs → bullish breakout
        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            if h2.price < h1.price and h2.index > h1.index:  # genuinely falling
                slope = (h2.price - h1.price) / (h2.index - h1.index)
                trendline_now = h2.price + slope * (last_idx - h2.index)
                if price > trendline_now * 1.002:  # broke above with a small buffer
                    strength = min(abs(price - trendline_now) / trendline_now * 15, 1.0)
                    return strength, (
                        f"Price ₹{price:.2f} broke above falling trendline "
                        f"(~₹{trendline_now:.2f})"
                    )

        # Rising trendline (support) from last 2 swing lows → bearish breakdown
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            if l2.price > l1.price and l2.index > l1.index:  # genuinely rising
                slope = (l2.price - l1.price) / (l2.index - l1.index)
                trendline_now = l2.price + slope * (last_idx - l2.index)
                if price < trendline_now * 0.998:
                    strength = min(abs(price - trendline_now) / trendline_now * 15, 1.0)
                    return -strength, (
                        f"Price ₹{price:.2f} broke below rising trendline "
                        f"(~₹{trendline_now:.2f})"
                    )

        return None

    # ── Pattern 3: Double bottom / double top ───────────────────────

    def _check_double_pattern(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> tuple[float, str] | None:
        lows  = sorted([s for s in swings if s.kind == "low"],  key=lambda s: s.index)
        highs = sorted([s for s in swings if s.kind == "high"], key=lambda s: s.index)
        price = df["close"].iloc[-1]

        # Double bottom: two similar-price lows with a peak between them, price now rising
        if len(lows) >= 2:
            l1, l2 = lows[-2], lows[-1]
            price_diff_pct = abs(l1.price - l2.price) / l1.price * 100
            if price_diff_pct <= self.level_tolerance_pct * 2 and l2.index > l1.index:
                # Confirm there's a peak between them
                between_highs = [h for h in highs if l1.index < h.index < l2.index]
                if between_highs and price > l2.price * 1.01:
                    strength = 0.7
                    return strength, (
                        f"Double bottom confirmed: ₹{l1.price:.2f} & ₹{l2.price:.2f}, "
                        f"price now recovering"
                    )

        # Double top: two similar-price highs with a trough between, price now falling
        if len(highs) >= 2:
            h1, h2 = highs[-2], highs[-1]
            price_diff_pct = abs(h1.price - h2.price) / h1.price * 100
            if price_diff_pct <= self.level_tolerance_pct * 2 and h2.index > h1.index:
                between_lows = [l for l in lows if h1.index < l.index < h2.index]
                if between_lows and price < h2.price * 0.99:
                    strength = 0.7
                    return -strength, (
                        f"Double top confirmed: ₹{h1.price:.2f} & ₹{h2.price:.2f}, "
                        f"price now declining"
                    )

        return None

    # ── Pattern 4: Engulfing candle at swing point ──────────────────

    def _check_engulfing(self, df: pd.DataFrame) -> tuple[float, str] | None:
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_bearish = prev["close"] < prev["open"]
        curr_bullish = curr["close"] > curr["open"]
        bullish_engulf = (
            prev_bearish and curr_bullish and
            curr["open"] <= prev["close"] and curr["close"] >= prev["open"]
        )

        prev_bullish = prev["close"] > prev["open"]
        curr_bearish = curr["close"] < curr["open"]
        bearish_engulf = (
            prev_bullish and curr_bearish and
            curr["open"] >= prev["close"] and curr["close"] <= prev["open"]
        )

        if bullish_engulf:
            body_size = abs(curr["close"] - curr["open"]) / curr["open"] * 100
            strength = min(0.3 + body_size * 0.1, 0.8)
            return strength, f"Bullish engulfing candle (body {body_size:.2f}%)"

        if bearish_engulf:
            body_size = abs(curr["close"] - curr["open"]) / curr["open"] * 100
            strength = min(0.3 + body_size * 0.1, 0.8)
            return -strength, f"Bearish engulfing candle (body {body_size:.2f}%)"

        return None

    # ── Volume confirmation ──────────────────────────────────────────

    def _volume_confirms(self, df: pd.DataFrame) -> bool:
        if len(df) < 21:
            return True   # not enough history to judge — don't penalise
        avg_vol = df["volume"].iloc[-21:-1].mean()
        curr_vol = df["volume"].iloc[-1]
        return curr_vol >= avg_vol * self.volume_surge_mult

    # ── Main signal generation ──────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        if not self._validate(df, min_rows=self.swing_lookback * 2 + 10):
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data for pattern detection", self.name)

        swings = self._find_swings(df)
        if len(swings) < self.min_swing_points:
            return StrategyResult(
                Signal.HOLD, 0.0,
                f"Only {len(swings)} swing points found — no clear pattern yet",
                self.name,
            )

        # Check all patterns, collect any that fire
        pattern_results: list[tuple[float, str]] = []

        for check in (
            self._check_sr_bounce,
            self._check_trendline_breakout,
            self._check_double_pattern,
        ):
            result = check(df, swings)
            if result:
                pattern_results.append(result)

        engulf = self._check_engulfing(df)
        if engulf:
            pattern_results.append(engulf)

        if not pattern_results:
            return StrategyResult(Signal.HOLD, 0.0, "No chart pattern detected", self.name)

        # Combine: average strength, majority direction wins
        bullish = [s for s, r in pattern_results if s > 0]
        bearish = [s for s, r in pattern_results if s < 0]
        reasons = [r for s, r in pattern_results]

        volume_ok = self._volume_confirms(df)
        vol_note  = " + volume surge" if volume_ok else " (no volume confirmation)"

        if len(bullish) > len(bearish):
            avg_strength = sum(bullish) / len(bullish)
            if volume_ok:
                avg_strength = min(avg_strength * 1.2, 1.0)
            else:
                avg_strength *= 0.7   # discount unconfirmed breakouts
            return StrategyResult(
                Signal.BUY, round(avg_strength, 2),
                f"{len(bullish)} bullish pattern(s){vol_note}: " + " | ".join(reasons),
                self.name,
            )

        if len(bearish) > len(bullish):
            avg_strength = sum(abs(s) for s in bearish) / len(bearish)
            if volume_ok:
                avg_strength = min(avg_strength * 1.2, 1.0)
            else:
                avg_strength *= 0.7
            return StrategyResult(
                Signal.SELL, round(avg_strength, 2),
                f"{len(bearish)} bearish pattern(s){vol_note}: " + " | ".join(reasons),
                self.name,
            )

        return StrategyResult(Signal.HOLD, 0.0, "Conflicting patterns — no clear direction", self.name)
