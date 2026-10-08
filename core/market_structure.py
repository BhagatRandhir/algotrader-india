"""
core/market_structure.py  –  Market Structure Detector

Phase 1 addition. Detects:
  - Swing highs and swing lows
  - Higher High / Higher Low (bullish structure)
  - Lower Low / Lower High (bearish structure)
  - Opening range (first 15 minutes)

Used as a filter: only take LONG in HH/HL structure,
only take SHORT in LL/LH structure.
Mixed/unclear structure → reduce score.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class Structure(Enum):
    BULLISH  = "BULLISH"   # HH + HL confirmed
    BEARISH  = "BEARISH"   # LL + LH confirmed
    MIXED    = "MIXED"     # contradictory
    UNCLEAR  = "UNCLEAR"   # not enough swings yet


@dataclass
class StructureResult:
    structure:      Structure
    last_swing_high:float
    last_swing_low: float
    prev_swing_high:float
    prev_swing_low: float
    hh:             bool   # higher high vs previous
    hl:             bool   # higher low vs previous
    ll:             bool   # lower low vs previous
    lh:             bool   # lower high vs previous
    score_bonus:    float  # +1 bullish, -1 bearish, 0 mixed
    reason:         str


@dataclass
class OpeningRange:
    high:      float
    low:       float
    mid:       float
    range_pct: float   # (high-low)/low * 100


def detect_swings(df: pd.DataFrame, lookback: int = 5) -> tuple[list[float], list[float]]:
    """
    Detect swing highs and lows using a simple N-bar lookback.
    A swing high: highest point in a window of lookback bars each side.
    A swing low:  lowest point in a window.
    Returns (swing_highs, swing_lows) as lists of price values.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(highs)

    swing_highs = []
    swing_lows  = []

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[ i - lookback: i + lookback + 1]

        if highs[i] == window_h.max():
            swing_highs.append(highs[i])
        if lows[i] == window_l.min():
            swing_lows.append(lows[i])

    return swing_highs, swing_lows


def detect_structure(df: pd.DataFrame, lookback: int = 5) -> StructureResult:
    """
    Classify current market structure from recent swings.
    Needs at least 2 swing highs and 2 swing lows to classify.
    """
    UNCLEAR = StructureResult(
        structure=Structure.UNCLEAR,
        last_swing_high=0, last_swing_low=0,
        prev_swing_high=0, prev_swing_low=0,
        hh=False, hl=False, ll=False, lh=False,
        score_bonus=0.0, reason="Insufficient swings to classify structure",
    )

    if df is None or len(df) < lookback * 4:
        return UNCLEAR

    swing_highs, swing_lows = detect_swings(df, lookback)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return UNCLEAR

    # Most recent swings
    sh1, sh2 = swing_highs[-1], swing_highs[-2]   # sh1 = most recent
    sl1, sl2 = swing_lows[-1],  swing_lows[-2]

    hh = sh1 > sh2   # higher high
    hl = sl1 > sl2   # higher low
    ll = sl1 < sl2   # lower low
    lh = sh1 < sh2   # lower high

    if hh and hl:
        structure = Structure.BULLISH
        bonus     = 1.0
        reason    = (f"Bullish structure HH+HL — "
                     f"SH: {sh2:.0f}→{sh1:.0f} SL: {sl2:.0f}→{sl1:.0f}")
    elif ll and lh:
        structure = Structure.BEARISH
        bonus     = -1.0
        reason    = (f"Bearish structure LL+LH — "
                     f"SL: {sl2:.0f}→{sl1:.0f} SH: {sh2:.0f}→{sh1:.0f}")
    elif hh and lh:
        structure = Structure.MIXED
        bonus     = 0.0
        reason    = f"Mixed — HH but LH (topping?)"
    elif ll and hl:
        structure = Structure.MIXED
        bonus     = 0.0
        reason    = f"Mixed — LL but HL (bottoming?)"
    else:
        structure = Structure.UNCLEAR
        bonus     = 0.0
        reason    = "No clear structure"

    return StructureResult(
        structure       = structure,
        last_swing_high = sh1,
        last_swing_low  = sl1,
        prev_swing_high = sh2,
        prev_swing_low  = sl2,
        hh=hh, hl=hl, ll=ll, lh=lh,
        score_bonus = bonus,
        reason      = reason,
    )


def get_opening_range(df: pd.DataFrame, open_bars: int = 3) -> OpeningRange:
    """
    First `open_bars` of 5-minute bars = opening range (15 minutes).
    Used to:
      - Avoid trading in opening volatility
      - Detect breakouts from the opening range
    """
    if df is None or len(df) < open_bars + 1:
        return OpeningRange(high=0, low=0, mid=0, range_pct=0)

    # Assume df is sorted by time, first bars = market open
    or_bars   = df.iloc[:open_bars]
    or_high   = float(or_bars["high"].max())
    or_low    = float(or_bars["low"].min())
    or_mid    = (or_high + or_low) / 2
    or_range  = (or_high - or_low) / or_low * 100 if or_low > 0 else 0

    return OpeningRange(
        high      = round(or_high, 2),
        low       = round(or_low,  2),
        mid       = round(or_mid,  2),
        range_pct = round(or_range, 3),
    )
