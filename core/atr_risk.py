"""
core/atr_risk.py  –  ATR-Based Dynamic Stop Loss & Position Sizing

Phase 1 addition. Replaces fixed % stop loss with ATR-based dynamic SL.

Stop loss logic (for LONG):
  SL = max(
    entry - ATR * atr_multiplier,        # ATR buffer
    entry - recent_swing_low - buffer,   # below swing low
    entry * (1 - min_sl_pct),            # absolute minimum
  )

This means:
  - Wide ATR day → wider SL (avoids getting stopped on noise)
  - Tight ATR day → tighter SL (less risk per trade)
  - Always below the most recent swing low (logical stop)

Position sizing from SL distance:
  risk_amount   = equity * risk_pct
  position_size = risk_amount / stop_distance

All values configurable via .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Config ────────────────────────────────────────────────────────
ATR_PERIOD      = int(os.getenv("ATR_PERIOD",        "14"))
ATR_SL_MULT     = float(os.getenv("ATR_SL_MULT",     "1.5"))  # SL = ATR × 1.5
ATR_TP1_MULT    = float(os.getenv("ATR_TP1_MULT",    "2.5"))  # TP1 = ATR × 2.5 (1.67R)
ATR_TP2_MULT    = float(os.getenv("ATR_TP2_MULT",    "4.0"))  # TP2 = ATR × 4.0 (2.67R)
MIN_SL_PCT      = float(os.getenv("MIN_SL_PCT",      "0.005"))# minimum 0.5% SL
MAX_SL_PCT      = float(os.getenv("MAX_SL_PCT",      "0.025"))# maximum 2.5% SL
MIN_RR          = float(os.getenv("MIN_RR_RATIO",    "1.5"))  # minimum reward:risk
RISK_PCT_FULL   = float(os.getenv("RISK_PCT_FULL",   "0.01")) # 1% risk at full size
RISK_PCT_HALF   = float(os.getenv("RISK_PCT_HALF",   "0.005"))# 0.5% risk at half size


@dataclass
class ATRRiskResult:
    entry:          float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  float
    stop_distance:  float
    atr:            float
    risk_reward_1:  float
    risk_reward_2:  float
    qty_full:       int     # full risk sizing
    qty_half:       int     # reduced risk (score 80-89)
    valid:          bool
    reason:         str


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """True Range ATR."""
    try:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        tr    = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return round(float(atr.iloc[-1]), 4)
    except Exception as exc:
        logger.debug(f"ATR calculation error: {exc}")
        return 0.0


def calculate_atr_risk(
    df:          pd.DataFrame,
    entry:       float,
    equity:      float,
    swing_low:   Optional[float] = None,   # from market_structure
    signal_strength: float       = 1.0,   # 0.6-1.0
) -> ATRRiskResult:
    """
    Calculate ATR-based SL, TP1, TP2, and position sizes.

    Args:
        df:              OHLCV bars (5m)
        entry:           proposed entry price
        equity:          current portfolio value
        swing_low:       recent swing low from market_structure (optional)
        signal_strength: from momentum_bot (0.6–1.0)
    """
    INVALID = ATRRiskResult(
        entry=entry, stop_loss=0, take_profit_1=0, take_profit_2=0,
        stop_distance=0, atr=0, risk_reward_1=0, risk_reward_2=0,
        qty_full=0, qty_half=0, valid=False, reason="ATR calculation failed",
    )

    atr = calculate_atr(df)
    if atr <= 0 or entry <= 0:
        return INVALID

    # ── Stop loss ─────────────────────────────────────────────────
    # 1. ATR-based stop
    sl_atr    = entry - atr * ATR_SL_MULT

    # 2. Swing-low stop (if available)
    sl_swing  = (swing_low - atr * 0.3) if swing_low and swing_low > 0 else sl_atr

    # 3. Use the higher SL (closer to entry) of ATR and swing
    #    but ensure it doesn't exceed max % loss
    sl_raw    = max(sl_atr, sl_swing)

    # 4. Apply min/max % constraints
    sl_min_price = entry * (1 - MAX_SL_PCT)   # never more than 2.5% away
    sl_max_price = entry * (1 - MIN_SL_PCT)   # never less than 0.5% away

    stop_loss = max(sl_min_price, min(sl_max_price, sl_raw))
    stop_loss = round(stop_loss, 2)

    stop_dist = entry - stop_loss
    if stop_dist <= 0:
        return INVALID._replace(reason=f"Invalid stop distance: {stop_dist}")

    # ── Take profits ──────────────────────────────────────────────
    tp1 = round(entry + stop_dist * ATR_TP1_MULT / ATR_SL_MULT, 2)  # ~1.67R
    tp2 = round(entry + stop_dist * ATR_TP2_MULT / ATR_SL_MULT, 2)  # ~2.67R

    rr1 = round((tp1 - entry) / stop_dist, 2)
    rr2 = round((tp2 - entry) / stop_dist, 2)

    # ── Validate R:R ─────────────────────────────────────────────
    if rr1 < MIN_RR:
        return ATRRiskResult(
            entry=entry, stop_loss=stop_loss,
            take_profit_1=tp1, take_profit_2=tp2,
            stop_distance=round(stop_dist, 2), atr=atr,
            risk_reward_1=rr1, risk_reward_2=rr2,
            qty_full=0, qty_half=0, valid=False,
            reason=f"R:R {rr1:.2f} < minimum {MIN_RR} — skip trade",
        )

    # ── Position sizing ───────────────────────────────────────────
    # Full risk: 1% of equity per trade
    risk_full = equity * RISK_PCT_FULL
    qty_full  = max(1, int(risk_full / stop_dist))

    # Half risk: for weaker signals
    risk_half = equity * RISK_PCT_HALF
    qty_half  = max(1, int(risk_half / stop_dist))

    # Cap: never more than 5% of equity in one trade
    max_value    = equity * 0.05
    qty_full     = min(qty_full, int(max_value / entry))
    qty_half     = min(qty_half, int(max_value / entry))

    # Apply signal strength scaling
    qty_final    = qty_full if signal_strength >= 0.80 else qty_half

    return ATRRiskResult(
        entry         = entry,
        stop_loss     = stop_loss,
        take_profit_1 = tp1,
        take_profit_2 = tp2,
        stop_distance = round(stop_dist, 2),
        atr           = round(atr, 2),
        risk_reward_1 = rr1,
        risk_reward_2 = rr2,
        qty_full      = qty_final,
        qty_half      = qty_half,
        valid         = True,
        reason        = (f"ATR={atr:.2f} SL={stop_loss:.2f} "
                        f"TP1={tp1:.2f}(R{rr1}) TP2={tp2:.2f}(R{rr2}) "
                        f"qty={qty_final}"),
    )
