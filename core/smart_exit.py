"""
core/smart_exit.py  –  Smart Exit Manager (Trail-Only)

Strategy:
  - NO partial exit — hold the full position
  - Trail SL starts immediately from entry (1% below rolling peak)
  - Once price gains +1%, SL moves to breakeven automatically
  - SL only ever moves UP — never loosens
  - Exit ALL shares when price hits the trail SL
  - Additional exits: volume dry-up, RSI divergence, 3PM force

This maximises profit on strong trending moves by letting the
position run as long as price keeps making new highs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class ExitDecision:
    action:  str    # "HOLD" or "FULL"
    qty:     int    # shares to sell (0 if HOLD)
    reason:  str
    new_sl:  float  # current SL price (for dashboard display)


@dataclass
class PositionState:
    symbol:       str
    entry_price:  float
    qty:          int
    sl_price:     float        # current trailing SL (moves up, never down)
    peak_price:   float        # highest price seen since entry
    entry_time:   datetime = field(default_factory=datetime.now)


class SmartExitManager:

    TRAIL_PCT        = 0.010   # trail 1% below rolling peak
    BREAKEVEN_TRIGGER= 0.010   # move SL to breakeven once +1% gained
    DRY_UP_MULT      = 0.50    # exit if volume < 50% of avg
    FORCE_EXIT_HOUR  = 15      # 3:00 PM IST
    FORCE_EXIT_MIN   = 0

    def __init__(self):
        self._positions: dict[str, PositionState] = {}

    def register(self, symbol: str, entry_price: float,
                 qty: int, initial_sl: float):
        """Call when a BUY order is placed."""
        self._positions[symbol] = PositionState(
            symbol      = symbol,
            entry_price = entry_price,
            qty         = qty,
            sl_price    = initial_sl,
            peak_price  = entry_price,
        )
        logger.info(
            f"📌 Trail registered: {symbol}  entry=₹{entry_price:.2f}"
            f"  qty={qty}  initial_SL=₹{initial_sl:.2f}"
        )

    def clear(self, symbol: str):
        self._positions.pop(symbol, None)

    def is_registered(self, symbol: str) -> bool:
        return symbol in self._positions

    # ── Main decision ─────────────────────────────────────────────

    def evaluate(self, symbol: str, df: pd.DataFrame,
                 current_price: float) -> ExitDecision:
        """
        Called every loop for each held position.
        Updates trailing SL and returns HOLD or FULL exit.
        """
        HOLD = ExitDecision("HOLD", 0, "", 0.0)

        pos = self._positions.get(symbol)
        if not pos or pos.qty <= 0:
            return HOLD

        # ── Update peak ───────────────────────────────────────────
        pos.peak_price = max(pos.peak_price, current_price)

        # ── Compute trail SL ──────────────────────────────────────
        trail_sl = round(pos.peak_price * (1 - self.TRAIL_PCT), 2)

        # Once price is up BREAKEVEN_TRIGGER%, SL floor = entry price
        gain_pct = (current_price - pos.entry_price) / pos.entry_price
        if gain_pct >= self.BREAKEVEN_TRIGGER:
            trail_sl = max(trail_sl, pos.entry_price)   # never below entry

        # SL only ever moves UP
        if trail_sl > pos.sl_price:
            logger.info(
                f"📈 Trail SL {symbol}: ₹{pos.sl_price:.2f} → ₹{trail_sl:.2f}"
                f"  (peak=₹{pos.peak_price:.2f}  gain={gain_pct:+.2%})"
            )
            pos.sl_price = trail_sl

        # ── 1. Force exit at 3 PM IST ─────────────────────────────
        now = datetime.now(IST)
        if now.hour == self.FORCE_EXIT_HOUR and now.minute >= self.FORCE_EXIT_MIN:
            return ExitDecision(
                "FULL", pos.qty,
                f"⏰ Force exit 3:00 PM IST  P=₹{current_price:.2f}",
                pos.sl_price,
            )

        # ── 2. Trail SL hit ───────────────────────────────────────
        if current_price <= pos.sl_price:
            return ExitDecision(
                "FULL", pos.qty,
                f"🛑 Trail SL hit ₹{pos.sl_price:.2f}  P=₹{current_price:.2f}"
                f"  peak=₹{pos.peak_price:.2f}  gain={gain_pct:+.2%}",
                pos.sl_price,
            )

        # ── 3. Volume dry-up ──────────────────────────────────────
        if self._volume_drying(df):
            return ExitDecision(
                "FULL", pos.qty,
                f"📉 Volume dry-up  P=₹{current_price:.2f}",
                pos.sl_price,
            )

        # ── 4. RSI divergence ─────────────────────────────────────
        if self._rsi_divergence(df):
            return ExitDecision(
                "FULL", pos.qty,
                f"⚠️ RSI divergence  P=₹{current_price:.2f}",
                pos.sl_price,
            )

        return ExitDecision("HOLD", 0, "", pos.sl_price)

    # ── Helpers ───────────────────────────────────────────────────

    def _volume_drying(self, df: pd.DataFrame) -> bool:
        if df is None or len(df) < 25:
            return False
        avg_vol = df["volume"].iloc[-22:-2].mean()
        recent  = df["volume"].iloc[-2:].mean()
        return bool(recent < self.DRY_UP_MULT * avg_vol)

    def _rsi_divergence(self, df: pd.DataFrame) -> bool:
        """Price making new high but RSI falling — reversal warning."""
        if df is None or len(df) < 20:
            return False
        try:
            close    = df["close"]
            delta    = close.diff()
            gain     = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss     = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
            rsi_ser  = 100 - 100 / (1 + gain / (loss + 1e-10))
            price_now  = float(close.iloc[-1])
            price_5ago = float(close.iloc[-6])
            rsi_now    = float(rsi_ser.iloc[-1])
            rsi_5ago   = float(rsi_ser.iloc[-6])
            return price_now > price_5ago and rsi_now < rsi_5ago - 3
        except Exception:
            return False

    def get_current_sl(self, symbol: str) -> Optional[float]:
        pos = self._positions.get(symbol)
        return pos.sl_price if pos else None

    def get_peak(self, symbol: str) -> Optional[float]:
        pos = self._positions.get(symbol)
        return pos.peak_price if pos else None

    def get_remaining_qty(self, symbol: str) -> int:
        pos = self._positions.get(symbol)
        return pos.qty if pos else 0
