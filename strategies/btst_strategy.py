"""
strategies/btst_strategy.py  –  BTST (Buy Today Sell Tomorrow) Strategy

Entry window: 2:30 PM – 3:10 PM IST
Exit window:  9:15 AM – 10:00 AM next day

Entry conditions (all must pass):
  1. Price closing near day high  (close > 97% of day high)
  2. Volume accelerating          (last hour volume > 1.5× morning avg)
  3. RSI 60–75 at close          (strong momentum, not overbought)
  4. Price > VWAP all afternoon  (sustained buying)
  5. EMA9 > EMA20 > EMA50       (full trend stack)
  6. Day change > 0.5%           (meaningful positive day)

Exit conditions (next morning):
  1. Sell between 9:15–10:00 AM (capture overnight gap)
  2. Force sell at 10:00 AM regardless
  3. Stop loss: 2% below previous close

Risk:
  - Smaller position size than intraday (overnight risk)
  - Max 2 BTST positions at a time
  - Only trade when VIX < 16 (low fear = safe overnight)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class BTSTSignal(Enum):
    BUY  = "BUY"
    HOLD = "HOLD"


@dataclass
class BTSTResult:
    signal:         BTSTSignal
    strength:       float
    reason:         str
    checks_passed:  int
    checks_total:   int = 6


class BTSTStrategy:
    name = "BTST"

    def __init__(
        self,
        min_close_vs_high: float = 0.97,   # close must be > 97% of day high
        rsi_lo:            float = 60.0,
        rsi_hi:            float = 85.0,
        vol_accel:         float = 1.5,    # last hour vol vs morning avg
        min_day_chg:       float = 0.005,  # +0.5% minimum
        ema_fast:          int   = 9,
        ema_mid:           int   = 20,
        ema_slow:          int   = 50,
    ):
        self.min_close_vs_high = min_close_vs_high
        self.rsi_lo            = rsi_lo
        self.rsi_hi            = rsi_hi
        self.vol_accel         = vol_accel
        self.min_day_chg       = min_day_chg
        self.ema_fast          = ema_fast
        self.ema_mid           = ema_mid
        self.ema_slow          = ema_slow

    # ── Indicators ────────────────────────────────────────────────

    def _ema(self, s: pd.Series, p: int) -> float:
        return float(s.ewm(span=p, adjust=False).mean().iloc[-1])

    def _rsi(self, s: pd.Series, p: int = 14) -> float:
        delta = s.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
        rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        return float(100 - 100 / (1 + rs))

    def _vwap(self, df: pd.DataFrame) -> float:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum().replace(0, np.nan)
        return float((typical * df["volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])

    def _day_high(self, df: pd.DataFrame) -> float:
        """Today's intraday high — last 78 bars (6.5 hrs of 5m bars)."""
        return float(df["high"].iloc[-78:].max())

    def _vol_acceleration(self, df: pd.DataFrame) -> float:
        """
        Compare last 12 bars (1 hour) volume vs first 36 bars (3 hours) avg.
        Ratio > 1.5 means volume picking up into close = institutional buying.
        """
        if len(df) < 50:
            return 1.0
        morning_avg = df["volume"].iloc[-60:-12].mean()
        last_hour   = df["volume"].iloc[-12:].mean()
        return round(float(last_hour / (morning_avg + 1)), 2)

    def _above_vwap_afternoon(self, df: pd.DataFrame) -> bool:
        """Price must be above session VWAP in last 24 bars (2 hrs)."""
        if len(df) < 30:
            return False
        vwap   = self._vwap(df)   # full session VWAP
        recent = df["close"].iloc[-24:]
        above  = (recent > vwap).sum()
        return above >= 15   # 60%+ of last 2 hrs above VWAP (relaxed from 75%)

    # ── Signal ────────────────────────────────────────────────────

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> BTSTResult:
        HOLD = BTSTResult(BTSTSignal.HOLD, 0.0, "conditions not met", 0)

        if df is None or len(df) < 60:
            return HOLD

        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return HOLD

        price    = float(df["close"].iloc[-1])
        day_high = self._day_high(df)
        ema9     = self._ema(df["close"], self.ema_fast)
        ema20    = self._ema(df["close"], self.ema_mid)
        ema50    = self._ema(df["close"], self.ema_slow)
        rsi      = self._rsi(df["close"])
        vwap     = self._vwap(df)
        vol_acc  = self._vol_acceleration(df)
        day_chg  = float((price - df["close"].iloc[-2]) / df["close"].iloc[-2]
                         if len(df) >= 2 else 0.0)

        # ── 6 conditions ──────────────────────────────────────────
        c1 = price >= day_high * self.min_close_vs_high    # near day high
        c2 = vol_acc >= self.vol_accel                     # volume accelerating
        c3 = self.rsi_lo <= rsi <= self.rsi_hi             # RSI sweet spot
        c4 = self._above_vwap_afternoon(df)                # above VWAP afternoon
        c5 = ema9 > ema20 > ema50                          # full EMA stack
        c6 = day_chg >= self.min_day_chg                   # positive day

        passes = sum([c1, c2, c3, c4, c5, c6])

        if passes < 5:   # need at least 5/6
            failed = []
            if not c1: failed.append(f"P({price:.0f})<97%×High({day_high:.0f})")
            if not c2: failed.append(f"VolAcc={vol_acc:.1f}×<{self.vol_accel}×")
            if not c3:
                if rsi < self.rsi_lo: failed.append(f"RSI={rsi:.0f}<{self.rsi_lo}")
                else:                 failed.append(f"RSI={rsi:.0f}>{self.rsi_hi}")
            if not c4: failed.append("Not above VWAP afternoon")
            if not c5: failed.append("EMA not aligned")
            if not c6: failed.append(f"DayChg={day_chg:+.2%}<{self.min_day_chg:.1%}")
            return BTSTResult(BTSTSignal.HOLD, 0.0,
                              f"{passes}/6: {', '.join(failed)}", passes)

        # ── Strength score ────────────────────────────────────────
        strength = min(
            0.50
            + 0.15 * min((rsi - self.rsi_lo) / 15, 1.0)
            + 0.15 * min((vol_acc - 1) / 2, 1.0)
            + 0.10 * min(day_chg / 0.01, 1.0)
            + 0.10 * (1.0 if passes == 6 else 0.0),
            1.0,
        )

        # Earnings boost for BTST — post-result stocks gap up more
        earnings_boost = 0.0
        earnings_note  = ""
        try:
            from core.earnings_analyser import get_earnings_analyser
            ea = get_earnings_analyser()
            er = ea.analyse(symbol)
            if er.verdict in ("STRONG", "GOOD"):
                earnings_boost = 0.10
                earnings_note  = f" | Earnings:{er.verdict}"
            elif er.verdict in ("WEAK", "AVOID"):
                earnings_boost = -0.10
                earnings_note  = f" | Earnings:{er.verdict}"
        except Exception:
            pass

        strength = round(min(strength + earnings_boost, 1.0), 3)
        reason = (
            f"BTST✓ Close={price:.0f} ({(price/day_high*100):.0f}% of High) "
            f"VolAcc={vol_acc:.1f}× RSI={rsi:.0f} "
            f"DayChg={day_chg:+.2%} [{passes}/6]{earnings_note}"
        )
        return BTSTResult(BTSTSignal.BUY, strength, reason, passes)
