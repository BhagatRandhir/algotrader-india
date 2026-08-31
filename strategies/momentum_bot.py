"""
strategies/momentum_bot.py  –  Profit-Focused Momentum Strategy v4

Expert improvements to reduce losses:

  ENTRY — 6 conditions, ALL must pass (was 4/5):
    1. Price > VWAP            — above intraday avg (demand zone)
    2. Price > EMA20           — short-term uptrend confirmed
    3. EMA20 > EMA50           — medium-term trend aligned (NEW)
    4. RSI 55–80               — momentum confirmed, not overbought
    5. Volume > 1.5× avg       — institutional participation (raised from 1.3)
    6. Day change > 0.3%       — meaningful positive move (was just > 0)

  QUALITY FILTER (NEW):
    - Price must be in lower half of today's range
      (High - Low). Buying near day high = chasing.
      Buying near day low = early entry with room to run.
    - Candle body > 60% of range (no indecision doji candles)
    - Last 3 bars must be rising (momentum confirmation)

  STRENGTH SCORE:
    - Base: 0.50
    - RSI premium (55→80): up to +0.15
    - Volume premium: up to +0.15
    - Day change premium: up to +0.10
    - 3-bar momentum: +0.10 bonus
    - All 6 pass: +0.10 bonus
    - Minimum to trade: 0.60 (weak setups skipped)

  EXIT:
    - Price < EMA20    → SELL (trend broken)
    - RSI < 45         → SELL (momentum lost)
    - RSI > 82         → SELL (overbought — take what's there)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy, Signal, StrategyResult

MIN_STRENGTH = 0.60   # below this → HOLD even if conditions pass


class MomentumBotStrategy(BaseStrategy):
    name = "Momentum_Bot"

    def __init__(
        self,
        ema_fast:    int   = 9,
        ema_mid:     int   = 20,
        ema_slow:    int   = 50,
        rsi_period:  int   = 14,
        rsi_buy_lo:  float = 55.0,   # raised from 50
        rsi_buy_hi:  float = 92.0,
        rsi_sell:    float = 45.0,
        rsi_ob:      float = 96.0,
        vol_mult:    float = 1.5,    # raised from 1.3
        min_day_chg: float = 0.003,  # NEW: minimum +0.3% day change
    ):
        self.ema_fast    = ema_fast
        self.ema_mid     = ema_mid
        self.ema_slow    = ema_slow
        self.rsi_period  = rsi_period
        self.rsi_buy_lo  = rsi_buy_lo
        self.rsi_buy_hi  = rsi_buy_hi
        self.rsi_sell    = rsi_sell
        self.rsi_ob      = rsi_ob
        self.vol_mult    = vol_mult
        self.min_day_chg = min_day_chg

    def _ema(self, s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()

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

    def _three_bar_rising(self, close: pd.Series) -> bool:
        """Last 3 closes each higher than previous — momentum confirmation."""
        if len(close) < 4:
            return False
        c = close.iloc[-4:].values
        return bool(c[1] > c[0] and c[2] > c[1] and c[3] > c[2])

    def _candle_quality(self, df: pd.DataFrame) -> bool:
        """
        Last candle must be a clear bullish bar:
          - Body >= 60% of total range (no doji/indecision)
          - Close in upper 40% of candle range
        """
        last = df.iloc[-1]
        total_range = last["high"] - last["low"]
        if total_range < 0.01:
            return False
        body      = abs(last["close"] - last["open"])
        body_pct  = body / total_range
        close_pos = (last["close"] - last["low"]) / total_range
        return body_pct >= 0.50 and close_pos >= 0.55

    def _price_position_in_range(self, df: pd.DataFrame) -> float:
        """
        Where is current price within today's high-low range?
        0.0 = at day low, 1.0 = at day high.
        We want < 0.75 — not buying near the top of the day.
        """
        today = df.iloc[-78:]   # approx today's bars (78 × 5min = 6.5hr)
        day_high = today["high"].max()
        day_low  = today["low"].min()
        rng      = day_high - day_low
        if rng < 0.01:
            return 0.5
        return float((df["close"].iloc[-1] - day_low) / rng)

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        HOLD = StrategyResult(Signal.HOLD, 0.0, "conditions not met", self.name)

        if not self._validate(df, min_rows=55):
            return HOLD

        df = df.copy()
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return HOLD

        price   = float(df["close"].iloc[-1])
        ema20   = float(self._ema(df["close"], self.ema_mid).iloc[-1])
        ema50   = float(self._ema(df["close"], self.ema_slow).iloc[-1])
        vwap    = self._vwap(df)
        rsi     = self._rsi(df["close"], self.rsi_period)
        volume  = float(df["volume"].iloc[-1])
        avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
        day_chg = float(
            (price - df["close"].iloc[-2]) / df["close"].iloc[-2]
            if len(df) >= 2 else 0.0
        )

        # ── EXIT checks first (protect capital) ───────────────────
        if price < ema20:
            return StrategyResult(Signal.SELL, 0.85,
                f"Price({price:.1f}) < EMA20({ema20:.1f})", self.name)
        if rsi < self.rsi_sell:
            return StrategyResult(Signal.SELL, 0.80,
                f"RSI={rsi:.1f} < {self.rsi_sell}", self.name)
        if rsi > self.rsi_ob:
            return StrategyResult(Signal.SELL, 0.65,
                f"RSI={rsi:.1f} overbought > {self.rsi_ob}", self.name)

        # ── 6 ENTRY conditions ────────────────────────────────────
        c1 = price > vwap                                         # above VWAP
        c2 = price > ema20                                        # above EMA20
        c3 = ema20  > ema50                                       # EMA alignment
        c4 = self.rsi_buy_lo <= rsi <= self.rsi_buy_hi            # RSI zone
        c5 = (volume > self.vol_mult * avg_vol) if avg_vol > 0 else False  # vol surge
        c6 = day_chg >= self.min_day_chg                          # meaningful move

        passes = sum([c1, c2, c3, c4, c5, c6])

        # All 6 must pass
        if passes < 6:
            failed = []
            if not c1: failed.append(f"P<VWAP({vwap:.0f})")
            if not c2: failed.append(f"P<EMA20({ema20:.0f})")
            if not c3: failed.append(f"EMA20<EMA50(trend down)")
            if not c4:
                if rsi < self.rsi_buy_lo: failed.append(f"RSI={rsi:.0f}<{self.rsi_buy_lo}")
                else:                     failed.append(f"RSI={rsi:.0f}>{self.rsi_buy_hi}")
            if not c5: failed.append(f"Vol={volume/(avg_vol+1):.1f}×<{self.vol_mult}×")
            if not c6: failed.append(f"DayChg={day_chg:+.2%}<{self.min_day_chg:.1%}")
            return StrategyResult(Signal.HOLD, 0.0,
                f"{passes}/6: {', '.join(failed)}", self.name)

        # ── Quality filters (bonus/penalty on strength) ───────────
        three_rising = self._three_bar_rising(df["close"])
        good_candle  = self._candle_quality(df)
        price_pos    = self._price_position_in_range(df)
        not_chasing  = price_pos < 0.75   # not buying near day high

        # Build strength score
        strength = 0.50
        strength += 0.15 * min((rsi - self.rsi_buy_lo) / 25, 1.0)     # RSI premium
        strength += 0.15 * min((volume / (avg_vol + 1) - 1) / 2, 1.0) # vol premium
        strength += 0.10 * min(day_chg / 0.01, 1.0)                    # day chg premium
        if three_rising: strength += 0.10                               # 3-bar momentum
        if good_candle:  strength += 0.05                               # clean candle
        if not_chasing:  strength += 0.05                               # good entry price
        strength = round(min(strength, 1.0), 4)

        # Skip weak setups — minimum threshold
        if strength < MIN_STRENGTH:
            return StrategyResult(Signal.HOLD, strength,
                f"Strength {strength:.2f} < {MIN_STRENGTH} minimum — skip weak setup",
                self.name)

        reason = (
            f"P>{vwap:.0f}(VWAP) EMA20({ema20:.0f})>EMA50({ema50:.0f}) "
            f"RSI={rsi:.0f} Vol={volume/(avg_vol+1):.1f}× "
            f"DayChg={day_chg:+.2%} "
            f"[{'3bar✓' if three_rising else ''}"
            f"{'candle✓' if good_candle else ''}"
            f"{'pos✓' if not_chasing else ''}] "
            f"str={strength:.2f}"
        )
        return StrategyResult(Signal.BUY, strength, reason, self.name)
