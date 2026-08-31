"""strategies/mean_reversion.py – Bollinger Band mean-reversion."""
from __future__ import annotations
import pandas as pd
from loguru import logger
from .base import BaseStrategy, Signal, StrategyResult


class MeanReversionStrategy(BaseStrategy):
    name = "Mean_Reversion"

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14):
        self.bb_period  = bb_period
        self.bb_std     = bb_std
        self.rsi_period = rsi_period

    def _bollinger(self, close):
        mid   = close.rolling(self.bb_period).mean()
        std   = close.rolling(self.bb_period).std()
        return mid + self.bb_std * std, mid, mid - self.bb_std * std

    def _rsi(self, close):
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=self.rsi_period - 1, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, adjust=False).mean()
        return 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        if not self._validate(df, self.bb_period + 5):
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", self.name)

        close      = df["close"]
        upper, mid, lower = self._bollinger(close)
        rsi        = self._rsi(close)
        price      = close.iloc[-1]
        prev_price = close.iloc[-2]
        curr_rsi   = rsi.iloc[-1]
        z = (price - mid.iloc[-1]) / ((upper.iloc[-1] - lower.iloc[-1]) / (2 * self.bb_std) + 1e-10)
        strength = min(abs(z) / 2, 1.0)

        if price <= lower.iloc[-1] and curr_rsi < 40:
            reason = (f"Price ₹{price:.2f} at lower BB ₹{lower.iloc[-1]:.2f}, "
                      f"RSI={curr_rsi:.1f}")
            return StrategyResult(Signal.BUY, strength, reason, self.name)

        if price >= upper.iloc[-1]:
            return StrategyResult(Signal.SELL, strength,
                                  f"Price ₹{price:.2f} at upper BB", self.name)

        if prev_price < mid.iloc[-1] and price >= mid.iloc[-1]:
            return StrategyResult(Signal.SELL, 0.6,
                                  f"Price reverted to midline ₹{mid.iloc[-1]:.2f}", self.name)

        return StrategyResult(Signal.HOLD, 0.0, f"BB z={z:+.2f}", self.name)
