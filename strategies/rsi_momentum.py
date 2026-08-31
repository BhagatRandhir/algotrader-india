"""strategies/rsi_momentum.py – RSI oversold/overbought with momentum confirm."""
from __future__ import annotations
import pandas as pd
from loguru import logger
from .base import BaseStrategy, Signal, StrategyResult


class RSIMomentumStrategy(BaseStrategy):
    name = "RSI_Momentum"

    def __init__(self, period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, momentum_bars: int = 5):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.momentum_bars = momentum_bars

    def _rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=self.period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=self.period - 1, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        if not self._validate(df, self.period * 3):
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", self.name)

        close    = df["close"]
        rsi      = self._rsi(close)
        prev_rsi = rsi.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        momentum = (close.iloc[-1] - close.iloc[-self.momentum_bars - 1]) / \
                   close.iloc[-self.momentum_bars - 1]
        strength = min(abs(curr_rsi - 50) / 50, 1.0)

        if prev_rsi < self.oversold and curr_rsi >= self.oversold and momentum > 0:
            reason = (f"RSI crossed out of oversold "
                      f"({prev_rsi:.1f}→{curr_rsi:.1f}), momentum={momentum:.2%}")
            return StrategyResult(Signal.BUY, strength, reason, self.name)

        if curr_rsi >= self.overbought:
            return StrategyResult(Signal.SELL, strength,
                                  f"RSI overbought ({curr_rsi:.1f})", self.name)

        if prev_rsi >= 50 and curr_rsi < 50:
            return StrategyResult(Signal.SELL, 0.5,
                                  f"RSI fell below 50 ({curr_rsi:.1f})", self.name)

        return StrategyResult(Signal.HOLD, 0.0, f"RSI={curr_rsi:.1f}", self.name)
