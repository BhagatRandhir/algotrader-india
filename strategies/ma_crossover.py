"""strategies/ma_crossover.py – EMA 9/21 crossover with 200-bar trend filter."""
from __future__ import annotations
import pandas as pd
from loguru import logger
from .base import BaseStrategy, Signal, StrategyResult


class MACrossoverStrategy(BaseStrategy):
    name = "MA_Crossover"

    def __init__(self, fast: int = 9, slow: int = 21, trend: int = 200):
        self.fast, self.slow, self.trend = fast, slow, trend

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult:
        if not self._validate(df, self.trend + 5):
            return StrategyResult(Signal.HOLD, 0.0, "Insufficient data", self.name)

        close = df["close"]
        ema_fast  = close.ewm(span=self.fast,  adjust=False).mean()
        ema_slow  = close.ewm(span=self.slow,  adjust=False).mean()
        ema_trend = close.ewm(span=self.trend, adjust=False).mean()

        prev_diff = ema_fast.iloc[-2] - ema_slow.iloc[-2]
        curr_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
        above_trend = close.iloc[-1] > ema_trend.iloc[-1]

        strength = min(abs(curr_diff) / ema_slow.iloc[-1] * 20, 1.0)

        if prev_diff <= 0 and curr_diff > 0 and above_trend:
            reason = (f"EMA{self.fast} crossed above EMA{self.slow} "
                      f"({ema_fast.iloc[-1]:.2f} vs {ema_slow.iloc[-1]:.2f}), "
                      f"above EMA{self.trend}")
            logger.debug(f"[{self.name}] {symbol} BUY — {reason}")
            return StrategyResult(Signal.BUY, strength, reason, self.name)

        if prev_diff >= 0 and curr_diff < 0:
            reason = (f"EMA{self.fast} crossed below EMA{self.slow} "
                      f"({ema_fast.iloc[-1]:.2f} vs {ema_slow.iloc[-1]:.2f})")
            logger.debug(f"[{self.name}] {symbol} SELL — {reason}")
            return StrategyResult(Signal.SELL, strength, reason, self.name)

        return StrategyResult(Signal.HOLD, 0.0, "No crossover", self.name)
