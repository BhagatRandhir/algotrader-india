"""strategies/base.py"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategyResult:
    signal: Signal
    strength: float          # 0.0 – 1.0
    reason: str
    strategy_name: str = ""


class BaseStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> StrategyResult: ...

    def _validate(self, df: pd.DataFrame, min_rows: int = 50) -> bool:
        return df is not None and len(df) >= min_rows
