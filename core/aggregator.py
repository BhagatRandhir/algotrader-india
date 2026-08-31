"""core/aggregator.py – Weighted ensemble vote."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from strategies.base import Signal, StrategyResult


@dataclass
class AggregatedSignal:
    final_signal: Signal
    combined_strength: float
    vote_breakdown: dict[str, str]
    reasons: list[str]


class SignalAggregator:
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def aggregate(self, results: List[StrategyResult]) -> AggregatedSignal:
        buy_score = sell_score = 0.0
        breakdown: dict[str, str] = {}
        reasons: list[str] = []

        for res in results:
            w = self.weights.get(res.strategy_name, 1.0 / len(results))
            breakdown[res.strategy_name] = res.signal.value
            reasons.append(f"[{res.strategy_name}] {res.signal.value}: {res.reason}")
            if res.signal == Signal.BUY:
                buy_score  += w * res.strength
            elif res.signal == Signal.SELL:
                sell_score += w * res.strength

        if buy_score > sell_score and buy_score > 0.30:
            return AggregatedSignal(Signal.BUY,  buy_score,  breakdown, reasons)
        if sell_score > buy_score and sell_score > 0.30:
            return AggregatedSignal(Signal.SELL, sell_score, breakdown, reasons)
        return AggregatedSignal(Signal.HOLD, 0.0, breakdown, reasons)
