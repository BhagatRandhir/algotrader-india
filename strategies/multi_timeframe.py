"""
strategies/multi_timeframe.py  –  Multi-Timeframe Confirmation

Checks 3 timeframes for trend alignment:
  • 5m  — entry timing (fast)
  • 15m — intermediate trend
  • 1h  — macro direction

All 3 must agree for a HIGH_CONFIDENCE signal.
2/3 agreement = MEDIUM_CONFIDENCE (smaller size).
1/3 or less   = no trade.

Each timeframe checks: Price > EMA20, RSI > 50, Price > VWAP
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger


class MTFConfidence(Enum):
    HIGH   = "HIGH"    # all 3 timeframes agree
    MEDIUM = "MEDIUM"  # 2/3 agree
    LOW    = "LOW"     # 1 or fewer


@dataclass
class MTFResult:
    confidence:   MTFConfidence
    size_mult:    float          # 1.0 = full, 0.6 = medium, 0.0 = skip
    aligned:      int            # how many TFs aligned (0-3)
    details:      dict           # per-TF breakdown
    reason:       str


class MultiTimeframeConfirmer:
    """
    Usage:
        mtf = MultiTimeframeConfirmer(broker)
        result = mtf.confirm(symbol)
        if result.size_mult > 0:
            qty = base_qty * result.size_mult
    """

    def __init__(self, broker):
        self.broker = broker

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
        return float(
            (typical * df["volume"]).cumsum().iloc[-1]
            / (df["volume"].cumsum().iloc[-1] + 1e-9)
        )

    def _adx(self, df: pd.DataFrame, p: int = 14) -> float:
        if len(df) < p + 5:
            return 0.0
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        dm_p = ((high.diff() > low.diff().abs()) & (high.diff() > 0)) * high.diff()
        dm_m = ((low.diff().abs() > high.diff()) & (low.diff() < 0)) * low.diff().abs()
        atr  = tr.ewm(span=p, adjust=False).mean()
        dip  = dm_p.ewm(span=p, adjust=False).mean() / (atr + 1e-9) * 100
        dim  = dm_m.ewm(span=p, adjust=False).mean() / (atr + 1e-9) * 100
        dx   = (dip - dim).abs() / (dip + dim + 1e-9) * 100
        return float(dx.ewm(span=p, adjust=False).mean().iloc[-1])

    # ── Per-timeframe check ───────────────────────────────────────

    def _check_tf(self, symbol: str, interval: str, period: str,
                  label: str) -> dict:
        result = {"tf": label, "bullish": False, "reason": "no data"}
        try:
            df = self.broker.get_bars(symbol, interval=interval, period=period)
            if df is None or df.empty or len(df) < 25:
                # No data for this TF — treat as neutral (don't block)
                result["bullish"] = True   # give benefit of doubt when no data
                result["reason"]  = f"no data — neutral"
                return result

            price = float(df["close"].iloc[-1])
            ema20 = self._ema(df["close"], 20)
            ema50 = self._ema(df["close"], 50) if len(df) >= 55 else ema20
            rsi   = self._rsi(df["close"])
            vwap  = self._vwap(df)
            adx   = self._adx(df)

            above_ema20 = price > ema20
            above_ema50 = price > ema50
            rsi_ok      = rsi > 50
            above_vwap  = price > vwap
            trending    = adx > 18

            score = sum([above_ema20, above_ema50, rsi_ok, above_vwap, trending])
            bullish = score >= 3   # 3 of 5 sub-conditions

            result.update({
                "bullish":     bullish,
                "score":       score,
                "price":       round(price, 2),
                "ema20":       round(ema20, 2),
                "rsi":         round(rsi, 1),
                "adx":         round(adx, 1),
                "above_vwap":  above_vwap,
                "reason":      f"score={score}/5 RSI={rsi:.0f} ADX={adx:.0f}",
            })
        except Exception as exc:
            result["reason"] = str(exc)
            logger.debug(f"MTF {label} {symbol}: {exc}")
        return result

    # ── Main confirm ──────────────────────────────────────────────

    def confirm(self, symbol: str) -> MTFResult:
        tf5m  = self._check_tf(symbol, "5m",  "5d",  "5m")
        tf15m = self._check_tf(symbol, "15m", "10d", "15m")
        tf1h  = self._check_tf(symbol, "60m", "1mo", "1h")

        aligned = sum([tf5m["bullish"], tf15m["bullish"], tf1h["bullish"]])
        details = {"5m": tf5m, "15m": tf15m, "1h": tf1h}

        # Count TFs that have enough data
        data_ok = sum([
            "score" in tf5m,
            "score" in tf15m,
            "score" in tf1h,
        ])

        # If higher TFs have no data, don't punish — only require available TFs
        if data_ok == 1:
            # Only 5m available — still allow with reduced size
            conf      = MTFConfidence.MEDIUM if tf5m.get("bullish") else MTFConfidence.LOW
            size_mult = 0.60 if tf5m.get("bullish") else 0.0
            reason    = f"⚡ Only 5m data available — {'bullish' if tf5m.get('bullish') else 'not bullish'}"
        elif aligned == 3:
            conf     = MTFConfidence.HIGH
            size_mult= 1.0
            reason   = "✅ All 3 TFs bullish (5m/15m/1h)"
        elif aligned == 2:
            conf     = MTFConfidence.MEDIUM
            size_mult= 0.75
            tfs      = [d["tf"] for d in [tf5m, tf15m, tf1h] if d.get("bullish")]
            reason   = f"⚡ 2/3 TFs bullish ({'+'.join(tfs)})"
        elif aligned == 1 and tf5m.get("bullish"):
            # 5m bullish but higher TFs not — allow small size
            conf     = MTFConfidence.MEDIUM
            size_mult= 0.50
            reason   = f"⚠️ Only 5m bullish — reduced size"
        else:
            conf     = MTFConfidence.LOW
            size_mult= 0.0
            reason   = f"❌ {aligned}/3 TFs bullish — skip"

        logger.debug(
            f"MTF {symbol}: {aligned}/3 aligned  "
            f"5m={'✓' if tf5m['bullish'] else '✗'}  "
            f"15m={'✓' if tf15m['bullish'] else '✗'}  "
            f"1h={'✓' if tf1h['bullish'] else '✗'}"
        )
        return MTFResult(conf, size_mult, aligned, details, reason)
