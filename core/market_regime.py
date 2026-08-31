"""
core/market_regime.py  –  Market Regime Filter

Detects whether today is a trending or choppy day using:
  1. INDIAVIX  — fear gauge (VIX > 20 = caution, > 25 = stop)
  2. Nifty ADX — trend strength (ADX < 20 = choppy, skip all trades)
  3. Nifty breadth — how many Nifty50 stocks are above their EMA20

Regime output:
  TRENDING_BULL  → full size, all trades allowed
  TRENDING_BEAR  → no new buys, manage exits only
  CHOPPY         → skip all trades (market going nowhere)
  HIGH_FEAR      → reduce size by 50%
  EXTREME_FEAR   → stop trading entirely
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

from utils.yf_helpers import flatten_yf_columns


class Regime(Enum):
    TRENDING_BULL  = "TRENDING_BULL"
    TRENDING_BEAR  = "TRENDING_BEAR"
    CHOPPY         = "CHOPPY"
    HIGH_FEAR      = "HIGH_FEAR"
    EXTREME_FEAR   = "EXTREME_FEAR"


@dataclass
class RegimeResult:
    regime:       Regime
    allow_entry:  bool
    size_mult:    float
    vix:          float
    nifty_adx:    float
    breadth_pct:  float   # % of Nifty50 stocks above EMA20
    reason:       str


# Nifty50 sample for breadth (top 20 most liquid — full 50 is slow)
NIFTY_BREADTH_SAMPLE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS",
]


class MarketRegimeFilter:

    VIX_HIGH    = 20.0
    VIX_EXTREME = 25.0
    ADX_TREND   = 20.0
    BREADTH_BULL= 0.60   # 60%+ stocks above EMA20 = broad bull

    def __init__(self):
        self._cache: RegimeResult | None = None
        self._cache_time: pd.Timestamp | None = None

    def _fetch_vix(self) -> float:
        """Fetch India VIX from NSE API."""
        try:
            from core.nse_data import get_client
            nse  = get_client()
            data = nse._get(f"https://www.nseindia.com/api/allIndices")
            if data:
                for idx in data.get("data", []):
                    if "VIX" in idx.get("index", "").upper():
                        return float(idx.get("last", 15.0))
        except Exception as exc:
            logger.debug(f"NSE VIX: {exc}")
        # yfinance fallback
        try:
            import yfinance as yf
            from utils.yf_helpers import flatten_yf_columns
            df = yf.download("^INDIAVIX", period="2d", interval="1d",
                             progress=False, auto_adjust=True)
            df = flatten_yf_columns(df)
            if not df.empty and "close" in df.columns:
                return float(df["close"].dropna().iloc[-1])
        except Exception:
            pass
        return 15.0   # safe default

    def _fetch_nifty_adx(self) -> float:
        try:
            df = yf.download("^NSEI", period="2mo", interval="1d",
                             progress=False, auto_adjust=True)
            df = flatten_yf_columns(df).dropna()
            if len(df) < 20:
                return 25.0  # assume trending
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            dm_p = ((high.diff() > low.diff().abs()) & (high.diff() > 0)) * high.diff()
            dm_m = ((low.diff().abs() > high.diff()) & (low.diff() < 0)) * low.diff().abs()
            atr  = tr.ewm(span=14, adjust=False).mean()
            dip  = dm_p.ewm(span=14, adjust=False).mean() / (atr + 1e-9) * 100
            dim  = dm_m.ewm(span=14, adjust=False).mean() / (atr + 1e-9) * 100
            dx   = (dip - dim).abs() / (dip + dim + 1e-9) * 100
            adx  = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])
            return round(adx, 2)
        except Exception as exc:
            logger.warning(f"Nifty ADX fetch: {exc}")
            return 25.0

    def _fetch_breadth(self) -> float:
        """Use NSE Nifty50 quotes for real-time breadth calculation."""
        try:
            from core.nse_data import get_client
            nse    = get_client()
            stocks = nse.get_nifty50_quotes()
            if not stocks:
                raise ValueError("no quotes")
            above = sum(1 for s in stocks
                        if s["ltp"] > 0 and s["change_pct"] > 0)
            total = len(stocks)
            pct   = round(above / total, 4) if total > 0 else 0.65
            logger.debug(f"NSE breadth: {above}/{total} stocks positive = {pct:.0%}")
            return pct
        except Exception as exc:
            logger.warning(f"NSE breadth: {exc} — using yfinance fallback")
            # yfinance fallback
            try:
                import yfinance as yf
                from utils.yf_helpers import flatten_yf_columns
                SAMPLE = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS",
                          "ICICIBANK.NS","SBIN.NS","AXISBANK.NS","WIPRO.NS",
                          "BAJFINANCE.NS","TITAN.NS"]
                above, total = 0, 0
                for tkr in SAMPLE:
                    try:
                        df = yf.download(tkr, period="3mo", interval="1d",
                                        progress=False, auto_adjust=True)
                        df = flatten_yf_columns(df).dropna()
                        if len(df) >= 22:
                            ema20 = df["close"].ewm(span=20,adjust=False).mean().iloc[-1]
                            above += int(df["close"].iloc[-1] > ema20)
                            total += 1
                    except Exception:
                        pass
                return round(above/total, 4) if total > 0 else 0.65
            except Exception:
                return 0.65   # default bullish

    def analyse(self, use_cache_mins: int = 30) -> RegimeResult:
        """Returns regime. Caches result for use_cache_mins minutes."""
        now = pd.Timestamp.now()
        if (self._cache is not None and self._cache_time is not None and
                (now - self._cache_time).total_seconds() < use_cache_mins * 60):
            return self._cache

        vix      = self._fetch_vix()
        adx      = self._fetch_nifty_adx()
        breadth  = self._fetch_breadth()

        # ── Regime logic ──────────────────────────────────────────
        if vix >= self.VIX_EXTREME:
            regime      = Regime.EXTREME_FEAR
            allow_entry = False
            size_mult   = 0.0
            reason      = f"VIX={vix:.1f} EXTREME — trading halted"

        elif vix >= self.VIX_HIGH:
            regime      = Regime.HIGH_FEAR
            allow_entry = True
            size_mult   = 0.50
            reason      = f"VIX={vix:.1f} HIGH — size halved"

        elif adx < self.ADX_TREND:
            regime      = Regime.CHOPPY
            allow_entry = False
            size_mult   = 0.0
            reason      = f"Nifty ADX={adx:.1f} < {self.ADX_TREND} — choppy day, skip"

        elif breadth >= self.BREADTH_BULL:
            regime      = Regime.TRENDING_BULL
            allow_entry = True
            size_mult   = 1.0
            reason      = (f"BULL: ADX={adx:.1f} Breadth={breadth:.0%} "
                           f"VIX={vix:.1f}")

        else:
            regime      = Regime.TRENDING_BEAR
            allow_entry = False
            size_mult   = 0.0
            reason      = (f"BEAR: Breadth={breadth:.0%}<{self.BREADTH_BULL:.0%} "
                           f"ADX={adx:.1f}")

        # If all data sources failed (yfinance blocked), allow trading at reduced size
        all_failed = (vix == 15.0 and adx == 25.0 and breadth in (0.5, 0.65)
                      and regime != Regime.EXTREME_FEAR)

        result = RegimeResult(
            regime=regime, allow_entry=allow_entry, size_mult=size_mult,
            vix=vix, nifty_adx=adx, breadth_pct=breadth, reason=reason,
        )
        self._cache      = result
        self._cache_time = now

        logger.info(
            f"🌡️  Regime: {regime.value}  VIX={vix:.1f}  "
            f"ADX={adx:.1f}  Breadth={breadth:.0%}  "
            f"allow={allow_entry}  mult={size_mult}"
        )
        return result
