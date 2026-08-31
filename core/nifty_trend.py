"""
core/nifty_trend.py  –  Priority 2: Nifty 50 Trend Filter

Logic:
  - Fetches daily Nifty 50 (^NSEI) bars via yfinance
  - Computes EMA(20), EMA(50), EMA(200) and ADX(14)
  - Classifies the market into 4 regimes:

      STRONG_UPTREND   → Nifty > EMA20 > EMA50 > EMA200, ADX > 25
                         Full position sizing, all entries allowed

      WEAK_UPTREND     → Nifty > EMA50, but below EMA20 or ADX weak
                         Entries allowed, size reduced to 75%

      DOWNTREND        → Nifty < EMA50
                         No new BUY entries; only hold existing positions

      SIDEWAYS         → EMA20 ≈ EMA50 (within 0.5%), ADX < 20
                         Entries allowed only for mean-reversion signals

  - Result cached for the trading day (re-fetched next morning)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich import box

from utils.yf_helpers import flatten_yf_columns

console = Console()


class MarketRegime(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    WEAK_UPTREND   = "WEAK_UPTREND"
    SIDEWAYS       = "SIDEWAYS"
    DOWNTREND      = "DOWNTREND"


@dataclass
class NiftyTrendResult:
    regime:          MarketRegime
    size_multiplier: float          # applied on top of global sentiment multiplier
    allow_entry:     bool
    nifty_price:     float
    ema20:           float
    ema50:           float
    ema200:          float
    adx:             float
    reason:          str

    # Only mean-reversion entries allowed in sideways market
    mean_reversion_only: bool = False


class NiftyTrendFilter:
    """
    Classifies the current Nifty 50 market regime.
    Call .analyse() once per day before 9:15 AM IST.
    """

    # Fallback list — tries each ticker until one returns valid daily data
    NIFTY_TICKERS = ["NIFTYBEES.NS", "^NSEI", "NIFTY_50.NS"]

    def __init__(
        self,
        ema_fast:   int   = 20,
        ema_mid:    int   = 50,
        ema_slow:   int   = 200,
        adx_period: int   = 14,
        adx_strong: float = 25.0,
        adx_weak:   float = 20.0,
        sideways_band_pct: float = 0.5,   # EMA20 within 0.5% of EMA50 = sideways
        sideways_adx_max:  float = 23.0,  # ADX below this = low trend strength
    ):
        self.ema_fast          = ema_fast
        self.ema_mid           = ema_mid
        self.ema_slow          = ema_slow
        self.adx_period        = adx_period
        self.adx_strong        = adx_strong
        self.adx_weak          = adx_weak
        self.sideways_band_pct = sideways_band_pct
        self.sideways_adx_max  = sideways_adx_max

    # ── Indicators ────────────────────────────────────────────────

    def _emas(self, close: pd.Series) -> tuple[float, float, float]:
        e20  = close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        e50  = close.ewm(span=self.ema_mid,  adjust=False).mean().iloc[-1]
        e200 = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        return round(e20, 2), round(e50, 2), round(e200, 2)

    def _adx(self, df: pd.DataFrame) -> float:
        """Average Directional Index (Wilder smoothing)."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        p     = self.adx_period

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        # Directional movement
        up   = high - high.shift()
        down = low.shift() - low
        pdm  = up.where((up > down) & (up > 0), 0.0)
        ndm  = down.where((down > up) & (down > 0), 0.0)

        # Wilder smoothing
        atr   = tr.ewm(alpha=1/p,  adjust=False).mean()
        pdi14 = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / (atr + 1e-10)
        ndi14 = 100 * ndm.ewm(alpha=1/p, adjust=False).mean() / (atr + 1e-10)

        dx    = 100 * (pdi14 - ndi14).abs() / (pdi14 + ndi14 + 1e-10)
        adx   = dx.ewm(alpha=1/p, adjust=False).mean()
        return round(float(adx.iloc[-1]), 2)

    # ── Fetch ─────────────────────────────────────────────────────

    def _fetch(self, retries: int = 2) -> Optional[pd.DataFrame]:
        """Try each ticker in NIFTY_TICKERS until one returns valid 1-year daily data."""
        for ticker in self.NIFTY_TICKERS:
            for attempt in range(1, retries + 1):
                try:
                    raw = yf.download(
                        ticker, period="1y", interval="1d",
                        auto_adjust=True, progress=False,
                    )
                    # Safely handle None (yfinance bug with ^ tickers)
                    if raw is None:
                        raise ValueError("yf.download returned None")
                    if raw.empty:
                        raise ValueError("empty response")
                    df = flatten_yf_columns(raw).dropna()
                    if "close" not in df.columns or df["close"].empty:
                        raise ValueError("no close data after flattening")
                    logger.debug(f"Nifty data fetched via {ticker}: {len(df)} rows")
                    return df
                except Exception as exc:
                    if attempt < retries:
                        time.sleep(1)
                    else:
                        logger.debug(f"Nifty fetch failed for {ticker}: {exc}")
        logger.warning(f"All Nifty tickers failed: {self.NIFTY_TICKERS}")
        return None

    # ── Classify ──────────────────────────────────────────────────

    def _classify(
        self,
        price: float,
        ema20: float, ema50: float, ema200: float,
        adx:   float,
    ) -> NiftyTrendResult:

        ema_spread_pct = abs(ema20 - ema50) / ema50 * 100

        # ── STRONG UPTREND ────────────────────────────────────────
        if (price > ema20 > ema50 > ema200) and adx >= self.adx_strong:
            return NiftyTrendResult(
                regime          = MarketRegime.STRONG_UPTREND,
                size_multiplier = 1.0,
                allow_entry     = True,
                nifty_price=price, ema20=ema20, ema50=ema50, ema200=ema200, adx=adx,
                reason = (
                    f"Nifty {price:,.0f} > EMA{self.ema_fast} {ema20:,.0f} > "
                    f"EMA{self.ema_mid} {ema50:,.0f} > EMA{self.ema_slow} {ema200:,.0f}, "
                    f"ADX={adx:.1f} (strong trend)"
                ),
            )

        # ── DOWNTREND ─────────────────────────────────────────────
        if price < ema50:
            return NiftyTrendResult(
                regime          = MarketRegime.DOWNTREND,
                size_multiplier = 0.0,
                allow_entry     = False,
                nifty_price=price, ema20=ema20, ema50=ema50, ema200=ema200, adx=adx,
                reason = (
                    f"Nifty {price:,.0f} below EMA{self.ema_mid} {ema50:,.0f} — "
                    f"market in downtrend, no new BUY entries"
                ),
            )

        # ── SIDEWAYS ──────────────────────────────────────────────
        if ema_spread_pct <= self.sideways_band_pct and adx < self.sideways_adx_max:
            return NiftyTrendResult(
                regime               = MarketRegime.SIDEWAYS,
                size_multiplier      = 0.6,
                allow_entry          = True,
                mean_reversion_only  = True,
                nifty_price=price, ema20=ema20, ema50=ema50, ema200=ema200, adx=adx,
                reason = (
                    f"EMA{self.ema_fast}/EMA{self.ema_mid} spread {ema_spread_pct:.2f}% "
                    f"(< {self.sideways_band_pct}%), ADX={adx:.1f} — "
                    f"sideways market, mean-reversion entries only"
                ),
            )

        # ── WEAK UPTREND ──────────────────────────────────────────
        return NiftyTrendResult(
            regime          = MarketRegime.WEAK_UPTREND,
            size_multiplier = 0.75,
            allow_entry     = True,
            nifty_price=price, ema20=ema20, ema50=ema50, ema200=ema200, adx=adx,
            reason = (
                f"Nifty {price:,.0f} above EMA{self.ema_mid} {ema50:,.0f} "
                f"but trend weak (ADX={adx:.1f}) — reduced sizing"
            ),
        )

    # ── Public API ────────────────────────────────────────────────

    def analyse(self) -> NiftyTrendResult:
        df = self._fetch()

        if df is None or len(df) < self.ema_slow + 5:
            logger.warning("Nifty data unavailable — defaulting to WEAK_UPTREND")
            return NiftyTrendResult(
                regime=MarketRegime.WEAK_UPTREND, size_multiplier=0.75,
                allow_entry=True, nifty_price=0, ema20=0, ema50=0, ema200=0,
                adx=0, reason="No data — proceeding cautiously at 75% size",
            )

        close = df["close"]
        price = round(float(close.iloc[-1]), 2)
        ema20, ema50, ema200 = self._emas(close)
        adx = self._adx(df)

        result = self._classify(price, ema20, ema50, ema200, adx)
        self._print(result)
        return result

    # ── Rich output ───────────────────────────────────────────────

    def _print(self, r: NiftyTrendResult):
        colors = {
            MarketRegime.STRONG_UPTREND: "green",
            MarketRegime.WEAK_UPTREND:   "yellow",
            MarketRegime.SIDEWAYS:       "cyan",
            MarketRegime.DOWNTREND:      "red",
        }
        c = colors[r.regime]
        console.print(Panel(
            f"  [bold]Nifty 50[/]  ₹{r.nifty_price:,.2f}  │  "
            f"EMA{20}={r.ema20:,.0f}  EMA{50}={r.ema50:,.0f}  EMA{200}={r.ema200:,.0f}  │  "
            f"ADX={r.adx:.1f}\n\n"
            f"  Regime: [{c} bold]{r.regime.value}[/]  │  "
            f"Size: [bold]{r.size_multiplier:.0%}[/]  │  "
            f"Entry: {'✅' if r.allow_entry else '❌ BLOCKED'}"
            + (f"  │  [cyan]Mean-reversion only[/]" if r.mean_reversion_only else "") +
            f"\n  {r.reason}",
            title="📉 Nifty Trend Filter",
            style=f"bold {c}",
            box=box.DOUBLE,
        ))
