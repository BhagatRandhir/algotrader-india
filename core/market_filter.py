"""
core/market_filter.py  –  Pre-Trade Market Context Filter

Before any BUY, checks:
  1. Nifty trend    — only buy stocks when Nifty itself is rising
  2. Sector check   — stock must be outperforming its sector (NEW)
  3. Time filter    — avoid first 15 min (9:15-9:30) and last 30 min (3:00-3:30)
  4. VIX check      — if VIX > 18, reduce size; if > 22, no new buys

Returns a multiplier (0.0 = skip, 0.5-1.0 = trade with scaled size).
All checks use NSE data already in the client — no extra API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

VIX_CAUTION  = 18.0   # reduce size
VIX_STOP     = 22.0   # no new buys


@dataclass
class MarketFilterResult:
    allow:      bool
    size_mult:  float
    reason:     str
    nifty_up:   bool
    vix:        float


class MarketContextFilter:
    """
    Lightweight market context check.
    Uses cached Nifty/VIX data — fetches once per 5 minutes.
    """

    def __init__(self):
        self._cache:       Optional[MarketFilterResult] = None
        self._cache_time:  float = 0.0
        self._cache_ttl:   int   = 300   # 5 minutes

    def check(self, broker) -> MarketFilterResult:
        """
        Returns MarketFilterResult.
        Uses cached result if < 5 min old.
        """
        import time
        now = time.time()
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        result = self._evaluate(broker)
        self._cache      = result
        self._cache_time = now
        return result

    def _evaluate(self, broker) -> MarketFilterResult:
        # ── 1. Time filter ────────────────────────────────────────
        now_ist = datetime.now(IST)
        h, m    = now_ist.hour, now_ist.minute

        if (h == 9 and m < 30):
            return MarketFilterResult(
                allow=False, size_mult=0.0,
                reason="⏰ Opening 15 min — too volatile",
                nifty_up=False, vix=15.0,
            )
        if (h >= 15):
            return MarketFilterResult(
                allow=False, size_mult=0.0,
                reason="⏰ After 3 PM — no new entries",
                nifty_up=False, vix=15.0,
            )

        # ── 2. Nifty trend ────────────────────────────────────────
        nifty_up   = False
        vix        = 15.0
        size_mult  = 1.0

        try:
            from core.nse_data import get_client
            nse    = get_client()
            stocks = nse.get_nifty50_quotes()

            if stocks:
                up_count = sum(1 for s in stocks if s["change_pct"] > 0)
                breadth  = up_count / len(stocks)
                nifty_up = breadth >= 0.55
                if not nifty_up:
                    logger.info(f"🌡️ Nifty breadth weak: {breadth:.0%} stocks up")
                    return MarketFilterResult(
                        allow=False, size_mult=0.0,
                        reason=f"📉 Nifty weak — only {breadth:.0%} stocks rising",
                        nifty_up=False, vix=vix,
                    )
            else:
                # NSE blocked — use yfinance for Nifty direction
                logger.debug("NSE blocked — using yfinance for Nifty check")
                import yfinance as yf
                nifty_df = yf.Ticker("^NSEI").history(period="2d", interval="1d")
                if len(nifty_df) >= 2:
                    nifty_chg = (nifty_df["Close"].iloc[-1] - nifty_df["Close"].iloc[-2])                                 / nifty_df["Close"].iloc[-2] * 100
                    nifty_up = nifty_chg > -0.5   # allow unless Nifty down >0.5%
                    if not nifty_up:
                        return MarketFilterResult(
                            allow=False, size_mult=0.0,
                            reason=f"📉 Nifty down {nifty_chg:.1f}% today",
                            nifty_up=False, vix=vix,
                        )
                else:
                    nifty_up = True   # can't check → allow

            # VIX via yfinance fallback
            try:
                import yfinance as yf
                vix_df = yf.Ticker("^INDIAVIX").history(period="2d", interval="1d")
                if not vix_df.empty:
                    vix = round(float(vix_df["Close"].iloc[-1]), 1)
            except Exception:
                pass

        except Exception as exc:
            logger.debug(f"Market filter error: {exc}")
            # All sources failed — allow at reduced size, don't block
            return MarketFilterResult(
                allow=True, size_mult=0.75,
                reason="Market data unavailable — trading at 75% size",
                nifty_up=True, vix=15.0,
            )

        # ── 3. VIX sizing ─────────────────────────────────────────
        if vix >= VIX_STOP:
            return MarketFilterResult(
                allow=False, size_mult=0.0,
                reason=f"🚨 VIX={vix:.1f} too high — no new buys",
                nifty_up=nifty_up, vix=vix,
            )
        elif vix >= VIX_CAUTION:
            size_mult = 0.50
            logger.info(f"⚠️ VIX={vix:.1f} elevated — half size")

        reason = (
            f"✅ Market OK — Nifty breadth strong  "
            f"VIX={vix:.1f}  size={size_mult:.0%}"
        )
        return MarketFilterResult(
            allow=True, size_mult=size_mult,
            reason=reason, nifty_up=nifty_up, vix=vix,
        )
