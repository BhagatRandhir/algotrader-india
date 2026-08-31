"""
core/global_sentiment.py  –  Priority 1: Global Market Sentiment Filter

Checks 4 global signals before market opens (or any time):
  1. SGX Nifty       — best early read on how NSE will open
  2. US Futures      — S&P 500 (ES=F) and Nasdaq (NQ=F) overnight direction
  3. Crude Oil       — Brent crude (BZ=F); spike hurts India
  4. Dollar Index    — DXY (UUP ETF proxy); strong dollar = FII outflow

Decision:
  BULLISH  → all clear, bot trades normally
  NEUTRAL  → cautious, reduce position sizes by 50%
  BEARISH  → block all new BUY entries for the session

Data source: yfinance (free, no API key needed)
"""

from __future__ import annotations

import time
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from utils.yf_helpers import flatten_yf_columns

console = Console()


class GlobalMood(Enum):
    BULLISH = "BULLISH"    # green light — trade normally
    NEUTRAL = "NEUTRAL"    # amber — reduce size
    BEARISH = "BEARISH"    # red — no new BUY entries


@dataclass
class GlobalSignal:
    ticker:      str
    name:        str
    last_price:  float
    change_pct:  float
    signal:      str        # "BULLISH" | "BEARISH" | "NEUTRAL"
    reason:      str


@dataclass
class GlobalSentimentResult:
    mood:           GlobalMood
    size_multiplier: float          # 1.0 = full size, 0.5 = half, 0.0 = no entry
    signals:        list[GlobalSignal] = field(default_factory=list)
    bull_count:     int = 0
    bear_count:     int = 0
    summary:        str = ""

    @property
    def allow_entry(self) -> bool:
        return self.mood != GlobalMood.BEARISH


class GlobalSentimentFilter:
    """
    Fetches global market data and scores overall sentiment.
    Designed to run once before 9:15 AM IST each morning.
    """

    # Thresholds — tweak in .env if needed
    def __init__(
        self,
        sgx_bear_pct:    float = -0.5,   # SGX Nifty down > 0.5% → bearish
        us_bear_pct:     float = -0.7,   # S&P500 futures down > 0.7% → bearish
        crude_bear_pct:  float =  2.0,   # Crude up > 2% → bearish for India
        dxy_bear_pct:    float =  0.5,   # Dollar up > 0.5% → bearish
        sgx_bull_pct:    float =  0.4,
        us_bull_pct:     float =  0.5,
        crude_bull_pct:  float = -1.5,   # Crude DOWN > 1.5% → bullish
        dxy_bull_pct:    float = -0.3,   # Dollar DOWN → bullish
    ):
        self.thresholds = {
            # (tickers_list, name, bear_threshold, bull_threshold, inverted)
            # Each entry has a LIST of tickers — tries each in order until one works.
            # inverted=True means higher value = bearish (crude, DXY)
            # ^NSEI fails on some yfinance versions — NIFTYBEES.NS is a reliable NSE ETF proxy
            "SGX":   (["NIFTYBEES.NS", "^NSEI", "NIFTY_50.NS"], "SGX/Nifty",     sgx_bear_pct,   sgx_bull_pct,   False),
            "SP500": (["ES=F", "SPY"],                           "S&P500 Futures", us_bear_pct,    us_bull_pct,    False),
            "NQ":    (["NQ=F", "QQQ"],                           "Nasdaq Futures", us_bear_pct,    us_bull_pct,    False),
            "CRUDE": (["BZ=F", "CL=F"],                          "Brent Crude",    crude_bear_pct, crude_bull_pct, True),
            "DXY":   (["UUP", "DX-Y.NYB"],                       "Dollar Index",   dxy_bear_pct,   dxy_bull_pct,   True),
        }

    # ── Fetch one ticker (with fallback list) ────────────────────

    def _fetch(self, tickers: list[str], retries: int = 2) -> Optional[pd.DataFrame]:
        """
        Try each ticker in `tickers` in order until one returns valid data.
        Handles None return (yfinance bug with ^ prefix tickers) safely.
        """
        for ticker in tickers:
            for attempt in range(1, retries + 1):
                try:
                    # Suppress yfinance's own stderr/warning output for failed
                    # tickers — it prints "Failed download: TypeError NoneType..."
                    # even when our code handles it gracefully. We log our own
                    # cleaner message instead.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
                        raw = yf.download(
                            ticker, period="2d", interval="1h",
                            auto_adjust=True, progress=False,
                        )
                    # yf.download can return None for some futures/index tickers
                    if raw is None:
                        raise ValueError("yf.download returned None")
                    if raw.empty:
                        raise ValueError("empty DataFrame")
                    df = flatten_yf_columns(raw)
                    if "close" not in df.columns:
                        raise ValueError("no close column after flattening")
                    if df["close"].dropna().empty:
                        raise ValueError("all close values are NaN")
                    logger.debug(f"Fetched {ticker} successfully")
                    return df
                except Exception as exc:
                    if attempt < retries:
                        time.sleep(1)
                    else:
                        logger.debug(f"{ticker}: failed ({exc})")
        logger.warning(f"All fallback tickers failed: {tickers}")
        return None

    def _change_pct(self, df: pd.DataFrame) -> float:
        """% change from previous session close to latest price."""
        if df is None or len(df) < 2:
            return 0.0
        close = df["close"].dropna()
        if len(close) < 2:
            return 0.0
        prev  = close.iloc[-2]
        curr  = close.iloc[-1]
        return round((curr - prev) / prev * 100, 3)

    # ── Score one signal ─────────────────────────────────────────

    def _score_signal(
        self,
        key: str,
        change_pct: float,
        last_price: float,
    ) -> GlobalSignal:
        tickers, name, bear_thr, bull_thr, inverted = self.thresholds[key]
        ticker = tickers[0]   # use primary ticker name for display

        if not inverted:
            # Normal: positive change = bullish
            if change_pct <= bear_thr:
                signal, reason = "BEARISH", f"{change_pct:+.2f}% (below {bear_thr}% threshold)"
            elif change_pct >= bull_thr:
                signal, reason = "BULLISH", f"{change_pct:+.2f}% (above {bull_thr}% threshold)"
            else:
                signal, reason = "NEUTRAL", f"{change_pct:+.2f}% (within neutral range)"
        else:
            # Inverted: positive change = bearish (crude oil, DXY)
            if change_pct >= bear_thr:
                signal, reason = "BEARISH", f"{change_pct:+.2f}% rise (above {bear_thr}% threshold)"
            elif change_pct <= bull_thr:
                signal, reason = "BULLISH", f"{change_pct:+.2f}% fall (below {bull_thr}% threshold)"
            else:
                signal, reason = "NEUTRAL", f"{change_pct:+.2f}% (within neutral range)"

        return GlobalSignal(
            ticker=ticker, name=name, last_price=last_price,
            change_pct=change_pct, signal=signal, reason=reason,
        )

    # ── Main run ─────────────────────────────────────────────────

    def analyse(self) -> GlobalSentimentResult:
        """
        Fetch all global signals and return an overall sentiment verdict.
        Call this once before 9:15 AM IST each trading day.
        """
        signals: list[GlobalSignal] = []

        for key, (tickers, name, *_) in self.thresholds.items():
            df = self._fetch(tickers)
            if df is None or df.empty:
                logger.warning(f"Could not fetch {name} ({tickers}) — skipping")
                continue
            chg   = self._change_pct(df)
            price = round(float(df["close"].dropna().iloc[-1]), 2)
            sig   = self._score_signal(key, chg, price)
            signals.append(sig)
            logger.debug(f"{name}: {chg:+.2f}%  →  {sig.signal}")

        if not signals:
            logger.warning("No global data available — defaulting to NEUTRAL")
            return GlobalSentimentResult(
                mood=GlobalMood.NEUTRAL, size_multiplier=0.5,
                summary="No global data — proceeding cautiously",
            )

        bull = sum(1 for s in signals if s.signal == "BULLISH")
        bear = sum(1 for s in signals if s.signal == "BEARISH")
        total = len(signals)

        # ── Decision logic ────────────────────────────────────────
        if bear >= 3:
            # 3+ bearish signals → block all buys
            mood       = GlobalMood.BEARISH
            multiplier = 0.0
            summary    = f"{bear}/{total} global signals bearish — BUY entries BLOCKED"
        elif bear >= 2 or bull == 0:
            # 2 bearish or no bullish → cautious
            mood       = GlobalMood.NEUTRAL
            multiplier = 0.5
            summary    = f"{bear}/{total} bearish, {bull}/{total} bullish — position sizes halved"
        elif bull >= 3:
            # 3+ bullish → full speed
            mood       = GlobalMood.BULLISH
            multiplier = 1.0
            summary    = f"{bull}/{total} global signals bullish — full position sizing"
        else:
            mood       = GlobalMood.NEUTRAL
            multiplier = 0.75
            summary    = f"Mixed signals ({bull} bull / {bear} bear) — sizing at 75%"

        result = GlobalSentimentResult(
            mood=mood, size_multiplier=multiplier,
            signals=signals, bull_count=bull, bear_count=bear,
            summary=summary,
        )
        self._print(result)
        return result

    # ── Rich output ──────────────────────────────────────────────

    def _print(self, result: GlobalSentimentResult):
        mood_color = {"BULLISH": "green", "NEUTRAL": "yellow", "BEARISH": "red"}
        mc = mood_color.get(result.mood.value, "white")

        tbl = Table(
            title=f"🌍 Global Sentiment — [{mc}]{result.mood.value}[/]",
            box=box.SIMPLE_HEAVY, title_style=f"bold {mc}",
        )
        tbl.add_column("Market",   style="bold white", width=18)
        tbl.add_column("Price",    justify="right", width=12)
        tbl.add_column("Change",   justify="right", width=10)
        tbl.add_column("Signal",   width=10)
        tbl.add_column("Reason",   width=40)

        colors = {"BULLISH": "green", "BEARISH": "red", "NEUTRAL": "yellow"}
        for s in result.signals:
            c   = colors.get(s.signal, "white")
            chg = f"[{c}]{s.change_pct:+.2f}%[/]"
            tbl.add_row(
                s.name, f"{s.last_price:,.2f}", chg,
                f"[{c}]{s.signal}[/]", s.reason,
            )

        console.print(tbl)
        console.print(
            f"  Verdict: [{mc}]{result.summary}[/]  "
            f"│  Size multiplier: [bold]{result.size_multiplier:.0%}[/]\n"
        )
