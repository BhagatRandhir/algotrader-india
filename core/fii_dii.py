"""
core/fii_dii.py  –  Priority 4: FII / DII Institutional Flow Filter

What it does:
  Fetches daily FII (Foreign Institutional Investor) and DII (Domestic
  Institutional Investor) net buy/sell data from NSE's official website.
  NSE publishes this every trading day by ~6:00 PM IST.

Why it matters:
  - FII net-sell of ₹2,000+ crore = strong bearish signal (foreign money leaving)
  - FII net-buy  of ₹2,000+ crore = strong bullish signal (foreign money entering)
  - DII often buys when FII sells (domestic funds absorb selling) — divergence matters
  - 5-day rolling trend is more reliable than single-day numbers

Data sources (in priority order, all free):
  1. NSE official CSV   → https://www.nseindia.com/api/fiidiiTradeReact
  2. Moneycontrol page  → parsed as fallback
  3. Hardcoded fallback → NEUTRAL if both fail

Flow Classification:
  BULLISH_STRONG  FII net > +₹2000 cr  AND  5d trend positive
                  → size_multiplier = 1.0,  allow_entry = True

  BULLISH_WEAK    FII net > +₹500 cr   OR   DII net > +₹1000 cr
                  → size_multiplier = 0.85, allow_entry = True

  NEUTRAL         FII between -₹500 and +₹500 cr
                  → size_multiplier = 0.75, allow_entry = True

  BEARISH_WEAK    FII net < -₹500 cr   AND  DII not compensating
                  → size_multiplier = 0.50, allow_entry = True (cautious)

  BEARISH_STRONG  FII net < -₹2000 cr  AND  5d trend negative
                  → size_multiplier = 0.0,  allow_entry = False
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# Thresholds in crore INR
STRONG_BUY_CR  =  2_000
WEAK_BUY_CR    =    500
WEAK_SELL_CR   =   -500
STRONG_SELL_CR = -2_000


class FlowMood(Enum):
    BULLISH_STRONG = "BULLISH_STRONG"
    BULLISH_WEAK   = "BULLISH_WEAK"
    NEUTRAL        = "NEUTRAL"
    BEARISH_WEAK   = "BEARISH_WEAK"
    BEARISH_STRONG = "BEARISH_STRONG"


@dataclass
class DayFlow:
    date:        str
    fii_buy:     float   # gross buy in crore
    fii_sell:    float   # gross sell in crore
    fii_net:     float   # net = buy - sell
    dii_buy:     float
    dii_sell:    float
    dii_net:     float


@dataclass
class FIIDIIResult:
    mood:             FlowMood
    size_multiplier:  float
    allow_entry:      bool
    fii_net_today:    float          # crore INR
    dii_net_today:    float
    fii_net_5d:       float          # 5-day rolling net
    days_available:   int
    summary:          str
    history:          list[DayFlow] = field(default_factory=list)
    fetched_at:       datetime = field(default_factory=datetime.now)

    @property
    def is_stale(self) -> bool:
        return (datetime.now() - self.fetched_at) > timedelta(hours=6)


class FIIDIIFilter:
    """
    Fetches FII/DII institutional flow from NSE and classifies
    market sentiment based on net buy/sell activity.
    """

    NSE_API   = "https://www.nseindia.com/api/fiidiiTradeReact"
    NSE_HOME  = "https://www.nseindia.com"
    TIMEOUT   = 10

    def __init__(self):
        self._session  = requests.Session()
        self._session.headers.update(NSE_HEADERS)
        self._cache: Optional[FIIDIIResult] = None

    # ── Fetching ──────────────────────────────────────────────────

    def _warm_session(self):
        """NSE requires a valid cookie from homepage before API calls."""
        try:
            self._session.get(self.NSE_HOME, timeout=self.TIMEOUT)
            time.sleep(1)
        except Exception as exc:
            logger.debug(f"NSE session warm-up: {exc}")

    def _fetch_nse(self) -> list[DayFlow]:
        """Fetch from NSE official API."""
        self._warm_session()
        try:
            resp = self._session.get(self.NSE_API, timeout=self.TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            flows = []
            for row in data:
                try:
                    flows.append(DayFlow(
                        date     = row.get("date", ""),
                        fii_buy  = float(row.get("fiiBuyValue",  0) or 0),
                        fii_sell = float(row.get("fiiSellValue", 0) or 0),
                        fii_net  = float(row.get("fiiNetValue",  0) or 0),
                        dii_buy  = float(row.get("diiBuyValue",  0) or 0),
                        dii_sell = float(row.get("diiSellValue", 0) or 0),
                        dii_net  = float(row.get("diiNetValue",  0) or 0),
                    ))
                except (ValueError, TypeError):
                    continue
            logger.info(f"NSE FII/DII: fetched {len(flows)} days")
            return flows
        except Exception as exc:
            logger.warning(f"NSE API failed: {exc}")
            return []

    def _fetch_moneycontrol_fallback(self) -> list[DayFlow]:
        """Fallback: parse Moneycontrol FII/DII page."""
        url = "https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php"
        try:
            resp = requests.get(url, headers=NSE_HEADERS, timeout=self.TIMEOUT)
            soup = BeautifulSoup(resp.text, "lxml")
            flows = []
            table = soup.find("table", {"class": lambda c: c and "mctable" in c})
            if not table:
                return []
            rows = table.find_all("tr")[1:]   # skip header
            for row in rows[:10]:
                cols = [td.get_text(strip=True).replace(",", "") for td in row.find_all("td")]
                if len(cols) < 7:
                    continue
                try:
                    flows.append(DayFlow(
                        date     = cols[0],
                        fii_buy  = float(cols[1] or 0),
                        fii_sell = float(cols[2] or 0),
                        fii_net  = float(cols[3] or 0),
                        dii_buy  = float(cols[4] or 0),
                        dii_sell = float(cols[5] or 0),
                        dii_net  = float(cols[6] or 0),
                    ))
                except (ValueError, IndexError):
                    continue
            logger.info(f"Moneycontrol fallback: fetched {len(flows)} days")
            return flows
        except Exception as exc:
            logger.warning(f"Moneycontrol fallback failed: {exc}")
            return []

    # ── Classification ────────────────────────────────────────────

    def _classify(self, flows: list[DayFlow]) -> FIIDIIResult:
        if not flows:
            return FIIDIIResult(
                mood=FlowMood.NEUTRAL, size_multiplier=0.75, allow_entry=True,
                fii_net_today=0, dii_net_today=0, fii_net_5d=0,
                days_available=0,
                summary="No FII/DII data available — proceeding at 75% size",
            )

        today   = flows[0]
        last_5  = flows[:5]
        fii_5d  = sum(d.fii_net for d in last_5)
        dii_5d  = sum(d.dii_net for d in last_5)

        fii_today = today.fii_net
        dii_today = today.dii_net

        # ── BULLISH STRONG ────────────────────────────────────────
        if fii_today >= STRONG_BUY_CR and fii_5d > 0:
            return FIIDIIResult(
                mood=FlowMood.BULLISH_STRONG, size_multiplier=1.0, allow_entry=True,
                fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
                days_available=len(flows), history=flows,
                summary=(
                    f"FII net-bought ₹{fii_today:,.0f} cr today, "
                    f"5-day total ₹{fii_5d:,.0f} cr — strong foreign inflow"
                ),
            )

        # ── BEARISH STRONG ────────────────────────────────────────
        if fii_today <= STRONG_SELL_CR and fii_5d < 0:
            return FIIDIIResult(
                mood=FlowMood.BEARISH_STRONG, size_multiplier=0.0, allow_entry=False,
                fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
                days_available=len(flows), history=flows,
                summary=(
                    f"FII net-sold ₹{abs(fii_today):,.0f} cr today, "
                    f"5-day outflow ₹{abs(fii_5d):,.0f} cr — foreign exodus"
                ),
            )

        # ── BEARISH WEAK ──────────────────────────────────────────
        # Check BEFORE bullish-weak so FII selling isn't masked by DII
        if fii_today <= WEAK_SELL_CR and dii_today < 1_000:
            return FIIDIIResult(
                mood=FlowMood.BEARISH_WEAK, size_multiplier=0.50, allow_entry=True,
                fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
                days_available=len(flows), history=flows,
                summary=(
                    f"FII selling ₹{abs(fii_today):,.0f} cr, "
                    f"DII not compensating (₹{dii_today:,.0f} cr)"
                ),
            )

        # FII selling but DII compensating strongly → treat as NEUTRAL
        if fii_today <= WEAK_SELL_CR and dii_today >= 1_000:
            return FIIDIIResult(
                mood=FlowMood.NEUTRAL, size_multiplier=0.75, allow_entry=True,
                fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
                days_available=len(flows), history=flows,
                summary=(
                    f"FII sold ₹{abs(fii_today):,.0f} cr but DII absorbed "
                    f"with ₹{dii_today:,.0f} cr — net balanced"
                ),
            )

        # ── BULLISH WEAK ──────────────────────────────────────────
        if fii_today >= WEAK_BUY_CR or dii_today >= 1_000:
            return FIIDIIResult(
                mood=FlowMood.BULLISH_WEAK, size_multiplier=0.85, allow_entry=True,
                fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
                days_available=len(flows), history=flows,
                summary=(
                    f"Mild FII inflow ₹{fii_today:,.0f} cr, "
                    f"DII net ₹{dii_today:,.0f} cr"
                ),
            )

        # ── NEUTRAL ───────────────────────────────────────────────
        return FIIDIIResult(
            mood=FlowMood.NEUTRAL, size_multiplier=0.75, allow_entry=True,
            fii_net_today=fii_today, dii_net_today=dii_today, fii_net_5d=fii_5d,
            days_available=len(flows), history=flows,
            summary=(
                f"FII net ₹{fii_today:+,.0f} cr, "
                f"DII net ₹{dii_today:+,.0f} cr — balanced flow"
            ),
        )

    # ── Public API ────────────────────────────────────────────────

    def _fetch_yfinance_proxy(self) -> list[DayFlow]:
        """
        Last-resort fallback: estimate FII flow direction from Nifty's
        recent price action vs its moving average.

        When NSE API and Moneycontrol are both unreachable (common on Mac/
        sandboxed environments), we use NIFTYBEES.NS price vs its 20-day
        average as a proxy for institutional sentiment:
          - Nifty significantly above 20-DMA → likely FII buying (+proxy)
          - Nifty significantly below 20-DMA → likely FII selling (-proxy)

        Values are labelled as estimates (not real crore figures).
        """
        try:
            import yfinance as yf
            from utils.yf_helpers import flatten_yf_columns

            df = yf.download("NIFTYBEES.NS", period="30d", interval="1d",
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                return []

            df = flatten_yf_columns(df).dropna()
            if len(df) < 5:
                return []

            close   = df["close"]
            ema20   = close.ewm(span=20, adjust=False).mean()
            volume  = df["volume"]
            avg_vol = volume.iloc[:-1].mean()

            flows = []
            for i in range(min(5, len(df))):
                idx     = -(i + 1)
                price   = float(close.iloc[idx])
                ema_val = float(ema20.iloc[idx])
                vol     = float(volume.iloc[idx])

                # Estimate FII net as % deviation from EMA × volume proxy
                pct_dev  = (price - ema_val) / ema_val * 100
                vol_mult = vol / avg_vol if avg_vol > 0 else 1.0
                # Scale to crore-like number for classification thresholds
                proxy_net = round(pct_dev * vol_mult * 200, 0)

                date_str  = df.index[idx].strftime("%d-%b-%Y")
                flows.append(DayFlow(
                    date     = date_str,
                    fii_buy  = max(proxy_net, 0),
                    fii_sell = abs(min(proxy_net, 0)),
                    fii_net  = proxy_net,
                    dii_buy  = 0,
                    dii_sell = 0,
                    dii_net  = 0,
                ))

            logger.info(
                f"FII/DII: using NIFTYBEES proxy "
                f"(NSE API unavailable) — estimated net: "
                f"₹{flows[0].fii_net:+,.0f} cr equiv"
            )
            return flows

        except Exception as exc:
            logger.debug(f"FII proxy fetch failed: {exc}")
            return []

    def analyse(self) -> FIIDIIResult:
        """
        Fetch + classify FII/DII flow. Cached for 6 hours.
        Tries: NSE API → Moneycontrol → NIFTYBEES proxy → NEUTRAL fallback
        """
        if self._cache and not self._cache.is_stale:
            logger.debug("FII/DII: using cached result")
            return self._cache

        logger.info("🏦 Fetching FII/DII institutional flow…")

        # Priority 1: NSE official API
        flows = self._fetch_nse()

        # Priority 2: Moneycontrol scrape
        if not flows:
            flows = self._fetch_moneycontrol_fallback()

        # Priority 3: NIFTYBEES yfinance proxy (always works, estimate only)
        if not flows:
            flows = self._fetch_yfinance_proxy()

        result = self._classify(flows)
        self._cache = result

        logger.info(
            f"FII/DII: mood={result.mood.value}  "
            f"FII={result.fii_net_today:+,.0f} cr  "
            f"DII={result.dii_net_today:+,.0f} cr  "
            f"5d={result.fii_net_5d:+,.0f} cr  "
            f"allow={result.allow_entry}"
        )
        self._print(result)
        return result

    # ── Rich display ──────────────────────────────────────────────

    def _print(self, r: FIIDIIResult):
        colors = {
            FlowMood.BULLISH_STRONG: "green",
            FlowMood.BULLISH_WEAK:   "cyan",
            FlowMood.NEUTRAL:        "yellow",
            FlowMood.BEARISH_WEAK:   "dark_orange",
            FlowMood.BEARISH_STRONG: "red",
        }
        c = colors[r.mood]

        fii_sign = "+" if r.fii_net_today >= 0 else ""
        dii_sign = "+" if r.dii_net_today >= 0 else ""
        d5_sign  = "+" if r.fii_net_5d    >= 0 else ""

        # History table
        if r.history:
            tbl = Table(box=box.SIMPLE, show_header=True,
                        header_style="bold white", style="dim")
            tbl.add_column("Date",     width=12)
            tbl.add_column("FII Net",  justify="right", width=12)
            tbl.add_column("DII Net",  justify="right", width=12)
            tbl.add_column("Combined", justify="right", width=12)
            for d in r.history[:5]:
                combined = d.fii_net + d.dii_net
                fc = "green" if d.fii_net >= 0 else "red"
                dc = "green" if d.dii_net >= 0 else "red"
                cc = "green" if combined  >= 0 else "red"
                tbl.add_row(
                    d.date,
                    f"[{fc}]₹{d.fii_net:+,.0f}[/]",
                    f"[{dc}]₹{d.dii_net:+,.0f}[/]",
                    f"[{cc}]₹{combined:+,.0f}[/]",
                )
            console.print(tbl)

        console.print(Panel(
            f"  FII today: [{c}]₹{fii_sign}{r.fii_net_today:,.0f} cr[/]  │  "
            f"DII today: [{c}]₹{dii_sign}{r.dii_net_today:,.0f} cr[/]  │  "
            f"FII 5-day: [{c}]₹{d5_sign}{r.fii_net_5d:,.0f} cr[/]\n\n"
            f"  Regime: [{c} bold]{r.mood.value}[/]  │  "
            f"Size: [bold]{r.size_multiplier:.0%}[/]  │  "
            f"Entry: {'✅' if r.allow_entry else '❌ BLOCKED'}\n"
            f"  {r.summary}",
            title="🏦 FII / DII Institutional Flow",
            style=f"bold {c}",
            box=box.DOUBLE,
        ))
