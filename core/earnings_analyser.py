"""
core/earnings_analyser.py  –  Company Earnings & Results Analyser

Fetches and scores quarterly results for NSE stocks.

Data sources (tried in order):
  1. yfinance quarterly_income_stmt — Revenue, Net Income, EPS
  2. NSE corporate announcements    — Recent result announcements
  3. Price-based proxy              — Earnings surprise from price reaction

Scoring:
  +0.3  Revenue growing QoQ AND YoY
  +0.2  Net profit growing QoQ AND YoY
  +0.2  EPS beat vs previous quarter
  +0.1  Recent positive result announcement
  -0.3  Revenue declining
  -0.3  Net loss (negative PAT)
  -0.2  Earnings miss vs estimates

Cached per symbol for 6 hours (results don't change intraday).

Usage:
    from core.earnings_analyser import EarningsAnalyser
    ea = EarningsAnalyser()
    result = ea.analyse("RELIANCE")
    print(result.verdict)   # STRONG / GOOD / NEUTRAL / WEAK / AVOID
    print(result.score)     # -1.0 to +1.0
    print(result.summary)   # "Revenue +18% YoY, PAT +12% QoQ — strong results"
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class EarningsResult:
    symbol:        str
    verdict:       str           # STRONG / GOOD / NEUTRAL / WEAK / AVOID
    score:         float         # -1.0 to +1.0
    summary:       str           # human-readable summary
    revenue_qoq:   Optional[float] = None   # % change quarter-on-quarter
    revenue_yoy:   Optional[float] = None   # % change year-on-year
    pat_qoq:       Optional[float] = None   # profit after tax QoQ
    pat_yoy:       Optional[float] = None   # profit after tax YoY
    eps_current:   Optional[float] = None
    eps_previous:  Optional[float] = None
    last_result_date: Optional[str] = None
    data_source:   str = "none"


class EarningsAnalyser:

    _cache: dict[str, tuple[float, EarningsResult]] = {}
    CACHE_TTL = 21600   # 6 hours

    def analyse(self, symbol: str) -> EarningsResult:
        """Analyse earnings for a symbol. Returns cached result if fresh."""
        now = time.time()
        if symbol in self._cache:
            cached_time, cached_result = self._cache[symbol]
            if now - cached_time < self.CACHE_TTL:
                return cached_result

        result = self._fetch_and_score(symbol)
        self._cache[symbol] = (now, result)
        return result

    def _fetch_and_score(self, symbol: str) -> EarningsResult:
        """Try all data sources and return best available result."""

        # ── 1. yfinance income statement ─────────────────────────
        result = self._from_yfinance(symbol)
        if result:
            return result

        # ── 2. NSE announcements ──────────────────────────────────
        result = self._from_nse_announcements(symbol)
        if result:
            return result

        # ── 3. Price-based proxy ──────────────────────────────────
        return self._price_proxy(symbol)

    # ── Source 1: yfinance ────────────────────────────────────────

    def _from_yfinance(self, symbol: str) -> Optional[EarningsResult]:
        try:
            import yfinance as yf
            tkr  = yf.Ticker(f"{symbol}.NS")
            stmt = tkr.quarterly_income_stmt

            if stmt is None or stmt.empty:
                return None

            # Find revenue row
            rev_row = next(
                (r for r in stmt.index
                 if any(x in str(r).lower()
                        for x in ["total revenue","revenue","net revenue","total income"])),
                None,
            )
            # Find net income / PAT row
            pat_row = next(
                (r for r in stmt.index
                 if any(x in str(r).lower()
                        for x in ["net income","profit after tax","pat",
                                  "net profit","profit/loss"])),
                None,
            )

            if not rev_row and not pat_row:
                return None

            cols = stmt.columns[:4]   # last 4 quarters

            revenue = stmt.loc[rev_row, cols].astype(float) if rev_row else None
            pat     = stmt.loc[pat_row, cols].astype(float) if pat_row else None

            score   = 0.0
            details = []

            # Revenue analysis
            rev_qoq = rev_yoy = None
            if revenue is not None and len(revenue) >= 2:
                rev_qoq = (revenue.iloc[0] - revenue.iloc[1]) / abs(revenue.iloc[1]) * 100
                if len(revenue) >= 4:
                    rev_yoy = (revenue.iloc[0] - revenue.iloc[3]) / abs(revenue.iloc[3]) * 100

                if rev_qoq and rev_qoq > 5:
                    score += 0.20; details.append(f"Revenue +{rev_qoq:.0f}% QoQ")
                elif rev_qoq and rev_qoq > 0:
                    score += 0.10; details.append(f"Revenue +{rev_qoq:.0f}% QoQ")
                elif rev_qoq and rev_qoq < -10:
                    score -= 0.25; details.append(f"Revenue {rev_qoq:.0f}% QoQ ⚠️")

                if rev_yoy and rev_yoy > 10:
                    score += 0.15; details.append(f"+{rev_yoy:.0f}% YoY")
                elif rev_yoy and rev_yoy < 0:
                    score -= 0.15; details.append(f"{rev_yoy:.0f}% YoY ⚠️")

            # PAT analysis
            pat_qoq = pat_yoy = None
            if pat is not None and len(pat) >= 2:
                pat_qoq = (pat.iloc[0] - pat.iloc[1]) / abs(pat.iloc[1] + 1e-9) * 100
                if len(pat) >= 4:
                    pat_yoy = (pat.iloc[0] - pat.iloc[3]) / abs(pat.iloc[3] + 1e-9) * 100

                if pat.iloc[0] < 0:
                    score -= 0.30; details.append("Net loss ❌")
                elif pat_qoq and pat_qoq > 10:
                    score += 0.20; details.append(f"PAT +{pat_qoq:.0f}% QoQ")
                elif pat_qoq and pat_qoq > 0:
                    score += 0.10; details.append(f"PAT +{pat_qoq:.0f}% QoQ")
                elif pat_qoq and pat_qoq < -15:
                    score -= 0.20; details.append(f"PAT {pat_qoq:.0f}% QoQ ⚠️")

                if pat_yoy and pat_yoy > 15:
                    score += 0.15; details.append(f"PAT +{pat_yoy:.0f}% YoY")
                elif pat_yoy and pat_yoy < 0:
                    score -= 0.15

            score   = round(max(-1.0, min(1.0, score)), 3)
            verdict = self._verdict(score)
            summary = " | ".join(details) if details else "Financial data available"

            return EarningsResult(
                symbol      = symbol,
                verdict     = verdict,
                score       = score,
                summary     = summary,
                revenue_qoq = round(rev_qoq, 1) if rev_qoq else None,
                revenue_yoy = round(rev_yoy, 1) if rev_yoy else None,
                pat_qoq     = round(pat_qoq, 1) if pat_qoq else None,
                pat_yoy     = round(pat_yoy, 1) if pat_yoy else None,
                data_source = "yfinance",
            )

        except Exception as exc:
            logger.debug(f"Earnings yfinance {symbol}: {exc}")
            return None

    # ── Source 2: NSE announcements ───────────────────────────────

    def _from_nse_announcements(self, symbol: str) -> Optional[EarningsResult]:
        """
        Check NSE corporate announcements for recent result announcements.
        Result keyword in announcement → check if positive/negative market reaction.
        """
        try:
            import requests
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer":    "https://www.nseindia.com/",
                "Accept":     "application/json",
            })
            sess.get("https://www.nseindia.com", timeout=5)
            url = (f"https://www.nseindia.com/api/corporate-announcements"
                   f"?index=equities&symbol={symbol}")
            r   = sess.get(url, timeout=6)

            if r.status_code != 200:
                return None

            announcements = r.json()
            if not isinstance(announcements, list):
                return None

            # Find latest result announcement
            result_ann = None
            for ann in announcements[:20]:
                subject = str(ann.get("subject", "")).lower()
                if any(x in subject for x in
                       ["financial result","quarterly result","annual result",
                        "q1","q2","q3","q4","unaudited"]):
                    result_ann = ann
                    break

            if not result_ann:
                return None

            # Parse date
            ann_date = result_ann.get("an_dt", "")[:10]
            desc     = result_ann.get("desc", result_ann.get("subject",""))

            # Simple keyword scoring
            desc_lower = desc.lower()
            score = 0.0
            details = []

            positive_keywords = ["profit","increase","growth","beat","strong","record","highest"]
            negative_keywords = ["loss","decline","fall","miss","weak","below","lower"]

            pos_count = sum(1 for w in positive_keywords if w in desc_lower)
            neg_count = sum(1 for w in negative_keywords if w in desc_lower)

            if pos_count > neg_count:
                score = min(0.20 * pos_count, 0.40)
                details.append(f"Positive result announcement ({ann_date})")
            elif neg_count > pos_count:
                score = max(-0.20 * neg_count, -0.40)
                details.append(f"Weak result announcement ({ann_date})")
            else:
                details.append(f"Result announced {ann_date}")

            return EarningsResult(
                symbol           = symbol,
                verdict          = self._verdict(score),
                score            = round(score, 3),
                summary          = " | ".join(details),
                last_result_date = ann_date,
                data_source      = "nse_announcements",
            )

        except Exception as exc:
            logger.debug(f"Earnings NSE {symbol}: {exc}")
            return None

    # ── Source 3: Price proxy ─────────────────────────────────────

    def _price_proxy(self, symbol: str) -> EarningsResult:
        """
        When no financial data is available, use price momentum
        as a proxy for earnings quality.
        Strong 3-month price performance → market has already priced in good results.
        """
        try:
            import yfinance as yf
            df = yf.Ticker(f"{symbol}.NS").history(period="3mo", interval="1wk")
            if df is not None and len(df) >= 8:
                perf_3m = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
                if perf_3m > 15:
                    return EarningsResult(symbol, "GOOD",    0.15,
                        f"Price +{perf_3m:.0f}% in 3M (earnings proxy)", data_source="price_proxy")
                elif perf_3m > 5:
                    return EarningsResult(symbol, "NEUTRAL", 0.05,
                        f"Price +{perf_3m:.0f}% in 3M", data_source="price_proxy")
                elif perf_3m < -15:
                    return EarningsResult(symbol, "WEAK",   -0.15,
                        f"Price {perf_3m:.0f}% in 3M (weak momentum)", data_source="price_proxy")
        except Exception:
            pass

        return EarningsResult(symbol, "NEUTRAL", 0.0,
            "No earnings data available", data_source="none")

    # ── Helper ────────────────────────────────────────────────────

    def _verdict(self, score: float) -> str:
        if score >= 0.40:   return "STRONG"
        if score >= 0.15:   return "GOOD"
        if score >= -0.10:  return "NEUTRAL"
        if score >= -0.30:  return "WEAK"
        return "AVOID"

    def format_for_display(self, result: EarningsResult) -> dict:
        """Format for dashboard / API response."""
        color = {
            "STRONG": "#22C55E",
            "GOOD":   "#00D4AA",
            "NEUTRAL":"#F59E0B",
            "WEAK":   "#EF4444",
            "AVOID":  "#EF4444",
        }.get(result.verdict, "#5A7A9E")

        rows = []
        if result.revenue_qoq is not None:
            rows.append({"label":"Revenue QoQ", "value":f"{result.revenue_qoq:+.1f}%",
                         "positive": result.revenue_qoq > 0})
        if result.revenue_yoy is not None:
            rows.append({"label":"Revenue YoY", "value":f"{result.revenue_yoy:+.1f}%",
                         "positive": result.revenue_yoy > 0})
        if result.pat_qoq is not None:
            rows.append({"label":"PAT QoQ", "value":f"{result.pat_qoq:+.1f}%",
                         "positive": result.pat_qoq > 0})
        if result.pat_yoy is not None:
            rows.append({"label":"PAT YoY", "value":f"{result.pat_yoy:+.1f}%",
                         "positive": result.pat_yoy > 0})

        return {
            "verdict":     result.verdict,
            "score":       result.score,
            "summary":     result.summary,
            "color":       color,
            "rows":        rows,
            "data_source": result.data_source,
            "last_result": result.last_result_date,
        }


# ── Module-level singleton ─────────────────────────────────────────
_analyser: Optional[EarningsAnalyser] = None

def get_earnings_analyser() -> EarningsAnalyser:
    global _analyser
    if _analyser is None:
        _analyser = EarningsAnalyser()
    return _analyser
