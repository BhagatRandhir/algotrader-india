"""
core/smart_signals.py  –  Smart Signal Engine

Provides 4 additional signals used in both screener and entry gate:

  1. NewsSentiment   — Google News RSS + VADER scoring
  2. SectorStrength  — Is the stock's sector outperforming Nifty today?
  3. LinearForecast  — Numpy linear regression: is price trending up?
  4. EarningsGuard   — Avoid stocks with earnings in next 3 days

Each returns a simple result: allow (bool), score (float), reason (str)
All are cached to avoid repeated API calls within the same loop.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ── Lazy imports (only load when needed) ─────────────────────────
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

def _feedparser():
    import feedparser
    return feedparser


@dataclass
class SignalResult:
    allow:  bool
    score:  float    # -1.0 (very bearish) to +1.0 (very bullish)
    reason: str
    detail: str = ""


# ── 1. News Sentiment ─────────────────────────────────────────────

class NewsSentimentSignal:
    """
    Fetches last 5 headlines from Google News RSS for the stock.
    Scores each with VADER. Returns average compound score.
    
    Score:  > 0.05 = positive (allow)
            < -0.05 = negative (block)
            else = neutral (allow with reduced confidence)
    
    Cached per symbol for 30 minutes.
    """
    _cache: dict[str, tuple[float, SignalResult]] = {}
    CACHE_TTL = 1800   # 30 min

    def analyse(self, symbol: str) -> SignalResult:
        # Check cache
        now = time.time()
        if symbol in self._cache:
            cached_time, cached_result = self._cache[symbol]
            if now - cached_time < self.CACHE_TTL:
                return cached_result

        result = self._fetch_and_score(symbol)
        self._cache[symbol] = (now, result)
        return result

    def _fetch_and_score(self, symbol: str) -> SignalResult:
        try:
            fp   = _feedparser()
            vader= _vader()
            url  = (f"https://news.google.com/rss/search?"
                    f"q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en")
            feed = fp.parse(url)

            if not feed.entries:
                # No news = neutral, always allow
                return SignalResult(True, 0.0, "No news — neutral", "neutral")

            scores = []
            headlines = []
            for entry in feed.entries[:5]:
                title = entry.get("title", "")
                vs    = vader.polarity_scores(title)
                scores.append(vs["compound"])
                headlines.append(f"{title[:50]} ({vs['compound']:+.2f})")

            avg = sum(scores) / len(scores)

            if avg > 0.05:
                allow  = True
                reason = f"News positive ({avg:+.2f})"
            elif avg < -0.05:
                allow  = False
                reason = f"News negative ({avg:+.2f}) — avoiding"
            else:
                allow  = True
                reason = f"News neutral ({avg:+.2f})"

            return SignalResult(allow, avg, reason, " | ".join(headlines[:2]))

        except Exception as exc:
            logger.debug(f"News sentiment {symbol}: {exc}")
            # Network blocked or error — never block trade due to news failure
            return SignalResult(True, 0.0, "News unavailable — allowing", "")


# ── 2. Sector Strength ────────────────────────────────────────────

# NSE sector index symbols (yfinance)
SECTOR_MAP = {
    "RELIANCE":"^CNXENERGY",  "ONGC":"^CNXENERGY",   "COALINDIA":"^CNXENERGY",
    "TCS":"^CNXIT",           "INFY":"^CNXIT",        "WIPRO":"^CNXIT",
    "HCLTECH":"^CNXIT",       "TECHM":"^CNXIT",
    "HDFCBANK":"^NSEBANK",    "ICICIBANK":"^NSEBANK", "SBIN":"^NSEBANK",
    "AXISBANK":"^NSEBANK",    "KOTAKBANK":"^NSEBANK", "BANDHANBNK":"^NSEBANK",
    "SUNPHARMA":"^CNXPHARMA", "DRREDDY":"^CNXPHARMA","CIPLA":"^CNXPHARMA",
    "DIVISLAB":"^CNXPHARMA",
    "MARUTI":"^CNXAUTO",      "TATAMOTORS":"^CNXAUTO","BAJAJ-AUTO":"^CNXAUTO",
    "EICHERMOT":"^CNXAUTO",   "HEROMOTOCO":"^CNXAUTO",
    "TITAN":"^CNXFMCG",       "HINDUNILVR":"^CNXFMCG","ITC":"^CNXFMCG",
    "NESTLEIND":"^CNXFMCG",
    "LT":"^CNXINFRA",         "ULTRACEMCO":"^CNXINFRA",
    "BAJFINANCE":"^CNXFIN",   "BAJAJFINSV":"^CNXFIN",
}
DEFAULT_SECTOR = "^CNX500"   # broad market fallback

class SectorStrengthSignal:
    """
    Compares stock's intraday change vs its sector index.
    If stock is outperforming sector → bullish signal.
    Cached per sector for 15 minutes.
    """
    _cache: dict[str, tuple[float, float]] = {}
    CACHE_TTL = 900

    def analyse(self, symbol: str, stock_change_pct: float) -> SignalResult:
        sector_sym = SECTOR_MAP.get(symbol, DEFAULT_SECTOR)
        sector_chg = self._get_sector_change(sector_sym)

        outperform = stock_change_pct - sector_chg
        if outperform > 0.3:
            return SignalResult(
                True, min(outperform / 2, 1.0),
                f"Outperforming sector by {outperform:+.2f}%",
                f"{symbol}={stock_change_pct:+.1f}% vs sector={sector_chg:+.1f}%",
            )
        elif outperform < -0.5:
            return SignalResult(
                False, max(outperform / 2, -1.0),
                f"Underperforming sector by {abs(outperform):.2f}%",
                f"{symbol}={stock_change_pct:+.1f}% vs sector={sector_chg:+.1f}%",
            )
        else:
            return SignalResult(
                True, 0.0,
                f"In line with sector ({outperform:+.2f}%)",
                f"sector={sector_chg:+.1f}%",
            )

    def _get_sector_change(self, sector_sym: str) -> float:
        now = time.time()
        if sector_sym in self._cache:
            cached_time, cached_val = self._cache[sector_sym]
            if now - cached_time < self.CACHE_TTL:
                return cached_val
        try:
            import yfinance as yf
            df  = yf.Ticker(sector_sym).history(period="2d", interval="1d")
            if len(df) >= 2:
                chg = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
                self._cache[sector_sym] = (now, round(float(chg), 2))
                return round(float(chg), 2)
        except Exception as exc:
            logger.debug(f"Sector {sector_sym}: {exc}")
        return 0.0


# ── 3. Linear Regression Forecast ────────────────────────────────

class LinearForecastSignal:
    """
    Fits a linear regression on last N bars of close price.
    Forecasts next 3 bars direction.
    
    slope > 0  = uptrend  → allow
    slope <= 0 = downtrend → reduce confidence / block
    
    R² score tells us how clean the trend is.
    """

    def analyse(self, df: pd.DataFrame, symbol: str,
                lookback: int = 20) -> SignalResult:
        if df is None or len(df) < lookback + 5:
            return SignalResult(True, 0.0, "Insufficient data for forecast", "")

        try:
            close  = df["close"].iloc[-lookback:].values
            x      = np.arange(len(close))
            slope, intercept = np.polyfit(x, close, 1)

            # Normalise slope as % per bar
            slope_pct = slope / (close[0] + 1e-9) * 100

            # R² — how well does the line fit?
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - close.mean()) ** 2)
            r2     = 1 - ss_res / (ss_tot + 1e-9)

            # Forecast next 3 bars
            forecast = [slope * (len(close) + i) + intercept for i in range(1, 4)]
            direction = "UP" if slope > 0 else "DOWN"
            confidence= min(abs(slope_pct) * r2 * 10, 1.0)

            if slope > 0 and r2 > 0.3:
                allow  = True
                score  = confidence
                reason = f"Forecast UP (slope={slope_pct:+.3f}%/bar R²={r2:.2f})"
            elif slope <= 0 and r2 > 0.4:
                allow  = False
                score  = -confidence
                reason = f"Forecast DOWN (slope={slope_pct:+.3f}%/bar R²={r2:.2f})"
            else:
                allow  = True
                score  = 0.0
                reason = f"Forecast unclear (R²={r2:.2f} too low)"

            detail = f"Next 3 bars: ₹{forecast[0]:.1f} → ₹{forecast[2]:.1f}"
            return SignalResult(allow, score, reason, detail)

        except Exception as exc:
            logger.debug(f"Forecast {symbol}: {exc}")
            return SignalResult(True, 0.0, "Forecast error — neutral", "")


# ── 4. Earnings Guard ─────────────────────────────────────────────

class EarningsGuardSignal:
    """
    Checks if the stock has earnings announcement within next 3 days.
    If yes → avoid (high volatility, unpredictable direction).
    Uses yfinance calendar data.
    Cached per symbol for 6 hours.
    """
    _cache: dict[str, tuple[float, SignalResult]] = {}
    CACHE_TTL = 21600   # 6 hours

    def analyse(self, symbol: str) -> SignalResult:
        now = time.time()
        if symbol in self._cache:
            cached_time, cached_result = self._cache[symbol]
            if now - cached_time < self.CACHE_TTL:
                return cached_result

        result = self._check_earnings(symbol)
        self._cache[symbol] = (now, result)
        return result

    def _check_earnings(self, symbol: str) -> SignalResult:
        try:
            import yfinance as yf
            cal = yf.Ticker(f"{symbol}.NS").calendar
            if cal is None or cal.empty:
                return SignalResult(True, 0.0, "No earnings date found", "")

            # Calendar has 'Earnings Date' column
            if "Earnings Date" in cal.index:
                earn_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])
                days_away = (earn_date - pd.Timestamp.now()).days
                if 0 <= days_away <= 3:
                    return SignalResult(
                        False, -0.5,
                        f"Earnings in {days_away} day(s) — avoiding",
                        f"Earnings: {earn_date.strftime('%d %b %Y')}",
                    )
                else:
                    return SignalResult(
                        True, 0.1,
                        f"No earnings soon ({days_away}d away)",
                        f"Next earnings: {earn_date.strftime('%d %b %Y')}",
                    )
        except Exception as exc:
            logger.debug(f"Earnings {symbol}: {exc}")
        return SignalResult(True, 0.0, "Earnings check unavailable", "")


# ── Module-level singletons ───────────────────────────────────────
_news     = NewsSentimentSignal()
_sector   = SectorStrengthSignal()
_forecast = LinearForecastSignal()
_earnings = EarningsGuardSignal()


def run_smart_signals(
    symbol:      str,
    df:          pd.DataFrame,
    day_chg_pct: float = 0.0,
) -> dict:
    """
    Run all 4 smart signals for a symbol.
    Returns dict with results for each signal + combined score.

    Usage:
        signals = run_smart_signals("RELIANCE", df, day_chg_pct=1.2)
        if signals["allow"]:
            # place order
    """
    news     = _news.analyse(symbol)
    sector   = _sector.analyse(symbol, day_chg_pct)
    forecast = _forecast.analyse(df, symbol)
    earnings = _earnings.analyse(symbol)

    # Combined score: weighted average
    combined = (
        news.score     * 0.30 +
        sector.score   * 0.25 +
        forecast.score * 0.35 +
        earnings.score * 0.10
    )

    # Only hard block on earnings (always block)
    # Soft signals: need 2+ to block (1 bad signal is not enough)
    hard_block  = not earnings.allow
    soft_blocks = sum([not news.allow, not sector.allow, not forecast.allow])
    allow       = not hard_block and soft_blocks <= 1

    return {
        "allow":    allow,
        "score":    round(combined, 3),
        "signals": {
            "news":     {"allow":news.allow,     "score":round(news.score,3),
                         "reason":news.reason,   "detail":news.detail},
            "sector":   {"allow":sector.allow,   "score":round(sector.score,3),
                         "reason":sector.reason, "detail":sector.detail},
            "forecast": {"allow":forecast.allow, "score":round(forecast.score,3),
                         "reason":forecast.reason,"detail":forecast.detail},
            "earnings": {"allow":earnings.allow, "score":round(earnings.score,3),
                         "reason":earnings.reason,"detail":earnings.detail},
        },
        "summary": (
            f"News: {'✅' if news.allow else '❌'} {news.reason} | "
            f"Sector: {'✅' if sector.allow else '❌'} {sector.reason} | "
            f"Forecast: {'✅' if forecast.allow else '❌'} {forecast.reason} | "
            f"Earnings: {'✅' if earnings.allow else '❌'} {earnings.reason}"
        ),
    }
