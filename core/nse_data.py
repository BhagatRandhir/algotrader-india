"""
core/nse_data.py  –  NSE India Free Data Layer

Replaces yfinance entirely. Uses NSE India's public APIs:

  Endpoint 1: Live Quote
    GET /api/quote-equity?symbol=RELIANCE
    → LTP, OHLC, volume, 52w high/low, delivery %

  Endpoint 2: Intraday OHLCV chart data
    GET /api/chart-databyindex?index=RELIANCE&indices=false
    → 1-minute OHLCV bars for today (live, real-time)

  Endpoint 3: Historical data via NSE chart API
    GET /api/historical/cm/equity?symbol=RELIANCE&series=EQ
      &from=2024-01-01&to=2024-01-31&csv=true
    → Daily OHLCV

  Endpoint 4: Market status
    GET /api/marketStatus
    → Whether market is open/closed

All free. No API key. No login.
Session is warmed up with cookies from homepage on first call.

Usage:
    nse = NSEDataClient()
    ltp     = nse.get_ltp("RELIANCE")
    df_5m   = nse.get_intraday_bars("RELIANCE", interval_min=5)
    df_hist = nse.get_historical_bars("RELIANCE", days=30)
"""
from __future__ import annotations

import time
import threading
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────
BASE        = "https://www.nseindia.com"
HEADERS     = {
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://www.nseindia.com/",
    "X-Requested-With":"XMLHttpRequest",
}
COOKIE_REFRESH_SEC  = 300   # refresh cookies every 5 min
REQUEST_TIMEOUT     = 8
RETRY_DELAY_SEC     = 1.5


class NSEDataClient:
    """
    Thread-safe NSE data client with automatic cookie management.
    Single instance should be shared across the bot (singleton pattern).
    """

    def __init__(self):
        self._session:      requests.Session  = requests.Session()
        self._last_cookie:  float             = 0.0
        self._ltp_cache:    dict[str, float]  = {}
        self._warmup()
        self._start_keepalive()

    def _start_keepalive(self):
        """Background thread that keeps NSE session alive automatically.
        Refreshes cookies every 4 minutes — no manual action needed."""
        def _loop():
            while True:
                time.sleep(240)   # every 4 minutes
                try:
                    self._warmup()
                    logger.debug("🔄 NSE session auto-refreshed")
                except Exception as exc:
                    logger.debug(f"NSE keepalive error: {exc}")

        t = threading.Thread(target=_loop, daemon=True, name="nse-keepalive")
        t.start()
        logger.debug("NSE keepalive thread started (auto-refresh every 4 min)")

    # ── Session management ────────────────────────────────────────

    def _warmup(self):
        """
        Full browser-like NSE session warmup.
        NSE requires: homepage → getquote page → then API calls work.
        """
        try:
            self._session.headers.update(HEADERS)

            # Step 1: Hit homepage (sets initial cookies)
            r1 = self._session.get(f"{BASE}/", timeout=REQUEST_TIMEOUT)
            time.sleep(0.5)

            # Step 2: Hit a stock page (NSE checks Referer chain)
            r2 = self._session.get(
                f"{BASE}/get-quotes/equity?symbol=RELIANCE",
                timeout=REQUEST_TIMEOUT,
            )
            time.sleep(0.3)

            if r1.status_code == 200:
                self._last_cookie = time.time()
                logger.info(
                    f"✅ NSE session ready  "
                    f"cookies={len(self._session.cookies)}"
                )
            else:
                logger.warning(f"NSE warmup homepage: {r1.status_code}")
        except Exception as exc:
            logger.warning(f"NSE warmup error: {exc}")

    def _ensure_fresh(self):
        """Re-warm cookies if stale."""
        if time.time() - self._last_cookie > COOKIE_REFRESH_SEC:
            logger.debug("NSE cookies stale — refreshing…")
            self._warmup()

    def _get(self, url: str, retries: int = 3) -> Optional[dict]:
        """GET with retry + cookie refresh on 401/403."""
        self._ensure_fresh()
        for attempt in range(1, retries + 1):
            try:
                r = self._session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"Referer": "https://www.nseindia.com/"},
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (401, 403):
                    logger.debug(f"NSE {r.status_code} — refreshing cookies (attempt {attempt})")
                    self._warmup()
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    logger.warning(f"NSE GET {url} → {r.status_code}")
                    time.sleep(RETRY_DELAY_SEC)
            except requests.exceptions.Timeout:
                logger.warning(f"NSE timeout (attempt {attempt})")
                time.sleep(RETRY_DELAY_SEC)
            except Exception as exc:
                logger.warning(f"NSE error: {exc}")
                time.sleep(RETRY_DELAY_SEC)
        return None

    # ── Live Quote ────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> Optional[float]:
        """
        Fetch latest traded price from NSE quote API.
        Returns None if unavailable.
        """
        url  = f"{BASE}/api/quote-equity?symbol={symbol.upper()}"
        data = self._get(url)
        if not data:
            return self._ltp_cache.get(symbol)

        try:
            price = (
                data.get("priceInfo", {}).get("lastPrice")
                or data.get("priceInfo", {}).get("close")
            )
            if price:
                price = float(price)
                self._ltp_cache[symbol] = price
                return round(price, 2)
        except Exception as exc:
            logger.debug(f"LTP parse error {symbol}: {exc}")

        return self._ltp_cache.get(symbol)

    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Full quote: LTP, OHLC, volume, change%, 52w high/low.
        Returns flat dict or None.
        """
        url  = f"{BASE}/api/quote-equity?symbol={symbol.upper()}"
        data = self._get(url)
        if not data:
            return None
        try:
            pi = data.get("priceInfo", {})
            ih = data.get("industryInfo", {})
            return {
                "symbol":       symbol,
                "ltp":          float(pi.get("lastPrice",  0)),
                "open":         float(pi.get("open",       0)),
                "high":         float(pi.get("intraDayHighLow", {}).get("max", 0)),
                "low":          float(pi.get("intraDayHighLow", {}).get("min", 0)),
                "prev_close":   float(pi.get("previousClose", 0)),
                "change":       float(pi.get("change",     0)),
                "change_pct":   float(pi.get("pChange",    0)),
                "volume":       int(data.get("marketDeptOrderBook", {})
                                    .get("tradeInfo", {}).get("totalTradedVolume", 0)),
                "week52_high":  float(pi.get("weekHighLow", {}).get("max", 0)),
                "week52_low":   float(pi.get("weekHighLow", {}).get("min", 0)),
                "sector":       ih.get("industry", ""),
            }
        except Exception as exc:
            logger.debug(f"Quote parse error {symbol}: {exc}")
            return None

    # ── Intraday bars ─────────────────────────────────────────────

    def get_intraday_bars(
        self,
        symbol:       str,
        interval_min: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch today's 1-min bars from NSE chart API, then resample
        to desired interval (5, 15, 30, 60 minutes).

        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex in IST
        """
        url  = (f"{BASE}/api/chart-databyindex"
                f"?index={symbol.upper()}&indices=false")
        data = self._get(url)

        if not data:
            return pd.DataFrame()

        try:
            # NSE returns: {"grapthData": [[timestamp_ms, close], ...], ...}
            # or {"grapthData": [[ts, open, high, low, close, volume], ...]}
            graph = data.get("grapthData") or data.get("graphData") or []
            if not graph:
                logger.debug(f"No chart data for {symbol}")
                return pd.DataFrame()

            rows = []
            for bar in graph:
                if len(bar) >= 5:
                    ts = pd.to_datetime(bar[0], unit="ms", utc=True).tz_convert("Asia/Kolkata")
                    rows.append({
                        "datetime": ts.tz_localize(None),
                        "open":     float(bar[1]),
                        "high":     float(bar[2]),
                        "low":      float(bar[3]),
                        "close":    float(bar[4]),
                        "volume":   float(bar[5]) if len(bar) > 5 else 0.0,
                    })
                elif len(bar) == 2:
                    # Only close price — create OHLCV from close
                    ts = pd.to_datetime(bar[0], unit="ms", utc=True).tz_convert("Asia/Kolkata")
                    px = float(bar[1])
                    rows.append({
                        "datetime": ts.tz_localize(None),
                        "open": px, "high": px, "low": px,
                        "close": px, "volume": 0.0,
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).set_index("datetime").sort_index()

            # Resample to desired interval
            if interval_min > 1:
                rule = f"{interval_min}min"
                df = df.resample(rule).agg({
                    "open":   "first",
                    "high":   "max",
                    "low":    "min",
                    "close":  "last",
                    "volume": "sum",
                }).dropna(subset=["close"])

            return df[["open", "high", "low", "close", "volume"]]

        except Exception as exc:
            logger.warning(f"Intraday parse error {symbol}: {exc}")
            return pd.DataFrame()

    # ── Historical bars ───────────────────────────────────────────

    def get_historical_bars(
        self,
        symbol: str,
        days:   int = 30,
        series: str = "EQ",
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV from NSE historical data API.
        Used for computing multi-day indicators (EMA50, ADX, etc.)
        """
        end   = datetime.now()
        start = end - timedelta(days=days + 10)   # buffer for weekends

        url = (
            f"{BASE}/api/historical/cm/equity"
            f"?symbol={symbol.upper()}&series=[%22{series}%22]"
            f"&from={start.strftime('%d-%m-%Y')}"
            f"&to={end.strftime('%d-%m-%Y')}&csv=true"
        )
        try:
            self._ensure_fresh()
            r = self._session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                logger.debug(f"Historical {symbol}: {r.status_code}")
                return pd.DataFrame()

            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # NSE CSV columns: Date, series, OPEN, HIGH, LOW, PREV. CLOSE, LTP,
            #                  CLOSE, vwap, 52W H, 52W L, VOLUME, VALUE, No of trades
            col_map = {
                "date": "date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            }
            # Find actual column names (NSE changes them sometimes)
            rename = {}
            for target, key in col_map.items():
                match = next((c for c in df.columns if key in c.lower()), None)
                if match:
                    rename[match] = target

            df = df.rename(columns=rename)
            needed = [c for c in ["date","open","high","low","close","volume"] if c in df.columns]
            df = df[needed].copy()

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], dayfirst=True)
                df = df.set_index("date").sort_index()

            for col in ["open","high","low","close","volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",",""), errors="coerce")

            return df.dropna(subset=["close"]).tail(days)

        except Exception as exc:
            logger.warning(f"Historical parse error {symbol}: {exc}")
            return pd.DataFrame()

    # ── Combined bars (intraday + historical) ─────────────────────

    def get_bars(
        self,
        symbol:      str,
        interval:    str = "5m",
        period:      str = "5d",
    ) -> pd.DataFrame:
        """
        Main method called by paper_broker.
        Maps interval/period strings to NSE API calls.

          interval: "1m","5m","15m","60m"
          period:   "1d","5d","10d","1mo","3mo"

        For intraday intervals → get_intraday_bars (today's data)
        For daily-level needs  → get_historical_bars (daily OHLCV)

        To get enough bars for indicators, combines:
          - Today's intraday bars
          - Yesterday's intraday bars (fetched from historical)
        """
        interval_min = int("".join(filter(str.isdigit, interval)) or "5")
        days_map     = {"1d":2,"5d":5,"10d":10,"1mo":30,"3mo":90}
        days         = days_map.get(period, 5)

        if interval_min >= 1440 or period in ("1mo","3mo"):
            # Daily bars
            return self.get_historical_bars(symbol, days=days)

        # Intraday: get today's bars
        df_today = self.get_intraday_bars(symbol, interval_min=interval_min)

        # For periods > 1d, also fetch historical to extend the series
        if days > 1:
            df_hist = self.get_historical_bars(symbol, days=days + 5)
            if not df_hist.empty:
                # Expand daily bars into intraday-like bars for indicator warmup
                rows = []
                for ts, row in df_hist.iterrows():
                    # Create 4 synthetic 5m bars from daily OHLCV
                    for sub_px in [row["open"], row["high"], row["low"], row["close"]]:
                        rows.append({
                            "open": row["open"], "high": row["high"],
                            "low": row["low"],   "close": sub_px,
                            "volume": row["volume"] / 4,
                        })
                df_warmup = pd.DataFrame(rows,
                    index=pd.date_range(
                        end=datetime.now().replace(hour=9, minute=15),
                        periods=len(rows),
                        freq=f"{interval_min}min",
                    )
                )
                df_today = pd.concat([df_warmup, df_today]).sort_index()

        return df_today if not df_today.empty else pd.DataFrame()

    # ── Market status ─────────────────────────────────────────────

    def is_market_open(self) -> bool:
        url  = f"{BASE}/api/marketStatus"
        data = self._get(url)
        if not data:
            return False
        try:
            markets = data.get("marketState", [])
            for m in markets:
                if "nse" in m.get("market", "").lower():
                    return m.get("marketStatus", "").lower() == "open"
        except Exception:
            pass
        return False

    # ── Screener data ─────────────────────────────────────────────

    def get_gainers(self, index: str = "NIFTY") -> list[dict]:
        """Top gainers for the day — useful for screener."""
        url  = f"{BASE}/api/live-analysis-variations?index=gainers&type={index}"
        data = self._get(url)
        if not data:
            return []
        try:
            return [
                {
                    "symbol":     d["symbol"],
                    "ltp":        float(d.get("ltp",    0)),
                    "change_pct": float(d.get("perChange", 0)),
                    "volume":     int(d.get("tradedVolume", 0)),
                }
                for d in data.get("NIFTY", {}).get("data", [])
            ]
        except Exception as exc:
            logger.debug(f"Gainers parse: {exc}")
            return []

    def get_most_active(self) -> list[dict]:
        """Most active stocks by volume — good momentum candidates."""
        url  = f"{BASE}/api/live-analysis-variations?index=mostactive"
        data = self._get(url)
        if not data:
            return []
        try:
            return [
                {
                    "symbol":     d["symbol"],
                    "ltp":        float(d.get("ltp",    0)),
                    "change_pct": float(d.get("perChange", 0)),
                    "volume":     int(d.get("tradedVolume", 0)),
                }
                for d in data.get("data", [])
            ]
        except Exception as exc:
            logger.debug(f"Most active parse: {exc}")
            return []

    def get_nifty50_quotes(self) -> list[dict]:
        """All Nifty50 stocks with live quotes — for breadth calculation."""
        url  = f"{BASE}/api/equity-stockIndices?index=NIFTY%2050"
        data = self._get(url)
        if not data:
            return []
        try:
            return [
                {
                    "symbol":     d["symbol"],
                    "ltp":        float(d.get("lastPrice",  0)),
                    "change_pct": float(d.get("pChange",    0)),
                    "open":       float(d.get("open",       0)),
                    "high":       float(d.get("dayHigh",    0)),
                    "low":        float(d.get("dayLow",     0)),
                    "volume":     int(d.get("totalTradedVolume", 0)),
                }
                for d in data.get("data", [])
                if d.get("symbol") != "NIFTY 50"   # skip index row
            ]
        except Exception as exc:
            logger.debug(f"Nifty50 parse: {exc}")
            return []


# ── Module-level singleton ─────────────────────────────────────────
_client: Optional[NSEDataClient] = None

def get_client() -> NSEDataClient:
    """Get or create the shared NSE client instance."""
    global _client
    if _client is None:
        _client = NSEDataClient()
    return _client
