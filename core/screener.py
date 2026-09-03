"""
core/screener.py  –  Smart Stock Screener (v2)

Scores stocks on 5 dimensions:
  1. Volume surge      — unusual activity vs 20-day avg
  2. Price momentum    — 5/10/20 day returns
  3. Trend strength    — EMA alignment + ADX
  4. RSI zone          — 45–65 sweet spot (not oversold or overbought)
  5. Intraday strength — price above VWAP on daily proxy

Only F&O-eligible / high-liquidity stocks get through the size filter.
Hourly refresh keeps the watchlist fresh during market hours.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich import box

from utils.nse_universe import NSE_UNIVERSE
from core.smart_signals import run_smart_signals
from utils.yf_helpers import flatten_yf_columns

KNOWN_BAD_SYMBOLS = {
    "MISHTANN", "TATAMOTORS", "LTIM", "HEXAWARE",
}

console      = Console()
WATCHLIST_FILE = Path("watchlist.json")


@dataclass
class ScreenerConfig:
    max_picks:               int   = 12
    volume_surge_multiplier: float = 1.8   # lowered: 2.0 was missing good stocks
    momentum_5d_min_pct:     float = 2.0   # lowered from 3 — catch earlier
    momentum_10d_min_pct:    float = 4.0
    momentum_20d_min_pct:    float = 7.0
    rsi_min:                 float = 45.0  # NEW: sweet spot filter
    rsi_max:                 float = 70.0  # NEW: avoid overbought entries
    adx_min:                 float = 20.0  # NEW: only trending stocks
    min_avg_volume:          int   = 300_000  # lowered to catch mid-caps
    min_price_inr:           float = 50.0
    lookback_period:         str   = "3mo"
    fetch_delay_sec:         float = 0.8
    batch_size:              int   = 10


@dataclass
class StockScore:
    symbol:          str
    last_price:      float
    score:           float = 0.0
    volume_signal:   bool  = False
    momentum_signal: bool  = False
    trend_signal:    bool  = False
    rsi_signal:      bool  = False
    volume_ratio:    float = 0.0
    momentum_5d:     float = 0.0
    momentum_10d:    float = 0.0
    rsi:             float = 50.0
    adx:             float = 0.0
    avg_volume:      int   = 0
    reasons:         list  = field(default_factory=list)


class StockScreener:

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.cfg = config or ScreenerConfig()

    # ── Indicators ────────────────────────────────────────────────

    def _rsi(self, close: pd.Series, p: int = 14) -> float:
        if len(close) < p + 1:
            return 50.0
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
        rs    = gain.iloc[-1] / (loss.iloc[-1] + 1e-10)
        return round(float(100 - 100 / (1 + rs)), 2)

    def _adx(self, df: pd.DataFrame, p: int = 14) -> float:
        """Average Directional Index — measures trend strength."""
        if len(df) < p + 5:
            return 0.0
        high, low, close = df["high"], df["low"], df["close"]
        tr  = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        dm_plus  = ((high.diff() >  low.diff().abs()) & (high.diff() > 0)) * high.diff()
        dm_minus = ((low.diff().abs() > high.diff()) & (low.diff() < 0)) * low.diff().abs()
        atr  = tr.ewm(span=p, adjust=False).mean()
        dip  = dm_plus.ewm(span=p,  adjust=False).mean()  / (atr + 1e-9) * 100
        dim  = dm_minus.ewm(span=p, adjust=False).mean() / (atr + 1e-9) * 100
        dx   = (dip - dim).abs() / (dip + dim + 1e-9) * 100
        adx  = dx.ewm(span=p, adjust=False).mean()
        return round(float(adx.iloc[-1]), 2)

    def _ema_aligned(self, close: pd.Series) -> bool:
        """True if EMA9 > EMA20 > EMA50 (full bull trend stack)."""
        if len(close) < 55:
            return False
        e9  = close.ewm(span=9,  adjust=False).mean().iloc[-1]
        e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        return bool(e9 > e20 > e50)

    # ── Fetch ─────────────────────────────────────────────────────

    def _fetch_batch(self, symbols: list[str], retries: int = 3) -> dict[str, pd.DataFrame]:
        tickers = [f"{s}.NS" for s in symbols]
        for attempt in range(1, retries + 1):
            try:
                raw = yf.download(tickers, period=self.cfg.lookback_period,
                                  interval="1d", auto_adjust=True,
                                  progress=False, threads=False)
                if raw is None or raw.empty:
                    raise ValueError("empty response")

                result = {}
                if isinstance(raw.columns, pd.MultiIndex):
                    level_0 = set(raw.columns.get_level_values(0))
                    tkl = 1 if any(c in level_0 for c in ("Open","High","Low","Close","Volume")) else 0
                    for sym, tkr in zip(symbols, tickers):
                        try:
                            if tkr not in raw.columns.get_level_values(tkl):
                                continue
                            df = raw.xs(tkr, level=tkl, axis=1).dropna()
                            df = flatten_yf_columns(df)
                            if not df.empty and "close" in df.columns:
                                result[sym] = df
                        except Exception:
                            pass
                else:
                    if symbols:
                        df = flatten_yf_columns(raw.dropna())
                        if "close" in df.columns:
                            result[symbols[0]] = df
                if result:
                    return result
                raise ValueError("no symbols parsed")
            except Exception as exc:
                wait = attempt * 2
                logger.warning(f"Batch attempt {attempt}/{retries}: {exc} — retry in {wait}s")
                time.sleep(wait)
        return {}

    # ── Score ─────────────────────────────────────────────────────

    def _score(self, symbol: str, df: pd.DataFrame) -> Optional[StockScore]:
        if df is None or len(df) < 30:
            return None

        close    = df["close"]
        volume   = df["volume"]
        last_px  = float(close.iloc[-1])
        avg_vol  = int(volume.iloc[-20:].mean())

        if avg_vol < self.cfg.min_avg_volume or last_px < self.cfg.min_price_inr:
            return None

        st = StockScore(symbol=symbol, last_price=round(last_px, 2), avg_volume=avg_vol)

        # ── 1. Volume surge ───────────────────────────────────────
        today_vol    = int(volume.iloc[-1])
        vol_ratio    = today_vol / (avg_vol + 1)
        st.volume_ratio = round(vol_ratio, 2)
        if vol_ratio >= self.cfg.volume_surge_multiplier:
            st.volume_signal = True
            st.score += 1.0 + min((vol_ratio - 1.8) * 0.3, 1.5)
            st.reasons.append(f"Vol surge {vol_ratio:.1f}×")

        # ── 2. Price momentum ─────────────────────────────────────
        mom5  = (close.iloc[-1]/close.iloc[-6]  - 1)*100 if len(close) > 5  else 0
        mom10 = (close.iloc[-1]/close.iloc[-11] - 1)*100 if len(close) > 10 else 0
        mom20 = (close.iloc[-1]/close.iloc[-21] - 1)*100 if len(close) > 20 else 0
        st.momentum_5d  = round(mom5,  2)
        st.momentum_10d = round(mom10, 2)

        mom_score = 0.0
        if mom5  >= self.cfg.momentum_5d_min_pct:
            mom_score += 0.8;  st.reasons.append(f"5d +{mom5:.1f}%")
        if mom10 >= self.cfg.momentum_10d_min_pct:
            mom_score += 0.6;  st.reasons.append(f"10d +{mom10:.1f}%")
        if mom20 >= self.cfg.momentum_20d_min_pct:
            mom_score += 0.5;  st.reasons.append(f"20d +{mom20:.1f}%")
        if mom_score > 0:
            st.momentum_signal = True
            st.score += mom_score

        # ── 3. Trend strength (EMA alignment + ADX) ───────────────
        aligned = self._ema_aligned(close)
        adx     = self._adx(df)
        st.adx  = adx
        if aligned and adx >= self.cfg.adx_min:
            st.trend_signal = True
            st.score += 1.0 + min((adx - 20) / 20, 1.0)
            st.reasons.append(f"Trend: EMA✓ ADX={adx:.0f}")
        elif aligned:
            st.score += 0.4
            st.reasons.append("EMA aligned")

        # ── 4. RSI sweet spot ─────────────────────────────────────
        rsi    = self._rsi(close)
        st.rsi = rsi
        if self.cfg.rsi_min <= rsi <= self.cfg.rsi_max:
            st.rsi_signal = True
            # Peak score at RSI=60, falls off toward extremes
            rsi_center  = 60.0
            rsi_bonus   = 1.0 - abs(rsi - rsi_center) / 20.0
            st.score   += max(rsi_bonus, 0.1)
            st.reasons.append(f"RSI={rsi:.0f} (sweet spot)")
        elif rsi < 35:
            # Oversold bounce candidate — lower weight for momentum bot
            st.score += 0.3
            st.reasons.append(f"RSI={rsi:.0f} oversold")

        return st if st.score > 0 else None

    # ── Run ───────────────────────────────────────────────────────

    def run_scored(self, universe: list[str] = NSE_UNIVERSE) -> list:
        """
        Same as run() but returns list[StockScore] instead of list[str].
        Used by WatchlistOptimiser for score-based comparison.
        """
        clean = [s for s in universe if s not in KNOWN_BAD_SYMBOLS]
        candidates: list[StockScore] = []
        batches = [clean[i:i+self.cfg.batch_size]
                   for i in range(0, len(clean), self.cfg.batch_size)]

        console.print(f"\n  [dim]🔍 Scoring {len(clean)} stocks for optimiser…[/]")
        for batch in batches:
            data = self._fetch_batch(batch)
            for sym, df in data.items():
                result = self._score(sym, df)
                if result:
                    candidates.append(result)
            time.sleep(self.cfg.fetch_delay_sec)

        qualified = [c for c in candidates
                     if sum([c.volume_signal, c.momentum_signal,
                             c.trend_signal,  c.rsi_signal]) >= 2]
        if not qualified:
            qualified = candidates

        # Smart signal boost/penalty on top scorers
        top = sorted(qualified, key=lambda x: x.score, reverse=True)[:30]
        logger.info(f"Running smart signals on top {len(top)} candidates…")
        for stock in top:
            try:
                import yfinance as yf
                df = yf.Ticker(f"{stock.symbol}.NS").history(period="5d", interval="5m")
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    smart = run_smart_signals(stock.symbol, df, stock.momentum_5d)
                    # Boost score if smart signals agree
                    if smart["allow"]:
                        stock.score += smart["score"] * 0.5
                        stock.reasons.append(f"Smart✓({smart['score']:+.2f})")
                    else:
                        stock.score *= 0.3   # heavy penalty for bad smart signals
                        stock.reasons.append(f"Smart✗")
            except Exception as exc:
                logger.debug(f"Smart signal {stock.symbol}: {exc}")

        return sorted(top, key=lambda x: x.score, reverse=True)

    def run(self, universe: list[str] = NSE_UNIVERSE) -> list[str]:
        clean = [s for s in universe if s not in KNOWN_BAD_SYMBOLS]
        skipped = set(universe) - set(clean)
        if skipped:
            logger.info(f"Skipping bad symbols: {sorted(skipped)}")

        console.print(
            f"\n  [bold cyan]📡 Smart Screener v2[/]  "
            f"scanning {len(clean)} stocks  "
            f"[dim]{datetime.now().strftime('%H:%M IST')}[/]\n"
        )

        candidates: list[StockScore] = []
        batches = [clean[i:i+self.cfg.batch_size]
                   for i in range(0, len(clean), self.cfg.batch_size)]

        with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                      BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      console=console) as prog:
            task = prog.add_task("Screening…", total=len(batches))
            for batch in batches:
                data = self._fetch_batch(batch)
                for sym, df in data.items():
                    result = self._score(sym, df)
                    if result:
                        candidates.append(result)
                time.sleep(self.cfg.fetch_delay_sec)
                prog.advance(task)

        # ── Rank: must have at least 2 signals to qualify ─────────
        qualified = [c for c in candidates
                     if sum([c.volume_signal, c.momentum_signal,
                             c.trend_signal,  c.rsi_signal]) >= 2]
        if not qualified:
            logger.warning("No stocks passed 2-signal filter — relaxing to 1")
            qualified = candidates

        qualified.sort(key=lambda x: x.score, reverse=True)
        top = qualified[:self.cfg.max_picks]

        self._print_results(top)

        symbols = [s.symbol for s in top]
        WATCHLIST_FILE.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(),
            "symbols": symbols,
            "details": [{
                "symbol":    s.symbol, "price":  s.last_price,
                "score":     round(s.score, 3),
                "rsi":       s.rsi,   "adx":    s.adx,
                "vol_ratio": s.volume_ratio,
                "mom_5d":    s.momentum_5d, "mom_10d": s.momentum_10d,
                "signals": {
                    "volume":   s.volume_signal, "momentum": s.momentum_signal,
                    "trend":    s.trend_signal,  "rsi":      s.rsi_signal,
                },
                "reasons": s.reasons,
            } for s in top],
        }, indent=2))
        logger.success(f"Watchlist saved → {WATCHLIST_FILE}  ({len(symbols)} stocks)")
        return symbols

    def _print_results(self, stocks: list[StockScore]):
        tbl = Table(title=f"🔍 Smart Screener — {datetime.now().strftime('%d %b %Y %H:%M')}",
                    box=box.SIMPLE_HEAVY, title_style="bold cyan")
        tbl.add_column("Rank", justify="right", width=5)
        tbl.add_column("Symbol",   style="bold white", width=12)
        tbl.add_column("Price",    justify="right", width=10)
        tbl.add_column("Score",    justify="right", width=7)
        tbl.add_column("RSI",      justify="right", width=6)
        tbl.add_column("ADX",      justify="right", width=6)
        tbl.add_column("Vol",      justify="right", width=7)
        tbl.add_column("5d Mom",   justify="right", width=8)
        tbl.add_column("Signals",  width=22)

        for i, s in enumerate(stocks, 1):
            sigs = []
            if s.volume_signal:   sigs.append("[yellow]VOL[/]")
            if s.momentum_signal: sigs.append("[green]MOM[/]")
            if s.trend_signal:    sigs.append("[blue]TRD[/]")
            if s.rsi_signal:      sigs.append("[cyan]RSI[/]")
            rc = "cyan" if s.rsi < 40 else ("red" if s.rsi > 70 else "white")
            mc = "green" if s.momentum_5d >= 0 else "red"
            tbl.add_row(
                f"#{i}", s.symbol, f"₹{s.last_price:,.2f}",
                f"[bold]{s.score:.2f}[/]",
                f"[{rc}]{s.rsi}[/]",
                f"{s.adx:.0f}",
                f"{s.volume_ratio:.1f}×",
                f"[{mc}]{s.momentum_5d:+.1f}%[/]",
                "  ".join(sigs) or "[dim]—[/]",
            )
        console.print(tbl)


    def run_nse(self) -> list[str]:
        """
        NSE-powered screener — uses live NSE data instead of yfinance.
        Fetches: most active + top gainers + Nifty50 quotes
        Scores each based on momentum, volume, change%.
        Much faster than yfinance batch download.
        """
        from core.nse_data import get_client
        nse = get_client()

        console.print("\n  [bold cyan]📡 NSE Live Screener[/]  "
                      f"[dim]{datetime.now().strftime('%H:%M IST')}[/]\n")

        # Gather candidates from multiple NSE endpoints
        candidates_raw = {}

        # 1. Most active stocks
        for s in nse.get_most_active():
            candidates_raw[s["symbol"]] = s

        # 2. Top gainers
        for s in nse.get_gainers():
            if s["symbol"] not in candidates_raw:
                candidates_raw[s["symbol"]] = s
            else:
                # merge — take max volume
                candidates_raw[s["symbol"]]["volume"] = max(
                    candidates_raw[s["symbol"]].get("volume",0), s.get("volume",0)
                )

        # 3. Nifty50 quotes (always include for liquidity)
        for s in nse.get_nifty50_quotes():
            if s["symbol"] not in candidates_raw:
                candidates_raw[s["symbol"]] = s

        if not candidates_raw:
            logger.warning("NSE screener: no data — raising to trigger yfinance fallback")
            raise ValueError("NSE screener returned no data")

        # Score each candidate
        scored = []
        for sym, q in candidates_raw.items():
            if sym in KNOWN_BAD_SYMBOLS:
                continue
            ltp        = float(q.get("ltp", q.get("lastPrice", 0)) or 0)
            change_pct = float(q.get("change_pct", q.get("pChange", 0)) or 0)
            volume     = int(q.get("volume", q.get("tradedVolume", 0)) or 0)

            if ltp < self.cfg.min_price_inr or volume < self.cfg.min_avg_volume:
                continue

            score = 0.0
            reasons = []

            # Positive day change
            if change_pct > 3:
                score += 2.0; reasons.append(f"+{change_pct:.1f}%")
            elif change_pct > 1.5:
                score += 1.2; reasons.append(f"+{change_pct:.1f}%")
            elif change_pct > 0.5:
                score += 0.6; reasons.append(f"+{change_pct:.1f}%")
            elif change_pct < 0:
                score -= 1.0

            # Volume
            if volume > 2_000_000:
                score += 1.5; reasons.append("high vol")
            elif volume > 1_000_000:
                score += 0.8; reasons.append("good vol")

            # Price sweet spot (₹100-₹5000)
            if 100 <= ltp <= 5000:
                score += 0.3

            if score > 0:
                scored.append({
                    "symbol":  sym,
                    "score":   round(score, 3),
                    "ltp":     ltp,
                    "chg":     change_pct,
                    "volume":  volume,
                    "reasons": reasons,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:self.cfg.max_picks]

        # Print results
        from rich.table import Table
        from rich import box as rbox
        tbl = Table(title=f"🔍 NSE Live Screener — {datetime.now().strftime('%H:%M')}",
                    box=rbox.SIMPLE_HEAVY, title_style="bold cyan")
        tbl.add_column("#",      width=4,  justify="right")
        tbl.add_column("Symbol", width=14, style="bold white")
        tbl.add_column("LTP",    width=10, justify="right")
        tbl.add_column("Chg%",   width=8,  justify="right")
        tbl.add_column("Volume", width=12, justify="right")
        tbl.add_column("Score",  width=8,  justify="right")

        for i, s in enumerate(top, 1):
            cc = "green" if s["chg"] >= 0 else "red"
            tbl.add_row(
                f"#{i}", s["symbol"], f"₹{s['ltp']:,.2f}",
                f"[{cc}]{s['chg']:+.2f}%[/]",
                f"{s['volume']:,}",
                f"[bold]{s['score']:.2f}[/]",
            )
        console.print(tbl)

        symbols = [s["symbol"] for s in top]

        # Save watchlist
        import json
        from pathlib import Path as P
        P("watchlist.json").write_text(json.dumps({
            "generated_at": datetime.now().isoformat(),
            "symbols":      symbols,
            "details":      top,
        }, indent=2))
        logger.success(f"NSE watchlist: {symbols}")
        return symbols


if __name__ == "__main__":
    picks = StockScreener().run()
    print(f"\nWatchlist: {picks}")
