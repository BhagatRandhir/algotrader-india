"""
main.py  –  AlgoTrader India (Zerodha Kite + NSE auto-screener)

How it works:
  1. Every morning before 9:15 AM IST, the auto-screener scans ~250 NSE
     stocks for volume surges, momentum, and oversold RSI — and picks the
     top 10 candidates automatically as the day's watchlist.
  2. Once market opens, the bot trades those stocks — buying and selling
     fully automatically based on the ensemble strategy signal.
  3. You do NOT need to provide any stock names. Everything is automatic.

Run:
    python auth/login.py    ← once every morning before 9:15 AM
    python main.py          ← start the bot

⚠️  Zerodha requires a STATIC IP to place orders (mandatory since April 2025).
    Whitelist your IP at: https://developers.kite.trade
"""

from __future__ import annotations

import json
import os
import signal as _signal
import time
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from core.broker import ZerodhaBroker
from core.risk import RiskConfig, RiskManager
from core.aggregator import SignalAggregator
from core.screener import StockScreener, ScreenerConfig
from core.global_sentiment import GlobalSentimentFilter, GlobalMood, GlobalSentimentResult
from core.nifty_trend import NiftyTrendFilter, MarketRegime, NiftyTrendResult
from core.news_sentiment import NewsSentimentAnalyser, NewsMood, NewsSentimentResult
from core.fii_dii import FIIDIIFilter, FlowMood, FIIDIIResult
from core.global_sentiment import GlobalSentimentFilter, GlobalSentimentResult, GlobalMood
from strategies.base import Signal
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_momentum import RSIMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from utils.dashboard import render_dashboard, log_trade
from utils.market_hours import is_market_open, is_safe_to_enter, minutes_to_open, now_ist

# ── Load config ───────────────────────────────────────────────────

load_dotenv(".env")

API_KEY      = os.environ["ZERODHA_API_KEY"]
ACCESS_TOKEN = os.environ["ZERODHA_ACCESS_TOKEN"]

# Manual watchlist fallback (used ONLY if screener fails or is disabled)
MANUAL_WATCHLIST = [s.strip() for s in os.getenv(
    "NSE_WATCHLIST", "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK").split(",")]

PRODUCT      = os.getenv("PRODUCT_TYPE",      "MIS")
ORDER_TYPE   = os.getenv("ORDER_TYPE",        "LIMIT")
LOOP_SEC     = int(os.getenv("LOOP_INTERVAL_SEC", "60"))
BAR_INTERVAL = os.getenv("BAR_INTERVAL",      "5m")
BAR_LOOKBACK = os.getenv("BAR_LOOKBACK",      "5d")
MAX_PICKS    = int(os.getenv("SCREENER_MAX_PICKS", "10"))
USE_SCREENER = os.getenv("USE_SCREENER", "true").lower() == "true"

WEIGHTS = {
    "MA_Crossover":        float(os.getenv("WEIGHT_MA_CROSSOVER",   "0.30")),
    "RSI_Momentum":        float(os.getenv("WEIGHT_RSI_MOMENTUM",   "0.25")),
    "Mean_Reversion":      float(os.getenv("WEIGHT_MEAN_REVERSION", "0.20")),
    "Pattern_Recognition": float(os.getenv("WEIGHT_PATTERN_RECOGNITION", "0.25")),
}

risk_cfg = RiskConfig(
    max_position_pct   = float(os.getenv("MAX_POSITION_PCT",   "0.05")),
    max_daily_loss_pct = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02")),
    max_open_positions = int(os.getenv("MAX_OPEN_POSITIONS",   "6")),
    stop_loss_pct      = float(os.getenv("STOP_LOSS_PCT",      "0.015")),
    target_pct         = float(os.getenv("TARGET_PCT",         "0.03")),
)

WATCHLIST_FILE = Path("watchlist.json")

# ── State ─────────────────────────────────────────────────────────

running     = True
signal_log: list[dict] = []
loop_count  = 0
sl_orders:  dict[str, str] = {}
_screener_date: date | None = None   # tracks when screener last ran
_global_sentiment: GlobalSentimentResult | None = None   # cached daily sentiment
_sentiment_date:   date | None = None
_news_analyser = NewsSentimentAnalyser()                  # 30-min internal cache
_fii_dii       = FIIDIIFilter()                           # 6-hour internal cache
_nifty_filter  = NiftyTrendFilter()                       # daily cache (see get_nifty_trend below)
_nifty_trend:      NiftyTrendResult | None = None
_nifty_trend_date: date | None = None
_sentiment: GlobalSentimentResult | None = None   # today's global sentiment
_sentiment_date: date | None = None


def shutdown(signum, frame):
    global running
    logger.warning("Shutdown requested.")
    running = False


_signal.signal(_signal.SIGINT,  shutdown)
_signal.signal(_signal.SIGTERM, shutdown)


# ── Screener integration ──────────────────────────────────────────

def load_watchlist_from_file() -> list[str] | None:
    """Load today's screener watchlist if it exists and was generated today."""
    if not WATCHLIST_FILE.exists():
        return None
    try:
        data = json.loads(WATCHLIST_FILE.read_text())
        generated = datetime.fromisoformat(data["generated_at"]).date()
        if generated == date.today():
            return data["symbols"]
    except Exception:
        pass
    return None


def run_screener() -> list[str]:
    """Run the auto-screener and return today's stock picks."""
    global _screener_date
    logger.info("🔍 Running auto-screener to build today's watchlist…")
    try:
        cfg      = ScreenerConfig(max_picks=MAX_PICKS)
        screener = StockScreener(cfg)
        picks    = screener.run()
        _screener_date = date.today()
        if picks:
            return picks
    except Exception as exc:
        logger.error(f"Screener failed: {exc}")
    # Fallback to manual watchlist
    logger.warning(f"Using manual watchlist: {MANUAL_WATCHLIST}")
    return MANUAL_WATCHLIST


def get_watchlist() -> list[str]:
    """
    Return today's watchlist.
    Priority:
      1. Already-run screener file from today  (fastest — no re-scan)
      2. Run screener now                      (takes ~2–3 min)
      3. Manual watchlist from .env            (fallback)
    """
    if not USE_SCREENER:
        return MANUAL_WATCHLIST

    # Check if we already have today's screener output
    cached = load_watchlist_from_file()
    if cached:
        logger.info(f"📋 Loaded screener watchlist from cache: {cached}")
        return cached

    return run_screener()


def should_refresh_screener() -> bool:
    """
    Refresh screener once per trading day at 8:45 AM IST
    (before market opens at 9:15 AM), or if no screener has run today.
    """
    global _screener_date
    now = now_ist()
    if _screener_date == date.today():
        return False
    # Run between 8:30–9:10 AM on weekdays (pre-market window)
    if now.weekday() < 5 and 8 <= now.hour <= 9 and now.minute <= 10:
        return True
    # Also run if market is about to open and we have no watchlist yet
    if _screener_date != date.today() and minutes_to_open() <= 30:
        return True
    return False


# ── Global sentiment ──────────────────────────────────────────────

def get_global_sentiment() -> GlobalSentimentResult:
    """
    Fetch global sentiment once per trading day (cached after first run).
    Refreshes at 8:30 AM IST before market opens.
    """
    global _sentiment, _sentiment_date
    today = date.today()

    if _sentiment is not None and _sentiment_date == today:
        return _sentiment   # already fetched today

    logger.info("🌍 Fetching global sentiment…")
    try:
        gsf        = GlobalSentimentFilter()
        _sentiment = gsf.analyse()
        _sentiment_date = today
        logger.info(f"Global mood: {_sentiment.mood.value}  "
                    f"| size multiplier: {_sentiment.size_multiplier:.0%}  "
                    f"| {_sentiment.summary}")
    except Exception as exc:
        logger.error(f"Global sentiment fetch failed: {exc} — defaulting to NEUTRAL")
        from core.global_sentiment import GlobalSentimentResult, GlobalMood
        _sentiment = GlobalSentimentResult(
            mood=GlobalMood.NEUTRAL, size_multiplier=0.5,
            summary="Fetch failed — cautious mode",
        )
        _sentiment_date = today

    return _sentiment


# ── Nifty trend filter ───────────────────────────────────────────

def get_nifty_trend() -> NiftyTrendResult:
    """
    Fetch Nifty trend regime once per trading day (cached after first run).
    Blocks/throttles BUY entries when the broader market is in a downtrend.
    """
    global _nifty_trend, _nifty_trend_date
    today = date.today()

    if _nifty_trend is not None and _nifty_trend_date == today:
        return _nifty_trend

    logger.info("📉 Fetching Nifty trend regime…")
    try:
        _nifty_trend = _nifty_filter.analyse()
        _nifty_trend_date = today
        logger.info(
            f"Nifty regime: {_nifty_trend.regime.value}  "
            f"| size multiplier: {_nifty_trend.size_multiplier:.0%}  "
            f"| allow_entry: {_nifty_trend.allow_entry}"
        )
    except Exception as exc:
        logger.error(f"Nifty trend fetch failed: {exc} — defaulting to cautious")
        _nifty_trend = NiftyTrendResult(
            regime=MarketRegime.WEAK_UPTREND, size_multiplier=0.75,
            allow_entry=True, nifty_price=0, ema20=0, ema50=0, ema200=0,
            adx=0, reason="Fetch failed — cautious mode",
        )
        _nifty_trend_date = today

    return _nifty_trend


def get_global_sentiment() -> GlobalSentimentResult:
    """
    Fetch global sentiment once per day (cached after first run).
    Falls back to NEUTRAL if data unavailable.
    """
    global _global_sentiment, _sentiment_date
    if _sentiment_date == date.today() and _global_sentiment is not None:
        return _global_sentiment   # use cached result

    logger.info("🌍 Fetching global market sentiment…")
    try:
        result = GlobalSentimentFilter().analyse()
        _global_sentiment = result
        _sentiment_date   = date.today()
        logger.info(f"Global mood: {result.mood.value}  size_multiplier={result.size_multiplier:.0%}")
        return result
    except Exception as exc:
        logger.error(f"Global sentiment failed: {exc} — defaulting to NEUTRAL")
        from core.global_sentiment import GlobalMood, GlobalSentimentResult
        fallback = GlobalSentimentResult(
            mood=GlobalMood.NEUTRAL, size_multiplier=0.75,
            summary="Data unavailable — proceeding at 75% size",
        )
        _global_sentiment = fallback
        _sentiment_date   = date.today()
        return fallback


# ── Per-symbol logic ──────────────────────────────────────────────

def process_symbol(
    symbol:     str,
    broker:     ZerodhaBroker,
    risk:       RiskManager,
    strategies: list,
    aggregator: SignalAggregator,
    positions:  dict,
    cash:       float,
    sentiment:  GlobalSentimentResult | None = None,
):
    df = broker.get_bars(symbol, interval=BAR_INTERVAL, period=BAR_LOOKBACK)
    if df.empty:
        return

    results = [s.generate_signal(df, symbol) for s in strategies]
    agg     = aggregator.aggregate(results)

    signal_log.append({
        "time":     datetime.now().strftime("%H:%M:%S"),
        "symbol":   symbol,
        "signal":   agg.final_signal.value,
        "strength": agg.combined_strength,
        "votes":    agg.vote_breakdown,
    })

    open_count = len(positions)

    # ── BUY ───────────────────────────────────────────────────────
    if agg.final_signal == Signal.BUY and symbol not in positions:

        # Global sentiment gate — block entries on bearish days
        if sentiment and not sentiment.allow_entry:
            logger.warning(
                f"{symbol}: BUY signal blocked — global sentiment BEARISH "
                f"({sentiment.summary})"
            )
            return

        # Nifty trend gate — block entries when the broader market is falling
        nifty = get_nifty_trend()
        if not nifty.allow_entry:
            logger.warning(
                f"{symbol}: BUY BLOCKED — Nifty trend bearish "
                f"({nifty.regime.value}: {nifty.reason})"
            )
            return

        # News sentiment gate — block on negative stock-specific news
        news = _news_analyser.analyse(symbol)
        if not news.allow_entry:
            logger.warning(
                f"{symbol}: BUY BLOCKED — negative news "
                f"(score={news.score:+.3f}  '{news.top_headlines[0][:55]}')"
            )
            return

        if not is_safe_to_enter():
            logger.warning(f"{symbol}: BUY signal but too close to market close")
            return
        if not risk.check_position_count(open_count):
            return

        ltp = broker.get_ltp(symbol)
        if not ltp:
            return

        # FII/DII gate — block on strong institutional outflow
        flow = _fii_dii.analyse()
        if not flow.allow_entry:
            logger.warning(
                f"{symbol}: BUY BLOCKED — FII/DII bearish "
                f"({flow.summary})"
            )
            return

        # Combined size multiplier: global × nifty × news × fii/dii
        global_mult   = sentiment.size_multiplier if sentiment else 1.0
        nifty_mult    = nifty.size_multiplier
        news_mult     = news.size_multiplier
        flow_mult     = flow.size_multiplier
        combined_mult = global_mult * nifty_mult * news_mult * flow_mult
        qty = risk.position_size(cash, ltp, agg.combined_strength * combined_mult)
        if qty <= 0:
            logger.warning(f"{symbol}: qty=0 — insufficient cash or signal too weak")
            return

        limit_px = round(ltp * 1.002, 2) if ORDER_TYPE == "LIMIT" else None
        order_id = broker.place_order(
            symbol=symbol, action="BUY", quantity=qty,
            order_type=ORDER_TYPE, product=PRODUCT, limit_price=limit_px,
        )
        if order_id:
            sl     = risk.stop_loss_price(ltp, "BUY")
            target = risk.target_price(ltp, "BUY")
            sl_id  = broker.place_sl_order(
                symbol=symbol, action="SELL", quantity=qty,
                trigger_price=round(sl * 0.999, 2), limit_price=sl,
                product=PRODUCT,
            )
            if sl_id:
                sl_orders[symbol] = sl_id
            log_trade("BUY", symbol, qty, ltp, sl, target, order_id)

    # ── SELL ──────────────────────────────────────────────────────
    elif agg.final_signal == Signal.SELL and symbol in positions:
        pos = positions[symbol]
        qty = abs(pos["qty"])
        if qty <= 0:
            return

        ltp = broker.get_ltp(symbol)
        if not ltp:
            return

        if symbol in sl_orders:
            broker.cancel_order(sl_orders.pop(symbol))

        limit_px = round(ltp * 0.998, 2) if ORDER_TYPE == "LIMIT" else None
        order_id = broker.place_order(
            symbol=symbol, action="SELL", quantity=qty,
            order_type=ORDER_TYPE, product=PRODUCT, limit_price=limit_px,
        )
        if order_id:
            log_trade("SELL", symbol, qty, ltp, sl=0, target=0, order_id=order_id)


# ── Main loop ─────────────────────────────────────────────────────

def main():
    global loop_count

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/trader.log", rotation="1 day", retention="30 days",
               level="DEBUG", enqueue=True)

    if not API_KEY or not ACCESS_TOKEN:
        raise SystemExit(
            "❌  ZERODHA_API_KEY and ZERODHA_ACCESS_TOKEN not set in .env\n"
            "    Run:  python auth/login.py"
        )

    broker     = ZerodhaBroker(API_KEY, ACCESS_TOKEN)
    risk       = RiskManager(risk_cfg)
    strategies = [
        MACrossoverStrategy(fast=9, slow=21, trend=200),
        RSIMomentumStrategy(period=14, oversold=30, overbought=70),
        MeanReversionStrategy(bb_period=20, bb_std=2.0),
        PatternRecognitionStrategy(),
    ]
    aggregator = SignalAggregator(WEIGHTS)

    # Initial watchlist — screener runs here on first boot
    watchlist = get_watchlist()
    logger.info(f"📋 Today's watchlist ({len(watchlist)} stocks): {watchlist}")
    logger.info(f"Product: {PRODUCT}  |  Order type: {ORDER_TYPE}  |  Loop: {LOOP_SEC}s")

    # Fetch global sentiment once at startup
    get_global_sentiment()

    while running:
        loop_count += 1

        # ── Refresh screener + sentiment once per day ─────────────
        if should_refresh_screener():
            watchlist = run_screener()
            logger.info(f"📋 Watchlist refreshed: {watchlist}")
            get_global_sentiment()   # re-fetch sentiment on same morning refresh

        mkt_open  = is_market_open()
        positions = broker.get_positions()
        cash      = broker.get_available_cash()
        daily_pnl = broker.get_daily_pnl()

        # ── Risk kill-switch ──────────────────────────────────────
        if not risk.check_daily_loss(daily_pnl, cash):
            broker.cancel_all_orders()
            render_dashboard(cash, daily_pnl, positions, signal_log,
                             halted=True, halt_reason=risk.halt_reason,
                             loop_count=loop_count, market_open=mkt_open)
            time.sleep(LOOP_SEC)
            continue

        render_dashboard(cash, daily_pnl, positions, signal_log,
                         halted=False, loop_count=loop_count,
                         market_open=mkt_open)

        # ── Trade ─────────────────────────────────────────────────
        if mkt_open:
            sentiment = get_global_sentiment()
            for symbol in watchlist:
                try:
                    process_symbol(symbol, broker, risk, strategies,
                                   aggregator, positions, cash, sentiment)
                except Exception as exc:
                    logger.error(f"Error on {symbol}: {exc}")
        else:
            mins = minutes_to_open()
            logger.info(
                "Market closed. " +
                (f"Opens in ~{mins} min." if mins > 0 else "Weekend or holiday.")
            )

        logger.debug(f"Loop {loop_count} done. Sleeping {LOOP_SEC}s…")
        time.sleep(LOOP_SEC)

    logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    main()
