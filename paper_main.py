"""
paper_main.py  –  AlgoTrader India — AGGRESSIVE MOMENTUM BOT

Aggressive settings:
  • Fewer gates — only 3 required (was 8)
  • Wider RSI range 50-80
  • Lower volume threshold 1.3x (was 1.5x)
  • No MTF block — uses it only to SIZE the trade
  • No regime block — only reduces size in bad conditions
  • Trailing SL wired directly into main loop
  • Targets 3-5 trades per day

Run:  python paper_main.py
Test: python paper_main.py --test
"""
from __future__ import annotations

import os, sys, json, signal as _signal, time
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.paper_broker import PaperBroker
from core.risk import RiskConfig, RiskManager
from core.screener import StockScreener, ScreenerConfig
from core.smart_exit import SmartExitManager
from core.performance_tracker import PerformanceTracker
from strategies.base import Signal
from core.smart_signals import run_smart_signals
from core.store import get_store
from core.market_filter import MarketContextFilter
from strategies.momentum_bot import MomentumBotStrategy
from utils.dashboard import render_dashboard, log_trade
from utils.market_hours import is_market_open, minutes_to_open, now_ist

console = Console()
load_dotenv(".env")

# ── Config ────────────────────────────────────────────────────────
INITIAL_CAPITAL      = float(os.getenv("INITIAL_CAPITAL",     "500000"))
MAX_PICKS            = int(os.getenv("SCREENER_MAX_PICKS",     "15"))
LOOP_SEC             = int(os.getenv("LOOP_INTERVAL_SEC",      "60"))
BAR_INTERVAL         = os.getenv("BAR_INTERVAL",  "5m")
BAR_LOOKBACK         = os.getenv("BAR_LOOKBACK",  "5d")
PRODUCT              = os.getenv("PRODUCT_TYPE",  "MIS")
USE_SCREENER         = os.getenv("USE_SCREENER",  "true").lower() == "true"
SCREENER_REFRESH_MIN = int(os.getenv("SCREENER_REFRESH_MIN",   "60"))
DAILY_TRADE_TARGET   = int(os.getenv("DAILY_TRADE_TARGET",     "5"))
MANUAL_WATCHLIST     = [s.strip() for s in os.getenv(
    "NSE_WATCHLIST",
    "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,SBIN,AXISBANK,"
    "BAJFINANCE,TITAN,MARUTI,SUNPHARMA,LT,TATAMOTORS,KOTAKBANK",
).split(",")]

risk_cfg = RiskConfig(
    max_position_pct    = float(os.getenv("MAX_POSITION_PCT",    "0.08")),
    max_daily_loss_pct  = float(os.getenv("MAX_DAILY_LOSS_PCT",  "0.03")),
    max_open_positions  = int(os.getenv("MAX_OPEN_POSITIONS",    "6")),
    stop_loss_pct       = float(os.getenv("STOP_LOSS_PCT",       "0.015")),
    target_pct          = float(os.getenv("TARGET_PCT",          "0.030")),
    min_rr_ratio        = float(os.getenv("MIN_RR_RATIO",        "1.5")),
    trail_trigger_pct   = float(os.getenv("TRAIL_TRIGGER_PCT",   "0.012")),
    consec_loss_limit   = int(os.getenv("CONSEC_LOSS_LIMIT",     "4")),
)

WATCHLIST_FILE = Path("watchlist.json")

# ── State ─────────────────────────────────────────────────────────
running        = True
signal_log:    list[dict] = []
loop_count     = 0
_last_refresh: datetime | None = None
_trades_today  = 0
_trade_date:   date | None = None

_mkt_filter = MarketContextFilter()
_strategy  = MomentumBotStrategy(
    ema_fast   = 9,
    ema_mid    = 20,
    ema_slow   = 50,
    rsi_period = 14,
    rsi_buy_lo = 50.0,   # AGGRESSIVE: was 55
    rsi_buy_hi = 88.0,   # AGGRESSIVE: wider window
    rsi_sell   = 42.0,   # AGGRESSIVE: lower exit threshold
    rsi_ob     = 95.0,
    vol_mult   = 1.3,    # AGGRESSIVE: was 1.5
)
_exit_mgr  = SmartExitManager()
_tracker   = PerformanceTracker()


def shutdown(signum, frame):
    global running
    logger.warning("Stopping…")
    running = False

_signal.signal(_signal.SIGINT,  shutdown)
_signal.signal(_signal.SIGTERM, shutdown)


def _write_signal_log(log: list):
    """Write latest signal to store (Redis or file)."""
    pass   # signals now written individually via store.append_signal()


def _write_nav_snapshot(broker):
    """Save NAV snapshot to store (Redis or file)."""
    try:
        get_store().save_nav_snapshot({
            "time": datetime.now().strftime("%H:%M"),
            "nav":  round(broker.get_portfolio_value(), 2),
            "cash": round(broker.get_available_cash(), 2),
            "pnl":  round(broker.get_daily_pnl(), 2),
        })
    except Exception:
        pass


# ── Daily counter ─────────────────────────────────────────────────
def _reset_daily():
    global _trades_today, _trade_date
    today = now_ist().date()
    if _trade_date != today:
        _trades_today = 0
        _trade_date   = today

def _inc(): 
    global _trades_today
    _trades_today += 1


# ── Watchlist ─────────────────────────────────────────────────────
def _load_cached() -> list[str] | None:
    if not WATCHLIST_FILE.exists():
        return None
    try:
        data = json.loads(WATCHLIST_FILE.read_text())
        gen  = datetime.fromisoformat(data["generated_at"])
        now  = now_ist()
        if gen.date() == now.date() and gen.hour == now.hour:
            return data["symbols"]
    except Exception:
        pass
    return None

def _run_screener(current: list[str] = None, positions: dict = None) -> list[str]:
    logger.info("🔍 Screener running…")
    try:
        from core.watchlist_optimiser import WatchlistOptimiser
        screener = StockScreener(ScreenerConfig(max_picks=MAX_PICKS))

        # Try NSE live screener first (faster, more accurate)
        try:
            symbols = screener.run_nse()
            if symbols:
                logger.info(f"NSE screener: {symbols}")
                return symbols
        except Exception as e:
            logger.warning(f"NSE screener failed: {e} — trying yfinance screener")

        # Fallback: yfinance-based screener with optimiser
        all_scored = screener.run_scored()
        if all_scored:
            opt = WatchlistOptimiser(swap_threshold=0.20)
            new_wl, swaps = opt.optimise(
                all_scored, current or MANUAL_WATCHLIST,
                positions or {}, MAX_PICKS,
            )
            opt.print_swap_report(swaps, new_wl)
            return new_wl
    except Exception as exc:
        logger.error(f"Screener: {exc}")
    return list(MANUAL_WATCHLIST)

def get_watchlist() -> list[str]:
    if not USE_SCREENER:
        return list(MANUAL_WATCHLIST)
    cached = _load_cached()
    return cached if cached else _run_screener()

def _should_refresh() -> bool:
    if not is_market_open():
        return False
    if _last_refresh is None:
        return True
    return (now_ist() - _last_refresh).total_seconds() / 60 >= SCREENER_REFRESH_MIN

def _merge(picks: list[str], positions: dict) -> list[str]:
    extra = sorted(set(positions.keys()) - set(picks))
    if extra:
        logger.info(f"📌 Keeping held: {extra}")
    return picks + extra


# ── Entry logic ───────────────────────────────────────────────────
def _try_buy(symbol: str, broker: PaperBroker, risk: RiskManager,
             positions: dict, cash: float, df):
    """
    Aggressive entry: only 3 hard requirements
      1. MomentumBot signal = BUY  (RSI 50-80, price>VWAP>EMA, vol>1.3x, day+)
      2. Not already holding
      3. Position count < max
    MTF and regime used only to scale size — never to block.
    """
    # ── Market context gate (Nifty trend + VIX + time) ──────────
    mkt = _mkt_filter.check(broker)
    if not mkt.allow:
        logger.debug(f"{symbol}: market filter block — {mkt.reason}")
        return

    result = _strategy.generate_signal(df, symbol)

    entry = {
        "type":     "signal",
        "time":     datetime.now().strftime("%H:%M:%S"),
        "symbol":   symbol,
        "signal":   result.signal.value,
        "strength": round(result.strength, 3),
        "rsi":      round(_strategy._rsi(df["close"], 14), 1) if not df.empty else 0,
        "vol":      round(float(df["volume"].iloc[-1]) / (float(df["volume"].rolling(20).mean().iloc[-1])+1), 2) if not df.empty else 0,
    }
    signal_log.append(entry)
    try:
        get_store().append_signal(entry)
    except Exception:
        pass

    if result.signal != Signal.BUY:
        return

    if symbol in positions:
        return

    if not risk.check_position_count(len(positions)):
        logger.debug(f"{symbol}: max positions reached")
        return

    if _trades_today >= DAILY_TRADE_TARGET * 2:
        logger.debug(f"{symbol}: daily hard limit {DAILY_TRADE_TARGET*2} reached")
        return

    ltp = broker.get_ltp(symbol)
    if not ltp or ltp <= 0:
        logger.warning(f"{symbol}: no LTP — skip")
        return

    sl     = risk.stop_loss_price(ltp)
    target = risk.target_price(ltp)

    # MTF sizing bonus (never blocks)
    mtf_mult = 1.0
    try:
        from strategies.multi_timeframe import MultiTimeframeConfirmer
        mtf_result = MultiTimeframeConfirmer(broker).confirm(symbol)
        mtf_mult   = max(mtf_result.size_mult, 0.60)   # minimum 60% even on bad MTF
    except Exception:
        pass

    # ── Smart signals gate ───────────────────────────────────────
    smart_mult = 1.0
    smart_summary = "not run"
    try:
        day_chg_pct = float(
            (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
            if len(df) >= 2 else 0.0
        )
        smart = run_smart_signals(symbol, df, day_chg_pct)
        smart_summary = smart["summary"]
        if not smart["allow"]:
            logger.info(f"{symbol}: ❌ Smart block — {smart_summary[:80]}")
            return
        smart_mult = max(0.6, min(1.0 + smart["score"] * 0.3, 1.3))
        logger.debug(f"{symbol}: ✅ Smart OK score={smart['score']:.2f} mult={smart_mult:.2f}")
    except Exception as exc:
        logger.debug(f"{symbol}: smart signals error: {exc}")

    # Combined size: signal × MTF × market context × smart signals
    combined_mult = result.strength * mtf_mult * mkt.size_mult * smart_mult
    qty = risk.position_size(cash, ltp, signal_strength=combined_mult)
    if qty <= 0:
        logger.debug(f"{symbol}: qty=0 (cash={cash:.0f} ltp={ltp:.0f})")
        return

    oid = broker.place_order(symbol, "BUY", qty, "MARKET", PRODUCT)
    if oid:
        _exit_mgr.register(symbol, ltp, qty, sl)
        log_trade("BUY", symbol, qty, ltp, sl, target, oid)
        _inc()
        _tracker.record_buy(
            trade_id=oid, symbol=symbol, quantity=qty, price=ltp,
            strategy_signals={"MomentumBot": "BUY"},
            ensemble_score=result.strength,
            global_mood="N/A", nifty_regime="N/A",
            news_mood="N/A", fii_mood="N/A",
            size_multiplier=combined_mult,
        )
        console.print(
            f"\n  ✅ [bold green]BUY #{_trades_today}[/]  "
            f"[bold]{symbol}[/]  qty=[cyan]{qty}[/]  "
            f"@₹[yellow]{ltp:.2f}[/]  "
            f"SL=₹{sl:.2f}  "
            f"Str=[magenta]{result.strength:.2f}[/]  "
            f"MTF={mtf_mult:.0%}  Smart={smart_mult:.0%}  VIX={mkt.vix:.1f}\n"
            f"  [dim]{result.reason}[/]\n"
            f"  [dim]{smart_summary[:80]}[/]\n"
        )


# ── Exit logic ────────────────────────────────────────────────────
def _handle_exits(symbol: str, broker: PaperBroker, risk: RiskManager,
                  positions: dict, df):
    """
    Smart exit: partial at +1.2%, trail after, full exit on signal/SL/target/3PM.
    """
    ltp = broker.get_ltp(symbol)
    if not ltp or symbol not in positions:
        return

    pos   = positions[symbol]
    entry = pos["average_price"]
    qty   = abs(pos["qty"])

    # Register with exit manager if new
    if not _exit_mgr.is_registered(symbol):
        sl = risk.stop_loss_price(entry)
        _exit_mgr.register(symbol, entry, qty, sl)

    # Smart exit decision (partial/trail/dryup/divergence/force)
    decision = _exit_mgr.evaluate(symbol, df, ltp)

    if decision.action in ("PARTIAL", "FULL"):
        sell_qty = decision.qty
        oid = broker.place_order(symbol, "SELL", sell_qty, "MARKET", PRODUCT)
        if oid:
            pnl = (ltp - entry) * sell_qty
            risk.record_outcome(pnl)
            log_trade("SELL", symbol, sell_qty, ltp, 0, 0, oid)
            _tracker.record_sell(symbol, ltp, exit_reason=decision.reason)
            icon = "✅" if pnl > 0 else "❌"
            console.print(
                f"\n  {icon} [bold]{'PARTIAL' if decision.action=='PARTIAL' else 'SELL'}[/]"
                f"  [bold]{symbol}[/]  qty={sell_qty}  @₹{ltp:.2f}"
                f"  P&L=[{'green' if pnl>0 else 'red'}]₹{pnl:+,.0f}[/]"
                f"  [{decision.reason}]\n"
            )
            if decision.action == "FULL":
                _exit_mgr.clear(symbol)
        return

    # Strategy SELL signal
    result = _strategy.generate_signal(df, symbol)
    if result.signal == Signal.SELL:
        remaining = _exit_mgr.get_remaining_qty(symbol) or qty
        oid = broker.place_order(symbol, "SELL", remaining, "MARKET", PRODUCT)
        if oid:
            pnl = (ltp - entry) * remaining
            risk.record_outcome(pnl)
            log_trade("SELL", symbol, remaining, ltp, 0, 0, oid)
            _tracker.record_sell(symbol, ltp, exit_reason="STRATEGY")
            _exit_mgr.clear(symbol)
            icon = "✅" if pnl > 0 else "❌"
            console.print(
                f"\n  {icon} [bold]SELL[/]  [bold]{symbol}[/]"
                f"  qty={remaining}  @₹{ltp:.2f}"
                f"  P&L=[{'green' if pnl>0 else 'red'}]₹{pnl:+,.0f}[/]"
                f"  [STRATEGY: {result.reason}]\n"
            )


# ── Status print ──────────────────────────────────────────────────
def _print_status(positions, cash, daily_pnl):
    console.print()
    t = Table(box=box.SIMPLE_HEAVY, title="📊 Session", title_style="bold cyan")
    t.add_column("", style="bold white", width=20)
    t.add_column("", justify="right")
    t.add_row("Trades today",    f"{_trades_today} / {DAILY_TRADE_TARGET} target")
    t.add_row("Open positions",  str(len(positions)))
    t.add_row("Available cash",  f"₹{cash:,.0f}")
    c = "green" if daily_pnl >= 0 else "red"
    t.add_row("Day P&L", f"[{c}]₹{daily_pnl:+,.0f}[/]")
    if positions:
        for sym, p in positions.items():
            ltp   = p.get("last_price", p["average_price"])
            pnl   = (ltp - p["average_price"]) * abs(p["qty"])
            c2    = "green" if pnl >= 0 else "red"
            sl    = _exit_mgr.get_current_sl(sym) or risk_cfg.stop_loss_pct
            t.add_row(f"  {sym}", f"qty={abs(p['qty'])} [{c2}]₹{pnl:+,.0f}[/]")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────
def main():
    global loop_count, _last_refresh

    os.makedirs("logs", exist_ok=True)
    logger.add("logs/trader.log", rotation="1 day",
               retention="30 days", level="DEBUG", enqueue=True)

    console.print(Panel(
        "  [bold cyan]AlgoTrader India — AGGRESSIVE MOMENTUM BOT[/]\n\n"
        "  Entry (3 conditions only — no unnecessary blocks):\n"
        "    ✅  Price > VWAP  &  Price > EMA20\n"
        "    ✅  RSI 50–80  (wide window)\n"
        "    ✅  Volume > 1.3× avg  +  Day Change > 0%\n\n"
        "  Exit (smart trailing):\n"
        "    💰  Partial exit at +1.2%  →  SL to breakeven\n"
        "    🔒  Trail 1% below peak on remaining shares\n"
        "    📉  Volume dry-up  /  RSI divergence  /  3PM force\n"
        "    🛑  Fixed SL 1.5%  |  Target 3.0%\n\n"
        f"  🎯  Target {DAILY_TRADE_TARGET} trades/day\n"
        f"  🔄  Screener refresh every {SCREENER_REFRESH_MIN} min\n"
        "  [dim]MTF used for sizing only — never blocks entry[/]\n"
        "  [dim]Regime used for sizing only — never blocks entry[/]",
        title="⚡ Aggressive Momentum Bot",
        style="bold green",
        box=box.DOUBLE_EDGE,
    ))

    broker        = PaperBroker(initial_capital=INITIAL_CAPITAL)
    risk          = RiskManager(risk_cfg)
    _starting_nav = broker.get_portfolio_value()

    base_watchlist = get_watchlist()
    _last_refresh  = now_ist()
    logger.info(f"📋 Watchlist ({len(base_watchlist)}): {base_watchlist}")
    logger.info(f"💰 Starting NAV: ₹{_starting_nav:,.0f}")

    while running:
        loop_count += 1
        _reset_daily()

        mkt_open  = is_market_open()
        positions = broker.get_positions()
        cash      = broker.get_available_cash()
        daily_pnl = broker.get_daily_pnl()

        # Hourly screener refresh
        if _should_refresh():
            base_watchlist = _run_screener(base_watchlist, positions)
            _last_refresh  = now_ist()
            logger.info(f"🔄 Watchlist refreshed: {base_watchlist}")

        watchlist = _merge(base_watchlist, positions)

        # Daily loss kill-switch
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

        if mkt_open:
            for symbol in watchlist:
                try:
                    positions = broker.get_positions()
                    cash      = broker.get_available_cash()
                    df = broker.get_bars(symbol, interval=BAR_INTERVAL,
                                         period=BAR_LOOKBACK)
                    if df is None or df.empty or len(df) < 30:
                        logger.debug(f"{symbol}: insufficient bars ({len(df) if df is not None else 0})")
                        continue

                    if symbol in positions:
                        _handle_exits(symbol, broker, risk, positions, df)
                    else:
                        _try_buy(symbol, broker, risk, positions, cash, df)

                except Exception as exc:
                    logger.error(f"{symbol}: {exc}")

            if loop_count % 5 == 0:
                positions = broker.get_positions()
                _print_status(positions, cash, daily_pnl)
                _write_nav_snapshot(broker)

        else:
            mins = minutes_to_open()
            logger.info(
                f"{'⏰ Opens in ' + str(mins) + ' min' if mins > 0 else '🌙 Market closed'}"
                f"  |  Trades today: {_trades_today}"
            )

        time.sleep(LOOP_SEC)

    # Shutdown
    console.print()
    broker.print_portfolio()
    broker.print_trade_history()
    _tracker.record_daily(starting_nav=_starting_nav,
                          ending_nav=broker.get_portfolio_value())
    _tracker.print_report()
    logger.info(f"Stopped. Total trades: {_trades_today}")


# ── Test mode ─────────────────────────────────────────────────────
def run_test():
    """
    Simulates a full trading day using real NSE data.
    Run with:  python paper_main.py --test
    """
    import random
    console.print(Panel("🧪 TEST MODE — simulating live trading", style="yellow"))

    broker = PaperBroker(initial_capital=500_000)
    risk   = RiskManager(risk_cfg)

    # Fetch real data for top stocks
    watchlist = MANUAL_WATCHLIST[:8]
    console.print(f"Fetching data for: {watchlist}")

    results = []
    for sym in watchlist:
        df = broker.get_bars(sym, interval="5m", period="5d")
        if df is None or len(df) < 30:
            console.print(f"  [red]✗[/] {sym}: no data ({len(df) if df is not None else 0} bars)")
            continue

        console.print(f"  [green]✓[/] {sym}: {len(df)} bars  last=₹{df['close'].iloc[-1]:.2f}")
        r = _strategy.generate_signal(df, sym)
        console.print(f"     signal=[bold]{r.signal.value}[/] str={r.strength:.2f}  {r.reason}")

        if r.signal == Signal.BUY:
            ltp = broker.get_ltp(sym)
            if ltp:
                sl  = risk.stop_loss_price(ltp)
                qty = risk.position_size(500_000, ltp, r.strength)
                oid = broker.place_order(sym, "BUY", qty, "MARKET", PRODUCT)
                if oid:
                    _exit_mgr.register(sym, ltp, qty, sl)
                    results.append(sym)
                    console.print(f"     [bold green]✅ BOUGHT {qty} @ ₹{ltp:.2f}[/]")

    console.print(f"\n  Total buys: {len(results)}: {results}")
    broker.print_portfolio()


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test()
    else:
        main()
