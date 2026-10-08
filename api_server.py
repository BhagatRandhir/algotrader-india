"""
api_server.py  –  REST API for the Trading Dashboard

Production-ready: uses Store (Redis on Render, file locally).

Start:  python3 api_server.py
Port:   http://localhost:5050  (or $PORT on Render)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from core.store import get_store
from core.paper_broker import PaperBroker
from core.nse_data import get_client as get_nse
from strategies.momentum_bot import MomentumBotStrategy
from core.smart_signals import run_smart_signals
from core.earnings_analyser import get_earnings_analyser
from core.risk import RiskConfig, RiskManager
from utils.market_hours import is_market_open, now_ist

app  = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, ngrok-skip-browser-warning"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["ngrok-skip-browser-warning"]   = "true"
    return response

@app.route("/api/options_preflight", methods=["OPTIONS"])
def handle_options():
    return "", 204

INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "500000"))
PORT            = int(os.getenv("PORT", "5050"))

_broker  = PaperBroker(initial_capital=INITIAL_CAPITAL)
_strat   = MomentumBotStrategy()
_nse     = get_nse()
_risk    = RiskManager(RiskConfig())


def _enrich_positions(positions: dict) -> list[dict]:
    result = []
    for sym, pos in positions.items():
        try:
            ltp = _broker.get_ltp(sym) or pos.get("average_price", 0)
            entry = pos.get("average_price", 0)
            qty   = abs(pos.get("qty", 0))
            unreal= round((ltp - entry) * qty, 2)
            result.append({
                "symbol":        sym,
                "qty":           qty,
                "entry":         round(entry, 2),
                "ltp":           round(ltp, 2),
                "unrealised":    unreal,
                "unrealised_pct":round((ltp - entry) / entry * 100, 2) if entry else 0,
            })
        except Exception:
            pass
    return result


def _nifty_breadth() -> dict:
    try:
        quotes = _nse.get_nifty50_quotes()
        positive = sum(1 for q in quotes if float(q.get("change_pct", 0)) > 0)
        negative = sum(1 for q in quotes if float(q.get("change_pct", 0)) < 0)
        neutral = max(len(quotes) - positive - negative, 0)
        index = _nse.get_index_snapshot("NIFTY 50") or {}
        return {
            "symbol": index.get("symbol", "NIFTY 50"),
            "price": round(float(index.get("price", 0)), 2),
            "change": round(float(index.get("change", 0)), 2),
            "change_pct": round(float(index.get("change_pct", 0)), 2),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "ratio": f"{positive}:{negative}",
            "breadth_pct": round((positive / len(quotes)) * 100, 1) if quotes else 0,
        }
    except Exception as exc:
        logger.debug(f"Nifty breadth error: {exc}")
        return {
            "symbol": "NIFTY 50",
            "price": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "ratio": "0:0",
            "breadth_pct": 0,
        }


def _pnl_by_day(orders: list[dict], days: int = 30) -> list[dict]:
    daily: dict[str, float] = {}
    tcount: dict[str, int]  = {}
    buys: dict[str, list]   = {}
    for o in orders:
        if o.get("status") != "COMPLETE":
            continue
        d = o.get("timestamp", "")[:10]
        if o["action"] == "BUY":
            buys.setdefault(o["symbol"], []).append(o)
        elif o["action"] == "SELL":
            q = buys.get(o["symbol"], [])
            if q:
                b = q.pop(0)
                pnl = (o["price"] - b["price"]) * o["quantity"]
                daily[d]  = round(daily.get(d, 0) + pnl, 2)
                tcount[d] = tcount.get(d, 0) + 1

    result = []
    for i in range(days - 1, -1, -1):
        d = (date.today() - timedelta(days=i)).isoformat()
        if tcount.get(d, 0) <= 0:
            continue
        result.append({
            "date":   d,
            "label":  (date.today() - timedelta(days=i)).strftime("%d %b"),
            "pnl":    daily.get(d, 0),
            "trades": tcount.get(d, 0),
        })
    return result


def _pnl_by_week(daily):
    w, wc = {}, {}
    for d in daily:
        dt  = datetime.strptime(d["date"], "%Y-%m-%d")
        key = f"W{dt.isocalendar()[1]} {dt.strftime('%b')}"
        w[key]  = round(w.get(key, 0)  + d["pnl"], 2)
        wc[key] = wc.get(key, 0) + d["trades"]
    return [{"week": k, "pnl": v, "trades": wc[k]} for k, v in w.items()]


def _pnl_by_month(daily):
    m, mc = {}, {}
    for d in daily:
        key = datetime.strptime(d["date"], "%Y-%m-%d").strftime("%b %Y")
        m[key]  = round(m.get(key, 0)  + d["pnl"], 2)
        mc[key] = mc.get(key, 0) + d["trades"]
    return [{"month": k, "pnl": v, "trades": mc[k]} for k, v in m.items()]


def _analyse_symbol(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    df = _broker.get_bars(symbol, interval="5m", period="5d")
    if df is None or len(df) < 30:
        return {"error": f"Insufficient data for {symbol}"}

    result  = _strat.generate_signal(df, symbol)
    quote   = _nse.get_quote(symbol) or {}
    ltp     = quote.get("ltp") or float(df["close"].iloc[-1])
    price   = ltp
    ema20   = float(_strat._ema(df["close"], 20).iloc[-1])
    ema50   = float(_strat._ema(df["close"], 50).iloc[-1])
    vwap    = _strat._vwap(df)
    rsi     = _strat._rsi(df["close"], 14)
    volume  = float(df["volume"].iloc[-1])
    avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])

    # Fetch accurate data from yfinance (prev close, 52W, day change)
    week52_high = 0.0
    week52_low  = 0.0
    prev_close  = 0.0
    try:
        import yfinance as yf
        info        = yf.Ticker(f"{symbol}.NS").fast_info
        week52_high = round(float(info.year_high      or 0), 2)
        week52_low  = round(float(info.year_low       or 0), 2)
        prev_close  = round(float(info.previous_close or 0), 2)
    except Exception:
        pass

    # Day change: use previous close for accuracy
    if prev_close > 0:
        day_chg = (price - prev_close) / prev_close
    elif quote.get("change_pct"):
        day_chg = float(quote["change_pct"]) / 100
    else:
        day_chg = float((price - df["close"].iloc[-2]) / df["close"].iloc[-2]
                        if len(df) >= 2 else 0.0)

    checks = [
        {"label":"Price > VWAP",      "pass": bool(price>vwap),           "detail":f"₹{price:.0f} vs ₹{vwap:.0f}"},
        {"label":"Price > EMA20",     "pass": bool(price>ema20),          "detail":f"EMA20=₹{ema20:.0f}"},
        {"label":"EMA20 > EMA50",     "pass": bool(ema20>ema50),          "detail":f"EMA50=₹{ema50:.0f}"},
        {"label":"RSI 55–92",         "pass": bool(55<=rsi<=92),          "detail":f"RSI={rsi:.0f}"},
        {"label":"Volume > 1.5×",     "pass": bool(volume>1.5*avg_vol),   "detail":f"{volume/(avg_vol+1):.1f}× avg"},
        {"label":"Day change > 0.3%", "pass": bool(day_chg>=0.003),       "detail":f"{day_chg:+.2%}"},
    ]
    tech_score = sum(1 for c in checks if c["pass"])

    # 30-day price history
    df_hist = _broker.get_bars(symbol, interval="1d", period="3mo")
    price_history = []
    if df_hist is not None and not df_hist.empty:
        for idx, row in df_hist.tail(30).iterrows():
            price_history.append({
                "date":  str(idx.date() if hasattr(idx,"date") else idx)[:10],
                "price": round(float(row["close"]), 2),
            })

    # Run smart signals
    smart = {}
    smart_score = 0.0
    smart_allow = True
    try:
        smart = run_smart_signals(symbol, df, round(day_chg * 100, 2))
        smart_score = smart.get("score", 0.0)
        smart_allow = smart.get("allow", True)
    except Exception as exc:
        logger.debug(f"Smart signals {symbol}: {exc}")

    # Run earnings analysis
    earnings = {}
    earnings_score = 0.0
    try:
        ea = get_earnings_analyser()
        er = ea.analyse(symbol)
        earnings = ea.format_for_display(er)
        earnings_score = er.score
    except Exception as exc:
        logger.debug(f"Earnings analysis {symbol}: {exc}")

    # ── Combined verdict using all 3 analyses ─────────────────────
    # Technical:    40% weight (0-6 conditions → 0.0-1.0)
    # Smart signals:35% weight (-1.0 to +1.0)
    # Earnings:     25% weight (-1.0 to +1.0)
    tech_norm    = (tech_score / 6.0) * 2 - 1      # convert 0-6 → -1.0 to +1.0
    combined     = (tech_norm * 0.40) + (smart_score * 0.35) + (earnings_score * 0.25)
    combined     = round(max(-1.0, min(1.0, combined)), 3)

    # Hard override: if smart signals block → cap at HOLD
    if not smart_allow and combined > 0:
        combined = min(combined, 0.0)

    # ── Final verdict — clear and actionable ────────────────────────
    #
    # Rules:
    #   BUY   → combined > 0.30 AND tech >= 4/6 AND smart_allow
    #   SELL  → combined < -0.20 OR (tech <= 2/6 AND combined < 0)
    #   HOLD  → everything else — not enough conviction either way
    #
    # Confidence tells user HOW SURE the system is:
    #   HIGH   → all 3 analyses agree
    #   MEDIUM → 2 of 3 agree
    #   LOW    → mixed signals — treat with caution

    # Count how many analyses agree with direction
    tech_bullish     = tech_score >= 4
    tech_bearish     = tech_score <= 2
    smart_bullish    = smart_score > 0.1
    smart_bearish    = smart_score < -0.1
    earnings_bullish = earnings_score > 0.1
    earnings_bearish = earnings_score < -0.1

    bull_count = sum([tech_bullish,    smart_bullish,    earnings_bullish])
    bear_count = sum([tech_bearish,    smart_bearish,    earnings_bearish])

    if combined >= 0.30 and tech_score >= 4 and smart_allow:
        verdict    = "BUY"
        confidence = "HIGH"   if bull_count >= 3 else "MEDIUM"
        action     = "Enter trade. All signals aligned." if bull_count >= 3 else "Enter trade. Watch for confirmation."
    elif combined >= 0.15 and tech_score >= 4 and smart_allow:
        verdict    = "BUY"
        confidence = "MEDIUM"
        action     = "Can enter. Monitor closely — not all signals agree."
    elif combined <= -0.30 or (tech_score <= 2 and combined < 0):
        verdict    = "SELL"
        confidence = "HIGH"   if bear_count >= 3 else "MEDIUM"
        action     = "Exit or avoid. Multiple signals bearish." if bear_count >= 3 else "Exit position. Trend weakening."
    elif combined <= -0.15 or not smart_allow:
        verdict    = "SELL"
        confidence = "MEDIUM"
        action     = "Avoid new entry. Wait for better setup."
    else:
        verdict    = "HOLD"
        confidence = "NEUTRAL"
        action     = "No clear signal. Wait and watch."

    score = tech_score   # keep for backward compat (6-point display)

    # Convert all numpy/pandas types to Python native for JSON serialization
    import numpy as np, json as _json

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_)):    return bool(obj)
        if isinstance(obj, (np.integer)):  return int(obj)
        if isinstance(obj, (np.floating)): return float(obj)
        if isinstance(obj, float) and (obj != obj):  return None  # NaN
        return obj

    return _sanitize({
        "symbol":        symbol,
        "price":         round(price, 2),
        "vwap":          round(vwap, 2),
        "ema20":         round(ema20, 2),
        "ema50":         round(ema50, 2),
        "rsi":           round(rsi, 1),
        "vol_ratio":     round(volume / (avg_vol + 1), 2),
        "day_change":    round(day_chg * 100, 2),
        "signal":        result.signal.value,
        "strength":      round(result.strength, 3),
        "reason":        result.reason,
        "verdict":       verdict,
        "confidence":    confidence,
        "action":        action,
        "score":         score,
        "combined_score":combined,
        "score_breakdown": {
            "technical":  round(tech_norm, 3),
            "smart":      round(smart_score, 3),
            "earnings":   round(earnings_score, 3),
            "combined":   combined,
        },
        "target":        round(_risk.target_price(price), 2),
        "stop_loss":     round(_risk.stop_loss_price(price), 2),
        "checks":        checks,
        "price_history": price_history,
        "quote": {
            "open":       quote.get("open", 0),
            "high":       quote.get("high", 0),
            "low":        quote.get("low", 0),
            "change_pct": round(day_chg * 100, 2),
            "week52_high": week52_high or quote.get("week52_high", 0),
            "week52_low":  week52_low  or quote.get("week52_low",  0),
            "sector":      quote.get("sector", ""),
        },
        "timestamp": now_ist().isoformat(),
        "smart_signals": smart,
        "earnings": earnings,
    })


# ── Routes ────────────────────────────────────────────────────────

@app.route("/api/live")
def live():
    try:
        store     = get_store()
        positions = store.get_positions()
        cash      = store.get_cash()
        daily_pnl = store.get_daily_pnl()
        nav       = cash + sum(
            (_broker.get_ltp(s) or p.get("average_price", 0)) * abs(p.get("qty", 0))
            for s, p in positions.items()
        )
        # Build today's trade history with entry/exit/P&L
        all_orders  = store.get_all_orders()
        today       = now_ist().strftime("%Y-%m-%d")
        today_orders= [o for o in all_orders
                       if o.get("timestamp","").startswith(today)
                       and o["status"]=="COMPLETE"]

        # Match BUY→SELL pairs
        buys_map = {}
        trade_history = []
        for o in sorted(today_orders, key=lambda x: x.get("timestamp","")):
            if o["action"] == "BUY":
                buys_map[o["symbol"]] = o
            elif o["action"] == "SELL" and o["symbol"] in buys_map:
                b   = buys_map.pop(o["symbol"])
                pnl = round((o["price"] - b["price"]) * o["quantity"], 2)
                trade_history.append({
                    "symbol":     o["symbol"],
                    "qty":        o["quantity"],
                    "entry":      round(b["price"], 2),
                    "exit":       round(o["price"], 2),
                    "pnl":        pnl,
                    "pnl_pct":    round((o["price"]-b["price"])/b["price"]*100, 2),
                    "entry_time": b.get("timestamp","")[-8:-3],
                    "exit_time":  o.get("timestamp","")[-8:-3],
                    "status":     "WIN" if pnl > 0 else "LOSS",
                })

        # Still open positions count as open trades
        open_trades = []
        for sym, pos in positions.items():
            if sym in buys_map:
                b   = buys_map[sym]
                ltp = (_broker.get_ltp(sym) or pos.get("average_price",0))
                unreal = round((ltp - b["price"]) * abs(pos.get("qty",0)), 2)
                open_trades.append({
                    "symbol":     sym,
                    "qty":        abs(pos.get("qty",0)),
                    "entry":      round(b["price"],2),
                    "exit":       None,
                    "pnl":        unreal,
                    "pnl_pct":    round((ltp-b["price"])/b["price"]*100,2) if b["price"] else 0,
                    "entry_time": b.get("timestamp","")[-8:-3],
                    "exit_time":  "OPEN",
                    "status":     "OPEN",
                })

        nifty = _nifty_breadth()
        return jsonify({
            "nav":           round(nav, 2),
            "cash":          round(cash, 2),
            "daily_pnl":     round(daily_pnl, 2),
            "trades_today":  store.get_trades_today(),
            "positions":     _enrich_positions(positions),
            "signal_log":    store.get_signals_today(),
            "nav_timeline":  store.get_nav_timeline(),
            "trade_history": open_trades + trade_history,
            "market_open":   is_market_open(),
            "nifty":         nifty,
            "timestamp":     now_ist().isoformat(),
        })
    except Exception as exc:
        logger.error(f"/api/live: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/pnl")
def pnl():
    try:
        store   = get_store()
        orders  = store.get_all_orders()
        daily   = _pnl_by_day(orders, days=30)
        weekly  = _pnl_by_week(daily)
        monthly = _pnl_by_month(daily)
        all_pnl = [d["pnl"] for d in daily]
        wins    = [p for p in all_pnl if p > 0]
        losses  = [p for p in all_pnl if p < 0]
        avg_win = round(sum(wins)/len(wins) if wins else 0, 2)
        avg_loss= round(sum(losses)/len(losses) if losses else 0, 2)
        active  = [p for p in all_pnl if p != 0]
        win_rate= round(len(wins)/len(active)*100 if active else 0, 1)
        return jsonify({
            "daily": daily, "weekly": weekly, "monthly": monthly,
            "summary": {
                "total_pnl": round(sum(all_pnl), 2),
                "win_rate":  win_rate,
                "avg_win":   avg_win,
                "avg_loss":  avg_loss,
                "best_day":  round(max(all_pnl) if all_pnl else 0, 2),
                "worst_day": round(min(all_pnl) if all_pnl else 0, 2),
                "rr_ratio":  round(avg_win/abs(avg_loss) if avg_loss else 0, 2),
            },
        })
    except Exception as exc:
        logger.error(f"/api/pnl: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyse/<symbol>")
def analyse(symbol):
    try:
        return jsonify(_analyse_symbol(symbol))
    except Exception as exc:
        logger.error(f"/api/analyse/{symbol}: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status")
def status():
    store = get_store()
    return jsonify({
        "market_open": is_market_open(),
        "store_mode":  store.mode,
        "bot_running": len(store.get_signals_today()) > 0,
        "timestamp":   now_ist().isoformat(),
    })


@app.route("/api/watchlist")
def watchlist():
    return jsonify(get_store().get_watchlist())



@app.route("/api/market")
def market():
    """Real market context: global sentiment + Nifty trend + FII."""
    result = {
        "global":  {},
        "nifty":   {},
        "fii":     {},
        "timestamp": now_ist().isoformat(),
    }
    try:
        from core.global_sentiment import GlobalSentimentFilter
        g = GlobalSentimentFilter().analyse()
        result["global"] = {
            "mood":       g.mood.value,
            "size_mult":  g.size_multiplier,
            "bull_count": g.bull_count,
            "bear_count": g.bear_count,
            "signals": [
                {
                    "name":       s.name,
                    "price":      s.last_price,
                    "change_pct": s.change_pct,
                    "signal":     s.signal,
                    "reason":     s.reason,
                }
                for s in g.signals
            ],
        }
    except Exception as exc:
        logger.debug(f"/api/market global: {exc}")

    try:
        from core.nifty_trend import NiftyTrendFilter
        n = NiftyTrendFilter().analyse()
        result["nifty"] = {
            "regime":      n.regime.value,
            "allow_entry": n.allow_entry,
            "size_mult":   n.size_multiplier,
            "price":       n.nifty_price,
            "ema20":       n.ema20,
            "ema50":       n.ema50,
            "ema200":      n.ema200,
            "adx":         getattr(n, "adx", 0),
        }
    except Exception as exc:
        logger.debug(f"/api/market nifty: {exc}")

    try:
        from core.fii_dii import FIIDIIFilter
        f = FIIDIIFilter().analyse()
        result["fii"] = {
            "mood":        f.mood.value,
            "allow_entry": f.allow_entry,
            "size_mult":   f.size_multiplier,
        }
    except Exception as exc:
        logger.debug(f"/api/market fii: {exc}")

    return jsonify(result)


@app.route("/")
def health():
    return jsonify({"status": "ok", "service": "AlgoTrader India API"})


if __name__ == "__main__":
    logger.info(f"🚀 API server on port {PORT}  store={get_store().mode}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
