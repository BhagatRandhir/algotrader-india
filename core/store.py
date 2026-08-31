"""
core/store.py  –  Shared State Store

Abstracts storage so the bot and API server can share data.
Works in two modes:

  REDIS mode  (production on Render):
    Set REDIS_URL env var → uses Redis for all shared state.
    Both bot and API server connect to same Redis instance.

  FILE mode   (local development):
    No REDIS_URL → reads/writes paper_trades.json + signal_log.json
    Same behaviour as before.

Usage:
    from core.store import Store
    store = Store()

    store.set_nav_snapshot({"time":"10:30","nav":502000})
    store.append_signal({"symbol":"RELIANCE","signal":"BUY",...})
    store.save_trade_state({"cash":490000,"orders":[...],...})

    data = store.get_live_data()
    pnl  = store.get_all_trades()
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

from loguru import logger

REDIS_URL        = os.getenv("REDIS_URL")
TRADE_LOG_FILE   = Path("paper_trades.json")
SIGNAL_LOG_FILE  = Path("signal_log.json")

# Redis key names
R_TRADES    = "algo:trades"
R_SIGNALS   = "algo:signals"
R_NAV       = "algo:nav"
R_POSITIONS = "algo:positions"
R_CASH      = "algo:cash"
R_WATCHLIST = "algo:watchlist"
R_DAILY_PNL = "algo:daily_pnl"
R_TRADES_TODAY = "algo:trades_today"


class Store:

    def __init__(self):
        self._redis = None
        if REDIS_URL:
            try:
                import redis
                self._redis = redis.from_url(REDIS_URL, decode_responses=True)
                self._redis.ping()
                logger.info("✅ Redis store connected")
            except Exception as exc:
                logger.warning(f"Redis connect failed: {exc} — falling back to file mode")
                self._redis = None
        else:
            logger.info("📁 File store mode (no REDIS_URL set)")

    @property
    def mode(self) -> str:
        return "redis" if self._redis else "file"

    # ── Write helpers ─────────────────────────────────────────────

    def save_trade_state(self, state: dict):
        """Save full broker state (cash, positions, orders)."""
        if self._redis:
            try:
                self._redis.set(R_TRADES, json.dumps(state))
                self._redis.set(R_CASH, str(state.get("cash", 0)))
                pos = state.get("positions", {})
                self._redis.set(R_POSITIONS, json.dumps(pos))
            except Exception as exc:
                logger.warning(f"Redis save_trade_state: {exc}")
        else:
            TRADE_LOG_FILE.write_text(json.dumps(state, indent=2))

    def append_signal(self, signal: dict):
        """Append one signal entry to the signal log."""
        signal.setdefault("date", date.today().isoformat())
        if self._redis:
            try:
                raw  = self._redis.get(R_SIGNALS) or "[]"
                data = json.loads(raw)
                data.append(signal)
                data = data[-500:]   # keep last 500
                self._redis.set(R_SIGNALS, json.dumps(data))
            except Exception as exc:
                logger.warning(f"Redis append_signal: {exc}")
        else:
            data = []
            if SIGNAL_LOG_FILE.exists():
                try: data = json.loads(SIGNAL_LOG_FILE.read_text())
                except: pass
            data.append(signal)
            SIGNAL_LOG_FILE.write_text(json.dumps(data[-500:], indent=2))

    def save_nav_snapshot(self, snapshot: dict):
        """Append a NAV snapshot for the intraday chart."""
        snapshot.setdefault("date", date.today().isoformat())
        snapshot.setdefault("type", "nav_snapshot")
        key = R_NAV
        if self._redis:
            try:
                raw  = self._redis.get(key) or "[]"
                data = json.loads(raw)
                data.append(snapshot)
                data = data[-200:]
                self._redis.set(key, json.dumps(data))
                self._redis.set(R_DAILY_PNL, str(snapshot.get("pnl", 0)))
            except Exception as exc:
                logger.warning(f"Redis save_nav_snapshot: {exc}")
        else:
            data = []
            if SIGNAL_LOG_FILE.exists():
                try:
                    existing = json.loads(SIGNAL_LOG_FILE.read_text())
                    data = [x for x in existing if x.get("type") != "nav_snapshot"]
                except: pass
            data.append(snapshot)
            SIGNAL_LOG_FILE.write_text(json.dumps(data[-500:], indent=2))

    def save_watchlist(self, watchlist_data: dict):
        if self._redis:
            try: self._redis.set(R_WATCHLIST, json.dumps(watchlist_data))
            except Exception as exc: logger.warning(f"Redis watchlist: {exc}")
        else:
            Path("watchlist.json").write_text(json.dumps(watchlist_data, indent=2))

    def increment_trades_today(self):
        today = date.today().isoformat()
        if self._redis:
            try:
                key = f"{R_TRADES_TODAY}:{today}"
                self._redis.incr(key)
                self._redis.expire(key, 86400)
            except: pass

    # ── Read helpers ──────────────────────────────────────────────

    def get_trade_state(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.get(R_TRADES)
                return json.loads(raw) if raw else {}
            except: return {}
        if TRADE_LOG_FILE.exists():
            try: return json.loads(TRADE_LOG_FILE.read_text())
            except: return {}
        return {}

    def get_signals_today(self) -> list[dict]:
        today = date.today().isoformat()
        if self._redis:
            try:
                raw  = self._redis.get(R_SIGNALS) or "[]"
                data = json.loads(raw)
                return [s for s in data if s.get("date") == today][-50:]
            except: return []
        if SIGNAL_LOG_FILE.exists():
            try:
                data = json.loads(SIGNAL_LOG_FILE.read_text())
                return [s for s in data
                        if s.get("date") == today
                        and s.get("type") != "nav_snapshot"][-50:]
            except: return []
        return []

    def get_nav_timeline(self) -> list[dict]:
        today = date.today().isoformat()
        if self._redis:
            try:
                raw  = self._redis.get(R_NAV) or "[]"
                data = json.loads(raw)
                return [s for s in data if s.get("date") == today]
            except: return []
        if SIGNAL_LOG_FILE.exists():
            try:
                data = json.loads(SIGNAL_LOG_FILE.read_text())
                return [s for s in data
                        if s.get("type") == "nav_snapshot"
                        and s.get("date") == today]
            except: return []
        return []

    def get_all_orders(self) -> list[dict]:
        state = self.get_trade_state()
        return state.get("orders", [])

    def get_positions(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.get(R_POSITIONS)
                return json.loads(raw) if raw else {}
            except: return {}
        state = self.get_trade_state()
        return state.get("positions", {})

    def get_cash(self) -> float:
        if self._redis:
            try:
                val = self._redis.get(R_CASH)
                return float(val) if val else 500_000.0
            except: return 500_000.0
        state = self.get_trade_state()
        return state.get("cash", 500_000.0)

    def get_daily_pnl(self) -> float:
        if self._redis:
            try:
                val = self._redis.get(R_DAILY_PNL)
                return float(val) if val else 0.0
            except: return 0.0
        # Calculate from orders
        orders = self.get_all_orders()
        today  = date.today().isoformat()
        today_orders = [o for o in orders if o.get("timestamp","").startswith(today)]
        buys  = {o["symbol"]: o for o in today_orders if o["action"] == "BUY"}
        total = 0.0
        for o in today_orders:
            if o["action"] == "SELL" and o["symbol"] in buys:
                b = buys[o["symbol"]]
                total += (o["price"] - b["price"]) * o["quantity"]
        return round(total, 2)

    def get_trades_today(self) -> int:
        today = date.today().isoformat()
        if self._redis:
            try:
                val = self._redis.get(f"{R_TRADES_TODAY}:{today}")
                return int(val) if val else 0
            except: return 0
        orders = self.get_all_orders()
        return sum(1 for o in orders
                   if o.get("timestamp","").startswith(today)
                   and o["action"] == "BUY")

    def get_watchlist(self) -> dict:
        if self._redis:
            try:
                raw = self._redis.get(R_WATCHLIST)
                return json.loads(raw) if raw else {"symbols":[], "details":[]}
            except: return {"symbols":[], "details":[]}
        wl = Path("watchlist.json")
        if wl.exists():
            try: return json.loads(wl.read_text())
            except: pass
        return {"symbols":[], "details":[]}


# Module-level singleton
_store: Store | None = None

def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store
