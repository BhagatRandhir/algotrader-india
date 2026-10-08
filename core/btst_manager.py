"""
core/btst_manager.py  –  BTST Position Manager

Manages the full BTST lifecycle:
  - Scans for BTST candidates at 2:30–3:10 PM IST
  - Buys and holds overnight (CNC product type)
  - Sells next morning at 9:15–10:00 AM IST
  - Sends WhatsApp alerts on buy/sell
  - Saves BTST positions to btst_positions.json

Completely separate from intraday bot — different:
  - Entry window (2:30–3:10 PM vs 10:15 AM–2:30 PM)
  - Product type (CNC vs MIS)
  - Position size (3% vs 6%)
  - Exit logic (next morning vs same day)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pytz
from loguru import logger

from strategies.btst_strategy import BTSTStrategy, BTSTSignal
from core.risk import RiskConfig, RiskManager

IST             = pytz.timezone("Asia/Kolkata")
BTST_FILE       = Path("btst_positions.json")
MAX_BTST_POS    = int(__import__("os").getenv("BTST_MAX_POSITIONS", "2"))
BTST_SIZE_PCT   = float(__import__("os").getenv("BTST_POSITION_PCT", "0.03"))
BTST_SL_PCT     = float(__import__("os").getenv("BTST_SL_PCT",       "0.020"))
BTST_VIX_MAX    = float(__import__("os").getenv("BTST_VIX_MAX",      "16.0"))

ENTRY_START_H, ENTRY_START_M = 14, 30   # 2:30 PM
ENTRY_END_H,   ENTRY_END_M   = 15, 10   # 3:10 PM
EXIT_START_H,  EXIT_START_M  = 9,  15   # 9:15 AM
EXIT_FORCE_H,  EXIT_FORCE_M  = 10, 0    # 10:00 AM force exit


@dataclass
class BTSTPosition:
    symbol:      str
    qty:         int
    entry_price: float
    sl_price:    float
    entry_date:  str
    entry_time:  str
    strength:    float
    reason:      str
    product:     str = "CNC"


class BTSTManager:

    def __init__(self, broker, notifier=None):
        self.broker    = broker
        self.notifier  = notifier
        self.strategy  = BTSTStrategy()
        self.risk      = RiskManager(RiskConfig(
            max_position_pct   = BTST_SIZE_PCT,
            stop_loss_pct      = BTST_SL_PCT,
            target_pct         = 0.03,
            max_open_positions = MAX_BTST_POS,
        ))
        self._positions: dict[str, BTSTPosition] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────

    def _save(self):
        try:
            BTST_FILE.write_text(json.dumps(
                {sym: asdict(pos) for sym, pos in self._positions.items()},
                indent=2,
            ))
        except Exception as exc:
            logger.warning(f"BTST save: {exc}")

    def _load(self):
        if not BTST_FILE.exists():
            return
        try:
            data = json.loads(BTST_FILE.read_text())
            for sym, d in data.items():
                self._positions[sym] = BTSTPosition(**d)
            if self._positions:
                logger.info(f"📋 BTST: loaded {len(self._positions)} overnight positions: "
                            f"{list(self._positions.keys())}")
        except Exception as exc:
            logger.warning(f"BTST load: {exc}")

    # ── Time helpers ──────────────────────────────────────────────

    def _in_entry_window(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=ENTRY_START_H, minute=ENTRY_START_M, second=0)
        end   = now.replace(hour=ENTRY_END_H,   minute=ENTRY_END_M,   second=0)
        return start <= now <= end

    def _in_exit_window(self) -> bool:
        now = datetime.now(IST)
        if now.weekday() >= 5:
            return False
        start = now.replace(hour=EXIT_START_H, minute=EXIT_START_M, second=0)
        force = now.replace(hour=EXIT_FORCE_H, minute=EXIT_FORCE_M, second=0)
        return start <= now <= force

    def _is_force_exit_time(self) -> bool:
        now = datetime.now(IST)
        return (now.hour == EXIT_FORCE_H and now.minute >= EXIT_FORCE_M)

    def _is_next_day_position(self, pos: BTSTPosition) -> bool:
        return pos.entry_date != date.today().isoformat()

    def _vix_ok(self) -> bool:
        try:
            import yfinance as yf
            df = yf.Ticker("^INDIAVIX").history(period="2d", interval="1d")
            if not df.empty:
                vix = float(df["Close"].iloc[-1])
                if vix > BTST_VIX_MAX:
                    logger.info(f"BTST blocked — VIX={vix:.1f} > {BTST_VIX_MAX}")
                    return False
        except Exception:
            pass
        return True

    # ── Entry ─────────────────────────────────────────────────────

    def scan_and_buy(self, watchlist: list[str], cash: float):
        """
        Called during 2:30–3:10 PM window.
        Scans watchlist for BTST candidates and buys the best ones.
        """
        if not self._in_entry_window():
            return
        if len(self._positions) >= MAX_BTST_POS:
            return
        if not self._vix_ok():
            return

        candidates = []
        for symbol in watchlist:
            if symbol in self._positions:
                continue
            try:
                df = self.broker.get_bars(symbol, interval="5m", period="5d")
                if df is None or len(df) < 60:
                    continue
                result = self.strategy.generate_signal(df, symbol)
                if result.signal == BTSTSignal.BUY:
                    candidates.append((symbol, result, df))
            except Exception as exc:
                logger.debug(f"BTST scan {symbol}: {exc}")

        if not candidates:
            return

        # Sort by strength — take top candidates up to max positions
        candidates.sort(key=lambda x: x[1].strength, reverse=True)
        slots = MAX_BTST_POS - len(self._positions)

        for symbol, result, df in candidates[:slots]:
            ltp = self.broker.get_ltp(symbol)
            if not ltp:
                continue

            sl  = self.risk.stop_loss_price(ltp)
            qty = self.risk.position_size(cash, ltp, result.strength)
            if qty <= 0:
                continue

            oid = self.broker.place_order(symbol, "BUY", qty, "MARKET", "CNC")
            if oid:
                pos = BTSTPosition(
                    symbol      = symbol,
                    qty         = qty,
                    entry_price = ltp,
                    sl_price    = sl,
                    entry_date  = date.today().isoformat(),
                    entry_time  = datetime.now(IST).strftime("%H:%M"),
                    strength    = result.strength,
                    reason      = result.reason,
                )
                self._positions[symbol] = pos
                self._save()

                logger.info(
                    f"🌙 BTST BUY {symbol} qty={qty} @₹{ltp:.2f} "
                    f"SL=₹{sl:.2f} strength={result.strength:.2f}"
                )

                if self.notifier:
                    self.notifier.send(
                        f"🌙 *BTST BUY {symbol}*\n"
                        f"Qty: {qty} shares\n"
                        f"Entry: ₹{ltp:.2f}\n"
                        f"Stop Loss: ₹{sl:.2f} (-{BTST_SL_PCT*100:.0f}%)\n"
                        f"Hold overnight → sell tomorrow 9:15 AM\n"
                        f"Strength: {result.strength:.0%}\n"
                        f"⏰ {datetime.now(IST).strftime('%H:%M IST')}"
                    )

    # ── Exit ──────────────────────────────────────────────────────

    def manage_exits(self):
        """
        Called every loop.
        Sells BTST positions that were bought yesterday, in exit window.
        Also checks SL breach.
        """
        if not self._positions:
            return

        for symbol, pos in list(self._positions.items()):

            ltp = self.broker.get_ltp(symbol)
            if not ltp:
                continue

            # ── SL check (any time) ───────────────────────────────
            if ltp <= pos.sl_price:
                self._sell(symbol, pos, ltp, "🛑 BTST SL hit")
                continue

            # ── Next morning exit window ──────────────────────────
            if self._is_next_day_position(pos):
                if self._in_exit_window() or self._is_force_exit_time():
                    reason = (
                        "⏰ BTST force exit 10:00 AM"
                        if self._is_force_exit_time()
                        else "🌅 BTST morning exit"
                    )
                    self._sell(symbol, pos, ltp, reason)

    def _sell(self, symbol: str, pos: BTSTPosition, ltp: float, reason: str):
        qty = pos.qty
        oid = self.broker.place_order(symbol, "SELL", qty, "MARKET", "CNC")
        if oid:
            pnl     = (ltp - pos.entry_price) * qty
            pnl_pct = (ltp - pos.entry_price) / pos.entry_price * 100
            icon    = "💰" if pnl >= 0 else "🔴"
            logger.info(
                f"{icon} BTST SELL {symbol} qty={qty} @₹{ltp:.2f} "
                f"P&L=₹{pnl:+,.0f} ({pnl_pct:+.2f}%) [{reason}]"
            )
            if self.notifier:
                self.notifier.send(
                    f"{icon} *BTST SELL {symbol}*\n"
                    f"Qty: {qty} shares\n"
                    f"Entry: ₹{pos.entry_price:.2f} → Exit: ₹{ltp:.2f}\n"
                    f"P&L: {'+'if pnl>=0 else ''}₹{pnl:.0f} ({pnl_pct:+.2f}%)\n"
                    f"Reason: {reason}\n"
                    f"⏰ {datetime.now(IST).strftime('%H:%M IST')}"
                )
            del self._positions[symbol]
            self._save()

    def get_positions(self) -> dict[str, BTSTPosition]:
        return dict(self._positions)

    def summary(self) -> str:
        if not self._positions:
            return "No BTST positions"
        return f"{len(self._positions)} BTST: " + ", ".join(
            f"{s}@₹{p.entry_price:.0f}" for s, p in self._positions.items()
        )
