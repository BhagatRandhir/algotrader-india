"""
core/paper_broker.py  –  Paper Trading Broker (no Zerodha account needed)

Drop-in replacement for core/broker.py.
- Fetches REAL NSE data via yfinance (free, no login)
- Simulates order execution at real market prices
- Tracks virtual portfolio, P&L, positions in memory
- Saves trade log to paper_trades.json after every order
- All the same method signatures as ZerodhaBroker — swap seamlessly

Usage:
    Set EXECUTION_MODE=paper in .env  (already the default)
    Run: python paper_main.py
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from utils.yf_helpers import flatten_yf_columns
from core.store import get_store
from core.nse_data import get_client as get_nse
from utils.nse_fetcher import fetch_stooq, fetch_nse_ltp, make_synthetic_bars

console = Console()

TRADE_LOG_FILE = Path("paper_trades.json")


@dataclass
class PaperOrder:
    order_id:    str
    symbol:      str
    action:      str        # BUY / SELL
    quantity:    int
    order_type:  str        # LIMIT / MARKET / SL
    price:       float      # execution price
    timestamp:   str
    product:     str = "MIS"
    status:      str = "COMPLETE"
    sl_trigger:  Optional[float] = None


@dataclass
class PaperPosition:
    symbol:        str
    qty:           int
    average_price: float
    product:       str = "MIS"

    @property
    def pnl(self) -> float:
        # Updated externally with LTP
        return 0.0


class PaperBroker:
    """
    Simulates a live broker using real yfinance data.
    No Zerodha account or API key required.
    """

    def __init__(self, initial_capital: float = 500_000.0, reset: bool = False):
        """
        reset=False (default): if paper_trades.json exists, RESUME that saved
            state (cash/positions/orders), ignoring `initial_capital`. This is
            intentional — across-session continuity is the whole point of the
            file. A warning is logged whenever the saved cash differs from
            the requested initial_capital, so a mismatch is never silent.
        reset=True: ignore any saved state and start completely fresh with
            `initial_capital`, overwriting paper_trades.json. Use this when
            you deliberately want a clean slate (e.g. new capital amount,
            new testing run).
        """
        self.initial_capital = initial_capital
        self._cash           = initial_capital
        self._positions:  dict[str, PaperPosition] = {}
        self._orders:     list[PaperOrder] = []
        self._ltp_cache:  dict[str, float] = {}

        if reset:
            if TRADE_LOG_FILE.exists():
                logger.warning(
                    f"📄 reset=True — ignoring existing {TRADE_LOG_FILE} "
                    f"and starting fresh with ₹{initial_capital:,.0f}"
                )
            self._save_state()
        else:
            self._load_state()
            if TRADE_LOG_FILE.exists() and abs(self._cash - initial_capital) > 0.01 and not self._positions and len(self._orders) == 0:
                # Loaded a stale/empty file whose cash differs from requested capital
                logger.warning(
                    f"📄 Resuming saved state with cash=₹{self._cash:,.0f}, "
                    f"which differs from requested initial_capital=₹{initial_capital:,.0f}. "
                    f"Pass reset=True to {self.__class__.__name__}() to start fresh instead."
                )

        logger.success(
            f"📄 Paper broker ready  cash=₹{self._cash:,.0f}  "
            f"(no Zerodha account needed)"
        )

    # ── Persistence ───────────────────────────────────────────────

    def _save_state(self):
        state = {
            "cash":      self._cash,
            "positions": {
                sym: {"symbol": p.symbol, "qty": p.qty,
                      "average_price": p.average_price, "product": p.product}
                for sym, p in self._positions.items()
            },
            "orders": [asdict(o) for o in self._orders],
            "saved_at": datetime.now().isoformat(),
        }
        try:
            get_store().save_trade_state(state)
        except Exception as exc:
            logger.warning(f"Store save: {exc}")
            TRADE_LOG_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        try:
            state = get_store().get_trade_state()
            if not state and TRADE_LOG_FILE.exists():
                state = json.loads(TRADE_LOG_FILE.read_text())
            if not state:
                return
            self._cash = state.get("cash", self.initial_capital)
            for sym, p in state.get("positions", {}).items():
                self._positions[sym] = PaperPosition(**p)
            for o in state.get("orders", []):
                self._orders.append(PaperOrder(**o))
            logger.info(
                f"📄 Restored paper state: cash=₹{self._cash:,.0f}  "
                f"positions={list(self._positions.keys())}"
            )
        except Exception as exc:
            logger.warning(f"Could not restore paper state: {exc}")

    # ── Market Data (real yfinance) ───────────────────────────────

    def get_bars(
        self,
        symbol:   str,
        interval: str = "5m",
        period:   str = "5d",
        exchange: str = "NSE",
        retries:  int = 2,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.
        Priority: NSE India API → yfinance → synthetic bars (last resort)
        """
        # ── 1. NSE India API (primary) ────────────────────────────
        try:
            nse = get_nse()
            df  = nse.get_bars(symbol, interval=interval, period=period)
            if df is not None and len(df) >= 10:
                logger.debug(f"{symbol}: {len(df)} bars from NSE API")
                return df
            logger.debug(f"{symbol}: NSE API returned {len(df) if df is not None else 0} bars")
        except Exception as exc:
            logger.debug(f"{symbol}: NSE API error: {exc}")

        # ── 2. yfinance fallback (Ticker API — more reliable than download) ─
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        ticker = f"{symbol}{suffix}"
        for attempt in range(1, retries + 1):
            try:
                import yfinance as yf
                tkr = yf.Ticker(ticker)
                df  = tkr.history(period=period, interval=interval,
                                  auto_adjust=True)
                if df is None or df.empty:
                    raise ValueError("empty")
                df.columns = [c.lower() for c in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
                df = df[["open","high","low","close","volume"]].dropna()
                if len(df) >= 10:
                    logger.debug(f"{symbol}: {len(df)} bars from yfinance Ticker")
                    return df
                raise ValueError(f"only {len(df)} bars")
            except Exception as exc:
                if attempt < retries:
                    time.sleep(1.5)
                else:
                    logger.debug(f"{symbol}: yfinance failed: {exc}")

        # ── 3. Synthetic bars (last resort) ───────────────────────
        from utils.nse_fetcher import make_synthetic_bars
        cached_px = self._ltp_cache.get(symbol, 1000.0)
        logger.warning(f"{symbol}: all sources failed — using synthetic bars")
        return make_synthetic_bars(symbol, base_price=cached_px)


    def get_ltp(self, symbol: str, exchange: str = "NSE", retries: int = 2) -> Optional[float]:
        """
        Fetch latest price.
        Priority: NSE India API → yfinance → cache
        """
        # ── 1. NSE India API ──────────────────────────────────────
        try:
            nse   = get_nse()
            price = nse.get_ltp(symbol)
            if price and price > 0:
                self._ltp_cache[symbol] = price
                return round(price, 2)
        except Exception as exc:
            logger.debug(f"{symbol}: NSE LTP error: {exc}")

        # ── 2. yfinance fallback (Ticker API) ────────────────────
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        ticker = f"{symbol}{suffix}"
        for attempt in range(1, retries + 1):
            try:
                import yfinance as yf
                tkr   = yf.Ticker(ticker)
                info  = tkr.fast_info
                price = float(info.last_price or 0)
                if price > 0:
                    self._ltp_cache[symbol] = price
                    return round(price, 2)
                # fallback to history
                df = tkr.history(period="1d", interval="1m", auto_adjust=True)
                if df is not None and not df.empty:
                    price = float(df["Close"].dropna().iloc[-1])
                    self._ltp_cache[symbol] = price
                    return round(price, 2)
            except Exception:
                if attempt < retries:
                    time.sleep(1)

        # ── 3. Cached price ───────────────────────────────────────
        cached = self._ltp_cache.get(symbol)
        if cached:
            logger.debug(f"{symbol}: using cached LTP ₹{cached:.2f}")
        return cached


    # ── Account ───────────────────────────────────────────────────

    def get_available_cash(self) -> float:
        return self._cash

    def get_portfolio_value(self) -> float:
        pos_value = 0.0
        for sym, pos in self._positions.items():
            ltp = self._ltp_cache.get(sym, pos.average_price)
            pos_value += pos.qty * ltp
        return self._cash + pos_value

    def get_positions(self) -> dict[str, dict]:
        """
        Return open positions with real-time P&L.
        Uses LTP cache if available (populated by get_ltp/get_bars calls).
        Falls back to average_price only if no market data available at all.
        """
        result = {}
        for sym, pos in self._positions.items():
            ltp = self._ltp_cache.get(sym)
            if ltp is None:
                # Try a quick daily bar fetch to get at least today's close
                try:
                    df = yf.download(f"{sym}.NS", period="1d", interval="1d",
                                     progress=False, auto_adjust=True)
                    if df is not None and not df.empty:
                        df = flatten_yf_columns(df)
                        ltp = float(df["close"].dropna().iloc[-1])
                        self._ltp_cache[sym] = ltp
                except Exception:
                    pass
            # Final fallback — use avg price (P&L = 0, but no crash)
            ltp = ltp or pos.average_price
            pnl = pos.qty * (ltp - pos.average_price)
            result[sym] = {
                "qty":           pos.qty,
                "average_price": pos.average_price,
                "pnl":           round(pnl, 2),
                "product":       pos.product,
            }
        return result

    def get_daily_pnl(self) -> float:
        """
        Total P&L for today = unrealised (open positions) + realised (closed trades).
        Realised P&L is computed by matching each today's SELL to its corresponding
        BUY using a FIFO stack per symbol — handles multiple buy/sell cycles
        on the same symbol correctly.
        """
        # ── Unrealised P&L from open positions ────────────────────
        total = 0.0
        for sym, pos in self._positions.items():
            ltp = self._ltp_cache.get(sym, pos.average_price)
            total += pos.qty * (ltp - pos.average_price)

        # ── Realised P&L from today's closed trades (FIFO matching) ──
        today = datetime.now().strftime("%Y-%m-%d")
        today_orders = [o for o in self._orders if o.timestamp.startswith(today)]

        # Build per-symbol FIFO buy queues from today's orders
        buy_queues: dict[str, list[PaperOrder]] = {}
        for o in today_orders:
            if o.action == "BUY" and o.status == "COMPLETE":
                buy_queues.setdefault(o.symbol, []).append(o)

        for o in today_orders:
            if o.action == "SELL" and o.status == "COMPLETE":
                queue = buy_queues.get(o.symbol, [])
                if queue:
                    buy = queue.pop(0)   # FIFO — match oldest buy first
                    total += o.quantity * (o.price - buy.price)

        return round(total, 2)

    def _find_buy_price(self, symbol: str) -> float:
        """Find most recent completed BUY price for a symbol (legacy fallback)."""
        for order in reversed(self._orders):
            if order.symbol == symbol and order.action == "BUY" and order.status == "COMPLETE":
                return order.price
        return 0.0

    # ── Orders (simulated) ────────────────────────────────────────

    def place_order(
        self,
        symbol:      str,
        action:      str,
        quantity:    int,
        exchange:    str = "NSE",
        order_type:  str = "LIMIT",
        product:     str = "MIS",
        limit_price: Optional[float] = None,
    ) -> Optional[str]:

        # Hard validation — quantity must be a positive integer. This is a
        # money-safety guard: a zero or negative quantity must never reach
        # the cash/position math below, since a negative quantity would
        # silently INCREASE cash (cost = qty * price goes negative) and
        # create a phantom negative position.
        if not isinstance(quantity, int) or quantity <= 0:
            logger.error(
                f"Paper order rejected — invalid quantity {quantity!r} "
                f"for {action} {symbol} (must be a positive integer)"
            )
            return None

        if action not in ("BUY", "SELL"):
            logger.error(f"Paper order rejected — invalid action {action!r}")
            return None

        # Get execution price
        ltp = self.get_ltp(symbol) or limit_price
        if not ltp:
            logger.error(f"Paper order failed — no price for {symbol}")
            return None

        # Simulate slippage: 0.1% on market orders
        if order_type == "MARKET":
            exec_price = ltp * (1.001 if action == "BUY" else 0.999)
        else:
            exec_price = limit_price or ltp

        exec_price = round(exec_price, 2)
        order_id   = str(uuid.uuid4())[:8].upper()

        if action == "BUY":
            cost = quantity * exec_price
            if cost > self._cash:
                logger.warning(
                    f"Paper order rejected — insufficient cash "
                    f"(need ₹{cost:,.0f}, have ₹{self._cash:,.0f})"
                )
                return None
            self._cash -= cost
            if symbol in self._positions:
                # Average up/down
                pos = self._positions[symbol]
                total_qty   = pos.qty + quantity
                avg_price   = (pos.qty * pos.average_price + quantity * exec_price) / total_qty
                pos.qty           = total_qty
                pos.average_price = round(avg_price, 2)
            else:
                self._positions[symbol] = PaperPosition(
                    symbol=symbol, qty=quantity,
                    average_price=exec_price, product=product
                )
            self._ltp_cache[symbol] = exec_price

        elif action == "SELL":
            if symbol not in self._positions:
                logger.warning(f"Paper SELL rejected — no position in {symbol}")
                return None
            pos      = self._positions[symbol]
            sell_qty = min(quantity, pos.qty)
            if sell_qty < quantity:
                logger.warning(
                    f"Paper SELL {symbol}: requested {quantity} but only "
                    f"{pos.qty} held — selling {sell_qty} instead"
                )
            proceeds = sell_qty * exec_price
            pnl      = sell_qty * (exec_price - pos.average_price)
            self._cash += proceeds
            quantity = sell_qty   # record the ACTUAL executed quantity, not the request

            if sell_qty >= pos.qty:
                del self._positions[symbol]
            else:
                pos.qty -= sell_qty

            logger.info(
                f"📄 Paper SELL {symbol}  qty={sell_qty}  "
                f"@₹{exec_price}  P&L=₹{pnl:+,.0f}"
            )

        order = PaperOrder(
            order_id=order_id, symbol=symbol, action=action,
            quantity=quantity, order_type=order_type,
            price=exec_price, product=product,
            timestamp=datetime.now().isoformat(),
        )
        self._orders.append(order)
        self._save_state()

        logger.success(
            f"📄 Paper ORDER  {action} {quantity}x {symbol}  "
            f"@₹{exec_price}  id={order_id}  cash=₹{self._cash:,.0f}"
        )
        return order_id

    def place_sl_order(
        self,
        symbol:        str,
        action:        str,
        quantity:      int,
        trigger_price: float,
        limit_price:   float,
        exchange:      str = "NSE",
        product:       str = "MIS",
    ) -> Optional[str]:
        """Paper SL order — logged but checked in main loop via LTP."""
        if not isinstance(quantity, int) or quantity <= 0:
            logger.error(
                f"Paper SL order rejected — invalid quantity {quantity!r} "
                f"for {action} {symbol}"
            )
            return None

        order_id = str(uuid.uuid4())[:8].upper()
        order = PaperOrder(
            order_id=order_id, symbol=symbol, action=action,
            quantity=quantity, order_type="SL",
            price=limit_price, product=product,
            timestamp=datetime.now().isoformat(),
            sl_trigger=trigger_price, status="PENDING",
        )
        self._orders.append(order)
        self._save_state()
        logger.info(
            f"📄 Paper SL  {action} {quantity}x {symbol}  "
            f"trigger=₹{trigger_price}  limit=₹{limit_price}  id={order_id}"
        )
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status == "PENDING":
                o.status = "CANCELLED"
                self._save_state()
                logger.info(f"📄 Paper order cancelled: {order_id}")
                return True
        return False

    def get_open_orders(self) -> list[dict]:
        return [
            {"order_id": o.order_id, "symbol": o.symbol,
             "status": o.status, "action": o.action}
            for o in self._orders if o.status == "PENDING"
        ]

    def cancel_all_orders(self):
        for o in self._orders:
            if o.status == "PENDING":
                o.status = "CANCELLED"
        self._save_state()
        logger.warning("📄 All paper orders cancelled")

    # ── Portfolio summary ─────────────────────────────────────────

    def print_portfolio(self):
        tbl = Table(
            title=f"📄 Paper Portfolio — {datetime.now().strftime('%d %b %Y %H:%M')}",
            box=box.SIMPLE_HEAVY, title_style="bold cyan",
        )
        tbl.add_column("Symbol",    style="bold white")
        tbl.add_column("Qty",       justify="right")
        tbl.add_column("Avg Price", justify="right")
        tbl.add_column("LTP",       justify="right")
        tbl.add_column("P&L",       justify="right")
        tbl.add_column("P&L %",     justify="right")

        total_pnl = 0.0
        for sym, pos in self._positions.items():
            ltp  = self._ltp_cache.get(sym, pos.average_price)
            pnl  = pos.qty * (ltp - pos.average_price)
            pct  = (ltp - pos.average_price) / pos.average_price * 100
            total_pnl += pnl
            c = "green" if pnl >= 0 else "red"
            tbl.add_row(
                sym, str(pos.qty),
                f"₹{pos.average_price:,.2f}",
                f"₹{ltp:,.2f}",
                f"[{c}]₹{pnl:+,.0f}[/]",
                f"[{c}]{pct:+.2f}%[/]",
            )

        console.print(tbl)
        nav = self.get_portfolio_value()
        total_return = (nav - self.initial_capital) / self.initial_capital * 100
        c = "green" if total_pnl >= 0 else "red"
        console.print(
            f"  Cash: [cyan]₹{self._cash:,.0f}[/]  │  "
            f"NAV: [cyan]₹{nav:,.0f}[/]  │  "
            f"Total P&L: [{c}]₹{total_pnl:+,.0f}  ({total_return:+.2f}%)[/]\n"
        )

    def print_trade_history(self, last_n: int = 20):
        tbl = Table(
            title=f"📄 Trade History (last {last_n})",
            box=box.SIMPLE_HEAVY, title_style="bold yellow",
        )
        tbl.add_column("Time",   style="dim", width=20)
        tbl.add_column("Symbol", style="bold white", width=12)
        tbl.add_column("Action", width=6)
        tbl.add_column("Qty",    justify="right", width=6)
        tbl.add_column("Price",  justify="right", width=10)
        tbl.add_column("Value",  justify="right", width=12)

        for o in self._orders[-last_n:]:
            c = "green" if o.action == "BUY" else "red"
            tbl.add_row(
                o.timestamp[:16],
                o.symbol,
                f"[{c}]{o.action}[/]",
                str(o.quantity),
                f"₹{o.price:,.2f}",
                f"₹{o.quantity*o.price:,.0f}",
            )
        console.print(tbl)
