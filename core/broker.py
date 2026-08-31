"""
core/broker.py  –  Zerodha Kite Connect wrapper.

Handles:
  - Session management
  - Market data via yfinance (free, no paid Kite subscription needed for data)
  - Live LTP via Kite quote() for order pricing
  - Order placement / cancellation
  - Portfolio & P&L queries
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import yfinance as yf
from kiteconnect import KiteConnect
from loguru import logger

from utils.yf_helpers import flatten_yf_columns


class ZerodhaBroker:

    def __init__(self, api_key: str, access_token: str):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        logger.success("Zerodha Kite session initialised.")

    # ── Market Data (yfinance – free, no subscription) ────────────

    def get_bars(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "5d",
        exchange: str = "NSE",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars via yfinance.
        NSE symbols → append .NS  |  BSE symbols → append .BO
        """
        suffix = ".NS" if exchange.upper() == "NSE" else ".BO"
        ticker = f"{symbol}{suffix}"
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                logger.warning(f"No yfinance data for {ticker}")
                return pd.DataFrame()
            df = flatten_yf_columns(df)
            df.index = pd.to_datetime(df.index)
            # Drop any timezone info to avoid pandas warnings
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as exc:
            logger.error(f"yfinance error for {ticker}: {exc}")
            return pd.DataFrame()

    # ── Live Pricing (Kite quote – always accurate for order sizing) ─

    def get_ltp(self, symbol: str, exchange: str = "NSE") -> Optional[float]:
        """Last traded price via Kite quote API."""
        instrument = f"{exchange.upper()}:{symbol}"
        try:
            quote = self.kite.quote([instrument])
            return quote[instrument]["last_price"]
        except Exception as exc:
            logger.error(f"LTP fetch failed for {instrument}: {exc}")
            return None

    # ── Account ───────────────────────────────────────────────────

    def get_margins(self) -> dict:
        """Return equity segment margins."""
        try:
            return self.kite.margins(segment="equity")
        except Exception as exc:
            logger.error(f"Margin fetch failed: {exc}")
            return {}

    def get_available_cash(self) -> float:
        margins = self.get_margins()
        return float(margins.get("net", 0))

    def get_positions(self) -> dict[str, dict]:
        """
        Returns {symbol: {qty, average_price, pnl, product}} for open positions.
        Uses net positions (day + carry-forward combined).
        """
        try:
            raw = self.kite.positions()
            result: dict[str, dict] = {}
            for pos in raw.get("net", []):
                if pos["quantity"] != 0:
                    result[pos["tradingsymbol"]] = {
                        "qty":           pos["quantity"],
                        "average_price": pos["average_price"],
                        "pnl":           pos["pnl"],
                        "product":       pos["product"],
                    }
            return result
        except Exception as exc:
            logger.error(f"Positions fetch failed: {exc}")
            return {}

    def get_daily_pnl(self) -> float:
        try:
            positions = self.kite.positions()
            return sum(p["pnl"] for p in positions.get("net", []))
        except Exception:
            return 0.0

    def get_portfolio_value(self) -> float:
        """Approximate NAV = cash + open position value."""
        try:
            cash = self.get_available_cash()
            positions = self.get_positions()
            position_value = sum(
                abs(p["qty"]) * p["average_price"]
                for p in positions.values()
            )
            return cash + position_value
        except Exception:
            return 0.0

    # ── Orders ────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        action: str,                   # "BUY" or "SELL"
        quantity: int,
        exchange: str = "NSE",
        order_type: str = "LIMIT",
        product: str = "MIS",
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """
        Place an order. Returns order_id on success, None on failure.
        order_type: LIMIT | MARKET
        product:    MIS (intraday) | CNC (delivery)
        """
        try:
            transaction_type = (
                self.kite.TRANSACTION_TYPE_BUY
                if action.upper() == "BUY"
                else self.kite.TRANSACTION_TYPE_SELL
            )
            kite_order_type = (
                self.kite.ORDER_TYPE_LIMIT
                if order_type.upper() == "LIMIT" and limit_price
                else self.kite.ORDER_TYPE_MARKET
            )
            kite_product = (
                self.kite.PRODUCT_MIS
                if product.upper() == "MIS"
                else self.kite.PRODUCT_CNC
            )
            kite_exchange = (
                self.kite.EXCHANGE_NSE
                if exchange.upper() == "NSE"
                else self.kite.EXCHANGE_BSE
            )

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=kite_exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=kite_order_type,
                product=kite_product,
                price=limit_price if kite_order_type == self.kite.ORDER_TYPE_LIMIT else None,
            )
            logger.success(
                f"✅ ORDER PLACED  {action} {quantity}x {exchange}:{symbol}  "
                f"type={order_type}  price=₹{limit_price or 'MKT'}  "
                f"order_id={order_id}"
            )
            return str(order_id)

        except Exception as exc:
            logger.error(f"Order failed {action} {symbol}: {exc}")
            return None

    def place_sl_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        trigger_price: float,
        limit_price: float,
        exchange: str = "NSE",
        product: str = "MIS",
    ) -> Optional[str]:
        """Place a stop-loss limit order (SL type)."""
        try:
            transaction_type = (
                self.kite.TRANSACTION_TYPE_BUY
                if action.upper() == "BUY"
                else self.kite.TRANSACTION_TYPE_SELL
            )
            kite_exchange = (
                self.kite.EXCHANGE_NSE
                if exchange.upper() == "NSE"
                else self.kite.EXCHANGE_BSE
            )
            kite_product = (
                self.kite.PRODUCT_MIS
                if product.upper() == "MIS"
                else self.kite.PRODUCT_CNC
            )

            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=kite_exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=self.kite.ORDER_TYPE_SL,
                product=kite_product,
                price=limit_price,
                trigger_price=trigger_price,
            )
            logger.info(
                f"SL order placed  {action} {quantity}x {symbol}  "
                f"trigger=₹{trigger_price}  limit=₹{limit_price}  id={order_id}"
            )
            return str(order_id)
        except Exception as exc:
            logger.error(f"SL order failed {symbol}: {exc}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR,
                                   order_id=order_id)
            logger.warning(f"Order cancelled: {order_id}")
            return True
        except Exception as exc:
            logger.error(f"Cancel failed {order_id}: {exc}")
            return False

    def get_open_orders(self) -> list[dict]:
        try:
            orders = self.kite.orders()
            return [o for o in orders if o["status"] in ("OPEN", "TRIGGER PENDING")]
        except Exception:
            return []

    def cancel_all_orders(self):
        for order in self.get_open_orders():
            self.cancel_order(str(order["order_id"]))
        logger.warning("All open orders cancelled.")
