"""
backtest.py  –  Backtest strategies on NSE historical data (yfinance, free).

Usage:
    python backtest.py --symbol RELIANCE --start 2023-01-01 --end 2024-12-31
    python backtest.py --symbol TCS      --start 2022-01-01 --end 2024-01-01
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_momentum import RSIMomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from core.aggregator import SignalAggregator
from strategies.base import Signal
from utils.yf_helpers import flatten_yf_columns


WEIGHTS = {
    "MA_Crossover": 0.30, "RSI_Momentum": 0.25,
    "Mean_Reversion": 0.20, "Pattern_Recognition": 0.25
}


def run_backtest(
    symbol: str,
    start: str,
    end: str,
    initial_capital: float = 500_000.0,
    stop_loss_pct: float = 0.015,
    target_pct: float = 0.030,
):
    ticker = f"{symbol}.NS"
    print(f"\nDownloading {ticker}  {start} → {end} …")
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        print(f"❌  No data for {ticker}")
        return
    df = flatten_yf_columns(df)

    strategies  = [
        MACrossoverStrategy(), RSIMomentumStrategy(),
        MeanReversionStrategy(), PatternRecognitionStrategy(),
    ]
    aggregator  = SignalAggregator(WEIGHTS)

    cash        = initial_capital
    position    = 0
    entry_price = 0.0
    equity      = []
    trades      = []

    for i in range(210, len(df)):
        window = df.iloc[:i + 1]
        results = [s.generate_signal(window, symbol) for s in strategies]
        agg     = aggregator.aggregate(results)
        price   = df["close"].iloc[i]
        date    = df.index[i]

        # Check SL / target on open position
        if position > 0:
            sl     = entry_price * (1 - stop_loss_pct)
            target = entry_price * (1 + target_pct)
            if price <= sl or price >= target:
                reason = "STOP_LOSS" if price <= sl else "TARGET"
                pnl    = position * (price - entry_price)
                cash  += position * price
                trades.append({"date": date, "action": "SELL", "price": price,
                               "pnl": pnl, "reason": reason})
                position = 0
                entry_price = 0.0

        if agg.final_signal == Signal.BUY and position == 0:
            qty = int((cash * 0.95) / price)
            if qty > 0:
                cash       -= qty * price
                position    = qty
                entry_price = price
                trades.append({"date": date, "action": "BUY",
                               "price": price, "pnl": None, "reason": "SIGNAL"})

        elif agg.final_signal == Signal.SELL and position > 0:
            pnl   = position * (price - entry_price)
            cash += position * price
            trades.append({"date": date, "action": "SELL", "price": price,
                           "pnl": pnl, "reason": "SIGNAL"})
            position = 0
            entry_price = 0.0

        equity.append(cash + position * price)

    # ── Metrics ───────────────────────────────────────────────────
    eq       = pd.Series(equity, index=df.index[210:])
    total_r  = (eq.iloc[-1] - initial_capital) / initial_capital
    days     = max((eq.index[-1] - eq.index[0]).days, 1)
    ann_r    = (1 + total_r) ** (365 / days) - 1
    d_ret    = eq.pct_change().dropna()
    sharpe   = (d_ret.mean() / d_ret.std() * np.sqrt(252)) if d_ret.std() > 0 else 0
    max_dd   = ((eq - eq.cummax()) / eq.cummax()).min()
    sells    = [t for t in trades if t["action"] == "SELL" and t["pnl"] is not None]
    wins     = [t for t in sells if t["pnl"] > 0]
    win_rate = len(wins) / len(sells) if sells else 0

    print(f"""
╔══════════════════════════════════════════════════╗
  Backtest — {symbol}.NS
  Period    : {start}  →  {end}
──────────────────────────────────────────────────
  Capital      : ₹{initial_capital:,.0f}
  Final NAV    : ₹{eq.iloc[-1]:,.0f}
  Total return : {total_r:+.2%}
  Ann. return  : {ann_r:+.2%}
  Sharpe ratio : {sharpe:.2f}
  Max drawdown : {max_dd:.2%}
  Win rate     : {win_rate:.2%}
  Total trades : {len(sells)}
╚══════════════════════════════════════════════════╝
""")

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        out = f"backtest_{symbol}_{start[:4]}_{end[:4]}.csv"
        trade_df.to_csv(out, index=False)
        print(f"Trade log saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="RELIANCE")
    parser.add_argument("--start",  default="2022-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    parser.add_argument("--capital", type=float, default=500_000.0)
    args = parser.parse_args()
    run_backtest(args.symbol, args.start, args.end, args.capital)
