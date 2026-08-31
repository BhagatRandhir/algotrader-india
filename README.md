# AlgoTrader India — Aggressive Momentum Bot

NSE paper trading bot. Free to run — no Zerodha API subscription,
no API keys, no paid data feeds. Uses NSE India's public API directly.

---

## Quick Start

```bash
# 1. Install dependencies
# On Mac:
pip3 install -r requirements.txt

# On Linux/Windows:
pip install -r requirements.txt

# 2. Create config
cp .env.example .env
# Edit .env → set INITIAL_CAPITAL=500000

# 3. Test your connection (always do this first)
python3 paper_main.py --test     # Mac
python  paper_main.py --test     # Linux/Windows

# 4. Run the bot
python3 paper_main.py            # Mac
python  paper_main.py            # Linux/Windows

# Stop anytime
Ctrl+C   ← prints full P&L summary before exiting
```

> **Mac users:** always use `python3` and `pip3` instead of `python` and `pip`.

---

## What the bot does

Scans NSE stocks every 60 seconds and buys when a stock shows strong
intraday momentum. Manages exits automatically with trailing stops and
partial profit booking. All session management is fully automatic —
no manual refresh needed.

### Entry (4 of 5 conditions must pass)
1. Price > VWAP (above intraday average price)
2. Price > EMA20 (short-term uptrend)
3. RSI between 50 and 88 (momentum without being overbought)
4. Volume > 1.3× 20-bar average (unusual activity)
5. Day change > 0% (positive on the day)

### Exit (smart trailing — all automatic)
- **Partial exit** at +1.2% → sell 50% of position, SL moves to breakeven
- **Trailing SL** → trails 1% below rolling peak on remaining shares
- **Volume dry-up** → exits if volume drops below 50% of average while holding
- **RSI divergence** → exits if price making new high but RSI falling
- **Fixed SL** → 1.5% below entry price
- **Target** → 3.0% above entry price
- **Force exit** → 3:00 PM IST every day (no overnight positions)

---

## Automation schedule

Everything runs automatically once you start the bot:

| Task | Frequency | Detail |
|------|-----------|--------|
| Fetch live prices | Every 60 sec | NSE API → yfinance → synthetic fallback |
| Check buy signals | Every 60 sec | All symbols in watchlist |
| Manage open positions | Every 60 sec | Trailing SL, partial exit, divergence |
| NSE session keepalive | Every 4 min | Background thread — fully automatic |
| Cookie renewal on 403 | Instant | Auto-retry inside data client |
| Watchlist refresh | Every 60 min | Screener scans NSE, swaps weak for strong |
| Watchlist optimisation | Every 60 min | Compares scores, swaps in 20%+ better stocks |
| Force exit all positions | 3:00 PM IST | Prevents overnight exposure |
| Trade log save | After every order | Saved to `paper_trades.json` |
| Resume after restart | On startup | Reads `paper_trades.json` automatically |

---

## Data sources (priority order)

1. **NSE India API** (primary) — `nseindia.com/api/*` — live, real-time, free
2. **yfinance** (fallback) — if NSE API fails
3. **Synthetic bars** (last resort) — bot never freezes, always has data

NSE session is kept alive automatically by a background thread.
No manual refresh or re-login needed.

---

## File structure

```
india_trader/
│
├── paper_main.py               ← Entry point — run this
├── main.py                     ← Live trading (needs Zerodha API)
├── .env.example                ← Copy to .env and configure
├── requirements.txt
├── README.md                   ← This file
│
├── strategies/
│   ├── base.py                 ← Signal / StrategyResult classes
│   ├── momentum_bot.py         ← Main strategy (VWAP/EMA/RSI/Vol)
│   ├── multi_timeframe.py      ← 5m/15m/1h alignment (sizing only)
│   ├── ma_crossover.py         ← Legacy (kept for reference)
│   ├── rsi_momentum.py         ← Legacy
│   ├── mean_reversion.py       ← Legacy
│   └── pattern_recognition.py  ← Legacy
│
├── core/
│   ├── nse_data.py             ← NSE India API client (primary data)
│   ├── paper_broker.py         ← Simulates orders, tracks P&L
│   ├── broker.py               ← Live Zerodha broker (for live mode)
│   ├── screener.py             ← Hourly stock scanner (NSE + yfinance)
│   ├── watchlist_optimiser.py  ← Swaps weak stocks for better ones
│   ├── smart_exit.py           ← Partial exits, trailing SL, divergence
│   ├── risk.py                 ← Position sizing, kill-switch, tight mode
│   ├── market_regime.py        ← VIX + Nifty ADX + breadth filter
│   ├── volume_profile.py       ← VWAP bands + order flow (sizing)
│   ├── aggregator.py           ← Signal aggregator
│   ├── performance_tracker.py  ← P&L log, auto-improvements
│   ├── nifty_trend.py          ← Nifty trend filter
│   ├── global_sentiment.py     ← Global market mood
│   ├── fii_dii.py              ← Foreign/domestic money flow
│   └── news_sentiment.py       ← News sentiment (legacy)
│
├── utils/
│   ├── nse_fetcher.py          ← Stooq + synthetic bar fallback
│   ├── dashboard.py            ← Rich terminal UI
│   ├── market_hours.py         ← IST open/close checks
│   ├── yf_helpers.py           ← yfinance column flattening
│   └── nse_universe.py         ← Full NSE stock list for screener
│
├── auth/
│   └── login.py                ← Zerodha login (live mode only)
│
├── logs/
│   └── trader.log              ← Full trade log (auto-created)
│
├── paper_trades.json           ← All orders + P&L (auto-saved)
└── watchlist.json              ← Active watchlist + scores (auto-updated)
```

---

## Configuration (.env)

```bash
# ── Capital ───────────────────────────────────────────────────────
INITIAL_CAPITAL=500000         # starting paper capital in ₹

# ── Trade sizing ──────────────────────────────────────────────────
MAX_POSITION_PCT=0.08          # 8% of capital per trade
MAX_OPEN_POSITIONS=6           # max simultaneous positions
DAILY_TRADE_TARGET=5           # target trades per day

# ── Risk management ───────────────────────────────────────────────
STOP_LOSS_PCT=0.015            # 1.5% hard stop loss
TARGET_PCT=0.030               # 3.0% profit target
TRAIL_TRIGGER_PCT=0.012        # trailing SL activates after +1.2%
MIN_RR_RATIO=1.5               # minimum reward:risk before entry
MAX_DAILY_LOSS_PCT=0.03        # halt if daily loss > 3% of capital
CONSEC_LOSS_LIMIT=4            # tight mode after 4 consecutive losses

# ── Screener ──────────────────────────────────────────────────────
USE_SCREENER=true              # auto-scan NSE for best stocks
SCREENER_REFRESH_MIN=60        # refresh watchlist every 60 minutes
SCREENER_MAX_PICKS=15          # top 15 stocks in watchlist

# ── Manual watchlist (used if USE_SCREENER=false) ─────────────────
NSE_WATCHLIST=RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,WIPRO,SBIN,AXISBANK,BAJFINANCE,TITAN

# ── Bar settings ──────────────────────────────────────────────────
BAR_INTERVAL=5m                # candle interval
BAR_LOOKBACK=5d                # history for indicators
PRODUCT_TYPE=MIS               # MIS (intraday) or CNC (delivery)
LOOP_INTERVAL_SEC=60           # main loop interval in seconds
```

---

## Tight mode (automatic)

After `CONSEC_LOSS_LIMIT` consecutive losing trades, the bot automatically:
- Reduces position size by 25%
- Lowers max open positions by 2
- Clears automatically after the next winning trade

---

## Troubleshooting

**"NSE warmup: 403"**
Your network is blocking nseindia.com. The bot falls back to yfinance
automatically. Confirm by opening https://www.nseindia.com in your browser.

**No trades for hours**
Run `python paper_main.py --test` — it shows exactly which gate is blocking.
The bot only enters trades between 9:30 AM and 3:00 PM IST on weekdays.

**Bot crashed / power cut**
Restart with `python paper_main.py` — it reads `paper_trades.json` and
resumes from exactly where it left off, including all open positions.

**Want to reset everything**
Delete `paper_trades.json` and `watchlist.json`, then restart.

---

## Changelog

| Version | What changed |
|---------|-------------|
| v1 | Basic momentum bot with 5 conditions (VWAP/EMA/RSI/Vol/DayChg) |
| v2 | Added AI signal scorer (Random Forest on trade history) |
| v3 | Added multi-timeframe (5m/15m/1h), volume profile, market regime |
| v4 | Added smart exit (partial profit, trailing SL, RSI divergence) |
| v5 | Added hourly watchlist optimiser (swaps weak stocks for stronger) |
| v6 | Rebuilt data layer — NSE India API replaces yfinance as primary |
| v6.1 | Auto keepalive thread — NSE session never expires manually |
| v6.2 | Aggressive settings — 4/5 conditions, wider RSI 50-88, vol 1.3× |

---

## Requirements

- Python 3.10+
- Internet access to nseindia.com (yfinance is fallback)
- No Zerodha account needed for paper trading
- ~50MB disk for logs
