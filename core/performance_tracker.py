"""
core/performance_tracker.py  –  15-Day Trade Performance Tracker

Stores every trade with full context, computes accuracy metrics daily,
and auto-suggests strategy improvements based on patterns found.

Data stored in performance_log.json:
  - Every trade: entry/exit price, P&L, which signals fired, which
    filters were active, hold duration, strategy that triggered it
  - Daily summary: win rate, avg P&L, Sharpe, drawdown
  - 15-day rolling window: auto-purges older data

Auto-improvement engine analyses:
  1. Which strategy has lowest win rate → reduce its weight
  2. Which filter is blocking too many good trades → loosen threshold
  3. Best/worst performing sectors → adjust screener bias
  4. Optimal hold time → suggest SL/target adjustment
  5. Time-of-day patterns → suggest entry window restrictions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

PERF_FILE     = Path("performance_log.json")
RETENTION_DAYS = 15


# ── Data Models ───────────────────────────────────────────────────

@dataclass
class TradeRecord:
    trade_id:       str
    symbol:         str
    action:         str           # BUY or SELL
    quantity:       int
    price:          float
    timestamp:      str
    date:           str           # YYYY-MM-DD

    # Context at time of trade
    strategy_signals: dict        # {MA_Crossover: BUY, RSI_Momentum: HOLD, ...}
    ensemble_score:   float       # aggregator combined_strength
    global_mood:      str         # BULLISH / NEUTRAL / BEARISH
    nifty_regime:     str         # STRONG_UPTREND / WEAK_UPTREND / SIDEWAYS / DOWNTREND
    news_mood:        str         # POSITIVE / NEUTRAL / NEGATIVE
    fii_mood:         str         # BULLISH_STRONG / ... / BEARISH_STRONG
    size_multiplier:  float       # combined multiplier applied

    # Filled after SELL
    exit_price:       Optional[float] = None
    exit_timestamp:   Optional[str]   = None
    pnl:              Optional[float] = None
    pnl_pct:          Optional[float] = None
    hold_minutes:     Optional[int]   = None
    exit_reason:      Optional[str]   = None   # SIGNAL / SL / TARGET / EOD


@dataclass
class DailySummary:
    date:            str
    total_trades:    int
    winning_trades:  int
    losing_trades:   int
    win_rate:        float
    total_pnl:       float
    avg_pnl:         float
    best_trade:      float
    worst_trade:     float
    avg_hold_min:    float
    starting_nav:    float
    ending_nav:      float
    nav_return_pct:  float


# ── Tracker ───────────────────────────────────────────────────────

class PerformanceTracker:

    def __init__(self):
        self._trades:  list[TradeRecord]  = []
        self._daily:   list[DailySummary] = []
        self._open:    dict[str, TradeRecord] = {}   # symbol → open trade
        self._load()

    # ── Persistence ───────────────────────────────────────────────

    def _load(self):
        if not PERF_FILE.exists():
            return
        try:
            data = json.loads(PERF_FILE.read_text())
            self._trades = [TradeRecord(**t) for t in data.get("trades", [])]
            self._daily  = [DailySummary(**d) for d in data.get("daily", [])]
            # Rebuild open trades (BUY without matching SELL)
            closed = {t.trade_id for t in self._trades
                      if t.action == "SELL" or t.exit_price is not None}
            self._open = {
                t.symbol: t for t in self._trades
                if t.action == "BUY" and t.trade_id not in closed
                and t.exit_price is None
            }
            logger.info(
                f"📊 Performance tracker loaded: "
                f"{len(self._trades)} trades  "
                f"{len(self._daily)} daily summaries"
            )
        except Exception as exc:
            logger.warning(f"Could not load performance log: {exc}")

    def _save(self):
        # Purge records older than 15 days
        cutoff = (date.today() - timedelta(days=RETENTION_DAYS)).isoformat()
        self._trades = [t for t in self._trades if t.date >= cutoff]
        self._daily  = [d for d in self._daily  if d.date >= cutoff]

        data = {
            "trades":       [asdict(t) for t in self._trades],
            "daily":        [asdict(d) for d in self._daily],
            "last_updated": datetime.now().isoformat(),
            "retention_days": RETENTION_DAYS,
        }
        PERF_FILE.write_text(json.dumps(data, indent=2))

    # ── Recording ─────────────────────────────────────────────────

    def record_buy(
        self,
        trade_id:         str,
        symbol:           str,
        quantity:         int,
        price:            float,
        strategy_signals: dict,
        ensemble_score:   float,
        global_mood:      str = "NEUTRAL",
        nifty_regime:     str = "WEAK_UPTREND",
        news_mood:        str = "NEUTRAL",
        fii_mood:         str = "NEUTRAL",
        size_multiplier:  float = 0.75,
    ):
        now = datetime.now()
        rec = TradeRecord(
            trade_id        = trade_id,
            symbol          = symbol,
            action          = "BUY",
            quantity        = quantity,
            price           = price,
            timestamp       = now.isoformat(),
            date            = now.strftime("%Y-%m-%d"),
            strategy_signals= strategy_signals,
            ensemble_score  = ensemble_score,
            global_mood     = global_mood,
            nifty_regime    = nifty_regime,
            news_mood       = news_mood,
            fii_mood        = fii_mood,
            size_multiplier = size_multiplier,
        )
        self._trades.append(rec)
        self._open[symbol] = rec
        self._save()
        logger.debug(f"📊 Recorded BUY {symbol} @ ₹{price}")

    def record_sell(
        self,
        symbol:       str,
        price:        float,
        exit_reason:  str = "SIGNAL",
    ):
        if symbol not in self._open:
            return
        buy_rec = self._open.pop(symbol)

        buy_dt  = datetime.fromisoformat(buy_rec.timestamp)
        sell_dt = datetime.now()
        hold_m  = int((sell_dt - buy_dt).total_seconds() / 60)
        pnl     = buy_rec.quantity * (price - buy_rec.price)
        pnl_pct = (price - buy_rec.price) / buy_rec.price * 100

        buy_rec.exit_price     = price
        buy_rec.exit_timestamp = sell_dt.isoformat()
        buy_rec.pnl            = round(pnl, 2)
        buy_rec.pnl_pct        = round(pnl_pct, 2)
        buy_rec.hold_minutes   = hold_m
        buy_rec.exit_reason    = exit_reason

        self._save()
        logger.info(
            f"📊 Recorded SELL {symbol} @ ₹{price}  "
            f"P&L=₹{pnl:+,.0f} ({pnl_pct:+.2f}%)  "
            f"held={hold_m}min  reason={exit_reason}"
        )

    def record_daily(
        self,
        starting_nav: float,
        ending_nav:   float,
    ):
        today      = date.today().isoformat()
        today_sells = [
            t for t in self._trades
            if t.date == today and t.exit_price is not None
        ]
        if not today_sells:
            return

        wins   = [t for t in today_sells if (t.pnl or 0) > 0]
        losses = [t for t in today_sells if (t.pnl or 0) <= 0]
        pnls   = [t.pnl for t in today_sells if t.pnl is not None]
        holds  = [t.hold_minutes for t in today_sells if t.hold_minutes]

        summary = DailySummary(
            date           = today,
            total_trades   = len(today_sells),
            winning_trades = len(wins),
            losing_trades  = len(losses),
            win_rate       = len(wins) / len(today_sells) if today_sells else 0,
            total_pnl      = round(sum(pnls), 2),
            avg_pnl        = round(sum(pnls) / len(pnls), 2) if pnls else 0,
            best_trade     = round(max(pnls), 2) if pnls else 0,
            worst_trade    = round(min(pnls), 2) if pnls else 0,
            avg_hold_min   = round(sum(holds) / len(holds), 1) if holds else 0,
            starting_nav   = starting_nav,
            ending_nav     = ending_nav,
            nav_return_pct = round((ending_nav - starting_nav) / starting_nav * 100, 3),
        )
        # Remove existing entry for today if re-recording
        self._daily = [d for d in self._daily if d.date != today]
        self._daily.append(summary)
        self._save()
        logger.info(f"📊 Daily summary recorded: {today}  P&L=₹{summary.total_pnl:+,.0f}")

    # ── Metrics ───────────────────────────────────────────────────

    def _closed_trades(self, days: int = 15) -> list[TradeRecord]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        return [
            t for t in self._trades
            if t.date >= cutoff and t.exit_price is not None
        ]

    def overall_win_rate(self) -> float:
        trades = self._closed_trades()
        if not trades:
            return 0.0
        return len([t for t in trades if (t.pnl or 0) > 0]) / len(trades)

    def total_pnl(self) -> float:
        return sum(t.pnl or 0 for t in self._closed_trades())

    def strategy_win_rates(self) -> dict[str, dict]:
        """Per-strategy win rate across closed trades."""
        result = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for t in self._closed_trades():
            for strat, sig in t.strategy_signals.items():
                if sig == "BUY":
                    if (t.pnl or 0) > 0:
                        result[strat]["wins"] += 1
                    else:
                        result[strat]["losses"] += 1
                    result[strat]["pnl"] += (t.pnl or 0)
        out = {}
        for strat, d in result.items():
            total = d["wins"] + d["losses"]
            out[strat] = {
                "wins":     d["wins"],
                "losses":   d["losses"],
                "win_rate": d["wins"] / total if total > 0 else 0,
                "total_pnl": round(d["pnl"], 2),
            }
        return out

    def exit_reason_breakdown(self) -> dict[str, int]:
        reasons = defaultdict(int)
        for t in self._closed_trades():
            reasons[t.exit_reason or "UNKNOWN"] += 1
        return dict(reasons)

    def best_worst_trades(self) -> tuple[Optional[TradeRecord], Optional[TradeRecord]]:
        trades = self._closed_trades()
        if not trades:
            return None, None
        best  = max(trades, key=lambda t: t.pnl or 0)
        worst = min(trades, key=lambda t: t.pnl or 0)
        return best, worst

    def avg_hold_time(self) -> float:
        holds = [t.hold_minutes for t in self._closed_trades() if t.hold_minutes]
        return round(sum(holds) / len(holds), 1) if holds else 0

    def symbol_performance(self) -> dict[str, dict]:
        perf = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        for t in self._closed_trades():
            perf[t.symbol]["trades"] += 1
            if (t.pnl or 0) > 0:
                perf[t.symbol]["wins"] += 1
            perf[t.symbol]["pnl"] += (t.pnl or 0)
        return {
            sym: {**d,
                  "win_rate": d["wins"] / d["trades"] if d["trades"] > 0 else 0,
                  "pnl": round(d["pnl"], 2)}
            for sym, d in perf.items()
        }

    # ── Auto-Improvement Engine ───────────────────────────────────

    def generate_improvements(self) -> list[dict]:
        """
        Analyse 15-day data and return list of improvement suggestions.
        Each suggestion has: area, finding, recommendation, priority
        """
        suggestions = []
        trades      = self._closed_trades()
        if len(trades) < 5:
            return [{"area": "Data", "finding": f"Only {len(trades)} closed trades",
                     "recommendation": "Need at least 5 closed trades for analysis",
                     "priority": "INFO"}]

        wins   = [t for t in trades if (t.pnl or 0) > 0]
        losses = [t for t in trades if (t.pnl or 0) <= 0]
        win_rate = len(wins) / len(trades)

        # ── 1. Strategy weight adjustment ─────────────────────────
        strat_wr = self.strategy_win_rates()
        for strat, d in strat_wr.items():
            wr = d["win_rate"]
            if wr < 0.35 and (d["wins"] + d["losses"]) >= 3:
                suggestions.append({
                    "area": "Strategy Weight",
                    "finding": f"{strat} win rate is only {wr:.0%} "
                               f"({d['wins']}W / {d['losses']}L)",
                    "recommendation": f"Reduce {strat} weight from current value by 10%",
                    "priority": "HIGH",
                    "action": f"REDUCE_WEIGHT:{strat}",
                })
            elif wr > 0.65 and (d["wins"] + d["losses"]) >= 3:
                suggestions.append({
                    "area": "Strategy Weight",
                    "finding": f"{strat} win rate is strong at {wr:.0%} "
                               f"({d['wins']}W / {d['losses']}L)",
                    "recommendation": f"Increase {strat} weight by 5%",
                    "priority": "MEDIUM",
                    "action": f"INCREASE_WEIGHT:{strat}",
                })

        # ── 2. Stop loss / target calibration ────────────────────
        sl_hits = [t for t in losses if t.exit_reason == "SL"]
        tgt_hits = [t for t in wins if t.exit_reason == "TARGET"]
        signal_exits = [t for t in trades if t.exit_reason == "SIGNAL"]

        sl_rate = len(sl_hits) / len(trades)
        if sl_rate > 0.40:
            avg_loss = abs(sum(t.pnl_pct or 0 for t in sl_hits) / len(sl_hits))
            suggestions.append({
                "area": "Stop Loss",
                "finding": f"SL hit on {sl_rate:.0%} of trades "
                           f"(avg loss {avg_loss:.2f}%)",
                "recommendation": "Widen SL from 1.5% to 2.0% — "
                                   "price fluctuation hitting stops too early",
                "priority": "HIGH",
                "action": "ADJUST_SL:0.020",
            })
        elif sl_rate < 0.10 and win_rate < 0.45:
            suggestions.append({
                "area": "Stop Loss",
                "finding": f"SL rarely triggered ({sl_rate:.0%}) but win rate low ({win_rate:.0%})",
                "recommendation": "Tighten SL from 1.5% to 1.0% — "
                                   "letting losses run too long on signal exits",
                "priority": "MEDIUM",
                "action": "ADJUST_SL:0.010",
            })

        if tgt_hits:
            tgt_rate = len(tgt_hits) / len(wins) if wins else 0
            if tgt_rate > 0.70:
                suggestions.append({
                    "area": "Target",
                    "finding": f"{tgt_rate:.0%} of wins hitting full target — "
                                "momentum may continue beyond target",
                    "recommendation": "Raise target from 3% to 4% to capture more upside",
                    "priority": "MEDIUM",
                    "action": "ADJUST_TARGET:0.040",
                })

        # ── 3. Hold time analysis ─────────────────────────────────
        avg_hold = self.avg_hold_time()
        if avg_hold < 30:
            suggestions.append({
                "area": "Hold Time",
                "finding": f"Average hold time only {avg_hold:.0f} min — "
                           "signals may be too trigger-happy",
                "recommendation": "Add minimum hold time of 15 min before allowing exit signal",
                "priority": "MEDIUM",
                "action": "MIN_HOLD:15",
            })

        # ── 4. Time of day analysis ───────────────────────────────
        morning_trades = [t for t in losses
                          if t.timestamp[11:13] in ("09", "10")]
        morning_loss_rate = (len(morning_trades) / len(losses)) if losses else 0
        if morning_loss_rate > 0.55:
            suggestions.append({
                "area": "Entry Window",
                "finding": f"{morning_loss_rate:.0%} of losses happen in first 60 min",
                "recommendation": "Delay entries until 10:15 AM — "
                                   "avoid volatile market open period",
                "priority": "HIGH",
                "action": "DELAY_ENTRY:10:15",
            })

        # ── 5. Symbol concentration ───────────────────────────────
        sym_perf = self.symbol_performance()
        bad_syms = [s for s, d in sym_perf.items()
                    if d["win_rate"] < 0.30 and d["trades"] >= 3]
        if bad_syms:
            suggestions.append({
                "area": "Symbol Filter",
                "finding": f"{bad_syms} consistently losing (win rate < 30%)",
                "recommendation": f"Blacklist {bad_syms} from watchlist for next 5 days",
                "priority": "HIGH",
                "action": f"BLACKLIST:{','.join(bad_syms)}",
            })

        # ── 6. Overall win rate ───────────────────────────────────
        if win_rate > 0.60:
            suggestions.append({
                "area": "Position Size",
                "finding": f"Win rate is strong at {win_rate:.0%} over {len(trades)} trades",
                "recommendation": "Increase max_position_pct from 5% to 7% — "
                                   "bot is performing well, deploy more capital per trade",
                "priority": "MEDIUM",
                "action": "INCREASE_SIZE:0.07",
            })
        elif win_rate < 0.40:
            suggestions.append({
                "area": "Position Size",
                "finding": f"Win rate below 40% ({win_rate:.0%}) — "
                            "strategy needs recalibration",
                "recommendation": "Reduce max_position_pct from 5% to 3% until "
                                   "win rate improves",
                "priority": "HIGH",
                "action": "REDUCE_SIZE:0.03",
            })

        return suggestions

    def auto_apply_improvements(self, env_path: str = ".env") -> list[str]:
        """
        Apply HIGH priority improvements automatically to .env.
        Returns list of changes made.
        """
        suggestions  = self.generate_improvements()
        high_priority = [s for s in suggestions if s["priority"] == "HIGH"]
        applied = []

        try:
            env_text = Path(env_path).read_text()
        except Exception:
            return []

        for s in high_priority:
            action = s.get("action", "")

            if action.startswith("ADJUST_SL:"):
                new_sl = action.split(":")[1]
                env_text = _replace_env(env_text, "STOP_LOSS_PCT", new_sl)
                applied.append(f"STOP_LOSS_PCT → {new_sl} ({s['finding']})")

            elif action.startswith("ADJUST_TARGET:"):
                new_tgt = action.split(":")[1]
                env_text = _replace_env(env_text, "TARGET_PCT", new_tgt)
                applied.append(f"TARGET_PCT → {new_tgt} ({s['finding']})")

            elif action.startswith("REDUCE_SIZE:"):
                new_sz = action.split(":")[1]
                env_text = _replace_env(env_text, "MAX_POSITION_PCT", new_sz)
                applied.append(f"MAX_POSITION_PCT → {new_sz} ({s['finding']})")

            elif action.startswith("INCREASE_SIZE:"):
                new_sz = action.split(":")[1]
                env_text = _replace_env(env_text, "MAX_POSITION_PCT", new_sz)
                applied.append(f"MAX_POSITION_PCT → {new_sz} ({s['finding']})")

        if applied:
            Path(env_path).write_text(env_text)
            logger.success(f"🤖 Auto-applied {len(applied)} improvements to .env")

        return applied

    # ── Display ───────────────────────────────────────────────────

    def print_report(self):
        trades  = self._closed_trades()
        console.print()
        console.print(Panel(
            f"  [bold]15-Day Performance Report[/]  —  "
            f"{(date.today() - timedelta(days=15)).strftime('%d %b')} → "
            f"{date.today().strftime('%d %b %Y')}\n\n"
            f"  Closed trades : [cyan]{len(trades)}[/]  │  "
            f"Win rate: [{'green' if self.overall_win_rate()>0.5 else 'red'}]"
            f"{self.overall_win_rate():.1%}[/]  │  "
            f"Total P&L: [{'green' if self.total_pnl()>=0 else 'red'}]"
            f"₹{self.total_pnl():+,.0f}[/]  │  "
            f"Avg hold: {self.avg_hold_time():.0f} min",
            title="📊 Performance Tracker",
            style="bold cyan", box=box.DOUBLE_EDGE,
        ))

        # Daily table
        if self._daily:
            tbl = Table(title="Daily P&L", box=box.SIMPLE_HEAVY,
                        title_style="bold cyan")
            tbl.add_column("Date",    width=12)
            tbl.add_column("Trades",  justify="right", width=7)
            tbl.add_column("Wins",    justify="right", width=6)
            tbl.add_column("Win%",    justify="right", width=7)
            tbl.add_column("P&L",     justify="right", width=12)
            tbl.add_column("Return",  justify="right", width=8)
            tbl.add_column("NAV",     justify="right", width=12)

            for d in sorted(self._daily, key=lambda x: x.date, reverse=True)[:15]:
                c = "green" if d.total_pnl >= 0 else "red"
                tbl.add_row(
                    d.date, str(d.total_trades),
                    str(d.winning_trades),
                    f"{d.win_rate:.0%}",
                    f"[{c}]₹{d.total_pnl:+,.0f}[/]",
                    f"[{c}]{d.nav_return_pct:+.2f}%[/]",
                    f"₹{d.ending_nav:,.0f}",
                )
            console.print(tbl)

        # Strategy performance
        strat_wr = self.strategy_win_rates()
        if strat_wr:
            tbl2 = Table(title="Strategy Performance", box=box.SIMPLE_HEAVY,
                         title_style="bold yellow")
            tbl2.add_column("Strategy",   style="bold white", width=20)
            tbl2.add_column("Wins",       justify="right", width=6)
            tbl2.add_column("Losses",     justify="right", width=8)
            tbl2.add_column("Win Rate",   justify="right", width=10)
            tbl2.add_column("Total P&L",  justify="right", width=12)
            for strat, d in strat_wr.items():
                c = "green" if d["win_rate"] >= 0.5 else "red"
                tbl2.add_row(
                    strat, str(d["wins"]), str(d["losses"]),
                    f"[{c}]{d['win_rate']:.0%}[/]",
                    f"[{'green' if d['total_pnl']>=0 else 'red'}]₹{d['total_pnl']:+,.0f}[/]",
                )
            console.print(tbl2)

        # Exit reasons
        reasons = self.exit_reason_breakdown()
        if reasons:
            console.print(
                "  Exit breakdown:  " +
                "  │  ".join(f"[cyan]{k}[/] {v}" for k, v in reasons.items())
            )

        # Improvement suggestions
        suggestions = self.generate_improvements()
        if suggestions:
            console.print()
            tbl3 = Table(title="🤖 Auto-Improvement Suggestions",
                         box=box.SIMPLE_HEAVY, title_style="bold magenta")
            tbl3.add_column("Priority", width=8)
            tbl3.add_column("Area",     width=16)
            tbl3.add_column("Finding",  width=38)
            tbl3.add_column("Recommendation", width=40)
            p_colors = {"HIGH": "red", "MEDIUM": "yellow", "INFO": "dim"}
            for s in suggestions:
                c = p_colors.get(s["priority"], "white")
                tbl3.add_row(
                    f"[{c}]{s['priority']}[/]",
                    s["area"],
                    s["finding"][:36],
                    s["recommendation"][:38],
                )
            console.print(tbl3)

            high = [s for s in suggestions if s["priority"] == "HIGH"]
            if high:
                console.print(
                    f"\n  [bold yellow]⚡ {len(high)} HIGH priority improvement(s) "
                    f"will be auto-applied to .env on next run.[/]\n"
                )


def _replace_env(text: str, key: str, value: str) -> str:
    """Replace a key=value line in .env text."""
    import re
    pattern = rf"^{key}=.*$"
    replacement = f"{key}={value}"
    new_text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    if key not in new_text:
        new_text += f"\n{replacement}"
    return new_text
