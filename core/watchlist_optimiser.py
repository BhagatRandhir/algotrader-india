"""
core/watchlist_optimiser.py  –  Hourly Watchlist Optimiser

Every hour:
  1. Runs full screener scan → gets scores for ALL candidates
  2. Loads current watchlist + their last known scores
  3. Compares: if a new stock scores 20%+ higher than the weakest
     stock currently in the watchlist → swap it in
  4. Never removes a symbol that has an open position
  5. Prints a swap report showing what changed and why

Result: watchlist continuously improves during the day, always
holding the best-scoring stocks available right now.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

WATCHLIST_FILE = Path("watchlist.json")
console = Console()

# How much better a new stock must score to displace an existing one
SWAP_THRESHOLD = 0.20   # 20% higher score required


@dataclass
class SwapEvent:
    removed:    str
    added:      str
    old_score:  float
    new_score:  float
    improvement: float   # percentage improvement


class WatchlistOptimiser:
    """
    Usage:
        opt = WatchlistOptimiser()
        new_watchlist = opt.optimise(
            all_scored_stocks,   # list[StockScore] from screener
            current_watchlist,   # list[str] current symbols
            open_positions,      # dict — these are never removed
            max_picks,           # max size of watchlist
        )
    """

    def __init__(self, swap_threshold: float = SWAP_THRESHOLD):
        self.swap_threshold = swap_threshold
        self._swap_history: list[SwapEvent] = []

    def optimise(
        self,
        all_candidates,       # list[StockScore] — full scored universe
        current_watchlist:    list[str],
        open_positions:       dict,
        max_picks:            int = 12,
    ) -> tuple[list[str], list[SwapEvent]]:
        """
        Returns (new_watchlist, swaps_made).
        new_watchlist always includes open_positions symbols.
        """
        if not all_candidates:
            return current_watchlist, []

        held = set(open_positions.keys())

        # Build score lookup for all candidates
        score_map: dict[str, float] = {
            s.symbol: s.score for s in all_candidates
        }

        # Load last known scores from watchlist.json
        current_scores = self._load_current_scores()

        # Current watchlist with their scores
        # Use last known score if not re-scored this run
        current_with_scores = []
        for sym in current_watchlist:
            score = score_map.get(sym) or current_scores.get(sym, 0.0)
            current_with_scores.append((sym, score))

        # All candidates sorted by score descending
        ranked_new = sorted(all_candidates, key=lambda x: x.score, reverse=True)

        # --- Optimisation logic ---
        # Start with held positions (always kept)
        protected = [(sym, score_map.get(sym, current_scores.get(sym, 0.0)))
                     for sym in held]
        protected_syms = held

        # Remaining slots
        slots = max_picks - len(protected)
        if slots <= 0:
            # All slots taken by positions — just return held
            result = list(held) + [s for s in current_watchlist if s not in held]
            return result[:max_picks], []

        # Pool to fill slots: current non-held watchlist
        current_pool = [(sym, sc) for sym, sc in current_with_scores
                        if sym not in protected_syms]

        # New candidates not already in watchlist
        new_pool = [s for s in ranked_new
                    if s.symbol not in set(current_watchlist) | protected_syms]

        # Fill slots: try to keep current, swap out weakest for better newcomers
        final_pool = list(current_pool)   # start with current
        swaps: list[SwapEvent] = []

        for new_stock in new_pool:
            if len(final_pool) < slots:
                # Empty slot — just add
                final_pool.append((new_stock.symbol, new_stock.score))
                logger.info(
                    f"📥 Added {new_stock.symbol} "
                    f"(score={new_stock.score:.2f}, slot was empty)"
                )
                continue

            # Find weakest in current pool
            if not final_pool:
                break
            weakest_sym, weakest_score = min(final_pool, key=lambda x: x[1])

            # Swap only if new stock is meaningfully better
            improvement = (new_stock.score - weakest_score) / (weakest_score + 1e-9)
            if improvement >= self.swap_threshold:
                final_pool.remove((weakest_sym, weakest_score))
                final_pool.append((new_stock.symbol, new_stock.score))
                swap = SwapEvent(
                    removed     = weakest_sym,
                    added       = new_stock.symbol,
                    old_score   = round(weakest_score, 3),
                    new_score   = round(new_stock.score, 3),
                    improvement = round(improvement * 100, 1),
                )
                swaps.append(swap)
                self._swap_history.append(swap)
                logger.info(
                    f"🔄 SWAP: {weakest_sym}(score={weakest_score:.2f}) → "
                    f"{new_stock.symbol}(score={new_stock.score:.2f}) "
                    f"+{improvement:.0%} better"
                )

        # Build final list: protected first, then optimised pool (sorted by score)
        final_pool.sort(key=lambda x: x[1], reverse=True)
        final_syms = [s for s in protected_syms] + [sym for sym, _ in final_pool]
        final_syms = list(dict.fromkeys(final_syms))[:max_picks]   # dedup, cap

        # Save updated scores to watchlist.json
        self._save_updated_watchlist(final_syms, score_map, current_scores)

        return final_syms, swaps

    def _load_current_scores(self) -> dict[str, float]:
        """Load last known scores from watchlist.json details section."""
        if not WATCHLIST_FILE.exists():
            return {}
        try:
            data    = json.loads(WATCHLIST_FILE.read_text())
            details = data.get("details", [])
            return {d["symbol"]: d["score"] for d in details if "score" in d}
        except Exception:
            return {}

    def _save_updated_watchlist(
        self,
        symbols:        list[str],
        score_map:      dict[str, float],
        fallback_scores: dict[str, float],
    ):
        """Update watchlist.json with new symbols and their scores."""
        try:
            existing = {}
            if WATCHLIST_FILE.exists():
                try:
                    data = json.loads(WATCHLIST_FILE.read_text())
                    existing = {d["symbol"]: d
                                for d in data.get("details", [])}
                except Exception:
                    pass

            details = []
            for sym in symbols:
                score = score_map.get(sym) or fallback_scores.get(sym, 0.0)
                detail = existing.get(sym, {"symbol": sym})
                detail["score"] = round(score, 3)
                details.append(detail)

            WATCHLIST_FILE.write_text(json.dumps({
                "generated_at": datetime.now().isoformat(),
                "symbols":      symbols,
                "details":      details,
            }, indent=2))
        except Exception as exc:
            logger.warning(f"Could not update watchlist.json: {exc}")

    def print_swap_report(self, swaps: list[SwapEvent], new_watchlist: list[str]):
        """Print a rich table showing what changed this hour."""
        console.print()
        if not swaps:
            console.print(
                "  [dim]🔄 Hourly optimisation: no swaps — current watchlist "
                "is already the best available[/]"
            )
        else:
            tbl = Table(
                title=f"🔄 Watchlist Optimisation — "
                      f"{datetime.now().strftime('%H:%M IST')}  "
                      f"({len(swaps)} swap{'s' if len(swaps)>1 else ''})",
                box=box.SIMPLE_HEAVY,
                title_style="bold yellow",
            )
            tbl.add_column("Removed",     style="red",   width=14)
            tbl.add_column("Old Score",   justify="right", width=10)
            tbl.add_column("Added",       style="green", width=14)
            tbl.add_column("New Score",   justify="right", width=10)
            tbl.add_column("Improvement", justify="right", width=13)

            for sw in swaps:
                tbl.add_row(
                    sw.removed,
                    f"{sw.old_score:.3f}",
                    sw.added,
                    f"{sw.new_score:.3f}",
                    f"[bold green]+{sw.improvement:.1f}%[/]",
                )
            console.print(tbl)

        console.print(
            f"  📋 Active watchlist ({len(new_watchlist)}): "
            f"[bold cyan]{', '.join(new_watchlist)}[/]\n"
        )

    def swap_summary(self) -> str:
        """One-line summary of all swaps made today."""
        if not self._swap_history:
            return "No swaps today"
        return (f"{len(self._swap_history)} swaps today: "
                + ", ".join(f"{s.removed}→{s.added}" for s in self._swap_history[-3:]))
