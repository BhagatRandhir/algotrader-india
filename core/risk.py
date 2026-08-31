"""
core/risk.py  –  Risk Manager (v2 — Profit-focused)

Key upgrades:
  • AI confidence gate: skips low-confidence trades (< 0.45)
  • Dynamic position sizing: scales with signal strength × AI confidence
  • Trailing stop: activates once position is up 1.5%
  • Reward:Risk enforcement — minimum 2:1 before entry
  • Time-based exit: auto-exit 30 min before market close (3:00 PM IST)
  • Win-rate tracking: auto-tighten filters after 3 consecutive losses
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import pytz

from loguru import logger


IST = pytz.timezone("Asia/Kolkata")


@dataclass
class RiskConfig:
    max_position_pct:    float = 0.06
    max_daily_loss_pct:  float = 0.02
    max_open_positions:  int   = 8
    stop_loss_pct:       float = 0.012
    target_pct:          float = 0.025
    min_rr_ratio:        float = 2.0      # minimum reward:risk
    ai_confidence_min:   float = 0.45     # skip trade if AI scores below this
    trail_trigger_pct:   float = 0.015    # activate trailing SL after +1.5%
    force_exit_hour:     int   = 15       # force exit at 3 PM IST
    force_exit_minute:   int   = 0
    consec_loss_limit:   int   = 3        # tighten after N consecutive losses


class RiskManager:

    def __init__(self, config: RiskConfig):
        self.cfg            = config
        self._halted        = False
        self._halt_reason   = ""
        self._consec_losses = 0
        self._tight_mode    = False        # activated after N consec losses
        # symbol → highest price seen since entry (for trailing stop)
        self._trail_high:   dict[str, float] = {}

    # ── Kill switch ───────────────────────────────────────────────

    def check_daily_loss(self, daily_pnl: float, nav: float) -> bool:
        if nav <= 0:
            return True
        loss_pct = abs(min(daily_pnl, 0)) / nav
        if loss_pct >= self.cfg.max_daily_loss_pct:
            self._halt(f"Daily loss {loss_pct:.2%} ≥ limit {self.cfg.max_daily_loss_pct:.2%}")
            return False
        return True

    def check_position_count(self, open_count: int) -> bool:
        limit = (self.cfg.max_open_positions - 2) if self._tight_mode \
                else self.cfg.max_open_positions
        if open_count >= limit:
            logger.warning(f"Max positions ({open_count}/{limit})"
                           + (" [TIGHT MODE]" if self._tight_mode else ""))
            return False
        return True

    def check_ai_confidence(self, confidence: float) -> bool:
        """Gate: skip trade if AI confidence is too low."""
        threshold = (self.cfg.ai_confidence_min + 0.10) if self._tight_mode \
                    else self.cfg.ai_confidence_min
        if confidence < threshold:
            logger.info(f"AI confidence {confidence:.3f} < {threshold:.3f} — skipping")
            return False
        return True

    def check_reward_risk(self, entry: float, sl: float, target: float) -> bool:
        """Enforce minimum 2:1 reward:risk before entry."""
        risk   = abs(entry - sl)
        reward = abs(target - entry)
        if risk == 0:
            return True
        rr = reward / risk
        if rr < self.cfg.min_rr_ratio:
            logger.info(f"R:R={rr:.2f} < {self.cfg.min_rr_ratio} — skipping")
            return False
        return True

    def is_force_exit_time(self) -> bool:
        """True if within 30 min of market close — force exit open positions."""
        now = datetime.now(IST)
        return (now.hour == self.cfg.force_exit_hour and
                now.minute >= self.cfg.force_exit_minute)

    # ── Trailing stop ─────────────────────────────────────────────

    def update_trail(self, symbol: str, current_price: float):
        """Update the highest-seen price for trailing stop."""
        self._trail_high[symbol] = max(
            self._trail_high.get(symbol, current_price),
            current_price,
        )

    def trail_stop_price(self, symbol: str, entry: float) -> float:
        """
        Returns the trailing SL price.
        Activates only once the position is up trail_trigger_pct.
        Before activation, returns the fixed SL.
        """
        fixed_sl  = self.stop_loss_price(entry)
        high      = self._trail_high.get(symbol, entry)
        gain_pct  = (high - entry) / entry if entry > 0 else 0

        if gain_pct >= self.cfg.trail_trigger_pct:
            # Trail at EMA20-proxy: 1% below the peak seen
            trail_sl = round(high * (1 - self.cfg.stop_loss_pct), 2)
            return max(trail_sl, fixed_sl)   # never loosen below fixed SL
        return fixed_sl

    def clear_trail(self, symbol: str):
        self._trail_high.pop(symbol, None)

    # ── Sizing ────────────────────────────────────────────────────

    def position_size(
        self,
        available_cash: float,
        price: float,
        signal_strength: float = 1.0,
        ai_confidence:   float = 0.5,
    ) -> int:
        if price <= 0 or available_cash <= 0:
            return 0
        signal_strength = max(0.0, min(signal_strength, 1.0))
        ai_confidence   = max(0.0, min(ai_confidence,   1.0))

        # Blend signal strength and AI confidence equally
        combined    = (signal_strength * 0.6 + ai_confidence * 0.4)

        # In tight mode, reduce size by 25%
        if self._tight_mode:
            combined *= 0.75

        max_rupees = available_cash * self.cfg.max_position_pct * combined
        return max(int(max_rupees / price), 0)

    def stop_loss_price(self, entry: float, side: str = "BUY") -> float:
        if side == "BUY":
            return round(entry * (1 - self.cfg.stop_loss_pct), 2)
        return round(entry * (1 + self.cfg.stop_loss_pct), 2)

    def target_price(self, entry: float, side: str = "BUY") -> float:
        if side == "BUY":
            return round(entry * (1 + self.cfg.target_pct), 2)
        return round(entry * (1 - self.cfg.target_pct), 2)

    # ── Win/Loss tracking ─────────────────────────────────────────

    def record_outcome(self, pnl: float):
        """Call after every closed trade to track consecutive losses."""
        if pnl < 0:
            self._consec_losses += 1
            if self._consec_losses >= self.cfg.consec_loss_limit:
                if not self._tight_mode:
                    self._tight_mode = True
                    logger.warning(
                        f"⚠️  {self._consec_losses} consecutive losses — "
                        f"TIGHT MODE ON (smaller sizes, fewer positions, higher AI threshold)"
                    )
        else:
            self._consec_losses = 0
            if self._tight_mode:
                self._tight_mode = False
                logger.info("✅ Win recorded — tight mode OFF")

    # ── State ─────────────────────────────────────────────────────

    def _halt(self, reason: str):
        self._halted      = True
        self._halt_reason = reason
        logger.critical(f"🛑 HALTED — {reason}")

    def reset_halt(self):
        self._halted      = False
        self._halt_reason = ""

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def tight_mode(self) -> bool:
        return self._tight_mode
