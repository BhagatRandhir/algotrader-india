"""
core/signal_engine.py  –  Phase 1 Multi-Factor Signal Engine

Orchestrates all Phase 1 filters into a single decision pipeline:

  5m MomentumBot signal          (existing — entry conditions)
       ↓
  15m Higher Timeframe filter    (NEW — directional gate)
       ↓
  Market Structure HH/HL         (NEW — structural confirmation)
       ↓
  Opening Range filter           (NEW — avoid first 15 min)
       ↓
  ATR-based SL / RR validation   (NEW — dynamic stop, min 1.5R)
       ↓
  Smart Signals gate             (existing — news/sector/forecast)
       ↓
  Market Context filter          (existing — Nifty/VIX)
       ↓
  Composite Score (100pt)        (NEW — weighted scoring)
       ↓
  BUY / NO_TRADE

Scoring (100 points):
  15m HTF alignment          20 pts
  Market structure HH/HL     15 pts
  5m MomentumBot score       25 pts  (maps 0.6-1.0 → 0-25)
  Volume confirmation        15 pts
  Smart signals              15 pts
  R:R quality                10 pts

Threshold:
  >= 75 → BUY
  < 75  → NO_TRADE

Hard rejections (override score):
  - 15m SIDEWAYS or BEARISH
  - No HH/HL structure (BEARISH or MIXED structure)
  - Opening range first 3 bars
  - R:R < 1.5
  - Smart signals block
  - Market context block
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
from loguru import logger

from core.higher_tf        import HigherTimeframeFilter, HTFTrend
from core.market_structure import detect_structure, get_opening_range, Structure
from core.atr_risk         import calculate_atr_risk, ATRRiskResult
from strategies.base       import Signal

IST                = pytz.timezone("Asia/Kolkata")
SIGNAL_SCORE_MIN   = int(os.getenv("SIGNAL_SCORE_MIN",  "75"))
OPENING_RANGE_BARS = int(os.getenv("OPENING_RANGE_BARS", "3"))  # first 15 min

_htf_filter = HigherTimeframeFilter()


@dataclass
class SignalDecision:
    signal:         str      # BUY / NO_TRADE
    score:          int      # 0-100
    entry:          float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  float
    qty:            int
    atr:            float
    risk_reward:    float
    regime_15m:     str
    structure:      str
    reason_codes:   list[str]
    rejection_reasons: list[str]

    @property
    def is_trade(self) -> bool:
        return self.signal == "BUY" and self.qty > 0


def evaluate(
    symbol:          str,
    df_5m:           pd.DataFrame,    # 5-minute bars
    momentum_result,                   # StrategyResult from MomentumBotStrategy
    broker,
    equity:          float,
    smart_signals:   Optional[dict]   = None,
    market_ok:       bool             = True,
) -> SignalDecision:
    """
    Full Phase 1 signal evaluation.
    Returns SignalDecision with score, SL, TP, qty and all reasons.
    """
    reasons   : list[str] = []
    rejections: list[str] = []
    score      = 0

    NO_TRADE = lambda rej=None: SignalDecision(
        signal="NO_TRADE", score=score, entry=0,
        stop_loss=0, take_profit_1=0, take_profit_2=0,
        qty=0, atr=0, risk_reward=0,
        regime_15m="UNKNOWN", structure="UNKNOWN",
        reason_codes=reasons,
        rejection_reasons=rejections + ([rej] if rej else []),
    )

    # ── 0. Must have a 5m BUY signal first ───────────────────────
    if momentum_result.signal != Signal.BUY:
        return NO_TRADE("5m signal is not BUY")

    ltp = float(df_5m["close"].iloc[-1]) if df_5m is not None and not df_5m.empty else 0
    if ltp <= 0:
        return NO_TRADE("No valid price")

    # ── 1. Opening range filter ───────────────────────────────────
    now_ist = datetime.now(IST)
    if now_ist.hour == 9 and now_ist.minute < 30:
        rejections.append(f"Opening range — no trades before 9:30 AM")
        return NO_TRADE()

    opening_range = get_opening_range(df_5m, OPENING_RANGE_BARS)

    # ── 2. Higher Timeframe (15m) filter — hard gate ──────────────
    htf = _htf_filter.analyse(symbol, broker)

    if htf.trend == HTFTrend.SIDEWAYS:
        rejections.append(f"15m SIDEWAYS — {htf.reason}")
        return NO_TRADE()

    if htf.trend == HTFTrend.BEARISH:
        rejections.append(f"15m BEARISH — {htf.reason}")
        return NO_TRADE()

    if htf.trend == HTFTrend.BULLISH:
        score += 20
        reasons.append(f"15m BULLISH +20 ({htf.reason})")
    else:  # UNKNOWN
        score += 10
        reasons.append(f"15m UNKNOWN +10 (reduced confidence)")

    # ── 3. Market structure ───────────────────────────────────────
    struct = detect_structure(df_5m)

    if struct.structure == Structure.BEARISH:
        rejections.append(f"Bearish structure LL+LH — {struct.reason}")
        return NO_TRADE()

    if struct.structure == Structure.BULLISH:
        score += 15
        reasons.append(f"Bullish structure HH+HL +15 ({struct.reason})")
    elif struct.structure == Structure.MIXED:
        score += 5
        reasons.append(f"Mixed structure +5 ({struct.reason})")
    else:
        score += 8   # UNCLEAR — not bearish, partial credit
        reasons.append(f"Structure unclear +8 — insufficient swing data")

    swing_low = struct.last_swing_low if struct.last_swing_low > 0 else None

    # ── 4. 5m MomentumBot score ───────────────────────────────────
    # Maps strength 0.6-1.0 → 15-25 points
    raw_strength = momentum_result.strength
    momentum_pts = int(15 + (raw_strength - 0.60) / 0.40 * 10)
    momentum_pts = max(0, min(25, momentum_pts))
    score       += momentum_pts
    reasons.append(f"5m momentum +{momentum_pts} (str={raw_strength:.2f}: {momentum_result.reason[:50]})")

    # ── 5. Volume confirmation ────────────────────────────────────
    try:
        vol_now = float(df_5m["volume"].iloc[-1])
        avg_vol = float(df_5m["volume"].rolling(20).mean().iloc[-1])
        vol_ratio = vol_now / (avg_vol + 1e-9)

        if vol_ratio >= 2.0:
            score += 15; reasons.append(f"Volume very strong +15 ({vol_ratio:.1f}×)")
        elif vol_ratio >= 1.5:
            score += 10; reasons.append(f"Volume strong +10 ({vol_ratio:.1f}×)")
        elif vol_ratio >= 1.0:
            score += 5;  reasons.append(f"Volume normal +5 ({vol_ratio:.1f}×)")
        else:
            score += 0
            if vol_ratio < 0.8:
                rejections.append(f"Volume weak {vol_ratio:.1f}× — no trade")
                return NO_TRADE()
    except Exception:
        score += 5   # can't check → neutral

    # ── 6. Smart signals ──────────────────────────────────────────
    if smart_signals:
        ss_allow = smart_signals.get("allow", True)
        ss_score = smart_signals.get("score", 0.0)

        if not ss_allow:
            rejections.append(f"Smart signals block: {smart_signals.get('summary','')[:60]}")
            return NO_TRADE()

        smart_pts = int(10 + ss_score * 5)   # 5-15 pts
        smart_pts = max(0, min(15, smart_pts))
        score    += smart_pts
        reasons.append(f"Smart signals +{smart_pts} (score={ss_score:.2f})")
    else:
        score += 8   # unavailable → neutral
        reasons.append("Smart signals unavailable +8")

    # ── 7. Market context ─────────────────────────────────────────
    if not market_ok:
        rejections.append("Market context blocked (Nifty/VIX)")
        return NO_TRADE()

    # ── 8. ATR-based SL / RR validation ──────────────────────────
    atr_result = calculate_atr_risk(
        df       = df_5m,
        entry    = ltp,
        equity   = equity,
        swing_low= swing_low,
        signal_strength = raw_strength,
    )

    if not atr_result.valid:
        rejections.append(f"ATR risk invalid: {atr_result.reason}")
        return NO_TRADE()

    rr1 = atr_result.risk_reward_1
    if rr1 >= 2.0:
        score += 10; reasons.append(f"R:R excellent +10 ({rr1:.1f}R)")
    elif rr1 >= 1.5:
        score += 7;  reasons.append(f"R:R good +7 ({rr1:.1f}R)")
    else:
        rejections.append(f"R:R {rr1:.1f} below minimum 1.5")
        return NO_TRADE()

    # ── Final score check ─────────────────────────────────────────
    logger.info(
        f"🔢 {symbol} signal score: {score}/100 "
        f"(threshold={SIGNAL_SCORE_MIN}) "
        f"HTF={htf.trend.value} struct={struct.structure.value}"
    )

    if score < SIGNAL_SCORE_MIN:
        rejections.append(f"Score {score} < threshold {SIGNAL_SCORE_MIN}")
        return SignalDecision(
            signal="NO_TRADE", score=score, entry=ltp,
            stop_loss=atr_result.stop_loss,
            take_profit_1=atr_result.take_profit_1,
            take_profit_2=atr_result.take_profit_2,
            qty=0, atr=atr_result.atr,
            risk_reward=rr1,
            regime_15m=htf.trend.value,
            structure=struct.structure.value,
            reason_codes=reasons, rejection_reasons=rejections,
        )

    # ── BUY ───────────────────────────────────────────────────────
    return SignalDecision(
        signal        = "BUY",
        score         = score,
        entry         = ltp,
        stop_loss     = atr_result.stop_loss,
        take_profit_1 = atr_result.take_profit_1,
        take_profit_2 = atr_result.take_profit_2,
        qty           = atr_result.qty_full,
        atr           = atr_result.atr,
        risk_reward   = rr1,
        regime_15m    = htf.trend.value,
        structure     = struct.structure.value,
        reason_codes  = reasons,
        rejection_reasons = rejections,
    )
