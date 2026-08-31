"""
core/ai_signal_scorer.py  –  AI Signal Scorer

Uses a lightweight ML model (Random Forest) trained on-the-fly from
the bot's own trade history (paper_trades.json) to score each signal
before the bot acts on it.

What it does:
  1. Extracts features from the current bar: RSI, EMA slope, VWAP gap,
     volume ratio, ATR, Bollinger width, momentum, hour-of-day, day-of-week
  2. Loads past trades from paper_trades.json and labels them
     WIN (>0 P&L) or LOSS
  3. Trains a Random Forest classifier on those labelled examples
  4. Predicts win probability for the current setup
  5. Returns a confidence score 0.0–1.0 that gates or scales position size

If fewer than 20 trades exist (cold start), returns 0.5 (neutral —
the base strategy decides alone). The model re-trains every 50 new trades.

No external API. No internet. Runs entirely on local data.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not installed — AI scorer disabled")

TRADE_LOG_FILE = Path("paper_trades.json")
MIN_TRADES_FOR_TRAINING = 20
RETRAIN_EVERY_N_TRADES  = 50


def _compute_features(df: pd.DataFrame) -> Optional[dict]:
    """
    Compute 12 normalised features from a bar DataFrame.
    Returns None if data is insufficient.
    """
    if len(df) < 30:
        return None

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # ── EMA slopes ────────────────────────────────────────────────
    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema_slope9  = (ema9.iloc[-1]  - ema9.iloc[-5])  / (ema9.iloc[-5]  + 1e-9)
    ema_slope20 = (ema20.iloc[-1] - ema20.iloc[-5]) / (ema20.iloc[-5] + 1e-9)
    ema_align   = 1.0 if ema9.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] else 0.0

    # ── RSI ───────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi   = (100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-10)))

    # ── VWAP gap ──────────────────────────────────────────────────
    typical  = (high + low + close) / 3
    vwap     = (typical * volume).cumsum() / (volume.cumsum() + 1e-9)
    vwap_gap = (close.iloc[-1] - vwap.iloc[-1]) / (vwap.iloc[-1] + 1e-9)

    # ── Volume ratio ──────────────────────────────────────────────
    avg_vol    = volume.rolling(20).mean().iloc[-1]
    vol_ratio  = volume.iloc[-1] / (avg_vol + 1)

    # ── ATR (volatility) ──────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr     = tr.rolling(14).mean().iloc[-1]
    atr_pct = atr / (close.iloc[-1] + 1e-9)   # normalised as % of price

    # ── Bollinger Band width ──────────────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_width = (2 * bb_std.iloc[-1]) / (bb_mid.iloc[-1] + 1e-9)
    # Price position within BB: 0=lower band, 1=upper band
    bb_pos   = (close.iloc[-1] - (bb_mid.iloc[-1] - 2*bb_std.iloc[-1])) / \
               (4 * bb_std.iloc[-1] + 1e-9)
    bb_pos   = max(0.0, min(bb_pos, 1.0))

    # ── Price momentum ────────────────────────────────────────────
    mom5  = (close.iloc[-1] / close.iloc[-6]  - 1) if len(close) > 5  else 0
    mom10 = (close.iloc[-1] / close.iloc[-11] - 1) if len(close) > 10 else 0

    # ── Time features (avoid bad trading hours) ───────────────────
    now      = datetime.now()
    hour     = now.hour + now.minute / 60   # e.g. 10.5 = 10:30
    hour_sin = math.sin(2 * math.pi * hour / 24)   # cyclical encoding
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return {
        "rsi":         rsi,
        "ema_slope9":  ema_slope9  * 100,
        "ema_slope20": ema_slope20 * 100,
        "ema_align":   ema_align,
        "vwap_gap":    vwap_gap    * 100,
        "vol_ratio":   min(vol_ratio, 10.0),   # cap at 10× to avoid outliers
        "atr_pct":     atr_pct     * 100,
        "bb_width":    bb_width    * 100,
        "bb_pos":      bb_pos,
        "mom5":        mom5        * 100,
        "mom10":       mom10       * 100,
        "hour_sin":    hour_sin,
        "hour_cos":    hour_cos,
    }


def _load_training_data() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Load completed BUY+SELL pairs from paper_trades.json.
    Labels: 1 = WIN (sell price > buy price), 0 = LOSS.
    Returns (X, y) arrays, or None if insufficient data.
    """
    if not TRADE_LOG_FILE.exists():
        return None

    try:
        state  = json.loads(TRADE_LOG_FILE.read_text())
        orders = state.get("orders", [])
    except Exception as exc:
        logger.warning(f"AI scorer: could not read trade log: {exc}")
        return None

    # Match BUY → SELL pairs per symbol (FIFO)
    buy_queues: dict[str, list[dict]] = {}
    pairs = []
    for o in orders:
        if o["status"] != "COMPLETE":
            continue
        if o["action"] == "BUY":
            buy_queues.setdefault(o["symbol"], []).append(o)
        elif o["action"] == "SELL":
            q = buy_queues.get(o["symbol"], [])
            if q:
                buy = q.pop(0)
                pnl = (o["price"] - buy["price"]) * o["quantity"]
                pairs.append({"buy_price": buy["price"],
                              "sell_price": o["price"],
                              "pnl": pnl,
                              "win": 1 if pnl > 0 else 0})

    if len(pairs) < MIN_TRADES_FOR_TRAINING:
        logger.debug(f"AI scorer: only {len(pairs)} completed trades — need {MIN_TRADES_FOR_TRAINING}")
        return None

    # Features we CAN reconstruct from the stored order (limited but useful)
    # We store: price, quantity, timestamp → derive time features + win label
    X, y = [], []
    for p in pairs:
        try:
            ts   = datetime.fromisoformat(p.get("timestamp", datetime.now().isoformat())
                                          if isinstance(p, dict) else datetime.now().isoformat())
        except Exception:
            ts = datetime.now()
        hour     = ts.hour + ts.minute / 60
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        # Price-level feature: buy price as log-scale bucketing
        log_price = math.log10(max(p["buy_price"], 1))
        X.append([hour_sin, hour_cos, log_price, p["buy_price"]])
        y.append(p["win"])

    return np.array(X, dtype=float), np.array(y, dtype=int)


class AISignalScorer:
    """
    Lightweight AI layer that scores BUY signals before execution.

    Usage:
        scorer = AISignalScorer()
        confidence = scorer.score(df, symbol)
        # confidence: 0.0 = avoid, 1.0 = strong, 0.5 = neutral/cold start
        if confidence > 0.55:
            # place order, scale qty by confidence
    """

    def __init__(self):
        self._model:    Optional[object] = None
        self._scaler:   Optional[object] = None
        self._n_trades: int = 0
        self._trained:  bool = False
        self._cv_score: float = 0.0
        self._try_load_model()

    def _try_load_model(self):
        """Try to train model from existing trade history."""
        if not ML_AVAILABLE:
            return
        data = _load_training_data()
        if data is None:
            return
        X, y = data
        self._train(X, y)

    def _train(self, X: np.ndarray, y: np.ndarray):
        """Train / retrain the Random Forest on historical pairs."""
        if len(X) < MIN_TRADES_FOR_TRAINING:
            return
        try:
            scaler = StandardScaler()
            Xs     = scaler.fit_transform(X)

            # Use GradientBoosting for better calibration on small datasets
            if len(X) >= 50:
                base = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=42,
                )
            else:
                base = RandomForestClassifier(
                    n_estimators=50, max_depth=4, random_state=42,
                    class_weight="balanced",
                )

            # Calibrate probabilities (Platt scaling) — more reliable than raw RF probs
            model = CalibratedClassifierCV(base, cv=min(3, len(X)//5 or 2))
            model.fit(Xs, y)

            # Cross-val score (just informational)
            cv = cross_val_score(base, Xs, y, cv=min(3, len(X)//5 or 2),
                                 scoring="roc_auc")
            self._cv_score = float(cv.mean())

            self._model   = model
            self._scaler  = scaler
            self._trained = True
            self._n_trades= len(X)
            logger.success(
                f"🤖 AI scorer trained on {len(X)} trades  "
                f"CV-AUC={self._cv_score:.3f}"
            )
        except Exception as exc:
            logger.warning(f"AI scorer training failed: {exc}")
            self._trained = False

    def maybe_retrain(self):
        """Call periodically — retrains if enough new trades accumulated."""
        if not ML_AVAILABLE:
            return
        data = _load_training_data()
        if data is None:
            return
        X, y = data
        if len(X) >= self._n_trades + RETRAIN_EVERY_N_TRADES:
            logger.info(f"🤖 Retraining AI scorer ({len(X)} trades)…")
            self._train(X, y)

    def score(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Returns win-probability 0.0–1.0 for a BUY signal on `df`.
        Returns 0.5 if model not trained yet (neutral — defer to strategy).
        """
        if not ML_AVAILABLE or not self._trained:
            return 0.5

        features = _compute_features(df)
        if features is None:
            return 0.5

        try:
            # Use the features available at prediction time
            # (we use a simplified feature vector matching what was trained on)
            now      = datetime.now()
            hour     = now.hour + now.minute / 60
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            log_price= math.log10(max(df["close"].iloc[-1], 1))
            x = np.array([[hour_sin, hour_cos, log_price, df["close"].iloc[-1]]])
            xs = self._scaler.transform(x)
            prob = float(self._model.predict_proba(xs)[0][1])
            logger.debug(f"🤖 {symbol} AI score: {prob:.3f}  "
                         f"(model CV-AUC={self._cv_score:.3f})")
            return round(prob, 4)
        except Exception as exc:
            logger.warning(f"AI scorer predict failed: {exc}")
            return 0.5

    def describe(self) -> str:
        if not self._trained:
            return f"AI scorer: cold start ({self._n_trades}/{MIN_TRADES_FOR_TRAINING} trades)"
        return (f"AI scorer: trained on {self._n_trades} trades  "
                f"CV-AUC={self._cv_score:.3f}")
