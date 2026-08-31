"""
core/volume_profile.py  –  Volume Profile & Order Flow Analysis

Three checks before entry:
  1. VWAP bands — price must be between VWAP and VWAP+1σ (not overextended)
  2. Volume delta — more up-volume than down-volume in last 3 bars
  3. Wick rejection — no long upper wicks (selling pressure overhead)

Also detects volume dry-up for exit signals while holding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class VolumeProfileResult:
    allow_entry:    bool
    exit_signal:    bool    # True = volume dying, consider exit
    vwap:           float
    vwap_upper:     float   # VWAP + 1σ
    vwap_lower:     float   # VWAP - 1σ
    price_zone:     str     # "SWEET", "EXTENDED", "BELOW_VWAP"
    volume_delta:   float   # positive = buyers dominating
    wick_ratio:     float   # upper_wick / total_range (lower = better)
    size_mult:      float   # 1.0 = full, 0.7 = partial, 0.0 = skip
    reason:         str


class VolumeProfileAnalyser:

    def __init__(
        self,
        vwap_band_mult:   float = 2.0,   # σ bands around VWAP
        min_delta_ratio:  float = 0.50,  # need 55%+ buyer dominance
        max_wick_ratio:   float = 0.55,  # reject if upper wick > 40% of range
        dry_up_mult:      float = 0.50,  # exit if volume < 50% of avg
    ):
        self.vwap_band_mult  = vwap_band_mult
        self.min_delta_ratio = min_delta_ratio
        self.max_wick_ratio  = max_wick_ratio
        self.dry_up_mult     = dry_up_mult

    # ── VWAP + bands ─────────────────────────────────────────────

    def _vwap_bands(self, df: pd.DataFrame) -> tuple[float, float, float]:
        typical  = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol  = df["volume"].cumsum().replace(0, np.nan)
        vwap     = (typical * df["volume"]).cumsum() / cum_vol

        # Standard deviation of price from VWAP
        deviation = (typical - vwap) ** 2
        cum_dev   = (deviation * df["volume"]).cumsum() / cum_vol
        sigma     = np.sqrt(cum_dev)

        vwap_val  = float(vwap.iloc[-1])
        sigma_val = float(sigma.iloc[-1])
        upper     = vwap_val + self.vwap_band_mult * sigma_val
        lower     = vwap_val - self.vwap_band_mult * sigma_val
        return round(vwap_val, 2), round(upper, 2), round(lower, 2)

    # ── Volume delta ──────────────────────────────────────────────

    def _volume_delta(self, df: pd.DataFrame, lookback: int = 3) -> float:
        """
        Approximate buyer vs seller volume.
        Up-bar (close > open) → buyer volume.
        Down-bar → seller volume.
        Returns buyer_ratio: 0.0 to 1.0 (0.5 = balanced)
        """
        recent = df.iloc[-lookback:]
        up_vol   = recent.loc[recent["close"] >= recent["open"], "volume"].sum()
        down_vol = recent.loc[recent["close"] <  recent["open"], "volume"].sum()
        total    = up_vol + down_vol
        if total == 0:
            return 0.5
        return round(float(up_vol / total), 4)

    # ── Wick analysis ─────────────────────────────────────────────

    def _wick_ratio(self, df: pd.DataFrame) -> float:
        """
        Upper wick ratio on last bar.
        upper_wick / total_range — high ratio = selling pressure.
        """
        last  = df.iloc[-1]
        total = last["high"] - last["low"]
        if total < 1e-9:
            return 0.0
        body_top   = max(last["open"], last["close"])
        upper_wick = last["high"] - body_top
        return round(float(upper_wick / total), 4)

    # ── Volume dry-up ─────────────────────────────────────────────

    def _is_drying_up(self, df: pd.DataFrame) -> bool:
        """True if last 2 bars' volume is below dry_up_mult × 20-bar average."""
        avg_vol = df["volume"].iloc[-22:-2].mean()
        recent  = df["volume"].iloc[-2:].mean()
        return bool(recent < self.dry_up_mult * avg_vol)

    # ── Main analyse ──────────────────────────────────────────────

    def analyse(self, df: pd.DataFrame, symbol: str = "") -> VolumeProfileResult:
        SKIP = VolumeProfileResult(
            allow_entry=False, exit_signal=False,
            vwap=0, vwap_upper=0, vwap_lower=0,
            price_zone="NO_DATA", volume_delta=0.5,
            wick_ratio=0, size_mult=0.0,
            reason="insufficient data",
        )

        if df is None or len(df) < 25:
            return SKIP

        try:
            price              = float(df["close"].iloc[-1])
            vwap, upper, lower = self._vwap_bands(df)
            vol_delta          = self._volume_delta(df)
            wick               = self._wick_ratio(df)
            drying             = self._is_drying_up(df)

            # ── Price zone ────────────────────────────────────────
            if lower <= price <= upper:
                zone      = "SWEET"       # between bands = ideal
                zone_mult = 1.0
            elif price > upper:
                zone      = "EXTENDED"    # above upper band = overextended
                zone_mult = 0.0
            else:
                zone      = "BELOW_VWAP"
                zone_mult = 0.0

            # ── Volume delta gate ─────────────────────────────────
            if vol_delta >= self.min_delta_ratio:
                delta_ok   = True
                delta_mult = 1.0
            elif vol_delta >= 0.45:
                delta_ok   = True     # borderline — allow but reduce size
                delta_mult = 0.75
            else:
                delta_ok   = False
                delta_mult = 0.0

            # ── Wick gate ─────────────────────────────────────────
            wick_ok = wick <= self.max_wick_ratio

            # ── Final decision ────────────────────────────────────
            allow  = zone_mult > 0 and delta_ok and wick_ok
            s_mult = zone_mult * delta_mult if allow else 0.0

            reasons = []
            if zone == "SWEET":       reasons.append(f"VWAP zone✓ ({lower:.0f}–{upper:.0f})")
            if zone == "EXTENDED":    reasons.append(f"Price overextended above VWAP+σ")
            if zone == "BELOW_VWAP":  reasons.append(f"Price below VWAP")
            if delta_ok:              reasons.append(f"Buyers {vol_delta:.0%}✓")
            else:                     reasons.append(f"Sellers dominating ({vol_delta:.0%})")
            if wick_ok:               reasons.append(f"Clean wick✓ ({wick:.0%})")
            else:                     reasons.append(f"Upper wick too long ({wick:.0%})")
            if drying:                reasons.append("⚠️ Volume drying up")

            logger.debug(
                f"VP {symbol}: zone={zone} delta={vol_delta:.2f} "
                f"wick={wick:.2f} allow={allow} mult={s_mult:.2f}"
            )

            return VolumeProfileResult(
                allow_entry  = allow,
                exit_signal  = drying,
                vwap         = vwap,
                vwap_upper   = upper,
                vwap_lower   = lower,
                price_zone   = zone,
                volume_delta = vol_delta,
                wick_ratio   = wick,
                size_mult    = round(s_mult, 2),
                reason       = " | ".join(reasons),
            )
        except Exception as exc:
            logger.warning(f"VolumeProfile {symbol}: {exc}")
            return SKIP
