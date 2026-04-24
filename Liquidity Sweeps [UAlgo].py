"""Python translation of `Liquidity Sweeps [UAlgo]` Pine Script.

Source: `S1_Silver_Bullet/Liquidity Sweeps [UAlgo].txt` (© UAlgo, MPL-2.0).

Detection logic is preserved 1:1 with the Pine source; only purely-cosmetic
TradingView settings (colours, line widths, label styles) are dropped.

Pine semantics that matter for faithful mirroring
-------------------------------------------------
1.  `ta.pivothigh(high, L, R)` with ``L == R == pivotPeriod`` is **strict**:
    the centre bar's high must be strictly greater than every other high in
    the ``[centre-L, centre+R]`` window. The pivot is *confirmed* on the bar
    that sits ``R`` bars after the centre.
2.  When a pivot is confirmed, a new "line" (level) is pushed onto the
    resistance / support array. If the array size exceeds ``maxLine`` the
    **oldest** entry is dropped (Pine: ``array.get(…, 0)`` then
    ``array.remove(…, 0)``).
3.  On every bar, every active level is checked (Pine uses a for-loop over
    the array). A resistance level is resolved if:
       * ``high > level  AND  close < level``  → *buy liquidity sweep*.
       * ``close > level``                     → invalidation (breakout).
    A support level mirrors the above:
       * ``low  < level  AND  close > level``  → *sell liquidity sweep*.
       * ``close < level``                     → invalidation (breakdown).
4.  Pine's ATR(14) is used only to position a text label; it does **not**
    participate in detection. It is therefore dropped here.

The module emits plain Python objects so the ICT pipeline adapter can
serialise them without extra plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class SweepLevel:
    """A pivot-based S/R level, mirrored from a Pine `line` object."""

    side: str  # "resistance" | "support"
    pivot_time: pd.Timestamp  # centre-bar timestamp (when the pivot formed)
    pivot_price: float
    created_time: pd.Timestamp  # bar on which the pivot was *confirmed*


@dataclass
class SweepEvent:
    """A resolved level: either a liquidity sweep or a breakout."""

    kind: str  # "buy_liquidity_sweep" | "sell_liquidity_sweep" | "breakout_up" | "breakdown"
    side: str  # "resistance" | "support"
    pivot_time: pd.Timestamp
    pivot_price: float
    event_time: pd.Timestamp
    event_price: float  # high for buy sweep, low for sell sweep, close otherwise


@dataclass
class LiquiditySweepsResult:
    events: List[SweepEvent] = field(default_factory=list)
    active_resistance: List[SweepLevel] = field(default_factory=list)
    active_support: List[SweepLevel] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pivot detection helper — matches Pine's strict `ta.pivothigh/pivotlow`
# ---------------------------------------------------------------------------
def _is_strict_pivot_high(series: pd.Series, centre_idx: int, window: int) -> bool:
    lo = centre_idx - window
    hi = centre_idx + window
    if lo < 0 or hi >= len(series):
        return False
    pivot = series.iat[centre_idx]
    for j in range(lo, hi + 1):
        if j == centre_idx:
            continue
        if series.iat[j] >= pivot:
            return False
    return True


def _is_strict_pivot_low(series: pd.Series, centre_idx: int, window: int) -> bool:
    lo = centre_idx - window
    hi = centre_idx + window
    if lo < 0 or hi >= len(series):
        return False
    pivot = series.iat[centre_idx]
    for j in range(lo, hi + 1):
        if j == centre_idx:
            continue
        if series.iat[j] <= pivot:
            return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_liquidity_sweeps(
    df: pd.DataFrame,
    *,
    pivot_period: int = 20,
    max_lines: int = 3,
) -> LiquiditySweepsResult:
    """Run `Liquidity Sweeps [UAlgo]` on an OHLC DataFrame.

    Parameters mirror the Pine inputs:
        ``pivot_period`` ⇄ ``pivotPeriod`` (default 20)
        ``max_lines``    ⇄ ``maxLine``     (default 3)

    Input
    -----
    df : pd.DataFrame indexed by DatetimeIndex (chronological, ascending)
         with columns ``open``, ``high``, ``low``, ``close``. Extra columns
         are ignored.
    """
    if pivot_period < 1:
        raise ValueError("pivot_period must be >= 1")
    if max_lines < 1:
        raise ValueError("max_lines must be >= 1")
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")
    if len(df) < 2 * pivot_period + 1:
        return LiquiditySweepsResult()

    highs = df["high"]
    lows = df["low"]
    closes = df["close"]
    index = df.index

    resistance: List[SweepLevel] = []
    support: List[SweepLevel] = []
    events: List[SweepEvent] = []

    # Pine evaluates per-bar: (1) detect pivot confirmation at `i`, (2) push
    # new line if confirmed, (3) iterate active lines and resolve. We mirror
    # that ordering so newly-added lines are also checked on the confirm bar
    # (same as Pine — though by definition of a strict pivot this cannot
    # resolve on the same bar).
    for i in range(len(df)):
        confirm_idx = i
        centre_idx = i - pivot_period

        if centre_idx >= pivot_period:  # both left & right windows exist
            # Pivot high confirmation → push resistance
            if _is_strict_pivot_high(highs, centre_idx, pivot_period):
                lvl = SweepLevel(
                    side="resistance",
                    pivot_time=index[centre_idx],
                    pivot_price=float(highs.iat[centre_idx]),
                    created_time=index[confirm_idx],
                )
                resistance.append(lvl)
                if len(resistance) > max_lines:
                    resistance.pop(0)  # Pine: array.remove(…, 0) — oldest dropped

            # Pivot low confirmation → push support
            if _is_strict_pivot_low(lows, centre_idx, pivot_period):
                lvl = SweepLevel(
                    side="support",
                    pivot_time=index[centre_idx],
                    pivot_price=float(lows.iat[centre_idx]),
                    created_time=index[confirm_idx],
                )
                support.append(lvl)
                if len(support) > max_lines:
                    support.pop(0)

        # --- Check active resistance lines against current bar ---
        high_i = float(highs.iat[i])
        low_i = float(lows.iat[i])
        close_i = float(closes.iat[i])
        t_i = index[i]

        # Iterate high-to-low index so in-loop removal stays safe (Pine does
        # `for i = size-1 to 0`).
        for k in range(len(resistance) - 1, -1, -1):
            lvl = resistance[k]
            price = lvl.pivot_price
            if high_i > price and close_i < price:
                events.append(
                    SweepEvent(
                        kind="buy_liquidity_sweep",
                        side="resistance",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        event_time=t_i,
                        event_price=high_i,
                    )
                )
                resistance.pop(k)
            elif close_i > price:
                events.append(
                    SweepEvent(
                        kind="breakout_up",
                        side="resistance",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                resistance.pop(k)

        # --- Check active support lines ---
        for k in range(len(support) - 1, -1, -1):
            lvl = support[k]
            price = lvl.pivot_price
            if low_i < price and close_i > price:
                events.append(
                    SweepEvent(
                        kind="sell_liquidity_sweep",
                        side="support",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        event_time=t_i,
                        event_price=low_i,
                    )
                )
                support.pop(k)
            elif close_i < price:
                events.append(
                    SweepEvent(
                        kind="breakdown",
                        side="support",
                        pivot_time=lvl.pivot_time,
                        pivot_price=price,
                        event_time=t_i,
                        event_price=close_i,
                    )
                )
                support.pop(k)

    return LiquiditySweepsResult(
        events=events,
        active_resistance=list(resistance),
        active_support=list(support),
    )


__all__ = [
    "SweepLevel",
    "SweepEvent",
    "LiquiditySweepsResult",
    "compute_liquidity_sweeps",
]
