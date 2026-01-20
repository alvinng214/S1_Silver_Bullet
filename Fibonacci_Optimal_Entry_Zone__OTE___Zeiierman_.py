"""Fibonacci Optimal Entry Zone [OTE] (Zeiierman).

Python translation of the Pine Script logic for swing tracking and Fibonacci OTE
levels. This mirrors the Pine flow by:
- Tracking pivots with ta.pivothigh/ta.pivotlow semantics.
- Maintaining Up/Dn swing bounds and direction state (pos).
- Emitting CHoCH events when structure flips.
- Updating Fibonacci levels using the same fibb() logic and follow mode.
- Extending levels forward when requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FibLevelState:
    index: int
    price: float


@dataclass
class StructureEvent:
    index: int
    price: float
    label: str
    bullish: bool


@dataclass
class FibonacciState:
    pos: int
    up: float
    dn: float
    i_up: int
    i_dn: int
    levels: List[float]


def _pivot_high(highs: np.ndarray, left: int, right: int, idx: int) -> Optional[float]:
    if idx < left or idx + right >= len(highs):
        return None
    pivot = highs[idx]
    for i in range(idx - left, idx + right + 1):
        if i != idx and highs[i] >= pivot:
            return None
    return pivot


def _pivot_low(lows: np.ndarray, left: int, right: int, idx: int) -> Optional[float]:
    if idx < left or idx + right >= len(lows):
        return None
    pivot = lows[idx]
    for i in range(idx - left, idx + right + 1):
        if i != idx and lows[i] <= pivot:
            return None
    return pivot


def _fibb(v: float, high: float, low: float, i_high: int, i_low: int) -> float:
    if i_low < i_high:
        return high - (high - low) * v
    if i_low > i_high:
        return low + (high - low) * v
    return np.nan


def calculate_fibonacci_ote(
    df: pd.DataFrame,
    *,
    pivot_period: int = 10,
    levels: List[float] = None,
    follow: bool = True,
    show_old: bool = False,
    extend: bool = True,
    enable_bull: bool = True,
    enable_bear: bool = True,
) -> Dict[str, object]:
    """Compute Fibonacci OTE structures and levels.

    Args:
        df: DataFrame with columns open, high, low, close.
        pivot_period: Pivot strength for ta.pivothigh/low (left/right).
        levels: Fibonacci levels to compute (default [0.5, 0.618]).
        follow: Whether to follow the latest swing points dynamically.
        show_old: Keep previous structures (otherwise overwrite).
        extend: Extend levels to current bar (flag in output).
        enable_bull: Allow bullish structure tracking.
        enable_bear: Allow bearish structure tracking.

    Returns:
        Dict with per-bar FibonacciState, events, and level history.
    """
    if levels is None:
        levels = [0.5, 0.618]

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    up = highs[0]
    dn = lows[0]
    i_up = 0
    i_dn = 0
    pos = 0

    swing_low = np.nan
    swing_high = np.nan
    i_swing_low = 0
    i_swing_high = 0

    level_history: List[List[float]] = []
    events: List[StructureEvent] = []
    states: List[FibonacciState] = []

    for b in range(len(df)):
        up = max(up, highs[b])
        dn = min(dn, lows[b])

        pvt_hi = _pivot_high(highs, pivot_period, pivot_period, b)
        pvt_lo = _pivot_low(lows, pivot_period, pivot_period, b)

        if pvt_hi is not None and pos <= 0:
            up = pvt_hi
        if pvt_lo is not None and pos >= 0:
            dn = pvt_lo

        if b > 0 and up > highs[b - 1]:
            i_up = b
            if pos <= 0:
                if enable_bull:
                    center = int(round((i_up - 1 + b) / 2))
                    events.append(StructureEvent(center, highs[b - 1], "CHoCH", True))
                    level_values = [_fibb(l, up, dn, i_up, i_dn) for l in levels]
                    level_history.append(level_values)
                pos = 1
                swing_low = dn
                i_swing_low = i_dn
            elif pos >= 1:
                if enable_bull:
                    target_low = dn if follow else swing_low
                    target_i = i_dn if follow else i_swing_low
                    level_values = [_fibb(l, up, target_low, i_up, target_i) for l in levels]
                    if level_history:
                        level_history[-1] = level_values
                    else:
                        level_history.append(level_values)
                pos = pos + 1 if pos > 1 else 2
        elif b > 0 and up < highs[b - 1]:
            i_up = b - pivot_period

        if b > 0 and dn < lows[b - 1]:
            i_dn = b
            if pos >= 0:
                if enable_bear:
                    center = int(round((i_dn - 1 + b) / 2))
                    events.append(StructureEvent(center, lows[b - 1], "CHoCH", False))
                    level_values = [_fibb(l, dn, up, i_dn, i_up) for l in levels]
                    level_history.append(level_values)
                pos = -1
                swing_high = up
                i_swing_high = i_up
            elif pos <= -1:
                if enable_bear:
                    target_high = up if follow else swing_high
                    target_i = i_up if follow else i_swing_high
                    level_values = [_fibb(l, target_high, dn, target_i, i_dn) for l in levels]
                    if level_history:
                        level_history[-1] = level_values
                    else:
                        level_history.append(level_values)
                pos = pos - 1 if pos < -1 else -2
        elif b > 0 and dn > lows[b - 1]:
            i_dn = b - pivot_period

        if not show_old and len(level_history) > 1:
            level_history = level_history[-1:]

        current_levels = level_history[-1] if level_history else [np.nan for _ in levels]
        states.append(FibonacciState(pos=pos, up=up, dn=dn, i_up=i_up, i_dn=i_dn, levels=current_levels))

    return {
        "states": states,
        "events": events,
        "levels": level_history,
        "extend": extend,
    }
