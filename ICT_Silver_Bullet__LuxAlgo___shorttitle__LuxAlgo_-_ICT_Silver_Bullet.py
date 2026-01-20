"""
ICT Silver Bullet [LuxAlgo] - Python Translation

Replicates the LuxAlgo Pine Script logic for ICT Silver Bullet sessions by:
- Tracking Silver Bullet sessions (LN/AM/PM) in America/New_York time.
- Building swing pivots, zigzag points, and market structure shifts (MSS).
- Creating and managing FVG boxes with strict/super-strict activation rules.
- Generating target lines from session pivots with hit detection.
- Preserving per-bar state for downstream plotting or backtesting.

License: CC BY-NC-SA 4.0
Original: LuxAlgo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass
class Pivot:
    index: int
    price: float


@dataclass
class ZigZagState:
    d: List[int]
    x: List[int]
    y: List[float]


@dataclass
class FVGBox:
    left: int
    right: int
    top: float
    bottom: float
    active: bool
    current: bool
    bullish: bool
    visible: bool = True


@dataclass
class ActLine:
    index: int
    price: float
    active: bool = True


@dataclass
class SessionState:
    swing_highs: List[Pivot] = field(default_factory=list)
    swing_lows: List[Pivot] = field(default_factory=list)
    min_pivot: float = 1e7
    max_pivot: float = 0.0
    target_highs: List[ActLine] = field(default_factory=list)
    target_lows: List[ActLine] = field(default_factory=list)


@dataclass
class BarState:
    index: int
    in_sb: bool
    in_ln: bool
    in_am: bool
    in_pm: bool
    start_sb: bool
    end_sb: bool
    trend: int
    target_hit_high: bool
    target_hit_low: bool


@dataclass
class TargetHit:
    index: int
    direction: str
    price: float


NY_TZ = ZoneInfo("America/New_York")


def _ny_time(ts: pd.Timestamp) -> datetime:
    if ts.tzinfo is None:
        return ts.tz_localize(NY_TZ).to_pydatetime()
    return ts.tz_convert(NY_TZ).to_pydatetime()


def _session_flags(ts: pd.Timestamp) -> Tuple[bool, bool, bool, bool]:
    dt = _ny_time(ts)
    in_ln = dt.hour == 3
    in_am = dt.hour == 10
    in_pm = dt.hour == 14
    in_sb = in_ln or in_am or in_pm
    return in_sb, in_ln, in_am, in_pm


def _pivot_at(values: pd.Series, index: int, left: int, right: int, is_high: bool) -> Optional[float]:
    pivot_idx = index - right
    if pivot_idx - left < 0 or pivot_idx + right >= len(values):
        return None
    center = float(values.iloc[pivot_idx])
    before = values.iloc[pivot_idx - left : pivot_idx]
    after = values.iloc[pivot_idx + 1 : pivot_idx + right + 1]
    if is_high:
        if (before >= center).any() or (after >= center).any():
            return None
    else:
        if (before <= center).any() or (after <= center).any():
            return None
    return center


def _in_out(zz: ZigZagState, direction: int, x2: int, y2: float) -> None:
    zz.d.insert(0, direction)
    zz.x.insert(0, x2)
    zz.y.insert(0, y2)
    zz.d.pop()
    zz.x.pop()
    zz.y.pop()


def _set_trend(zz: ZigZagState, trend: int, close: float) -> int:
    if len(zz.d) < 3:
        return trend
    i_h = 2 if zz.d[2] == 1 else 1
    i_l = 2 if zz.d[2] == -1 else 1
    y_h = zz.y[i_h]
    y_l = zz.y[i_l]
    if y_h is not None and not np.isnan(y_h) and close > y_h and zz.d[i_h] == 1 and trend < 1:
        return 1
    if y_l is not None and not np.isnan(y_l) and close < y_l and zz.d[i_l] == -1 and trend > -1:
        return -1
    return trend


def _minimum_trade_framework(symbol_type: str, min_tick: float) -> float:
    if symbol_type == "forex":
        return min_tick * 15 * 10
    if symbol_type in {"index", "futures"}:
        return min_tick * 40
    return 0.0


def calculate_luxalgo_silver_bullet(
    df: pd.DataFrame,
    *,
    pivot_left: int = 5,
    pivot_right: int = 1,
    filter_mode: str = "Super-Strict",
    extend_fvg: bool = True,
    target_mode: str = "previous session (similar)",
    keep_lines: bool = True,
    symbol_type: str = "forex",
    min_tick: float = 0.0001,
) -> Dict[str, object]:
    """Replicate LuxAlgo ICT Silver Bullet logic.

    Args:
        df: DataFrame with columns open, high, low, close and datetime index.
        pivot_left: Left bars for pivots.
        pivot_right: Right bars for pivots (Pine uses 1).
        filter_mode: 'All FVG', 'Only FVG in the same direction of trend', 'Strict', 'Super-Strict'.
        extend_fvg: Extend FVG boxes while active.
        target_mode: 'previous session (any)' or 'previous session (similar)'.
        keep_lines: Keep target lines across sessions when strict.
        symbol_type: 'forex', 'index', 'futures', etc.
        min_tick: Minimum tick size for the instrument.
    """
    max_size = 250
    zz = ZigZagState(d=[0] * max_size, x=[0] * max_size, y=[np.nan] * max_size)
    trend = 0

    use_trend = filter_mode != "All FVG"
    strict = filter_mode == "Strict"
    super_strict = filter_mode == "Super-Strict"
    strict_mode = strict or super_strict

    sessions: Dict[str, SessionState] = {
        "GN": SessionState(),
        "LN": SessionState(),
        "AM": SessionState(),
        "PM": SessionState(),
    }

    fvg_bull: List[FVGBox] = []
    fvg_bear: List[FVGBox] = []
    highs: List[ActLine] = []
    lows: List[ActLine] = []
    target_hits: List[TargetHit] = []
    bar_states: List[BarState] = []

    min_val = 1e7
    max_val = 0.0
    hilo_high = 0.0
    hilo_low = 1e7

    last_in_sb = False
    last_in_ln = False
    last_in_am = False
    last_in_pm = False

    for i in range(len(df)):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        close = float(df["close"].iloc[i])
        ts = df.index[i]

        in_sb, in_ln, in_am, in_pm = _session_flags(ts)
        start_sb = in_sb and not last_in_sb
        end_sb = (not in_sb) and last_in_sb
        start_ln = in_ln and not last_in_ln
        start_am = in_am and not last_in_am
        start_pm = in_pm and not last_in_pm
        end_ln = (not in_ln) and last_in_ln
        end_am = (not in_am) and last_in_am
        end_pm = (not in_pm) and last_in_pm

        trend = _set_trend(zz, trend, close)

        target_hit_high = False
        target_hit_low = False

        if len(highs) > 200:
            highs.pop(0)
        if len(lows) > 200:
            lows.pop(0)

        for line in highs:
            if line.active and high > line.price:
                line.active = False
                target_hit_high = True
                target_hits.append(TargetHit(i, "high", line.price))
        for line in lows:
            if line.active and low < line.price:
                line.active = False
                target_hit_low = True
                target_hits.append(TargetHit(i, "low", line.price))

        if start_sb:
            min_val = 1e7
            max_val = 0.0
            for box in fvg_bull:
                if i > box.right - 1 and box.current:
                    box.current = False
            for box in fvg_bear:
                if i > box.right - 1 and box.current:
                    box.current = False

        if in_sb:
            if use_trend:
                if trend == 1 and i >= 2 and low > float(df["high"].iloc[i - 2]):
                    top = low
                    bottom = float(df["high"].iloc[i - 2])
                    fvg_bull.insert(0, FVGBox(i - 2, i, top, bottom, active=False, current=True, bullish=True))
                elif trend != 1 and i >= 2 and high < float(df["low"].iloc[i - 2]):
                    top = float(df["low"].iloc[i - 2])
                    bottom = high
                    fvg_bear.insert(0, FVGBox(i - 2, i, top, bottom, active=False, current=True, bullish=False))
            else:
                if i >= 2 and low > float(df["high"].iloc[i - 2]):
                    top = low
                    bottom = float(df["high"].iloc[i - 2])
                    fvg_bull.insert(0, FVGBox(i, i, top, bottom, active=False, current=True, bullish=True))
                if i >= 2 and high < float(df["low"].iloc[i - 2]):
                    top = float(df["low"].iloc[i - 2])
                    bottom = high
                    fvg_bear.insert(0, FVGBox(i, i, top, bottom, active=False, current=True, bullish=False))

        for box in fvg_bull:
            if i - box.left >= 1000:
                continue
            if box.current:
                if in_sb:
                    if close < box.bottom:
                        if super_strict:
                            box.current = False
                            box.visible = False
                            box.right = box.left
                        if strict_mode:
                            box.active = False
                    else:
                        if extend_fvg and box.active:
                            box.right = i
                    if not box.active:
                        if low < box.top and close > box.bottom:
                            box.active = True
                            if extend_fvg:
                                box.right = i
                if end_sb:
                    if box.active:
                        if strict and close < box.bottom:
                            box.active = False
                        if super_strict and close < box.top:
                            box.active = False
                    if not box.active:
                        box.visible = False
                        box.right = box.left
                    if box.active:
                        min_val = min(min_val, box.bottom + _minimum_trade_framework(symbol_type, min_tick))
                        if extend_fvg:
                            box.right = i
                if last_in_sb and not in_sb:
                    box.active = False

        for box in fvg_bear:
            if i - box.left >= 1000:
                continue
            if box.current:
                if in_sb:
                    if close > box.top:
                        if super_strict:
                            box.current = False
                            box.visible = False
                            box.right = box.left
                        if strict_mode:
                            box.active = False
                    else:
                        if extend_fvg and box.active:
                            box.right = i
                    if not box.active:
                        if high > box.bottom and close < box.top:
                            box.active = True
                            if extend_fvg:
                                box.right = i
                if end_sb:
                    if box.active:
                        if strict and close > box.top:
                            box.active = False
                        if super_strict and close > box.bottom:
                            box.active = False
                    if not box.active:
                        box.visible = False
                        box.right = box.left
                    if box.active:
                        max_val = max(max_val, box.top - _minimum_trade_framework(symbol_type, min_tick))
                        if extend_fvg:
                            box.right = i
                if last_in_sb and not in_sb:
                    box.active = False

        ph = _pivot_at(df["high"], i, pivot_left, pivot_right, is_high=True)
        pl = _pivot_at(df["low"], i, pivot_left, pivot_right, is_high=False)

        def apply_swings(session_key: str, start: bool, end: bool, active: bool) -> None:
            nonlocal trend
            nonlocal hilo_high, hilo_low
            state = sessions[session_key]
            if start:
                hilo_high = 0.0
                hilo_low = 1e7
                if strict_mode:
                    should_clear = not keep_lines
                else:
                    should_clear = True
                if should_clear:
                    highs.clear()
                    lows.clear()
                    state.target_highs.clear()
                    state.target_lows.clear()
                    for other in sessions.values():
                        other.target_highs.clear()
                        other.target_lows.clear()
                state.min_pivot = 1e7
                state.max_pivot = 0.0
            if active:
                hilo_high = max(hilo_high, high)
                hilo_low = min(hilo_low, low)
            if ph is not None:
                state.max_pivot = max(state.max_pivot, ph)
                state.swing_highs = [p for p in state.swing_highs if ph < p.price]
                state.swing_highs.insert(0, Pivot(i - 1, ph))
                if session_key in {"GN", "LN"}:
                    dir0 = zz.d[0]
                    y1 = zz.y[0]
                    y2 = float(df["high"].iloc[i - 1])
                    if dir0 < 1:
                        _in_out(zz, 1, i - 1, y2)
                    else:
                        if dir0 == 1 and ph > y1:
                            zz.x[0] = i - 1
                            zz.y[0] = y2
            if pl is not None:
                state.min_pivot = min(state.min_pivot, pl)
                state.swing_lows = [p for p in state.swing_lows if pl > p.price]
                state.swing_lows.insert(0, Pivot(i - 1, pl))
                if session_key in {"GN", "LN"}:
                    dir0 = zz.d[0]
                    y1 = zz.y[0]
                    y2 = float(df["low"].iloc[i - 1])
                    if dir0 > -1:
                        _in_out(zz, -1, i - 1, y2)
                    else:
                        if dir0 == -1 and pl < y1:
                            zz.x[0] = i - 1
                            zz.y[0] = y2

            i_h = 2 if zz.d[2] == 1 else 1
            i_l = 2 if zz.d[2] == -1 else 1
            y_h = zz.y[i_h]
            y_l = zz.y[i_l]
            if y_h is not None and not np.isnan(y_h) and close > y_h and zz.d[i_h] == 1 and trend < 1:
                trend = 1
            if y_l is not None and not np.isnan(y_l) and close < y_l and zz.d[i_l] == -1 and trend > -1:
                trend = -1

            if end:
                for pivot in state.swing_highs:
                    if pivot.price > (min_val if strict_mode else hilo_high):
                        state.target_highs.insert(0, ActLine(pivot.index, pivot.price, True))
                        highs.insert(0, ActLine(i, pivot.price, True))
                for pivot in state.swing_lows:
                    if pivot.price < (max_val if strict_mode else hilo_low):
                        state.target_lows.insert(0, ActLine(pivot.index, pivot.price, True))
                        lows.insert(0, ActLine(i, pivot.price, True))
                state.swing_highs.clear()
                state.swing_lows.clear()
                state.min_pivot = 1e7
                state.max_pivot = 0.0

        if target_mode == "previous session (any)":
            apply_swings("GN", start_sb, end_sb, in_sb)
        else:
            apply_swings("LN", start_ln, end_ln, in_ln)
            apply_swings("AM", start_am, end_am, in_am)
            apply_swings("PM", start_pm, end_pm, in_pm)

        bar_states.append(
            BarState(
                index=i,
                in_sb=in_sb,
                in_ln=in_ln,
                in_am=in_am,
                in_pm=in_pm,
                start_sb=start_sb,
                end_sb=end_sb,
                trend=trend,
                target_hit_high=target_hit_high,
                target_hit_low=target_hit_low,
            )
        )

        last_in_sb = in_sb
        last_in_ln = in_ln
        last_in_am = in_am
        last_in_pm = in_pm

    return {
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "sessions": sessions,
        "targets_active_highs": highs,
        "targets_active_lows": lows,
        "target_hits": target_hits,
        "bar_states": bar_states,
    }
