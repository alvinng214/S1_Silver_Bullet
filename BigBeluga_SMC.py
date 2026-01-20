"""BigBeluga - Smart Money Concepts (Pine Script translation).

This module mirrors the algorithmic logic of the BigBeluga SMC Pine script for
use in Python/backtrader workflows. It focuses on the signal-generation logic
and returns structured outputs (market-structure events, order blocks, and FVGs)
without any charting primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SMCConfig:
    window_enabled: bool = True
    ms_window: int = 5000
    show_swing: bool = True
    swing_limit: int = 100
    show_mapping: bool = False
    candle_css: bool = False
    mstext: str = "Tiny"
    msmode: str = "Adjusted Points"  # "Extreme Points" | "Adjusted Points"
    mslen: int = 5
    build_sweep: bool = True
    msbubble: bool = True

    ob_show: bool = True
    ob_last: int = 5
    ob_show_activity: bool = True
    ob_show_breakers: bool = False
    ob_mode: str = "Length"  # "Length" | "Full"
    ob_len: int = 5
    ob_mitigation: str = "Close"  # "Close" | "Wick" | "Avg"
    ob_metric_size: str = "Normal"
    ob_show_metric: bool = True
    ob_show_midline: bool = True
    ob_overlap_hide: bool = True
    ob_overlap_mode: str = "Recent"  # "Recent" | "Old"

    fvg_enable: bool = False
    fvg_mode: str = "FVG"  # "FVG" | "Breakers"
    fvg_num: int = 5
    fvg_src: str = "Close"  # "Close" | "Wick" | "Avg"
    fvg_thresh: float = 0.0
    fvg_overlap_hide: bool = True
    fvg_extend: bool = False
    fvg_show_midline: bool = True
    fvg_show_raids: bool = False


@dataclass
class StructureEvent:
    index: int
    kind: str  # "bos" | "choch"
    direction: int  # 1 bullish, -1 bearish
    price: float
    sweep: bool = False
    internal: bool = False


@dataclass
class OrderBlock:
    bull: bool
    top: float
    bottom: float
    avg: float
    index: int
    volume: float
    direction: int = 0
    move: int = 1
    bl_pos: int = 1
    br_pos: int = 1
    xlocbl: Optional[pd.Timestamp] = None
    xlocbr: Optional[pd.Timestamp] = None
    mitigated: bool = False
    breaker: bool = False
    breaker_index: Optional[int] = None
    mitigation_index: Optional[int] = None
    active: bool = True


@dataclass
class FVG:
    bull: bool
    top: float
    bottom: float
    index: int
    breaker: bool = False
    breaker_index: Optional[int] = None
    raid: bool = False
    raid_price: Optional[float] = None
    raid_index: Optional[int] = None
    raid_index_end: Optional[int] = None
    active: bool = False


@dataclass
class SMCOutputs:
    trend: pd.Series
    swing_highs: pd.Series
    swing_lows: pd.Series
    events: List[StructureEvent]
    order_blocks: List[OrderBlock]
    fvgs: List[FVG]
    internal_events: List[StructureEvent]
    sfps: List["SFP"]


@dataclass
class StructureState:
    zn: Optional[int] = None
    zz: Optional[float] = None
    trend: int = 0
    start: int = 0
    bos: Optional[float] = None
    choch: Optional[float] = None
    main: Optional[float] = None
    loc: Optional[int] = None
    temp: Optional[int] = None
    xloc: Optional[int] = None
    upsweep: bool = False
    dnsweep: bool = False
    txt: Optional[str] = None
    up: Optional[float] = None
    dn: Optional[float] = None


@dataclass
class SFP:
    price: float
    index: int
    anchor: float


def _sfp_data(df: pd.DataFrame, idx: int) -> Tuple[float, float, float, float, float, float, float, float]:
    if idx < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    return (
        float(df["high"].iloc[idx]),
        float(df["high"].iloc[idx - 1]),
        float(df["high"].iloc[idx - 2]),
        float(df["low"].iloc[idx]),
        float(df["low"].iloc[idx - 1]),
        float(df["low"].iloc[idx - 2]),
        float(df["close"].iloc[idx]),
        float(df.get("volume", pd.Series(index=df.index)).iloc[idx])
        if "volume" in df.columns
        else np.nan,
    )


def _atr(df: pd.DataFrame, length: int = 200, ob_len: int = 5) -> pd.Series:
    """Calculate ATR with Pine Script scaling factor (5/ob_len)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    base_atr = tr.rolling(length).mean()
    # Pine Script: float atr = (ta.atr(200) / (5/len))
    # This scales the ATR based on ob_len parameter
    scale_factor = 5.0 / ob_len if ob_len > 0 else 1.0
    return base_atr / scale_factor


def _pivot_points(series: pd.Series, left: int, right: int, is_high: bool) -> pd.Series:
    pivot = pd.Series(np.nan, index=series.index)
    if len(series) < left + right + 1:
        return pivot
    values = series.to_numpy()
    for i in range(left, len(series) - right):
        center = values[i]
        left_slice = values[i - left : i]
        right_slice = values[i + 1 : i + right + 1]
        if is_high:
            if (left_slice >= center).any() or (right_slice >= center).any():
                continue
        else:
            if (left_slice <= center).any() or (right_slice <= center).any():
                continue
        pivot.iloc[i + right] = center
    return pivot


def _apply_swing_limit(pivot: pd.Series, limit: int) -> pd.Series:
    if limit <= 0 or len(pivot) == 0:
        return pivot
    mask = pd.Series(False, index=pivot.index)
    mask.iloc[-limit:] = True
    pivot = pivot.copy()
    pivot.loc[~mask] = np.nan
    return pivot


def _find_swing_extreme(
    df: pd.DataFrame,
    current_idx: int,
    ref_loc: int,
    use_max: bool,
    use_sweep_loc: bool = False,
    xloc: Optional[int] = None,
    check_adjacent: bool = False,
) -> Tuple[int, float, float]:
    """
    Mirror Pine's ms.find() method.

    Finds the highest high (use_max=True) or lowest low (use_max=False)
    between the current bar and a reference location.

    Args:
        df: DataFrame with OHLC data
        current_idx: Current bar index
        ref_loc: Reference location (ms.loc or ms.xloc)
        use_max: True to find highest high, False to find lowest low
        use_sweep_loc: If True, use xloc instead of ref_loc
        xloc: Alternative reference location for sweeps
        check_adjacent: If True, also check the adjacent bar for OB detection

    Returns:
        Tuple of (index_offset, extreme_value, opposite_value)
        where index_offset is bars back from current_idx
    """
    loc = xloc if use_sweep_loc and xloc is not None else ref_loc

    if loc is None:
        loc = current_idx

    lookback = current_idx - loc
    if lookback < 0:
        lookback = 0

    min_val = float('inf')
    max_val = float('-inf')
    idx = 0
    opposite = 0.0

    for i in range(lookback + 1):
        if current_idx - i < 0:
            break
        bar_high = float(df["high"].iloc[current_idx - i])
        bar_low = float(df["low"].iloc[current_idx - i])

        if use_max:
            if bar_high > max_val:
                max_val = bar_high
                min_val = bar_low
                idx = i
        else:
            if bar_low < min_val:
                min_val = bar_low
                max_val = bar_high
                idx = i

    # Pine's useob flag - check adjacent bar for potentially better OB candle
    if check_adjacent and idx + 1 <= lookback and current_idx - idx - 1 >= 0:
        adj_high = float(df["high"].iloc[current_idx - idx - 1])
        adj_low = float(df["low"].iloc[current_idx - idx - 1])
        if use_max:
            if adj_high > max_val:
                max_val = adj_high
                min_val = adj_low
                idx = idx + 1
        else:
            if adj_low < min_val:
                min_val = adj_low
                max_val = adj_high
                idx = idx + 1

    if use_max:
        return idx, max_val, min_val
    return idx, min_val, max_val


def _find_ob_at_swing(
    df: pd.DataFrame,
    direction: int,
    current_idx: int,
    ref_loc: Optional[int],
    ob_mode: str,
    atr: pd.Series,
) -> Optional[OrderBlock]:
    """
    Find Order Block at swing extreme, mirroring Pine's logic.

    Pine logic:
    - For bullish OB (direction=1): Find lowest low candle (idbull = ms.find(false, false, true))
    - For bearish OB (direction=-1): Find highest high candle (idbear = ms.find(true, false, true))

    The OB is created at the candle where the swing extreme occurred.
    """
    if ref_loc is None:
        ref_loc = max(0, current_idx - 1)

    # Use _find_swing_extreme to locate the swing candle
    # For bullish OB: find lowest low (use_max=False)
    # For bearish OB: find highest high (use_max=True)
    use_max = direction == -1  # Bearish OB needs highest high

    idx_offset, extreme_val, opposite_val = _find_swing_extreme(
        df, current_idx, ref_loc, use_max=use_max, check_adjacent=True
    )

    ob_bar = current_idx - idx_offset
    if ob_bar < 0 or ob_bar >= len(df):
        return None

    candle = df.iloc[ob_bar]
    atr_val = float(atr.iloc[ob_bar]) if ob_bar < len(atr) else np.nan

    if direction == 1:  # Bullish OB
        # Pine: topP = obmode == "Length" ? (low[idbull] + 1 * atr[idbull]) > high[idbull]
        #              ? high[idbull] : (low[idbull] + 1 * atr[idbull]) : high[idbull]
        # OB created with: top=topP, bottom=low[idx]
        bottom = float(candle["low"])
        if ob_mode == "Length" and not np.isnan(atr_val):
            top = min(float(candle["high"]), bottom + atr_val)
        else:
            top = float(candle["high"])
        avg = (top + bottom) / 2.0
        return OrderBlock(
            bull=True,
            top=top,
            bottom=bottom,
            avg=avg,
            index=ob_bar,
            volume=float(candle.get("volume", np.nan)),
            direction=1 if candle["close"] > candle["open"] else -1,
        )
    else:  # Bearish OB (direction == -1)
        # Pine: btmP = obmode == "Length" ? (high[idbear] - 1 * atr[idbear]) < low[idbear]
        #              ? low[idbear] : (high[idbear] - 1 * atr[idbear]) : low[idbear]
        # OB created with: top=high[idx], bottom=btmP
        top = float(candle["high"])
        if ob_mode == "Length" and not np.isnan(atr_val):
            bottom = max(float(candle["low"]), top - atr_val)
        else:
            bottom = float(candle["low"])
        avg = (top + bottom) / 2.0
        return OrderBlock(
            bull=False,
            top=top,
            bottom=bottom,
            avg=avg,
            index=ob_bar,
            volume=float(candle.get("volume", np.nan)),
            direction=1 if candle["close"] > candle["open"] else -1,
        )


def _mitigation_trigger(row: pd.Series, ob: OrderBlock, mode: str) -> bool:
    """
    Check if an Order Block has been mitigated.

    Pine logic uses strict inequalities (<, >):
    - Bullish OB: Close mode: min(close, open) < btm
                  Wick mode: low < btm
                  Avg mode: low < avg
    - Bearish OB: Close mode: max(close, open) > top
                  Wick mode: high > top
                  Avg mode: high > avg
    """
    if ob.bull:
        if mode == "Close":
            return min(row["open"], row["close"]) < ob.bottom
        elif mode == "Wick":
            return row["low"] < ob.bottom
        else:  # Avg mode - Pine: low < stuff.avg
            return row["low"] < ob.avg
    else:  # Bearish OB
        if mode == "Close":
            return max(row["open"], row["close"]) > ob.top
        elif mode == "Wick":
            return row["high"] > ob.top
        else:  # Avg mode - Pine: high > stuff.avg
            return row["high"] > ob.avg


def _fvg_levels(row: pd.Series, src: str) -> Tuple[float, float]:
    if src == "Close":
        return min(row["open"], row["close"]), max(row["open"], row["close"])
    if src == "Avg":
        avg = (row["open"] + row["close"]) / 2.0
        return avg, avg
    return row["low"], row["high"]


def _fvg_mitigated(row: pd.Series, fvg: FVG, src: str) -> bool:
    """
    Check if an FVG has been mitigated.

    Pine logic uses strict inequalities (<, >):
    - Bullish FVG: Close mode: min(c, o) < fvg.btm
                   Wick mode: l < fvg.btm
                   Avg mode: l < math.avg(fvg.top, fvg.btm)  (low < FVG midpoint)
    - Bearish FVG: Close mode: max(c, o) > fvg.top
                   Wick mode: h > fvg.top
                   Avg mode: h > math.avg(fvg.top, fvg.btm)  (high > FVG midpoint)
    """
    fvg_midpoint = (fvg.top + fvg.bottom) / 2.0

    if fvg.bull:
        if src == "Close":
            return min(row["open"], row["close"]) < fvg.bottom
        elif src == "Wick":
            return row["low"] < fvg.bottom
        else:  # Avg mode - Pine: l < math.avg(fvg.top, fvg.btm)
            return row["low"] < fvg_midpoint
    else:  # Bearish FVG
        if src == "Close":
            return max(row["open"], row["close"]) > fvg.top
        elif src == "Wick":
            return row["high"] > fvg.top
        else:  # Avg mode - Pine: h > math.avg(fvg.top, fvg.btm)
            return row["high"] > fvg_midpoint


def _fvg_raid(row: pd.Series, fvg: FVG) -> bool:
    if fvg.bull:
        return row["low"] < fvg.top and row["close"] > fvg.top
    return row["high"] > fvg.bottom and row["close"] < fvg.bottom


def _overlap_range(a_top: float, a_bottom: float, b_top: float, b_bottom: float) -> bool:
    return a_bottom <= b_top and a_top >= b_bottom


def _overlap_fvg(bull: List[FVG], bear: List[FVG]) -> None:
    if len(bull) > 1:
        for i in range(len(bull) - 1, 0, -1):
            stuff = bull[i]
            current = bull[0]
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bull.pop(i)
    if len(bear) > 1:
        for i in range(len(bear) - 1, 0, -1):
            stuff = bear[i]
            current = bear[0]
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bear.pop(i)
    if bull and bear:
        for i in range(len(bull) - 1, -1, -1):
            stuff = bull[i]
            current = bear[0]
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bull.pop(i)
    if bull and bear:
        for i in range(len(bear) - 1, -1, -1):
            stuff = bear[i]
            current = bull[0]
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bear.pop(i)


def _overlap_obs(bull: List[OrderBlock], bear: List[OrderBlock], mode: str) -> None:
    if len(bull) > 1:
        for i in range(len(bull) - 1, 0, -1):
            stuff = bull[i]
            current = bull[0]
            remove_idx = i if mode == "Recent" else 0
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bull.pop(remove_idx)
    if len(bear) > 1:
        for i in range(len(bear) - 1, 0, -1):
            stuff = bear[i]
            current = bear[0]
            remove_idx = i if mode == "Recent" else 0
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bear.pop(remove_idx)
    if bull and bear:
        for i in range(len(bull) - 1, -1, -1):
            stuff = bull[i]
            current = bear[0]
            remove_idx = 0 if mode == "Recent" else i
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bull.pop(remove_idx)
    if bull and bear:
        for i in range(len(bear) - 1, -1, -1):
            stuff = bear[i]
            current = bull[0]
            remove_idx = 0 if mode == "Recent" else i
            if _overlap_range(stuff.top, stuff.bottom, current.top, current.bottom):
                bear.pop(remove_idx)


def _breaker_resolved(row: pd.Series, ob: OrderBlock, mode: str) -> bool:
    """
    Check if an OB breaker has been resolved (invalidated).

    Pine logic for breaker resolution:
    - Bullish breaker: Close mode: max(close, open) > top
                       Wick mode: high > top
                       Avg mode: high > avg (NOT candle_avg > top)
    - Bearish breaker: Close mode: min(close, open) < btm
                       Wick mode: low < btm
                       Avg mode: low < avg (NOT candle_avg < btm)
    """
    if ob.bull:  # Bullish breaker gets resolved when price breaks above
        if mode == "Close":
            return max(row["open"], row["close"]) > ob.top
        elif mode == "Wick":
            return row["high"] > ob.top
        else:  # Avg mode - Pine: high > stuff.avg
            return row["high"] > ob.avg
    else:  # Bearish breaker gets resolved when price breaks below
        if mode == "Close":
            return min(row["open"], row["close"]) < ob.bottom
        elif mode == "Wick":
            return row["low"] < ob.bottom
        else:  # Avg mode - Pine: low < stuff.avg
            return row["low"] < ob.avg


def _update_ob_metrics(
    ob: OrderBlock,
    base_time: pd.Timestamp,
    current_time: pd.Timestamp,
    prev_time: Optional[pd.Timestamp],
    prev_prev_time: Optional[pd.Timestamp],
) -> None:
    if ob.direction == 1:
        if ob.move == 1:
            ob.bl_pos += 1
            ob.move = 2
        elif ob.move == 2:
            ob.bl_pos += 1
            ob.move = 3
        else:
            ob.br_pos += 1
            ob.move = 1
    else:
        if ob.move == 1:
            ob.br_pos += 1
            ob.move = 2
        elif ob.move == 2:
            ob.br_pos += 1
            ob.move = 3
        else:
            ob.bl_pos += 1
            ob.move = 1

    if prev_time is None or prev_prev_time is None:
        return
    dt = current_time - prev_time
    prev_dt = prev_time - prev_prev_time
    if dt == prev_dt:
        ob.xlocbl = base_time + dt * ob.bl_pos
        ob.xlocbr = base_time + dt * ob.br_pos


def _fvg_breaker_resolved(row: pd.Series, fvg: FVG, src: str) -> bool:
    """
    Check if an FVG breaker has been resolved (invalidated).

    Pine logic for FVG breaker resolution:
    - Bullish FVG breaker: Close mode: max(c, o) > fvg.top
                           Wick mode: h > fvg.top
                           Avg mode: h > math.avg(fvg.top, fvg.btm)
    - Bearish FVG breaker: Close mode: min(c, o) < fvg.btm
                           Wick mode: l < fvg.btm
                           Avg mode: l < math.avg(fvg.top, fvg.btm)
    """
    fvg_midpoint = (fvg.top + fvg.bottom) / 2.0

    if fvg.bull:  # Bullish FVG breaker resolved when price breaks above
        if src == "Close":
            return max(row["open"], row["close"]) > fvg.top
        elif src == "Wick":
            return row["high"] > fvg.top
        else:  # Avg mode - Pine: h > math.avg(fvg.top, fvg.btm)
            return row["high"] > fvg_midpoint
    else:  # Bearish FVG breaker resolved when price breaks below
        if src == "Close":
            return min(row["open"], row["close"]) < fvg.bottom
        elif src == "Wick":
            return row["low"] < fvg.bottom
        else:  # Avg mode - Pine: l < math.avg(fvg.top, fvg.btm)
            return row["low"] < fvg_midpoint


def _update_structure(
    state: StructureState,
    idx: int,
    high: float,
    low: float,
    close: float,
    open_: float,
    prev_close: float,
    prev_open: float,
    swing_high: Optional[float],
    swing_low: Optional[float],
    pivot_highs: List[float],
    pivot_high_idx: List[int],
    pivot_lows: List[float],
    pivot_low_idx: List[int],
    crossup: bool,
    crossdn: bool,
    config: SMCConfig,
    df: Optional[pd.DataFrame] = None,
) -> List[StructureEvent]:
    events: List[StructureEvent] = []
    if state.start == 0:
        state.zn = idx
        state.zz = None
        state.bos = high
        state.choch = low
        state.loc = idx
        state.temp = idx
        state.trend = 0
        state.start = 1
        state.main = None
        state.xloc = idx

    state.upsweep = False
    state.dnsweep = False

    if state.start == 1:
        if config.build_sweep and state.choch is not None:
            if low <= state.choch and close >= state.choch:
                state.dnsweep = True
                state.choch = low
                state.xloc = idx
                events.append(StructureEvent(idx, "choch", -1, low, sweep=True))
        if config.build_sweep and state.bos is not None:
            if high >= state.bos and close <= state.bos:
                state.upsweep = True
                state.bos = high
                state.xloc = idx
                events.append(StructureEvent(idx, "choch", 1, high, sweep=True))

        if state.choch is not None and close <= state.choch:
            state.txt = "choch"
            state.trend = -1
            state.choch = state.bos
            state.bos = None
            state.start = 2
            state.loc = idx
            state.main = low
            state.temp = state.loc
            state.xloc = idx
            events.append(StructureEvent(idx, "choch", -1, low))
            return events
        if state.bos is not None and close >= state.bos:
            state.txt = "choch"
            state.trend = 1
            state.bos = None
            state.start = 2
            state.loc = idx
            state.main = high
            state.temp = state.loc
            state.xloc = idx
            events.append(StructureEvent(idx, "choch", 1, high))
            return events

    if state.start == 2:
        if state.trend == -1:
            if state.main is None or low <= state.main:
                state.main = low
                state.temp = idx
            if idx % (config.mslen * 2) == 0 and config.msmode == "Adjusted Points":
                if pivot_highs and state.choch is not None and pivot_highs[0] < state.choch:
                    state.choch = pivot_highs[0]
                    state.loc = pivot_high_idx[0]
                    state.xloc = pivot_high_idx[0]
                    state.temp = pivot_high_idx[0]

            if state.bos is None and crossup and close > open_ and prev_close > prev_open:
                state.bos = state.main
                state.loc = state.temp
                state.xloc = state.loc

            if config.build_sweep and state.bos is not None:
                if low <= state.bos and close >= state.bos:
                    state.dnsweep = True
                    state.bos = low
                    state.xloc = idx
                    events.append(StructureEvent(idx, "bos", -1, low, sweep=True))

            if state.bos is not None and close <= state.bos:
                state.txt = "bos"
                state.zz = state.bos
                state.zn = idx
                events.append(StructureEvent(idx, "bos", -1, state.bos))
                state.bos = None
                if swing_high is not None:
                    state.choch = swing_high
                    state.loc = idx

        if state.trend == 1:
            if state.main is None or high >= state.main:
                state.main = high
                state.temp = idx
            if idx % (config.mslen * 2) == 0 and config.msmode == "Adjusted Points":
                if pivot_lows and state.choch is not None and pivot_lows[0] > state.choch:
                    state.choch = pivot_lows[0]
                    state.loc = pivot_low_idx[0]
                    state.xloc = pivot_low_idx[0]
                    state.temp = pivot_low_idx[0]

            if state.bos is None and crossdn and close < open_ and prev_close < prev_open:
                state.bos = state.main
                state.loc = state.temp
                state.xloc = state.loc

            if config.build_sweep and state.bos is not None:
                if high >= state.bos and close <= state.bos:
                    state.upsweep = True
                    state.bos = high
                    state.xloc = idx
                    events.append(StructureEvent(idx, "bos", 1, high, sweep=True))

            if state.bos is not None and close >= state.bos:
                state.txt = "bos"
                state.zz = state.bos
                state.zn = idx
                events.append(StructureEvent(idx, "bos", 1, state.bos))
                state.bos = None
                if swing_low is not None:
                    state.choch = swing_low
                    state.loc = idx

        if state.choch is not None:
            if state.trend == 1 and close <= state.choch:
                # Bullish to Bearish CHoCH
                state.txt = "choch"
                events.append(StructureEvent(idx, "choch", -1, state.choch))
                state.trend = -1

                # Pine logic: when BOS is None, use ms.find() to locate new CHoCH level
                # For bearish trend: find highest high as new CHoCH
                if state.bos is None and df is not None:
                    # ms.find(true, false, false) - find highest high
                    idx_offset, found_high, _ = _find_swing_extreme(
                        df, idx, state.loc or idx, use_max=True
                    )
                    state.choch = found_high
                    state.loc = idx - idx_offset
                else:
                    state.choch = state.bos

                state.bos = None
                state.loc = idx
                state.main = low
                state.temp = state.loc
                state.xloc = idx

            elif state.trend == -1 and close >= state.choch:
                # Bearish to Bullish CHoCH
                state.txt = "choch"
                events.append(StructureEvent(idx, "choch", 1, state.choch))
                state.trend = 1

                # Pine logic: when BOS is None, use ms.find() to locate new CHoCH level
                # For bullish trend: find lowest low as new CHoCH
                if state.bos is None and df is not None:
                    # ms.find(false, false, false) - find lowest low
                    idx_offset, found_low, _ = _find_swing_extreme(
                        df, idx, state.loc or idx, use_max=False
                    )
                    state.choch = found_low
                    state.loc = idx - idx_offset
                else:
                    state.choch = state.bos

                state.bos = None
                state.loc = idx
                state.main = high
                state.temp = state.loc
                state.xloc = idx

    return events


def _detect_fvg(
    df: pd.DataFrame,
    idx: int,
    fvg_src: str,
    thresh: float,
    atr: pd.Series,
    pending_bull_fvg: List[bool],
    pending_bear_fvg: List[bool],
) -> Optional[FVG]:
    """
    Detect Fair Value Gap mirroring Pine Script logic.

    Pine Script creates FVG one bar AFTER detection (using upfvg[1]/dnfvg[1]).
    To match this behavior, we track pending FVGs and create them on the next bar.

    Pine FVG detection:
    - Bullish: l > h2 (current low > high 2 bars ago) - gap up
    - Bearish: l2 > h (low 2 bars ago > current high) - gap down

    Pine FVG boundaries (at creation bar, one bar after detection):
    - Bullish: top = l[1] (low of detection bar), bottom = h2[1] (high[3] at creation)
    - Bearish: top = l2[1] (low[3] at creation), bottom = h[1] (high of detection bar)
    """
    if idx < 3:
        return None

    # Current bar values
    high = df["high"].iloc[idx]
    low = df["low"].iloc[idx]
    close = df["close"].iloc[idx]

    # Previous bar values (for threshold check - Pine uses c1)
    prev_close = df["close"].iloc[idx - 1]
    prev_low = df["low"].iloc[idx - 1]
    prev_high = df["high"].iloc[idx - 1]

    # Two bars ago
    high_2 = df["high"].iloc[idx - 2]
    low_2 = df["low"].iloc[idx - 2]

    # Three bars ago (for FVG boundaries at creation time)
    high_3 = df["high"].iloc[idx - 3] if idx >= 3 else np.nan
    low_3 = df["low"].iloc[idx - 3] if idx >= 3 else np.nan

    # ATR for threshold
    atr_prev = atr.iloc[idx - 1] if idx >= 1 else np.nan

    # Pine threshold calculation: blth = l1 + (fvatr[1] * fvgthresh)
    bullish_thresh = prev_low + (atr_prev * thresh) if not pd.isna(atr_prev) else prev_low
    bearish_thresh = prev_high - (atr_prev * thresh) if not pd.isna(atr_prev) else prev_high

    # Check if there's a pending FVG from previous bar to create now
    result = None

    if pending_bull_fvg and pending_bull_fvg[0]:
        # Create bullish FVG now (one bar after detection)
        # Pine: top = l[1], bottom = h2[1] = high[3] at current bar
        top = prev_low  # l[1] - low of detection bar (now prev bar)
        bottom = high_3  # h2[1] = high[3] at creation bar
        pending_bull_fvg[0] = False
        if not pd.isna(top) and not pd.isna(bottom) and top > bottom:
            result = FVG(bull=True, top=float(top), bottom=float(bottom), index=idx - 2)

    if pending_bear_fvg and pending_bear_fvg[0]:
        # Create bearish FVG now (one bar after detection)
        # Pine: top = l2[1] = low[3], bottom = h[1] (high of detection bar)
        top = low_3  # l2[1] = low[3] at creation bar
        bottom = prev_high  # h[1] - high of detection bar (now prev bar)
        pending_bear_fvg[0] = False
        if not pd.isna(top) and not pd.isna(bottom) and top > bottom:
            if result is None:
                result = FVG(bull=False, top=float(top), bottom=float(bottom), index=idx - 2)

    # Detect new FVGs (to be created next bar)
    # Pine: if l > h2 and cc and c1 > blth => upfvg := true
    if low > high_2 and prev_close > bullish_thresh:
        pending_bull_fvg[0] = True

    # Pine: if l2 > h and cc and c1 < brth => dnfvg := true
    if low_2 > high and prev_close < bearish_thresh:
        pending_bear_fvg[0] = True

    return result


def calculate_bigbeluga_smc(df: pd.DataFrame, config: Optional[SMCConfig] = None) -> SMCOutputs:
    """Translate BigBeluga SMC Pine logic into Python output series."""
    config = config or SMCConfig()
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    window_start = max(0, len(df) - config.ms_window) if config.window_enabled else 0
    swing_highs = _pivot_points(df["high"], config.mslen, config.mslen, True)
    swing_lows = _pivot_points(df["low"], config.mslen, config.mslen, False)
    if config.show_swing:
        swing_highs = _apply_swing_limit(swing_highs, config.swing_limit)
        swing_lows = _apply_swing_limit(swing_lows, config.swing_limit)
    else:
        swing_highs[:] = np.nan
        swing_lows[:] = np.nan

    trend = pd.Series(0, index=df.index)
    events: List[StructureEvent] = []
    bull_obs: List[OrderBlock] = []
    bear_obs: List[OrderBlock] = []
    bull_fvgs: List[FVG] = []
    bear_fvgs: List[FVG] = []
    internal_events: List[StructureEvent] = []
    sfps: List[SFP] = []

    last_swing_high = np.nan
    last_swing_low = np.nan
    last_swing_high_idx = None
    last_swing_low_idx = None
    current_trend = 0
    structure_state = StructureState()

    # ATR with Pine Script scaling factor
    atr = _atr(df, ob_len=config.ob_len)
    pivot_high_idx: List[int] = []
    pivot_highs: List[float] = []
    pivot_low_idx: List[int] = []
    pivot_lows: List[float] = []

    # Pending FVG tracking (Pine creates FVG one bar after detection)
    pending_bull_fvg: List[bool] = [False]
    pending_bear_fvg: List[bool] = [False]

    for i in range(window_start, len(df)):
        if i == 0:
            prev_close = float(df["close"].iloc[i])
            prev_open = float(df["open"].iloc[i])
        else:
            prev_close = float(df["close"].iloc[i - 1])
            prev_open = float(df["open"].iloc[i - 1])

        if not np.isnan(swing_highs.iloc[i]):
            last_swing_high = float(swing_highs.iloc[i])
            last_swing_high_idx = i
        if not np.isnan(swing_lows.iloc[i]):
            last_swing_low = float(swing_lows.iloc[i])
            last_swing_low_idx = i

        close = float(df["close"].iloc[i])
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        open_ = float(df["open"].iloc[i])

        if not np.isnan(swing_highs.iloc[i]):
            # Pine stores bar_index[mslen] - the actual pivot bar, not the confirmation bar
            # The pivot at swing_highs.iloc[i] was actually detected at bar (i - mslen)
            actual_pivot_bar = i - config.mslen
            pivot_high_idx.insert(0, actual_pivot_bar)
            pivot_highs.insert(0, float(swing_highs.iloc[i]))
        if not np.isnan(swing_lows.iloc[i]):
            # Same for swing lows
            actual_pivot_bar = i - config.mslen
            pivot_low_idx.insert(0, actual_pivot_bar)
            pivot_lows.insert(0, float(swing_lows.iloc[i]))

        if pivot_highs and high > pivot_highs[0]:
            pivot_highs.clear()
            pivot_high_idx.clear()
        if pivot_lows and low < pivot_lows[0]:
            pivot_lows.clear()
            pivot_low_idx.clear()

        if structure_state.up is None:
            structure_state.up = high
        if structure_state.dn is None:
            structure_state.dn = low

        crossup = False
        crossdn = False
        if structure_state.up is not None and high > structure_state.up:
            structure_state.up = high
            structure_state.dn = low
            crossup = True
        if structure_state.dn is not None and low < structure_state.dn:
            structure_state.up = high
            structure_state.dn = low
            crossdn = True

        swing_high = float(last_swing_high) if last_swing_high_idx is not None else None
        swing_low = float(last_swing_low) if last_swing_low_idx is not None else None
        events.extend(
            _update_structure(
                structure_state,
                i,
                high,
                low,
                close,
                open_,
                prev_close,
                prev_open,
                swing_high,
                swing_low,
                pivot_highs,
                pivot_high_idx,
                pivot_lows,
                pivot_low_idx,
                crossup,
                crossdn,
                config,
                df,  # Pass DataFrame for CHoCH find() logic
            )
        )
        if structure_state.trend != current_trend:
            current_trend = structure_state.trend
            # Use structure_state.loc as reference for finding swing extreme
            ob = _find_ob_at_swing(
                df,
                current_trend,
                i,
                structure_state.loc,
                config.ob_mode,
                atr,
            )
            if ob:
                if ob.bull:
                    bull_obs.insert(0, ob)
                else:
                    bear_obs.insert(0, ob)

        # Note: Extra sweep detection on swing points was removed.
        # Pine only detects sweeps on structure levels (BOS/CHoCH), not on all swing points.
        # Sweep detection is handled inside _update_structure.

        trend.iloc[i] = current_trend

        if config.ob_show:
            for ob in list(bull_obs + bear_obs):
                if not ob.active:
                    continue
                base_time = df.index[ob.index] if ob.index < len(df.index) else None
                if base_time is not None:
                    prev_time = df.index[i - 1] if i - 1 >= 0 else None
                    prev_prev_time = df.index[i - 2] if i - 2 >= 0 else None
                    _update_ob_metrics(
                        ob,
                        base_time=base_time,
                        current_time=df.index[i],
                        prev_time=prev_time,
                        prev_prev_time=prev_prev_time,
                    )
                if not ob.breaker:
                    if _mitigation_trigger(df.iloc[i], ob, config.ob_mitigation):
                        ob.mitigated = True
                        ob.breaker = True
                        ob.breaker_index = i
                        ob.mitigation_index = i
                        if not config.ob_show_breakers:
                            ob.active = False
                else:
                    if _breaker_resolved(df.iloc[i], ob, config.ob_mitigation):
                        ob.active = False

        if config.fvg_enable:
            # FVG detection with Pine-style delayed creation
            fvg = _detect_fvg(
                df, i, config.fvg_src, config.fvg_thresh, atr,
                pending_bull_fvg, pending_bear_fvg
            )
            if fvg:
                if fvg.bull:
                    # Pine logic: reset raid state when new FVG is created
                    if bull_fvgs and bull_fvgs[0].raid and not bull_fvgs[0].active:
                        bull_fvgs[0].active = True
                        bull_fvgs[0].raid = False
                        bull_fvgs[0].raid_price = None
                        bull_fvgs[0].raid_index = None
                        bull_fvgs[0].raid_index_end = None
                    bull_fvgs.insert(0, fvg)
                else:
                    if bear_fvgs and bear_fvgs[0].raid and not bear_fvgs[0].active:
                        bear_fvgs[0].active = True
                        bear_fvgs[0].raid = False
                        bear_fvgs[0].raid_price = None
                        bear_fvgs[0].raid_index = None
                        bear_fvgs[0].raid_index_end = None
                    bear_fvgs.insert(0, fvg)
            for gaps in (bull_fvgs, bear_fvgs):
                for idx_gap in range(len(gaps) - 1, -1, -1):
                    gap = gaps[idx_gap]
                    if not gap.breaker and _fvg_mitigated(df.iloc[i], gap, config.fvg_src):
                        if config.fvg_mode == "Breakers":
                            gap.breaker = True
                            gap.breaker_index = i
                        else:
                            gaps.pop(idx_gap)
                            continue
                    if gap.breaker and config.fvg_mode == "Breakers":
                        if _fvg_breaker_resolved(df.iloc[i], gap, config.fvg_src):
                            gaps.pop(idx_gap)
                            continue
                    if config.fvg_show_raids and not gap.raid and _fvg_raid(df.iloc[i], gap):
                        gap.raid = True
                        gap.raid_price = low if gap.bull else high
                        gap.raid_index = i
                        gap.raid_index_end = i
                    elif gap.raid:
                        if gap.active is False:
                            gap.raid_index_end = i
                            if gap.bull and low <= (gap.raid_price or gap.top):
                                gap.active = True
                            if not gap.bull and high >= (gap.raid_price or gap.bottom):
                                gap.active = True

        h, h1, h2, l, l1, l2, c, v = _sfp_data(df, i)
        if not np.isnan(h1) and h > h1 and c < h1:
            sfps.append(SFP(price=h, index=i, anchor=h1))
        if not np.isnan(l1) and l < l1 and c > l1:
            sfps.append(SFP(price=l, index=i, anchor=l1))

    if config.ob_show and config.ob_overlap_hide:
        _overlap_obs(bull_obs, bear_obs, config.ob_overlap_mode)

    bull_obs = [ob for ob in bull_obs if ob.active][: config.ob_last]
    bear_obs = [ob for ob in bear_obs if ob.active][: config.ob_last]
    order_blocks = bull_obs + bear_obs

    fvgs: List[FVG] = []
    if config.fvg_enable:
        if config.fvg_overlap_hide:
            _overlap_fvg(bull_fvgs, bear_fvgs)
        bull_fvgs = bull_fvgs[: config.fvg_num]
        bear_fvgs = bear_fvgs[: config.fvg_num]
        fvgs = bull_fvgs + bear_fvgs

    return SMCOutputs(
        trend=trend,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        events=events,
        order_blocks=order_blocks,
        fvgs=fvgs,
        internal_events=internal_events,
        sfps=sfps,
    )
