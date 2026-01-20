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


def _atr(df: pd.DataFrame, length: int = 200) -> pd.Series:
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
    return tr.rolling(length).mean()


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


def _find_ob_range(
    df: pd.DataFrame,
    direction: int,
    lookback: int,
    ob_mode: str,
    idx: int,
    atr: pd.Series,
) -> Optional[OrderBlock]:
    start = max(0, idx - lookback)
    end = idx
    if direction == 1:
        candidates = df.iloc[start:end]
        down_candles = candidates[candidates["close"] < candidates["open"]]
        if down_candles.empty:
            return None
        ob_idx = down_candles.index[-1]
        candle = df.loc[ob_idx]
        atr_val = float(atr.loc[ob_idx]) if ob_idx in atr.index else np.nan
        top = candle["high"]
        if ob_mode == "Length" and not np.isnan(atr_val):
            top = min(candle["high"], candle["low"] + atr_val)
        bottom = candle["low"]
        avg = (top + bottom) / 2.0
        return OrderBlock(
            bull=True,
            top=float(top),
            bottom=float(bottom),
            avg=float(avg),
            index=df.index.get_loc(ob_idx),
            volume=float(candle.get("volume", np.nan)),
            direction=1 if candle["close"] > candle["open"] else -1,
        )
    candidates = df.iloc[start:end]
    up_candles = candidates[candidates["close"] > candidates["open"]]
    if up_candles.empty:
        return None
    ob_idx = up_candles.index[-1]
    candle = df.loc[ob_idx]
    atr_val = float(atr.loc[ob_idx]) if ob_idx in atr.index else np.nan
    top = candle["high"]
    bottom = candle["low"]
    if ob_mode == "Length" and not np.isnan(atr_val):
        bottom = max(candle["low"], candle["high"] - atr_val)
    avg = (top + bottom) / 2.0
    return OrderBlock(
        bull=False,
        top=float(top),
        bottom=float(bottom),
        avg=float(avg),
        index=df.index.get_loc(ob_idx),
        volume=float(candle.get("volume", np.nan)),
        direction=1 if candle["close"] > candle["open"] else -1,
    )


def _mitigation_trigger(row: pd.Series, ob: OrderBlock, mode: str) -> bool:
    if ob.bull:
        level = ob.bottom if mode == "Close" else ob.bottom if mode == "Wick" else ob.avg
        if mode == "Close":
            return min(row["open"], row["close"]) < level
        return row["low"] < level
    level = ob.top if mode == "Close" else ob.top if mode == "Wick" else ob.avg
    if mode == "Close":
        return max(row["open"], row["close"]) > level
    return row["high"] > level


def _fvg_levels(row: pd.Series, src: str) -> Tuple[float, float]:
    if src == "Close":
        return min(row["open"], row["close"]), max(row["open"], row["close"])
    if src == "Avg":
        avg = (row["open"] + row["close"]) / 2.0
        return avg, avg
    return row["low"], row["high"]


def _fvg_mitigated(row: pd.Series, fvg: FVG, src: str) -> bool:
    if fvg.bull:
        if src == "Close":
            return min(row["open"], row["close"]) <= fvg.bottom
        if src == "Avg":
            avg = (row["open"] + row["close"]) / 2.0
            return avg <= fvg.bottom
        return row["low"] <= fvg.bottom
    if src == "Close":
        return max(row["open"], row["close"]) >= fvg.top
    if src == "Avg":
        avg = (row["open"] + row["close"]) / 2.0
        return avg >= fvg.top
    return row["high"] >= fvg.top


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
    if ob.bull:
        if mode == "Close":
            return max(row["open"], row["close"]) > ob.top
        if mode == "Avg":
            avg = (row["open"] + row["close"]) / 2.0
            return avg > ob.top
        return row["high"] > ob.top
    if mode == "Close":
        return min(row["open"], row["close"]) < ob.bottom
    if mode == "Avg":
        avg = (row["open"] + row["close"]) / 2.0
        return avg < ob.bottom
    return row["low"] < ob.bottom


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
    if fvg.bull:
        if src == "Close":
            return max(row["open"], row["close"]) > fvg.top
        if src == "Avg":
            avg = (row["open"] + row["close"]) / 2.0
            return avg > fvg.top
        return row["high"] > fvg.top
    if src == "Close":
        return min(row["open"], row["close"]) < fvg.bottom
    if src == "Avg":
        avg = (row["open"] + row["close"]) / 2.0
        return avg < fvg.bottom
    return row["low"] < fvg.bottom
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
                state.txt = "choch"
                events.append(StructureEvent(idx, "choch", -1, state.choch))
                state.trend = -1
                state.choch = state.bos
                state.bos = None
                state.loc = idx
                state.main = low
                state.temp = state.loc
                state.xloc = idx
            elif state.trend == -1 and close >= state.choch:
                state.txt = "choch"
                events.append(StructureEvent(idx, "choch", 1, state.choch))
                state.trend = 1
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
) -> Optional[FVG]:
    if idx < 3:
        return None
    high_2 = df["high"].iloc[idx - 2]
    low_2 = df["low"].iloc[idx - 2]
    high_3 = df["high"].iloc[idx - 3]
    low_3 = df["low"].iloc[idx - 3]
    high = df["high"].iloc[idx]
    low = df["low"].iloc[idx]
    prev_close = df["close"].iloc[idx - 1]
    prev_low = df["low"].iloc[idx - 1]
    prev_high = df["high"].iloc[idx - 1]
    atr_prev = atr.iloc[idx - 1]
    bullish_thresh = prev_low + (atr_prev * thresh) if not pd.isna(atr_prev) else prev_low
    bearish_thresh = prev_high - (atr_prev * thresh) if not pd.isna(atr_prev) else prev_high
    if low > high_2 and prev_close > bullish_thresh:
        top = prev_low
        bottom = high_3
        if thresh > 0 and not pd.isna(atr_prev) and (top - bottom) < (atr_prev * thresh):
            return None
        return FVG(bull=True, top=float(top), bottom=float(bottom), index=idx - 1)
    if low_2 > high and prev_close < bearish_thresh:
        top = low_3
        bottom = prev_high
        if thresh > 0 and not pd.isna(atr_prev) and (top - bottom) < (atr_prev * thresh):
            return None
        return FVG(bull=False, top=float(top), bottom=float(bottom), index=idx - 1)
    return None


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

    atr = _atr(df)
    pivot_high_idx: List[int] = []
    pivot_highs: List[float] = []
    pivot_low_idx: List[int] = []
    pivot_lows: List[float] = []

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
            pivot_high_idx.insert(0, i)
            pivot_highs.insert(0, float(swing_highs.iloc[i]))
        if not np.isnan(swing_lows.iloc[i]):
            pivot_low_idx.insert(0, i)
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
            )
        )
        if structure_state.trend != current_trend:
            current_trend = structure_state.trend
            ob = _find_ob_range(df, current_trend, config.ob_len, config.ob_mode, i, atr)
            if ob:
                if ob.bull:
                    bull_obs.insert(0, ob)
                else:
                    bear_obs.insert(0, ob)

        if config.build_sweep:
            if last_swing_high_idx is not None and high > last_swing_high and close < last_swing_high:
                events.append(StructureEvent(i, "bos", 1, high, sweep=True))
            if last_swing_low_idx is not None and low < last_swing_low and close > last_swing_low:
                events.append(StructureEvent(i, "bos", -1, low, sweep=True))

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
            fvg = _detect_fvg(df, i, config.fvg_src, config.fvg_thresh, atr)
            if fvg:
                if fvg.bull:
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
