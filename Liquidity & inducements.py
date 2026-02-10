"""Liquidity & inducements (Pine translation).

Implements the PriceAction-based logic from the Pine Script by:
- Detecting market structure pivots and trend (CHoCH/BOS).
- Tracking liquidity grabs, sweeps, turtle soups, equal pivots, and external liquidity.
- Building retracement inducement levels after structure breaks.
- Supporting HTF pivot sourcing via resampling (request.security analogue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import PriceAction as pa


Pivot = pa.Pivot


@dataclass
class LiquidityGrab:
    pivot: Pivot
    taken: bool = False
    invalidated: bool = False
    limit: Optional[pa.Line] = None
    break_line: Optional[pa.Line] = None
    linefill: Optional[pa.LineFill] = None
    label: Optional[pa.Label] = None


@dataclass
class EqualPivotInducement:
    stop_losses: float
    first_pivot: Pivot
    second_pivot: Pivot
    label: Optional[pa.Label] = None
    line: Optional[pa.Line] = None
    liquidity_taken: bool = False


@dataclass
class EqualPivotState:
    highs: List[Pivot] = field(default_factory=list)
    lows: List[Pivot] = field(default_factory=list)
    bearish_inducements: List[EqualPivotInducement] = field(default_factory=list)
    bullish_inducements: List[EqualPivotInducement] = field(default_factory=list)


@dataclass
class RetracementInducement:
    pivot: Pivot
    line: Optional[pa.Line] = None
    label: Optional[pa.Label] = None
    taken: bool = False
    invalidated: bool = False
    stop_index: Optional[int] = None


@dataclass
class RetracementState:
    highs: List[RetracementInducement] = field(default_factory=list)
    lows: List[RetracementInducement] = field(default_factory=list)
    high_pivots: List[Pivot] = field(default_factory=list)
    low_pivots: List[Pivot] = field(default_factory=list)
    historical_highs: List[RetracementInducement] = field(default_factory=list)
    historical_lows: List[RetracementInducement] = field(default_factory=list)


Structure = pa.Structure


@dataclass
class ExternalLiquidity:
    price: float
    pivot: Pivot
    line: Optional[pa.Line] = None
    label: Optional[pa.Label] = None
    hidden: bool = False


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
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


def _timeframe_rule(tf: str) -> Optional[str]:
    if tf == "":
        return None
    if tf.isdigit():
        return f"{int(tf)}min"
    if tf in {"D", "W", "M"}:
        return tf
    raise ValueError(f"Unsupported timeframe: {tf}")


def _floor_time(ts: pd.Timestamp, rule: str) -> pd.Timestamp:
    if rule in {"D", "W", "M"}:
        return ts.to_period(rule).start_time
    return ts.floor(rule)


def _build_htf_view(df: pd.DataFrame, rule: str) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[bool]]:
    htf = (
        df.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    buckets = [_floor_time(ts, rule) for ts in df.index]
    bucket_series = pd.Series(buckets, index=df.index)
    last_in_bucket = bucket_series.groupby(bucket_series).transform("size")
    pos_in_bucket = bucket_series.groupby(bucket_series).cumcount()
    htf_closed = (pos_in_bucket + 1) == last_in_bucket
    return htf, buckets, list(htf_closed)


def _pivot_series(high: pd.Series, low: pd.Series, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
    pivot_high = [np.nan] * len(high)
    pivot_low = [np.nan] * len(low)
    for i in range(left + right, len(high)):
        pivot_idx = i - right
        if pivot_idx - left < 0 or pivot_idx + right >= len(high):
            continue
        center_high = high.iloc[pivot_idx]
        center_low = low.iloc[pivot_idx]
        # Pivot high and low must be detected independently (Pine's ta.pivothigh
        # and ta.pivotlow are separate functions).
        is_pivot_high = True
        if (high.iloc[pivot_idx - left : pivot_idx] >= center_high).any():
            is_pivot_high = False
        if is_pivot_high and (high.iloc[pivot_idx + 1 : pivot_idx + right + 1] >= center_high).any():
            is_pivot_high = False
        if is_pivot_high:
            pivot_high[i] = center_high
        is_pivot_low = True
        if (low.iloc[pivot_idx - left : pivot_idx] <= center_low).any():
            is_pivot_low = False
        if is_pivot_low and (low.iloc[pivot_idx + 1 : pivot_idx + right + 1] <= center_low).any():
            is_pivot_low = False
        if is_pivot_low:
            pivot_low[i] = center_low
    return pd.Series(pivot_high, index=high.index), pd.Series(pivot_low, index=high.index)


def _bar_index_for_time(times: pd.Index, ts: pd.Timestamp) -> int:
    return int(times.searchsorted(ts, side="left"))


def _htf_pivots(df: pd.DataFrame, tf: str, left: int, right: int) -> Tuple[List[Optional[Pivot]], List[Optional[Pivot]], List[bool]]:
    rule = _timeframe_rule(tf)
    if rule is None:
        htf = df.copy()
        buckets = list(df.index)
        closed_flags = [True] * len(df)
    else:
        htf, buckets, closed_flags = _build_htf_view(df, rule)
    piv_high, piv_low = _pivot_series(htf["high"], htf["low"], left, right)
    piv_high_list: List[Optional[Pivot]] = [None] * len(df)
    piv_low_list: List[Optional[Pivot]] = [None] * len(df)
    for i in range(len(df)):
        bucket = buckets[i]
        if bucket not in htf.index:
            continue
        htf_idx = htf.index.get_loc(bucket)
        if not closed_flags[i] and htf_idx > 0:
            htf_idx -= 1
        ph = piv_high.iloc[htf_idx]
        pl = piv_low.iloc[htf_idx]
        if not np.isnan(ph):
            time_val = htf.index[htf_idx - right]
            piv_high_list[i] = Pivot(float(ph), _bar_index_for_time(df.index, time_val), 1, time_val)
        if not np.isnan(pl):
            time_val = htf.index[htf_idx - right]
            piv_low_list[i] = Pivot(float(pl), _bar_index_for_time(df.index, time_val), -1, time_val)
    return piv_high_list, piv_low_list, closed_flags




def _stop_retracement_inducements(
    retr_state: RetracementState,
    *,
    high: float,
    low: float,
    bar_index: int,
    stop_reason: str,
    keep_invalidated: bool,
) -> None:
    if stop_reason not in {"take", "invalidate"}:
        raise ValueError("stop_reason must be 'take' or 'invalidate'")

    remaining_highs: List[RetracementInducement] = []
    for inducement in retr_state.highs:
        stop = stop_reason == "invalidate" or high >= inducement.pivot.price
        if stop:
            inducement.stop_index = bar_index
            if stop_reason == "take":
                inducement.taken = True
            else:
                inducement.invalidated = True
            keep_line = stop_reason == "take" or keep_invalidated
            if keep_line and inducement.line:
                inducement.line.set_x2(bar_index)
                inducement.line.set_extend_right(False)
            else:
                if inducement.line:
                    inducement.line.delete()
                if inducement.label:
                    inducement.label.delete()
            if keep_line:
                retr_state.historical_highs.append(inducement)
        else:
            remaining_highs.append(inducement)
    retr_state.highs = remaining_highs

    remaining_lows: List[RetracementInducement] = []
    for inducement in retr_state.lows:
        stop = stop_reason == "invalidate" or low <= inducement.pivot.price
        if stop:
            inducement.stop_index = bar_index
            if stop_reason == "take":
                inducement.taken = True
            else:
                inducement.invalidated = True
            keep_line = stop_reason == "take" or keep_invalidated
            if keep_line and inducement.line:
                inducement.line.set_x2(bar_index)
                inducement.line.set_extend_right(False)
            else:
                if inducement.line:
                    inducement.line.delete()
                if inducement.label:
                    inducement.label.delete()
            if keep_line:
                retr_state.historical_lows.append(inducement)
        else:
            remaining_lows.append(inducement)
    retr_state.lows = remaining_lows


def calculate_liquidity_inducements(
    df: pd.DataFrame,
    *,
    market_left: int = 5,
    market_right: int = 5,
    grabs_enabled: bool = True,
    big_grabs_enabled: bool = True,
    sweeps_enabled: bool = True,
    turtle_soups_enabled: bool = True,
    equal_pivots_enabled: bool = True,
    external_liquidity_enabled: bool = True,
    retracement_inducements_enabled: bool = True,
    grabs_left: int = 3,
    grabs_right: int = 3,
    grabs_lookback: int = 5,
    grabs_tf: str = "",
    big_grabs_left: int = 10,
    big_grabs_right: int = 10,
    big_grabs_lookback: int = 5,
    big_grabs_tf: str = "",
    sweeps_left: int = 3,
    sweeps_right: int = 3,
    sweeps_lookback: int = 5,
    sweeps_tf: str = "",
    turtle_left: int = 1,
    turtle_right: int = 1,
    turtle_lookback: int = 5,
    turtle_tf: str = "",
    turtle_confirmation: bool = True,
    turtle_color: str = "orange@70",
    equal_left: int = 1,
    equal_right: int = 1,
    equal_atr_factor: float = 0.5,
    equal_lookback: int = 3,
    equal_tf: str = "",
    equal_liquidity_color: str = "orange",
    equal_bullish_inducement_color: str = "teal",
    equal_bearish_inducement_color: str = "red",
    equal_font_size: int = 7,
    line_style: str = "dotted",
    liquidity_font_size: int = 7,
    grabs_color: str = "orange",
    big_grabs_color: str = "aqua",
    sweeps_bullish_color: str = "teal",
    sweeps_bearish_color: str = "red",
    external_show: int = 1,
    external_line_style: str = "dotted",
    external_bullish_color: str = "teal",
    external_bearish_color: str = "red",
    retr_left: int = 1,
    retr_right: int = 1,
    retr_lookback: int = 5,
    retr_tf: str = "",
    retr_keep_invalidated: bool = False,
    retr_bullish_color: str = "teal",
    retr_bearish_color: str = "red",
    retr_line_style: str = "dotted",
    retr_font_size: int = 7,
) -> Dict[str, object]:
    atr = _atr(df, 14)
    buyside_targets = pd.Series(np.nan, index=df.index)
    sellside_targets = pd.Series(np.nan, index=df.index)

    structure = Structure(
        left_length=market_left,
        right_length=market_right,
        type=pa.StructureType.SWING,
        trend=0,
        equal_pivots_factor=0.0,
    )
    change_of_character: Optional[Pivot] = None
    break_of_structure: Optional[Pivot] = None
    previous_structure_break_pivot: Optional[Pivot] = None
    previous_structure_break_index: Optional[int] = None
    retracement_structure_break_index: Optional[int] = None

    grabs_highs: List[LiquidityGrab] = []
    grabs_lows: List[LiquidityGrab] = []
    big_grabs_highs: List[LiquidityGrab] = []
    big_grabs_lows: List[LiquidityGrab] = []
    sweeps_highs: List[LiquidityGrab] = []
    sweeps_lows: List[LiquidityGrab] = []
    equal_state = EqualPivotState()
    retr_state = RetracementState()
    turtle_context = pa.TurtleSoups()
    turtle_settings = pa.TurtleSoupSettings(
        pivot_left_length=turtle_left,
        pivot_right_length=turtle_right,
        lookback=turtle_lookback,
        confirmation=turtle_confirmation,
        color=turtle_color,
    )
    turtle_screener = pa.Screener()
    buyside: List[ExternalLiquidity] = []
    sellside: List[ExternalLiquidity] = []

    grab_high_series, grab_low_series, grab_closed = _htf_pivots(df, grabs_tf, grabs_left, grabs_right)
    big_grab_high_series, big_grab_low_series, big_grab_closed = _htf_pivots(df, big_grabs_tf, big_grabs_left, big_grabs_right)
    sweep_high_series, sweep_low_series, sweep_closed = _htf_pivots(df, sweeps_tf, sweeps_left, sweeps_right)
    turtle_high_series, turtle_low_series, turtle_closed = _htf_pivots(df, turtle_tf, turtle_left, turtle_right)
    equal_high_series, equal_low_series, _ = _htf_pivots(df, equal_tf, equal_left, equal_right)
    retr_high_series, retr_low_series, _ = _htf_pivots(df, retr_tf, retr_left, retr_right)

    for i in range(len(df)):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        close = float(df["close"].iloc[i])
        pa.pivot_step(structure, df["high"], df["low"], df.index, i)

        last_high = next((p for p in structure.pivots if p.type == 1), None)
        last_low = next((p for p in structure.pivots if p.type == -1), None)

        change_of_character = pa.change_of_character(structure, closes=df["close"], bar_index=i)
        structure_break_event = False
        if change_of_character:
            break_of_structure = None
            previous_structure_break_pivot = change_of_character
            structure_break_event = True

        bos_pivot = pa.break_of_structure(structure, closes=df["close"], bar_index=i)
        if bos_pivot:
            break_of_structure = bos_pivot
            previous_structure_break_pivot = bos_pivot
            structure_break_event = True

        # Pine Script detects grabs/sweeps on existing pivots BEFORE storing
        # new ones (LiquidityGrabs() runs before SetLiquidityGrabs()).
        if i > 0:
            prev_high = float(df["high"].iloc[i - 1])
            prev_low = float(df["low"].iloc[i - 1])
            def _process_grabs(grabs: List[LiquidityGrab], color: str) -> None:
                for grab in grabs:
                    if grab.taken or grab.invalidated:
                        continue
                    grabbed = False
                    if grab.pivot.type == -1:
                        if prev_low <= grab.pivot.price and close >= grab.pivot.price:
                            grabbed = True
                        elif close < grab.pivot.price:
                            grab.invalidated = True
                    else:
                        if prev_high >= grab.pivot.price and close <= grab.pivot.price:
                            grabbed = True
                        elif close > grab.pivot.price:
                            grab.invalidated = True
                    if grabbed:
                        grab_bar_index = i - 1
                        grab_price = prev_low if grab.pivot.type == -1 else prev_high
                        grab.limit = pa.Line(
                            grab.pivot.bar_index,
                            grab.pivot.price,
                            grab_bar_index,
                            grab.pivot.price,
                            color=color,
                            style=line_style,
                        )
                        grab.break_line = pa.Line(
                            grab.pivot.bar_index,
                            grab_price,
                            grab_bar_index,
                            grab_price,
                        )
                        grab.linefill = pa.LineFill(grab.limit, grab.break_line, color=f"{color}@80")
                        grab.label = pa.Label(
                            int(grab_bar_index - ((grab_bar_index - grab.pivot.bar_index) / 2)),
                            grab.pivot.price,
                            "$$$",
                            textcolor=f"{color}@30",
                            style="label_up" if grab.pivot.type == -1 else None,
                            size=liquidity_font_size,
                        )
                        grab.taken = True

            _process_grabs(grabs_highs + grabs_lows, grabs_color)
            _process_grabs(big_grabs_highs + big_grabs_lows, big_grabs_color)

            for sweep in sweeps_highs + sweeps_lows:
                if sweep.taken or sweep.invalidated:
                    continue
                swept = False
                if sweep.pivot.type == -1:
                    if prev_low <= sweep.pivot.price and close <= sweep.pivot.price:
                        if previous_structure_break_pivot and sweep.pivot.bar_index == previous_structure_break_pivot.bar_index:
                            sweep.invalidated = True
                        else:
                            swept = True
                    elif prev_low <= sweep.pivot.price and close >= sweep.pivot.price:
                        sweep.invalidated = True
                else:
                    if prev_high >= sweep.pivot.price and close >= sweep.pivot.price:
                        if previous_structure_break_pivot and sweep.pivot.bar_index == previous_structure_break_pivot.bar_index:
                            sweep.invalidated = True
                        else:
                            swept = True
                    elif prev_high >= sweep.pivot.price and close <= sweep.pivot.price:
                        sweep.invalidated = True
                if swept:
                    sweep_bar_index = i - 1
                    sweep_price = prev_low if sweep.pivot.type == -1 else prev_high
                    sweep_color = sweeps_bearish_color if sweep.pivot.type == -1 else sweeps_bullish_color
                    sweep.limit = pa.Line(
                        sweep.pivot.bar_index,
                        sweep.pivot.price,
                        sweep_bar_index,
                        sweep.pivot.price,
                        color=sweep_color,
                        style=line_style,
                    )
                    sweep.label = pa.Label(
                        int(sweep_bar_index - ((sweep_bar_index - sweep.pivot.bar_index) / 2)),
                        sweep.pivot.price,
                        "$",
                        textcolor=f"{sweep_color}@30",
                        style="label_up" if sweep.pivot.type == -1 else None,
                        size=liquidity_font_size,
                    )
                    sweep.taken = True
                else:
                    grabbed = False
                    if sweep.pivot.type == -1:
                        grabbed = prev_low <= sweep.pivot.price and close >= sweep.pivot.price
                    else:
                        grabbed = prev_high >= sweep.pivot.price and close <= sweep.pivot.price
                    if grabbed:
                        sweep.invalidated = True

        # Store new pivots AFTER detection (matching Pine Script order).
        if grabs_enabled and grab_closed[i] and grab_high_series[i]:
            grabs_highs.insert(0, LiquidityGrab(grab_high_series[i]))
            grabs_highs = grabs_highs[:grabs_lookback]
        if grabs_enabled and grab_closed[i] and grab_low_series[i]:
            grabs_lows.insert(0, LiquidityGrab(grab_low_series[i]))
            grabs_lows = grabs_lows[:grabs_lookback]
        if big_grabs_enabled and big_grab_closed[i] and big_grab_high_series[i]:
            big_grabs_highs.insert(0, LiquidityGrab(big_grab_high_series[i]))
            big_grabs_highs = big_grabs_highs[:big_grabs_lookback]
        if big_grabs_enabled and big_grab_closed[i] and big_grab_low_series[i]:
            big_grabs_lows.insert(0, LiquidityGrab(big_grab_low_series[i]))
            big_grabs_lows = big_grabs_lows[:big_grabs_lookback]

        if sweeps_enabled and sweep_closed[i] and sweep_high_series[i]:
            sweeps_highs.insert(0, LiquidityGrab(sweep_high_series[i]))
            sweeps_highs = sweeps_highs[:sweeps_lookback]
        if sweeps_enabled and sweep_closed[i] and sweep_low_series[i]:
            sweeps_lows.insert(0, LiquidityGrab(sweep_low_series[i]))
            sweeps_lows = sweeps_lows[:sweeps_lookback]
        if sweeps_enabled and change_of_character and previous_structure_break_index is not None:
            sweeps_highs.clear()
            sweeps_lows.clear()

        if turtle_soups_enabled:
            pa.visualize_turtle_soups(
                turtle_context.highs,
                turtle_context.bearish,
                turtle_context,
                turtle_settings,
                highs=df["high"],
                lows=df["low"],
                times=df.index,
                bar_index=i,
            )
            pa.visualize_turtle_soups(
                turtle_context.lows,
                turtle_context.bullish,
                turtle_context,
                turtle_settings,
                highs=df["high"],
                lows=df["low"],
                times=df.index,
                bar_index=i,
            )
            if turtle_confirmation and change_of_character and previous_structure_break_index is not None:
                pa.confirm(
                    turtle_context.bullish,
                    turtle_context,
                    turtle_settings,
                    previous_structure_break_index,
                    turtle_screener,
                    i,
                )
                pa.confirm(
                    turtle_context.bearish,
                    turtle_context,
                    turtle_settings,
                    previous_structure_break_index,
                    turtle_screener,
                    i,
                )
            if turtle_closed[i]:
                ph = turtle_high_series[i]
                pl = turtle_low_series[i]
                pa.set_pivots(turtle_context, turtle_settings, ph, pl)

        if equal_pivots_enabled:
            atr_val = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
            eq_high = equal_high_series[i]
            eq_low = equal_low_series[i]
            if eq_high:
                equal_state.highs.insert(0, eq_high)
                equal_state.highs = equal_state.highs[:equal_lookback]
            if eq_low:
                equal_state.lows.insert(0, eq_low)
                equal_state.lows = equal_state.lows[:equal_lookback]

            for pivots, inducements, direction in (
                (equal_state.highs, equal_state.bearish_inducements, -1),
                (equal_state.lows, equal_state.bullish_inducements, 1),
            ):
                if len(pivots) < 2:
                    continue
                latest = pivots[0]
                if latest.bar_index != i - 1:
                    continue
                for equal_pivot in pivots[1:]:
                    max_price = equal_pivot.price + (atr_val * equal_atr_factor) if latest.type == -1 else equal_pivot.price
                    min_price = equal_pivot.price if latest.type == -1 else equal_pivot.price - (atr_val * equal_atr_factor)
                    if latest.price > max_price or latest.price < min_price:
                        continue
                    broken = False
                    if latest.type == 1:
                        step = (equal_pivot.price - latest.price) / max(1, latest.bar_index - equal_pivot.bar_index)
                    else:
                        bar_span = equal_pivot.bar_index - latest.bar_index
                        if bar_span == 0:
                            continue
                        step = (latest.price - equal_pivot.price) / bar_span
                    for j in range(2, latest.bar_index - equal_pivot.bar_index + 1):
                        bar_price = latest.price + (step * (j - 1))
                        if latest.type == 1 and df["high"].iloc[i - j] > bar_price:
                            broken = True
                            break
                        if latest.type == -1 and df["low"].iloc[i - j] < bar_price:
                            broken = True
                            break
                    if broken:
                        continue
                    trend_inducement = (latest.type == 1 and structure.trend == -1) or (latest.type == -1 and structure.trend == 1)
                    label_text = "IDM" if trend_inducement else "$$$"
                    label_color = equal_liquidity_color
                    if trend_inducement:
                        label_color = equal_bullish_inducement_color if structure.trend == 1 else equal_bearish_inducement_color
                    label_style = "label_up" if latest.type == -1 and structure.trend == -1 else "label_down"
                    label = pa.Label(
                        int(latest.bar_index - ((latest.bar_index - equal_pivot.bar_index) / 2)),
                        latest.price + ((equal_pivot.price - latest.price) / 2),
                        label_text,
                        textcolor=label_color,
                        style=label_style,
                        size=equal_font_size,
                    )
                    line = pa.Line(
                        latest.bar_index,
                        latest.price,
                        equal_pivot.bar_index,
                        equal_pivot.price,
                        color=label_color,
                        style=line_style,
                    )
                    if trend_inducement:
                        if latest.type == 1:
                            stop_price = equal_pivot.price + (atr_val * 0.1)
                            inducements.insert(0, EqualPivotInducement(stop_price, equal_pivot, latest, label, line))
                        else:
                            label.style = "label_up"
                            stop_price = equal_pivot.price - (atr_val * 0.1)
                            inducements.insert(0, EqualPivotInducement(stop_price, equal_pivot, latest, label, line))

            for inducement in equal_state.bearish_inducements:
                if structure.trend == -1 and not inducement.liquidity_taken and high >= inducement.stop_losses:
                    inducement.liquidity_taken = True
                    pa.Line(
                        inducement.first_pivot.bar_index,
                        inducement.stop_losses,
                        inducement.second_pivot.bar_index,
                        inducement.stop_losses,
                        color=equal_liquidity_color,
                        style=line_style,
                    )
                    pa.Label(
                        int(
                            inducement.second_pivot.bar_index
                            - ((inducement.second_pivot.bar_index - inducement.first_pivot.bar_index) / 2)
                        ),
                        inducement.stop_losses,
                        "$$$",
                        textcolor=equal_liquidity_color,
                        size=equal_font_size,
                    )
            for inducement in equal_state.bullish_inducements:
                if structure.trend == 1 and not inducement.liquidity_taken and low <= inducement.stop_losses:
                    inducement.liquidity_taken = True
                    pa.Line(
                        inducement.first_pivot.bar_index,
                        inducement.stop_losses,
                        inducement.second_pivot.bar_index,
                        inducement.stop_losses,
                        color=equal_liquidity_color,
                        style=line_style,
                    )
                    pa.Label(
                        int(
                            inducement.second_pivot.bar_index
                            - ((inducement.second_pivot.bar_index - inducement.first_pivot.bar_index) / 2)
                        ),
                        inducement.stop_losses,
                        "$$$",
                        textcolor=equal_liquidity_color,
                        style="label_up",
                        size=equal_font_size,
                    )
            if structure_break_event:
                equal_state.bullish_inducements.clear()
                equal_state.bearish_inducements.clear()

        if external_liquidity_enabled:
            if last_high and last_high.bar_index == i - market_right:
                for pool in buyside:
                    if not pool.hidden:
                        pool.hidden = True
                        if pool.line:
                            pool.line.set_color(None)
                        if pool.label:
                            pool.label.set_textcolor(None)
                line = pa.Line(
                    last_high.bar_index,
                    last_high.price,
                    i,
                    last_high.price,
                    style=external_line_style,
                )
                line.set_extend_right(True)
                label = pa.Label(
                    last_high.bar_index,
                    last_high.price,
                    "Buyside liquidity",
                    style="label_down",
                )
                buyside.insert(0, ExternalLiquidity(last_high.price, last_high, line, label, hidden=True))
            if last_low and last_low.bar_index == i - market_right:
                for pool in sellside:
                    if not pool.hidden:
                        pool.hidden = True
                        if pool.line:
                            pool.line.set_color(None)
                        if pool.label:
                            pool.label.set_textcolor(None)
                line = pa.Line(
                    last_low.bar_index,
                    last_low.price,
                    i,
                    last_low.price,
                    style=external_line_style,
                )
                line.set_extend_right(True)
                label = pa.Label(
                    last_low.bar_index,
                    last_low.price,
                    "Sellside liquidity",
                    style="label_up",
                )
                sellside.insert(0, ExternalLiquidity(last_low.price, last_low, line, label, hidden=True))
            remaining_sellside: List[ExternalLiquidity] = []
            for pool in sellside:
                if low <= pool.price:
                    if pool.label:
                        pool.label.delete()
                    if pool.line:
                        pool.line.delete()
                    continue
                remaining_sellside.append(pool)
            sellside = remaining_sellside
            remaining_buyside: List[ExternalLiquidity] = []
            for pool in buyside:
                if high >= pool.price:
                    if pool.label:
                        pool.label.delete()
                    if pool.line:
                        pool.line.delete()
                    continue
                remaining_buyside.append(pool)
            buyside = remaining_buyside
            for i_pool, pool in enumerate(buyside):
                if i_pool + 1 <= external_show:
                    pool.hidden = False
                    if pool.line:
                        pool.line.set_color(external_bullish_color)
                    if pool.label:
                        pool.label.set_textcolor(external_bullish_color)
            for i_pool, pool in enumerate(sellside):
                if i_pool + 1 <= external_show:
                    pool.hidden = False
                    if pool.line:
                        pool.line.set_color(external_bearish_color)
                    if pool.label:
                        pool.label.set_textcolor(external_bearish_color)

        if retracement_inducements_enabled:
            rh = retr_high_series[i]
            rl = retr_low_series[i]
            if rh:
                retr_state.high_pivots.insert(0, rh)
                retr_state.high_pivots = retr_state.high_pivots[:retr_lookback]
            if rl:
                retr_state.low_pivots.insert(0, rl)
                retr_state.low_pivots = retr_state.low_pivots[:retr_lookback]
            if structure.trend != 0:
                pivots = retr_state.high_pivots if structure.trend == -1 else retr_state.low_pivots
                if len(pivots) > 1:
                    latest = pivots[0]
                    next_latest = pivots[1]
                    if retracement_structure_break_index is not None:
                        latest_after_break = latest.bar_index > retracement_structure_break_index
                        if (
                            latest.bar_index == i - retr_right
                            and latest_after_break
                            and next_latest.bar_index < retracement_structure_break_index
                        ):
                            target_list = retr_state.highs if structure.trend == -1 else retr_state.lows
                            line_color = retr_bearish_color if structure.trend == -1 else retr_bullish_color
                            label_style = "label_down" if structure.trend == -1 else "label_up"
                            line = pa.Line(
                                latest.bar_index,
                                latest.price,
                                i,
                                latest.price,
                                style=retr_line_style,
                                color=line_color,
                            )
                            line.set_extend_right(True)
                            label = pa.Label(
                                latest.bar_index,
                                latest.price,
                                "IDM",
                                textcolor=line_color,
                                style=label_style,
                                size=retr_font_size,
                            )
                            target_list.insert(0, RetracementInducement(latest, line=line, label=label))
            _stop_retracement_inducements(
                retr_state,
                high=high,
                low=low,
                bar_index=i,
                stop_reason="take",
                keep_invalidated=retr_keep_invalidated,
            )
            if structure_break_event:
                _stop_retracement_inducements(
                    retr_state,
                    high=high,
                    low=low,
                    bar_index=i,
                    stop_reason="invalidate",
                    keep_invalidated=retr_keep_invalidated,
                )
            if structure_break_event:
                retracement_structure_break_index = i

        buyside_targets.iloc[i] = buyside[0].price if buyside else np.nan
        sellside_targets.iloc[i] = sellside[0].price if sellside else np.nan

        if structure_break_event:
            previous_structure_break_index = i

    return {
        "trend": structure.trend,
        "change_of_character": change_of_character,
        "break_of_structure": break_of_structure,
        "previous_structure_break_pivot": previous_structure_break_pivot,
        "grabs_highs": grabs_highs,
        "grabs_lows": grabs_lows,
        "big_grabs_highs": big_grabs_highs,
        "big_grabs_lows": big_grabs_lows,
        "sweeps_highs": sweeps_highs,
        "sweeps_lows": sweeps_lows,
        "equal_pivots": equal_state,
        "turtle_soups": turtle_context,
        "buyside_liquidity": buyside,
        "sellside_liquidity": sellside,
        "buyside_targets": buyside_targets,
        "sellside_targets": sellside_targets,
        "retracement_inducements": retr_state,
    }
