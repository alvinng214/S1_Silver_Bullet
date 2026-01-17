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


@dataclass
class Pivot:
    price: float
    type: int  # 1 high, -1 low
    time: pd.Timestamp
    bar_index: int
    break_of_structure_broken: bool = False
    liquidity_broken: bool = False
    change_of_character_broken: bool = False


@dataclass
class Liquidity:
    pivot: Pivot
    taken: bool = False
    invalidated: bool = False


@dataclass
class EqualPivotInducement:
    stop_losses: float
    first_pivot: Pivot
    second_pivot: Pivot
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


@dataclass
class RetracementState:
    highs: List[RetracementInducement] = field(default_factory=list)
    lows: List[RetracementInducement] = field(default_factory=list)
    high_pivots: List[Pivot] = field(default_factory=list)
    low_pivots: List[Pivot] = field(default_factory=list)


@dataclass
class TurtleSoupEvent:
    pivot: Pivot
    direction: int
    index: int
    confirmed: bool


@dataclass
class ExternalLiquidity:
    price: float
    pivot: Pivot
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
        if (high.iloc[pivot_idx - left : pivot_idx] >= center_high).any():
            continue
        if (high.iloc[pivot_idx + 1 : pivot_idx + right + 1] >= center_high).any():
            continue
        pivot_high[i] = center_high
        if (low.iloc[pivot_idx - left : pivot_idx] <= center_low).any():
            continue
        if (low.iloc[pivot_idx + 1 : pivot_idx + right + 1] <= center_low).any():
            continue
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
            piv_high_list[i] = Pivot(float(ph), 1, time_val, _bar_index_for_time(df.index, time_val))
        if not np.isnan(pl):
            time_val = htf.index[htf_idx - right]
            piv_low_list[i] = Pivot(float(pl), -1, time_val, _bar_index_for_time(df.index, time_val))
    return piv_high_list, piv_low_list, closed_flags


def _append_structure_pivot(structure_pivots: List[Pivot], pivot: Pivot, max_len: int = 5) -> None:
    structure_pivots.insert(0, pivot)
    if len(structure_pivots) > max_len:
        structure_pivots.pop()


def _detect_break_of_structure(structure_pivots: List[Pivot], trend: int, close: float) -> Optional[Pivot]:
    if trend == 1:
        for pivot in structure_pivots:
            if pivot.type != 1 or pivot.break_of_structure_broken:
                continue
            if close > pivot.price:
                pivot.break_of_structure_broken = True
                return pivot
    if trend == -1:
        for pivot in structure_pivots:
            if pivot.type != -1 or pivot.break_of_structure_broken:
                continue
            if close < pivot.price:
                pivot.break_of_structure_broken = True
                return pivot
    return None


def _detect_change_of_character(
    structure_pivots: List[Pivot],
    trend: int,
    close: float,
    prev_close: float,
) -> Tuple[Optional[Pivot], int]:
    if trend <= 0:
        for pivot in structure_pivots:
            if pivot.type != 1 or pivot.change_of_character_broken:
                continue
            if close > pivot.price and prev_close < pivot.price:
                pivot.change_of_character_broken = True
                remaining: List[Pivot] = []
                for existing in structure_pivots:
                    if existing.bar_index <= pivot.bar_index:
                        continue
                    existing.break_of_structure_broken = True
                    remaining.append(existing)
                for existing in remaining:
                    if existing.bar_index != pivot.bar_index:
                        existing.change_of_character_broken = False
                structure_pivots[:] = remaining
                return pivot, 1
    if trend >= 0:
        for pivot in structure_pivots:
            if pivot.type != -1 or pivot.change_of_character_broken:
                continue
            if close < pivot.price and prev_close > pivot.price:
                pivot.change_of_character_broken = True
                remaining = []
                for existing in structure_pivots:
                    if existing.bar_index <= pivot.bar_index:
                        continue
                    existing.break_of_structure_broken = True
                    remaining.append(existing)
                for existing in remaining:
                    if existing.bar_index != pivot.bar_index:
                        existing.change_of_character_broken = False
                structure_pivots[:] = remaining
                return pivot, -1
    return None, trend


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
    equal_left: int = 1,
    equal_right: int = 1,
    equal_atr_factor: float = 0.5,
    equal_lookback: int = 3,
    equal_tf: str = "",
    external_show: int = 1,
    retr_left: int = 1,
    retr_right: int = 1,
    retr_lookback: int = 5,
    retr_tf: str = "",
    retr_keep_invalidated: bool = False,
) -> Dict[str, object]:
    atr = _atr(df, 14)

    structure_high, structure_low = _pivot_series(df["high"], df["low"], market_left, market_right)
    structure_pivots: List[Pivot] = []
    trend = 0
    change_of_character: Optional[Pivot] = None
    break_of_structure: Optional[Pivot] = None
    previous_structure_break_pivot: Optional[Pivot] = None
    previous_structure_break_index: Optional[int] = None

    grabs_highs: List[Liquidity] = []
    grabs_lows: List[Liquidity] = []
    sweeps_highs: List[Liquidity] = []
    sweeps_lows: List[Liquidity] = []
    equal_state = EqualPivotState()
    retr_state = RetracementState()
    turtle_events: List[TurtleSoupEvent] = []
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
        prev_close = float(df["close"].iloc[i - 1]) if i > 0 else close

        struct_high_val = structure_high.iloc[i]
        struct_low_val = structure_low.iloc[i]
        if not np.isnan(struct_high_val):
            pivot = Pivot(float(struct_high_val), 1, df.index[i - market_right], i - market_right)
            _append_structure_pivot(structure_pivots, pivot)
        if not np.isnan(struct_low_val):
            pivot = Pivot(float(struct_low_val), -1, df.index[i - market_right], i - market_right)
            _append_structure_pivot(structure_pivots, pivot)

        last_high = next((p for p in structure_pivots if p.type == 1), None)
        last_low = next((p for p in structure_pivots if p.type == -1), None)

        change_of_character, trend = _detect_change_of_character(structure_pivots, trend, close, prev_close)
        if change_of_character:
            break_of_structure = None
            previous_structure_break_pivot = change_of_character
            previous_structure_break_index = i

        bos_pivot = _detect_break_of_structure(structure_pivots, trend, close)
        if bos_pivot:
            break_of_structure = bos_pivot
            previous_structure_break_pivot = bos_pivot
            previous_structure_break_index = i

        if grabs_enabled and grab_closed[i] and grab_high_series[i]:
            grabs_highs.insert(0, Liquidity(grab_high_series[i]))
            grabs_highs = grabs_highs[:grabs_lookback]
        if grabs_enabled and grab_closed[i] and grab_low_series[i]:
            grabs_lows.insert(0, Liquidity(grab_low_series[i]))
            grabs_lows = grabs_lows[:grabs_lookback]
        if big_grabs_enabled and big_grab_closed[i] and big_grab_high_series[i]:
            grabs_highs.insert(0, Liquidity(big_grab_high_series[i]))
            grabs_highs = grabs_highs[:big_grabs_lookback]
        if big_grabs_enabled and big_grab_closed[i] and big_grab_low_series[i]:
            grabs_lows.insert(0, Liquidity(big_grab_low_series[i]))
            grabs_lows = grabs_lows[:big_grabs_lookback]

        if sweeps_enabled and sweep_closed[i] and sweep_high_series[i]:
            sweeps_highs.insert(0, Liquidity(sweep_high_series[i]))
            sweeps_highs = sweeps_highs[:sweeps_lookback]
        if sweeps_enabled and sweep_closed[i] and sweep_low_series[i]:
            sweeps_lows.insert(0, Liquidity(sweep_low_series[i]))
            sweeps_lows = sweeps_lows[:sweeps_lookback]

        if i > 0:
            prev_high = float(df["high"].iloc[i - 1])
            prev_low = float(df["low"].iloc[i - 1])
            for grab in grabs_highs + grabs_lows:
                if grab.taken or grab.invalidated:
                    continue
                if grab.pivot.type == -1:
                    if prev_low <= grab.pivot.price and close >= grab.pivot.price:
                        grab.taken = True
                    elif close < grab.pivot.price:
                        grab.invalidated = True
                else:
                    if prev_high >= grab.pivot.price and close <= grab.pivot.price:
                        grab.taken = True
                    elif close > grab.pivot.price:
                        grab.invalidated = True

            for sweep in sweeps_highs + sweeps_lows:
                if sweep.taken or sweep.invalidated:
                    continue
                if sweep.pivot.type == -1:
                    if prev_low <= sweep.pivot.price and close <= sweep.pivot.price:
                        if previous_structure_break_pivot and sweep.pivot.bar_index == previous_structure_break_pivot.bar_index:
                            sweep.invalidated = True
                        else:
                            sweep.taken = True
                    elif prev_low <= sweep.pivot.price and close >= sweep.pivot.price:
                        sweep.invalidated = True
                else:
                    if prev_high >= sweep.pivot.price and close >= sweep.pivot.price:
                        if previous_structure_break_pivot and sweep.pivot.bar_index == previous_structure_break_pivot.bar_index:
                            sweep.invalidated = True
                        else:
                            sweep.taken = True
                    elif prev_high >= sweep.pivot.price and close <= sweep.pivot.price:
                        sweep.invalidated = True

        if turtle_soups_enabled and turtle_closed[i]:
            ph = turtle_high_series[i]
            pl = turtle_low_series[i]
            if ph:
                pivot = ph
                if i > 0 and df["high"].iloc[i - 1] >= pivot.price and close <= pivot.price:
                    confirmed = not turtle_confirmation
                    if turtle_confirmation and change_of_character and change_of_character.type == -1:
                        confirmed = True
                    turtle_events.append(TurtleSoupEvent(pivot, -1, i - 1, confirmed))
            if pl:
                pivot = pl
                if i > 0 and df["low"].iloc[i - 1] <= pivot.price and close >= pivot.price:
                    confirmed = not turtle_confirmation
                    if turtle_confirmation and change_of_character and change_of_character.type == 1:
                        confirmed = True
                    turtle_events.append(TurtleSoupEvent(pivot, 1, i - 1, confirmed))

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
                    step = (equal_pivot.price - latest.price) / max(1, latest.bar_index - equal_pivot.bar_index)
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
                    trend_inducement = (latest.type == 1 and trend == -1) or (latest.type == -1 and trend == 1)
                    if trend_inducement:
                        stop_price = equal_pivot.price + (atr_val * 0.1) if latest.type == 1 else equal_pivot.price - (atr_val * 0.1)
                        inducements.insert(0, EqualPivotInducement(stop_price, equal_pivot, latest))

            for inducement in equal_state.bearish_inducements:
                if trend == -1 and not inducement.liquidity_taken and high >= inducement.stop_losses:
                    inducement.liquidity_taken = True
            for inducement in equal_state.bullish_inducements:
                if trend == 1 and not inducement.liquidity_taken and low <= inducement.stop_losses:
                    inducement.liquidity_taken = True

        if external_liquidity_enabled:
            if last_high and last_high.bar_index == i - market_right:
                for pool in buyside:
                    pool.hidden = True
                buyside.insert(0, ExternalLiquidity(last_high.price, last_high, hidden=True))
            if last_low and last_low.bar_index == i - market_right:
                for pool in sellside:
                    pool.hidden = True
                sellside.insert(0, ExternalLiquidity(last_low.price, last_low, hidden=True))
            sellside = [p for p in sellside if low > p.price]
            buyside = [p for p in buyside if high < p.price]
            for i_pool, pool in enumerate(buyside):
                if i_pool + 1 <= external_show:
                    pool.hidden = False
            for i_pool, pool in enumerate(sellside):
                if i_pool + 1 <= external_show:
                    pool.hidden = False

        if retracement_inducements_enabled:
            rh = retr_high_series[i]
            rl = retr_low_series[i]
            if rh:
                retr_state.high_pivots.insert(0, rh)
                retr_state.high_pivots = retr_state.high_pivots[:retr_lookback]
            if rl:
                retr_state.low_pivots.insert(0, rl)
                retr_state.low_pivots = retr_state.low_pivots[:retr_lookback]
            if trend != 0:
                pivots = retr_state.high_pivots if trend == -1 else retr_state.low_pivots
                if len(pivots) > 1:
                    latest = pivots[0]
                    next_latest = pivots[1]
                    if previous_structure_break_index is not None:
                        latest_after_break = latest.bar_index > previous_structure_break_index
                        if latest.bar_index == i - retr_right and latest_after_break and next_latest.bar_index < previous_structure_break_index:
                            target_list = retr_state.highs if trend == -1 else retr_state.lows
                            target_list.insert(0, RetracementInducement(latest))
            if previous_structure_break_index is not None:
                if change_of_character or break_of_structure:
                    if retr_keep_invalidated:
                        pass
                    else:
                        retr_state.highs.clear()
                        retr_state.lows.clear()

    return {
        "trend": trend,
        "change_of_character": change_of_character,
        "break_of_structure": break_of_structure,
        "previous_structure_break_pivot": previous_structure_break_pivot,
        "grabs_highs": grabs_highs,
        "grabs_lows": grabs_lows,
        "sweeps_highs": sweeps_highs,
        "sweeps_lows": sweeps_lows,
        "equal_pivots": equal_state,
        "turtle_soups": turtle_events,
        "buyside_liquidity": buyside,
        "sellside_liquidity": sellside,
        "retracement_inducements": retr_state,
    }
