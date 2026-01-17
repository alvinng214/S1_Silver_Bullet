"""
Silver Bullet ICT Strategy [TradingFinder] 10-11 AM NY Time +FVG
TFlab Silver Bullet - Python Translation

This module mirrors the Pine Script logic by:
- Tracking the New York opening range (09:00-10:00) highs/lows.
- Tracking the New York trading window (10:00-11:00) for breaks.
- Detecting FVGs with optional width filtering.
- Building CISD levels and trigger signals with TradingFinder-style logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FVGDetection:
    demand_condition: pd.Series
    demand_distal: pd.Series
    demand_proximal: pd.Series
    demand_bar: pd.Series
    supply_condition: pd.Series
    supply_distal: pd.Series
    supply_proximal: pd.Series
    supply_bar: pd.Series


@dataclass
class CISDOutputs:
    bull_trigger: pd.Series
    bear_trigger: pd.Series
    fvg_bull_trigger: pd.Series
    fvg_bear_trigger: pd.Series
    bull_fvg_bar: pd.Series
    bear_fvg_bar: pd.Series
    bull_fvg_distal: pd.Series
    bear_fvg_distal: pd.Series
    bull_fvg_proximal: pd.Series
    bear_fvg_proximal: pd.Series
    bull_ob_index: pd.Series
    bear_ob_index: pd.Series
    cisd_high_level: pd.Series
    cisd_low_level: pd.Series
    cisd_high_index: pd.Series
    cisd_low_index: pd.Series


@dataclass
class SessionLevels:
    or_high: pd.Series
    or_low: pd.Series
    or_start_time: pd.Series
    or_range: pd.Series
    trading_range: pd.Series
    high_break: pd.Series
    low_break: pd.Series


@dataclass
class OBRefinerOutput:
    xd1: pd.Series
    xd2: pd.Series
    yd12: pd.Series
    xp1: pd.Series
    xp2: pd.Series
    yp12: pd.Series


@dataclass
class OBDrawingOutput:
    alert: pd.Series
    proximal: pd.Series
    distal: pd.Series
    index: pd.Series
    alert_bb: pd.Series
    proximal_bb: pd.Series
    distal_bb: pd.Series
    index_bb: pd.Series


def _atr(df: pd.DataFrame, length: int = 55) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(length).mean()


def _series_value(series: pd.Series, idx: int, offset: int) -> float:
    target = idx - offset
    if target < 0 or target >= len(series):
        return float("nan")
    return float(series.iloc[target])


def _session_mask(index: pd.DatetimeIndex, session: str, tz: str) -> pd.Series:
    start_str, end_str = session.split("-")
    start_hour = int(start_str[:2])
    start_min = int(start_str[2:])
    end_hour = int(end_str[:2])
    end_min = int(end_str[2:])
    local = index.tz_convert(tz) if index.tzinfo else index.tz_localize(tz)
    times = local.time
    start = pd.Timestamp(year=2000, month=1, day=1, hour=start_hour, minute=start_min).time()
    end = pd.Timestamp(year=2000, month=1, day=1, hour=end_hour, minute=end_min).time()
    mask = (times >= start) & (times <= end)
    return pd.Series(mask.astype(int), index=index)


def _low_high_session_detector(df: pd.DataFrame, on_session: pd.Series) -> SessionLevels:
    high_series = pd.Series(np.nan, index=df.index)
    low_series = pd.Series(np.nan, index=df.index)
    time_series = pd.Series(np.nan, index=df.index)

    current_high = 0.0
    current_low = 0.0
    current_time = np.nan

    for i in range(len(df)):
        prior_session = on_session.iloc[i - 1] if i > 0 else 0
        session_now = on_session.iloc[i]

        if prior_session == 0 and session_now == 1:
            current_time = df.index[i].value
            current_high = df["high"].iloc[i]
            current_low = df["low"].iloc[i]
        elif prior_session == 1 or session_now == 1:
            current_high = max(current_high, df["high"].iloc[i])
            current_low = min(current_low, df["low"].iloc[i])

        high_series.iloc[i] = current_high
        low_series.iloc[i] = current_low
        time_series.iloc[i] = current_time

    return SessionLevels(
        or_high=high_series,
        or_low=low_series,
        or_start_time=time_series,
        or_range=on_session.copy(),
        trading_range=pd.Series(np.nan, index=df.index),
        high_break=pd.Series(False, index=df.index),
        low_break=pd.Series(False, index=df.index),
    )


def _fvg_detector(
    df: pd.DataFrame,
    filter_on: bool,
    filter_type: str,
) -> FVGDetection:
    atr = _atr(df) if filter_on else None
    multipliers = {
        "Very Aggressive": 0.0,
        "Aggressive": 0.5,
        "Defensive": 0.7,
        "Very Defensive": 1.0,
    }
    multiplier = multipliers.get(filter_type, 0.7)

    atr = _atr(df, length=55)
    demand_condition = pd.Series(False, index=df.index)
    supply_condition = pd.Series(False, index=df.index)
    demand_distal = pd.Series(np.nan, index=df.index)
    demand_proximal = pd.Series(np.nan, index=df.index)
    supply_distal = pd.Series(np.nan, index=df.index)
    supply_proximal = pd.Series(np.nan, index=df.index)
    demand_bar = pd.Series(0, index=df.index)
    supply_bar = pd.Series(0, index=df.index)

    for i in range(2, len(df)):
        high_2 = df["high"].iloc[i - 2]
        low_2 = df["low"].iloc[i - 2]
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        if low > high_2:
            width = low - high_2
            if filter_on and atr is not None:
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * multiplier:
                    continue
    highs = df["high"]
    lows = df["low"]
    opens = df["open"]
    closes = df["close"]

    for i in range(2, len(df)):
        high_2 = highs.iloc[i - 2]
        low_2 = lows.iloc[i - 2]
        high = highs.iloc[i]
        low = lows.iloc[i]
        high_1 = highs.iloc[i - 1]
        low_1 = lows.iloc[i - 1]

        if not filter_on:
            d_condition = low > high_2
            s_condition = high < low_2
        else:
            atr_val = atr.iloc[i]
            body_ratio_1 = abs((closes.iloc[i - 1] - opens.iloc[i - 1]) / (high_1 - low_1)) if high_1 != low_1 else 0.0
            body_ratio_2 = abs((closes.iloc[i - 2] - opens.iloc[i - 2]) / (high_2 - low_2)) if high_2 != low_2 else 0.0
            body_ratio_0 = abs((closes.iloc[i] - opens.iloc[i]) / (high - low)) if high != low else 0.0
            if filter_type == "Very Aggressive":
                d_condition = (low > high_2) and (high > high_1)
                s_condition = (high < low_2) and (low_1 > low)
            elif filter_type == "Aggressive":
                d_condition = (low > high_2) and ((high_1 - low_1) >= (1.0 * atr_val)) and (high > high_1)
                s_condition = (high < low_2) and ((high_1 - low_1) >= (1.0 * atr_val)) and (low_1 > low)
            elif filter_type == "Defensive":
                d_condition = (
                    (low > high_2)
                    and ((high_1 - low_1) >= (1.5 * atr_val))
                    and (high > high_1)
                    and (
                        (closes.iloc[i - 2] - opens.iloc[i - 2] > 0 and closes.iloc[i - 1] - opens.iloc[i - 1] > 0)
                        or body_ratio_1 > 0.7
                    )
                )
                s_condition = (
                    (high < low_2)
                    and ((high_1 - low_1) >= (1.5 * atr_val))
                    and (low_1 > low)
                    and (
                        (closes.iloc[i - 2] - opens.iloc[i - 2] < 0 and closes.iloc[i - 1] - opens.iloc[i - 1] < 0)
                        or body_ratio_1 > 0.7
                    )
                )
            else:
                d_condition = (
                    (low > high_2)
                    and ((high_1 - low_1) >= (1.5 * atr_val))
                    and (high > high_1)
                    and (closes.iloc[i - 2] - opens.iloc[i - 2] > 0 and closes.iloc[i - 1] - opens.iloc[i - 1] > 0)
                    and body_ratio_1 > 0.7
                    and body_ratio_2 > 0.35
                    and body_ratio_0 > 0.35
                )
                s_condition = (
                    (high < low_2)
                    and ((high_1 - low_1) >= (1.5 * atr_val))
                    and (low_1 > low)
                    and (closes.iloc[i - 2] - opens.iloc[i - 2] < 0 and closes.iloc[i - 1] - opens.iloc[i - 1] < 0)
                    and body_ratio_1 > 0.7
                    and body_ratio_2 > 0.35
                    and body_ratio_0 > 0.35
                )

        if d_condition:
            demand_condition.iloc[i] = True
            demand_distal.iloc[i] = high_2
            demand_proximal.iloc[i] = low
            demand_bar.iloc[i] = i
        elif high < low_2:
            width = low_2 - high
            if filter_on and atr is not None:
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * multiplier:
                    continue
        if s_condition:
            supply_condition.iloc[i] = True
            supply_distal.iloc[i] = low_2
            supply_proximal.iloc[i] = high
            supply_bar.iloc[i] = i

    return FVGDetection(
        demand_condition=demand_condition,
        demand_distal=demand_distal,
        demand_proximal=demand_proximal,
        demand_bar=demand_bar,
        supply_condition=supply_condition,
        supply_distal=supply_distal,
        supply_proximal=supply_proximal,
        supply_bar=supply_bar,
    )


def _refine_ob(
    df: pd.DataFrame,
    *,
    direction: str,
    refine_on: bool,
    refine_mode: str,
    trigger: pd.Series,
    ob_index: pd.Series,
) -> OBRefinerOutput:
    xd1 = pd.Series(np.nan, index=df.index)
    xd2 = pd.Series(np.nan, index=df.index)
    yd12 = pd.Series(np.nan, index=df.index)
    xp1 = pd.Series(np.nan, index=df.index)
    xp2 = pd.Series(np.nan, index=df.index)
    yp12 = pd.Series(np.nan, index=df.index)

    current_index = np.nan
    current_distal = np.nan
    current_proximal = np.nan

    atr = _atr(df, length=55)
    highs = df["high"]
    lows = df["low"]
    opens = df["open"]
    closes = df["close"]
    ranges = highs - lows
    bodies = closes - opens

    for i in range(len(df)):
        if trigger.iloc[i]:
            idx = ob_index.iloc[i]
            idx = int(idx) if not pd.isna(idx) else i
            pos_index = i - idx if (i - idx) < 4999 else 2
            if idx < 0 or idx >= len(df):
                continue

            atr_val = atr.iloc[i]
            range_pos = _series_value(ranges, i, pos_index)
            body_pos = _series_value(bodies, i, pos_index)

            def _min_val(series: pd.Series, offsets: List[int]) -> float:
                values = [val for val in (_series_value(series, i, off) for off in offsets) if not np.isnan(val)]
                return min(values) if values else float("nan")

            def _max_val(series: pd.Series, offsets: List[int]) -> float:
                values = [val for val in (_series_value(series, i, off) for off in offsets) if not np.isnan(val)]
                return max(values) if values else float("nan")

            if direction == "Demand" and i > 4:
                offsets_full = [pos_index - 2, pos_index + 1, pos_index, pos_index - 1]
                offsets_mid = [pos_index + 1, pos_index, pos_index - 1]
                offsets_short = [pos_index + 1, pos_index]
                if i > idx + 1:
                    dmin_open = _min_val(opens, offsets_full)
                    dmin_high = _min_val(highs, offsets_full)
                    dmin_low = _min_val(lows, offsets_full)
                    dmin_close = _min_val(closes, offsets_full)
                    dmax_open = _max_val(opens, offsets_full)
                    dmax_high = _max_val(highs, offsets_full)
                    dmax_low = _max_val(lows, offsets_full)
                    dmax_close = _max_val(closes, offsets_full)
                elif i == idx + 1:
                    dmin_open = _min_val(opens, offsets_mid)
                    dmin_high = _min_val(highs, offsets_mid)
                    dmin_low = _min_val(lows, offsets_mid)
                    dmin_close = _min_val(closes, offsets_mid)
                    dmax_open = _max_val(opens, offsets_mid)
                    dmax_high = _max_val(highs, offsets_mid)
                    dmax_low = _max_val(lows, offsets_mid)
                    dmax_close = _max_val(closes, offsets_mid)
                else:
                    dmin_open = _min_val(opens, offsets_short)
                    dmin_high = _min_val(highs, offsets_short)
                    dmin_low = _min_val(lows, offsets_short)
                    dmin_close = _min_val(closes, offsets_short)
                    dmax_open = _max_val(opens, offsets_short)
                    dmax_high = _max_val(highs, offsets_short)
                    dmax_low = _max_val(lows, offsets_short)
                    dmax_close = _max_val(closes, offsets_short)

                dpa = dmax_close
                dda = dmin_low
                if range_pos <= (atr_val * 0.5):
                    dpa = _series_value(highs, i, pos_index)
                    dda = dmin_low
                elif (range_pos > (atr_val * 0.5)) and (range_pos <= atr_val):
                    if body_pos >= 0:
                        dpa = _series_value(closes, i, pos_index)
                        dda = dmin_low
                    else:
                        dpa = _series_value(opens, i, pos_index)
                        dda = dmin_low
                elif range_pos > atr_val:
                    if i > idx:
                        low_next = _series_value(lows, i, pos_index + 1)
                        high_pos = _series_value(highs, i, pos_index)
                        if low_next < high_pos:
                            if low_next > _series_value(lows, i, pos_index):
                                dpa = low_next
                                dda = _series_value(lows, i, pos_index)
                            else:
                                body_next = _series_value(bodies, i, pos_index + 1)
                                dcr_next = _series_value(ranges, i, pos_index + 1)
                                if body_next >= 0:
                                    if dcr_next > atr_val:
                                        dpa = dmax_open
                                        dda = dmin_low
                                    else:
                                        dpa = dmax_close
                                        dda = dmin_low
                                else:
                                    if dcr_next > atr_val:
                                        dpa = dmax_open
                                        dda = dmin_low
                                    else:
                                        dpa = dmax_open
                                        dda = dmin_low
                        else:
                            dmin_open_gap = dmin_low - dmin_open
                            dmin_close_gap = dmin_low - dmin_close
                            if (dmin_open_gap > dmin_close_gap) and ((dmin_open_gap <= atr_val) and (dmin_open_gap >= (atr_val * 0.5))):
                                dpa = dmin_open
                                dda = dmin_low
                            elif (dmin_open_gap < dmin_close_gap) and ((dmin_close_gap <= atr_val) and (dmin_close_gap >= (atr_val * 0.5))):
                                dpa = dmin_close
                                dda = dmin_low
                            else:
                                dpa = dmin_close
                                dda = dmin_low
                    else:
                        dcr_next = _series_value(ranges, i, pos_index + 1)
                        if dcr_next > atr_val:
                            dpa = dmax_open
                            dda = dmin_low
                        else:
                            dpa = dmax_close
                            dda = dmin_low
                else:
                    if body_pos > 0:
                        dpa = dmax_high
                        dda = dmin_low
                    else:
                        dpa = dmax_high
                        dda = dmax_low

                if refine_mode == "Defensive":
                    span = dpa - dda
                    if span >= atr_val * 4:
                        distal_refine = dda
                        proximal_refine = dpa - span * 0.8
                    elif span >= atr_val * 3:
                        distal_refine = dda
                        proximal_refine = dpa - span * 0.7
                    elif span >= atr_val * 2:
                        distal_refine = dda
                        proximal_refine = dpa - span * 0.6
                    elif span >= atr_val * 1.6:
                        distal_refine = dda
                        proximal_refine = dpa - span * 0.5
                    elif span > atr_val:
                        distal_refine = dda
                        proximal_refine = dpa - span * 0.25
                    else:
                        distal_refine = dda
                        proximal_refine = dpa
                else:
                    distal_refine = dda
                    proximal_refine = dpa

                y_p = proximal_refine if refine_on else _series_value(highs, i, pos_index)
                y_d = distal_refine if refine_on else _series_value(lows, i, pos_index)

            elif direction == "Supply" and i > 4:
                offsets_full = [pos_index - 2, pos_index + 1, pos_index, pos_index - 1]
                offsets_mid = [pos_index + 1, pos_index, pos_index - 1]
                offsets_short = [pos_index + 1, pos_index]
                if i > idx + 1:
                    smin_open = _min_val(opens, offsets_full)
                    smin_high = _min_val(highs, offsets_full)
                    smin_low = _min_val(lows, offsets_full)
                    smin_close = _min_val(closes, offsets_full)
                    smax_open = _max_val(opens, offsets_full)
                    smax_high = _max_val(highs, offsets_full)
                    smax_low = _max_val(lows, offsets_full)
                    smax_close = _max_val(closes, offsets_full)
                elif i == idx + 1:
                    smin_open = _min_val(opens, offsets_mid)
                    smin_high = _min_val(highs, offsets_mid)
                    smin_low = _min_val(lows, offsets_mid)
                    smin_close = _min_val(closes, offsets_mid)
                    smax_open = _max_val(opens, offsets_mid)
                    smax_high = _max_val(highs, offsets_mid)
                    smax_low = _max_val(lows, offsets_mid)
                    smax_close = _max_val(closes, offsets_mid)
                else:
                    smin_open = _min_val(opens, offsets_short)
                    smin_high = _min_val(highs, offsets_short)
                    smin_low = _min_val(lows, offsets_short)
                    smin_close = _min_val(closes, offsets_short)
                    smax_open = _max_val(opens, offsets_short)
                    smax_high = _max_val(highs, offsets_short)
                    smax_low = _max_val(lows, offsets_short)
                    smax_close = _max_val(closes, offsets_short)

                spa = smin_close
                sda = smax_high
                if range_pos <= (atr_val * 0.5):
                    spa = _series_value(lows, i, pos_index)
                    sda = smax_high
                elif (range_pos > (atr_val * 0.5)) and (range_pos <= atr_val):
                    if body_pos >= 0:
                        spa = _series_value(closes, i, pos_index)
                        sda = smax_high
                    else:
                        spa = _series_value(opens, i, pos_index)
                        sda = smax_high
                elif range_pos > atr_val:
                    if i > idx:
                        high_prev = _series_value(highs, i, pos_index - 1)
                        low_pos = _series_value(lows, i, pos_index)
                        if high_prev > low_pos:
                            if high_prev > _series_value(highs, i, pos_index):
                                spa = high_prev
                                sda = _series_value(highs, i, pos_index)
                            else:
                                body_prev = _series_value(bodies, i, pos_index - 1)
                                scr_prev = _series_value(ranges, i, pos_index - 1)
                                if body_prev >= 0:
                                    if scr_prev > atr_val:
                                        spa = smin_close
                                        sda = smax_high
                                    else:
                                        spa = smin_open
                                        sda = smax_high
                                else:
                                    if scr_prev > atr_val:
                                        spa = smin_open
                                        sda = smax_high
                                    else:
                                        spa = smax_close
                                        sda = smax_high
                        else:
                            smax_open_gap = smax_high - smin_open
                            smax_close_gap = smax_high - smin_close
                            if (smax_open_gap > smax_close_gap) and ((smax_open_gap <= atr_val) and (smax_open_gap >= (atr_val * 0.5))):
                                spa = smin_open
                                sda = smax_high
                            elif (smax_open_gap < smax_close_gap) and ((smax_close_gap <= atr_val) and (smax_close_gap >= (atr_val * 0.5))):
                                spa = smin_close
                                sda = smax_high
                            else:
                                spa = smin_close
                                sda = smax_high
                    else:
                        scr_next = _series_value(ranges, i, pos_index + 1)
                        if scr_next > atr_val:
                            spa = smin_close
                            sda = smax_high
                        else:
                            spa = smin_open
                            sda = smax_high
                else:
                    if body_pos > 0:
                        spa = smin_low
                        sda = smax_high
                    else:
                        spa = smin_low
                        sda = smax_high

                if refine_mode == "Defensive":
                    span = sda - spa
                    if span >= atr_val * 4:
                        distal_refine = sda
                        proximal_refine = spa + span * 0.8
                    elif span >= atr_val * 3:
                        distal_refine = sda
                        proximal_refine = spa + span * 0.7
                    elif span >= atr_val * 2:
                        distal_refine = sda
                        proximal_refine = spa + span * 0.6
                    elif span >= atr_val * 1.6:
                        distal_refine = sda
                        proximal_refine = spa + span * 0.5
                    elif span > atr_val:
                        distal_refine = sda
                        proximal_refine = spa + span * 0.25
                    else:
                        distal_refine = sda
                        proximal_refine = spa
                else:
                    distal_refine = sda
                    proximal_refine = spa

                y_p = proximal_refine if refine_on else _series_value(lows, i, pos_index)
                y_d = distal_refine if refine_on else _series_value(highs, i, pos_index)
            else:
                y_p = np.nan
                y_d = np.nan

            current_index = idx
            current_distal = y_d
            current_proximal = y_p

        xd1.iloc[i] = current_index
        xd2.iloc[i] = current_index
        yd12.iloc[i] = current_distal
        xp1.iloc[i] = current_index
        xp2.iloc[i] = current_index
        yp12.iloc[i] = current_proximal

    return OBRefinerOutput(
        xd1=xd1,
        xd2=xd2,
        yd12=yd12,
        xp1=xp1,
        xp2=xp2,
        yp12=yp12,
    )


def _mitigation_level(distal: float, proximal: float, level: str) -> float:
    if level == "Distal":
        return distal
    if level == "50 % OB":
        return (distal + proximal) / 2.0
    return proximal


def _ob_drawing(
    df: pd.DataFrame,
    *,
    direction: str,
    trigger: pd.Series,
    distal_series: pd.Series,
    proximal_series: pd.Series,
    index_series: pd.Series,
    ob_valid_global: pd.Series,
    validity: int,
    mitigation_level: str,
    mitigation_level_bb: str,
    show_all: bool,
    show_all_bb: bool,
    show: bool,
    show_bb: bool,
) -> OBDrawingOutput:
    alert = pd.Series(False, index=df.index)
    proximal_out = pd.Series(np.nan, index=df.index)
    distal_out = pd.Series(np.nan, index=df.index)
    index_out = pd.Series(np.nan, index=df.index)
    alert_bb = pd.Series(False, index=df.index)
    proximal_bb = pd.Series(np.nan, index=df.index)
    distal_bb = pd.Series(np.nan, index=df.index)
    index_bb = pd.Series(np.nan, index=df.index)

    check = True
    check_bb = False
    distal_price = np.nan
    proximal_price = np.nan
    idx_price = np.nan

    distal_price_bb = np.nan
    proximal_price_bb = np.nan
    idx_price_bb = np.nan

    cbb_history: List[bool] = []

    for i in range(len(df)):
        prev_check = check
        prev_check_bb = check_bb

        if trigger.iloc[i]:
            distal_price = distal_series.iloc[i]
            proximal_price = proximal_series.iloc[i]
            idx_price = index_series.iloc[i]
            check = True

        if check and not pd.isna(idx_price):
            if (i - idx_price) >= validity:
                check = False
            else:
                ml = _mitigation_level(distal_price, proximal_price, mitigation_level)
                if direction == "Demand":
                    if df["low"].iloc[i] < ml:
                        check = False
                else:
                    if df["high"].iloc[i] > ml:
                        check = False

        cbb = (prev_check is True) and (check is False)
        cbb_history.append(cbb)
        if len(cbb_history) > 4:
            cbb_history.pop(0)

        if any(cbb_history) and not prev_check_bb and not check_bb:
            if direction == "Demand" and df["close"].iloc[i] < distal_price:
                idx_price_bb = i
                check_bb = True
                distal_price_bb = proximal_price
                proximal_price_bb = distal_price
            if direction == "Supply" and df["close"].iloc[i] > distal_price:
                idx_price_bb = i
                check_bb = True
                distal_price_bb = proximal_price
                proximal_price_bb = distal_price

        if check_bb:
            ml_bb = _mitigation_level(distal_price_bb, proximal_price_bb, mitigation_level_bb)
            if direction == "Demand":
                if df["high"].iloc[i] > ml_bb or (not pd.isna(idx_price_bb) and (i - idx_price_bb) >= validity) or cbb:
                    check_bb = False
            else:
                if df["low"].iloc[i] < ml_bb or (not pd.isna(idx_price_bb) and (i - idx_price_bb) >= validity) or cbb:
                    check_bb = False

        if prev_check and not check:
            alert.iloc[i] = True

        if prev_check_bb and not check_bb and not alert.iloc[i]:
            alert_bb.iloc[i] = True

        if check:
            proximal_out.iloc[i] = proximal_price
            distal_out.iloc[i] = distal_price
            index_out.iloc[i] = idx_price

        if check_bb:
            proximal_bb.iloc[i] = proximal_price_bb
            distal_bb.iloc[i] = distal_price_bb
            index_bb.iloc[i] = idx_price_bb

    return OBDrawingOutput(
        alert=alert,
        proximal=proximal_out,
        distal=distal_out,
        index=index_out,
        alert_bb=alert_bb,
        proximal_bb=proximal_bb,
        distal_bb=distal_bb,
        index_bb=index_bb,
    )


def _cisd_level_detector(
    df: pd.DataFrame,
    *,
    cond_high: pd.Series,
    cond_low: pd.Series,
    trading_range: pd.Series,
    fvg_detection: FVGDetection,
    bar_back_check: int,
    cisd_valid: int,
) -> CISDOutputs:
    body = df["close"] - df["open"]

    bull_trigger = pd.Series(False, index=df.index)
    bear_trigger = pd.Series(False, index=df.index)
    fvg_bull_trigger = pd.Series(False, index=df.index)
    fvg_bear_trigger = pd.Series(False, index=df.index)
    bull_fvg_bar = pd.Series(0, index=df.index)
    bear_fvg_bar = pd.Series(0, index=df.index)
    bull_fvg_distal = pd.Series(0.0, index=df.index)
    bear_fvg_distal = pd.Series(0.0, index=df.index)
    bull_fvg_proximal = pd.Series(0.0, index=df.index)
    bear_fvg_proximal = pd.Series(0.0, index=df.index)
    bull_ob_index = pd.Series(np.nan, index=df.index)
    bear_ob_index = pd.Series(np.nan, index=df.index)
    cisd_high_level = pd.Series(np.nan, index=df.index)
    cisd_low_level = pd.Series(np.nan, index=df.index)
    cisd_high_index = pd.Series(np.nan, index=df.index)
    cisd_low_index = pd.Series(np.nan, index=df.index)

    permit_h_set = True
    permit_l_set = True
    permit_h_reset = True
    permit_l_reset = True
    prev_permit_h_reset = True
    prev_permit_l_reset = True

    fvg_bear_d: list[float] = []
    fvg_bear_p: list[float] = []
    fvg_bear_i: list[int] = []

    fvg_bull_d: list[float] = []
    fvg_bull_p: list[float] = []
    fvg_bull_i: list[int] = []

    high_ob = None
    low_ob = None
    bear_i = None
    bull_i = None

    bear_fvg_idx = 0
    bull_fvg_idx = 0
    bear_fvg_d = 0.0
    bear_fvg_p = 0.0
    bull_fvg_d = 0.0
    bull_fvg_p = 0.0

    current_cisd_high = np.nan
    current_cisd_low = np.nan
    current_cisd_high_idx = np.nan
    current_cisd_low_idx = np.nan

    for i in range(len(df)):
        if i >= 2 and trading_range.iloc[i - 1] == 0 and trading_range.iloc[i - 2] == 1:
            bear_fvg_idx = 0
            bear_fvg_d = 0.0
            bear_fvg_p = 0.0
            bull_fvg_idx = 0
            bull_fvg_d = 0.0
            bull_fvg_p = 0.0
            high_ob = None
            low_ob = None
            bear_i = None
            bull_i = None
            fvg_bear_d.clear()
            fvg_bear_p.clear()
            fvg_bear_i.clear()
            fvg_bull_d.clear()
            fvg_bull_p.clear()
            fvg_bull_i.clear()
            current_cisd_high = np.nan
            current_cisd_low = np.nan
            current_cisd_high_idx = np.nan
            current_cisd_low_idx = np.nan

        if i > 0 and trading_range.iloc[i - 1] == 0 and trading_range.iloc[i] == 1:
            high_ob = df["high"].iloc[i]
            bear_i = i
            low_ob = df["low"].iloc[i]
            bull_i = i

        if i > 0 and trading_range.iloc[i - 1] == 1:
            if high_ob is not None and df["high"].iloc[i] > high_ob:
                high_ob = df["high"].iloc[i]
                bear_i = i
            if low_ob is not None and df["low"].iloc[i] < low_ob:
                low_ob = df["low"].iloc[i]
                bull_i = i

        if cond_high.iloc[i]:
            permit_h_set = True
            for j in range(1, bar_back_check + 1):
                idx = i - j
                if idx < 0 or not permit_h_set:
                    continue
                if body.iloc[idx] < 0:
                    permit_h_reset = True
                    if bar_back_check > 1 and j > 1:
                        open_1 = df["open"].iloc[i - j + 1]
                        open_2 = df["open"].iloc[i - j + 2]
                        current_cisd_high = min(open_1, open_2)
                        current_cisd_high_idx = i - j + (2 if open_2 < open_1 else 1)
                    else:
                        current_cisd_high = df["open"].iloc[i]
                        current_cisd_high_idx = i
                        open_1 = df["open"].iloc[i - j - 1]
                        open_2 = df["open"].iloc[i - j - 2]
                        current_cisd_high = min(open_1, open_2)
                        current_cisd_high_idx = i - j - 2 if open_2 == current_cisd_high else i - j - 1
                    else:
                        current_cisd_high = df["open"].iloc[i - j - 1]
                        current_cisd_high_idx = i - j - 1
                    permit_h_set = False

        if cond_low.iloc[i]:
            permit_l_set = True
            for j in range(1, bar_back_check + 1):
                idx = i - j
                if idx < 0 or not permit_l_set:
                    continue
                if body.iloc[idx] > 0:
                    permit_l_reset = True
                    if bar_back_check > 1 and j > 1:
                        open_1 = df["open"].iloc[i - j + 1]
                        open_2 = df["open"].iloc[i - j + 2]
                        current_cisd_low = max(open_1, open_2)
                        current_cisd_low_idx = i - j + (2 if open_2 > open_1 else 1)
                    else:
                        current_cisd_low = df["open"].iloc[i]
                        current_cisd_low_idx = i
                        open_1 = df["open"].iloc[i - j - 1]
                        open_2 = df["open"].iloc[i - j - 2]
                        current_cisd_low = max(open_1, open_2)
                        current_cisd_low_idx = i - j - 2 if open_2 == current_cisd_low else i - j - 1
                    else:
                        current_cisd_low = df["open"].iloc[i - j - 1]
                        current_cisd_low_idx = i - j - 1
                    permit_l_set = False

        cisd_high_level.iloc[i] = current_cisd_high
        cisd_low_level.iloc[i] = current_cisd_low
        cisd_high_index.iloc[i] = current_cisd_high_idx
        cisd_low_index.iloc[i] = current_cisd_low_idx

        if permit_h_reset and trading_range.iloc[i] == 1:
            if fvg_detection.supply_condition.iloc[i]:
                fvg_bear_i.append(int(fvg_detection.supply_bar.iloc[i]))
                fvg_bear_d.append(float(fvg_detection.supply_distal.iloc[i]))
                fvg_bear_p.append(float(fvg_detection.supply_proximal.iloc[i]))
            if fvg_bear_i:
                if df["high"].iloc[i] > fvg_bear_p[-1]:
                    fvg_bear_i.pop()
                    fvg_bear_d.pop()
                    fvg_bear_p.pop()

            if not pd.isna(current_cisd_high) and i - current_cisd_high_idx <= cisd_valid:
                if df["close"].iloc[i] <= current_cisd_high:
                    permit_h_reset = False

        if permit_l_reset and trading_range.iloc[i] == 1:
            if fvg_detection.demand_condition.iloc[i]:
                fvg_bull_i.append(int(fvg_detection.demand_bar.iloc[i]))
                fvg_bull_d.append(float(fvg_detection.demand_distal.iloc[i]))
                fvg_bull_p.append(float(fvg_detection.demand_proximal.iloc[i]))
            if fvg_bull_i:
                if df["low"].iloc[i] < fvg_bull_p[-1]:
                    fvg_bull_i.pop()
                    fvg_bull_d.pop()
                    fvg_bull_p.pop()

            if not pd.isna(current_cisd_low) and i - current_cisd_low_idx <= cisd_valid:
                if df["close"].iloc[i] >= current_cisd_low:
                    permit_l_reset = False

        if trading_range.iloc[i] == 1:
            if fvg_bull_i:
                bull_fvg_idx = fvg_bull_i[-1]
                bull_fvg_d = fvg_bull_d[-1]
                bull_fvg_p = fvg_bull_p[-1]
            if fvg_bear_i:
                bear_fvg_idx = fvg_bear_i[-1]
                bear_fvg_d = fvg_bear_d[-1]
                bear_fvg_p = fvg_bear_p[-1]

        if prev_permit_h_reset and not permit_h_reset:
            bear_trigger.iloc[i] = True
            if fvg_bear_i and bear_fvg_idx != 0:
                fvg_bear_trigger.iloc[i] = True
        if prev_permit_l_reset and not permit_l_reset:
            bull_trigger.iloc[i] = True
            if fvg_bull_i and bull_fvg_idx != 0:
                fvg_bull_trigger.iloc[i] = True

        prev_permit_h_reset = permit_h_reset
        prev_permit_l_reset = permit_l_reset

        if i >= 2 and (trading_range.iloc[i - 1] == 0 and trading_range.iloc[i - 2] == 1):
            bear_fvg_idx = 0
            bear_fvg_d = 0.0
            bear_fvg_p = 0.0
            bull_fvg_idx = 0
            bull_fvg_d = 0.0
            bull_fvg_p = 0.0
            high_ob = None
            low_ob = None
            bear_i = None
            bull_i = None
            fvg_bear_d.clear()
            fvg_bear_p.clear()
            fvg_bear_i.clear()
            fvg_bull_d.clear()
            fvg_bull_p.clear()
            fvg_bull_i.clear()
            current_cisd_high = np.nan
            current_cisd_low = np.nan
            current_cisd_high_idx = np.nan
            current_cisd_low_idx = np.nan

        bull_fvg_bar.iloc[i] = bull_fvg_idx
        bear_fvg_bar.iloc[i] = bear_fvg_idx
        bull_fvg_distal.iloc[i] = bull_fvg_d
        bear_fvg_distal.iloc[i] = bear_fvg_d
        bull_fvg_proximal.iloc[i] = bull_fvg_p
        bear_fvg_proximal.iloc[i] = bear_fvg_p
        bull_ob_index.iloc[i] = bull_i if bull_i is not None else np.nan
        bear_ob_index.iloc[i] = bear_i if bear_i is not None else np.nan

    return CISDOutputs(
        bull_trigger=bull_trigger,
        bear_trigger=bear_trigger,
        fvg_bull_trigger=fvg_bull_trigger,
        fvg_bear_trigger=fvg_bear_trigger,
        bull_fvg_bar=bull_fvg_bar,
        bear_fvg_bar=bear_fvg_bar,
        bull_fvg_distal=bull_fvg_distal,
        bear_fvg_distal=bear_fvg_distal,
        bull_fvg_proximal=bull_fvg_proximal,
        bear_fvg_proximal=bear_fvg_proximal,
        bull_ob_index=bull_ob_index,
        bear_ob_index=bear_ob_index,
        cisd_high_level=cisd_high_level,
        cisd_low_level=cisd_low_level,
        cisd_high_index=cisd_high_index,
        cisd_low_index=cisd_low_index,
    )


def calculate_tradingfinder_silver_bullet(
    df: pd.DataFrame,
    *,
    fvg_filter: bool = False,
    fvg_filter_type: str = "Defensive",
    bar_back_check: int = 120,
    cisd_valid: int = 90,
    ob_valid: int = 60,
    fvg_valid: int = 60,
    refine: bool = True,
    refine_mode: str = "Defensive",
    mitigation_level_ob: str = "Proximal",
    mitigation_level_fvg: str = "Proximal",
    ny_or_session: str = "0900-1000",
    ny_trading_session: str = "1000-1100",
) -> Dict[str, object]:
    """Run the TradingFinder Silver Bullet calculation."""
    or_range = _session_mask(df.index, ny_or_session, "America/New_York")
    trading_range = _session_mask(df.index, ny_trading_session, "America/New_York")

    session_levels = _low_high_session_detector(df, or_range)
    session_levels.trading_range = trading_range

    high_break = pd.Series(False, index=df.index)
    low_break = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if trading_range.iloc[i] == 0:
            high_break.iloc[i] = False
            low_break.iloc[i] = False
            continue

        if i > 0 and high_break.iloc[i - 1]:
            high_break.iloc[i] = True
        elif df["high"].iloc[i] > session_levels.or_high.iloc[i]:
            high_break.iloc[i] = True

        if i > 0 and low_break.iloc[i - 1]:
            low_break.iloc[i] = True
        elif df["low"].iloc[i] < session_levels.or_low.iloc[i]:
        if df["high"].iloc[i] > session_levels.or_high.iloc[i]:
            high_break.iloc[i] = True
        if df["low"].iloc[i] < session_levels.or_low.iloc[i]:
            low_break.iloc[i] = True

    session_levels.high_break = high_break
    session_levels.low_break = low_break

    cond_high = (trading_range == 1) & high_break & (~high_break.shift(1).fillna(False)) & (~low_break)
    cond_low = (trading_range == 1) & low_break & (~low_break.shift(1).fillna(False)) & (~high_break)

    fvg_detection = _fvg_detector(df, fvg_filter, fvg_filter_type)

    cisd_outputs = _cisd_level_detector(
        df,
        cond_high=cond_high,
        cond_low=cond_low,
        trading_range=trading_range,
        fvg_detection=fvg_detection,
        bar_back_check=bar_back_check,
        cisd_valid=cisd_valid,
    )

    demand_refined = _refine_ob(
        df,
        direction="Demand",
        refine_on=refine,
        refine_mode=refine_mode,
        trigger=cisd_outputs.bull_trigger,
        ob_index=cisd_outputs.bull_ob_index,
    )
    supply_refined = _refine_ob(
        df,
        direction="Supply",
        refine_on=refine,
        refine_mode=refine_mode,
        trigger=cisd_outputs.bear_trigger,
        ob_index=cisd_outputs.bear_ob_index,
    )

    demand_ob = _ob_drawing(
        df,
        direction="Demand",
        trigger=cisd_outputs.bull_trigger,
        distal_series=demand_refined.yd12,
        proximal_series=demand_refined.yp12,
        index_series=demand_refined.xd1,
        ob_valid_global=trading_range == 1,
        validity=ob_valid,
        mitigation_level=mitigation_level_ob,
        mitigation_level_bb=mitigation_level_ob,
        show_all=True,
        show_all_bb=False,
        show=True,
        show_bb=False,
    )
    supply_ob = _ob_drawing(
        df,
        direction="Supply",
        trigger=cisd_outputs.bear_trigger,
        distal_series=supply_refined.yd12,
        proximal_series=supply_refined.yp12,
        index_series=supply_refined.xd1,
        ob_valid_global=trading_range == 1,
        validity=ob_valid,
        mitigation_level=mitigation_level_ob,
        mitigation_level_bb=mitigation_level_ob,
        show_all=True,
        show_all_bb=False,
        show=True,
        show_bb=False,
    )

    demand_fvg = _ob_drawing(
        df,
        direction="Demand",
        trigger=cisd_outputs.fvg_bull_trigger,
        distal_series=cisd_outputs.bull_fvg_distal,
        proximal_series=cisd_outputs.bull_fvg_proximal,
        index_series=cisd_outputs.bull_fvg_bar,
        ob_valid_global=trading_range == 1,
        validity=fvg_valid,
        mitigation_level=mitigation_level_fvg,
        mitigation_level_bb=mitigation_level_fvg,
        show_all=True,
        show_all_bb=False,
        show=True,
        show_bb=False,
    )
    supply_fvg = _ob_drawing(
        df,
        direction="Supply",
        trigger=cisd_outputs.fvg_bear_trigger,
        distal_series=cisd_outputs.bear_fvg_distal,
        proximal_series=cisd_outputs.bear_fvg_proximal,
        index_series=cisd_outputs.bear_fvg_bar,
        ob_valid_global=trading_range == 1,
        validity=fvg_valid,
        mitigation_level=mitigation_level_fvg,
        mitigation_level_bb=mitigation_level_fvg,
        show_all=True,
        show_all_bb=False,
        show=True,
        show_bb=False,
    )

    return {
        "session_levels": session_levels,
        "fvg_detection": fvg_detection,
        "cisd": cisd_outputs,
        "demand_refined": demand_refined,
        "supply_refined": supply_refined,
        "demand_ob": demand_ob,
        "supply_ob": supply_ob,
        "demand_fvg": demand_fvg,
        "supply_fvg": supply_fvg,
    }


if __name__ == "__main__":
    data = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    data["datetime"] = pd.to_datetime(data["time"])
    data = data.set_index("datetime").sort_index()

    results = calculate_tradingfinder_silver_bullet(
        data,
        fvg_filter=False,
        fvg_filter_type="Defensive",
        bar_back_check=120,
        cisd_valid=90,
    )

    print("Results summary:")
    print(results["session_levels"].high_break.tail())
