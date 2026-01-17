"""Smart Money Concept [TradingFinder] Major Minor OB + FVG (SMC).

Python translation focused on the Pine logic flow:
- Pivot-based zigzag for major/minor structure.
- BoS/ChoCh detection with external/internal trend tracking.
- Order block trigger indices for major ChoCh/BoS events.
- FVG detection with optional width filtering.
- Liquidity line detection from static/dynamic pivots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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
class LiquidityLine:
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    kind: str  # "static_high", "static_low", "dynamic_high", "dynamic_low"


@dataclass
class LiquidityLevels:
    static_high: pd.Series
    static_low: pd.Series
    dynamic_high: pd.Series
    dynamic_low: pd.Series
    lines: List[LiquidityLine]


@dataclass
class StructureOutputs:
    major_high_level: pd.Series
    major_low_level: pd.Series
    major_high_index: pd.Series
    major_low_index: pd.Series
    minor_high_level: pd.Series
    minor_low_level: pd.Series
    minor_high_index: pd.Series
    minor_low_index: pd.Series
    external_trend: pd.Series
    internal_trend: pd.Series
    bullish_major_bos: pd.Series
    bearish_major_bos: pd.Series
    bullish_major_choch: pd.Series
    bearish_major_choch: pd.Series
    bullish_minor_bos: pd.Series
    bearish_minor_bos: pd.Series
    bullish_minor_choch: pd.Series
    bearish_minor_choch: pd.Series
    bu_mch_main_trigger: pd.Series
    bu_mch_sub_trigger: pd.Series
    bu_mbos_trigger: pd.Series
    be_mch_main_trigger: pd.Series
    be_mch_sub_trigger: pd.Series
    be_mbos_trigger: pd.Series
    bu_mch_main_index: pd.Series
    bu_mch_sub_index: pd.Series
    bu_mbos_index: pd.Series
    be_mch_main_index: pd.Series
    be_mch_sub_index: pd.Series
    be_mbos_index: pd.Series


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


@dataclass
class AlertEvent:
    index: int
    alert_type: str
    detection_type: str
    message: str


def _pivot_high(series: pd.Series, length: int) -> pd.Series:
    pivots = pd.Series(False, index=series.index)
    for i in range(length, len(series) - length):
        window = series.iloc[i - length : i + length + 1]
        if series.iloc[i] == window.max():
            pivots.iloc[i] = True
    return pivots


def _pivot_low(series: pd.Series, length: int) -> pd.Series:
    pivots = pd.Series(False, index=series.index)
    for i in range(length, len(series) - length):
        window = series.iloc[i - length : i + length + 1]
        if series.iloc[i] == window.min():
            pivots.iloc[i] = True
    return pivots


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
            else:  # Very Defensive
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


def _liquidity_levels(
    df: pd.DataFrame,
    static_period: int,
    dynamic_period: int,
    static_sensitivity: float,
    dynamic_sensitivity: float,
) -> LiquidityLevels:
    static_high = pd.Series(np.nan, index=df.index)
    static_low = pd.Series(np.nan, index=df.index)
    dynamic_high = pd.Series(np.nan, index=df.index)
    dynamic_low = pd.Series(np.nan, index=df.index)
    lines: List[LiquidityLine] = []

    atr = _atr(df, length=55)

    static_high_pivot = _pivot_high(df["high"], static_period)
    static_low_pivot = _pivot_low(df["low"], static_period)
    dynamic_high_pivot = _pivot_high(df["high"], dynamic_period)
    dynamic_low_pivot = _pivot_low(df["low"], dynamic_period)

    def _update_valuewhen(pivot_series: pd.Series, value_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        current = pd.Series(np.nan, index=df.index)
        previous = pd.Series(np.nan, index=df.index)
        last_val = np.nan
        last_prev = np.nan
        for i in range(len(df)):
            if pivot_series.iloc[i]:
                last_prev = last_val
                last_val = value_series.iloc[i]
            current.iloc[i] = last_val
            previous.iloc[i] = last_prev
        return current, previous

    def _update_indexwhen(pivot_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        current = pd.Series(np.nan, index=df.index)
        previous = pd.Series(np.nan, index=df.index)
        last_idx = np.nan
        prev_idx = np.nan
        for i in range(len(df)):
            if pivot_series.iloc[i]:
                prev_idx = last_idx
                last_idx = i
            current.iloc[i] = last_idx
            previous.iloc[i] = prev_idx
        return current, previous

    stsh_vc, stsh_vp = _update_valuewhen(static_high_pivot, df["high"])
    stsl_vc, stsl_vp = _update_valuewhen(static_low_pivot, df["low"])
    stsh_ic, stsh_ip = _update_indexwhen(static_high_pivot)
    stsl_ic, stsl_ip = _update_indexwhen(static_low_pivot)

    dtsh_vc, dtsh_vp = _update_valuewhen(dynamic_high_pivot, df["high"])
    dtsl_vc, dtsl_vp = _update_valuewhen(dynamic_low_pivot, df["low"])
    dtsh_ic, dtsh_ip = _update_indexwhen(dynamic_high_pivot)
    dtsl_ic, dtsl_ip = _update_indexwhen(dynamic_low_pivot)

    last_static_high = np.nan
    last_static_low = np.nan
    last_dynamic_high = np.nan
    last_dynamic_low = np.nan

    for i in range(len(df)):
        if static_high_pivot.iloc[i]:
            level = df["high"].iloc[i]
            if np.isnan(last_static_high) or abs(level - last_static_high) / level >= static_sensitivity:
                last_static_high = level
        if static_low_pivot.iloc[i]:
            level = df["low"].iloc[i]
            if np.isnan(last_static_low) or abs(level - last_static_low) / level >= static_sensitivity:
                last_static_low = level
        if dynamic_high_pivot.iloc[i]:
            level = df["high"].iloc[i]
            if np.isnan(last_dynamic_high) or abs(level - last_dynamic_high) / level >= dynamic_sensitivity:
                last_dynamic_high = level
        if dynamic_low_pivot.iloc[i]:
            level = df["low"].iloc[i]
            if np.isnan(last_dynamic_low) or abs(level - last_dynamic_low) / level >= dynamic_sensitivity:
                last_dynamic_low = level
        if not np.isnan(stsh_ic.iloc[i]) and stsh_ic.iloc[i] != stsh_ic.shift(1).iloc[i]:
            if stsh_vc.iloc[i] <= stsh_vp.iloc[i] and (stsh_ic.iloc[i] - stsh_ip.iloc[i]) >= 8:
                atr_idx = max(i - static_period, 0)
                atr_val = atr.iloc[atr_idx]
                if atr_val and abs((stsh_vp.iloc[i] - stsh_vc.iloc[i]) / atr_val) <= static_sensitivity:
                    lines.append(
                        LiquidityLine(
                            start_index=int(stsh_ip.iloc[i]),
                            end_index=int(stsh_ic.iloc[i]),
                            start_price=float(stsh_vp.iloc[i]),
                            end_price=float(stsh_vp.iloc[i]),
                            kind="static_high",
                        )
                    )
                    last_static_high = stsh_vp.iloc[i]
        if not np.isnan(stsl_ic.iloc[i]) and stsl_ic.iloc[i] != stsl_ic.shift(1).iloc[i]:
            if stsl_vc.iloc[i] >= stsl_vp.iloc[i] and (stsl_ic.iloc[i] - stsl_ip.iloc[i]) >= 8:
                atr_idx = max(i - static_period, 0)
                atr_val = atr.iloc[atr_idx]
                if atr_val and abs((stsl_vp.iloc[i] - stsl_vc.iloc[i]) / atr_val) <= static_sensitivity:
                    lines.append(
                        LiquidityLine(
                            start_index=int(stsl_ip.iloc[i]),
                            end_index=int(stsl_ic.iloc[i]),
                            start_price=float(stsl_vp.iloc[i]),
                            end_price=float(stsl_vp.iloc[i]),
                            kind="static_low",
                        )
                    )
                    last_static_low = stsl_vp.iloc[i]

        if not np.isnan(dtsh_ic.iloc[i]) and dtsh_ic.iloc[i] != dtsh_ic.shift(1).iloc[i]:
            if dtsh_vc.iloc[i] <= dtsh_vp.iloc[i] and (dtsh_ic.iloc[i] - dtsh_ip.iloc[i]) >= 4:
                atr_idx = max(i - dynamic_period, 0)
                atr_val = atr.iloc[atr_idx]
                diff_ratio = abs((dtsh_vp.iloc[i] - dtsh_vc.iloc[i]) / atr_val) if atr_val else 0.0
                if dynamic_sensitivity <= diff_ratio <= 1.95:
                    lines.append(
                        LiquidityLine(
                            start_index=int(dtsh_ip.iloc[i]),
                            end_index=int(dtsh_ic.iloc[i]),
                            start_price=float(dtsh_vp.iloc[i]),
                            end_price=float(dtsh_vc.iloc[i]),
                            kind="dynamic_high",
                        )
                    )
                    last_dynamic_high = dtsh_vc.iloc[i]

        if not np.isnan(dtsl_ic.iloc[i]) and dtsl_ic.iloc[i] != dtsl_ic.shift(1).iloc[i]:
            if dtsl_vc.iloc[i] >= dtsl_vp.iloc[i] and (dtsl_ic.iloc[i] - dtsl_ip.iloc[i]) >= 4:
                atr_idx = max(i - dynamic_period, 0)
                atr_val = atr.iloc[atr_idx]
                diff_ratio = abs((dtsl_vp.iloc[i] - dtsl_vc.iloc[i]) / atr_val) if atr_val else 0.0
                if dynamic_sensitivity <= diff_ratio <= 1.95:
                    lines.append(
                        LiquidityLine(
                            start_index=int(dtsl_ip.iloc[i]),
                            end_index=int(dtsl_ic.iloc[i]),
                            start_price=float(dtsl_vp.iloc[i]),
                            end_price=float(dtsl_vc.iloc[i]),
                            kind="dynamic_low",
                        )
                    )
                    last_dynamic_low = dtsl_vc.iloc[i]

        static_high.iloc[i] = last_static_high
        static_low.iloc[i] = last_static_low
        dynamic_high.iloc[i] = last_dynamic_high
        dynamic_low.iloc[i] = last_dynamic_low

    return LiquidityLevels(
        static_high=static_high,
        static_low=static_low,
        dynamic_high=dynamic_high,
        dynamic_low=dynamic_low,
        lines=lines,
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


def _mitigation_level(distal: float, proximal: float, mode: str) -> float:
    if mode == "Distal":
        return distal
    if mode == "50 % OB":
        return (proximal + distal) / 2.0
    return proximal


def _ob_drawing(
    df: pd.DataFrame,
    *,
    direction: str,
    trigger: pd.Series,
    distal_series: pd.Series,
    proximal_series: pd.Series,
    index_series: pd.Series,
    validity: int,
    show: bool,
    mitigation_level: str = "Proximal",
    show_bb: bool = False,
    mitigation_level_bb: str = "Proximal",
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
    prev_check = False
    prev_check_bb = False
    distal_price = np.nan
    proximal_price = np.nan
    idx_price = np.nan

    distal_price_bb = np.nan
    proximal_price_bb = np.nan
    idx_price_bb = np.nan

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
        if cbb and not check_bb and show_bb:
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

        if check_bb and not pd.isna(idx_price_bb):
            if (i - idx_price_bb) >= validity:
                check_bb = False
            else:
                ml_bb = _mitigation_level(distal_price_bb, proximal_price_bb, mitigation_level_bb)
                if direction == "Demand":
                    if df["high"].iloc[i] > ml_bb:
                        check_bb = False
                else:
                    if df["low"].iloc[i] < ml_bb:
                        check_bb = False

        if prev_check and not check:
            alert.iloc[i] = True

        if prev_check_bb and not check_bb and not alert.iloc[i] and show_bb:
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


def calculate_smc_tradingfinder(
    df: pd.DataFrame,
    *,
    pivot_period: int = 5,
    ob_validity: int = 500,
    show_demand_main: bool = True,
    show_demand_sub: bool = True,
    show_demand_bos: bool = True,
    show_supply_main: bool = True,
    show_supply_sub: bool = True,
    show_supply_bos: bool = True,
    refine_demand_main: bool = True,
    refine_demand_sub: bool = True,
    refine_demand_bos: bool = True,
    refine_supply_main: bool = True,
    refine_supply_sub: bool = True,
    refine_supply_bos: bool = True,
    refine_mode_demand_main: str = "Defensive",
    refine_mode_demand_sub: str = "Defensive",
    refine_mode_demand_bos: str = "Defensive",
    refine_mode_supply_main: str = "Defensive",
    refine_mode_supply_sub: str = "Defensive",
    refine_mode_supply_bos: str = "Defensive",
    fvg_filter: bool = True,
    fvg_filter_type: str = "Very Defensive",
    static_pivot_period: int = 8,
    dynamic_pivot_period: int = 3,
    static_sensitivity: float = 0.30,
    dynamic_sensitivity: float = 1.00,
) -> Dict[str, object]:
    """Calculate Smart Money Concept structure, OB triggers, FVG, and liquidity outputs."""
    high_pivot = _pivot_high(df["high"], pivot_period)
    low_pivot = _pivot_low(df["low"], pivot_period)

    major_high_level = pd.Series(np.nan, index=df.index)
    major_low_level = pd.Series(np.nan, index=df.index)
    major_high_index = pd.Series(np.nan, index=df.index)
    major_low_index = pd.Series(np.nan, index=df.index)
    minor_high_level = pd.Series(np.nan, index=df.index)
    minor_low_level = pd.Series(np.nan, index=df.index)
    minor_high_index = pd.Series(np.nan, index=df.index)
    minor_low_index = pd.Series(np.nan, index=df.index)

    external_trend = pd.Series("No Trend", index=df.index)
    internal_trend = pd.Series("No Trend", index=df.index)

    bullish_major_bos = pd.Series(False, index=df.index)
    bearish_major_bos = pd.Series(False, index=df.index)
    bullish_major_choch = pd.Series(False, index=df.index)
    bearish_major_choch = pd.Series(False, index=df.index)
    bullish_minor_bos = pd.Series(False, index=df.index)
    bearish_minor_bos = pd.Series(False, index=df.index)
    bullish_minor_choch = pd.Series(False, index=df.index)
    bearish_minor_choch = pd.Series(False, index=df.index)

    bu_mch_main_trigger = pd.Series(False, index=df.index)
    bu_mch_sub_trigger = pd.Series(False, index=df.index)
    bu_mbos_trigger = pd.Series(False, index=df.index)
    be_mch_main_trigger = pd.Series(False, index=df.index)
    be_mch_sub_trigger = pd.Series(False, index=df.index)
    be_mbos_trigger = pd.Series(False, index=df.index)

    bu_mch_main_index = pd.Series(np.nan, index=df.index)
    bu_mch_sub_index = pd.Series(np.nan, index=df.index)
    bu_mbos_index = pd.Series(np.nan, index=df.index)
    be_mch_main_index = pd.Series(np.nan, index=df.index)
    be_mch_sub_index = pd.Series(np.nan, index=df.index)
    be_mbos_index = pd.Series(np.nan, index=df.index)

    last_major_high = np.nan
    last_major_low = np.nan
    last_major_high_idx = np.nan
    last_major_low_idx = np.nan
    last_minor_high = np.nan
    last_minor_low = np.nan
    last_minor_high_idx = np.nan
    last_minor_low_idx = np.nan

    last_pivot_type = None
    last_pivot_value = np.nan
    last_pivot_index = np.nan
    last_high_value = np.nan
    last_low_value = np.nan

    lock_break_major = np.nan
    lock_break_minor = np.nan

    for i in range(len(df)):
        if high_pivot.iloc[i]:
            if last_high_value is np.nan or df["high"].iloc[i] > last_high_value:
                pivot_type = "HH"
            else:
                pivot_type = "LH"
            last_high_value = df["high"].iloc[i]
            last_pivot_type = pivot_type
            last_pivot_value = df["high"].iloc[i]
            last_pivot_index = i
            last_minor_high = df["high"].iloc[i]
            last_minor_high_idx = i
            if np.isnan(last_major_high) or pivot_type == "HH":
                last_major_high = df["high"].iloc[i]
                last_major_high_idx = i

        if low_pivot.iloc[i]:
            if last_low_value is np.nan or df["low"].iloc[i] < last_low_value:
                pivot_type = "LL"
            else:
                pivot_type = "HL"
            last_low_value = df["low"].iloc[i]
            last_pivot_type = pivot_type
            last_pivot_value = df["low"].iloc[i]
            last_pivot_index = i
            last_minor_low = df["low"].iloc[i]
            last_minor_low_idx = i
            if np.isnan(last_major_low) or pivot_type == "LL":
                last_major_low = df["low"].iloc[i]
                last_major_low_idx = i

        major_high_level.iloc[i] = last_major_high
        major_low_level.iloc[i] = last_major_low
        major_high_index.iloc[i] = last_major_high_idx
        major_low_index.iloc[i] = last_major_low_idx
        minor_high_level.iloc[i] = last_minor_high
        minor_low_level.iloc[i] = last_minor_low
        minor_high_index.iloc[i] = last_minor_high_idx
        minor_low_index.iloc[i] = last_minor_low_idx

        if not np.isnan(last_major_high) and df["close"].iloc[i] > last_major_high and lock_break_major != last_major_high_idx:
            if external_trend.iloc[i - 1] in ("No Trend", "Up Trend") if i > 0 else True:
                bullish_major_bos.iloc[i] = True
                external_trend.iloc[i] = "Up Trend"
            else:
                bullish_major_choch.iloc[i] = True
                external_trend.iloc[i] = "Up Trend"
            lock_break_major = last_major_high_idx

        if not np.isnan(last_major_low) and df["close"].iloc[i] < last_major_low and lock_break_major != last_major_low_idx:
            if external_trend.iloc[i - 1] in ("No Trend", "Down Trend") if i > 0 else True:
                bearish_major_bos.iloc[i] = True
                external_trend.iloc[i] = "Down Trend"
            else:
                bearish_major_choch.iloc[i] = True
                external_trend.iloc[i] = "Down Trend"
            lock_break_major = last_major_low_idx

        if i > 0 and external_trend.iloc[i] == "No Trend":
            external_trend.iloc[i] = external_trend.iloc[i - 1]

        if not np.isnan(last_minor_high) and df["close"].iloc[i] > last_minor_high and lock_break_minor != last_minor_high_idx:
            if internal_trend.iloc[i - 1] in ("No Trend", "Up Trend") if i > 0 else True:
                bullish_minor_bos.iloc[i] = True
                internal_trend.iloc[i] = "Up Trend"
            else:
                bullish_minor_choch.iloc[i] = True
                internal_trend.iloc[i] = "Up Trend"
            lock_break_minor = last_minor_high_idx

        if not np.isnan(last_minor_low) and df["close"].iloc[i] < last_minor_low and lock_break_minor != last_minor_low_idx:
            if internal_trend.iloc[i - 1] in ("No Trend", "Down Trend") if i > 0 else True:
                bearish_minor_bos.iloc[i] = True
                internal_trend.iloc[i] = "Down Trend"
            else:
                bearish_minor_choch.iloc[i] = True
                internal_trend.iloc[i] = "Down Trend"
            lock_break_minor = last_minor_low_idx

        if i > 0 and internal_trend.iloc[i] == "No Trend":
            internal_trend.iloc[i] = internal_trend.iloc[i - 1]

        if bullish_major_choch.iloc[i]:
            bu_mch_main_trigger.iloc[i] = True
            bu_mch_main_index.iloc[i] = last_major_low_idx
            if not np.isnan(last_minor_low_idx) and last_minor_low_idx != last_major_low_idx:
                bu_mch_sub_trigger.iloc[i] = True
                bu_mch_sub_index.iloc[i] = last_minor_low_idx

        if bullish_major_bos.iloc[i]:
            bu_mbos_trigger.iloc[i] = True
            bu_mbos_index.iloc[i] = last_pivot_index

        if bearish_major_choch.iloc[i]:
            be_mch_main_trigger.iloc[i] = True
            be_mch_main_index.iloc[i] = last_major_high_idx
            if not np.isnan(last_minor_high_idx) and last_minor_high_idx != last_major_high_idx:
                be_mch_sub_trigger.iloc[i] = True
                be_mch_sub_index.iloc[i] = last_minor_high_idx

        if bearish_major_bos.iloc[i]:
            be_mbos_trigger.iloc[i] = True
            be_mbos_index.iloc[i] = last_pivot_index

    fvg_detection = _fvg_detector(df, fvg_filter, fvg_filter_type)
    liquidity = _liquidity_levels(
        df,
        static_period=static_pivot_period,
        dynamic_period=dynamic_pivot_period,
        static_sensitivity=static_sensitivity,
        dynamic_sensitivity=dynamic_sensitivity,
    )

    bu_mch_main_refined = _refine_ob(
        df,
        direction="Demand",
        refine_on=refine_demand_main,
        refine_mode=refine_mode_demand_main,
        trigger=bu_mch_main_trigger,
        ob_index=bu_mch_main_index,
    )
    bu_mch_sub_refined = _refine_ob(
        df,
        direction="Demand",
        refine_on=refine_demand_sub,
        refine_mode=refine_mode_demand_sub,
        trigger=bu_mch_sub_trigger,
        ob_index=bu_mch_sub_index,
    )
    bu_mbos_refined = _refine_ob(
        df,
        direction="Demand",
        refine_on=refine_demand_bos,
        refine_mode=refine_mode_demand_bos,
        trigger=bu_mbos_trigger,
        ob_index=bu_mbos_index,
    )
    be_mch_main_refined = _refine_ob(
        df,
        direction="Supply",
        refine_on=refine_supply_main,
        refine_mode=refine_mode_supply_main,
        trigger=be_mch_main_trigger,
        ob_index=be_mch_main_index,
    )
    be_mch_sub_refined = _refine_ob(
        df,
        direction="Supply",
        refine_on=refine_supply_sub,
        refine_mode=refine_mode_supply_sub,
        trigger=be_mch_sub_trigger,
        ob_index=be_mch_sub_index,
    )
    be_mbos_refined = _refine_ob(
        df,
        direction="Supply",
        refine_on=refine_supply_bos,
        refine_mode=refine_mode_supply_bos,
        trigger=be_mbos_trigger,
        ob_index=be_mbos_index,
    )

    ob_alerts = {
        "demand_main": _ob_drawing(
            df,
            direction="Demand",
            trigger=bu_mch_main_trigger,
            distal_series=bu_mch_main_refined.yd12,
            proximal_series=bu_mch_main_refined.yp12,
            index_series=bu_mch_main_refined.xd1,
            validity=ob_validity,
            show=show_demand_main,
            show_bb=False,
        ),
        "demand_sub": _ob_drawing(
            df,
            direction="Demand",
            trigger=bu_mch_sub_trigger,
            distal_series=bu_mch_sub_refined.yd12,
            proximal_series=bu_mch_sub_refined.yp12,
            index_series=bu_mch_sub_refined.xd1,
            validity=ob_validity,
            show=show_demand_sub,
            show_bb=False,
        ),
        "demand_bos": _ob_drawing(
            df,
            direction="Demand",
            trigger=bu_mbos_trigger,
            distal_series=bu_mbos_refined.yd12,
            proximal_series=bu_mbos_refined.yp12,
            index_series=bu_mbos_refined.xd1,
            validity=ob_validity,
            show=show_demand_bos,
            show_bb=False,
        ),
        "supply_main": _ob_drawing(
            df,
            direction="Supply",
            trigger=be_mch_main_trigger,
            distal_series=be_mch_main_refined.yd12,
            proximal_series=be_mch_main_refined.yp12,
            index_series=be_mch_main_refined.xd1,
            validity=ob_validity,
            show=show_supply_main,
            show_bb=False,
        ),
        "supply_sub": _ob_drawing(
            df,
            direction="Supply",
            trigger=be_mch_sub_trigger,
            distal_series=be_mch_sub_refined.yd12,
            proximal_series=be_mch_sub_refined.yp12,
            index_series=be_mch_sub_refined.xd1,
            validity=ob_validity,
            show=show_supply_sub,
            show_bb=False,
        ),
        "supply_bos": _ob_drawing(
            df,
            direction="Supply",
            trigger=be_mbos_trigger,
            distal_series=be_mbos_refined.yd12,
            proximal_series=be_mbos_refined.yp12,
            index_series=be_mbos_refined.xd1,
            validity=ob_validity,
            show=show_supply_bos,
            show_bb=False,
        ),
    }

    structure = StructureOutputs(
        major_high_level=major_high_level,
        major_low_level=major_low_level,
        major_high_index=major_high_index,
        major_low_index=major_low_index,
        minor_high_level=minor_high_level,
        minor_low_level=minor_low_level,
        minor_high_index=minor_high_index,
        minor_low_index=minor_low_index,
        external_trend=external_trend,
        internal_trend=internal_trend,
        bullish_major_bos=bullish_major_bos,
        bearish_major_bos=bearish_major_bos,
        bullish_major_choch=bullish_major_choch,
        bearish_major_choch=bearish_major_choch,
        bullish_minor_bos=bullish_minor_bos,
        bearish_minor_bos=bearish_minor_bos,
        bullish_minor_choch=bullish_minor_choch,
        bearish_minor_choch=bearish_minor_choch,
        bu_mch_main_trigger=bu_mch_main_trigger,
        bu_mch_sub_trigger=bu_mch_sub_trigger,
        bu_mbos_trigger=bu_mbos_trigger,
        be_mch_main_trigger=be_mch_main_trigger,
        be_mch_sub_trigger=be_mch_sub_trigger,
        be_mbos_trigger=be_mbos_trigger,
        bu_mch_main_index=bu_mch_main_index,
        bu_mch_sub_index=bu_mch_sub_index,
        bu_mbos_index=bu_mbos_index,
        be_mch_main_index=be_mch_main_index,
        be_mch_sub_index=be_mch_sub_index,
        be_mbos_index=be_mbos_index,
    )

    return {
        "structure": structure,
        "fvg": fvg_detection,
        "liquidity": liquidity,
        "ob_refined": {
            "bu_mch_main": bu_mch_main_refined,
            "bu_mch_sub": bu_mch_sub_refined,
            "bu_mbos": bu_mbos_refined,
            "be_mch_main": be_mch_main_refined,
            "be_mch_sub": be_mch_sub_refined,
            "be_mbos": be_mbos_refined,
        },
        "ob_alerts": ob_alerts,
    }


if __name__ == "__main__":
    data = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    data["datetime"] = pd.to_datetime(data["time"])
    data = data.set_index("datetime").sort_index()

    results = calculate_smc_tradingfinder(data)
    print(results["structure"].external_trend.tail())
