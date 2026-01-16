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
class LiquidityLevels:
    static_high: pd.Series
    static_low: pd.Series
    dynamic_high: pd.Series
    dynamic_low: pd.Series


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
            demand_condition.iloc[i] = True
            demand_distal.iloc[i] = high_2
            demand_proximal.iloc[i] = low
            demand_bar.iloc[i] = i
        elif high < low_2:
            width = low_2 - high
            if filter_on and atr is not None:
                if pd.isna(atr.iloc[i]) or width < atr.iloc[i] * multiplier:
                    continue
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

    static_high_pivot = _pivot_high(df["high"], static_period)
    static_low_pivot = _pivot_low(df["low"], static_period)
    dynamic_high_pivot = _pivot_high(df["high"], dynamic_period)
    dynamic_low_pivot = _pivot_low(df["low"], dynamic_period)

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

        static_high.iloc[i] = last_static_high
        static_low.iloc[i] = last_static_low
        dynamic_high.iloc[i] = last_dynamic_high
        dynamic_low.iloc[i] = last_dynamic_low

    return LiquidityLevels(
        static_high=static_high,
        static_low=static_low,
        dynamic_high=dynamic_high,
        dynamic_low=dynamic_low,
    )


def calculate_smc_tradingfinder(
    df: pd.DataFrame,
    *,
    pivot_period: int = 5,
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
    }


if __name__ == "__main__":
    data = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    data["datetime"] = pd.to_datetime(data["time"])
    data = data.set_index("datetime").sort_index()

    results = calculate_smc_tradingfinder(data)
    print(results["structure"].external_trend.tail())
