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


def _atr(df: pd.DataFrame, length: int = 55) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(length).mean()


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

    return {
        "session_levels": session_levels,
        "fvg_detection": fvg_detection,
        "cisd": cisd_outputs,
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
