"""One Trading Setup for Life ICT [TradingFinder] Sweep Session FVG.

Python translation of the TradingFinder Pine Script by TFlab.

This module mirrors the Pine logic by:
- Tracking NY PM session highs/lows and NY AM session breaks.
- Detecting Fair Value Gaps (FVG) with optional filtering.
- Building CISD levels and emitting OB/FVG trigger signals.
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


@dataclass
class SessionLevels:
    pm_high: pd.Series
    pm_low: pd.Series
    pm_start_time: pd.Series
    am_range: pd.Series
    pm_range: pd.Series
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
    demand_bar = pd.Series(np.nan, index=df.index)
    supply_bar = pd.Series(np.nan, index=df.index)

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


def _low_high_session_detector(df: pd.DataFrame, on_session: pd.Series) -> SessionLevels:
    high_series = pd.Series(np.nan, index=df.index)
    low_series = pd.Series(np.nan, index=df.index)
    time_series = pd.Series(np.nan, index=df.index)

    current_high = 0.0
    current_low = 0.0
    current_time = np.nan

    for i in range(len(df)):
        if i == 0:
            prior_session = 0
        else:
            prior_session = on_session.iloc[i - 1]
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
        pm_high=high_series,
        pm_low=low_series,
        pm_start_time=time_series,
        am_range=pd.Series(np.nan, index=df.index),
        pm_range=pd.Series(np.nan, index=df.index),
        high_break=pd.Series(np.nan, index=df.index),
        low_break=pd.Series(np.nan, index=df.index),
    )


def _cisd_level_detector(
    df: pd.DataFrame,
    *,
    cond_high: pd.Series,
    cond_low: pd.Series,
    am_range: pd.Series,
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

    permit_h_set = True
    permit_l_set = True
    permit_h_reset = True
    permit_l_reset = True
    prev_permit_h_reset = False
    prev_permit_l_reset = False

    cisd_level_h = 0.0
    cisd_level_l = 0.0
    cisd_index_h = 0
    cisd_index_l = 0

    fvg_bear_i: List[int] = []
    fvg_bear_d: List[float] = []
    fvg_bear_p: List[float] = []

    fvg_bull_i: List[int] = []
    fvg_bull_d: List[float] = []
    fvg_bull_p: List[float] = []

    high_ob: Optional[float] = None
    low_ob: Optional[float] = None
    bear_i: Optional[int] = None
    bull_i: Optional[int] = None

    for i in range(len(df)):
        if am_range.iloc[i] == 1 and i > 0 and am_range.iloc[i - 1] == 0:
            high_ob = df["high"].iloc[i]
            low_ob = df["low"].iloc[i]
            bear_i = i
            bull_i = i

        if am_range.iloc[i] == 1 and i > 0 and am_range.iloc[i - 1] == 1:
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
                if idx < 0:
                    continue
                if body.iloc[idx] < 0 and permit_h_set:
                    permit_h_reset = True
                    if bar_back_check > 1 and j > 1:
                        cisd_level_h = min(df["open"].iloc[i - j - 1], df["open"].iloc[i - j - 2])
                        cisd_index_h = i - j - 2 if df["open"].iloc[i - j - 2] == cisd_level_h else i - j - 1
                    else:
                        cisd_level_h = df["open"].iloc[i - j - 1]
                        cisd_index_h = i - j - 1
                    permit_h_set = False
                    break

        if permit_h_reset and am_range.iloc[i] == 1:
            if fvg_detection.supply_condition.iloc[i]:
                fvg_bear_i.append(int(fvg_detection.supply_bar.iloc[i]))
                fvg_bear_d.append(float(fvg_detection.supply_distal.iloc[i]))
                fvg_bear_p.append(float(fvg_detection.supply_proximal.iloc[i]))
            if fvg_bear_i:
                if df["high"].iloc[i] > fvg_bear_p[-1]:
                    fvg_bear_i.pop()
                    fvg_bear_d.pop()
                    fvg_bear_p.pop()

            if df["close"].iloc[i] <= cisd_level_h and i - cisd_index_h <= cisd_valid:
                permit_h_reset = False

        if cond_low.iloc[i]:
            permit_l_set = True
            for j in range(1, bar_back_check + 1):
                idx = i - j
                if idx < 0:
                    continue
                if body.iloc[idx] > 0 and permit_l_set:
                    permit_l_reset = True
                    if bar_back_check > 1 and j > 1:
                        cisd_level_l = max(df["open"].iloc[i - j - 1], df["open"].iloc[i - j - 2])
                        cisd_index_l = i - j - 2 if df["open"].iloc[i - j - 2] == cisd_level_l else i - j - 1
                    else:
                        cisd_level_l = df["open"].iloc[i - j - 1]
                        cisd_index_l = i - j - 1
                    permit_l_set = False
                    break

        if permit_l_reset and am_range.iloc[i] == 1:
            if fvg_detection.demand_condition.iloc[i]:
                fvg_bull_i.append(int(fvg_detection.demand_bar.iloc[i]))
                fvg_bull_d.append(float(fvg_detection.demand_distal.iloc[i]))
                fvg_bull_p.append(float(fvg_detection.demand_proximal.iloc[i]))
            if fvg_bull_i:
                if df["low"].iloc[i] < fvg_bull_p[-1]:
                    fvg_bull_i.pop()
                    fvg_bull_d.pop()
                    fvg_bull_p.pop()

            if df["close"].iloc[i] >= cisd_level_l and i - cisd_index_l <= cisd_valid:
                permit_l_reset = False

        if am_range.iloc[i] == 1:
            if fvg_bull_i:
                bull_fvg_bar.iloc[i] = fvg_bull_i[-1]
                bull_fvg_distal.iloc[i] = fvg_bull_d[-1]
                bull_fvg_proximal.iloc[i] = fvg_bull_p[-1]
            if fvg_bear_i:
                bear_fvg_bar.iloc[i] = fvg_bear_i[-1]
                bear_fvg_distal.iloc[i] = fvg_bear_d[-1]
                bear_fvg_proximal.iloc[i] = fvg_bear_p[-1]

        if i > 0 and (am_range.iloc[i - 1] == 1 and am_range.iloc[i] == 0):
            fvg_bear_i.clear()
            fvg_bear_d.clear()
            fvg_bear_p.clear()
            fvg_bull_i.clear()
            fvg_bull_d.clear()
            fvg_bull_p.clear()
            high_ob = None
            low_ob = None
            bear_i = None
            bull_i = None

        prev_h_reset = prev_permit_h_reset
        prev_l_reset = prev_permit_l_reset

        if prev_h_reset and not permit_h_reset:
            bear_trigger.iloc[i] = True
            if fvg_bear_i and bear_fvg_bar.iloc[i] != 0:
                fvg_bear_trigger.iloc[i] = True
        if prev_l_reset and not permit_l_reset:
            bull_trigger.iloc[i] = True
            if fvg_bull_i and bull_fvg_bar.iloc[i] != 0:
                fvg_bull_trigger.iloc[i] = True

        prev_permit_h_reset = permit_h_reset
        prev_permit_l_reset = permit_l_reset

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
    )


def calculate_one_trading_setup(
    df: pd.DataFrame,
    *,
    bar_back_check: int = 5,
    cisd_valid: int = 90,
    pfvg_filter: bool = False,
    pfvg_filter_type: str = "Defensive",
    ny_pm_session: str = "1330-1600",
    ny_am_session: str = "0930-1600",
) -> Dict[str, object]:
    """Run the One Trading Setup for Life ICT calculation."""
    pm_range = _session_mask(df.index, ny_pm_session, "America/New_York")
    am_range = _session_mask(df.index, ny_am_session, "America/New_York")

    session_levels = _low_high_session_detector(df, pm_range)
    session_levels.pm_range = pm_range
    session_levels.am_range = am_range

    high_break = pd.Series(False, index=df.index)
    low_break = pd.Series(False, index=df.index)

    for i in range(len(df)):
        if am_range.iloc[i] == 0:
            high_break.iloc[i] = False
            low_break.iloc[i] = False
            continue
        if i > 0 and df["high"].iloc[i] > session_levels.pm_high.iloc[i] and df["high"].iloc[i - 1] < session_levels.pm_high.iloc[i]:
            high_break.iloc[i] = True
        if i > 0 and df["low"].iloc[i] < session_levels.pm_low.iloc[i] and df["low"].iloc[i - 1] > session_levels.pm_low.iloc[i]:
            low_break.iloc[i] = True

    session_levels.high_break = high_break
    session_levels.low_break = low_break

    cond_high = (am_range == 1) & high_break & (~high_break.shift(1).fillna(False)) & (~low_break)
    cond_low = (am_range == 1) & low_break & (~low_break.shift(1).fillna(False)) & (~high_break)

    fvg_detection = _fvg_detector(df, pfvg_filter, pfvg_filter_type)

    cisd_outputs = _cisd_level_detector(
        df,
        cond_high=cond_high,
        cond_low=cond_low,
        am_range=am_range,
        fvg_detection=fvg_detection,
        bar_back_check=bar_back_check,
        cisd_valid=cisd_valid,
    )

    return {
        "session_levels": session_levels,
        "fvg_detection": fvg_detection,
        "cisd": cisd_outputs,
    }
