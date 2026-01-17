"""One Trading Setup for Life ICT [TradingFinder] Sweep Session FVG.

Python translation of the TradingFinder Pine Script by TFlab.

This module mirrors the Pine logic by:
- Tracking NY PM session highs/lows and NY AM session breaks.
- Detecting Fair Value Gaps (FVG) with optional filtering.
- Building CISD levels and emitting OB/FVG trigger signals.
- Refining order blocks and tracking mitigation alerts.
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


@dataclass
class AlertEvent:
    index: int
    direction: str
    category: str
    message: str


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
        prior_session = 0 if i == 0 else on_session.iloc[i - 1]
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

    for i in range(len(df)):
        if trigger.iloc[i]:
            idx = ob_index.iloc[i]
            if not pd.isna(idx):
                idx = int(idx)
            else:
                idx = i
            if 0 <= idx < len(df):
                open_i = float(df["open"].iloc[idx])
                close_i = float(df["close"].iloc[idx])
                high_i = float(df["high"].iloc[idx])
                low_i = float(df["low"].iloc[idx])

                if direction == "Demand":
                    distal = low_i
                    if refine_on:
                        proximal = max(open_i, close_i) if refine_mode == "Defensive" else high_i
                    else:
                        proximal = high_i
                else:
                    distal = high_i
                    if refine_on:
                        proximal = min(open_i, close_i) if refine_mode == "Defensive" else low_i
                    else:
                        proximal = low_i

                current_index = idx
                current_distal = distal
                current_proximal = proximal

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
    session_on: pd.Series,
    validity: int,
    mitigation_level: str,
) -> OBDrawingOutput:
    alert = pd.Series(False, index=df.index)
    proximal_out = pd.Series(np.nan, index=df.index)
    distal_out = pd.Series(np.nan, index=df.index)
    index_out = pd.Series(np.nan, index=df.index)

    active_zone: Optional[Dict[str, float]] = None

    for i in range(len(df)):
        if trigger.iloc[i]:
            distal = distal_series.iloc[i]
            proximal = proximal_series.iloc[i]
            index_val = index_series.iloc[i]
            if not pd.isna(distal) and not pd.isna(proximal) and not pd.isna(index_val):
                active_zone = {
                    "distal": float(distal),
                    "proximal": float(proximal),
                    "index": int(index_val),
                }

        if active_zone is not None:
            zone_age = i - active_zone["index"]
            if zone_age > validity:
                active_zone = None
            else:
                distal = active_zone["distal"]
                proximal = active_zone["proximal"]
                level = _mitigation_level(distal, proximal, mitigation_level)
                high_i = float(df["high"].iloc[i])
                low_i = float(df["low"].iloc[i])
                if session_on.iloc[i] == 1 and low_i <= level <= high_i:
                    alert.iloc[i] = True
                    active_zone = None

        if active_zone is not None:
            proximal_out.iloc[i] = active_zone["proximal"]
            distal_out.iloc[i] = active_zone["distal"]
            index_out.iloc[i] = active_zone["index"]

    return OBDrawingOutput(alert=alert, proximal=proximal_out, distal=distal_out, index=index_out)


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

        if i >= 2 and (am_range.iloc[i - 1] == 0 and am_range.iloc[i - 2] == 1):
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
            bull_fvg_bar.iloc[i] = 0
            bear_fvg_bar.iloc[i] = 0
            bull_fvg_distal.iloc[i] = 0.0
            bear_fvg_distal.iloc[i] = 0.0
            bull_fvg_proximal.iloc[i] = 0.0
            bear_fvg_proximal.iloc[i] = 0.0

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


def _alert_enabled(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() == "on"


def calculate_one_trading_setup(
    df: pd.DataFrame,
    *,
    bar_back_check: int = 120,
    cisd_valid: int = 90,
    ob_valid: int = 60,
    fvg_valid: int = 60,
    refine: bool = True,
    refine_mode: str = "Defensive",
    mitigation_level_ob: str = "Proximal",
    mitigation_level_fvg: str = "Proximal",
    pfvg_filter: bool = False,
    pfvg_filter_type: str = "Defensive",
    ny_pm_session: str = "1330-1600",
    ny_am_session: str = "0930-1600",
    alert_name: str = "One Trading Setup ICT [TradingFinder]",
    alert_ob_bull: str | bool = "On",
    alert_ob_bear: str | bool = "On",
    alert_fvg_bull: str | bool = "On",
    alert_fvg_bear: str | bool = "On",
) -> Dict[str, object]:
    """Run the One Trading Setup for Life ICT calculation."""
    pm_range = _session_mask(df.index, ny_pm_session, "America/New_York")
    am_range = _session_mask(df.index, ny_am_session, "America/New_York")

    session_levels = _low_high_session_detector(df, pm_range)
    session_levels.pm_range = pm_range
    session_levels.am_range = am_range

    high_break = pd.Series(False, index=df.index)
    low_break = pd.Series(False, index=df.index)

    high_break_flag = False
    low_break_flag = False

    for i in range(len(df)):
        if am_range.iloc[i] == 0:
            high_break_flag = False
            low_break_flag = False
        else:
            if i > 0 and df["high"].iloc[i] > session_levels.pm_high.iloc[i] and df["high"].iloc[i - 1] < session_levels.pm_high.iloc[i]:
                high_break_flag = True
            if i > 0 and df["low"].iloc[i] < session_levels.pm_low.iloc[i] and df["low"].iloc[i - 1] > session_levels.pm_low.iloc[i]:
                low_break_flag = True

        high_break.iloc[i] = high_break_flag
        low_break.iloc[i] = low_break_flag

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
        session_on=am_range,
        validity=ob_valid,
        mitigation_level=mitigation_level_ob,
    )
    supply_ob = _ob_drawing(
        df,
        direction="Supply",
        trigger=cisd_outputs.bear_trigger,
        distal_series=supply_refined.yd12,
        proximal_series=supply_refined.yp12,
        index_series=supply_refined.xd1,
        session_on=am_range,
        validity=ob_valid,
        mitigation_level=mitigation_level_ob,
    )

    demand_fvg = _ob_drawing(
        df,
        direction="Demand",
        trigger=cisd_outputs.fvg_bull_trigger,
        distal_series=cisd_outputs.bull_fvg_distal,
        proximal_series=cisd_outputs.bull_fvg_proximal,
        index_series=cisd_outputs.bull_fvg_bar,
        session_on=am_range,
        validity=fvg_valid,
        mitigation_level=mitigation_level_fvg,
    )
    supply_fvg = _ob_drawing(
        df,
        direction="Supply",
        trigger=cisd_outputs.fvg_bear_trigger,
        distal_series=cisd_outputs.bear_fvg_distal,
        proximal_series=cisd_outputs.bear_fvg_proximal,
        index_series=cisd_outputs.bear_fvg_bar,
        session_on=am_range,
        validity=fvg_valid,
        mitigation_level=mitigation_level_fvg,
    )

    alert_events: List[AlertEvent] = []
    for i in range(len(df)):
        if demand_ob.alert.iloc[i] and _alert_enabled(alert_ob_bull):
            alert_events.append(
                AlertEvent(
                    index=i,
                    direction="Bullish",
                    category="Order Block Signal",
                    message=f"{alert_name}: Alert Demand Mitigation in Son Model ICT Setup",
                )
            )
        if supply_ob.alert.iloc[i] and _alert_enabled(alert_ob_bear):
            alert_events.append(
                AlertEvent(
                    index=i,
                    direction="Bearish",
                    category="Order Block Signal",
                    message=f"{alert_name}: Alert Supply Mitigation in Son Model ICT Setup",
                )
            )
        if demand_fvg.alert.iloc[i] and _alert_enabled(alert_fvg_bull):
            alert_events.append(
                AlertEvent(
                    index=i,
                    direction="Bullish",
                    category="Order Block Signal",
                    message=f"{alert_name}: Alert Demand Mitigation in Son Model ICT Setup",
                )
            )
        if supply_fvg.alert.iloc[i] and _alert_enabled(alert_fvg_bear):
            alert_events.append(
                AlertEvent(
                    index=i,
                    direction="Bearish",
                    category="Order Block Signal",
                    message=f"{alert_name}: Alert Supply Mitigation in Son Model ICT Setup",
                )
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
        "alert_events": alert_events,
    }
