"""Monday Range (Lines) - Pine translation.

Mirrors the Pine Script logic:
- Stores Monday (opening day) OHLC per week using weekly/daily HTF data.
- Builds Monday high/low/open/close levels and custom range multipliers.
- Supports line extension modes (end of week, current bar, fixed daily bars).
- Tracks breakout and reclaim events after Monday completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class Monday:
    wk_start: pd.Timestamp
    wk_end: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    def price_range(self) -> float:
        return self.high - self.low


@dataclass
class RangeEvent:
    bar_time: pd.Timestamp
    price: float
    week_key: pd.Timestamp


@dataclass
class LevelConfig:
    enabled: bool
    text: str
    value: float
    color: str
    line_style: str
    line_width: int


@dataclass
class LevelOutput:
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    price: float
    text: str
    color: str
    line_style: str
    line_width: int
    week_key: pd.Timestamp


@dataclass
class MondayRangeOutputs:
    mondays: Dict[pd.Timestamp, Monday]
    monday_order: List[pd.Timestamp]
    levels: List[LevelOutput]
    high_breakouts: List[RangeEvent]
    low_breakouts: List[RangeEvent]
    high_reclaims: List[RangeEvent]
    low_reclaims: List[RangeEvent]
    alert_flags: Dict[str, pd.Series]


def _parse_timeframe_to_minutes(timeframe: str) -> Optional[int]:
    if timeframe.isdigit():
        return int(timeframe)
    suffix = timeframe[-1].lower()
    value = timeframe[:-1]
    if not value:
        return None
    if suffix == "h":
        return int(value) * 60
    if suffix == "d":
        return int(value) * 60 * 24
    if suffix == "w":
        return int(value) * 60 * 24 * 7
    if suffix == "m":
        return int(value)
    return None


def _resample_ohlc(df: pd.DataFrame, rule: str, label: str = "left") -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label=label, closed=label)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _align_series(series: pd.Series, target_index: pd.Index) -> pd.Series:
    return series.reindex(target_index, method="ffill").fillna(method="bfill")


def _get_extension_end(
    monday: Monday,
    extension_type: str,
    fixed_bars: int,
    current_time: pd.Timestamp,
) -> pd.Timestamp:
    if extension_type == "End of Week":
        return monday.wk_end
    if extension_type == "Current Bar":
        return current_time
    return monday.wk_start + pd.Timedelta(days=fixed_bars)


def _is_timeframe_disallowed(chart_timeframe: str) -> bool:
    minutes = _parse_timeframe_to_minutes(chart_timeframe)
    if minutes is None:
        return False
    return minutes >= 60 * 24 * 7


def calculate_monday_range(
    df: pd.DataFrame,
    *,
    chart_timeframe: str = "15",
    max_mondays: int = 4,
    extension_type: str = "End of Week",
    fixed_bars_count: int = 5,
    use_mh: bool = True,
    mh_text: str = "MH",
    mh_color: str = "#007FFF",
    mh_line_style: str = "Solid",
    mh_line_width: int = 1,
    use_ml: bool = True,
    ml_text: str = "ML",
    ml_color: str = "#007FFF",
    ml_line_style: str = "Solid",
    ml_line_width: int = 1,
    use_mo: bool = False,
    mo_text: str = "MO",
    mo_color: str = "#007FFF",
    mo_line_style: str = "Solid",
    mo_line_width: int = 1,
    use_mc: bool = False,
    mc_text: str = "MC",
    mc_color: str = "#007FFF",
    mc_line_style: str = "Solid",
    mc_line_width: int = 1,
    custom_levels: Optional[List[LevelConfig]] = None,
    enable_alerts: bool = True,
) -> MondayRangeOutputs:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    can_show_monday_range = not _is_timeframe_disallowed(chart_timeframe)

    weekly = _resample_ohlc(df, "W", label="left")
    weekly_close = weekly.index + pd.Timedelta(days=7) - pd.Timedelta(milliseconds=1)
    daily = _resample_ohlc(df, "1D", label="left")

    wk_start_series = _align_series(pd.Series(weekly.index, index=weekly.index), df.index)
    wk_end_series = _align_series(pd.Series(weekly_close, index=weekly.index), df.index)

    monday_open = _align_series(daily["open"], df.index)
    monday_high = _align_series(daily["high"], df.index)
    monday_low = _align_series(daily["low"], df.index)
    monday_close = _align_series(daily["close"], df.index)

    new_week = wk_start_series.ne(wk_start_series.shift(1)).fillna(False)

    mondays_map: Dict[pd.Timestamp, Monday] = {}
    mondays_order: List[pd.Timestamp] = []

    high_breakouts: List[RangeEvent] = []
    low_breakouts: List[RangeEvent] = []
    high_reclaims: List[RangeEvent] = []
    low_reclaims: List[RangeEvent] = []

    high_touched: Dict[pd.Timestamp, bool] = {}
    low_touched: Dict[pd.Timestamp, bool] = {}

    if can_show_monday_range:
        for idx, is_new in enumerate(new_week):
            if not bool(is_new):
                continue
            wk_start = wk_start_series.iloc[idx]
            wk_end = wk_end_series.iloc[idx]
            if wk_start not in mondays_map:
                while len(mondays_order) >= 52:
                    oldest_key = mondays_order.pop()
                    mondays_map.pop(oldest_key, None)
                mondays_map[wk_start] = Monday(
                    wk_start=wk_start,
                    wk_end=wk_end,
                    open=float(monday_open.iloc[idx]),
                    high=float(monday_high.iloc[idx]),
                    low=float(monday_low.iloc[idx]),
                    close=float(monday_close.iloc[idx]),
                )
                mondays_order.insert(0, wk_start)

    if custom_levels is None:
        custom_levels = [
            LevelConfig(True, "EQ", 0.5, "#007FFF", "Solid", 1),
            LevelConfig(False, "", 0.0, "#007FFF", "Solid", 1),
            LevelConfig(False, "", 0.0, "#007FFF", "Solid", 1),
            LevelConfig(False, "", 0.0, "#007FFF", "Solid", 1),
            LevelConfig(False, "", 0.0, "#007FFF", "Solid", 1),
            LevelConfig(False, "", 0.0, "#007FFF", "Solid", 1),
        ]

    levels: List[LevelOutput] = []
    if can_show_monday_range and mondays_order:
        end_index = min(max_mondays, len(mondays_order))
        current_time = df.index[-1]
        for i in range(end_index):
            wk_key = mondays_order[i]
            monday = mondays_map.get(wk_key)
            if monday is None:
                continue
            monday_range = monday.price_range()
            extension_end = _get_extension_end(monday, extension_type, fixed_bars_count, current_time)

            if use_mh:
                levels.append(
                    LevelOutput(monday.wk_start, extension_end, monday.high, mh_text, mh_color, mh_line_style, mh_line_width, wk_key)
                )
            if use_ml:
                levels.append(
                    LevelOutput(monday.wk_start, extension_end, monday.low, ml_text, ml_color, ml_line_style, ml_line_width, wk_key)
                )
            if use_mo:
                levels.append(
                    LevelOutput(monday.wk_start, extension_end, monday.open, mo_text, mo_color, mo_line_style, mo_line_width, wk_key)
                )
            if use_mc:
                levels.append(
                    LevelOutput(monday.wk_start, extension_end, monday.close, mc_text, mc_color, mc_line_style, mc_line_width, wk_key)
                )

            for level_config in custom_levels:
                if level_config.enabled:
                    level_price = monday.low + (monday_range * level_config.value)
                    levels.append(
                        LevelOutput(
                            monday.wk_start,
                            extension_end,
                            level_price,
                            level_config.text,
                            level_config.color,
                            level_config.line_style,
                            level_config.line_width,
                            wk_key,
                        )
                    )

    high_break = pd.Series(False, index=df.index)
    low_break = pd.Series(False, index=df.index)
    high_reclaim = pd.Series(False, index=df.index)
    low_reclaim = pd.Series(False, index=df.index)

    if enable_alerts and can_show_monday_range:
        for idx, timestamp in enumerate(df.index):
            wk_start = wk_start_series.iloc[idx]
            monday = mondays_map.get(wk_start)
            if monday is None:
                continue
            if timestamp < monday.wk_start or timestamp > monday.wk_end:
                continue
            opening_day_end = monday.wk_start + pd.Timedelta(days=1)
            if timestamp < opening_day_end:
                continue

            if df["high"].iloc[idx] > monday.high or df["close"].iloc[idx] > monday.high:
                high_touched[monday.wk_start] = True

            if df["low"].iloc[idx] < monday.low or df["close"].iloc[idx] < monday.low:
                low_touched[monday.wk_start] = True

            prev_close = df["close"].iloc[idx - 1] if idx > 0 else df["close"].iloc[idx]

            if prev_close <= monday.high and df["close"].iloc[idx] > monday.high:
                high_breakouts.append(RangeEvent(timestamp, df["high"].iloc[idx], monday.wk_start))
                high_break.iloc[idx] = True

            if prev_close >= monday.low and df["close"].iloc[idx] < monday.low:
                low_breakouts.append(RangeEvent(timestamp, df["low"].iloc[idx], monday.wk_start))
                low_break.iloc[idx] = True

            if high_touched.get(monday.wk_start) and df["close"].iloc[idx] < monday.high:
                high_reclaims.append(RangeEvent(timestamp, df["high"].iloc[idx], monday.wk_start))
                high_reclaim.iloc[idx] = True
                high_touched[monday.wk_start] = False

            if low_touched.get(monday.wk_start) and df["close"].iloc[idx] > monday.low:
                low_reclaims.append(RangeEvent(timestamp, df["low"].iloc[idx], monday.wk_start))
                low_reclaim.iloc[idx] = True
                low_touched[monday.wk_start] = False

    alert_flags = {
        "high_break": high_break,
        "low_break": low_break,
        "high_reclaim": high_reclaim,
        "low_reclaim": low_reclaim,
        "range_break": high_break | low_break,
        "range_reclaim": high_reclaim | low_reclaim,
        "any_setup": high_break | low_break | high_reclaim | low_reclaim,
    }

    return MondayRangeOutputs(
        mondays=mondays_map,
        monday_order=mondays_order,
        levels=levels,
        high_breakouts=high_breakouts,
        low_breakouts=low_breakouts,
        high_reclaims=high_reclaims,
        low_reclaims=low_reclaims,
        alert_flags=alert_flags,
    )
