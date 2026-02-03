"""ICT Sessions_One Setup for Life [MK] - Python translation.

This module mirrors the Pine Script logic found in:
`S1_Silver_Bullet/ICT Sessions_One Setup for Life [MK].txt`.

It processes OHLCV data with a DatetimeIndex and produces session ranges,
RTH gap objects, opening lines, and weekly/monthly/yearly open levels.
The implementation focuses on faithfully reproducing the calculation flow
rather than rendering visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass
class SessionLine:
    name: str
    kind: str  # "high", "low", "mid"
    start: pd.Timestamp
    end: pd.Timestamp
    price: float
    active: bool = True


@dataclass
class SessionBox:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    high: float
    low: float


@dataclass
class SessionLabel:
    name: str
    time: pd.Timestamp
    price: float
    text: str


@dataclass
class RTHGap:
    start: pd.Timestamp
    end: pd.Timestamp
    top: float
    bottom: float
    mid: float
    q25: float
    q75: float
    four_pm_close: float
    open_price: float


@dataclass
class RTHLine:
    kind: str  # "mid", "q25", "q75", "close"
    start: pd.Timestamp
    end: pd.Timestamp
    price: float
    active: bool = True
    extend_right: bool = False


@dataclass
class RTHCloseLabel:
    time: pd.Timestamp
    price: float
    text: str


@dataclass
class OpeningLine:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp
    price: float
    active: bool = True


@dataclass
class OpenLevel:
    timeframe: str  # "weekly", "monthly", "yearly"
    time: pd.Timestamp
    price: float
    label_index: int
    anchor_index: int
    padding: int


@dataclass
class IndicatorConfig:
    timezone: str = "America/New_York"
    max_timeframe_minutes: int = 15
    event_days: int = 10
    show_days_background: bool = True

    asia_session: str = "2000-0000"
    europe_session: str = "0200-0500"
    usa_session: str = "0930-1200"
    nyl_session: str = "1200-1330"
    usa2_session: str = "1330-1600"

    show_asia: bool = True
    show_europe: bool = True
    show_usa: bool = True
    show_nyl: bool = True
    show_usa2: bool = True
    show_lines: bool = True
    show_mids: bool = False

    show_rth_gap: bool = True
    boxes_to_show: int = 3
    extend_gap_boxes: bool = False
    hours_forward: float = 1.0
    lines_to_show: int = 3
    show_4pm_line: bool = False
    show_4pm_label: bool = False
    extend_4pm_line_right: bool = False

    show_open_0000: bool = True
    show_open_0830: bool = True
    show_open_0930: bool = False
    history_open_lines: bool = False
    hide_open_lines_after_close: bool = True

    show_weekly: bool = True
    show_monthly: bool = True
    show_yearly: bool = False
    discover_prices: bool = False
    extended_hours: bool = False
    right_offset: int = 20


@dataclass
class IndicatorOutput:
    session_boxes: Dict[str, List[SessionBox]] = field(default_factory=dict)
    session_lines: Dict[str, List[SessionLine]] = field(default_factory=dict)
    session_labels: Dict[str, List[SessionLabel]] = field(default_factory=dict)
    rth_gaps: List[RTHGap] = field(default_factory=list)
    rth_lines: List[RTHLine] = field(default_factory=list)
    rth_close_labels: List[RTHCloseLabel] = field(default_factory=list)
    opening_lines: List[OpeningLine] = field(default_factory=list)
    open_levels: List[OpenLevel] = field(default_factory=list)
    open_series: pd.DataFrame = field(default_factory=pd.DataFrame)


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index
    elif "time" in df.columns:
        index = pd.to_datetime(df["time"])
    else:
        index = pd.to_datetime(df.index)
    if index.tz is None:
        index = index.tz_localize(tz)
    else:
        index = index.tz_convert(tz)
    df = df.copy()
    df.index = index
    return df


def _infer_bar_delta(index: pd.DatetimeIndex) -> timedelta:
    if len(index) < 2:
        return timedelta(minutes=1)
    deltas = index.to_series().diff().dropna()
    median = deltas.median()
    if pd.isna(median):
        return timedelta(minutes=1)
    return median.to_pytimedelta()


def _timeframe_flags(bar_delta: timedelta) -> Tuple[bool, bool, bool, bool]:
    is_intraday = bar_delta < timedelta(days=1)
    is_daily = timedelta(days=1) <= bar_delta < timedelta(days=7)
    is_weekly = timedelta(days=7) <= bar_delta < timedelta(days=28)
    is_monthly = bar_delta >= timedelta(days=28)
    return is_intraday, is_daily, is_weekly, is_monthly


def _get_padding(last_opens: List[float], index: int, can_show_weekly: bool, can_show_monthly: bool) -> int:
    padding = 0
    if index > 1 and can_show_weekly and last_opens[1] == last_opens[index]:
        padding += 1
    if index > 2 and can_show_monthly and last_opens[2] == last_opens[index]:
        padding += 1
    return padding


def _parse_session(session: str) -> Tuple[time, time]:
    start, end = session.split("-")
    return time(int(start[:2]), int(start[2:])), time(int(end[:2]), int(end[2:]))


def _in_session(dt: pd.Timestamp, start: time, end: time) -> bool:
    if start <= end:
        return start <= dt.time() < end
    return dt.time() >= start or dt.time() < end


def _session_start(dt: pd.Timestamp, start: time, end: time) -> pd.Timestamp:
    if start <= end:
        return dt.normalize() + timedelta(hours=start.hour, minutes=start.minute)
    if dt.time() >= start:
        return dt.normalize() + timedelta(hours=start.hour, minutes=start.minute)
    return (dt.normalize() - timedelta(days=1)) + timedelta(hours=start.hour, minutes=start.minute)


def _session_end(dt: pd.Timestamp, start: time, end: time) -> pd.Timestamp:
    start_ts = _session_start(dt, start, end)
    if start <= end:
        return start_ts + timedelta(hours=end.hour, minutes=end.minute) - timedelta(
            hours=start.hour, minutes=start.minute
        )
    return start_ts + timedelta(days=1, hours=end.hour, minutes=end.minute)


def _is_new_session(
    dt: pd.Timestamp,
    prev_dt: Optional[pd.Timestamp],
    start: time,
    end: time,
) -> bool:
    if not _in_session(dt, start, end):
        return False
    if prev_dt is None:
        return True
    if not _in_session(prev_dt, start, end):
        return True
    return _session_start(dt, start, end) != _session_start(prev_dt, start, end)


def _build_pre_range(index: pd.DatetimeIndex, event_days: int, tz: str) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=bool)
    timenow = index[-1].tz_convert(tz)
    starttime = datetime(
        timenow.year,
        timenow.month,
        timenow.day,
        23,
        59,
        tzinfo=ZoneInfo(tz),
    )
    pre_ts = starttime - timedelta(days=event_days)
    return (index >= pre_ts) & (index < starttime)


def _bar_contains_time(idx: pd.Timestamp, bar_end: pd.Timestamp, target: time) -> bool:
    for offset in (0, 1):
        day = idx.normalize() + timedelta(days=offset)
        target_ts = day + timedelta(hours=target.hour, minutes=target.minute)
        if idx <= target_ts < bar_end:
            return True
    return False


def _update_session(
    name: str,
    dt: pd.Timestamp,
    bar_end: pd.Timestamp,
    high: float,
    low: float,
    new_session: bool,
    show_mids: bool,
    show_lines: bool,
    allow_mid: bool,
    state: Dict[str, Dict[str, Optional[object]]],
    output: IndicatorOutput,
) -> None:
    session_state = state[name]
    if new_session:
        start = _session_start(dt, session_state["start"], session_state["end"])
        end = bar_end
        box = SessionBox(name=name, start=start, end=end, high=high, low=low)
        output.session_boxes.setdefault(name, []).append(box)
        high_line = None
        low_line = None
        if show_lines:
            high_line = SessionLine(name=name, kind="high", start=start, end=end, price=high)
            low_line = SessionLine(name=name, kind="low", start=start, end=end, price=low)
            output.session_lines.setdefault(name, []).extend([high_line, low_line])
        session_state["active_box"] = box
        session_state["active_high"] = high_line
        session_state["active_low"] = low_line
        if show_mids and allow_mid and show_lines:
            mid_price = (high + low) / 2
            mid_line = SessionLine(name=name, kind="mid", start=start, end=end, price=mid_price)
            output.session_lines[name].append(mid_line)
            session_state["active_mid"] = mid_line
        label = SessionLabel(name=name, time=(start + (end - start) / 2), price=high, text=name)
        output.session_labels.setdefault(name, []).append(label)
        session_state["active_label"] = label
    else:
        box: Optional[SessionBox] = session_state["active_box"]
        high_line: Optional[SessionLine] = session_state["active_high"]
        low_line: Optional[SessionLine] = session_state["active_low"]
        if box is None:
            return
        box.end = bar_end
        box.high = max(box.high, high)
        box.low = min(box.low, low)
        if high_line is not None:
            high_line.end = bar_end
            high_line.price = box.high
        if low_line is not None:
            low_line.end = bar_end
            low_line.price = box.low
        if show_mids and show_lines and allow_mid:
            mid_line: Optional[SessionLine] = session_state["active_mid"]
            if mid_line is not None:
                mid_line.end = bar_end
                mid_line.price = (box.high + box.low) / 2
        label: SessionLabel = session_state["active_label"]
        if label is not None:
            label.time = box.start + (box.end - box.start) / 2
            label.price = box.high


def _update_session_breaks(
    name: str,
    high: float,
    low: float,
    state: Dict[str, Dict[str, Optional[object]]],
) -> None:
    session_state = state[name]
    high_line: SessionLine = session_state["active_high"]
    low_line: SessionLine = session_state["active_low"]
    if high_line is not None and high > high_line.price:
        high_line.active = False
        session_state["active_high"] = None
    if low_line is not None and low < low_line.price:
        low_line.active = False
        session_state["active_low"] = None


def compute_indicator(df: pd.DataFrame, config: IndicatorConfig | None = None) -> IndicatorOutput:
    config = config or IndicatorConfig()
    tz = config.timezone
    df = _ensure_datetime_index(df, tz)
    bar_delta = _infer_bar_delta(df.index)
    intraday, is_daily, is_weekly, is_monthly = _timeframe_flags(bar_delta)
    minutes = bar_delta.total_seconds() / 60
    disp_rth = intraday and minutes <= config.max_timeframe_minutes
    pre_range = _build_pre_range(df.index, config.event_days, tz)

    sessions = {
        "Asia": (config.asia_session, config.show_asia, True),
        "London": (config.europe_session, config.show_europe, True),
        "NY AM": (config.usa_session, config.show_usa, True),
        "NY\nLunch": (config.nyl_session, config.show_nyl, False),
        "NY PM": (config.usa2_session, config.show_usa2, True),
    }

    state: Dict[str, Dict[str, Optional[object]]] = {}
    for name, (session, _, _) in sessions.items():
        start, end = _parse_session(session)
        state[name] = {
            "start": start,
            "end": end,
            "active_box": None,
            "active_high": None,
            "active_low": None,
            "active_mid": None,
            "active_label": None,
        }

    output = IndicatorOutput()

    can_show_weekly = config.show_weekly and (intraday or is_daily)
    can_show_monthly = config.show_monthly and not is_monthly
    can_show_yearly = config.show_yearly and not (is_monthly and bar_delta >= timedelta(days=365))

    four_pm_close = np.nan
    rth_boxes: List[RTHGap] = []
    rth_mid_lines: List[RTHLine] = []
    rth_q25_lines: List[RTHLine] = []
    rth_q75_lines: List[RTHLine] = []
    rth_close_lines: List[RTHLine] = []
    rth_close_labels: List[RTHCloseLabel] = []
    open_lines_buffer: List[OpeningLine] = []

    open_levels: List[OpenLevel] = []
    last_opens = [np.nan, np.nan, np.nan, np.nan]
    offset_padding = 4
    weekly_open = np.nan
    monthly_open = np.nan
    yearly_open = np.nan

    if len(df.index) > 0:
        last_bar_time = df.index[-1] + bar_delta
    else:
        last_bar_time = pd.Timestamp.now(tz=tz)

    weekly_frame = (
        df["open"].resample("W-SUN", label="left", closed="left").first().to_frame("open")
    )
    weekly_frame["time"] = weekly_frame.index
    monthly_frame = df["open"].resample("MS", label="left", closed="left").first().to_frame("open")
    monthly_frame["time"] = monthly_frame.index
    yearly_frame = df["open"].resample("AS-JAN", label="left", closed="left").first().to_frame("open")
    yearly_frame["time"] = yearly_frame.index

    weekly_frame = weekly_frame.reindex(df.index, method="ffill")
    monthly_frame = monthly_frame.reindex(df.index, method="ffill")
    yearly_frame = yearly_frame.reindex(df.index, method="ffill")

    prev_dt: Optional[pd.Timestamp] = None
    prev_open_0000 = False
    prev_open_0830 = False
    prev_open_0930 = False
    prev_open_0000_session = False
    prev_open_0830_session = False
    prev_open_0930_session = False
    openprice_0000 = 0.0
    openprice_0830 = 0.0
    openprice_0930 = 0.0

    open_series = pd.DataFrame(index=df.index, columns=["weekly", "monthly", "yearly"], dtype=float)

    for bar_index, (idx, row) in enumerate(df.iterrows()):
        bar_end = idx + bar_delta
        in_pre_range = bool(pre_range.loc[idx]) if len(pre_range) else True

        # Session logic
        if disp_rth and in_pre_range:
            for name, (session, enabled, allow_mid) in sessions.items():
                if not enabled:
                    continue
                start, end = _parse_session(session)
                in_session = _in_session(idx, start, end)
                new_session = _is_new_session(idx, prev_dt, start, end)
                if in_session:
                    _update_session(
                        name,
                        idx,
                        bar_end,
                        float(row["high"]),
                        float(row["low"]),
                        new_session,
                        config.show_mids,
                        config.show_lines,
                        allow_mid,
                        state,
                        output,
                    )
                else:
                    _update_session_breaks(
                        name,
                        float(row["high"]),
                        float(row["low"]),
                        state,
                    )

        # RTH gap logic (Pine Script has pre_range/disp_RTHsess commented out)
        if config.show_rth_gap:
            if bar_end.time() == time(16, 15):
                four_pm_close = float(row["close"])
            if idx.time() == time(9, 30) and not np.isnan(four_pm_close):
                gap_end = idx + timedelta(hours=config.hours_forward)
                projected_end = last_bar_time if config.extend_gap_boxes else gap_end
                gap_mid = (four_pm_close + float(row["open"])) / 2
                gap = RTHGap(
                    start=idx,
                    end=projected_end,
                    top=four_pm_close,
                    bottom=float(row["open"]),
                    mid=gap_mid,
                    q25=(gap_mid + float(row["open"])) / 2,
                    q75=(gap_mid + four_pm_close) / 2,
                    four_pm_close=four_pm_close,
                    open_price=float(row["open"]),
                )
                rth_boxes.append(gap)
                if len(rth_boxes) > config.boxes_to_show:
                    rth_boxes.pop(0)

                rth_q75_lines.append(RTHLine(kind="q75", start=idx, end=projected_end, price=gap.q75))
                if len(rth_q75_lines) > config.boxes_to_show:
                    rth_q75_lines.pop(0)

                rth_q25_lines.append(RTHLine(kind="q25", start=idx, end=projected_end, price=gap.q25))
                if len(rth_q25_lines) > config.boxes_to_show:
                    rth_q25_lines.pop(0)

                rth_mid_lines.append(RTHLine(kind="mid", start=idx, end=projected_end, price=gap.mid))
                if len(rth_mid_lines) > config.boxes_to_show:
                    rth_mid_lines.pop(0)

                close_end = last_bar_time
                close_line = RTHLine(
                    kind="close",
                    start=idx,
                    end=close_end,
                    price=four_pm_close,
                    active=config.show_4pm_line,
                    extend_right=config.extend_4pm_line_right,
                )
                rth_close_lines.append(close_line)
                if len(rth_close_lines) > config.lines_to_show:
                    rth_close_lines.pop(0)
                if config.show_4pm_label:
                    rth_close_labels.append(
                        RTHCloseLabel(time=last_bar_time, price=four_pm_close, text=str(four_pm_close))
                    )
                    if len(rth_close_labels) > config.lines_to_show:
                        rth_close_labels.pop(0)

        # Opening lines (00:00, 08:30, 09:30)
        open_0000_session = _bar_contains_time(idx, bar_end, time(0, 0))
        open_0830_session = _bar_contains_time(idx, bar_end, time(8, 30))
        open_0930_session = _bar_contains_time(idx, bar_end, time(9, 30))

        if open_0000_session:
            if not prev_open_0000_session:
                openprice_0000 = float(row["open"])
            else:
                openprice_0000 = max(openprice_0000, float(row["open"]))

        if open_0830_session:
            if not prev_open_0830_session:
                openprice_0830 = float(row["open"])
            else:
                openprice_0830 = max(openprice_0830, float(row["open"]))

        if open_0930_session:
            if not prev_open_0930_session:
                openprice_0930 = float(row["open"])
            else:
                openprice_0930 = max(openprice_0930, float(row["open"]))

        open_0000 = open_0000_session
        open_0830 = open_0830_session
        open_0930 = open_0930_session

        if open_0000 and config.show_open_0000 and (not prev_open_0000):
            line = OpeningLine(
                kind="00:00",
                start=idx,
                end=idx + timedelta(hours=16) + timedelta(days=2 if idx.weekday() == 4 else 0),
                price=float(openprice_0000),
            )
            open_lines_buffer.append(line)
            if not config.history_open_lines:
                for existing in open_lines_buffer[:-1]:
                    if existing.kind == "00:00":
                        existing.active = False

        if open_0830 and config.show_open_0830 and (not prev_open_0830):
            line = OpeningLine(
                kind="08:30",
                start=idx,
                end=idx + timedelta(hours=3, minutes=30),
                price=float(openprice_0830),
            )
            open_lines_buffer.append(line)
            if not config.history_open_lines:
                for existing in open_lines_buffer[:-1]:
                    if existing.kind == "08:30":
                        existing.active = False

        if open_0930 and config.show_open_0930 and (not prev_open_0930):
            line = OpeningLine(
                kind="09:30",
                start=idx,
                end=idx + timedelta(hours=2, minutes=30),
                price=float(openprice_0930),
            )
            open_lines_buffer.append(line)
            if not config.history_open_lines:
                for existing in open_lines_buffer[:-1]:
                    if existing.kind == "09:30":
                        existing.active = False

        if config.hide_open_lines_after_close and _in_session(idx, time(15, 0), time(20, 0)):
            for line in open_lines_buffer:
                if line.kind in {"08:30", "09:30"}:
                    line.active = False

        # Weekly/Monthly/Yearly opens
        weekly_time = weekly_frame.at[idx, "time"] if idx in weekly_frame.index else pd.NaT
        weekly_open_value = weekly_frame.at[idx, "open"] if idx in weekly_frame.index else np.nan
        monthly_time = monthly_frame.at[idx, "time"] if idx in monthly_frame.index else pd.NaT
        monthly_open_value = monthly_frame.at[idx, "open"] if idx in monthly_frame.index else np.nan
        yearly_time = yearly_frame.at[idx, "time"] if idx in yearly_frame.index else pd.NaT
        yearly_open_value = yearly_frame.at[idx, "open"] if idx in yearly_frame.index else np.nan

        if can_show_weekly:
            week_changed = False
            if config.extended_hours:
                prev_week = prev_dt.isocalendar().week if prev_dt is not None else None
                week_changed = prev_week is None or idx.isocalendar().week != prev_week
            else:
                prev_week_time = (
                    weekly_frame.at[prev_dt, "time"] if prev_dt in weekly_frame.index else pd.NaT
                )
                week_changed = prev_dt is None or weekly_time != prev_week_time
            if week_changed:
                weekly_open = float(row["open"]) if config.discover_prices else float(weekly_open_value)
                last_opens[1] = weekly_open
                padding = _get_padding(last_opens, 1, can_show_weekly, can_show_monthly)
                label_index = bar_index + config.right_offset + padding * offset_padding
                anchor_index = bar_index
                if is_weekly and pd.notna(weekly_time) and idx.day > weekly_time.day and bar_index > 0:
                    anchor_index = bar_index - 1
                open_levels.append(
                    OpenLevel(
                        timeframe="weekly",
                        time=idx,
                        price=weekly_open,
                        label_index=label_index,
                        anchor_index=anchor_index,
                        padding=padding,
                    )
                )

        if can_show_monthly:
            month_changed = False
            if config.extended_hours:
                month_changed = prev_dt is None or idx.month != prev_dt.month
            else:
                prev_month_time = (
                    monthly_frame.at[prev_dt, "time"] if prev_dt in monthly_frame.index else pd.NaT
                )
                month_changed = prev_dt is None or monthly_time != prev_month_time
            if month_changed:
                monthly_open = float(row["open"]) if config.discover_prices else float(monthly_open_value)
                last_opens[2] = monthly_open
                padding = _get_padding(last_opens, 2, can_show_weekly, can_show_monthly)
                label_index = bar_index + config.right_offset + padding * offset_padding
                anchor_index = bar_index
                if is_weekly and pd.notna(monthly_time) and idx.day > monthly_time.day and bar_index > 0:
                    anchor_index = bar_index - 1
                open_levels.append(
                    OpenLevel(
                        timeframe="monthly",
                        time=idx,
                        price=monthly_open,
                        label_index=label_index,
                        anchor_index=anchor_index,
                        padding=padding,
                    )
                )

        if can_show_yearly:
            year_changed = False
            if config.extended_hours:
                year_changed = prev_dt is None or idx.year != prev_dt.year
            else:
                prev_year_time = (
                    yearly_frame.at[prev_dt, "time"] if prev_dt in yearly_frame.index else pd.NaT
                )
                year_changed = prev_dt is None or yearly_time != prev_year_time
            if year_changed:
                yearly_open = float(row["open"]) if config.discover_prices else float(yearly_open_value)
                last_opens[3] = yearly_open
                padding = _get_padding(last_opens, 3, can_show_weekly, can_show_monthly)
                label_index = bar_index + config.right_offset + padding * offset_padding
                anchor_index = bar_index
                if is_weekly and pd.notna(yearly_time) and idx.day > yearly_time.day and bar_index > 0:
                    anchor_index = bar_index - 1
                open_levels.append(
                    OpenLevel(
                        timeframe="yearly",
                        time=idx,
                        price=yearly_open,
                        label_index=label_index,
                        anchor_index=anchor_index,
                        padding=padding,
                    )
                )

        open_series.loc[idx, "weekly"] = weekly_open
        open_series.loc[idx, "monthly"] = monthly_open
        open_series.loc[idx, "yearly"] = yearly_open

        prev_dt = idx
        prev_open_0000 = open_0000
        prev_open_0830 = open_0830
        prev_open_0930 = open_0930
        prev_open_0000_session = open_0000_session
        prev_open_0830_session = open_0830_session
        prev_open_0930_session = open_0930_session

    output.rth_gaps = rth_boxes
    output.rth_lines = rth_q75_lines + rth_q25_lines + rth_mid_lines + rth_close_lines
    output.rth_close_labels = rth_close_labels
    output.opening_lines = open_lines_buffer
    output.open_levels = open_levels
    output.open_series = open_series
    return output


__all__ = [
    "IndicatorConfig",
    "IndicatorOutput",
    "SessionBox",
    "SessionLine",
    "SessionLabel",
    "RTHGap",
    "RTHLine",
    "RTHCloseLabel",
    "OpeningLine",
    "OpenLevel",
    "compute_indicator",
]
