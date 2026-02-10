"""One Setup for Life ICT - Python translation.

This module mirrors the Pine Script logic found in:
`S1_Silver_Bullet/One Setup for Life ICT.txt`.

It processes OHLC data with a DatetimeIndex and emits the session boxes,
prior-session high/low lines, divider markers, and optional text boxes that
would normally be rendered on a chart. The implementation aims to preserve the
original control flow and state transitions rather than rendering visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class DividerLine:
    kind: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class BackgroundRange:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class TextBox:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    top: float
    bottom: float
    color: str
    text: str


@dataclass
class TextBoxLine:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    price: float


@dataclass
class SessionBox:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    high: float
    low: float
    show_box: bool = True


@dataclass
class SessionLine:
    name: str
    kind: str  # "high" or "low"
    start: pd.Timestamp
    end: pd.Timestamp
    price: float
    active: bool = True


@dataclass
class SessionLabel:
    name: str
    time: pd.Timestamp
    price: float
    text: str
    active: bool = True


@dataclass
class IndicatorConfig:
    timezone: str = "America/New_York"
    max_timeframe_minutes: int = 15

    show_daily: bool = True
    show_br_sess: bool = False
    show_br_sb: bool = False
    show_br_ln: bool = False
    show_br_ny: bool = False
    show_br_pm: bool = False

    show_session: bool = True
    show_box_session: bool = True
    show_text_box: bool = True
    show_prev_lines: bool = True
    show_text_lines: bool = False
    show_price_session: bool = False
    show_only_today: bool = False

    choice_start_osfl: str = "09:30"
    show_line_start: bool = True

    show_sess_0: bool = True
    show_sess_1: bool = True
    show_sess_2: bool = True
    show_sess_3: bool = True

    sess_0: str = "0200-0500"
    sess_1: str = "0930-1200"
    sess_2: str = "1200-1330"
    sess_3: str = "1330-1600"

    txt_sess_0: str = "London"
    txt_sess_1: str = "Prev.AM"
    txt_sess_2: str = "Lunch"
    txt_sess_3: str = "PM"


@dataclass
class IndicatorOutput:
    divider_lines: List[DividerLine] = field(default_factory=list)
    background_ranges: List[BackgroundRange] = field(default_factory=list)
    text_boxes: List[TextBox] = field(default_factory=list)
    text_box_lines: List[TextBoxLine] = field(default_factory=list)
    session_boxes: Dict[str, List[SessionBox]] = field(default_factory=dict)
    session_lines: Dict[str, List[SessionLine]] = field(default_factory=dict)
    session_labels: Dict[str, List[SessionLabel]] = field(default_factory=dict)


def _ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df


def _infer_bar_delta(index: pd.DatetimeIndex) -> timedelta:
    if len(index) < 2:
        return timedelta(minutes=1)
    diffs = index.to_series().diff().dropna()
    return diffs.mode().iloc[0].to_pytimedelta()


def _session_mask(index: pd.DatetimeIndex, session: str, tz: str) -> pd.Series:
    start_str, end_str = session.split("-")
    start_time = time(int(start_str[:2]), int(start_str[2:]))
    end_time = time(int(end_str[:2]), int(end_str[2:]))
    local = index.tz_convert(tz)
    times = local.time
    if start_time <= end_time:
        mask = (times >= start_time) & (times <= end_time)
    else:
        mask = (times >= start_time) | (times <= end_time)
    return pd.Series(mask, index=index)


def _bar_contains_time(bar_start: pd.Timestamp, bar_end: pd.Timestamp, target: time) -> bool:
    for offset in (0, 1):
        day = bar_start.normalize() + timedelta(days=offset)
        target_ts = day + timedelta(hours=target.hour, minutes=target.minute)
        if bar_start <= target_ts < bar_end:
            return True
    return False


def _parse_choice_time(choice: str) -> time:
    hour, minute = choice.split(":")
    return time(int(hour), int(minute))


def compute_indicator(df: pd.DataFrame, config: IndicatorConfig | None = None) -> IndicatorOutput:
    config = config or IndicatorConfig()
    tz = config.timezone
    df = _ensure_datetime_index(df, tz)
    bar_delta = _infer_bar_delta(df.index)
    minutes = bar_delta.total_seconds() / 60
    intraday = bar_delta < timedelta(days=1)
    dom = intraday and minutes <= config.max_timeframe_minutes

    output = IndicatorOutput()

    sess_masks = {
        "sess_0": _session_mask(df.index, config.sess_0, tz),
        "sess_1": _session_mask(df.index, config.sess_1, tz),
        "sess_2": _session_mask(df.index, config.sess_2, tz),
        "sess_3": _session_mask(df.index, config.sess_3, tz),
        "ln": _session_mask(df.index, "0200-0500", tz),
        "ny": _session_mask(df.index, "0930-1200", tz),
        "pm": _session_mask(df.index, "1330-1600", tz),
        "sb_ln": _session_mask(df.index, "0300-0400", tz),
        "sb_ny": _session_mask(df.index, "1000-1100", tz),
        "sb_pm": _session_mask(df.index, "1400-1500", tz),
        "txt_sess_2": _session_mask(df.index, "1201-1330", tz),
        "txt_sess_3": _session_mask(df.index, "1331-1600", tz),
    }

    state = {
        "text": {
            "in_kz": False,
            "box": None,
            "start_line": None,
        },
        "sess_0": {
            "box": None,
            "high": None,
            "low": None,
            "label_high": None,
            "label_low": None,
            "can_high": False,
            "can_low": False,
        },
        "sess_1": {
            "box": None,
            "high": None,
            "low": None,
            "label_high": None,
            "label_low": None,
            "can_high": False,
            "can_low": False,
        },
        "sess_2": {
            "box": None,
            "high": None,
            "low": None,
            "label_high": None,
            "label_low": None,
            "can_high": False,
            "can_low": False,
        },
        "sess_3": {
            "box": None,
            "high": None,
            "low": None,
            "label_high": None,
            "label_low": None,
            "can_high": False,
            "can_low": False,
        },
    }

    choice_time = _parse_choice_time(config.choice_start_osfl)
    time_start_time = choice_time
    if config.choice_start_osfl == "02:00":
        line_start_time = time(2, 0)
    elif config.choice_start_osfl == "08:30":
        line_start_time = time(8, 30)
    else:
        line_start_time = time(9, 30)
    time_start_ln = time(2, 0)
    time_start_lc = time(13, 30)

    prev_in_kz = False

    for i, (dt, row) in enumerate(df.iterrows()):
        if not dom:
            continue

        bar_end = dt + bar_delta
        high = float(row["high"])
        low = float(row["low"])

        def in_session(mask_name: str) -> bool:
            return bool(sess_masks[mask_name].iloc[i])

        def new_bar(mask_name: str) -> bool:
            current = bool(sess_masks[mask_name].iloc[i])
            prev = bool(sess_masks[mask_name].iloc[i - 1]) if i > 0 else False
            return current and not prev

        is_new_session = _bar_contains_time(dt, bar_end, time(0, 0))
        is_new_session_ln = _bar_contains_time(dt, bar_end, time(5, 0))
        is_new_session_ny = _bar_contains_time(dt, bar_end, time(12, 0))
        is_new_session_lc = _bar_contains_time(dt, bar_end, time(13, 30))

        is_line_start = _bar_contains_time(dt, bar_end, line_start_time)

        if is_new_session and config.show_daily:
            output.divider_lines.append(DividerLine(kind="daily", start=dt, end=bar_end))

        if is_line_start and config.show_line_start:
            output.divider_lines.append(DividerLine(kind="line_start", start=dt, end=bar_end))

        if config.show_br_sess:
            if config.show_br_ln and in_session("ln"):
                output.background_ranges.append(BackgroundRange(name="Session LN", start=dt, end=bar_end))
            if config.show_br_ny and in_session("ny"):
                output.background_ranges.append(BackgroundRange(name="Session NY", start=dt, end=bar_end))
            if config.show_br_pm and in_session("pm"):
                output.background_ranges.append(BackgroundRange(name="Session PM", start=dt, end=bar_end))
        if config.show_br_sb:
            if config.show_br_ln and in_session("sb_ln"):
                output.background_ranges.append(BackgroundRange(name="Silver Bullet London", start=dt, end=bar_end))
            if config.show_br_ny and in_session("sb_ny"):
                output.background_ranges.append(BackgroundRange(name="Silver Bullet NewYork", start=dt, end=bar_end))
            if config.show_br_pm and in_session("sb_pm"):
                output.background_ranges.append(BackgroundRange(name="Silver Bullet PM", start=dt, end=bar_end))

        if config.show_session:
            if config.show_text_box:
                if config.show_sess_0 and in_session("sess_0"):
                    in_kz = True
                    kz_color = "sess_0"
                    kz_text = config.txt_sess_0
                elif config.show_sess_1 and in_session("sess_1"):
                    in_kz = True
                    kz_color = "sess_1"
                    kz_text = config.txt_sess_1
                elif config.show_sess_2 and in_session("txt_sess_2"):
                    in_kz = True
                    kz_color = "sess_2"
                    kz_text = config.txt_sess_2
                elif config.show_sess_3 and in_session("txt_sess_3"):
                    in_kz = True
                    kz_color = "sess_3"
                    kz_text = config.txt_sess_3
                else:
                    in_kz = False
            else:
                in_kz = True
                kz_color = ""
                kz_text = ""

            if config.show_text_box:
                if in_kz and not prev_in_kz:
                    box = TextBox(
                        name="session_text",
                        start=dt,
                        end=bar_end,
                        top=high,
                        bottom=low,
                        color=kz_color,
                        text=kz_text,
                    )
                    output.text_boxes.append(box)
                    output.text_box_lines.append(TextBoxLine(name="kz_start", start=dt, end=bar_end, price=high))
                    state["text"]["box"] = box
                    state["text"]["start_line"] = output.text_box_lines[-1]
                if prev_in_kz:
                    box = state["text"]["box"]
                    if box is not None:
                        box.end = bar_end
                        if high > box.bottom:
                            box.bottom = 1.0010 * high
                            box.top = 1.0015 * high
                        start_line = state["text"]["start_line"]
                        if start_line is not None:
                            start_line.end = bar_end
                            start_line.price = box.top
                if prev_in_kz and not in_kz:
                    box = state["text"]["box"]
                    end_price = box.top if box is not None else high
                    output.text_box_lines.append(TextBoxLine(name="kz_end", start=dt, end=bar_end, price=end_price))

            prev_in_kz = in_kz

        def update_session(
            session_key: str,
            show_session: bool,
            label_text: str,
            new_session_flag: bool,
            stop_time: time,
            high_or_break: bool = False,
        ) -> None:
            if not (config.show_session and show_session):
                return
            mask_name = session_key
            state_key = session_key

            if new_bar(mask_name):
                box = SessionBox(
                    name=label_text,
                    start=dt,
                    end=bar_end,
                    high=high,
                    low=low,
                    show_box=config.show_box_session,
                )
                output.session_boxes.setdefault(label_text, []).append(box)
                state[state_key]["box"] = box
                if config.show_only_today and len(output.session_boxes[label_text]) > 1:
                    removal_index = -3 if session_key == "sess_3" else -2
                    if len(output.session_boxes[label_text]) >= abs(removal_index):
                        output.session_boxes[label_text].pop(removal_index)
            elif in_session(mask_name):
                box = state[state_key]["box"]
                if box is not None:
                    box.end = bar_end
                    if high > box.high:
                        box.high = high
                    if low < box.low:
                        box.low = low

            if new_session_flag and config.show_prev_lines:
                box = state[state_key]["box"]
                if box is None:
                    return
                high_line = SessionLine(name=label_text, kind="high", start=dt, end=bar_end, price=box.high)
                low_line = SessionLine(name=label_text, kind="low", start=dt, end=bar_end, price=box.low)
                output.session_lines.setdefault(label_text, []).extend([high_line, low_line])
                state[state_key]["high"] = high_line
                state[state_key]["low"] = low_line
                state[state_key]["can_high"] = True
                state[state_key]["can_low"] = True
                if config.show_only_today and len(output.session_lines[label_text]) > 2:
                    output.session_lines[label_text].pop(-3)
                    output.session_lines[label_text].pop(-3)
                if config.show_text_lines:
                    high_text = f"{label_text} High"
                    low_text = f"{label_text} Low"
                    if config.show_price_session:
                        high_text = f"{label_text} High {box.high}"
                        low_text = f"{label_text} Low {box.low}"
                    label_high = SessionLabel(name=label_text, time=dt, price=box.high, text=high_text)
                    label_low = SessionLabel(name=label_text, time=dt, price=box.low, text=low_text)
                    output.session_labels.setdefault(label_text, []).extend([label_high, label_low])
                    prev_label_high = state[state_key]["label_high"]
                    prev_label_low = state[state_key]["label_low"]
                    if prev_label_high is not None:
                        prev_label_high.active = False
                    if prev_label_low is not None:
                        prev_label_low.active = False
                    state[state_key]["label_high"] = label_high
                    state[state_key]["label_low"] = label_low
                    if config.show_only_today and len(output.session_labels[label_text]) > 2:
                        output.session_labels[label_text].pop(-3)
                        output.session_labels[label_text].pop(-3)
            else:
                if state[state_key]["can_high"]:
                    high_line = state[state_key]["high"]
                    if high_line is not None:
                        stop_dt = dt.normalize() + timedelta(hours=stop_time.hour, minutes=stop_time.minute)
                        if high_or_break:
                            should_stop = dt > stop_dt or high > high_line.price
                        else:
                            should_stop = dt > stop_dt and high > high_line.price
                        if should_stop:
                            state[state_key]["can_high"] = False
                            high_line.active = False
                            if config.show_text_lines:
                                label_high = state[state_key]["label_high"]
                                if label_high is not None:
                                    label_high.active = False
                        else:
                            high_line.end = bar_end
                if state[state_key]["can_low"]:
                    low_line = state[state_key]["low"]
                    if low_line is not None:
                        if dt > dt.normalize() + timedelta(hours=stop_time.hour, minutes=stop_time.minute) and low < low_line.price:
                            state[state_key]["can_low"] = False
                            low_line.active = False
                            if config.show_text_lines:
                                label_low = state[state_key]["label_low"]
                                if label_low is not None:
                                    label_low.active = False
                        else:
                            low_line.end = bar_end

        update_session("sess_0", config.show_sess_0, config.txt_sess_0, is_new_session_ln, time_start_ln)
        update_session("sess_1", config.show_sess_1, config.txt_sess_1, is_new_session_ny, time_start_time)
        update_session(
            "sess_2",
            config.show_sess_2,
            config.txt_sess_2,
            is_new_session_lc,
            time_start_lc,
            high_or_break=True,
        )
        update_session("sess_3", config.show_sess_3, config.txt_sess_3, is_new_session, time_start_time)

    return output


__all__ = [
    "BackgroundRange",
    "DividerLine",
    "IndicatorConfig",
    "IndicatorOutput",
    "SessionBox",
    "SessionLabel",
    "SessionLine",
    "TextBox",
    "TextBoxLine",
    "compute_indicator",
]
