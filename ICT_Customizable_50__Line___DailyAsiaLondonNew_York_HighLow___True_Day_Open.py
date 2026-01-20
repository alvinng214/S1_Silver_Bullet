"""
ICT Customizable 50% Line & Daily/Asia/London/New York High/Low + True Day Open
Python implementation that mirrors the Pine Script logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


@dataclass
class Line:
    x1: Union[int, datetime]
    y1: float
    x2: Union[int, datetime]
    y2: float
    color: str
    width: int
    style: str
    kind: str
    session_name: Optional[str] = None
    break_idx: Optional[int] = None


@dataclass
class Label:
    x: Union[int, datetime]
    y: float
    text: str
    color: str
    size: str
    kind: str


@dataclass
class DailyState:
    start_bar: int
    end_bar: int
    day_high: float
    day_low: float
    mid_level: float
    line_50: Line
    high_line: Optional[Line]
    low_line: Optional[Line]
    market_open_line: Optional[Line]
    label_50: Optional[Label]
    label_high: Optional[Label]
    label_low: Optional[Label]
    label_market_open: Optional[Label]


@dataclass
class SessionLine:
    line: Line
    label: Optional[Label]
    session_high: bool


@dataclass
class ChecklistUpdate:
    bar_index: int
    rows: List[Tuple[str, bool]]
    summary: str


@dataclass
class Watermark:
    text_primary: str
    text_secondary: str


@dataclass
class IndicatorConfig:
    mintick: float = 0.01

    line_color: str = "#CEE9FF"
    line_width: int = 1
    dotted_line: bool = True
    opacity_50: int = 50

    show_50_text: bool = True
    label_text_50: str = "50% Level"
    text_color_50: str = "#FFFFFF"
    text_size_50: str = "small"
    text_opacity_50: int = 100
    text_pos_50: str = "Middle"

    show_high_low: bool = True
    high_low_color: str = "#FF0000"
    high_low_width: int = 1
    dotted_high_low: bool = True
    show_hl_text: bool = True
    high_text: str = "Daily High"
    low_text: str = "Daily Low"
    high_low_text_color: str = "#FFFFFF"
    high_low_text_size: str = "small"
    high_low_text_opacity: int = 100
    high_low_text_pos: str = "Middle"

    show_market_open: bool = True
    market_open_line_color: str = "#FFFFFF"
    market_open_line_opacity: int = 70
    market_open_line_width: int = 1
    market_open_text: str = "Market Open"
    market_open_text_color: str = "#FFFFFF"
    market_open_text_size: str = "small"
    market_open_text_opacity: int = 50
    market_open_text_pos: str = "Middle"

    enable_watermark: bool = True
    watermark_text: str = "Motivational Words"
    watermark_text2: str = "Customize me!"

    show_true_day_open: bool = True
    true_day_open_color: str = "#808080"
    true_day_open_width: int = 1
    true_day_open_text_color: str = "#FFFFFF"
    true_day_open_text_size: str = "small"
    true_day_open_text_position: str = "Middle"

    enable_asia_session: bool = True
    enable_london_session: bool = True
    enable_ny_session: bool = True
    enable_ny_am: bool = True
    enable_ny_lunch: bool = True
    enable_ny_pm: bool = True

    line_color_asia: str = "#0000FF"
    line_color_london: str = "#FD7200"
    line_color_ny: str = "#00FF96"
    line_color_ny_am: str = "#00FFFF"
    line_color_ny_lunch: str = "#FFFF00"
    line_color_ny_pm: str = "#FF00FF"
    killzone_dotted_lines: bool = False

    show_labels: bool = True
    label_align_opt: str = "Center"
    extend_bars: int = 1500
    session_label_text_color: str = "#FFFFFF"

    show_checklist: bool = True
    checklist_size: str = "Medium"
    checklist_position: str = "Top Right"

    c1: bool = False
    c2: bool = False
    c3: bool = False
    c4: bool = False
    c5: bool = False
    c6: bool = False
    c7: bool = False
    c8: bool = False


def _calc_label_x(position: str, bar_index: int) -> int:
    if position == "Left":
        return bar_index - 10
    if position == "Right":
        return bar_index + 300
    return bar_index + 150


def _calc_label_y(base: float, position: str, mintick: float) -> float:
    if position == "Top":
        return base + 100 * mintick
    if position == "Bottom":
        return base - 100 * mintick
    return base


def _opacity_to_alpha(opacity: int) -> int:
    return max(0, min(100, 100 - opacity))


def build_daily_states(df: pd.DataFrame, config: IndicatorConfig) -> List[DailyState]:
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df_chi = df.tz_convert("America/Chicago")

    states: List[DailyState] = []
    day_high = None
    day_low = None
    start_bar = None

    for idx in range(len(df_chi)):
        current = df_chi.index[idx]
        prev = df_chi.index[idx - 1] if idx > 0 else None
        new_session = current.hour >= 17 and (prev is None or prev.hour < 17)

        high = float(df_chi.iloc[idx]["high"])
        low = float(df_chi.iloc[idx]["low"])

        if new_session:
            start_bar = idx
            day_high = high
            day_low = low
        else:
            day_high = max(day_high, high) if day_high is not None else high
            day_low = min(day_low, low) if day_low is not None else low

        mid_level = (day_high + day_low) / 2
        end_bar = idx + 500

        line_50 = Line(
            x1=start_bar,
            y1=mid_level,
            x2=end_bar,
            y2=mid_level,
            color=config.line_color,
            width=config.line_width,
            style="dotted" if config.dotted_line else "solid",
            kind="50%",
        )

        label_50 = None
        if config.show_50_text:
            label_50 = Label(
                x=_calc_label_x(config.text_pos_50, idx),
                y=mid_level,
                text=config.label_text_50,
                color=config.text_color_50,
                size=config.text_size_50,
                kind="50%",
            )

        high_line = None
        low_line = None
        label_high = None
        label_low = None

        if config.show_high_low:
            style = "dotted" if config.dotted_high_low else "solid"
            high_line = Line(
                x1=start_bar,
                y1=day_high,
                x2=end_bar,
                y2=day_high,
                color=config.high_low_color,
                width=config.high_low_width,
                style=style,
                kind="daily_high",
            )
            low_line = Line(
                x1=start_bar,
                y1=day_low,
                x2=end_bar,
                y2=day_low,
                color=config.high_low_color,
                width=config.high_low_width,
                style=style,
                kind="daily_low",
            )

            if config.show_hl_text:
                label_high = Label(
                    x=_calc_label_x(config.high_low_text_pos, idx),
                    y=day_high,
                    text=config.high_text,
                    color=config.high_low_text_color,
                    size=config.high_low_text_size,
                    kind="daily_high",
                )
                label_low = Label(
                    x=_calc_label_x(config.high_low_text_pos, idx),
                    y=day_low,
                    text=config.low_text,
                    color=config.high_low_text_color,
                    size=config.high_low_text_size,
                    kind="daily_low",
                )

        market_open_line = None
        market_open_label = None
        if config.show_market_open:
            market_open_line = Line(
                x1=start_bar,
                y1=day_high + 100 * config.mintick,
                x2=start_bar,
                y2=day_low - 100 * config.mintick,
                color=config.market_open_line_color,
                width=config.market_open_line_width,
                style="dotted",
                kind="market_open",
            )
            market_open_label = Label(
                x=start_bar,
                y=_calc_label_y(mid_level, config.market_open_text_pos, config.mintick),
                text=config.market_open_text,
                color=config.market_open_text_color,
                size=config.market_open_text_size,
                kind="market_open",
            )

        states.append(
            DailyState(
                start_bar=start_bar,
                end_bar=end_bar,
                day_high=day_high,
                day_low=day_low,
                mid_level=mid_level,
                line_50=line_50,
                high_line=high_line,
                low_line=low_line,
                market_open_line=market_open_line,
                label_50=label_50,
                label_high=label_high,
                label_low=label_low,
                label_market_open=market_open_label,
            )
        )

    return states


def build_true_day_open_states(
    df: pd.DataFrame,
    daily_states: List[DailyState],
    config: IndicatorConfig,
) -> Dict[int, Tuple[Line, List[Label]]]:
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df_chi = df.tz_convert("America/Chicago")

    outputs: Dict[int, Tuple[Line, List[Label]]] = {}

    for idx in range(len(df_chi)):
        if not config.show_true_day_open:
            continue
        current = df_chi.index[idx]
        day_state = daily_states[idx]
        true_day_open_time = current.replace(hour=23, minute=0, second=0, microsecond=0)

        if current >= true_day_open_time:
            line = Line(
                x1=true_day_open_time,
                y1=day_state.day_high + 450 * config.mintick,
                x2=true_day_open_time,
                y2=day_state.day_low - 500 * config.mintick,
                color=config.true_day_open_color,
                width=config.true_day_open_width,
                style="dotted",
                kind="true_day_open",
            )
            if config.true_day_open_text_position == "Top":
                text_y = day_state.day_high + 200 * config.mintick
            elif config.true_day_open_text_position == "Bottom":
                text_y = day_state.day_low - 200 * config.mintick
            else:
                text_y = day_state.mid_level

            labels: List[Label] = []
            vertical_text = "True Day Open"
            for i, char in enumerate(vertical_text):
                labels.append(
                    Label(
                        x=true_day_open_time,
                        y=text_y - i * 4 * config.mintick,
                        text=char,
                        color=config.true_day_open_text_color,
                        size=config.true_day_open_text_size,
                        kind="true_day_open",
                    )
                )

            outputs[idx] = (line, labels)

    return outputs


def _day_index_utc(dt: datetime) -> int:
    start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    delta = dt - start
    return int(delta.total_seconds() // 86400) + 1


def _handle_session(
    df: pd.DataFrame,
    session_name: str,
    enabled: bool,
    start_hour: int,
    end_hour: int,
    session_color: str,
    config: IndicatorConfig,
    labeled_highs: Dict[float, Label],
    labeled_lows: Dict[float, Label],
) -> List[SessionLine]:
    if not enabled:
        return []

    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    df_ny = df.tz_convert("America/New_York")
    df_utc = df.tz_convert("UTC")

    high_val = None
    low_val = None
    high_idx = None
    low_idx = None
    high_line: Optional[Line] = None
    low_line: Optional[Line] = None
    high_end = None
    low_end = None
    high_label: Optional[Label] = None
    low_label: Optional[Label] = None
    prev_day = None

    session_lines: List[SessionLine] = []

    for idx in range(len(df_ny)):
        current = df_ny.index[idx]
        utc_dt = df_utc.index[idx]
        current_day = _day_index_utc(utc_dt)

        if prev_day is not None and current_day != prev_day:
            high_val = None
            low_val = None
            high_idx = None
            low_idx = None
            high_line = None
            low_line = None
            high_label = None
            low_label = None
            high_end = None
            low_end = None
        prev_day = current_day

        start_session = current.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        end_session = current.replace(
            hour=end_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
        if end_hour < start_hour:
            end_session = end_session + pd.Timedelta(days=1)

        in_session = start_session <= current < end_session
        was_in_session = False
        if idx > 0:
            prev_current = df_ny.index[idx - 1]
            prev_start = prev_current.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            prev_end = prev_current.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            if end_hour < start_hour:
                prev_end = prev_end + pd.Timedelta(days=1)
            was_in_session = prev_start <= prev_current < prev_end

        new_session = in_session and not was_in_session

        high = float(df_ny.iloc[idx]["high"])
        low = float(df_ny.iloc[idx]["low"])

        if new_session:
            high_val = high
            low_val = low
            high_idx = idx
            low_idx = idx
        elif in_session:
            if high_val is None or high > high_val:
                high_val = high
                high_idx = idx
            if low_val is None or low < low_val:
                low_val = low
                low_idx = idx

        if not in_session and was_in_session and high_val is not None and low_val is not None:
            high_line = Line(
                x1=high_idx,
                y1=high_val,
                x2=high_idx + 1,
                y2=high_val,
                color=session_color,
                width=1,
                style="dotted" if config.killzone_dotted_lines else "solid",
                kind="killzone_high",
                session_name=session_name,
            )
            low_line = Line(
                x1=low_idx,
                y1=low_val,
                x2=low_idx + 1,
                y2=low_val,
                color=session_color,
                width=1,
                style="dotted" if config.killzone_dotted_lines else "solid",
                kind="killzone_low",
                session_name=session_name,
            )
            high_end = high_idx + config.extend_bars
            low_end = low_idx + config.extend_bars

            if config.show_labels:
                if high_val in labeled_highs:
                    labeled_highs.pop(high_val)
                high_name = "NY" if session_name == "NY" else session_name
                high_label = Label(
                    x=high_idx + 10,
                    y=high_val,
                    text=f"{high_name} High",
                    color=config.session_label_text_color,
                    size="small",
                    kind="killzone_high",
                )
                labeled_highs[high_val] = high_label

                if low_val in labeled_lows:
                    labeled_lows.pop(low_val)
                low_name = "NY" if session_name == "NY" else session_name
                low_label = Label(
                    x=low_idx + 10,
                    y=low_val,
                    text=f"{low_name} Low",
                    color=config.session_label_text_color,
                    size="small",
                    kind="killzone_low",
                )
                labeled_lows[low_val] = low_label

            session_lines.append(SessionLine(line=high_line, label=high_label, session_high=True))
            session_lines.append(SessionLine(line=low_line, label=low_label, session_high=False))

        if high_line is not None and high_val is not None:
            if idx <= high_end:
                high_line.x2 = idx
            if high > high_val:
                high_end = idx
            if config.show_labels and high_label is not None:
                x1 = high_line.x1
                x2 = high_line.x2
                if config.label_align_opt == "Left":
                    new_x = x1 + 10
                elif config.label_align_opt == "Right":
                    new_x = x2 - 25
                else:
                    new_x = int(round((x1 + x2) / 2))
                high_label.x = new_x

        if low_line is not None and low_val is not None:
            if idx <= low_end:
                low_line.x2 = idx
            if low < low_val:
                low_end = idx
            if config.show_labels and low_label is not None:
                x1 = low_line.x1
                x2 = low_line.x2
                if config.label_align_opt == "Left":
                    new_x = x1 + 10
                elif config.label_align_opt == "Right":
                    new_x = x2 - 25
                else:
                    new_x = int(round((x1 + x2) / 2))
                low_label.x = new_x

        if high_line is not None and high_val is not None and high >= high_val:
            high_line.break_idx = idx
            high_line = None

        if low_line is not None and low_val is not None and low <= low_val:
            low_line.break_idx = idx
            low_line = None

    return session_lines


def build_killzone_levels(df: pd.DataFrame, config: IndicatorConfig) -> List[SessionLine]:
    labeled_highs: Dict[float, Label] = {}
    labeled_lows: Dict[float, Label] = {}
    session_lines: List[SessionLine] = []

    session_lines += _handle_session(
        df,
        session_name="Asia",
        enabled=config.enable_asia_session,
        start_hour=20,
        end_hour=0,
        session_color=config.line_color_asia,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )
    session_lines += _handle_session(
        df,
        session_name="London",
        enabled=config.enable_london_session,
        start_hour=2,
        end_hour=5,
        session_color=config.line_color_london,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )
    session_lines += _handle_session(
        df,
        session_name="NY",
        enabled=config.enable_ny_session,
        start_hour=9,
        end_hour=16,
        session_color=config.line_color_ny,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )
    session_lines += _handle_session(
        df,
        session_name="NY AM",
        enabled=config.enable_ny_am,
        start_hour=9,
        end_hour=11,
        session_color=config.line_color_ny_am,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )
    session_lines += _handle_session(
        df,
        session_name="NY Lunch",
        enabled=config.enable_ny_lunch,
        start_hour=11,
        end_hour=13,
        session_color=config.line_color_ny_lunch,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )
    session_lines += _handle_session(
        df,
        session_name="NY PM",
        enabled=config.enable_ny_pm,
        start_hour=13,
        end_hour=15,
        session_color=config.line_color_ny_pm,
        config=config,
        labeled_highs=labeled_highs,
        labeled_lows=labeled_lows,
    )

    return session_lines


def build_checklist_updates(df: pd.DataFrame, config: IndicatorConfig) -> List[ChecklistUpdate]:
    if not config.show_checklist:
        return []

    labels = [
        "Premium & Discount",
        "Major Liq Sweep",
        "Good Reaction",
        "Inside Killzone",
        "Good Momentum",
        "Clear DOL",
        "SMT",
        "Delivering from HTF Gap",
    ]
    statuses = [config.c1, config.c2, config.c3, config.c4, config.c5, config.c6, config.c7, config.c8]

    updates: List[ChecklistUpdate] = []
    for idx in range(len(df)):
        if idx % 5 != 0:
            continue
        rows = list(zip(labels, statuses))
        checked = sum(1 for status in statuses if status)
        summary = f"{checked}/{len(statuses)} Confluences"
        updates.append(ChecklistUpdate(bar_index=idx, rows=rows, summary=summary))

    return updates


def build_watermark(df: pd.DataFrame, config: IndicatorConfig) -> Optional[Watermark]:
    if not config.enable_watermark:
        return None
    if len(df) == 0:
        return None
    return Watermark(text_primary=config.watermark_text, text_secondary=config.watermark_text2)


def calculate_indicator(df: pd.DataFrame, config: Optional[IndicatorConfig] = None) -> Dict[str, object]:
    config = config or IndicatorConfig()

    daily_states = build_daily_states(df, config)
    true_day_open_states = build_true_day_open_states(df, daily_states, config)
    killzone_levels = build_killzone_levels(df, config)
    checklist_updates = build_checklist_updates(df, config)
    watermark = build_watermark(df, config)

    return {
        "daily_states": daily_states,
        "true_day_open_states": true_day_open_states,
        "killzone_levels": killzone_levels,
        "checklist_updates": checklist_updates,
        "watermark": watermark,
    }
