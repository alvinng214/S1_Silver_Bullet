"""SW's Asia/London H/L's - Pine translation.

Mirrors the Pine Script logic:
- Tracks Asia/London/NY/Sydney session highs/lows in LA time with optional offset.
- Freezes session ranges when the next session starts and anchors lines at the bar of the extreme.
- Builds Previous Day High/Low from a custom trading day (15:00 -> 14:00 LA time).
- Emits data structures for session lines, PDH/PDL/mid, and vertical line markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class SessionLine:
    session: str
    is_high: bool
    price: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    bar_of_extreme: pd.Timestamp
    frozen: bool


@dataclass
class SessionState:
    live_high: Optional[float] = None
    live_low: Optional[float] = None
    start_time: Optional[pd.Timestamp] = None
    high_time: Optional[pd.Timestamp] = None
    low_time: Optional[pd.Timestamp] = None
    frozen: bool = False


@dataclass
class PDLevels:
    high: Optional[float]
    low: Optional[float]
    high_time: Optional[pd.Timestamp]
    low_time: Optional[pd.Timestamp]
    mid: Optional[float]
    start_time: Optional[pd.Timestamp]


@dataclass
class VerticalLine:
    timestamp: pd.Timestamp
    label: str
    color: str
    thickness: int
    text_color: str
    opacity: int
    anchor_high: Optional[float] = None
    label_offset: float = 0.0


@dataclass
class CustomVerticalConfig:
    enabled: bool
    hour: int
    minute: int
    label: str
    color: str
    opacity: int
    thickness: int
    text_color: str


@dataclass
class CustomVerticalState:
    anchor_high: Optional[float] = None
    last_date: Optional[pd.Timestamp] = None


@dataclass
class AsiaLondonOutputs:
    session_lines: List[SessionLine]
    pd_levels: PDLevels
    vertical_lines: List[VerticalLine]
    custom_vertical_lines: List[VerticalLine]


def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def _session_window(
    ts: pd.Timestamp,
    start_hour: int,
    end_hour: int,
    tz_offset: int,
    start_minute: int = 0,
    end_minute: int = 0,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    cross_midnight = start_hour > end_hour or (start_hour == end_hour and start_minute > end_minute)
    base_date = ts.normalize()
    if cross_midnight and ts.hour < start_hour:
        start_date = base_date - pd.Timedelta(days=1)
    else:
        start_date = base_date
    end_date = start_date + pd.Timedelta(days=1) if cross_midnight else start_date

    start = start_date + pd.Timedelta(hours=start_hour + tz_offset, minutes=start_minute)
    end = end_date + pd.Timedelta(hours=end_hour + tz_offset, minutes=end_minute)
    return start, end


def _in_session(ts: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return start <= ts < end


def _maybe_freeze(
    state: SessionState,
    next_session_started: bool,
    session: str,
    end_time: pd.Timestamp,
    lines: List[SessionLine],
) -> None:
    if not next_session_started or state.frozen:
        return
    if state.live_high is None or state.live_low is None:
        return
    high_time = state.high_time or state.start_time
    low_time = state.low_time or state.start_time
    lines.append(
        SessionLine(
            session=session,
            is_high=True,
            price=state.live_high,
            start_time=high_time,
            end_time=end_time,
            bar_of_extreme=high_time,
            frozen=True,
        )
    )
    lines.append(
        SessionLine(
            session=session,
            is_high=False,
            price=state.live_low,
            start_time=low_time,
            end_time=end_time,
            bar_of_extreme=low_time,
            frozen=True,
        )
    )
    state.frozen = True


def calculate_asia_london_levels(
    df: pd.DataFrame,
    *,
    timezone_offset: int = 0,
    asia_enabled: bool = True,
    london_enabled: bool = True,
    ny_enabled: bool = True,
    sydney_enabled: bool = False,
    pdhl_enabled: bool = True,
    pd_mid_enabled: bool = False,
    vert_lines_enabled: bool = True,
    cv_label_offset: float = 100.0,
    custom_verticals: Optional[List[CustomVerticalConfig]] = None,
) -> AsiaLondonOutputs:
    df = _ensure_index(df)

    session_lines: List[SessionLine] = []

    asia = SessionState()
    london = SessionState()
    ny = SessionState()
    sydney = SessionState()

    prev_session_high: Optional[float] = None
    prev_session_low: Optional[float] = None
    prev_session_high_time: Optional[pd.Timestamp] = None
    prev_session_low_time: Optional[pd.Timestamp] = None

    session_high: Optional[float] = None
    session_low: Optional[float] = None
    session_high_time: Optional[pd.Timestamp] = None
    session_low_time: Optional[pd.Timestamp] = None
    pd_start_time: Optional[pd.Timestamp] = None

    vertical_lines: List[VerticalLine] = []
    custom_vertical_lines: List[VerticalLine] = []

    if custom_verticals is None:
        custom_verticals = []

    custom_states: Dict[int, CustomVerticalState] = {
        idx: CustomVerticalState() for idx in range(len(custom_verticals))
    }

    prev_flags = {
        "asia": False,
        "london": False,
        "ny": False,
        "sydney": False,
    }

    for idx, ts in enumerate(df.index):
        asia_start, asia_end = _session_window(ts, 17, 2, timezone_offset)
        london_start, london_end = _session_window(ts, 0, 9, timezone_offset)
        ny_start, ny_end = _session_window(ts, 5, 14, timezone_offset, start_minute=30)
        sydney_start, sydney_end = _session_window(ts, 15, 0, timezone_offset)

        asia_session = asia_enabled and _in_session(ts, asia_start, asia_end)
        london_session = london_enabled and _in_session(ts, london_start, london_end)
        ny_session = ny_enabled and _in_session(ts, ny_start, ny_end)
        sydney_session = sydney_enabled and _in_session(ts, sydney_start, sydney_end)

        asia_started = asia_session and not prev_flags["asia"]
        london_started = london_session and not prev_flags["london"]
        ny_started = ny_session and not prev_flags["ny"]
        sydney_started = sydney_session and not prev_flags["sydney"]

        _maybe_freeze(asia, london_started, "Asia", ts, session_lines)
        _maybe_freeze(london, ny_started, "London", ts, session_lines)
        _maybe_freeze(ny, sydney_started, "NY", ts, session_lines)
        _maybe_freeze(sydney, asia_started, "Sydney", ts, session_lines)

        if asia_started:
            asia.frozen = False
            asia.start_time = ts
            asia.live_high = df["high"].iloc[idx]
            asia.live_low = df["low"].iloc[idx]
            asia.high_time = ts
            asia.low_time = ts
        if london_started:
            london.frozen = False
            london.start_time = ts
            london.live_high = df["high"].iloc[idx]
            london.live_low = df["low"].iloc[idx]
            london.high_time = ts
            london.low_time = ts
        if ny_started:
            ny.frozen = False
            ny.start_time = ts
            ny.live_high = df["high"].iloc[idx]
            ny.live_low = df["low"].iloc[idx]
            ny.high_time = ts
            ny.low_time = ts
        if sydney_started:
            sydney.frozen = False
            sydney.start_time = ts
            sydney.live_high = df["high"].iloc[idx]
            sydney.live_low = df["low"].iloc[idx]
            sydney.high_time = ts
            sydney.low_time = ts

        if asia_session and not asia.frozen and asia.live_high is not None:
            if df["high"].iloc[idx] > asia.live_high:
                asia.live_high = df["high"].iloc[idx]
                asia.high_time = ts
            if df["low"].iloc[idx] < asia.live_low:
                asia.live_low = df["low"].iloc[idx]
                asia.low_time = ts
        if london_session and not london.frozen and london.live_high is not None:
            if df["high"].iloc[idx] > london.live_high:
                london.live_high = df["high"].iloc[idx]
                london.high_time = ts
            if df["low"].iloc[idx] < london.live_low:
                london.live_low = df["low"].iloc[idx]
                london.low_time = ts
        if ny_session and not ny.frozen and ny.live_high is not None:
            if df["high"].iloc[idx] > ny.live_high:
                ny.live_high = df["high"].iloc[idx]
                ny.high_time = ts
            if df["low"].iloc[idx] < ny.live_low:
                ny.live_low = df["low"].iloc[idx]
                ny.low_time = ts
        if sydney_session and not sydney.frozen and sydney.live_high is not None:
            if df["high"].iloc[idx] > sydney.live_high:
                sydney.live_high = df["high"].iloc[idx]
                sydney.high_time = ts
            if df["low"].iloc[idx] < sydney.live_low:
                sydney.live_low = df["low"].iloc[idx]
                sydney.low_time = ts

        sess_start, sess_end = _session_window(ts, 15, 14, timezone_offset)
        new_session_start = idx > 0 and df.index[idx - 1] < sess_start <= ts
        if new_session_start:
            prev_session_high = session_high
            prev_session_low = session_low
            prev_session_high_time = session_high_time
            prev_session_low_time = session_low_time
            session_high = df["high"].iloc[idx]
            session_low = df["low"].iloc[idx]
            session_high_time = ts
            session_low_time = ts
            pd_start_time = sess_start - pd.Timedelta(days=1)
        else:
            if sess_start <= ts < sess_end:
                if session_high is None or df["high"].iloc[idx] > session_high:
                    session_high = df["high"].iloc[idx]
                    session_high_time = ts
                if session_low is None or df["low"].iloc[idx] < session_low:
                    session_low = df["low"].iloc[idx]
                    session_low_time = ts

        prev_flags = {
            "asia": asia_session,
            "london": london_session,
            "ny": ny_session,
            "sydney": sydney_session,
        }

        for idx_cfg, config in enumerate(custom_verticals):
            if not config.enabled:
                custom_states[idx_cfg] = CustomVerticalState()
                continue
            base_date = ts.normalize()
            if custom_states[idx_cfg].last_date != base_date:
                custom_states[idx_cfg] = CustomVerticalState(last_date=base_date)
            ts_custom = base_date + pd.Timedelta(
                hours=config.hour + timezone_offset, minutes=config.minute
            )
            prev_ts = df.index[idx - 1] if idx > 0 else ts
            if prev_ts < ts_custom <= ts:
                custom_states[idx_cfg].anchor_high = df["high"].iloc[idx]
            anchor_high = custom_states[idx_cfg].anchor_high
            custom_vertical_lines.append(
                VerticalLine(
                    timestamp=ts_custom,
                    label=config.label,
                    color=config.color,
                    thickness=config.thickness,
                    text_color=config.text_color,
                    opacity=config.opacity,
                    anchor_high=anchor_high,
                    label_offset=cv_label_offset,
                )
            )

    pd_mid = None
    if pdhl_enabled and prev_session_high is not None and prev_session_low is not None:
        pd_mid = (prev_session_high + prev_session_low) / 2

    pd_levels = PDLevels(
        high=prev_session_high if pdhl_enabled else None,
        low=prev_session_low if pdhl_enabled else None,
        high_time=prev_session_high_time,
        low_time=prev_session_low_time,
        mid=pd_mid if pd_mid_enabled else None,
        start_time=pd_start_time if pdhl_enabled else None,
    )

    last_ts = df.index[-1] if not df.empty else pd.Timestamp.min
    for line in session_lines:
        if line.frozen:
            line.end_time = last_ts
    def _add_live(state: SessionState, name: str) -> None:
        if state.frozen or state.live_high is None or state.live_low is None or state.start_time is None:
            return
        high_time = state.high_time or state.start_time
        low_time = state.low_time or state.start_time
        session_lines.append(
            SessionLine(
                session=name,
                is_high=True,
                price=state.live_high,
                start_time=state.start_time,
                end_time=last_ts,
                bar_of_extreme=high_time,
                frozen=False,
            )
        )
        session_lines.append(
            SessionLine(
                session=name,
                is_high=False,
                price=state.live_low,
                start_time=state.start_time,
                end_time=last_ts,
                bar_of_extreme=low_time,
                frozen=False,
            )
        )

    _add_live(asia, "Asia")
    _add_live(london, "London")
    _add_live(ny, "NY")
    _add_live(sydney, "Sydney")

    if vert_lines_enabled and not df.empty:
        unique_dates = pd.Series(df.index.normalize()).unique()

        def _vl(base_date: pd.Timestamp, hour: int, minute: int, label: str, color: str) -> None:
            ts = base_date + pd.Timedelta(hours=hour + timezone_offset, minutes=minute)
            vertical_lines.append(
                VerticalLine(
                    timestamp=ts,
                    label=label,
                    color=color,
                    thickness=2,
                    text_color="white",
                    opacity=70,
                )
            )

        for base_date in unique_dates:
            _vl(base_date, 5, 30, "Pre-market", "#ffa726")
            _vl(base_date, 6, 30, "Market Open", "#f57c00")
            _vl(base_date, 14, 0, "Market Close", "#801922")

    return AsiaLondonOutputs(
        session_lines=session_lines,
        pd_levels=pd_levels,
        vertical_lines=vertical_lines,
        custom_vertical_lines=custom_vertical_lines,
    )
