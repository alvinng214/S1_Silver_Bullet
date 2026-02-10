"""Line-by-line parity translation of `ICT MTF Order Block Wicks [MK].txt` (Pine v5).

This module mirrors the Pine indicator logic as faithfully as practical in Python,
including script quirks/bugs and alert flag composition.
"""

from __future__ import annotations

from datetime import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# -----------------------------
# Pine-like runtime structures
# -----------------------------
@dataclass
class OBLabel:
    x: int
    y: float
    text: str
    text_color: str = "orange"


@dataclass
class OBBox:
    left: int
    right: int
    top: float
    bottom: float
    bgcolor: str
    border_color: str
    border_width: int = 1
    border_style: str = "dotted"


@dataclass
class TimeframeState:
    bull_boxes: List[OBBox] = field(default_factory=list)
    bear_boxes: List[OBBox] = field(default_factory=list)
    bull_labels: List[OBLabel] = field(default_factory=list)
    bear_labels: List[OBLabel] = field(default_factory=list)
    new_bull: bool = False
    new_bear: bool = False


@dataclass
class Settings:
    # Core toggles
    display: bool = True
    only_market_hours: bool = False  # OnlyMktHrs
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    session_timezone: Optional[str] = None
    enforce_session_timezone: bool = True
    security_merge_policy: str = "tv_like_developing"
    fvgmethod_body: bool = True      # fmthds = "Body"
    show_labels: bool = True
    show_timeonlabels: bool = False

    # Label settings
    hours_offset_input: float = -5.0
    label_shift: int = 10
    label_color: str = "orange"

    # Intrusion / mitigation
    incursion_alerts: bool = True
    incursion_pct: int = 20
    mitigation_mode: str = "Normal"   # Normal|Dynamic|None|Half
    show_mitigated_text: bool = False
    use_body_for_mitigation: bool = False  # mitig_type default "Wicks"

    # Colors
    bullfvgcolor: str = "yellow"
    bearfvgcolor: str = "blue"
    entrychangecolor: bool = True
    entry_bull_color: str = "white"
    entry_bear_color: str = "white"
    nomiticolor: str = "yellow"

    # TF enable flags
    enable_current_timeframe: bool = False
    enable_5min: bool = False
    enable_10min: bool = True
    enable_15min: bool = True
    enable_30min: bool = True
    enable_1hr: bool = True
    enable_4hr: bool = False
    enable_8hr: bool = False
    enable_12hr: bool = False
    enable_daily: bool = False
    enable_week: bool = False
    enable_month: bool = False

    # Per-TF max sizes
    curr_max_array_size: int = 8
    max_5min: int = 8
    max_10min: int = 8
    max_15min: int = 8
    max_30min: int = 8
    max_1hr: int = 8
    max_4hr: int = 8
    max_8hr: int = 8
    max_12hr: int = 8
    max_daily: int = 8
    max_weekly: int = 8
    max_monthly: int = 8


@dataclass
class Result:
    states: Dict[str, TimeframeState]
    bull_fvg_creation_alert: bool
    bear_fvg_creation_alert: bool
    both_fvg_creation_alert: bool
    incursion_messages: List[str]
    is_error: bool
    bar_events: List[dict] = field(default_factory=list)


def _mitigationaction(mode: str) -> int:
    if mode == "Normal":
        return 1
    if mode == "Dynamic":
        return 2
    if mode == "None":
        return 3
    return 4


def _gettf_interval_str(index: int) -> str:
    return {
        0: "5", 1: "10", 2: "15", 3: "30", 4: "60", 5: "240",
        6: "480", 7: "720", 8: "D", 9: "W", 10: "M",
    }.get(index, "Unsupported Timeframe")


def _gettf_label_str(index: int) -> str:
    return {
        0: "5 Min", 1: "10 Min", 2: "15 Min", 3: "30 Min", 4: "1 Hr", 5: "4 Hr",
        6: "8 Hr", 7: "12Hr", 8: "Daily", 9: "Weekly", 10: "Monthly",
    }.get(index, "Unsupported Timeframe")


def _timestring_from_current_tf(current_timeframe: str) -> str:
    if current_timeframe.isdigit():
        n = int(current_timeframe)
        if n > 59:
            return f"{n / 60:g} Hr"
        return f"{current_timeframe} Min"
    if current_timeframe == "D":
        return "Daily"
    if current_timeframe == "W":
        return "Weekly"
    return "Monthly"


def _not_current_timeframe_equal_enabled_tfs(current_tf: str, s: Settings) -> bool:
    return {
        "5": not s.enable_5min,
        "10": not s.enable_10min,
        "15": not s.enable_15min,
        "30": not s.enable_30min,
        "60": not s.enable_1hr,
        "240": not s.enable_4hr,
        "480": not s.enable_8hr,
        "720": not s.enable_12hr,
        "D": not s.enable_daily,
        "W": not s.enable_week,
        "M": not s.enable_month,
    }.get(current_tf, True)


def _period_start_index(idx: pd.DatetimeIndex, period: str) -> pd.Series:
    if period in {"5", "10", "15", "30", "60", "240", "480", "720"}:
        minutes = int(period)
        return pd.Series(idx.floor(f"{minutes}min"), index=idx)
    if period == "D":
        return pd.Series(idx.floor("D"), index=idx)
    if period == "W":
        # Weekly start (Monday) for stable grouping.
        return pd.Series(idx.to_period("W-SUN").start_time, index=idx)
    if period == "M":
        return pd.Series(idx.to_period("M").start_time, index=idx)
    raise ValueError(f"Unsupported period: {period}")


def _security_context(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Mirror request.security outputs used by Pine call:
    [_open1,_close1,_open,_close,_high1,_low1]

    - _open/_close correspond to current HTF bar open/developing close
    - *_1 correspond to previous fully closed HTF bar values
    """
    grp = _period_start_index(df.index, period)
    work = df.copy()
    work["_grp"] = grp.values

    # Current HTF bar fields (open is group-first, close is developing current close).
    curr_open = work.groupby("_grp")["open"].transform("first")
    curr_close = work["close"]

    # Completed HTF OHLC per group, then shift one HTF bar back.
    g_ohlc = work.groupby("_grp").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    g_prev = g_ohlc.shift(1)

    prev_open = grp.map(g_prev["open"])
    prev_close = grp.map(g_prev["close"])
    prev_high = grp.map(g_prev["high"])
    prev_low = grp.map(g_prev["low"])

    return pd.DataFrame(
        {
            "open1": prev_open.astype(float),
            "close1": prev_close.astype(float),
            "open0": curr_open.astype(float),
            "close0": curr_close.astype(float),
            "high1": prev_high.astype(float),
            "low1": prev_low.astype(float),
        },
        index=df.index,
    )


def _isfvgbull(display: bool, fvgmethod: bool, open1: float, close1: float, op: float, cl: float, high1: float) -> bool:
    if not display:
        return False
    if fvgmethod:
        return open1 > close1 and op < cl and cl > high1
    return op < high1


def _isfvgbear(display: bool, fvgmethod: bool, open1: float, close1: float, op: float, cl: float, low1: float) -> bool:
    if not display:
        return False
    if fvgmethod:
        return open1 < close1 and op > cl and cl < low1
    return op < low1


def _duplicate_box(boxes: List[OBBox], top: float) -> bool:
    # Pine only compares top, bottom commented out.
    return any(b.top == top for b in boxes)


def _in_session(ts: pd.Timestamp, s: Settings) -> bool:
    # Pine: not na(time(timeframe.period, "0930-1600"))
    if s.session_timezone:
        if ts.tzinfo is None:
            ts = ts.tz_localize(s.session_timezone)
        else:
            ts = ts.tz_convert(s.session_timezone)

    t = ts.time()
    return s.session_start <= t <= s.session_end


def _resolve_session_timezone(df: pd.DataFrame, s: Settings) -> Optional[str]:
    if s.session_timezone:
        return s.session_timezone
    tz_from_attrs = df.attrs.get("timezone")
    if isinstance(tz_from_attrs, str) and tz_from_attrs:
        return tz_from_attrs
    if df.index.tz is not None:
        return str(df.index.tz)
    return None


def _security_context_with_policy(df: pd.DataFrame, period: str, policy: str) -> pd.DataFrame:
    if policy != "tv_like_developing":
        raise ValueError("Unsupported security_merge_policy. Use 'tv_like_developing'.")
    return _security_context(df, period)


def run_mtf_order_block_wicks(
    df: pd.DataFrame,
    settings: Optional[Settings] = None,
    current_timeframe: str = "15",
) -> Result:
    s = settings or Settings()

    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input DataFrame must include columns: {sorted(required)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Input DataFrame index must be monotonic increasing")

    resolved_session_timezone = _resolve_session_timezone(df, s)
    if s.only_market_hours and s.enforce_session_timezone and not resolved_session_timezone:
        raise ValueError(
            "Session timezone required for strict market-hours parity. Set settings.session_timezone "
            "or df.attrs['timezone'], or use a tz-aware DatetimeIndex."
        )
    if resolved_session_timezone:
        s = Settings(**{**s.__dict__, "session_timezone": resolved_session_timezone})

    # Pine bug/behavior parity: 30m/12h intentionally excluded from total sum expression.
    tot_user_max_boxes = (
        s.curr_max_array_size
        + s.max_5min
        + s.max_10min
        + s.max_15min
        + s.max_1hr
        + s.max_4hr
        + s.max_8hr
        + s.max_daily
        + s.max_weekly
        + s.max_monthly
    )
    is_error = tot_user_max_boxes > 500

    intrusion_percentage = s.incursion_pct / 100.0
    mitigationaction = _mitigationaction(s.mitigation_mode)
    last_bar_index = len(df) - 1

    tfs = ["chart", "5", "10", "15", "30", "60", "240", "480", "720", "D", "W", "M"]
    states: Dict[str, TimeframeState] = {k: TimeframeState() for k in tfs}
    incursion_messages: List[str] = []
    bar_events: List[dict] = []
    last_bar_flags: Dict[str, Dict[str, bool]] = {k: {"new_bull": False, "new_bear": False} for k in tfs}

    def _addlabel(labels: List[OBLabel], bar_index: int, ts: pd.Timestamp, string_: str, y: float) -> None:
        if not s.show_labels:
            return
        text = string_
        if s.show_timeonlabels:
            shifted = ts + pd.Timedelta(hours=s.hours_offset_input)
            text = f"{string_}  {shifted:%H:%M %m/%d/%y}"
        labels.append(OBLabel(x=bar_index + s.label_shift, y=y, text=text, text_color=s.label_color))

    def _update_bull_fvgs(st: TimeframeState, timestring: str, bar_index: int, low: float, close: float, lastlow: float) -> None:
        for i in range(len(st.bull_boxes) - 1, -1, -1):
            bx = st.bull_boxes[i]
            top, bottom = bx.top, bx.bottom

            midpt = (top + bottom) / 2
            threshold = top - (intrusion_percentage * (top - bottom))
            have_intrusion = low < threshold and lastlow > threshold

            lowundertop = low < top
            lowunderbtm = low < bottom
            lowundermid = low < midpt
            closeundertop = close < top
            closeunderbtm = close < bottom
            closeundermid = low < midpt  # parity with Pine typo

            if (mitigationaction in (1, 3)) and have_intrusion and s.incursion_alerts:
                incursion_messages.append(f"Bull OB Wick Incursion {timestring}")

            if s.entrychangecolor and lowundertop:
                bx.bgcolor = s.entry_bull_color

            if s.show_labels:
                if i < len(st.bull_labels):
                    st.bull_labels[i].x = bar_index + s.label_shift
                    st.bull_labels[i].y = (bx.top + bx.bottom) / 2
                bx.left = last_bar_index + s.label_shift

            if mitigationaction == 2 and s.use_body_for_mitigation and closeundertop:
                bx.top = close
                if s.show_labels and i < len(st.bull_labels):
                    st.bull_labels[i].y = (close + bottom) / 2
            elif mitigationaction == 2 and lowundertop:
                bx.top = low
                if s.show_labels and i < len(st.bull_labels):
                    st.bull_labels[i].y = (low + bottom) / 2

            if mitigationaction == 3:
                if (s.use_body_for_mitigation and closeunderbtm) or lowunderbtm:
                    bx.bgcolor = s.nomiticolor
                    if s.show_labels and s.show_mitigated_text and i < len(st.bull_labels):
                        if "Mitigated" not in st.bull_labels[i].text:
                            st.bull_labels[i].text += " Mitigated"

            delete_now = False
            if mitigationaction in (1, 2):
                delete_now = (s.use_body_for_mitigation and closeunderbtm) or lowunderbtm
            if mitigationaction == 4:
                delete_now = (s.use_body_for_mitigation and closeundermid) or lowundermid

            if delete_now:
                st.bull_boxes.pop(i)
                if s.show_labels and i < len(st.bull_labels):
                    st.bull_labels.pop(i)

    def _update_bear_fvgs(st: TimeframeState, timestring: str, bar_index: int, high: float, close: float, lasthigh: float) -> None:
        for i in range(len(st.bear_boxes) - 1, -1, -1):
            bx = st.bear_boxes[i]
            top, bottom = bx.top, bx.bottom

            midpt = (top + bottom) / 2
            threshold = bottom + (intrusion_percentage * (top - bottom))
            have_intrusion = high > threshold and lasthigh < threshold

            highovertop = high > top
            highoverbtm = high > bottom
            highovermid = high > midpt
            closeovertop = close > top
            closeoverbtm = close > bottom
            closeovermid = close > midpt

            if (mitigationaction in (1, 3)) and have_intrusion and s.incursion_alerts:
                incursion_messages.append(f"Bear OB Wick Incursion {timestring}")

            if s.entrychangecolor:
                if highoverbtm:
                    bx.bgcolor = s.entry_bear_color
                else:
                    bx.bgcolor = s.bearfvgcolor

            if s.show_labels:
                if i < len(st.bear_labels):
                    st.bear_labels[i].x = bar_index + s.label_shift
                    st.bear_labels[i].y = (bx.top + bx.bottom) / 2
                bx.left = last_bar_index + s.label_shift

            if mitigationaction == 2 and s.use_body_for_mitigation and closeoverbtm:
                bx.bottom = close
                if s.show_labels and i < len(st.bear_labels):
                    st.bear_labels[i].y = (close + bottom) / 2
            elif mitigationaction == 2 and highoverbtm:
                bx.bottom = high
                if s.show_labels and i < len(st.bear_labels):
                    st.bear_labels[i].y = (top + high) / 2

            if mitigationaction == 3:
                if (s.use_body_for_mitigation and closeovertop) or highovertop:
                    bx.bgcolor = s.nomiticolor
                    if s.show_labels and s.show_mitigated_text and i < len(st.bear_labels):
                        if "Mitigated" not in st.bear_labels[i].text:
                            st.bear_labels[i].text += " Mitigated"

            delete_now = False
            if mitigationaction in (1, 2):
                delete_now = (s.use_body_for_mitigation and closeovertop) or highovertop
            if mitigationaction == 4:
                delete_now = (s.use_body_for_mitigation and closeovermid) or highovermid

            if delete_now:
                st.bear_boxes.pop(i)
                if s.show_labels and i < len(st.bear_labels):
                    st.bear_labels.pop(i)

    def _handle_all(
        tf_key: str,
        tstring: str,
        maxarraysize: int,
        ctx: pd.DataFrame,
    ) -> None:
        st = states[tf_key]
        st.new_bull = False
        st.new_bear = False

        high1_prev_shift = ctx["high1"].shift(1)

        for i in range(1, len(df)):
            ts = df.index[i]
            in_session = _in_session(ts, s)
            gate_ok = (s.only_market_hours and in_session) or (not s.only_market_hours)
            if not gate_ok:
                st.new_bull = False
                st.new_bear = False
                last_bar_flags[tf_key]["new_bull"] = False
                last_bar_flags[tf_key]["new_bear"] = False
                continue

            open1 = float(ctx["open1"].iloc[i]) if pd.notna(ctx["open1"].iloc[i]) else float("nan")
            close1 = float(ctx["close1"].iloc[i]) if pd.notna(ctx["close1"].iloc[i]) else float("nan")
            op = float(ctx["open0"].iloc[i])
            cl = float(ctx["close0"].iloc[i])
            high1 = float(ctx["high1"].iloc[i]) if pd.notna(ctx["high1"].iloc[i]) else float("nan")
            low1 = float(ctx["low1"].iloc[i]) if pd.notna(ctx["low1"].iloc[i]) else float("nan")

            if pd.isna(open1) or pd.isna(close1) or pd.isna(high1) or pd.isna(low1):
                st.new_bull = False
                st.new_bear = False
                last_bar_flags[tf_key]["new_bull"] = False
                last_bar_flags[tf_key]["new_bear"] = False
                continue

            new_bull = _isfvgbull(s.display, s.fvgmethod_body, open1, close1, op, cl, high1)
            new_bear = _isfvgbear(s.display, s.fvgmethod_body, open1, close1, op, cl, low1)
            st.new_bull = bool(new_bull)
            st.new_bear = bool(new_bear)
            last_bar_flags[tf_key]["new_bull"] = bool(new_bull)
            last_bar_flags[tf_key]["new_bear"] = bool(new_bear)

            if new_bull:
                st.new_bull = True
                if len(st.bull_boxes) > maxarraysize:
                    st.bull_boxes.pop(0)
                    if s.show_labels and st.bull_labels:
                        st.bull_labels.pop(0)

                if not _duplicate_box(st.bull_boxes, high1):
                    st.bull_boxes.append(
                        OBBox(
                            left=last_bar_index + 20,
                            right=last_bar_index + 200,
                            top=high1,
                            bottom=open1,
                            bgcolor=s.bullfvgcolor,
                            border_color="yellow",
                        )
                    )
                    # Pine uses (_high1[1] + _low1) / 2 in label call.
                    high1_prev = float(high1_prev_shift.iloc[i]) if pd.notna(high1_prev_shift.iloc[i]) else high1
                    _addlabel(st.bull_labels, i, ts, f"{tstring} OB BULL", (high1_prev + low1) / 2)

            if new_bear:
                st.new_bear = True
                if len(st.bear_boxes) > maxarraysize:
                    st.bear_boxes.pop(0)
                    if s.show_labels and st.bear_labels:
                        st.bear_labels.pop(0)

                if not _duplicate_box(st.bear_boxes, open1):
                    st.bear_boxes.append(
                        OBBox(
                            left=last_bar_index + 20,
                            right=last_bar_index + 200,
                            top=open1,
                            bottom=low1,
                            bgcolor=s.bearfvgcolor,
                            border_color="blue",
                        )
                    )
                    high1_prev = float(high1_prev_shift.iloc[i]) if pd.notna(high1_prev_shift.iloc[i]) else high1
                    _addlabel(st.bear_labels, i, ts, f"{tstring} OB BEAR", (high1_prev + low1) / 2)

            if st.bull_boxes:
                _update_bull_fvgs(
                    st,
                    tstring,
                    i,
                    float(df["low"].iloc[i]),
                    float(df["close"].iloc[i]),
                    float(df["low"].iloc[i - 1]),
                )

            if st.bear_boxes:
                _update_bear_fvgs(
                    st,
                    tstring,
                    i,
                    float(df["high"].iloc[i]),
                    float(df["close"].iloc[i]),
                    float(df["high"].iloc[i - 1]),
                )

            if new_bull or new_bear:
                bar_events.append(
                    {
                        "bar_index": i,
                        "timestamp": ts,
                        "timeframe": tf_key,
                        "timeframe_label": tstring,
                        "new_bull": bool(new_bull),
                        "new_bear": bool(new_bear),
                    }
                )

    currtfstring = _timestring_from_current_tf(current_timeframe)

    if s.enable_current_timeframe and _not_current_timeframe_equal_enabled_tfs(current_timeframe, s):
        _handle_all("chart", currtfstring, s.curr_max_array_size, _security_context_with_policy(df, current_timeframe, s.security_merge_policy))

    if s.enable_5min:
        _handle_all("5", _gettf_label_str(0), s.max_5min, _security_context_with_policy(df, _gettf_interval_str(0), s.security_merge_policy))
    if s.enable_10min:
        _handle_all("10", _gettf_label_str(1), s.max_10min, _security_context_with_policy(df, _gettf_interval_str(1), s.security_merge_policy))
    if s.enable_15min:
        _handle_all("15", _gettf_label_str(2), s.max_15min, _security_context_with_policy(df, _gettf_interval_str(2), s.security_merge_policy))
    if s.enable_30min:
        _handle_all("30", _gettf_label_str(3), s.max_30min, _security_context_with_policy(df, _gettf_interval_str(3), s.security_merge_policy))
    if s.enable_1hr:
        _handle_all("60", _gettf_label_str(4), s.max_1hr, _security_context_with_policy(df, _gettf_interval_str(4), s.security_merge_policy))
    if s.enable_4hr:
        _handle_all("240", _gettf_label_str(5), s.max_4hr, _security_context_with_policy(df, _gettf_interval_str(5), s.security_merge_policy))
    if s.enable_8hr:
        _handle_all("480", _gettf_label_str(6), s.max_8hr, _security_context_with_policy(df, _gettf_interval_str(6), s.security_merge_policy))
    if s.enable_12hr:
        _handle_all("720", _gettf_label_str(7), s.max_12hr, _security_context_with_policy(df, _gettf_interval_str(7), s.security_merge_policy))
    if s.enable_daily:
        _handle_all("D", _gettf_label_str(8), s.max_daily, _security_context_with_policy(df, _gettf_interval_str(8), s.security_merge_policy))
    if s.enable_week:
        _handle_all("W", _gettf_label_str(9), s.max_weekly, _security_context_with_policy(df, _gettf_interval_str(9), s.security_merge_policy))
    if s.enable_month:
        _handle_all("M", _gettf_label_str(10), s.max_monthly, _security_context_with_policy(df, _gettf_interval_str(10), s.security_merge_policy))

    # Pine alert expressions explicitly omit 12hr flags from aggregate expressions.
    bull_fvg_creation_alert = (
        last_bar_flags["chart"]["new_bull"]
        or last_bar_flags["5"]["new_bull"]
        or last_bar_flags["10"]["new_bull"]
        or last_bar_flags["15"]["new_bull"]
        or last_bar_flags["60"]["new_bull"]
        or last_bar_flags["240"]["new_bull"]
        or last_bar_flags["480"]["new_bull"]
        or last_bar_flags["D"]["new_bull"]
        or last_bar_flags["W"]["new_bull"]
        or last_bar_flags["M"]["new_bull"]
    )
    bear_fvg_creation_alert = (
        last_bar_flags["chart"]["new_bear"]
        or last_bar_flags["5"]["new_bear"]
        or last_bar_flags["10"]["new_bear"]
        or last_bar_flags["15"]["new_bear"]
        or last_bar_flags["60"]["new_bear"]
        or last_bar_flags["240"]["new_bear"]
        or last_bar_flags["480"]["new_bear"]
        or last_bar_flags["D"]["new_bear"]
        or last_bar_flags["W"]["new_bear"]
        or last_bar_flags["M"]["new_bear"]
    )
    both_fvg_creation_alert = bull_fvg_creation_alert or bear_fvg_creation_alert

    return Result(
        states=states,
        bull_fvg_creation_alert=bull_fvg_creation_alert,
        bear_fvg_creation_alert=bear_fvg_creation_alert,
        both_fvg_creation_alert=both_fvg_creation_alert,
        incursion_messages=incursion_messages,
        is_error=is_error,
        bar_events=bar_events,
    )


def run_mtf_order_block_wicks_from_records(
    records: List[dict],
    settings: Optional[Settings] = None,
    current_timeframe: str = "15",
) -> Result:
    """Backtrader-friendly wrapper.

    Accepts a list of dict records with keys: `datetime`, `open`, `high`, `low`, `close`.
    This keeps the core Pine-parity logic intact while providing a feed-neutral entrypoint
    commonly used when exporting bars inside Backtrader strategies/analyzers.
    """
    df = pd.DataFrame.from_records(records)
    if "datetime" not in df.columns:
        raise ValueError("records must include a 'datetime' key")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    return run_mtf_order_block_wicks(df[["open", "high", "low", "close"]], settings, current_timeframe)


def run_mtf_order_block_wicks_from_backtrader(
    datetimes: List[object],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    settings: Optional[Settings] = None,
    current_timeframe: str = "15",
) -> Result:
    """Convenience bridge for Backtrader line buffers.

    Typical use in Backtrader:
    - collect `bt.num2date(data.datetime[0])`, `data.open[0]`, `data.high[0]`,
      `data.low[0]`, `data.close[0]` into Python lists during `next()`
    - call this function in `stop()` or analyzer finalization
    """
    if not (len(datetimes) == len(opens) == len(highs) == len(lows) == len(closes)):
        raise ValueError("All input arrays must have equal length")
    records = [
        {
            "datetime": datetimes[i],
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
        }
        for i in range(len(datetimes))
    ]
    return run_mtf_order_block_wicks_from_records(records, settings, current_timeframe)
