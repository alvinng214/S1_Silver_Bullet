"""Python translation of TradingView Pine Script: ICT MTF Order Block Wicks [MK].

This module mirrors the source script's logic (box creation, update, mitigation,
duplicate filtering, max-array limits, MTF dispatch, and alert booleans) using
plain Python data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------
# Pine-like settings/constants
# ---------------------------
@dataclass
class MKSettings:
    display: bool = True
    only_market_hours: bool = False
    fvgmethod_body: bool = True
    show_labels: bool = True
    show_timeonlabels: bool = False
    hours_offset_input: float = -5.0
    label_shift: int = 10
    incursion_alerts: bool = True
    incursion_pct: int = 20

    mitigation_mode: str = "Normal"  # Normal | Dynamic | None | Half
    show_mitigated_text: bool = False
    use_body_for_mitigation: bool = False

    entry_change_color: bool = True
    bull_color: str = "yellow"
    bear_color: str = "blue"
    entry_bull_color: str = "white"
    entry_bear_color: str = "white"
    no_mitigation_color: str = "yellow"

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

    curr_max_array_size: int = 8
    max_5: int = 8
    max_10: int = 8
    max_15: int = 8
    max_30: int = 8
    max_60: int = 8
    max_240: int = 8
    max_480: int = 8
    max_720: int = 8
    max_d: int = 8
    max_w: int = 8
    max_m: int = 8


@dataclass
class MKLabel:
    x: int
    y: float
    text: str
    color: str = "orange"


@dataclass
class MKBox:
    left: int
    right: int
    top: float
    bottom: float
    bgcolor: str
    border_color: str
    timeframe: str
    direction: str  # bull|bear


@dataclass
class TFState:
    bull_boxes: List[MKBox] = field(default_factory=list)
    bear_boxes: List[MKBox] = field(default_factory=list)
    bull_labels: List[MKLabel] = field(default_factory=list)
    bear_labels: List[MKLabel] = field(default_factory=list)
    new_bull: bool = False
    new_bear: bool = False


@dataclass
class MKResult:
    states: Dict[str, TFState]
    alerts: Dict[str, bool]
    incursion_alert_messages: List[str]
    is_error: bool


def _mitigation_action(mode: str) -> int:
    return {"Normal": 1, "Dynamic": 2, "None": 3}.get(mode, 4)


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


def _not_current_timeframe_equal_enabled_tfs(current_tf: str, s: MKSettings) -> bool:
    enabled_map = {
        "5": s.enable_5min,
        "10": s.enable_10min,
        "15": s.enable_15min,
        "30": s.enable_30min,
        "60": s.enable_1hr,
        "240": s.enable_4hr,
        "480": s.enable_8hr,
        "720": s.enable_12hr,
        "D": s.enable_daily,
        "W": s.enable_week,
        "M": s.enable_month,
    }
    return not enabled_map.get(current_tf, False)


def _tf_rule(tf: str) -> str:
    if tf in {"5", "10", "15", "30", "60", "240", "480", "720"}:
        return f"{tf}min"
    return {"D": "1D", "W": "1W", "M": "1ME"}[tf]


def _security_like(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _tf_rule(tf)
    htf = (
        df.resample(rule)
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
    )
    return htf.reindex(df.index, method="ffill")


def _is_bull(fvgmethod: bool, display: bool, open1: float, close1: float, op: float, cl: float, high1: float) -> bool:
    if not display:
        return False
    return (open1 > close1 and op < cl and cl > high1) if fvgmethod else (op < high1)


def _is_bear(fvgmethod: bool, display: bool, open1: float, close1: float, op: float, cl: float, low1: float) -> bool:
    if not display:
        return False
    return (open1 < close1 and op > cl and cl < low1) if fvgmethod else (op < low1)


def _duplicate_box(boxes: List[MKBox], top: float) -> bool:
    return any(b.top == top for b in boxes)


def _add_label(labels: List[MKLabel], bar_index: int, high: float, text: str, s: MKSettings, ts: pd.Timestamp, y: float) -> None:
    if not s.show_labels:
        return
    if s.show_timeonlabels:
        stamp = ts + pd.Timedelta(hours=s.hours_offset_input)
        text = f"{text}  {stamp:%H:%M %m/%d/%y}"
    labels.append(MKLabel(x=bar_index + s.label_shift, y=y, text=text))


def _in_session(ts: pd.Timestamp) -> bool:
    return (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)) and (ts.hour < 16 or (ts.hour == 16 and ts.minute == 0))


def run_ict_mtf_order_block_wicks(df: pd.DataFrame, settings: Optional[MKSettings] = None, current_timeframe: str = "15") -> MKResult:
    s = settings or MKSettings()
    mit = _mitigation_action(s.mitigation_mode)
    intrusion_pct = s.incursion_pct / 100.0

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    total_max = s.curr_max_array_size + s.max_5 + s.max_10 + s.max_15 + s.max_60 + s.max_240 + s.max_480 + s.max_d + s.max_w + s.max_m
    is_error = total_max > 500

    states: Dict[str, TFState] = {k: TFState() for k in ["chart", "5", "10", "15", "30", "60", "240", "480", "720", "D", "W", "M"]}
    incursion_messages: List[str] = []

    def _update_bull(tfstate: TFState, bar_index: int, low: float, close: float, lastlow: float, timestr: str) -> None:
        for i in range(len(tfstate.bull_boxes) - 1, -1, -1):
            bx = tfstate.bull_boxes[i]
            top, bottom = bx.top, bx.bottom
            mid = (top + bottom) / 2
            threshold = top - (intrusion_pct * (top - bottom))
            lowundertop = low < top
            lowunderbtm = low < bottom
            lowundermid = low < mid
            closeundertop = close < top
            closeunderbtm = close < bottom
            closeundermid = low < mid  # mirrors script typo exactly
            intrusion = low < threshold and lastlow > threshold

            if (mit in (1, 3)) and intrusion and s.incursion_alerts:
                incursion_messages.append(f"Bull OB Wick Incursion {timestr}")

            if s.entry_change_color and lowundertop:
                bx.bgcolor = s.entry_bull_color

            if s.show_labels and i < len(tfstate.bull_labels):
                tfstate.bull_labels[i].x = bar_index + s.label_shift
                tfstate.bull_labels[i].y = (bx.top + bx.bottom) / 2
                bx.left = len(df) - 1 + s.label_shift

            if mit == 2 and s.use_body_for_mitigation and closeundertop:
                bx.top = close
            elif mit == 2 and lowundertop:
                bx.top = low

            if mit == 3:
                if (s.use_body_for_mitigation and closeunderbtm) or lowunderbtm:
                    bx.bgcolor = s.no_mitigation_color
                    if s.show_labels and s.show_mitigated_text and i < len(tfstate.bull_labels):
                        if "Mitigated" not in tfstate.bull_labels[i].text:
                            tfstate.bull_labels[i].text += " Mitigated"

            should_delete = False
            if mit in (1, 2):
                should_delete = (s.use_body_for_mitigation and closeunderbtm) or lowunderbtm
            if mit == 4:
                should_delete = (s.use_body_for_mitigation and closeundermid) or lowundermid
            if should_delete:
                tfstate.bull_boxes.pop(i)
                if s.show_labels and i < len(tfstate.bull_labels):
                    tfstate.bull_labels.pop(i)

    def _update_bear(tfstate: TFState, bar_index: int, high: float, close: float, lasthigh: float, timestr: str) -> None:
        for i in range(len(tfstate.bear_boxes) - 1, -1, -1):
            bx = tfstate.bear_boxes[i]
            top, bottom = bx.top, bx.bottom
            mid = (top + bottom) / 2
            threshold = bottom + (intrusion_pct * (top - bottom))
            highovertop = high > top
            highoverbtm = high > bottom
            highovermid = high > mid
            closeovertop = close > top
            closeoverbtm = close > bottom
            closeovermid = close > mid
            intrusion = high > threshold and lasthigh < threshold

            if (mit in (1, 3)) and intrusion and s.incursion_alerts:
                incursion_messages.append(f"Bear OB Wick Incursion {timestr}")

            if s.entry_change_color:
                bx.bgcolor = s.entry_bear_color if highoverbtm else s.bear_color

            if s.show_labels and i < len(tfstate.bear_labels):
                tfstate.bear_labels[i].x = bar_index + s.label_shift
                tfstate.bear_labels[i].y = (bx.top + bx.bottom) / 2
                bx.left = len(df) - 1 + s.label_shift

            if mit == 2 and s.use_body_for_mitigation and closeoverbtm:
                bx.bottom = close
            elif mit == 2 and highoverbtm:
                bx.bottom = high

            if mit == 3:
                if (s.use_body_for_mitigation and closeovertop) or highovertop:
                    bx.bgcolor = s.no_mitigation_color
                    if s.show_labels and s.show_mitigated_text and i < len(tfstate.bear_labels):
                        if "Mitigated" not in tfstate.bear_labels[i].text:
                            tfstate.bear_labels[i].text += " Mitigated"

            should_delete = False
            if mit in (1, 2):
                should_delete = (s.use_body_for_mitigation and closeovertop) or highovertop
            if mit == 4:
                should_delete = (s.use_body_for_mitigation and closeovermid) or highovermid
            if should_delete:
                tfstate.bear_boxes.pop(i)
                if s.show_labels and i < len(tfstate.bear_labels):
                    tfstate.bear_labels.pop(i)

    def _handle_all(tf_key: str, timestr: str, enabled: bool, max_size: int, aligned: pd.DataFrame) -> None:
        if not enabled:
            return
        st = states[tf_key]
        st.new_bull = False
        st.new_bear = False

        for i in range(1, len(df)):
            ts = df.index[i]
            if s.only_market_hours and not _in_session(ts):
                pass_condition = False
            else:
                pass_condition = True
            if not pass_condition:
                continue

            open1 = float(aligned["open"].iloc[i - 1])
            close1 = float(aligned["close"].iloc[i - 1])
            op = float(aligned["open"].iloc[i])
            cl = float(aligned["close"].iloc[i])
            high1 = float(aligned["high"].iloc[i - 1])
            low1 = float(aligned["low"].iloc[i - 1])

            new_bull = _is_bull(s.fvgmethod_body, s.display, open1, close1, op, cl, high1)
            new_bear = _is_bear(s.fvgmethod_body, s.display, open1, close1, op, cl, low1)
            st.new_bull = st.new_bull or new_bull
            st.new_bear = st.new_bear or new_bear

            if new_bull:
                if len(st.bull_boxes) > max_size:
                    st.bull_boxes.pop(0)
                    if s.show_labels and st.bull_labels:
                        st.bull_labels.pop(0)
                if not _duplicate_box(st.bull_boxes, high1):
                    st.bull_boxes.append(MKBox(len(df)-1+20, len(df)-1+200, high1, open1, s.bull_color, "yellow", tf_key, "bull"))
                    _add_label(st.bull_labels, i, high1, f"{timestr} OB BULL", s, ts, (high1 + low1) / 2)

            if new_bear:
                if len(st.bear_boxes) > max_size:
                    st.bear_boxes.pop(0)
                    if s.show_labels and st.bear_labels:
                        st.bear_labels.pop(0)
                if not _duplicate_box(st.bear_boxes, open1):
                    st.bear_boxes.append(MKBox(len(df)-1+20, len(df)-1+200, open1, low1, s.bear_color, "blue", tf_key, "bear"))
                    _add_label(st.bear_labels, i, high1, f"{timestr} OB BEAR", s, ts, (high1 + low1) / 2)

            if st.bull_boxes:
                _update_bull(st, i, float(df["low"].iloc[i]), float(df["close"].iloc[i]), float(df["low"].iloc[i - 1]), timestr)
            if st.bear_boxes:
                _update_bear(st, i, float(df["high"].iloc[i]), float(df["close"].iloc[i]), float(df["high"].iloc[i - 1]), timestr)

    if s.enable_current_timeframe and _not_current_timeframe_equal_enabled_tfs(current_timeframe, s):
        _handle_all("chart", current_timeframe, True, s.curr_max_array_size, _security_like(df, current_timeframe))

    tf_plan: List[Tuple[str, bool, int, str]] = [
        ("5", s.enable_5min, s.max_5, _gettf_label_str(0)),
        ("10", s.enable_10min, s.max_10, _gettf_label_str(1)),
        ("15", s.enable_15min, s.max_15, _gettf_label_str(2)),
        ("30", s.enable_30min, s.max_30, _gettf_label_str(3)),
        ("60", s.enable_1hr, s.max_60, _gettf_label_str(4)),
        ("240", s.enable_4hr, s.max_240, _gettf_label_str(5)),
        ("480", s.enable_8hr, s.max_480, _gettf_label_str(6)),
        ("720", s.enable_12hr, s.max_720, _gettf_label_str(7)),
        ("D", s.enable_daily, s.max_d, _gettf_label_str(8)),
        ("W", s.enable_week, s.max_w, _gettf_label_str(9)),
        ("M", s.enable_month, s.max_m, _gettf_label_str(10)),
    ]
    for tf, enabled, mx, lbl in tf_plan:
        _handle_all(tf, lbl, enabled, mx, _security_like(df, tf))

    alerts = {
        "bull_fvg_creation_alert": any(states[k].new_bull for k in states),
        "bear_fvg_creation_alert": any(states[k].new_bear for k in states),
        "both_fvg_creation_alert": any(states[k].new_bull or states[k].new_bear for k in states),
    }
    return MKResult(states=states, alerts=alerts, incursion_alert_messages=incursion_messages, is_error=is_error)
