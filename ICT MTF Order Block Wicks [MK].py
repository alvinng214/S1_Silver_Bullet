"""Python translation of the Pine Script `ICT MTF Order Block Wicks [MK]`.

This module mirrors the Pine logic by:
- Detecting MTF order block wick zones based on prior candle direction and breakout.
- Tracking mitigation/entry/half-mitigation behavior and intrusions.
- Enforcing per-timeframe max box counts and duplicate suppression.

TradingView drawings (boxes/labels) are represented as data structures so the
behavior can be consumed programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class OBBox:
    direction: str  # "bull" or "bear"
    timeframe: str
    left_index: int
    right_index: int
    top: float
    bottom: float
    label: Optional[str] = None
    label_time: Optional[pd.Timestamp] = None
    mitigated: bool = False
    intrusion: bool = False
    entered: bool = False


@dataclass
class OBSettings:
    display: bool = True
    only_market_hours: bool = False
    session_hours: str = "0930-1600"
    fvgmethod_body: bool = True
    show_labels: bool = True
    show_time_on_labels: bool = False
    hours_offset: float = -5.0
    label_shift: int = 10
    incursion_pct: int = 20
    mitigation_mode: str = "Normal"  # Normal, Dynamic, None, Half
    use_body_for_mitigation: bool = False
    entry_change_color: bool = True
    entry_bull_color: str = "bull"
    entry_bear_color: str = "bear"

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
    enable_weekly: bool = False
    enable_monthly: bool = False

    max_per_tf: Dict[str, int] = field(
        default_factory=lambda: {
            "chart": 8,
            "5": 8,
            "10": 8,
            "15": 8,
            "30": 8,
            "60": 8,
            "240": 8,
            "480": 8,
            "720": 8,
            "D": 8,
            "W": 8,
            "M": 8,
        }
    )
    current_timeframe_guard: bool = True


def _tf_to_rule(tf: str) -> str:
    if tf in {"5", "10", "15", "30", "60", "240", "480", "720"}:
        return f"{tf}min"
    if tf == "D":
        return "1D"
    if tf == "W":
        return "1W"
    if tf == "M":
        return "1M"
    raise ValueError(f"Unsupported timeframe: {tf}")


def _timeframe_label(tf: str) -> str:
    mapping = {
        "5": "5 Min",
        "10": "10 Min",
        "15": "15 Min",
        "30": "30 Min",
        "60": "1 Hr",
        "240": "4 Hr",
        "480": "8 Hr",
        "720": "12Hr",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
    }
    return mapping.get(tf, tf)


def _not_current_timeframe_equal_enabled(settings: OBSettings, current_timeframe: str) -> bool:
    enabled_map = {
        "5": settings.enable_5min,
        "10": settings.enable_10min,
        "15": settings.enable_15min,
        "30": settings.enable_30min,
        "60": settings.enable_1hr,
        "240": settings.enable_4hr,
        "480": settings.enable_8hr,
        "720": settings.enable_12hr,
        "D": settings.enable_daily,
        "W": settings.enable_weekly,
        "M": settings.enable_monthly,
    }
    return not enabled_map.get(current_timeframe, False)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def _align_htf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _tf_to_rule(tf)
    htf = _resample_ohlc(df, rule)
    return htf.reindex(df.index, method="ffill")


def _mitigation_mode(mode: str) -> int:
    return {"Normal": 1, "Dynamic": 2, "None": 3}.get(mode, 4)


def _in_session(ts: pd.Timestamp, settings: OBSettings) -> bool:
    if not settings.only_market_hours:
        return True
    start, end = settings.session_hours.split("-")
    start_h, start_m = int(start[:2]), int(start[2:])
    end_h, end_m = int(end[:2]), int(end[2:])
    return (start_h, start_m) <= (ts.hour, ts.minute) <= (end_h, end_m)


def _detect_bull(settings: OBSettings, open1: float, close1: float, open0: float, close0: float, high1: float) -> bool:
    if not settings.display:
        return False
    if settings.fvgmethod_body:
        return open1 > close1 and open0 < close0 and close0 > high1
    return open0 < high1


def _detect_bear(settings: OBSettings, open1: float, close1: float, open0: float, close0: float, low1: float) -> bool:
    if not settings.display:
        return False
    if settings.fvgmethod_body:
        return open1 < close1 and open0 > close0 and close0 < low1
    return open0 < low1


def _bull_action(top: float, bottom: float, intrusion_pct: float, low: float, close: float, last_low: float) -> Dict[str, bool]:
    mid = (top + bottom) / 2
    threshold = top - (intrusion_pct * (top - bottom))
    return {
        "lowundertop": low < top,
        "lowunderbtm": low < bottom,
        "closeundertop": close < top,
        "closeunderbtm": close < bottom,
        "lowundermid": low < mid,
        "closeundermid": close < mid,
        "intrusion": low < threshold and last_low > threshold,
    }


def _bear_action(top: float, bottom: float, intrusion_pct: float, high: float, close: float, last_high: float) -> Dict[str, bool]:
    mid = (top + bottom) / 2
    threshold = bottom + (intrusion_pct * (top - bottom))
    return {
        "highovertop": high > top,
        "highoverbtm": high > bottom,
        "closeovertop": close > top,
        "closeoverbtm": close > bottom,
        "highovermid": high > mid,
        "closeovermid": close > mid,
        "intrusion": high > threshold and last_high < threshold,
    }


def compute_ict_mtf_order_block_wicks(
    df: pd.DataFrame,
    *,
    settings: Optional[OBSettings] = None,
    current_timeframe: Optional[str] = None,
) -> Dict[str, List[OBBox]]:
    if settings is None:
        settings = OBSettings()

    mitigationaction = _mitigation_mode(settings.mitigation_mode)
    intrusion_pct = settings.incursion_pct / 100.0

    boxes: List[OBBox] = []
    error = sum(settings.max_per_tf.values()) > 500

    def process_tf(tf: str, enabled: bool) -> None:
        if not enabled:
            return
        aligned = _align_htf(df, tf)
        open0 = aligned["open"]
        close0 = aligned["close"]
        open1 = aligned["open"].shift(1)
        close1 = aligned["close"].shift(1)
        high1 = aligned["high"].shift(1)
        low1 = aligned["low"].shift(1)

        for idx in range(len(df)):
            if idx < 1:
                continue
            if not _in_session(df.index[idx], settings):
                continue
            o0 = float(open0.iloc[idx])
            c0 = float(close0.iloc[idx])
            o1 = float(open1.iloc[idx])
            c1 = float(close1.iloc[idx])
            h1 = float(high1.iloc[idx])
            l1 = float(low1.iloc[idx])

            if np.isnan(o1) or np.isnan(c1) or np.isnan(h1) or np.isnan(l1):
                continue

            new_bull = _detect_bull(settings, o1, c1, o0, c0, h1)
            new_bear = _detect_bear(settings, o1, c1, o0, c0, l1)

            if new_bull:
                duplicate = any(b.direction == "bull" and b.top == h1 for b in boxes if b.timeframe == tf)
                if not duplicate:
                    boxes.append(
                        OBBox(
                            direction="bull",
                            timeframe=tf,
                            left_index=idx,
                            right_index=idx,
                            top=h1,
                            bottom=o1,
                            label=(f"{_timeframe_label(tf)} OB BULL" if settings.show_labels else None),
                            label_time=(df.index[idx] + pd.Timedelta(hours=settings.hours_offset)
                                        if settings.show_time_on_labels else None),
                        )
                    )

            if new_bear:
                duplicate = any(b.direction == "bear" and b.top == o1 for b in boxes if b.timeframe == tf)
                if not duplicate:
                    boxes.append(
                        OBBox(
                            direction="bear",
                            timeframe=tf,
                            left_index=idx,
                            right_index=idx,
                            top=o1,
                            bottom=l1,
                            label=(f"{_timeframe_label(tf)} OB BEAR" if settings.show_labels else None),
                            label_time=(df.index[idx] + pd.Timedelta(hours=settings.hours_offset)
                                        if settings.show_time_on_labels else None),
                        )
                    )

        last_low = df["low"].shift(1)
        last_high = df["high"].shift(1)
        for idx in range(len(df)):
            low = float(df["low"].iloc[idx])
            high = float(df["high"].iloc[idx])
            close = float(df["close"].iloc[idx])
            ll = float(last_low.iloc[idx]) if idx > 0 else low
            lh = float(last_high.iloc[idx]) if idx > 0 else high

            for box in list(boxes):
                if box.timeframe != tf:
                    continue
                if box.right_index < idx:
                    box.right_index = idx
                if box.direction == "bull":
                    action = _bull_action(box.top, box.bottom, intrusion_pct, low, close, ll)
                    box.intrusion = action["intrusion"]
                    box.entered = action["lowundertop"]
                    if mitigationaction == 2:
                        if settings.use_body_for_mitigation and action["closeundertop"]:
                            box.top = close
                        elif action["lowundertop"]:
                            box.top = low
                    if mitigationaction == 3:
                        if settings.use_body_for_mitigation and action["closeunderbtm"]:
                            box.mitigated = True
                        elif action["lowunderbtm"]:
                            box.mitigated = True
                    if mitigationaction in {1, 2}:
                        if settings.use_body_for_mitigation and action["closeunderbtm"]:
                            boxes.remove(box)
                        elif action["lowunderbtm"]:
                            boxes.remove(box)
                    if mitigationaction == 4:
                        if settings.use_body_for_mitigation and action["closeundermid"]:
                            boxes.remove(box)
                        elif action["lowundermid"]:
                            boxes.remove(box)
                else:
                    action = _bear_action(box.top, box.bottom, intrusion_pct, high, close, lh)
                    box.intrusion = action["intrusion"]
                    box.entered = action["highoverbtm"]
                    if mitigationaction == 2:
                        if settings.use_body_for_mitigation and action["closeoverbtm"]:
                            box.bottom = close
                        elif action["highoverbtm"]:
                            box.bottom = high
                    if mitigationaction == 3:
                        if settings.use_body_for_mitigation and action["closeovertop"]:
                            box.mitigated = True
                        elif action["highovertop"]:
                            box.mitigated = True
                    if mitigationaction in {1, 2}:
                        if settings.use_body_for_mitigation and action["closeovertop"]:
                            boxes.remove(box)
                        elif action["highovertop"]:
                            boxes.remove(box)
                    if mitigationaction == 4:
                        if settings.use_body_for_mitigation and action["closeovermid"]:
                            boxes.remove(box)
                        elif action["highovermid"]:
                            boxes.remove(box)

        max_allowed = settings.max_per_tf.get(tf, 8)
        if max_allowed > 0:
            tf_boxes = [b for b in boxes if b.timeframe == tf]
            if len(tf_boxes) > max_allowed:
                remove = len(tf_boxes) - max_allowed
                keep = tf_boxes[remove:]
                boxes[:] = [b for b in boxes if b.timeframe != tf] + keep

    if settings.enable_current_timeframe and current_timeframe:
        if not settings.current_timeframe_guard or _not_current_timeframe_equal_enabled(settings, current_timeframe):
            process_tf(current_timeframe, True)
    process_tf("5", settings.enable_5min)
    process_tf("10", settings.enable_10min)
    process_tf("15", settings.enable_15min)
    process_tf("30", settings.enable_30min)
    process_tf("60", settings.enable_1hr)
    process_tf("240", settings.enable_4hr)
    process_tf("480", settings.enable_8hr)
    process_tf("720", settings.enable_12hr)
    process_tf("D", settings.enable_daily)
    process_tf("W", settings.enable_weekly)
    process_tf("M", settings.enable_monthly)

    return {"boxes": boxes, "error": error}
