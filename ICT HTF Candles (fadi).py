"""
ICT HTF Candles (fadi)
Python translation of the TradingView Pine Script.

This module mirrors the Pine logic by:
- Building HTF candle sequences for multiple timeframes.
- Tracking per-candle OHLC indices and intra-HTF updates.
- Computing Fair Value Gap (FVG) and Volume Imbalance (VI) regions.
- Preparing trace level data for open/close/high/low lines.

The output structures are designed for plotting or downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import pandas as pd

NY_TZ = "America/New_York"


@dataclass
class Candle:
    o: float
    c: float
    h: float
    l: float
    o_time: pd.Timestamp
    o_idx: int
    c_idx: int
    h_idx: int
    l_idx: int
    dow: str
    bucket_start: pd.Timestamp
    body_left: Optional[float] = None
    body_right: Optional[float] = None
    wick_x: Optional[float] = None
    dow_x: Optional[float] = None
    dow_y: Optional[float] = None
    dow_label: Optional["LabelInfo"] = None


@dataclass
class Imbalance:
    kind: str  # "fvg" or "vi"
    top: float
    bottom: float
    left_idx: float
    right_idx: float
    color: Optional[str] = None


@dataclass
class TraceLevel:
    kind: str  # "open", "close", "high", "low"
    price: float
    start_x: float
    end_x: float
    color: Optional[str] = None
    style: Optional[str] = None
    size: Optional[int] = None
    label: Optional["LabelInfo"] = None


@dataclass
class CandleSettings:
    show: bool
    htf: str
    max_display: int


@dataclass
class Settings:
    max_sets: int = 6
    use_custom_daily: bool = False
    custom_daily: str = "Midnight"  # "Midnight", "8:30", "9:30"
    trace_show: bool = False
    trace_o_color: str = "gray"
    trace_o_style: str = "····"
    trace_o_size: int = 1
    trace_c_color: str = "gray"
    trace_c_style: str = "····"
    trace_c_size: int = 1
    trace_h_color: str = "gray"
    trace_h_style: str = "····"
    trace_h_size: int = 1
    trace_l_color: str = "gray"
    trace_l_style: str = "····"
    trace_l_size: int = 1
    trace_anchor: str = "First Timeframe"  # "First Timeframe" or "Last Timeframe"
    offset: int = 10
    buffer: int = 1
    htf_buffer: int = 10
    width: int = 2
    max_bars_back: int = 5000
    daily_name: bool = False
    label_position: str = "Both"  # "Both", "Top", "Bottom"
    label_alignment: str = "Align"  # "Align", "Follow Candles"
    htf_label_show: bool = True
    htf_label_color: str = "black"
    htf_label_size: str = "large"
    htf_timer_show: bool = True
    htf_timer_color: str = "black"
    htf_timer_size: str = "normal"
    label_show: bool = False
    label_color: str = "black"
    label_size: str = "small"
    bull_body: str = "green"
    bull_border: str = "black"
    bull_wick: str = "black"
    bear_body: str = "red"
    bear_border: str = "black"
    bear_wick: str = "black"
    fvg_show: bool = True
    fvg_color: str = "gray"
    vi_show: bool = True
    vi_color: str = "red"
    dow_color: str = "black"
    dow_size: str = "small"


@dataclass
class CandleSet:
    settings: CandleSettings
    candles: List[Candle] = field(default_factory=list)
    imbalances: List[Imbalance] = field(default_factory=list)
    traces: List[TraceLevel] = field(default_factory=list)
    label_top: Optional["LabelInfo"] = None
    label_bottom: Optional["LabelInfo"] = None
    timer_top: Optional["LabelInfo"] = None
    timer_bottom: Optional["LabelInfo"] = None
    offset_x: Optional[float] = None


@dataclass
class HTFResult:
    candle_sets: Dict[str, CandleSet]
    base_timeframe_seconds: int


@dataclass
class LabelInfo:
    x: float
    y: float
    text: str
    color: Optional[str] = None
    text_color: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None


def _parse_tf_seconds(tf: str) -> int:
    tf = tf.strip()
    if tf.endswith("H") and tf[:-1].isdigit():
        return int(tf[:-1]) * 3600
    if tf.endswith("min") and tf[:-3].isdigit():
        return int(tf[:-3]) * 60
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.isdigit():
        return int(tf) * 60
    if tf in {"1D", "1W", "1M"}:
        return {"1D": 86400, "1W": 604800, "1M": 2592000}[tf]
    raise ValueError(f"Unsupported timeframe: {tf}")


def _infer_base_timeframe_seconds(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        raise ValueError("Need at least two rows to infer base timeframe.")
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    median = diffs.median()
    if pd.isna(median) or median <= 0:
        raise ValueError("Unable to infer base timeframe from index.")
    return int(median)


def _valid_timeframe(htf: str, base_seconds: int) -> bool:
    htf_seconds = _parse_tf_seconds(htf)
    if htf_seconds >= 86400 and htf_seconds > base_seconds:
        return True
    if base_seconds < htf_seconds:
        ratio = htf_seconds / base_seconds
        return round(ratio) == ratio
    return False


def _floor_time(ts: pd.Timestamp, tf: str, custom_daily: Optional[str]) -> pd.Timestamp:
    if tf in {"1D", "1W", "1M"}:
        if tf == "1D" and custom_daily:
            offset_hours = {"Midnight": 0, "8:30": 8.5, "9:30": 9.5}[custom_daily]
            shifted = ts - pd.Timedelta(hours=offset_hours)
            return shifted.floor("D") + pd.Timedelta(hours=offset_hours)
        if tf == "1W":
            return ts.to_period("W").start_time
        if tf == "1M":
            return ts.to_period("M").start_time
        return ts.floor("D")
    minutes = _parse_tf_seconds(tf) // 60
    return ts.floor(f"{minutes}min")


def _day_of_week_label(ts: pd.Timestamp) -> str:
    labels = {0: "M", 1: "T", 2: "W", 3: "T", 4: "F", 5: "S", 6: "S"}
    return labels.get(ts.dayofweek, "")


def _htf_name(tf: str) -> str:
    seconds = _parse_tf_seconds(tf)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}H"
    return tf


def _dow_for_candle(ts: pd.Timestamp, tf: str) -> str:
    if tf == "1D":
        return _day_of_week_label(ts)
    if tf.isdigit() and int(tf) < 60:
        return ts.strftime("%M")
    if tf.isdigit() and int(tf) >= 60:
        return ts.strftime("%H")
    if tf == "1M":
        return ts.strftime("%m")
    return ""


def _ensure_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(tz)


def _update_candle(candle: Candle, row: pd.Series, idx: int) -> None:
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    if high > candle.h:
        candle.h = high
        candle.h_idx = idx
    if low < candle.l:
        candle.l = low
        candle.l_idx = idx
    candle.c = close
    candle.c_idx = idx


def _remaining_time_text(
    ts: pd.Timestamp,
    tf: str,
    custom_daily: Optional[str],
    realtime: bool,
    now_ts: Optional[pd.Timestamp],
) -> str:
    if not realtime:
        return "n/a"
    if now_ts is None:
        now_ts = pd.Timestamp.now(tz=NY_TZ)
    else:
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC").tz_convert(NY_TZ)
        else:
            now_ts = now_ts.tz_convert(NY_TZ)
    if tf in {"1D", "1W", "1M"}:
        bucket = _floor_time(now_ts, tf, custom_daily)
        if tf == "1W":
            next_bucket = (bucket.to_period("W") + 1).start_time
        elif tf == "1M":
            next_bucket = (bucket.to_period("M") + 1).start_time
        else:
            next_bucket = bucket + pd.Timedelta(days=1)
    else:
        seconds = _parse_tf_seconds(tf)
        bucket = _floor_time(now_ts, tf, custom_daily)
        next_bucket = bucket + pd.Timedelta(seconds=seconds)
    remaining = max(0, int((next_bucket - now_ts).total_seconds()))
    days = remaining // 86400
    hours = (remaining - days * 86400) // 3600
    minutes = (remaining - days * 86400 - hours * 3600) // 60
    seconds = remaining - days * 86400 - hours * 3600 - minutes * 60
    result = f"{seconds:02d}"
    if minutes > 0 or hours > 0 or days > 0:
        result = f"{minutes:02d}:{result}"
    if hours > 0 or days > 0:
        result = f"{hours:02d}:{result}"
    if days > 0:
        result = f"{days}D {result}"
    return result


def _is_new_custom_daily(
    ts: pd.Timestamp,
    prev_ts: Optional[pd.Timestamp],
    custom_daily: str,
    bar_seconds: int,
) -> bool:
    if prev_ts is None:
        return True
    if custom_daily == "Midnight":
        return ts.date() != prev_ts.date()
    if custom_daily == "8:30":
        target_hour, target_minute = 8, 30
    else:
        target_hour, target_minute = 9, 30
    target_dt = ts.normalize() + pd.Timedelta(hours=target_hour, minutes=target_minute)
    bar_end = ts + pd.Timedelta(seconds=bar_seconds)
    prev_end = prev_ts + pd.Timedelta(seconds=bar_seconds)
    in_bar = ts <= target_dt < bar_end
    in_prev = prev_ts <= target_dt < prev_end
    return in_bar and not in_prev


def _reorder_positions(
    candles: List[Candle],
    offset: int,
    width: int,
    buffer: int,
    bar_index: int,
    daily_name: bool,
    dow_color: str,
    dow_size: str,
) -> None:
    size = len(candles)
    if size == 0:
        return
    for i in range(size - 1, -1, -1):
        candle = candles[i]
        t_buffer = offset + (width + buffer) * (size - i - 1)
        body_left = bar_index + t_buffer
        body_right = bar_index + width + t_buffer
        wick_x = bar_index + width / 2 + t_buffer
        candle.body_left = float(body_left)
        candle.body_right = float(body_right)
        candle.wick_x = float(wick_x)
        candle.dow_x = float(wick_x)
        candle.dow_y = candle.h
        if daily_name:
            candle.dow_label = LabelInfo(
                x=float(wick_x),
                y=candle.h,
                text=candle.dow,
                text_color=dow_color,
                size=dow_size,
                style="label_down",
            )
        else:
            candle.dow_label = None


def _build_trace_levels(
    candle: Candle,
    last_bar_index: int,
    max_bars_back: int,
    label_show: bool,
    settings: Settings,
) -> List[TraceLevel]:
    body_left = candle.body_left if candle.body_left is not None else float(candle.o_idx)
    body_right = candle.body_right if candle.body_right is not None else float(candle.c_idx)
    wick_x = candle.wick_x if candle.wick_x is not None else float(candle.h_idx)
    traces: List[TraceLevel] = []
    if last_bar_index - candle.o_idx < max_bars_back:
        label = (
            LabelInfo(
                x=body_right,
                y=candle.o,
                text=str(candle.o),
                text_color=settings.label_color,
                size=settings.label_size,
                style="label_left",
            )
            if label_show
            else None
        )
        traces.append(
            TraceLevel(
                kind="open",
                price=candle.o,
                start_x=float(candle.o_idx),
                end_x=body_left,
                color=settings.trace_o_color,
                style=settings.trace_o_style,
                size=settings.trace_o_size,
                label=label,
            )
        )
    if last_bar_index - candle.c_idx < max_bars_back:
        label = (
            LabelInfo(
                x=body_right,
                y=candle.c,
                text=str(candle.c),
                text_color=settings.label_color,
                size=settings.label_size,
                style="label_left",
            )
            if label_show
            else None
        )
        traces.append(
            TraceLevel(
                kind="close",
                price=candle.c,
                start_x=float(candle.c_idx),
                end_x=body_left,
                color=settings.trace_c_color,
                style=settings.trace_c_style,
                size=settings.trace_c_size,
                label=label,
            )
        )
    if last_bar_index - candle.h_idx < max_bars_back:
        label = (
            LabelInfo(
                x=body_right,
                y=candle.h,
                text=str(candle.h),
                text_color=settings.label_color,
                size=settings.label_size,
                style="label_left",
            )
            if label_show
            else None
        )
        traces.append(
            TraceLevel(
                kind="high",
                price=candle.h,
                start_x=float(candle.h_idx),
                end_x=wick_x,
                color=settings.trace_h_color,
                style=settings.trace_h_style,
                size=settings.trace_h_size,
                label=label,
            )
        )
    if last_bar_index - candle.l_idx < max_bars_back:
        label = (
            LabelInfo(
                x=body_right,
                y=candle.l,
                text=str(candle.l),
                text_color=settings.label_color,
                size=settings.label_size,
                style="label_left",
            )
            if label_show
            else None
        )
        traces.append(
            TraceLevel(
                kind="low",
                price=candle.l,
                start_x=float(candle.l_idx),
                end_x=wick_x,
                color=settings.trace_l_color,
                style=settings.trace_l_style,
                size=settings.trace_l_size,
                label=label,
            )
        )
    return traces


def _candle_set_high(candles: List[Candle], seed: float) -> float:
    high = seed
    for candle in candles:
        if candle.h > high:
            high = candle.h
    return high


def _candle_set_low(candles: List[Candle], seed: float) -> float:
    low = seed
    for candle in candles:
        if candle.l < low:
            low = candle.l
    return low


def _find_imbalances(candles: List[Candle], settings: Settings) -> List[Imbalance]:
    imbalances: List[Imbalance] = []
    if len(candles) > 3 and settings.fvg_show:
        for i in range(0, len(candles) - 2):
            candle1 = candles[i]
            candle2 = candles[i + 2]
            candle3 = candles[i + 1]

            if candle1.l > candle2.h and min(candle1.o, candle1.c) > max(candle2.o, candle2.c):
                # Pine: box.new(candle2.body.left, candle2.h, candle1.body.right, candle1.l)
                # Parameters: left, top, right, bottom
                imbalances.append(
                    Imbalance(
                        kind="fvg",
                        top=candle2.h,
                        bottom=candle1.l,
                        left_idx=candle2.body_left if candle2.body_left is not None else float(candle2.o_idx),
                        right_idx=candle1.body_right if candle1.body_right is not None else float(candle1.c_idx),
                        color=settings.fvg_color,
                    )
                )
            if candle1.h < candle2.l and max(candle1.o, candle1.c) < min(candle2.o, candle2.c):
                # Pine: box.new(candle1.body.right, candle1.h, candle2.body.left, candle2.l)
                # Parameters: left, top, right, bottom
                imbalances.append(
                    Imbalance(
                        kind="fvg",
                        top=candle1.h,
                        bottom=candle2.l,
                        left_idx=candle1.body_right if candle1.body_right is not None else float(candle1.o_idx),
                        right_idx=candle2.body_left if candle2.body_left is not None else float(candle2.c_idx),
                        color=settings.fvg_color,
                    )
                )
            _ = candle3
    if len(candles) > 2 and settings.vi_show:
        for i in range(0, len(candles) - 1):
            candle1 = candles[i]
            candle2 = candles[i + 1]
            if candle1.l < candle2.h and min(candle1.o, candle1.c) > max(candle2.o, candle2.c):
                imbalances.append(
                    Imbalance(
                        kind="vi",
                        top=min(candle1.o, candle1.c),
                        bottom=max(candle2.o, candle2.c),
                        left_idx=candle2.body_left if candle2.body_left is not None else float(candle2.o_idx),
                        right_idx=candle1.body_right if candle1.body_right is not None else float(candle1.c_idx),
                        color=settings.vi_color,
                    )
                )
            if candle1.h > candle2.l and max(candle1.o, candle1.c) < min(candle2.o, candle2.c):
                imbalances.append(
                    Imbalance(
                        kind="vi",
                        top=min(candle2.o, candle2.c),
                        bottom=max(candle1.o, candle1.c),
                        left_idx=candle1.body_right if candle1.body_right is not None else float(candle1.o_idx),
                        right_idx=candle2.body_left if candle2.body_left is not None else float(candle2.c_idx),
                        color=settings.vi_color,
                    )
                )
    return imbalances


def calculate_ict_htf_candles(
    df: pd.DataFrame,
    settings: Optional[Settings] = None,
    htf_settings: Optional[Sequence[CandleSettings]] = None,
    base_timeframe_seconds: Optional[int] = None,
    realtime: bool = False,
    now_ts: Optional[pd.Timestamp] = None,
) -> HTFResult:
    """
    Calculate ICT HTF candle sets from LTF OHLC data.

    Args:
        df: DataFrame with columns [open, high, low, close] and a DatetimeIndex or a "time" column.
        settings: Optional Settings override.
        htf_settings: Optional list of CandleSettings (max 6).
        base_timeframe_seconds: Optional base timeframe in seconds; inferred if omitted.
        realtime: Whether the final bar should mimic TradingView realtime mode.
        now_ts: Optional timestamp to use for realtime remaining time calculation.

    Returns:
        HTFResult with candle sets, imbalances, and trace levels.
    """
    settings = settings or Settings()
    if htf_settings is None:
        htf_settings = [
            CandleSettings(show=True, htf="5", max_display=10),
            CandleSettings(show=True, htf="15", max_display=10),
            CandleSettings(show=True, htf="60", max_display=10),
            CandleSettings(show=True, htf="240", max_display=10),
            CandleSettings(show=True, htf="1D", max_display=10),
            CandleSettings(show=True, htf="1W", max_display=10),
        ]

    data = df.copy()
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], utc=True)
        data = data.set_index("time")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must use a DatetimeIndex or include a 'time' column.")
    data = data.sort_index()
    data.index = _ensure_tz(data.index, NY_TZ)

    if base_timeframe_seconds is None:
        base_timeframe_seconds = _infer_base_timeframe_seconds(data.index)

    candle_sets: Dict[str, CandleSet] = {}
    for setting in htf_settings:
        candle_sets[setting.htf] = CandleSet(settings=setting)

    enabled = [s for s in htf_settings if s.show]
    max_sets = min(settings.max_sets, len(enabled))

    for bar_idx, (ts, row) in enumerate(data.iterrows()):
        prev_ts = data.index[bar_idx - 1] if bar_idx > 0 else None
        cnt = 0
        for setting in htf_settings:
            if not setting.show:
                continue
            if cnt >= max_sets:
                break
            if not _valid_timeframe(setting.htf, base_timeframe_seconds):
                continue

            candle_set = candle_sets[setting.htf]
            bucket = _floor_time(ts, setting.htf, settings.custom_daily if settings.use_custom_daily else None)
            current = candle_set.candles[0] if candle_set.candles else None
            is_new = current is None or bucket != current.bucket_start
            if (
                settings.use_custom_daily
                and setting.htf == "1D"
                and current is not None
            ):
                is_new = _is_new_custom_daily(
                    ts,
                    prev_ts,
                    settings.custom_daily,
                    base_timeframe_seconds,
                )

            if is_new:
                candle = Candle(
                    o=float(row["open"]),
                    c=float(row["close"]),
                    h=float(row["high"]),
                    l=float(row["low"]),
                    o_time=ts,
                    o_idx=bar_idx,
                    c_idx=bar_idx,
                    h_idx=bar_idx,
                    l_idx=bar_idx,
                    dow=_dow_for_candle(ts, setting.htf),
                    bucket_start=bucket,
                )
                candle_set.candles.insert(0, candle)
                if len(candle_set.candles) > setting.max_display:
                    candle_set.candles.pop()
            else:
                _update_candle(current, row, bar_idx)

            cnt += 1

    offset = settings.offset
    cnt = 0
    last_bar_index = len(data) - 1
    last_ts = data.index[-1] if len(data) else None
    ordered_sets: List[CandleSet] = []
    for setting in htf_settings:
        if not setting.show:
            continue
        if cnt >= max_sets:
            break
        if not _valid_timeframe(setting.htf, base_timeframe_seconds):
            continue

        candle_set = candle_sets[setting.htf]
        _reorder_positions(
            candle_set.candles,
            offset,
            settings.width,
            settings.buffer,
            last_bar_index,
            settings.daily_name,
            settings.dow_color,
            settings.dow_size,
        )
        candle_set.offset_x = float(offset)

        ordered_sets.append(candle_set)

        show_trace = False
        if settings.trace_anchor == "First Timeframe":
            show_trace = cnt == 0
        elif settings.trace_anchor == "Last Timeframe":
            show_trace = cnt == max_sets - 1
        if settings.trace_show and show_trace and candle_set.candles:
            candle = candle_set.candles[0]
            candle_set.traces = _build_trace_levels(
                candle,
                last_bar_index,
                settings.max_bars_back,
                settings.label_show,
                settings,
            )
        else:
            candle_set.traces = []

        candle_set.imbalances = _find_imbalances(candle_set.candles, settings)

        size = len(candle_set.candles)
        offset += (
            size * settings.width
            + ((size - 1) * settings.buffer if size > 0 else 0)
            + settings.htf_buffer
        )
        cnt += 1

    if ordered_sets and last_ts is not None:
        if settings.label_alignment == "Align":
            global_high = 0.0
            for candle_set in ordered_sets:
                global_high = _candle_set_high(candle_set.candles, global_high)
            global_low = _candle_set_low(ordered_sets[0].candles, global_high)
            for candle_set in ordered_sets[1:]:
                global_low = _candle_set_low(candle_set.candles, global_low)
        for candle_set in ordered_sets:
            size = len(candle_set.candles)
            if size == 0:
                continue
            left = last_bar_index + (candle_set.offset_x or settings.offset) + (
                settings.width + settings.buffer
            ) * (size - 1) / 2
            if settings.label_alignment == "Align":
                top = global_high
                bottom = global_low
            else:
                top = _candle_set_high(candle_set.candles, 0.0)
                bottom = _candle_set_low(candle_set.candles, top)
            htf_text = _htf_name(candle_set.settings.htf)
            timer_text = _remaining_time_text(
                last_ts,
                candle_set.settings.htf,
                settings.custom_daily if settings.use_custom_daily else None,
                realtime,
                now_ts,
            )
            label_top_text = htf_text
            label_bottom_text = htf_text
            if settings.htf_timer_show:
                label_top_text = f"{label_top_text}\n"
                label_bottom_text = f"\n{label_bottom_text}"
            if settings.daily_name:
                label_top_text = f"{label_top_text}\n"
            if settings.htf_label_show:
                if settings.label_position in {"Both", "Top"}:
                    candle_set.label_top = LabelInfo(
                        x=left,
                        y=top,
                        text=label_top_text,
                        text_color=settings.htf_label_color,
                        size=settings.htf_label_size,
                        style="label_down",
                    )
                if settings.label_position in {"Both", "Bottom"}:
                    candle_set.label_bottom = LabelInfo(
                        x=left,
                        y=bottom,
                        text=label_bottom_text,
                        text_color=settings.htf_label_color,
                        size=settings.htf_label_size,
                        style="label_up",
                    )
            if settings.htf_timer_show:
                timer_value = f"({timer_text})"
                if settings.daily_name:
                    timer_value = f"{timer_value}\n"
                if settings.label_position in {"Both", "Top"}:
                    candle_set.timer_top = LabelInfo(
                        x=left,
                        y=top,
                        text=timer_value,
                        text_color=settings.htf_timer_color,
                        size=settings.htf_timer_size,
                        style="label_down",
                    )
                if settings.label_position in {"Both", "Bottom"}:
                    candle_set.timer_bottom = LabelInfo(
                        x=left,
                        y=bottom,
                        text=timer_value,
                        text_color=settings.htf_timer_color,
                        size=settings.htf_timer_size,
                        style="label_up",
                    )

    return HTFResult(candle_sets=candle_sets, base_timeframe_seconds=base_timeframe_seconds)
