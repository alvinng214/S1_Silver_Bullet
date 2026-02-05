"""MirPapa ICT HTF FVG OB Threeple (EN).

Python translation of the MirPapa ICT Pine Script indicator.

The implementation mirrors the Pine flow:
- Auto-select mid/high timeframes based on the chart timeframe.
- Cache HTF OHLC values (current and prior bars) and detect new HTF bars.
- Create FOB (FVG OB) boxes on high/mid/current timeframes with cooldown filters.
- Process/close active boxes with configurable close-count thresholds.
- Periodically clean inactive boxes when max size is exceeded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BoxData:
    name: str
    is_bullish: bool
    breach_mode: str
    top: float
    bottom: float
    mid: float
    start_index: int
    end_index: int
    timeframe: str
    htf_bar_index: int
    top_breached: bool = False
    bottom_breached: bool = False
    is_active: bool = True
    close_count: int = 0
    created_bar: int = 0


@dataclass
class HTFCache:
    timeframe: str
    last_bar_index: int
    is_new_bar: bool
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    open1: float
    close1: float
    high1: float
    low1: float
    open2: float
    close2: float
    high2: float
    low2: float
    high3: float
    low3: float
    time1: Optional[pd.Timestamp]
    time2: Optional[pd.Timestamp]


@dataclass
class FOBSettings:
    use_box: bool
    use_line: bool
    close_count: int
    cooldown: int
    bull_color: str
    bear_color: str
    transparency: int


@dataclass
class ThreepleOutputs:
    high_tf_boxes: List[BoxData]
    mid_tf_boxes: List[BoxData]
    current_tf_boxes: List[BoxData]
    debug: Dict[str, int]


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


def _infer_base_minutes(df: pd.DataFrame) -> int:
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return 0
    return int(diffs.median().total_seconds() // 60)


def _select_timeframes(chart_tf: str) -> Tuple[str, str]:
    mapping = {
        "1": ("5", "15"),
        "5": ("15", "60"),
        "15": ("60", "240"),
        "30": ("120", "480"),
        "60": ("240", "1D"),
        "240": ("1D", "1W"),
        "1D": ("1W", "1M"),
        "1W": ("1M", "6M"),
        "1M": ("6M", "12M"),
    }
    return mapping.get(chart_tf, ("240", "1D"))


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _align_series(series: pd.Series, target_index: pd.Index) -> pd.Series:
    return series.reindex(target_index, method="ffill")


def _build_htf_cache(df: pd.DataFrame, timeframe: str) -> HTFCache:
    tf_minutes = _parse_timeframe_to_minutes(timeframe)
    if tf_minutes is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if tf_minutes <= _infer_base_minutes(df) or tf_minutes == 0:
        htf = df[["open", "high", "low", "close"]].copy()
    else:
        htf = _resample_ohlc(df, f"{tf_minutes}min")

    htf["bar_index"] = np.arange(len(htf))
    htf = htf.assign(
        open1=htf["open"].shift(1),
        close1=htf["close"].shift(1),
        high1=htf["high"].shift(1),
        low1=htf["low"].shift(1),
        open2=htf["open"].shift(2),
        close2=htf["close"].shift(2),
        high2=htf["high"].shift(2),
        low2=htf["low"].shift(2),
        high3=htf["high"].shift(3),
        low3=htf["low"].shift(3),
        time1=htf.index.to_series().shift(1),
        time2=htf.index.to_series().shift(2),
    )

    aligned = pd.DataFrame(index=df.index)
    for col in [
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "open1",
        "close1",
        "high1",
        "low1",
        "open2",
        "close2",
        "high2",
        "low2",
        "high3",
        "low3",
        "time1",
        "time2",
    ]:
        aligned[col] = _align_series(htf[col], df.index)

    if aligned["bar_index"].isna().all():
        return HTFCache(
            timeframe=timeframe,
            last_bar_index=0,
            is_new_bar=False,
            bar_index=0,
            open=np.nan,
            high=np.nan,
            low=np.nan,
            close=np.nan,
            open1=np.nan,
            close1=np.nan,
            high1=np.nan,
            low1=np.nan,
            open2=np.nan,
            close2=np.nan,
            high2=np.nan,
            low2=np.nan,
            high3=np.nan,
            low3=np.nan,
            time1=None,
            time2=None,
        )

    bar_index = int(aligned["bar_index"].iloc[-1])
    is_new_bar = aligned["bar_index"].iloc[-1] != aligned["bar_index"].iloc[-2] if len(aligned) > 1 else True

    return HTFCache(
        timeframe=timeframe,
        last_bar_index=bar_index,
        is_new_bar=is_new_bar,
        bar_index=bar_index,
        open=float(aligned["open"].iloc[-1]),
        high=float(aligned["high"].iloc[-1]),
        low=float(aligned["low"].iloc[-1]),
        close=float(aligned["close"].iloc[-1]),
        open1=float(aligned["open1"].iloc[-1]) if not pd.isna(aligned["open1"].iloc[-1]) else np.nan,
        close1=float(aligned["close1"].iloc[-1]) if not pd.isna(aligned["close1"].iloc[-1]) else np.nan,
        high1=float(aligned["high1"].iloc[-1]) if not pd.isna(aligned["high1"].iloc[-1]) else np.nan,
        low1=float(aligned["low1"].iloc[-1]) if not pd.isna(aligned["low1"].iloc[-1]) else np.nan,
        open2=float(aligned["open2"].iloc[-1]) if not pd.isna(aligned["open2"].iloc[-1]) else np.nan,
        close2=float(aligned["close2"].iloc[-1]) if not pd.isna(aligned["close2"].iloc[-1]) else np.nan,
        high2=float(aligned["high2"].iloc[-1]) if not pd.isna(aligned["high2"].iloc[-1]) else np.nan,
        low2=float(aligned["low2"].iloc[-1]) if not pd.isna(aligned["low2"].iloc[-1]) else np.nan,
        high3=float(aligned["high3"].iloc[-1]) if not pd.isna(aligned["high3"].iloc[-1]) else np.nan,
        low3=float(aligned["low3"].iloc[-1]) if not pd.isna(aligned["low3"].iloc[-1]) else np.nan,
        time1=aligned["time1"].iloc[-1] if not pd.isna(aligned["time1"].iloc[-1]) else None,
        time2=aligned["time2"].iloc[-1] if not pd.isna(aligned["time2"].iloc[-1]) else None,
    )


def _build_htf_cache_series(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf_minutes = _parse_timeframe_to_minutes(timeframe)
    if tf_minutes is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if tf_minutes <= _infer_base_minutes(df) or tf_minutes == 0:
        htf = df[["open", "high", "low", "close"]].copy()
    else:
        htf = _resample_ohlc(df, f"{tf_minutes}min")

    htf["bar_index"] = np.arange(len(htf))
    htf = htf.assign(
        open1=htf["open"].shift(1),
        close1=htf["close"].shift(1),
        high1=htf["high"].shift(1),
        low1=htf["low"].shift(1),
        open2=htf["open"].shift(2),
        close2=htf["close"].shift(2),
        high2=htf["high"].shift(2),
        low2=htf["low"].shift(2),
        high3=htf["high"].shift(3),
        low3=htf["low"].shift(3),
        time1=htf.index.to_series().shift(1),
        time2=htf.index.to_series().shift(2),
    )

    aligned = pd.DataFrame(index=df.index)
    for col in [
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "open1",
        "close1",
        "high1",
        "low1",
        "open2",
        "close2",
        "high2",
        "low2",
        "high3",
        "low3",
        "time1",
        "time2",
    ]:
        aligned[col] = _align_series(htf[col], df.index)
    return aligned


def _htf_cache_from_aligned(aligned: pd.DataFrame, bar_index: int, timeframe: str) -> HTFCache:
    row = aligned.iloc[bar_index]
    prev_row = aligned.iloc[bar_index - 1] if bar_index > 0 else row
    if pd.isna(row["bar_index"]):
        return HTFCache(
            timeframe=timeframe,
            last_bar_index=0,
            is_new_bar=False,
            bar_index=0,
            open=np.nan,
            high=np.nan,
            low=np.nan,
            close=np.nan,
            open1=np.nan,
            close1=np.nan,
            high1=np.nan,
            low1=np.nan,
            open2=np.nan,
            close2=np.nan,
            high2=np.nan,
            low2=np.nan,
            high3=np.nan,
            low3=np.nan,
            time1=None,
            time2=None,
        )

    bar_index_value = int(row["bar_index"])
    is_new_bar = row["bar_index"] != prev_row["bar_index"] if bar_index > 0 else True
    return HTFCache(
        timeframe=timeframe,
        last_bar_index=bar_index_value,
        is_new_bar=is_new_bar,
        bar_index=bar_index_value,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        open1=float(row["open1"]) if not pd.isna(row["open1"]) else np.nan,
        close1=float(row["close1"]) if not pd.isna(row["close1"]) else np.nan,
        high1=float(row["high1"]) if not pd.isna(row["high1"]) else np.nan,
        low1=float(row["low1"]) if not pd.isna(row["low1"]) else np.nan,
        open2=float(row["open2"]) if not pd.isna(row["open2"]) else np.nan,
        close2=float(row["close2"]) if not pd.isna(row["close2"]) else np.nan,
        high2=float(row["high2"]) if not pd.isna(row["high2"]) else np.nan,
        low2=float(row["low2"]) if not pd.isna(row["low2"]) else np.nan,
        high3=float(row["high3"]) if not pd.isna(row["high3"]) else np.nan,
        low3=float(row["low3"]) if not pd.isna(row["low3"]) else np.nan,
        time1=row["time1"] if not pd.isna(row["time1"]) else None,
        time2=row["time2"] if not pd.isna(row["time2"]) else None,
    )


def _is_chart_tf_comparison_htf(chart_tf: str, target_tf: str) -> bool:
    chart_minutes = _parse_timeframe_to_minutes(chart_tf)
    target_minutes = _parse_timeframe_to_minutes(target_tf)
    if chart_minutes is None or target_minutes is None:
        return False
    return target_minutes >= chart_minutes


def _is_fob_condition(is_bullish: bool, price_prev: float, price_now: float) -> bool:
    if np.isnan(price_prev) or np.isnan(price_now):
        return False
    if is_bullish:
        return price_now > price_prev
    return price_prev > price_now


def _is_sob_condition(is_bullish: bool, open2: float, close1: float, open1: float, close0: float) -> bool:
    if np.isnan(open2) or np.isnan(close1) or np.isnan(open1) or np.isnan(close0):
        return False
    if is_bullish:
        return open2 > close1 and open1 < close0
    return open2 < close1 and open1 > close0


def _process_box_datas(
    boxes: List[BoxData],
    close_count: int,
    high_price: float,
    low_price: float,
    close_price: float,
    bar_index: int,
) -> None:
    for box in boxes:
        if not box.is_active:
            continue
        box.end_index = bar_index
        breach_mode = box.breach_mode.lower()
        if breach_mode == "directionalhighlow":
            if box.is_bullish:
                if high_price > box.top:
                    box.top_breached = True
                if box.top_breached and low_price < box.bottom:
                    box.close_count += 1
                    box.top_breached = False
                    box.bottom_breached = False
            else:
                if low_price < box.bottom:
                    box.bottom_breached = True
                if box.bottom_breached and high_price > box.top:
                    box.close_count += 1
                    box.top_breached = False
                    box.bottom_breached = False
        elif breach_mode == "directionalclose":
            if box.is_bullish:
                if close_price > box.top:
                    box.top_breached = True
                if box.top_breached and close_price < box.bottom:
                    box.close_count += 1
                    box.top_breached = False
                    box.bottom_breached = False
            else:
                if close_price < box.bottom:
                    box.bottom_breached = True
                if box.bottom_breached and close_price > box.top:
                    box.close_count += 1
                    box.top_breached = False
                    box.bottom_breached = False
        elif breach_mode == "sobclose":
            if box.is_bullish and close_price < box.bottom:
                box.close_count += 1
            if not box.is_bullish and close_price > box.top:
                box.close_count += 1
        else:
            new_top = box.top_breached or (high_price > box.top)
            new_bottom = box.bottom_breached or (low_price < box.bottom)
            if new_top and new_bottom and not (box.top_breached and box.bottom_breached):
                box.close_count += 1
                box.top_breached = False
                box.bottom_breached = False
            else:
                box.top_breached = new_top
                box.bottom_breached = new_bottom

        if box.close_count >= close_count:
            box.is_active = False


def _cleanup_box_array(boxes: List[BoxData], max_size: int) -> int:
    removed = 0
    if len(boxes) > max_size:
        for i in range(len(boxes) - 1, -1, -1):
            if len(boxes) <= max_size:
                break
            if not boxes[i].is_active:
                boxes.pop(i)
                removed += 1
    return removed


def _create_box_data(
    name: str,
    is_bullish: bool,
    low2: float,
    high2: float,
    timeframe: str,
    htf_bar_index: int,
    start_index: int,
    *,
    open1: float | None = None,
    close1: float | None = None,
) -> Tuple[bool, Optional[BoxData]]:
    if name == "sob":
        if open1 is None or close1 is None or np.isnan(open1) or np.isnan(close1):
            return False, None
        top = max(open1, close1)
        bottom = min(open1, close1)
        breach_mode = "sobClose"
    else:
        if np.isnan(low2) or np.isnan(high2):
            return False, None
        top = max(low2, high2)
        bottom = min(low2, high2)
        breach_mode = "directionalHighLow" if name in {"fob", "rb", "custom"} else "both"
    mid = (top + bottom) * 0.5
    return True, BoxData(
        name=name,
        is_bullish=is_bullish,
        breach_mode=breach_mode,
        top=top,
        bottom=bottom,
        mid=mid,
        start_index=start_index,
        end_index=start_index,
        timeframe=timeframe,
        htf_bar_index=htf_bar_index,
        created_bar=start_index,
    )


def calculate_fvg_ob_threeple(
    df: pd.DataFrame,
    *,
    chart_timeframe: str,
    mid_tf_override: Optional[str] = None,
    high_tf_override: Optional[str] = None,
    max_boxes: int = 300,
    enable_cleanup: bool = True,
    high_tf_settings: FOBSettings = FOBSettings(
        use_box=True,
        use_line=False,
        close_count=1,
        cooldown=3,
        bull_color="aqua",
        bear_color="orange",
        transparency=70,
    ),
    mid_tf_settings: FOBSettings = FOBSettings(
        use_box=True,
        use_line=False,
        close_count=1,
        cooldown=3,
        bull_color="blue",
        bear_color="yellow",
        transparency=70,
    ),
    current_tf_settings: FOBSettings = FOBSettings(
        use_box=True,
        use_line=False,
        close_count=1,
        cooldown=3,
        bull_color="green",
        bear_color="red",
        transparency=80,
    ),
    offset_fob: int = 2,
) -> ThreepleOutputs:
    if mid_tf_override and high_tf_override:
        mid_tf = mid_tf_override
        high_tf = high_tf_override
    else:
        mid_tf, high_tf = _select_timeframes(chart_timeframe)

    high_tf_boxes_bull: List[BoxData] = []
    high_tf_boxes_bear: List[BoxData] = []
    mid_tf_boxes_bull: List[BoxData] = []
    mid_tf_boxes_bear: List[BoxData] = []
    current_tf_boxes_bull: List[BoxData] = []
    current_tf_boxes_bear: List[BoxData] = []

    last_high_tf_fob_bar: Optional[int] = None
    last_high_tf_fob_bull: bool = False
    last_high_tf_sob_bar: Optional[int] = None
    last_high_tf_sob_bull: bool = False
    last_mid_tf_fob_bar: Optional[int] = None
    last_mid_tf_fob_bull: bool = False
    last_mid_tf_sob_bar: Optional[int] = None
    last_mid_tf_sob_bull: bool = False
    last_current_tf_fob_bar: Optional[int] = None
    last_current_tf_fob_bull: bool = False
    last_current_tf_sob_bar: Optional[int] = None
    last_current_tf_sob_bull: bool = False

    high_tf_aligned = (
        _build_htf_cache_series(df, high_tf)
        if high_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, high_tf)
        else None
    )
    mid_tf_aligned = (
        _build_htf_cache_series(df, mid_tf)
        if mid_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, mid_tf)
        else None
    )

    for bar_index, (timestamp, row) in enumerate(df.iterrows()):
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        if high_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, high_tf):
            _process_box_datas(
                high_tf_boxes_bull,
                high_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )
            _process_box_datas(
                high_tf_boxes_bear,
                high_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )
        if mid_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, mid_tf):
            _process_box_datas(
                mid_tf_boxes_bull,
                mid_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )
            _process_box_datas(
                mid_tf_boxes_bear,
                mid_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )
        if current_tf_settings.use_box:
            _process_box_datas(
                current_tf_boxes_bull,
                current_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )
            _process_box_datas(
                current_tf_boxes_bear,
                current_tf_settings.close_count,
                high_price,
                low_price,
                close_price,
                bar_index,
            )

        if enable_cleanup and bar_index % 20 == 0:
            _cleanup_box_array(high_tf_boxes_bull, max_boxes)
            _cleanup_box_array(high_tf_boxes_bear, max_boxes)
            _cleanup_box_array(mid_tf_boxes_bull, max_boxes)
            _cleanup_box_array(mid_tf_boxes_bear, max_boxes)
            _cleanup_box_array(current_tf_boxes_bull, max_boxes)
            _cleanup_box_array(current_tf_boxes_bear, max_boxes)

        if bar_index < offset_fob:
            continue

        if high_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, high_tf):
            htf_cache = _htf_cache_from_aligned(high_tf_aligned, bar_index, high_tf)
            if htf_cache.is_new_bar and htf_cache.bar_index > offset_fob:
                can_bull = (
                    last_high_tf_fob_bar is None
                    or not last_high_tf_fob_bull
                    or (htf_cache.bar_index - last_high_tf_fob_bar) >= high_tf_settings.cooldown
                )
                can_bear = (
                    last_high_tf_fob_bar is None
                    or last_high_tf_fob_bull
                    or (htf_cache.bar_index - last_high_tf_fob_bar) >= high_tf_settings.cooldown
                )
                is_bull = can_bull and _is_fob_condition(True, htf_cache.high2, htf_cache.low)
                is_bear = can_bear and _is_fob_condition(False, htf_cache.low2, htf_cache.high)
                if is_bull:
                    result, data = _create_box_data(
                        "fob",
                        True,
                        htf_cache.low2,
                        htf_cache.high2,
                        high_tf,
                        htf_cache.bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        high_tf_boxes_bull.append(data)
                        last_high_tf_fob_bar = htf_cache.bar_index
                        last_high_tf_fob_bull = True
                if is_bear:
                    result, data = _create_box_data(
                        "fob",
                        False,
                        htf_cache.low2,
                        htf_cache.high2,
                        high_tf,
                        htf_cache.bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        high_tf_boxes_bear.append(data)
                        last_high_tf_fob_bar = htf_cache.bar_index
                        last_high_tf_fob_bull = False
                can_sob_bull = (
                    last_high_tf_sob_bar is None
                    or not last_high_tf_sob_bull
                    or (htf_cache.bar_index - last_high_tf_sob_bar) >= high_tf_settings.cooldown
                )
                can_sob_bear = (
                    last_high_tf_sob_bar is None
                    or last_high_tf_sob_bull
                    or (htf_cache.bar_index - last_high_tf_sob_bar) >= high_tf_settings.cooldown
                )
                is_sob_bull = can_sob_bull and _is_sob_condition(
                    True,
                    htf_cache.open2,
                    htf_cache.close1,
                    htf_cache.open1,
                    htf_cache.close,
                )
                is_sob_bear = can_sob_bear and _is_sob_condition(
                    False,
                    htf_cache.open2,
                    htf_cache.close1,
                    htf_cache.open1,
                    htf_cache.close,
                )
                if is_sob_bull:
                    result, data = _create_box_data(
                        "sob",
                        True,
                        htf_cache.low2,
                        htf_cache.high2,
                        high_tf,
                        htf_cache.bar_index,
                        bar_index - 1,
                        open1=htf_cache.open1,
                        close1=htf_cache.close1,
                    )
                    if result and data:
                        high_tf_boxes_bull.append(data)
                        last_high_tf_sob_bar = htf_cache.bar_index
                        last_high_tf_sob_bull = True
                if is_sob_bear:
                    result, data = _create_box_data(
                        "sob",
                        False,
                        htf_cache.low2,
                        htf_cache.high2,
                        high_tf,
                        htf_cache.bar_index,
                        bar_index - 1,
                        open1=htf_cache.open1,
                        close1=htf_cache.close1,
                    )
                    if result and data:
                        high_tf_boxes_bear.append(data)
                        last_high_tf_sob_bar = htf_cache.bar_index
                        last_high_tf_sob_bull = False

        if mid_tf_settings.use_box and _is_chart_tf_comparison_htf(chart_timeframe, mid_tf):
            mid_cache = _htf_cache_from_aligned(mid_tf_aligned, bar_index, mid_tf)
            if mid_cache.is_new_bar and mid_cache.bar_index > offset_fob:
                can_bull = (
                    last_mid_tf_fob_bar is None
                    or not last_mid_tf_fob_bull
                    or (mid_cache.bar_index - last_mid_tf_fob_bar) >= mid_tf_settings.cooldown
                )
                can_bear = (
                    last_mid_tf_fob_bar is None
                    or last_mid_tf_fob_bull
                    or (mid_cache.bar_index - last_mid_tf_fob_bar) >= mid_tf_settings.cooldown
                )
                is_bull = can_bull and _is_fob_condition(True, mid_cache.high2, mid_cache.low)
                is_bear = can_bear and _is_fob_condition(False, mid_cache.low2, mid_cache.high)
                if is_bull:
                    result, data = _create_box_data(
                        "fob",
                        True,
                        mid_cache.low2,
                        mid_cache.high2,
                        mid_tf,
                        mid_cache.bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        mid_tf_boxes_bull.append(data)
                        last_mid_tf_fob_bar = mid_cache.bar_index
                        last_mid_tf_fob_bull = True
                if is_bear:
                    result, data = _create_box_data(
                        "fob",
                        False,
                        mid_cache.low2,
                        mid_cache.high2,
                        mid_tf,
                        mid_cache.bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        mid_tf_boxes_bear.append(data)
                        last_mid_tf_fob_bar = mid_cache.bar_index
                        last_mid_tf_fob_bull = False
                can_sob_bull = (
                    last_mid_tf_sob_bar is None
                    or not last_mid_tf_sob_bull
                    or (mid_cache.bar_index - last_mid_tf_sob_bar) >= mid_tf_settings.cooldown
                )
                can_sob_bear = (
                    last_mid_tf_sob_bar is None
                    or last_mid_tf_sob_bull
                    or (mid_cache.bar_index - last_mid_tf_sob_bar) >= mid_tf_settings.cooldown
                )
                is_sob_bull = can_sob_bull and _is_sob_condition(
                    True,
                    mid_cache.open2,
                    mid_cache.close1,
                    mid_cache.open1,
                    mid_cache.close,
                )
                is_sob_bear = can_sob_bear and _is_sob_condition(
                    False,
                    mid_cache.open2,
                    mid_cache.close1,
                    mid_cache.open1,
                    mid_cache.close,
                )
                if is_sob_bull:
                    result, data = _create_box_data(
                        "sob",
                        True,
                        mid_cache.low2,
                        mid_cache.high2,
                        mid_tf,
                        mid_cache.bar_index,
                        bar_index - 1,
                        open1=mid_cache.open1,
                        close1=mid_cache.close1,
                    )
                    if result and data:
                        mid_tf_boxes_bull.append(data)
                        last_mid_tf_sob_bar = mid_cache.bar_index
                        last_mid_tf_sob_bull = True
                if is_sob_bear:
                    result, data = _create_box_data(
                        "sob",
                        False,
                        mid_cache.low2,
                        mid_cache.high2,
                        mid_tf,
                        mid_cache.bar_index,
                        bar_index - 1,
                        open1=mid_cache.open1,
                        close1=mid_cache.close1,
                    )
                    if result and data:
                        mid_tf_boxes_bear.append(data)
                        last_mid_tf_sob_bar = mid_cache.bar_index
                        last_mid_tf_sob_bull = False

        if current_tf_settings.use_box:
            if bar_index > offset_fob:
                can_bull = (
                    last_current_tf_fob_bar is None
                    or not last_current_tf_fob_bull
                    or (bar_index - last_current_tf_fob_bar) >= current_tf_settings.cooldown
                )
                can_bear = (
                    last_current_tf_fob_bar is None
                    or last_current_tf_fob_bull
                    or (bar_index - last_current_tf_fob_bar) >= current_tf_settings.cooldown
                )
                high2 = df["high"].iloc[bar_index - 2]
                low2 = df["low"].iloc[bar_index - 2]
                current_low = df["low"].iloc[bar_index]
                current_high = df["high"].iloc[bar_index]
                is_bull = can_bull and _is_fob_condition(True, high2, current_low)
                is_bear = can_bear and _is_fob_condition(False, low2, current_high)
                if is_bull:
                    result, data = _create_box_data(
                        "fob",
                        True,
                        low2,
                        high2,
                        chart_timeframe,
                        bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        current_tf_boxes_bull.append(data)
                        last_current_tf_fob_bar = bar_index
                        last_current_tf_fob_bull = True
                if is_bear:
                    result, data = _create_box_data(
                        "fob",
                        False,
                        low2,
                        high2,
                        chart_timeframe,
                        bar_index,
                        bar_index - offset_fob,
                    )
                    if result and data:
                        current_tf_boxes_bear.append(data)
                        last_current_tf_fob_bar = bar_index
                        last_current_tf_fob_bull = False
                can_sob_bull = (
                    last_current_tf_sob_bar is None
                    or not last_current_tf_sob_bull
                    or (bar_index - last_current_tf_sob_bar) >= current_tf_settings.cooldown
                )
                can_sob_bear = (
                    last_current_tf_sob_bar is None
                    or last_current_tf_sob_bull
                    or (bar_index - last_current_tf_sob_bar) >= current_tf_settings.cooldown
                )
                open2 = df["open"].iloc[bar_index - 2]
                close1 = df["close"].iloc[bar_index - 1]
                open1 = df["open"].iloc[bar_index - 1]
                close0 = df["close"].iloc[bar_index]
                is_sob_bull = can_sob_bull and _is_sob_condition(True, open2, close1, open1, close0)
                is_sob_bear = can_sob_bear and _is_sob_condition(False, open2, close1, open1, close0)
                if is_sob_bull:
                    result, data = _create_box_data(
                        "sob",
                        True,
                        low2,
                        high2,
                        chart_timeframe,
                        bar_index,
                        bar_index - 1,
                        open1=open1,
                        close1=close1,
                    )
                    if result and data:
                        current_tf_boxes_bull.append(data)
                        last_current_tf_sob_bar = bar_index
                        last_current_tf_sob_bull = True
                if is_sob_bear:
                    result, data = _create_box_data(
                        "sob",
                        False,
                        low2,
                        high2,
                        chart_timeframe,
                        bar_index,
                        bar_index - 1,
                        open1=open1,
                        close1=close1,
                    )
                    if result and data:
                        current_tf_boxes_bear.append(data)
                        last_current_tf_sob_bar = bar_index
                        last_current_tf_sob_bull = False

    high_tf_boxes = high_tf_boxes_bull + high_tf_boxes_bear
    mid_tf_boxes = mid_tf_boxes_bull + mid_tf_boxes_bear
    current_tf_boxes = current_tf_boxes_bull + current_tf_boxes_bear

    debug = {
        "high": len(high_tf_boxes),
        "mid": len(mid_tf_boxes),
        "current": len(current_tf_boxes),
        "total": len(high_tf_boxes) + len(mid_tf_boxes) + len(current_tf_boxes),
        "max": max_boxes,
    }

    return ThreepleOutputs(
        high_tf_boxes=high_tf_boxes,
        mid_tf_boxes=mid_tf_boxes,
        current_tf_boxes=current_tf_boxes,
        debug=debug,
    )
