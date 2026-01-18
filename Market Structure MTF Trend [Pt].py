"""Market Structure MTF Trend [Pt].

Python translation of the PtGambler Pine Script indicator.

This module mirrors the Pine logic:
- Per-timeframe market-structure trend detection using pivot highs/lows.
- Break of Structure (BoS) vs Change of Character (CHoCH) transitions.
- request.security-style MTF alignment with lookahead on/off behavior.
- Trend color state based on BoS/CHoCH transitions.
- Alert condition series for bullish/bearish CHoCH on each timeframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MarketStructureSeries:
    trend: pd.Series
    bos: pd.Series
    pivot_high_time: pd.Series
    pivot_low_time: pd.Series
    prev_pivot_high: pd.Series
    prev_pivot_low: pd.Series


@dataclass
class TrendOutputs:
    data: MarketStructureSeries
    trend_change: pd.Series
    bos_edge: pd.Series
    color: pd.Series
    bullish_choch: pd.Series
    bearish_choch: pd.Series


@dataclass
class MarketStructureMTFOutputs:
    tf1: TrendOutputs
    tf2: TrendOutputs
    tf3: TrendOutputs
    tf4: TrendOutputs
    timeframe_labels: Dict[str, str]
    tf_mismatch_higher: bool
    tf_mismatch_lower: bool


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


def _timeframe_label(timeframe: str) -> str:
    minutes = _parse_timeframe_to_minutes(timeframe)
    if minutes is None:
        return timeframe
    if minutes % (60 * 24 * 7) == 0:
        return f"{minutes // (60 * 24 * 7)}W"
    if minutes % (60 * 24) == 0:
        return f"{minutes // (60 * 24)}D"
    if minutes % 60 == 0:
        return f"{minutes // 60}H"
    return f"{minutes}m"


def _infer_base_minutes(df: pd.DataFrame) -> int:
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return 0
    return int(diffs.median().total_seconds() // 60)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _align_series(series: pd.Series, target_index: pd.Index, lookahead_on: bool) -> pd.Series:
    if lookahead_on:
        aligned = series.reindex(target_index, method="bfill")
        return aligned.fillna(method="ffill")
    return series.reindex(target_index, method="ffill").fillna(method="bfill")


def _is_pivot_high(highs: np.ndarray, idx: int, pivot_len: int) -> bool:
    left = idx - pivot_len
    right = idx + pivot_len
    if left < 0 or right >= len(highs):
        return False
    pivot = highs[idx]
    return pivot == np.max(highs[left : right + 1]) and np.sum(highs[left : right + 1] == pivot) == 1


def _is_pivot_low(lows: np.ndarray, idx: int, pivot_len: int) -> bool:
    left = idx - pivot_len
    right = idx + pivot_len
    if left < 0 or right >= len(lows):
        return False
    pivot = lows[idx]
    return pivot == np.min(lows[left : right + 1]) and np.sum(lows[left : right + 1] == pivot) == 1


def calculate_market_structure_trend(df: pd.DataFrame, pivot_len: int) -> MarketStructureSeries:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df.index

    n = len(df)
    trend = np.full(n, False, dtype=object)
    bos = np.full(n, False, dtype=object)
    pivot_high_time = np.full(n, pd.NaT, dtype="datetime64[ns]")
    pivot_low_time = np.full(n, pd.NaT, dtype="datetime64[ns]")
    prev_pivot_high = np.full(n, np.nan, dtype=float)
    prev_pivot_low = np.full(n, np.nan, dtype=float)

    last_pivot_high = np.nan
    last_pivot_low = np.nan
    last_broken_high = np.nan
    last_broken_low = np.nan
    last_pivot_high_time = pd.NaT
    last_pivot_low_time = pd.NaT
    current_trend = False

    for i in range(n):
        prev_last_pivot_high = last_pivot_high
        prev_last_pivot_low = last_pivot_low

        if i >= pivot_len * 2:
            pivot_idx = i - pivot_len
            if _is_pivot_high(highs, pivot_idx, pivot_len):
                pivot_price = highs[pivot_idx]
                if current_trend:
                    last_pivot_high = (
                        np.nanmax([pivot_price, last_pivot_high])
                        if not np.isnan(last_pivot_high)
                        else pivot_price
                    )
                else:
                    last_pivot_high = pivot_price
                if last_pivot_high != prev_last_pivot_high:
                    last_pivot_high_time = times[pivot_idx]

            if _is_pivot_low(lows, pivot_idx, pivot_len):
                pivot_price = lows[pivot_idx]
                if not current_trend:
                    last_pivot_low = (
                        np.nanmin([pivot_price, last_pivot_low])
                        if not np.isnan(last_pivot_low)
                        else pivot_price
                    )
                else:
                    last_pivot_low = pivot_price
                if last_pivot_low != prev_last_pivot_low:
                    last_pivot_low_time = times[pivot_idx]

        break_of_structure = False
        if not np.isnan(last_pivot_high):
            prev_close = closes[i - 1] if i > 0 else closes[i]
            if prev_close <= last_pivot_high and closes[i] > last_pivot_high:
                break_of_structure = bool(current_trend and last_pivot_high != last_broken_high)
                current_trend = True
                last_broken_high = last_pivot_high
                last_broken_low = np.nan

        if not np.isnan(last_pivot_low):
            prev_close = closes[i - 1] if i > 0 else closes[i]
            if prev_close >= last_pivot_low and closes[i] < last_pivot_low:
                break_of_structure = bool((not current_trend) and last_pivot_low != last_broken_low)
                current_trend = False
                last_broken_low = last_pivot_low
                last_broken_high = np.nan

        trend[i] = current_trend
        bos[i] = break_of_structure
        pivot_high_time[i] = last_pivot_high_time
        pivot_low_time[i] = last_pivot_low_time
        prev_pivot_high[i] = prev_last_pivot_high
        prev_pivot_low[i] = prev_last_pivot_low

    return MarketStructureSeries(
        trend=pd.Series(trend, index=df.index),
        bos=pd.Series(bos, index=df.index),
        pivot_high_time=pd.Series(pivot_high_time, index=df.index),
        pivot_low_time=pd.Series(pivot_low_time, index=df.index),
        prev_pivot_high=pd.Series(prev_pivot_high, index=df.index),
        prev_pivot_low=pd.Series(prev_pivot_low, index=df.index),
    )


def _trend_color_series(
    trend: pd.Series,
    bos: pd.Series,
    trend_change: pd.Series,
    choch_bull: Tuple[int, int, int],
    choch_bear: Tuple[int, int, int],
    bos_bull: str,
    bos_bear: str,
) -> pd.Series:
    colors: List[Optional[object]] = []
    current_color: Optional[object] = None
    for idx in range(len(trend)):
        if bool(bos.iloc[idx]):
            current_color = bos_bull if bool(trend.iloc[idx]) else bos_bear
        elif bool(trend_change.iloc[idx]):
            current_color = choch_bull if bool(trend.iloc[idx]) else choch_bear
        colors.append(current_color)
    return pd.Series(colors, index=trend.index)


def _align_lower_tf_series(
    series: MarketStructureSeries,
    base_index: pd.Index,
    base_rule: str,
    lookahead_on: bool,
) -> MarketStructureSeries:
    def _downsample(values: pd.Series) -> pd.Series:
        resampled = values.resample(base_rule, label="right", closed="right").last().dropna()
        return _align_series(resampled, base_index, lookahead_on)

    return MarketStructureSeries(
        trend=_downsample(series.trend),
        bos=_downsample(series.bos),
        pivot_high_time=_downsample(series.pivot_high_time),
        pivot_low_time=_downsample(series.pivot_low_time),
        prev_pivot_high=_downsample(series.prev_pivot_high),
        prev_pivot_low=_downsample(series.prev_pivot_low),
    )


def _market_structure_for_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    pivot_len: int,
    lookahead_on: bool,
    lower_tf_data: Optional[pd.DataFrame] = None,
) -> MarketStructureSeries:
    base_minutes = _infer_base_minutes(df)
    tf_minutes = _parse_timeframe_to_minutes(timeframe)

    if tf_minutes is None or tf_minutes == base_minutes:
        series = calculate_market_structure_trend(df, pivot_len)
        return series

    if tf_minutes < base_minutes:
        if lower_tf_data is None or lower_tf_data.empty:
            return calculate_market_structure_trend(df, pivot_len)
        lower_series = calculate_market_structure_trend(lower_tf_data, pivot_len)
        base_rule = f"{base_minutes}min"
        return _align_lower_tf_series(lower_series, df.index, base_rule, lookahead_on)

    rule = f"{tf_minutes}min"
    htf = _resample_ohlc(df, rule)
    htf_series = calculate_market_structure_trend(htf, pivot_len)

    aligned_trend = _align_series(htf_series.trend, df.index, lookahead_on)
    aligned_bos = _align_series(htf_series.bos, df.index, lookahead_on)
    aligned_ph_time = _align_series(htf_series.pivot_high_time, df.index, lookahead_on)
    aligned_pl_time = _align_series(htf_series.pivot_low_time, df.index, lookahead_on)
    aligned_prev_ph = _align_series(htf_series.prev_pivot_high, df.index, lookahead_on)
    aligned_prev_pl = _align_series(htf_series.prev_pivot_low, df.index, lookahead_on)

    return MarketStructureSeries(
        trend=aligned_trend,
        bos=aligned_bos,
        pivot_high_time=aligned_ph_time,
        pivot_low_time=aligned_pl_time,
        prev_pivot_high=aligned_prev_ph,
        prev_pivot_low=aligned_prev_pl,
    )


def _build_trend_output(
    series: MarketStructureSeries,
    choch_bull: Tuple[int, int, int],
    choch_bear: Tuple[int, int, int],
    bos_bull: str,
    bos_bear: str,
) -> TrendOutputs:
    trend_change = series.trend.ne(series.trend.shift(1)).fillna(False)
    bos_edge = series.bos & ~series.bos.shift(1).fillna(False)
    color = _trend_color_series(series.trend, series.bos, trend_change, choch_bull, choch_bear, bos_bull, bos_bear)
    bullish_choch = trend_change & series.trend
    bearish_choch = trend_change & ~series.trend

    return TrendOutputs(
        data=series,
        trend_change=trend_change,
        bos_edge=bos_edge,
        color=color,
        bullish_choch=bullish_choch,
        bearish_choch=bearish_choch,
    )


def calculate_market_structure_mtf(
    df: pd.DataFrame,
    *,
    timeframes: Tuple[str, str, str, str] = ("15", "30", "60", "240"),
    pivot_strengths: Tuple[int, int, int, int] = (15, 15, 15, 15),
    is_lower_tf: Tuple[bool, bool, bool, bool] = (False, False, False, False),
    lower_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
    choch_bull_colors: Tuple[Tuple[int, int, int], ...] = (
        (46, 104, 48),
        (46, 104, 48),
        (46, 104, 48),
        (46, 104, 48),
    ),
    choch_bear_colors: Tuple[Tuple[int, int, int], ...] = (
        (128, 41, 41),
        (128, 41, 41),
        (128, 41, 41),
        (128, 41, 41),
    ),
    bos_bull_colors: Tuple[str, ...] = ("green", "green", "green", "green"),
    bos_bear_colors: Tuple[str, ...] = ("red", "red", "red", "red"),
) -> MarketStructureMTFOutputs:
    base_minutes = _infer_base_minutes(df)
    tf_minutes = [_parse_timeframe_to_minutes(tf) for tf in timeframes]

    tf_mismatch_higher = False
    tf_mismatch_lower = False
    for tf_min, lower_tf in zip(tf_minutes, is_lower_tf):
        if tf_min is None or base_minutes == 0:
            continue
        if not lower_tf and base_minutes > tf_min:
            tf_mismatch_higher = True
        if lower_tf and base_minutes < tf_min:
            tf_mismatch_lower = True

    lower_tf_data = lower_tf_data or {}
    tf1_series = _market_structure_for_timeframe(
        df, timeframes[0], pivot_strengths[0], is_lower_tf[0], lower_tf_data.get(timeframes[0])
    )
    tf2_series = _market_structure_for_timeframe(
        df, timeframes[1], pivot_strengths[1], is_lower_tf[1], lower_tf_data.get(timeframes[1])
    )
    tf3_series = _market_structure_for_timeframe(
        df, timeframes[2], pivot_strengths[2], is_lower_tf[2], lower_tf_data.get(timeframes[2])
    )
    tf4_series = _market_structure_for_timeframe(
        df, timeframes[3], pivot_strengths[3], is_lower_tf[3], lower_tf_data.get(timeframes[3])
    )

    tf1_output = _build_trend_output(
        tf1_series, choch_bull_colors[0], choch_bear_colors[0], bos_bull_colors[0], bos_bear_colors[0]
    )
    tf2_output = _build_trend_output(
        tf2_series, choch_bull_colors[1], choch_bear_colors[1], bos_bull_colors[1], bos_bear_colors[1]
    )
    tf3_output = _build_trend_output(
        tf3_series, choch_bull_colors[2], choch_bear_colors[2], bos_bull_colors[2], bos_bear_colors[2]
    )
    tf4_output = _build_trend_output(
        tf4_series, choch_bull_colors[3], choch_bear_colors[3], bos_bull_colors[3], bos_bear_colors[3]
    )

    timeframe_labels = {
        "tf1": _timeframe_label(timeframes[0]),
        "tf2": _timeframe_label(timeframes[1]),
        "tf3": _timeframe_label(timeframes[2]),
        "tf4": _timeframe_label(timeframes[3]),
    }

    return MarketStructureMTFOutputs(
        tf1=tf1_output,
        tf2=tf2_output,
        tf3=tf3_output,
        tf4=tf4_output,
        timeframe_labels=timeframe_labels,
        tf_mismatch_higher=tf_mismatch_higher,
        tf_mismatch_lower=tf_mismatch_lower,
    )
