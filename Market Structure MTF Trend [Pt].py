"""Market Structure MTF Trend [Pt].

Python translation of the PtGambler Pine Script indicator.

This module mirrors the Pine logic:
- Per-timeframe market-structure trend detection using pivot highs/lows.
- Break of Structure (BoS) vs Change of Character (CHoCH) transitions.
- request.security-style MTF alignment with lookahead on/off behavior.
- Trend color state based on BoS/CHoCH transitions.
- Alert condition series for bullish/bearish CHoCH on each timeframe.

HTF publishing timing (aligned with ``Market Structure MTF Trend [Pt].cs``):
The Pine source published HTF values one full bar late on the chart, which
caused CHoCH / BoS labels to appear on the LTF bar AFTER the breakout window
had already ended. The ``.cs`` port fixed this by the
``ResolveTfBarForChartBar`` (``!IsLowerTf`` branch) helper:

    For each LTF bar at time T:
      - j = index of HTF bar at-or-before T
      - T' = next LTF bar's open time (with live-edge fallback = last diff)
      - k = index of HTF bar at-or-before T'
      - if k > j  -> this LTF bar is the LAST one in HTF bar j's window ->
                     publish HTF bar j (it is now confirmed/closed)
      - else      -> still inside the forming HTF bar -> publish j-1
                     (the most recently closed HTF bar)

This mirrors ``barmerge.lookahead_off`` without the full-HTF-period delay the
naive ``shift(1)`` approach introduced. The ``lookahead_on=True`` path and the
lower-TF (``IsLowerTf=True``) path are unchanged.
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

    # Pine convention: uppercase 'M' = month, lowercase 'm' / no-suffix = minute.
    # Do NOT lowercase the suffix — '1M' (month) and '1m' (minute) are different.
    suffix = timeframe[-1]
    value = timeframe[:-1]
    if not value:
        return None

    if suffix in ("h", "H"):
        return int(value) * 60
    if suffix in ("d", "D"):
        return int(value) * 60 * 24
    if suffix in ("w", "W"):
        return int(value) * 60 * 24 * 7
    if suffix == "M":  # month — must come BEFORE 'm' to avoid case collision
        return int(value) * 60 * 24 * 30
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
    """Resample OHLC data to a higher timeframe with TV-style bar labels.

    TradingView labels each HTF bar at the FIRST TRADING DAY in the bin,
    not the calendar period start. When a Monday is a holiday (MLK Day,
    Presidents Day), TV's weekly bar is labeled with that Tuesday's open;
    pandas' default ``W-MON`` keeps the calendar Monday label, which
    shifts pivot windows and causes CHoCH / BOS divergence around holiday
    weeks (e.g. 2020-01-20 → TV labels 2020-01-21).

    The fix: do the standard left-anchored resample to define the bins,
    then re-label each bin with its FIRST actual daily-bar timestamp.
    """
    if rule in ("1D",) or rule.endswith("min"):
        # No re-anchoring needed for intraday or daily — bin == bar.
        return (
            df[["open", "high", "low", "close"]]
            .resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )

    binned = (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    # For each bin, find the first daily-bar timestamp inside it. That's
    # TV's bar label.
    bin_starts = binned.index
    daily_idx = df.index
    new_labels = []
    for i, bin_start in enumerate(bin_starts):
        bin_end = bin_starts[i + 1] if i + 1 < len(bin_starts) else daily_idx[-1] + pd.Timedelta(days=1)
        first_in_bin = daily_idx[(daily_idx >= bin_start) & (daily_idx < bin_end)]
        if len(first_in_bin) > 0:
            new_labels.append(first_in_bin[0])
        else:
            new_labels.append(bin_start)
    binned.index = pd.DatetimeIndex(new_labels, name=daily_idx.name)
    return binned


def _resample_rule_for_minutes(tf_minutes: int) -> str:
    """Map an HTF minute count to a *calendar-aware* pandas resample rule.

    Pine's ``request.security`` aggregates by the exchange calendar
    (a week = Mon→Fri, a month = 1st→last), not by fixed-width minute
    bins anchored to the data's first timestamp. Using ``f"{tf_minutes}min"``
    for weekly/monthly drifts out of phase by up to ±N days as the data
    history grows, which manifests as wrong CHoCH / pivot dates against
    TradingView. The mappings below match TV's bar labels:

      * 1D  (1440 min)   → ``"1D"``     — daily bars
      * 1W  (10080 min)  → ``"W-MON"``  — week starts Mon, label = Mon open
      * 1M  (43200 min)  → ``"MS"``     — month-start, label = 1st of month
      * 3M  (129600 min) → ``"QS"``     — quarter-start
      * intraday minutes → ``f"{tf_minutes}min"`` (no calendar phase issue)

    Other multi-day buckets (2D / 3D / 5D / 2W) fall through to the
    minute-string form because they don't align to a single canonical
    exchange-calendar rule.
    """
    if tf_minutes == 60 * 24:                # 1D
        return "1D"
    if tf_minutes == 60 * 24 * 7:            # 1W
        return "W-MON"
    if tf_minutes == 60 * 24 * 30:           # 1M (the alias ict_structure emits)
        return "MS"
    if tf_minutes == 60 * 24 * 30 * 3:       # 3M
        return "QS"
    return f"{tf_minutes}min"


def _align_series(series: pd.Series, target_index: pd.Index, lookahead_on: bool) -> pd.Series:
    if series.dtype == object:
        series = series.infer_objects(copy=False)
    if lookahead_on:
        aligned = series.reindex(target_index, method="bfill")
        return aligned.ffill()
    return series.reindex(target_index, method="ffill").bfill()


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

    n = int(len(df))
    trend = np.full(n, False, dtype=object)
    bos = np.full(n, False, dtype=object)
    nat = np.datetime64("NaT")
    pivot_high_time = np.full(n, nat, dtype="datetime64[ns]")
    pivot_low_time = np.full(n, nat, dtype="datetime64[ns]")
    prev_pivot_high = np.full(n, np.nan, dtype=float)
    prev_pivot_low = np.full(n, np.nan, dtype=float)

    last_pivot_high = np.nan
    last_pivot_low = np.nan
    last_broken_high = np.nan
    last_broken_low = np.nan
    last_pivot_high_time = nat
    last_pivot_low_time = nat
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
                    last_pivot_high_time = np.datetime64(times[pivot_idx])

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
                    last_pivot_low_time = np.datetime64(times[pivot_idx])

        break_of_structure = False
        if not np.isnan(last_pivot_high):
            prev_close = closes[i - 1] if i > 0 else closes[i]
            if prev_close <= prev_last_pivot_high and closes[i] > last_pivot_high:
                break_of_structure = bool(current_trend and last_pivot_high != last_broken_high)
                current_trend = True
                last_broken_high = last_pivot_high
                last_broken_low = np.nan

        if not np.isnan(last_pivot_low):
            prev_close = closes[i - 1] if i > 0 else closes[i]
            if prev_close >= prev_last_pivot_low and closes[i] < last_pivot_low:
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


def _confirmed_htf_positions(
    ltf_index: pd.DatetimeIndex,
    htf_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Vectorized port of cAlgo's ``ResolveTfBarForChartBar`` (``!IsLowerTf`` branch).

    Returns an ``int64`` array of length ``len(ltf_index)``: at each LTF bar,
    the position of the HTF bar whose value is considered "confirmed and
    publishable" at that LTF bar. Returns ``-1`` where no HTF bar has yet
    closed (pre-history).

    Semantics:
      - At an LTF bar inside a forming HTF bar (i.e. the next LTF bar still
        belongs to the same HTF window) -> position = j - 1 (the previously
        closed HTF bar).
      - At the LTF bar whose close coincides with the HTF bar's close (i.e.
        the next LTF bar starts a later HTF window) -> position = j (that
        HTF bar is now confirmed).

    The last LTF bar has no "next LTF bar" available; we estimate its
    neighbour using the last pairwise diff, matching the ``.cs`` fallback
    ``Bars.OpenTimes[Count-1] - Bars.OpenTimes[Count-2]``.
    """
    n = len(ltf_index)
    if n == 0:
        return np.array([], dtype=np.int64)
    if len(htf_index) == 0:
        return np.full(n, -1, dtype=np.int64)

    ltf_arr = np.asarray(ltf_index.values, dtype="datetime64[ns]")
    htf_arr = np.asarray(htf_index.values, dtype="datetime64[ns]")

    # j[i] = largest k s.t. htf_arr[k] <= ltf_arr[i]; -1 when none.
    j = np.searchsorted(htf_arr, ltf_arr, side="right") - 1

    if n >= 2:
        last_span = ltf_arr[-1] - ltf_arr[-2]
        if last_span <= np.timedelta64(0, "ns"):
            last_span = np.timedelta64(1, "m")
    else:
        last_span = np.timedelta64(1, "m")

    next_ltf = np.empty(n, dtype="datetime64[ns]")
    if n >= 2:
        next_ltf[:-1] = ltf_arr[1:]
    next_ltf[-1] = ltf_arr[-1] + last_span

    k = np.searchsorted(htf_arr, next_ltf, side="right") - 1

    confirmed = np.where(k > j, j, j - 1).astype(np.int64)
    return confirmed


def _publish_htf_to_ltf(
    htf_series: pd.Series,
    ltf_index: pd.DatetimeIndex,
    confirmed_pos: np.ndarray,
) -> pd.Series:
    """Map an HTF-indexed series onto an LTF index via confirmed-bar positions.

    ``confirmed_pos`` entries that equal -1 (no HTF bar confirmed yet) map
    to a dtype-appropriate "missing" sentinel:
      - floats   -> ``np.nan``
      - datetime -> ``NaT``
      - object / bool -> ``False`` (matches the ``.cs`` initial-state defaults
        for ``Trend`` and ``Bos``).
    """
    n = len(ltf_index)
    if n == 0:
        return pd.Series([], index=ltf_index, dtype=htf_series.dtype)

    htf_vals = htf_series.to_numpy()
    if len(htf_vals) == 0:
        # No HTF bars at all -> degrade to "no data" per dtype.
        if np.issubdtype(htf_series.dtype, np.floating):
            return pd.Series(np.full(n, np.nan), index=ltf_index)
        if np.issubdtype(htf_series.dtype, np.datetime64):
            return pd.Series(
                np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
                index=ltf_index,
            )
        return pd.Series([False] * n, index=ltf_index, dtype=object)

    safe_pos = np.where(confirmed_pos >= 0, confirmed_pos, 0)
    picked = htf_vals[safe_pos]
    invalid = confirmed_pos < 0

    if not invalid.any():
        return pd.Series(picked, index=ltf_index)

    if np.issubdtype(picked.dtype, np.floating):
        out = picked.astype(float, copy=True)
        out[invalid] = np.nan
    elif np.issubdtype(picked.dtype, np.datetime64):
        out = picked.astype("datetime64[ns]", copy=True)
        out[invalid] = np.datetime64("NaT")
    else:
        out = picked.astype(object, copy=True)
        out[invalid] = False

    return pd.Series(out, index=ltf_index)


def _align_lower_tf_series(
    series: MarketStructureSeries,
    base_index: pd.Index,
    base_rule: str,
    lookahead_on: bool,
) -> MarketStructureSeries:
    def _downsample(values: pd.Series) -> pd.Series:
        resampled = values.resample(base_rule, label="left", closed="left").last().dropna()
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
    """Calculate market structure for a specific timeframe.

    For HTF data with lookahead_off:
    - The HTF bar's value is only available AFTER the bar closes
    - We shift the HTF series by 1 period before aligning to LTF
    - This matches TradingView's request.security behavior
    """
    base_minutes = _infer_base_minutes(df)
    tf_minutes = _parse_timeframe_to_minutes(timeframe)

    if tf_minutes is None or tf_minutes == base_minutes:
        series = calculate_market_structure_trend(df, pivot_len)
        return series

    if tf_minutes < base_minutes:
        if lower_tf_data is None or lower_tf_data.empty:
            return calculate_market_structure_trend(df, pivot_len)
        lower_series = calculate_market_structure_trend(lower_tf_data, pivot_len)
        base_rule = _resample_rule_for_minutes(base_minutes)
        return _align_lower_tf_series(lower_series, df.index, base_rule, lookahead_on)

    rule = _resample_rule_for_minutes(tf_minutes)
    htf = _resample_ohlc(df, rule)
    htf_series = calculate_market_structure_trend(htf, pivot_len)

    if lookahead_on:
        # Mirror Pine ``barmerge.lookahead_on``: the HTF value is visible from
        # the HTF bar's open, i.e. any contemporaneous LTF bar sees it.
        aligned_trend = _align_series(htf_series.trend, df.index, True)
        aligned_bos = _align_series(htf_series.bos, df.index, True)
        aligned_ph_time = _align_series(htf_series.pivot_high_time, df.index, True)
        aligned_pl_time = _align_series(htf_series.pivot_low_time, df.index, True)
        aligned_prev_ph = _align_series(htf_series.prev_pivot_high, df.index, True)
        aligned_prev_pl = _align_series(htf_series.prev_pivot_low, df.index, True)
    else:
        # Mirror Pine ``barmerge.lookahead_off`` via the .cs
        # ``ResolveTfBarForChartBar`` confirmed-bar mapping. The HTF bar's
        # value is published to the LTF bar whose close coincides with the
        # HTF bar's close (not one full HTF period later, as the naive
        # ``shift(1)`` approach produced).
        confirmed_pos = _confirmed_htf_positions(df.index, htf.index)
        aligned_trend = _publish_htf_to_ltf(htf_series.trend, df.index, confirmed_pos)
        aligned_bos = _publish_htf_to_ltf(htf_series.bos, df.index, confirmed_pos)
        aligned_ph_time = _publish_htf_to_ltf(htf_series.pivot_high_time, df.index, confirmed_pos)
        aligned_pl_time = _publish_htf_to_ltf(htf_series.pivot_low_time, df.index, confirmed_pos)
        aligned_prev_ph = _publish_htf_to_ltf(htf_series.prev_pivot_high, df.index, confirmed_pos)
        aligned_prev_pl = _publish_htf_to_ltf(htf_series.prev_pivot_low, df.index, confirmed_pos)

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
    trend_bool = series.trend.astype(bool)
    bos_bool = series.bos.astype(bool)
    # Mirror Pine's ta.change() on the first bar: it returns 0 (false), not NaN.
    # Seed shift(1) with the first bar's own value so bar-0 diff collapses to False.
    first_trend = bool(trend_bool.iloc[0]) if len(trend_bool) else False
    trend_change = trend_bool.ne(trend_bool.shift(1, fill_value=first_trend))
    bos_edge = bos_bool & ~bos_bool.shift(1, fill_value=False)
    color = _trend_color_series(trend_bool, bos_bool, trend_change, choch_bull, choch_bear, bos_bull, bos_bear)
    bullish_choch = trend_change & trend_bool
    bearish_choch = trend_change & ~trend_bool

    return TrendOutputs(
        data=MarketStructureSeries(
            trend=trend_bool,
            bos=bos_bool,
            pivot_high_time=series.pivot_high_time,
            pivot_low_time=series.pivot_low_time,
            prev_pivot_high=series.prev_pivot_high,
            prev_pivot_low=series.prev_pivot_low,
        ),
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
