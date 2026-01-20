"""Ighodalo Gold - CRT (Candles are Ranges Theory).

Python translation of the Pine Script that:
- Detects CRT ranges across up to four timeframes.
- Tracks active CRTs with optional overlap control.
- Emits Turtle Soup signals with ATR tolerance.
- Records per-timeframe alert fires aligned to HTF bar closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CRTLevel:
    high: float
    low: float
    mid: float
    detected_time: pd.Timestamp
    active: bool
    buy_signaled: bool
    sell_signaled: bool
    start_index: int
    end_index: Optional[int] = None


@dataclass
class CRTManagerState:
    timeframe: str
    label: str
    levels: List[CRTLevel] = field(default_factory=list)


@dataclass
class SignalEvent:
    index: int
    direction: str
    price: float
    level: float
    timeframe: str


@dataclass
class AlertFire:
    index: int
    direction: str
    timeframe: str


def _timeframe_rule(tf: str) -> Optional[str]:
    if tf == "":
        return None
    if tf.isdigit():
        return f"{int(tf)}min"
    if tf in {"D", "W", "M"}:
        return tf
    raise ValueError(f"Unsupported timeframe: {tf}")


def _floor_time(ts: pd.Timestamp, rule: str) -> pd.Timestamp:
    if rule in {"D", "W", "M"}:
        return ts.to_period(rule).start_time
    return ts.floor(rule)


def _build_htf_view(df: pd.DataFrame, rule: str) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[bool]]:
    htf = (
        df.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    buckets = [_floor_time(ts, rule) for ts in df.index]
    bucket_series = pd.Series(buckets, index=df.index)
    last_in_bucket = bucket_series.groupby(bucket_series).transform("size")
    pos_in_bucket = bucket_series.groupby(bucket_series).cumcount()
    htf_closed = (pos_in_bucket + 1) == last_in_bucket
    return htf, buckets, list(htf_closed)


def _crt_calc_all(high: pd.Series, low: pd.Series, times: pd.Index, lookback: int, end_idx: int) -> Tuple[List[float], List[float], List[pd.Timestamp]]:
    highs: List[float] = []
    lows: List[float] = []
    tvals: List[pd.Timestamp] = []
    if end_idx < 0:
        return highs, lows, tvals
    for k in range(1, lookback + 1):
        idx = end_idx - k
        if idx < 0:
            continue
        pot_high = float(high.iloc[idx])
        pot_low = float(low.iloc[idx])
        recent_high = float(high.iloc[end_idx - k + 1 : end_idx + 1].max())
        recent_low = float(low.iloc[end_idx - k + 1 : end_idx + 1].min())
        within = recent_high <= pot_high and recent_low >= pot_low
        hh = float(high.iloc[end_idx - k : end_idx + 1].max())
        ll = float(low.iloc[end_idx - k : end_idx + 1].min())
        unique = True
        for j in range(0, k):
            if float(high.iloc[end_idx - j]) == hh and float(low.iloc[end_idx - j]) == ll:
                unique = False
                break
        if within and unique:
            highs.append(pot_high)
            lows.append(pot_low)
            tvals.append(times[idx])
    return highs, lows, tvals


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _tf_label(tf: str) -> str:
    per = tf if tf else ""
    mapping = {
        "1": "1m",
        "3": "3m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "45": "45m",
        "60": "H1",
        "120": "H2",
        "180": "H3",
        "240": "H4",
        "360": "H6",
        "480": "H8",
        "720": "H12",
        "D": "D1",
        "W": "W1",
        "M": "M1",
    }
    return mapping.get(per, per or "")


def calculate_ighodalo_crt(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    enable_overlapping: bool = False,
    enable_turtle_soup: bool = True,
    enable_tolerance: bool = True,
    atr_len: int = 14,
    atr_mult: float = 0.1,
    tf1: str = "60",
    tf2: str = "",
    tf3: str = "",
    tf4: str = "",
) -> Dict[str, object]:
    """Replicate CRT detection and Turtle Soup signals across timeframes."""
    managers: List[CRTManagerState] = []
    for tf in (tf1, tf2, tf3, tf4):
        if tf:
            managers.append(CRTManagerState(timeframe=tf, label=_tf_label(tf)))

    atr = _atr(df, atr_len)

    htf_cache: Dict[str, Tuple[pd.DataFrame, List[pd.Timestamp], List[bool]]] = {}
    for mgr in managers:
        rule = _timeframe_rule(mgr.timeframe)
        if rule is None:
            htf_cache[mgr.timeframe] = (df, list(df.index), [True] * len(df))
        else:
            htf_cache[mgr.timeframe] = _build_htf_view(df, rule)

    signals: List[SignalEvent] = []
    alerts: List[AlertFire] = []

    tf_fire = {"buy": [False] * 4, "sell": [False] * 4}

    for i in range(len(df)):
        row = df.iloc[i]
        tol = float(atr.iloc[i]) * atr_mult if enable_tolerance and not pd.isna(atr.iloc[i]) else 0.0
        buy_signal = False
        sell_signal = False

        for idx, mgr in enumerate(managers):
            htf_df, buckets, closed_flags = htf_cache[mgr.timeframe]
            rule = _timeframe_rule(mgr.timeframe)
            if rule is None:
                effective_bucket = buckets[i]
            else:
                current_bucket = buckets[i]
                if closed_flags[i]:
                    effective_bucket = current_bucket
                else:
                    bucket_pos = htf_df.index.get_loc(current_bucket) if current_bucket in htf_df.index else None
                    if bucket_pos is None or bucket_pos == 0:
                        effective_bucket = current_bucket
                    else:
                        effective_bucket = htf_df.index[bucket_pos - 1]
            if rule is None:
                effective_idx = i
            else:
                effective_idx = htf_df.index.get_loc(effective_bucket) if effective_bucket in htf_df.index else -1

            htf_closed = closed_flags[i]
            htf_close = float(htf_df["close"].iloc[effective_idx]) if effective_idx >= 0 else float(row["close"])

            has_active = any(level.active for level in mgr.levels)
            detect_new = enable_overlapping or not has_active

            if detect_new:
                highs, lows, times = _crt_calc_all(
                    htf_df["high"],
                    htf_df["low"],
                    htf_df.index,
                    lookback,
                    effective_idx,
                )
                num = min(len(highs), len(lows), len(times))
                if num > 0 and htf_closed:
                    if enable_overlapping:
                        indices = range(num)
                    else:
                        indices = [num - 1]
                    for m in indices:
                        c_time = times[m]
                        if any(level.detected_time == c_time for level in mgr.levels):
                            continue
                        if not enable_overlapping:
                            for level in mgr.levels:
                                if level.active:
                                    level.active = False
                        p_high = highs[m]
                        p_low = lows[m]
                        mid = (p_high + p_low) / 2
                        mgr.levels.append(
                            CRTLevel(
                                high=p_high,
                                low=p_low,
                                mid=mid,
                                detected_time=c_time,
                                active=True,
                                buy_signaled=False,
                                sell_signaled=False,
                                start_index=i,
                            )
                        )

            if enable_turtle_soup:
                for level in reversed(mgr.levels):
                    if not level.active:
                        continue
                    break_cond = htf_close > level.high or htf_close < level.low
                    if break_cond:
                        level.active = False
                        level.end_index = i - 1 if i > 0 else i

                    if not level.buy_signaled and row["low"] <= (level.low + tol) and row["close"] > (level.low - tol):
                        level.buy_signaled = True
                        buy_signal = True
                        signals.append(
                            SignalEvent(i, "buy", float(row["close"]), level.low, mgr.label)
                        )
                        if htf_closed:
                            tf_fire["buy"][idx] = True
                            alerts.append(AlertFire(i, "buy", mgr.label))

                    if not level.sell_signaled and row["high"] >= (level.high - tol) and row["close"] < (level.high + tol):
                        level.sell_signaled = True
                        sell_signal = True
                        signals.append(
                            SignalEvent(i, "sell", float(row["close"]), level.high, mgr.label)
                        )
                        if htf_closed:
                            tf_fire["sell"][idx] = True
                            alerts.append(AlertFire(i, "sell", mgr.label))

    return {
        "managers": managers,
        "signals": signals,
        "alerts": alerts,
        "tf_fire": tf_fire,
    }
