"""CandelaCharts - HTF Sweeps

Python translation of the CandelaCharts HTF Sweeps Pine Script.

This module mirrors the Pine logic by:
- Building HTF candle sequences for multiple timeframes.
- Detecting sweeps of prior HTF highs/lows with in-range close rules.
- Tracking sweep invalidation and formation states.
- Computing LTF mapping data for HTF candle traces.

It returns structured data to allow downstream plotting or alerting in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Sweep:
    x1: int
    x2: int
    y: float
    bull: bool
    tf: str
    invalidated: bool = False
    invalidated_on: Optional[int] = None
    removed: bool = False
    formed: bool = False


@dataclass
class HTFCandle:
    tf: str
    o: float
    h: float
    l: float
    c: float
    o_idx: int
    c_idx: int
    h_idx: int
    l_idx: int
    ot: pd.Timestamp
    ct: pd.Timestamp
    bull: bool
    is_closed: bool = True
    htf_sweeps: List[Sweep] = field(default_factory=list)
    ltf_sweeps: List[Sweep] = field(default_factory=list)
    bull_sweep: bool = False
    bear_sweep: bool = False


def _is_bullish_candle(close: float, open_: float, high: float, low: float) -> bool:
    if close == open_:
        return abs(open_ - high) < abs(open_ - low)
    return close > open_


def _parse_tf_to_minutes(tf: str) -> int:
    tf = tf.strip()
    if tf.endswith("M") and tf[:-1].isdigit():
        return int(tf[:-1]) * 43200
    if tf.endswith("W") and tf[:-1].isdigit():
        return int(tf[:-1]) * 10080
    if tf.endswith("D") and tf[:-1].isdigit():
        return int(tf[:-1]) * 1440
    if tf.endswith("H") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("min") and tf[:-3].isdigit():
        return int(tf[:-3])
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.isdigit():
        return int(tf)
    if tf == "1D":
        return 1440
    if tf == "1W":
        return 10080
    if tf == "1M":
        return 43200
    raise ValueError(f"Unsupported timeframe: {tf}")


def _resample_htf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    minutes = _parse_tf_to_minutes(tf)
    resampled = (
        df[["open", "high", "low", "close"]]
        .resample(f"{minutes}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return resampled


def _build_htf_candles(df: pd.DataFrame, tf: str, limit: int) -> List[HTFCandle]:
    htf = _resample_htf(df, tf)
    candles: List[HTFCandle] = []

    for _, row in htf.tail(limit).iterrows():
        start_time = row.name
        end_time = start_time + (htf.index[1] - htf.index[0]) if len(htf) > 1 else start_time

        mask = (df.index >= start_time) & (df.index < end_time)
        if not mask.any():
            continue

        ltf_slice = df.loc[mask]
        o_idx = df.index.get_loc(ltf_slice.index[0])
        c_idx = df.index.get_loc(ltf_slice.index[-1])
        h_idx = df.index.get_loc(ltf_slice["high"].idxmax())
        l_idx = df.index.get_loc(ltf_slice["low"].idxmin())

        candle = HTFCandle(
            tf=tf,
            o=float(row["open"]),
            h=float(row["high"]),
            l=float(row["low"]),
            c=float(row["close"]),
            o_idx=o_idx,
            c_idx=c_idx,
            h_idx=h_idx,
            l_idx=l_idx,
            ot=start_time,
            ct=ltf_slice.index[-1],
            bull=_is_bullish_candle(float(row["close"]), float(row["open"]), float(row["high"]), float(row["low"])),
            is_closed=True,
        )
        candles.append(candle)

    return candles


def _detect_sweep(c1: HTFCandle, c2: HTFCandle) -> Optional[Tuple[Sweep, Sweep]]:
    c1_bull = c1.bull
    c2_bull = c2.bull

    bull_sweep_in_range = (c2.c < c1.h) if c2_bull else (c2.o < c1.h)
    is_bull_sweep = c2.h > c1.h and bull_sweep_in_range

    bear_sweep_in_range = (c2.o > c1.l) if c2_bull else (c2.c > c1.l)
    is_bear_sweep = c2.l < c1.l and bear_sweep_in_range

    if is_bull_sweep and not c1.bull_sweep:
        sweep = Sweep(x1=c1.h_idx, x2=c2.c_idx, y=c1.h, bull=True, tf=c1.tf)
        return sweep, sweep
    if is_bear_sweep and not c1.bear_sweep:
        sweep = Sweep(x1=c1.l_idx, x2=c2.c_idx, y=c1.l, bull=False, tf=c1.tf)
        return sweep, sweep
    return None


def _invalidate_sweep(sweep: Sweep, c2: HTFCandle) -> bool:
    c2_bull = c2.bull
    if sweep.bull:
        return sweep.y < (c2.c if c2_bull else c2.o)
    return sweep.y > (c2.o if c2_bull else c2.c)


def _invalidate_sweeps(candles: List[HTFCandle]) -> None:
    for i in range(1, len(candles)):
        c1 = candles[i - 1]
        c2 = candles[i]
        for sweep in c1.ltf_sweeps:
            if sweep.removed or sweep.invalidated_on is not None:
                continue
            invalidated = _invalidate_sweep(sweep, c2)

            if sweep.x2 <= c2.c_idx and sweep.x2 >= c2.o_idx:
                sweep.invalidated = invalidated
                if invalidated:
                    sweep.invalidated_on = c2.o_idx
            else:
                sweep.invalidated = invalidated
                if invalidated:
                    sweep.invalidated_on = c2.o_idx

            if sweep.invalidated and not sweep.formed:
                sweep.removed = True

        for sweep in c1.ltf_sweeps:
            if not sweep.formed and not sweep.removed:
                sweep.formed = True


def _detect_sweeps(candles: List[HTFCandle]) -> None:
    for i in range(1, len(candles)):
        c1 = candles[i - 1]
        c2 = candles[i]
        result = _detect_sweep(c1, c2)
        if result:
            htf_sweep, ltf_sweep = result
            c1.htf_sweeps.append(htf_sweep)
            c1.ltf_sweeps.append(ltf_sweep)
            c1.bull_sweep = htf_sweep.bull
            c1.bear_sweep = not htf_sweep.bull
    _invalidate_sweeps(candles)


def calculate_htf_sweeps(
    df: pd.DataFrame,
    *,
    timeframes: List[Tuple[str, int, bool]],
) -> Dict[str, List[HTFCandle]]:
    """Calculate HTF sweeps for configured timeframes.

    Args:
        df: DataFrame with datetime index and OHLC columns.
        timeframes: List of tuples (timeframe, number_of_candles, map_to_ltf).

    Returns:
        Dict mapping timeframe to list of HTFCandle objects.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    output: Dict[str, List[HTFCandle]] = {}
    for tf, count, _ in timeframes:
        candles = _build_htf_candles(df, tf, count)
        _detect_sweeps(candles)
        output[tf] = candles

    return output
