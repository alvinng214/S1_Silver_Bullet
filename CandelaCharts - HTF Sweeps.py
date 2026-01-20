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
    if tf.endswith("H") and tf[:-1].isdigit():
        return int(tf[:-1]) * 60
    if tf.endswith("min") and tf[:-3].isdigit():
        return int(tf[:-3])
    if tf.endswith("m") and tf[:-1].isdigit():
        return int(tf[:-1])
    if tf.isdigit():
        return int(tf)
    if tf in {"1D", "1W", "1M"}:
        return {"1D": 1440, "1W": 10080, "1M": 43200}[tf]
    raise ValueError(f"Unsupported timeframe: {tf}")


def _tf_rule(tf: str) -> str:
    if tf in {"1D", "1W", "1M"}:
        return tf
    minutes = _parse_tf_to_minutes(tf)
    return f"{minutes}min"


def _floor_time(ts: pd.Timestamp, rule: str) -> pd.Timestamp:
    if rule in {"1D", "1W", "1M"}:
        return ts.to_period(rule[1]).start_time
    return ts.floor(rule)


def _update_htf_candles(
    candles: List[HTFCandle],
    df_index: pd.Index,
    tf: str,
    rule: str,
    idx: pd.Timestamp,
    row: pd.Series,
    limit: int,
) -> List[HTFCandle]:
    bucket = _floor_time(idx, rule)
    current = candles[-1] if candles else None
    current_bucket = _floor_time(current.ot, rule) if current else None

    if current is None or bucket != current_bucket:
        if current is not None:
            current.is_closed = True
        new_candle = HTFCandle(
            tf=tf,
            o=float(row["open"]),
            h=float(row["high"]),
            l=float(row["low"]),
            c=float(row["close"]),
            o_idx=df_index.get_loc(idx),
            c_idx=df_index.get_loc(idx),
            h_idx=df_index.get_loc(idx),
            l_idx=df_index.get_loc(idx),
            ot=idx,
            ct=idx,
            bull=_is_bullish_candle(float(row["close"]), float(row["open"]), float(row["high"]), float(row["low"])),
            is_closed=False,
        )
        candles.append(new_candle)
    else:
        current.c = float(row["close"])
        current.ct = idx
        current.c_idx = df_index.get_loc(idx)
        if float(row["high"]) > current.h:
            current.h = float(row["high"])
            current.h_idx = df_index.get_loc(idx)
        if float(row["low"]) < current.l:
            current.l = float(row["low"])
            current.l_idx = df_index.get_loc(idx)
        current.bull = _is_bullish_candle(current.c, current.o, current.h, current.l)

    if len(candles) > limit:
        return candles[-limit:]
    return candles


def _detect_sweep(c1: HTFCandle, c2: HTFCandle) -> Optional[Tuple[Sweep, Sweep]]:
    c1_bull = c1.bull
    c2_bull = c2.bull

    bull_sweep_in_range = (c2.c < c1.h) if c2_bull else (c2.o < c1.h)
    is_bull_sweep = c2.h > c1.h and bull_sweep_in_range

    bear_sweep_in_range = (c2.o > c1.l) if c2_bull else (c2.c > c1.l)
    is_bear_sweep = c2.l < c1.l and bear_sweep_in_range

    if is_bull_sweep and not c1.bull_sweep:
        htf_sweep = Sweep(x1=c1.h_idx, x2=c2.c_idx, y=c1.h, bull=True, tf=c1.tf)
        ltf_sweep = Sweep(x1=c1.h_idx, x2=c2.c_idx, y=c1.h, bull=True, tf=c1.tf)
        return htf_sweep, ltf_sweep
    if is_bear_sweep and not c1.bear_sweep:
        htf_sweep = Sweep(x1=c1.l_idx, x2=c2.c_idx, y=c1.l, bull=False, tf=c1.tf)
        ltf_sweep = Sweep(x1=c1.l_idx, x2=c2.c_idx, y=c1.l, bull=False, tf=c1.tf)
        return htf_sweep, ltf_sweep
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
        for sweep in c1.ltf_sweeps + c1.htf_sweeps:
            if sweep.removed or sweep.invalidated_on is not None:
                continue
            invalidated = _invalidate_sweep(sweep, c2)
            if sweep.x2 <= c2.c_idx and sweep.x2 >= c2.o_idx:
                sweep.invalidated = invalidated
                if invalidated:
                    sweep.invalidated_on = c2.o_idx
            elif invalidated:
                sweep.invalidated = True
                sweep.invalidated_on = c2.o_idx

        if c2.is_closed:
            for sweep in c1.ltf_sweeps + c1.htf_sweeps:
                if not sweep.formed and not sweep.removed:
                    if sweep.invalidated and sweep.invalidated_on is not None:
                        sweep.removed = True
                    else:
                        sweep.formed = True


def _detect_sweeps(candles: List[HTFCandle]) -> None:
    if len(candles) < 2:
        return
    size = min(4, len(candles) - 1)
    start = max(1, len(candles) - size - 1)
    for i in range(start, len(candles)):
        c1 = candles[i - 1]
        c2 = candles[i]
        if c2.is_closed or len(c1.htf_sweeps) > 2:
            continue
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
        rule = _tf_rule(tf)
        candles: List[HTFCandle] = []
        for idx, row in df.iterrows():
            candles = _update_htf_candles(candles, df.index, tf, rule, idx, row, count)
            _detect_sweeps(candles)
        output[tf] = candles

    return output
