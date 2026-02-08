"""Python translation of the Pine Script `MTF Order Block Finder`.

This module mirrors the Pine logic by:
- Detecting bullish/bearish order blocks from higher timeframes.
- Filtering patterns by required move %, doji thresholds, and fuzzy candles.
- Selecting order block sources via High/Low, OHLC, or Context wick search.
- Tracking box/line objects and enforcing per-side limits.

TradingView drawing primitives are represented as data structures and
returned to the caller for downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class OBZone:
    direction: str  # "bull" or "bear"
    source_time: pd.Timestamp
    high: float
    low: float
    avg: float
    selector_shift: int


@dataclass
class OBSettings:
    resolution: str = ""
    ob_period: int = 5
    threshold: float = 0.3
    bull_channels: int = 4
    bear_channels: int = 4
    doji: float = 0.05
    fuzzy: float = 0.01
    near_price: float = 0.0
    style: str = "BOX"  # BOTH, BOX, LINE
    ob_shift: int = 1
    ob_selector: str = "OHLC"  # High/Low, OHLC, Context
    ob_search: int = 2


def _resolution_to_minutes(res: str) -> int:
    mapping = {
        "1": 1,
        "3": 3,
        "5": 5,
        "10": 10,
        "15": 15,
        "30": 30,
        "45": 45,
        "60": 60,
        "120": 120,
        "180": 180,
        "240": 240,
        "1D": 1440,
        "1W": 10080,
        "1M": 43200,
    }
    return mapping.get(res, 0)


def _timeframe_to_minutes(index: pd.Index) -> int:
    if len(index) < 2:
        return 0
    delta = index.to_series().diff().dropna().median()
    return int(delta.total_seconds() // 60)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _resample_rule_from_resolution(res: str) -> str:
    if res.endswith("D") or res.endswith("W") or res.endswith("M"):
        return res
    return f"{res}min"


def _remove_gaps(series: pd.Series, warmup: int) -> Tuple[List[float], bool]:
    values: List[float] = []
    if series.isna().all() or len(series) <= warmup:
        values.append(np.nan)
        return values, False
    for i in range(warmup + 1):
        val = series.iloc[-(i + 1)]
        if not pd.isna(val):
            values.append(val)
    # Pine array indexing: index 0 == most recent bar.
    return values, not series.isna().iloc[-1]


def _is_zero_candle(idx: int, o: List[float], h: List[float], l: List[float], c: List[float]) -> bool:
    base = o[idx]
    return base == c[idx] and base == h[idx] and base == l[idx]


def _low_wick_search(start_index: int, length: int, o: List[float], l: List[float], c: List[float]) -> Tuple[float, float, int]:
    wick_h = np.nan
    wick_l = np.nan
    index = 0
    for i in range(length + 1):
        if i > 0 and l[start_index - i] > wick_l:
            continue
        bar_dir = np.sign(c[start_index - i] - o[start_index - i])
        wick_h = c[start_index - i] if bar_dir == 1 else o[start_index - i]
        wick_l = l[start_index - i]
        index = i
    return wick_h, wick_l, index


def _high_wick_search(start_index: int, length: int, o: List[float], h: List[float], c: List[float]) -> Tuple[float, float, int]:
    wick_h = np.nan
    wick_l = np.nan
    index = 0
    for i in range(length + 1):
        if i > 0 and h[start_index - i] < wick_h:
            continue
        bar_dir = np.sign(c[start_index - i] - o[start_index - i])
        wick_l = o[start_index - i] if bar_dir == 1 else c[start_index - i]
        wick_h = h[start_index - i]
        index = i
    return wick_h, wick_l, index


def compute_mtf_order_block_finder(
    df: pd.DataFrame,
    *,
    settings: Optional[OBSettings] = None,
) -> Dict[str, List[OBZone]]:
    if settings is None:
        settings = OBSettings()

    ob_search = settings.ob_search
    ob_shift = settings.ob_shift
    ob_period = settings.ob_period

    if ob_search >= ob_period:
        ob_search = ob_period
    if ob_shift >= ob_period:
        ob_shift = ob_period
    ob_period += 1

    chart_minutes = _timeframe_to_minutes(df.index)
    res_minutes = _resolution_to_minutes(settings.resolution)
    if chart_minutes == 0:
        return {"zones": []}
    res = "" if chart_minutes > res_minutes and res_minutes > 0 else settings.resolution

    warmup = ob_period + (-ob_shift if np.sign(ob_shift) == -1 else 0)
    mtf_warmup = warmup if res == "" else int(warmup * (res_minutes / chart_minutes))

    data = df.copy()
    if res:
        rule = _resample_rule_from_resolution(res)
        data = _resample_ohlc(df, rule)

    copen, naopen = _remove_gaps(data["open"], mtf_warmup)
    chigh, nahigh = _remove_gaps(data["high"], mtf_warmup)
    clow, nalow = _remove_gaps(data["low"], mtf_warmup)
    cclose, naclose = _remove_gaps(data["close"], mtf_warmup)
    ctime, natime = _remove_gaps(pd.Series(data.index), mtf_warmup)

    nastate = naopen and nahigh and nalow and naclose and natime
    min_size = min(len(copen), len(chigh), len(clow), len(cclose))

    zones: List[OBZone] = []
    if min_size > warmup:
        relmove = abs(cclose[ob_period] - cclose[1]) / cclose[ob_period] * 100 > settings.threshold
        doji_candle = abs(cclose[ob_period] - copen[ob_period]) / copen[ob_period] * 100 > settings.doji

        bullish_ob = cclose[ob_period] < copen[ob_period]
        bearish_ob = cclose[ob_period] > copen[ob_period]

        upcandles = 0
        downcandles = 0
        for i in range(1, ob_period):
            if _is_zero_candle(i, copen, chigh, clow, cclose):
                continue
            t_close = cclose[i]
            t_open = copen[i]
            if abs(100 * (t_close - t_open) / t_open) < settings.fuzzy:
                upcandles += 1
                downcandles += 1
                continue
            if t_close > t_open:
                upcandles += 1
            elif t_close < t_open:
                downcandles += 1

        if doji_candle and relmove and nastate:
            ob_bull = bullish_ob and (upcandles == (ob_period - 1))
            ob_bear = bearish_ob and (downcandles == (ob_period - 1))

            if ob_bull:
                selector_shift = ob_shift
                if settings.ob_selector == "Context":
                    temp_high, temp_low, selector_shift = _low_wick_search(ob_period, ob_search, copen, clow, cclose)
                    ob_bull_chigh = temp_high
                    ob_bull_clow = temp_low
                elif settings.ob_selector == "High/Low":
                    ob_bull_chigh = chigh[ob_period - selector_shift]
                    ob_bull_clow = clow[ob_period - selector_shift]
                else:
                    temp_high, temp_low, _index = _low_wick_search(ob_period - selector_shift, 0, copen, clow, cclose)
                    ob_bull_chigh = temp_high
                    ob_bull_clow = temp_low
                if settings.bull_channels > 0:
                    existing = [z for z in zones if z.direction == "bull"]
                    if len(existing) == settings.bull_channels:
                        for idx, z in enumerate(zones):
                            if z.direction == "bull":
                                zones.pop(idx)
                                break
                    zones.append(
                        OBZone(
                            direction="bull",
                            source_time=pd.to_datetime(ctime[ob_period - selector_shift]),
                            high=float(ob_bull_chigh),
                            low=float(ob_bull_clow),
                            avg=float((ob_bull_chigh + ob_bull_clow) / 2),
                            selector_shift=selector_shift,
                        )
                    )

            if ob_bear:
                selector_shift = ob_shift
                if settings.ob_selector == "Context":
                    temp_high, temp_low, selector_shift = _high_wick_search(ob_period, ob_search, copen, chigh, cclose)
                    ob_bear_chigh = temp_high
                    ob_bear_clow = temp_low
                elif settings.ob_selector == "High/Low":
                    ob_bear_chigh = chigh[ob_period - selector_shift]
                    ob_bear_clow = clow[ob_period - selector_shift]
                else:
                    temp_high, temp_low, _index = _high_wick_search(ob_period - selector_shift, 0, copen, chigh, cclose)
                    ob_bear_chigh = temp_high
                    ob_bear_clow = temp_low
                if settings.bear_channels > 0:
                    existing = [z for z in zones if z.direction == "bear"]
                    if len(existing) == settings.bear_channels:
                        for idx, z in enumerate(zones):
                            if z.direction == "bear":
                                zones.pop(idx)
                                break
                    zones.append(
                        OBZone(
                            direction="bear",
                            source_time=pd.to_datetime(ctime[ob_period - selector_shift]),
                            high=float(ob_bear_chigh),
                            low=float(ob_bear_clow),
                            avg=float((ob_bear_chigh + ob_bear_clow) / 2),
                            selector_shift=selector_shift,
                        )
                    )

    if settings.near_price > 0 and len(df) > 0:
        source = float(df["open"].iloc[-1])
        limit = source * settings.near_price / 100.0
        zones = [
            zone
            for zone in zones
            if not (abs(zone.high - source) > limit and abs(zone.low - source) > limit)
        ]

    return {"zones": zones}
