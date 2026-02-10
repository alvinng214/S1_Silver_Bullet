"""Python translation of the Pine Script "Order Blocks & Imbalance MTF".

This module mirrors the Pine logic by:
- Detecting higher-timeframe (HTF) order blocks based on FVG size vs ATR threshold.
- Tracking mitigation and removal rules for bullish (demand) and bearish (supply) zones.
- Supporting smart visibility to show only a limited number of zones per side.

TradingView box drawings are represented as data structures so the behavior can be
used programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class OBZone:
    top: float
    bottom: float
    is_bullish: bool
    mitigated: bool
    created_time: pd.Timestamp
    left_time: pd.Timestamp
    right_time: pd.Timestamp
    visible: bool
    label: str
    bg_color: str = "white"
    bg_alpha: int = 100
    border_color: str = "white"
    border_alpha: int = 100
    border_style: str = "none"
    text_color: str = "white"
    text_alpha: int = 100


@dataclass
class OBSettings:
    timeframe: str = ""
    fvg_threshold: float = 0.5
    mitigation_type: str = "Wick"  # "Wick" or "Close"
    show_bull: bool = True
    show_bear: bool = True
    use_smart_view: bool = True
    visible_limit: int = 10
    extend_active: bool = True

    def __post_init__(self) -> None:
        # Pine input.int enforces [1, 20] for visibleLimit.
        self.visible_limit = max(1, min(20, int(self.visible_limit)))
        # Pine input.string constrains mitigationType to ["Close", "Wick"].
        if self.mitigation_type not in {"Wick", "Close"}:
            self.mitigation_type = "Wick"


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregations = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        aggregations["volume"] = "sum"
    return df.resample(rule).agg(aggregations).dropna()


def _resolve_timeframe_rule(df: pd.DataFrame, timeframe: str) -> Optional[str]:
    if timeframe == "":
        return None
    normalized = timeframe.upper()
    if normalized.isdigit():
        return f"{normalized}min"
    if normalized in {"1H", "H"}:
        return "1H"
    if normalized in {"1D", "D"}:
        return "1D"
    if normalized in {"1W", "W"}:
        return "1W"
    if normalized in {"1M", "M"}:
        return "1M"
    return normalized


def _rma(series: pd.Series, length: int) -> pd.Series:
    rma = pd.Series(index=series.index, dtype="float64")
    if len(series) < length:
        return rma
    rma.iloc[: length - 1] = pd.NA
    rma.iloc[length - 1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        rma.iloc[i] = ((rma.iloc[i - 1] * (length - 1)) + series.iloc[i]) / length
    return rma


def _atr(htf: pd.DataFrame, length: int = 14) -> pd.Series:
    high = htf["high"]
    low = htf["low"]
    close = htf["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _rma(tr, length)


def _align_htf(df: pd.DataFrame, htf: pd.DataFrame) -> pd.DataFrame:
    return htf.reindex(df.index, method="ffill")


def _toggle_zone(zone: OBZone, visible: bool) -> None:
    base_color = "green" if zone.is_bullish else "red"
    zone.visible = visible
    if visible:
        zone.text_color = "white"
        zone.text_alpha = 0
        zone.border_color = base_color
        zone.border_alpha = 0
        if zone.mitigated:
            zone.bg_color = base_color
            zone.bg_alpha = 90
            zone.border_style = "dotted"
        else:
            zone.bg_color = base_color
            zone.bg_alpha = 70
            zone.border_style = "solid"
    else:
        zone.bg_color = "white"
        zone.bg_alpha = 100
        zone.border_color = "white"
        zone.border_alpha = 100
        zone.border_style = "none"
        zone.text_color = "white"
        zone.text_alpha = 100


def build_order_blocks(df: pd.DataFrame, settings: OBSettings) -> List[OBZone]:
    """Return the list of OB zones after processing the full OHLCV series.

    The input dataframe must be indexed by timestamps and include columns:
    open, high, low, close (volume optional).
    """

    rule = _resolve_timeframe_rule(df, settings.timeframe)
    if rule is None:
        htf = df.copy()
    else:
        htf = _resample_ohlc(df, rule)

    if htf.empty:
        return []

    atr = _atr(htf, 14)
    bull_fvg_size = htf["low"] - htf["high"].shift(2)
    bear_fvg_size = htf["low"].shift(2) - htf["high"]
    bull = bull_fvg_size > (atr.shift(1) * settings.fvg_threshold)
    bear = bear_fvg_size > (atr.shift(1) * settings.fvg_threshold)

    htf_calc = pd.DataFrame(
        {
            "is_bull": bull,
            "is_bear": bear,
            "high_shift2": htf["high"].shift(2),
            "low_shift2": htf["low"].shift(2),
            "time_shift2": htf.index.to_series().shift(2),
        }
    )

    htf_aligned = _align_htf(df, htf_calc)
    zones: List[OBZone] = []
    last_created_time: Optional[pd.Timestamp] = None

    for i, (ts, row) in enumerate(df.iterrows()):
        htf_row = htf_aligned.loc[ts]
        htf_time = htf_row["time_shift2"]
        if pd.notna(htf_time) and htf_time != last_created_time:
            if settings.show_bull and bool(htf_row["is_bull"]):
                zone = OBZone(
                    top=float(htf_row["high_shift2"]),
                    bottom=float(htf_row["low_shift2"]),
                    is_bullish=True,
                    mitigated=False,
                    created_time=htf_time,
                    left_time=htf_time,
                    right_time=ts + pd.Timedelta(milliseconds=60000),
                    visible=not settings.use_smart_view,
                    label="OB Demand",
                )
                _toggle_zone(zone, not settings.use_smart_view)
                zones.append(zone)
                last_created_time = htf_time
            elif settings.show_bear and bool(htf_row["is_bear"]):
                zone = OBZone(
                    top=float(htf_row["high_shift2"]),
                    bottom=float(htf_row["low_shift2"]),
                    is_bullish=False,
                    mitigated=False,
                    created_time=htf_time,
                    left_time=htf_time,
                    right_time=ts + pd.Timedelta(milliseconds=60000),
                    visible=not settings.use_smart_view,
                    label="OB Supply",
                )
                _toggle_zone(zone, not settings.use_smart_view)
                zones.append(zone)
                last_created_time = htf_time

        if zones:
            remove_indices = []
            for idx in range(len(zones) - 1, -1, -1):
                zone = zones[idx]
                if zone.is_bullish:
                    if row["close"] < zone.bottom:
                        remove_indices.append(idx)
                    else:
                        touch = row["low"] if settings.mitigation_type == "Wick" else row["close"]
                        if touch <= zone.top and not zone.mitigated:
                            zone.mitigated = True
                            zone.label = "Mitigated"
                            if not settings.use_smart_view:
                                _toggle_zone(zone, True)
                else:
                    if row["close"] > zone.top:
                        remove_indices.append(idx)
                    else:
                        touch = row["high"] if settings.mitigation_type == "Wick" else row["close"]
                        if touch >= zone.bottom and not zone.mitigated:
                            zone.mitigated = True
                            zone.label = "Mitigated"
                            if not settings.use_smart_view:
                                _toggle_zone(zone, True)

            for idx in remove_indices:
                zones.pop(idx)

        if len(zones) > 450:
            zones.pop(0)

    if len(df.index) < 2:
        return zones

    last_time = df.index[-1]
    prev_time = df.index[-2]
    time_delta = last_time - prev_time
    future_time = last_time + (time_delta * 10)

    if settings.use_smart_view and zones:
        for zone in zones:
            _toggle_zone(zone, False)

        bull = [(zone.top, idx) for idx, zone in enumerate(zones) if zone.is_bullish]
        bear = [(zone.bottom, idx) for idx, zone in enumerate(zones) if not zone.is_bullish]

        bull.sort(key=lambda x: x[0], reverse=True)
        bear.sort(key=lambda x: x[0])

        for _, idx in bull[: settings.visible_limit]:
            _toggle_zone(zones[idx], True)
            if settings.extend_active:
                zones[idx].right_time = future_time

        for _, idx in bear[: settings.visible_limit]:
            _toggle_zone(zones[idx], True)
            if settings.extend_active:
                zones[idx].right_time = future_time
    elif not settings.use_smart_view and zones:
        for zone in zones:
            zone.right_time = last_time + (time_delta * 5)

    return zones
