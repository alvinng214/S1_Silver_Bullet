"""Smart Money Zones (FVG + OB) + MTF Trend Panel.

Python translation of the Pine Script logic with:
- Per-bar FVG/OB detection and zone lifecycle.
- Trend-filtered signals.
- Mitigation tracking and optional removal.
- Multi-timeframe trend series aligned to the chart index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SMZone:
    top: float
    bottom: float
    start_idx: int
    end_idx: int
    created_at: int
    is_fvg: bool
    is_bullish: bool
    strength: str
    is_live: bool = True
    mitigated_amount: float = 0.0


def _calc_body_pct(open_val: float, high_val: float, low_val: float, close_val: float) -> float:
    candle_range = high_val - low_val
    body = abs(close_val - open_val)
    return body / candle_range if candle_range > 0 else 0.0


def _get_strength(size: float, atr_val: float, min_tick: float = 1e-8) -> str:
    denom = max(atr_val, min_tick)
    ratio = size / denom
    if ratio >= 2.0:
        return "VERY STRONG"
    if ratio >= 1.5:
        return "STRONG"
    if ratio >= 1.0:
        return "MEDIUM"
    return "WEAK"


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    return tr.rolling(window=length).mean()


def _mitigate_zone(zone: SMZone, row: pd.Series, remove_on_mit: bool) -> bool:
    zone_size = max(zone.top - zone.bottom, 1e-8)
    if not zone.is_live:
        return False

    if zone.is_bullish:
        if row["low"] <= zone.top and row["low"] >= zone.bottom:
            penetration = zone.top - row["low"]
            zone.mitigated_amount = (penetration / zone_size) * 100.0
        is_mitigated = zone.mitigated_amount >= 50.0 or row["close"] <= zone.bottom
    else:
        if row["high"] >= zone.bottom and row["high"] <= zone.top:
            penetration = row["high"] - zone.bottom
            zone.mitigated_amount = (penetration / zone_size) * 100.0
        is_mitigated = zone.mitigated_amount >= 50.0 or row["close"] >= zone.top

    if is_mitigated:
        if remove_on_mit:
            return True
        zone.is_live = False
    return False


def _enforce_cap(zones: List[SMZone], cap: int) -> None:
    while len(zones) > cap:
        zones.pop(0)


def _trend_series(df: pd.DataFrame, tf: str, ma_period: int) -> pd.Series:
    resampled = df.resample(tf).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()
    ma = resampled["close"].rolling(window=ma_period).mean()
    trend = resampled["close"] > ma
    return trend.reindex(df.index, method="ffill").fillna(False)


def calculate_smart_money_zones(
    df: pd.DataFrame,
    *,
    show_fvg: bool = True,
    show_ob: bool = True,
    max_zones: int = 20,
    remove_on_mit: bool = False,
    min_body_pct: float = 0.5,
    ob_lookback: int = 10,
    atr_length: int = 14,
    use_trend_filter: bool = True,
    trend_ma_period: int = 50,
    mtf_ma_period: int = 50,
) -> Dict[str, object]:
    """Detect Smart Money Zones (FVG + OB) with Pine-style logic."""
    atr = _atr(df, atr_length)
    trend_ma = df["close"].rolling(window=trend_ma_period).mean()
    is_uptrend = df["close"] > trend_ma
    is_dntrend = df["close"] < trend_ma

    bull_fvg: List[SMZone] = []
    bear_fvg: List[SMZone] = []
    bull_ob: List[SMZone] = []
    bear_ob: List[SMZone] = []

    for i in range(len(df)):
        if i >= 2:
            if show_fvg and df["low"].iloc[i] > df["high"].iloc[i - 2]:
                mid_bull = df["close"].iloc[i - 1] > df["open"].iloc[i - 1]
                mid_body = _calc_body_pct(
                    df["open"].iloc[i - 1],
                    df["high"].iloc[i - 1],
                    df["low"].iloc[i - 1],
                    df["close"].iloc[i - 1],
                )
                if mid_bull and mid_body >= min_body_pct and (not use_trend_filter or is_uptrend.iloc[i]):
                    top = df["low"].iloc[i]
                    bottom = df["high"].iloc[i - 2]
                    gap = top - bottom
                    strength = _get_strength(gap, atr.iloc[i])
                    bull_fvg.append(
                        SMZone(
                            top=top,
                            bottom=bottom,
                            start_idx=i - 2,
                            end_idx=i,
                            created_at=i,
                            is_fvg=True,
                            is_bullish=True,
                            strength=strength,
                        )
                    )
                    _enforce_cap(bull_fvg, max_zones)

            if show_fvg and df["high"].iloc[i] < df["low"].iloc[i - 2]:
                mid_bear = df["close"].iloc[i - 1] < df["open"].iloc[i - 1]
                mid_body = _calc_body_pct(
                    df["open"].iloc[i - 1],
                    df["high"].iloc[i - 1],
                    df["low"].iloc[i - 1],
                    df["close"].iloc[i - 1],
                )
                if mid_bear and mid_body >= min_body_pct and (not use_trend_filter or is_dntrend.iloc[i]):
                    top = df["low"].iloc[i - 2]
                    bottom = df["high"].iloc[i]
                    gap = top - bottom
                    strength = _get_strength(gap, atr.iloc[i])
                    bear_fvg.append(
                        SMZone(
                            top=top,
                            bottom=bottom,
                            start_idx=i - 2,
                            end_idx=i,
                            created_at=i,
                            is_fvg=True,
                            is_bullish=False,
                            strength=strength,
                        )
                    )
                    _enforce_cap(bear_fvg, max_zones)

        if i >= ob_lookback and show_ob:
            atr_val = atr.iloc[i]
            is_bull_break = df["close"].iloc[i] > df["close"].iloc[i - 1] and (df["high"].iloc[i] - df["low"].iloc[i]) > atr_val * 1.2
            prev_bear = df["close"].iloc[i - 1] < df["open"].iloc[i - 1]
            strong_up = df["close"].iloc[i] > df["open"].iloc[i] and _calc_body_pct(
                df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            ) >= min_body_pct
            if is_bull_break and prev_bear and strong_up and (not use_trend_filter or is_uptrend.iloc[i]):
                top = max(df["open"].iloc[i - 1], df["close"].iloc[i - 1])
                bottom = df["low"].iloc[i - 1]
                size = top - bottom
                strength = _get_strength(size, atr_val)
                bull_ob.append(
                    SMZone(
                        top=top,
                        bottom=bottom,
                        start_idx=i - 1,
                        end_idx=i,
                        created_at=i,
                        is_fvg=False,
                        is_bullish=True,
                        strength=strength,
                    )
                )
                _enforce_cap(bull_ob, max_zones)

            is_bear_break = df["close"].iloc[i] < df["close"].iloc[i - 1] and (df["high"].iloc[i] - df["low"].iloc[i]) > atr_val * 1.2
            prev_bull = df["close"].iloc[i - 1] > df["open"].iloc[i - 1]
            strong_dn = df["close"].iloc[i] < df["open"].iloc[i] and _calc_body_pct(
                df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            ) >= min_body_pct
            if is_bear_break and prev_bull and strong_dn and (not use_trend_filter or is_dntrend.iloc[i]):
                top = df["high"].iloc[i - 1]
                bottom = min(df["open"].iloc[i - 1], df["close"].iloc[i - 1])
                size = top - bottom
                strength = _get_strength(size, atr_val)
                bear_ob.append(
                    SMZone(
                        top=top,
                        bottom=bottom,
                        start_idx=i - 1,
                        end_idx=i,
                        created_at=i,
                        is_fvg=False,
                        is_bullish=False,
                        strength=strength,
                    )
                )
                _enforce_cap(bear_ob, max_zones)

        for zones in (bull_fvg, bear_fvg, bull_ob, bear_ob):
            for zone in list(zones):
                zone.end_idx = i
                if _mitigate_zone(zone, df.iloc[i], remove_on_mit):
                    zones.remove(zone)

    mtf_trends = {
        "1m": _trend_series(df, "1min", mtf_ma_period),
        "5m": _trend_series(df, "5min", mtf_ma_period),
        "15m": _trend_series(df, "15min", mtf_ma_period),
        "30m": _trend_series(df, "30min", mtf_ma_period),
        "1h": _trend_series(df, "60min", mtf_ma_period),
        "4h": _trend_series(df, "240min", mtf_ma_period),
        "1d": _trend_series(df, "1D", mtf_ma_period),
    }

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "trend_ma": trend_ma,
        "mtf_trends": mtf_trends,
    }


if __name__ == "__main__":
    data = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    data["datetime"] = pd.to_datetime(data["time"])
    data = data.set_index("datetime").sort_index()

    results = calculate_smart_money_zones(data)
    print(results["mtf_trends"]["1h"].tail())
