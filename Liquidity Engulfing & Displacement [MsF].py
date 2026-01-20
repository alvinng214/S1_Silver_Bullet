"""Liquidity Engulfing & Displacement [MsF].

Python translation of the Trader_Morry Pine Script:
- Liquidity Engulfing Candles (LEC) on H1/H4/current timeframe.
- Displacement detection with optional FVG requirement.

The translation mirrors `request.security` lookahead_off behavior by
building higher-timeframe bars and forward-filling their signals to LTF bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class LECSignals:
    bullish: pd.Series
    bearish: pd.Series
    bullish_edge: pd.Series
    bearish_edge: pd.Series


@dataclass
class LECOutputs:
    h1: LECSignals
    h4: LECSignals
    current: LECSignals


@dataclass
class DisplacementOutput:
    displacement_raw: pd.Series
    displacement_bar: pd.Series
    candle_range: pd.Series
    std_threshold: pd.Series
    fvg: pd.Series


def _detect_lec(df: pd.DataFrame, filter_liquidity: bool = True, filter_close: bool = True) -> Tuple[pd.Series, pd.Series]:
    prior_open = df["open"].shift(1)
    prior_close = df["close"].shift(1)
    current_open = df["open"]
    current_close = df["close"]

    bull = (
        (current_open <= prior_close)
        & (current_open < prior_open)
        & (current_close > prior_open)
    )
    bear = (
        (current_open >= prior_close)
        & (current_open > prior_open)
        & (current_close < prior_open)
    )

    if filter_liquidity:
        bull = bull & (df["low"] <= df["low"].shift(1))
        bear = bear & (df["high"] >= df["high"].shift(1))

    if filter_close:
        bull = bull & (df["close"] >= df["high"].shift(1))
        bear = bear & (df["close"] <= df["low"].shift(1))

    return bull.fillna(False), bear.fillna(False)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _align_signals(signal: pd.Series, target_index: pd.Index) -> pd.Series:
    return signal.reindex(target_index, method="ffill").fillna(False)


def _lec_signals_for_timeframe(
    df: pd.DataFrame,
    rule: str,
    filter_liquidity: bool,
    filter_close: bool,
) -> LECSignals:
    htf_df = _resample_ohlc(df, rule)
    bull, bear = _detect_lec(htf_df, filter_liquidity=filter_liquidity, filter_close=filter_close)
    bull = _align_signals(bull, df.index)
    bear = _align_signals(bear, df.index)
    bull_edge = bull & ~bull.shift(1).fillna(False)
    bear_edge = bear & ~bear.shift(1).fillna(False)
    return LECSignals(bullish=bull, bearish=bear, bullish_edge=bull_edge, bearish_edge=bear_edge)


def calculate_lec_signals(
    df: pd.DataFrame,
    *,
    filter_liquidity: bool = True,
    filter_close: bool = True,
) -> LECOutputs:
    """Calculate LEC signals on H1, H4, and current timeframe."""
    bull_cur, bear_cur = _detect_lec(df, filter_liquidity=filter_liquidity, filter_close=filter_close)
    bull_edge_cur = bull_cur & ~bull_cur.shift(1).fillna(False)
    bear_edge_cur = bear_cur & ~bear_cur.shift(1).fillna(False)
    current = LECSignals(bull_cur, bear_cur, bull_edge_cur, bear_edge_cur)

    h1 = _lec_signals_for_timeframe(df, "60min", filter_liquidity, filter_close)
    h4 = _lec_signals_for_timeframe(df, "240min", filter_liquidity, filter_close)

    return LECOutputs(h1=h1, h4=h4, current=current)


def calculate_displacement(
    df: pd.DataFrame,
    *,
    require_fvg: bool = True,
    disp_type: str = "Open to Close",
    std_len: int = 100,
    std_x: int = 2,
) -> DisplacementOutput:
    """Calculate displacement signals matching the Pine Script rules."""
    if disp_type == "Open to Close":
        candle_range = (df["open"] - df["close"]).abs()
    else:
        candle_range = df["high"] - df["low"]

    std_threshold = candle_range.rolling(std_len).std() * std_x

    prior_bull = df["close"].shift(1) > df["open"].shift(1)
    fvg = np.where(prior_bull, df["high"].shift(2) < df["low"], df["low"].shift(2) > df["high"])
    fvg = pd.Series(fvg, index=df.index, dtype=bool).fillna(False)

    if require_fvg:
        displacement_raw = (candle_range.shift(1) > std_threshold.shift(1)) & fvg
        displacement_raw = displacement_raw.fillna(False)
        displacement_bar = displacement_raw.shift(-1).fillna(False)
    else:
        displacement_raw = (candle_range > std_threshold).fillna(False)
        displacement_bar = displacement_raw.copy()

    return DisplacementOutput(
        displacement_raw=displacement_raw,
        displacement_bar=displacement_bar,
        candle_range=candle_range,
        std_threshold=std_threshold,
        fvg=fvg,
    )


def calculate_liquidity_engulfing_displacement(
    df: pd.DataFrame,
    *,
    filter_liquidity: bool = True,
    filter_close: bool = True,
    require_fvg: bool = True,
    disp_type: str = "Open to Close",
    std_len: int = 100,
    std_x: int = 2,
) -> Dict[str, object]:
    """Run both LEC and displacement calculations and return results."""
    lec = calculate_lec_signals(df, filter_liquidity=filter_liquidity, filter_close=filter_close)
    displacement = calculate_displacement(
        df,
        require_fvg=require_fvg,
        disp_type=disp_type,
        std_len=std_len,
        std_x=std_x,
    )

    return {
        "lec": lec,
        "displacement": displacement,
    }
