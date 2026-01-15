"""Liquidity Engulfing & Displacement [MsF]

Python translation of the TradingView Pine Script by Trader_Morry.

This module reproduces:
- Liquidity Engulfing Candles (LEC) on H1/H4/current timeframe.
- Displacement detection with optional FVG requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class LECSignals:
    bullish: pd.Series
    bearish: pd.Series


@dataclass
class LECOutputs:
    h1: LECSignals
    h4: LECSignals
    current: LECSignals


@dataclass
class DisplacementOutput:
    displacement: pd.Series
    candle_range: pd.Series
    std_threshold: pd.Series
    fvg: pd.Series


def _detect_lec(df: pd.DataFrame, filter_liquidity: bool = True, filter_close: bool = True) -> LECSignals:
    """Detect Liquidity Engulfing Candles on a given dataframe."""
    prior_open = df["open"].shift(1)
    prior_close = df["close"].shift(1)

    current_open = df["open"]
    current_close = df["close"]

    bull_engulf = (
        (current_open <= prior_close)
        & (current_open < prior_open)
        & (current_close > prior_open)
    )
    bear_engulf = (
        (current_open >= prior_close)
        & (current_open > prior_open)
        & (current_close < prior_open)
    )

    if filter_liquidity:
        bull_engulf = bull_engulf & (df["low"] <= df["low"].shift(1))
        bear_engulf = bear_engulf & (df["high"] >= df["high"].shift(1))

    if filter_close:
        bull_engulf = bull_engulf & (df["close"] >= df["high"].shift(1))
        bear_engulf = bear_engulf & (df["close"] <= df["low"].shift(1))

    return LECSignals(bullish=bull_engulf, bearish=bear_engulf)


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["open", "high", "low", "close"]]
        .resample(rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def _align_signals(source_signals: LECSignals, target_index: pd.Index) -> LECSignals:
    aligned_bull = source_signals.bullish.reindex(target_index, method="ffill").fillna(False)
    aligned_bear = source_signals.bearish.reindex(target_index, method="ffill").fillna(False)
    return LECSignals(bullish=aligned_bull, bearish=aligned_bear)


def calculate_lec_signals(
    df: pd.DataFrame,
    *,
    filter_liquidity: bool = True,
    filter_close: bool = True,
) -> LECOutputs:
    """Calculate LEC signals on H1, H4, and current timeframe."""
    current = _detect_lec(df, filter_liquidity=filter_liquidity, filter_close=filter_close)

    h1_df = _resample_ohlc(df, "60min")
    h4_df = _resample_ohlc(df, "240min")

    h1 = _align_signals(_detect_lec(h1_df, filter_liquidity, filter_close), df.index)
    h4 = _align_signals(_detect_lec(h4_df, filter_liquidity, filter_close), df.index)

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
    fvg = pd.Series(fvg, index=df.index, dtype=bool)

    if require_fvg:
        displacement = (candle_range.shift(1) > std_threshold.shift(1)) & fvg
        displacement = displacement.shift(1).fillna(False)
    else:
        displacement = (candle_range > std_threshold).fillna(False)

    return DisplacementOutput(
        displacement=displacement,
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
