"""
HTF Bias Indicator - Python translation of CandelaCharts HTF Sweeps
Simplified version focusing on core functionality:

1. Display HTF candles (1H, 4H, Daily) as mini-candles on the chart
2. Show HTF highs/lows as horizontal lines
3. Detect sweeps (wick beyond HTF level, close back inside)
4. Show HTF bias (bullish/bearish trend)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HTFCandle:
    """Represents a Higher Timeframe candle"""
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    open_idx: int
    close_idx: int
    high_idx: int
    low_idx: int
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    is_bullish: bool

    def __post_init__(self):
        self.bull_sweep = False
        self.bear_sweep = False


@dataclass
class Sweep:
    """Represents a liquidity sweep"""
    price: float
    index: int
    time: pd.Timestamp
    is_bullish: bool  # True = bullish sweep (swept high), False = bearish sweep (swept low)
    invalidated: bool = False
    htf_timeframe: str = ""


def resample_to_htf(df, timeframe_minutes):
    """
    Resample the dataframe to a higher timeframe.

    Args:
        df: DataFrame with time index and OHLC columns
        timeframe_minutes: Number of minutes for the new timeframe

    Returns:
        Resampled DataFrame with additional index column
    """
    # Create a copy with time as column for reference
    df_copy = df.copy()

    # Resample
    df_resampled = df_copy.resample(f'{timeframe_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    return df_resampled


def build_htf_candles(df, df_htf, timeframe_name):
    """
    Build HTFCandle objects from resampled data, mapped to LTF indices.

    Args:
        df: Original LTF dataframe with index positions
        df_htf: Resampled HTF dataframe
        timeframe_name: Name of the timeframe (e.g., "1H", "4H")

    Returns:
        List of HTFCandle objects
    """
    htf_candles = []

    for htf_time, htf_row in df_htf.iterrows():
        # Find LTF bars that fall within this HTF candle
        ltf_bars_in_htf = df[(df.index >= htf_time) & (df.index < htf_time + pd.Timedelta(minutes=int(timeframe_name[:-1]) if 'H' in timeframe_name else 1440))]

        if len(ltf_bars_in_htf) == 0:
            continue

        # Find indices for O/H/L/C
        open_idx = ltf_bars_in_htf.iloc[0]['index']
        close_idx = ltf_bars_in_htf.iloc[-1]['index']

        # Find high index (first occurrence if multiple bars have same high)
        high_matches = ltf_bars_in_htf[ltf_bars_in_htf['high'] == htf_row['high']]
        if len(high_matches) > 0:
            high_idx = int(high_matches.iloc[0]['index'])
        else:
            high_idx = open_idx

        # Find low index (first occurrence if multiple bars have same low)
        low_matches = ltf_bars_in_htf[ltf_bars_in_htf['low'] == htf_row['low']]
        if len(low_matches) > 0:
            low_idx = int(low_matches.iloc[0]['index'])
        else:
            low_idx = open_idx

        is_bullish = htf_row['close'] >= htf_row['open']

        candle = HTFCandle(
            timeframe=timeframe_name,
            open=htf_row['open'],
            high=htf_row['high'],
            low=htf_row['low'],
            close=htf_row['close'],
            open_idx=open_idx,
            close_idx=close_idx,
            high_idx=high_idx,
            low_idx=low_idx,
            open_time=htf_time,
            close_time=ltf_bars_in_htf.index[-1],
            is_bullish=is_bullish
        )

        htf_candles.append(candle)

    return htf_candles


def detect_htf_sweeps(htf_candles):
    """
    Detect sweeps between consecutive HTF candles.

    A sweep occurs when:
    - Bullish sweep: Current candle high > previous candle high, but close < previous high
    - Bearish sweep: Current candle low < previous candle low, but close > previous low

    Args:
        htf_candles: List of HTFCandle objects

    Returns:
        List of Sweep objects
    """
    sweeps = []

    for i in range(1, len(htf_candles)):
        prev_candle = htf_candles[i-1]
        curr_candle = htf_candles[i]

        # Check for bullish sweep (swept previous high)
        if curr_candle.high > prev_candle.high:
            # Check if close is back inside (below the previous high)
            if curr_candle.is_bullish:
                close_back_inside = curr_candle.close < prev_candle.high
            else:
                close_back_inside = curr_candle.close < prev_candle.high

            if close_back_inside:
                sweep = Sweep(
                    price=prev_candle.high,
                    index=prev_candle.high_idx,
                    time=prev_candle.close_time,
                    is_bullish=True,
                    htf_timeframe=prev_candle.timeframe
                )
                sweeps.append(sweep)
                prev_candle.bull_sweep = True

        # Check for bearish sweep (swept previous low)
        if curr_candle.low < prev_candle.low:
            # Check if close is back inside (above the previous low)
            if curr_candle.is_bullish:
                close_back_inside = curr_candle.close > prev_candle.low
            else:
                close_back_inside = curr_candle.close > prev_candle.low

            if close_back_inside:
                sweep = Sweep(
                    price=prev_candle.low,
                    index=prev_candle.low_idx,
                    time=prev_candle.close_time,
                    is_bullish=False,
                    htf_timeframe=prev_candle.timeframe
                )
                sweeps.append(sweep)
                prev_candle.bear_sweep = True

    return sweeps


def get_htf_bias(htf_candles, num_candles=3):
    """
    Determine HTF bias based on recent candles.

    Bullish bias: Recent candles making higher highs and higher lows
    Bearish bias: Recent candles making lower highs and lower lows

    Args:
        htf_candles: List of HTFCandle objects
        num_candles: Number of recent candles to analyze

    Returns:
        1 for bullish, -1 for bearish, 0 for neutral
    """
    if len(htf_candles) < 2:
        return 0

    recent = htf_candles[-min(num_candles, len(htf_candles)):]

    # Count bullish vs bearish candles
    bullish_count = sum(1 for c in recent if c.is_bullish)
    bearish_count = len(recent) - bullish_count

    # Check if making higher highs and higher lows
    if len(recent) >= 2:
        last = recent[-1]
        prev = recent[-2]

        if last.high > prev.high and last.low > prev.low:
            return 1  # Bullish bias
        elif last.high < prev.high and last.low < prev.low:
            return -1  # Bearish bias

    # Fallback to candle color majority
    if bullish_count > bearish_count:
        return 1
    elif bearish_count > bullish_count:
        return -1
    else:
        return 0


def calculate_htf_data(df, timeframes_minutes={'1H': 60, '4H': 240, 'Daily': 1440}):
    """
    Calculate HTF candles, sweeps, and bias for multiple timeframes.

    Args:
        df: DataFrame with OHLC data and index column
        timeframes_minutes: Dict of timeframe names to minutes

    Returns:
        Dictionary with HTF candles, sweeps, and bias for each timeframe
    """
    results = {}

    for tf_name, tf_minutes in timeframes_minutes.items():
        # Resample to HTF
        df_htf = resample_to_htf(df, tf_minutes)

        # Build HTF candles
        htf_candles = build_htf_candles(df, df_htf, tf_name)

        # Detect sweeps
        sweeps = detect_htf_sweeps(htf_candles)

        # Get bias
        bias = get_htf_bias(htf_candles)

        results[tf_name] = {
            'candles': htf_candles,
            'sweeps': sweeps,
            'bias': bias
        }

    return results
