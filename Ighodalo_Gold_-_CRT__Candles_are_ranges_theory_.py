"""
Ighodalo Gold - CRT (Candles are Ranges Theory)
Python implementation of CRT detection and Turtle Soup signals
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class CRTRange:
    """Represents a CRT (Candle Range Theory) level"""
    high: float
    low: float
    mid: float
    start_idx: int
    end_idx: int
    detection_time_idx: int
    is_active: bool
    buy_signaled: bool
    sell_signaled: bool
    timeframe: str


def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def detect_crt_ranges(df, lookback=20, timeframe_minutes=60):
    """
    Detect CRT (Candles are Ranges Theory) levels

    A CRT is formed when a candle's range contains all subsequent candles
    up to the lookback period.

    Args:
        df: DataFrame with OHLC data
        lookback: Maximum lookback period to check for CRT
        timeframe_minutes: Timeframe for CRT detection (e.g., 60 for 1H)

    Returns:
        List of CRTRange objects
    """
    crt_ranges = []

    # Resample to higher timeframe if needed
    if timeframe_minutes > 5:  # Assuming base is 5min
        df_htf = df.resample(f'{timeframe_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        df_htf['htf_idx'] = range(len(df_htf))

        # Create mapping from HTF index to LTF index range
        htf_to_ltf_map = {}
        for htf_idx, htf_time in enumerate(df_htf.index):
            # Find LTF bars within this HTF candle
            ltf_bars = df[(df.index >= htf_time) &
                          (df.index < htf_time + pd.Timedelta(minutes=timeframe_minutes))]
            if len(ltf_bars) > 0:
                htf_to_ltf_map[htf_idx] = {
                    'start': df.index.get_loc(ltf_bars.index[0]),
                    'end': df.index.get_loc(ltf_bars.index[-1])
                }
    else:
        df_htf = df.copy()
        df_htf['htf_idx'] = range(len(df_htf))
        # For same timeframe, identity mapping
        htf_to_ltf_map = {i: {'start': i, 'end': i} for i in range(len(df_htf))}

    # Detect CRTs on the HTF data
    for i in range(lookback, len(df_htf)):
        # Look back for potential CRT candles
        for k in range(1, min(lookback + 1, i + 1)):
            pot_high = df_htf.iloc[i - k]['high']
            pot_low = df_htf.iloc[i - k]['low']

            # Check if all candles from [i-k+1] to [i] are within the potential range
            # This means the highest high and lowest low in that period should be <= pot_high and >= pot_low
            period_slice = df_htf.iloc[i - k + 1:i + 1]

            highest_in_period = period_slice['high'].max()
            lowest_in_period = period_slice['low'].min()

            within = highest_in_period <= pot_high and lowest_in_period >= pot_low

            if not within:
                continue

            # Check uniqueness - ensure this exact high/low combo hasn't been seen before in the lookback
            # We want the first occurrence
            unique = True
            for j in range(max(0, i - k - lookback), i - k):
                if df_htf.iloc[j]['high'] == pot_high and df_htf.iloc[j]['low'] == pot_low:
                    unique = False
                    break

            if within and unique:
                mid = (pot_high + pot_low) / 2

                # Map HTF indices back to LTF indices
                detection_htf_idx = i - k
                end_htf_idx = i

                # Get LTF index range for start and end
                ltf_start_idx = htf_to_ltf_map.get(detection_htf_idx, {}).get('start', detection_htf_idx)
                ltf_end_idx = htf_to_ltf_map.get(end_htf_idx, {}).get('end', end_htf_idx)

                crt_ranges.append(CRTRange(
                    high=pot_high,
                    low=pot_low,
                    mid=mid,
                    start_idx=ltf_start_idx,  # Now using LTF index
                    end_idx=ltf_end_idx,  # Now using LTF index
                    detection_time_idx=ltf_start_idx,
                    is_active=True,
                    buy_signaled=False,
                    sell_signaled=False,
                    timeframe=f"{timeframe_minutes}min"
                ))

    return crt_ranges


def detect_turtle_soup_signals(df, crt_ranges, atr_multiplier=0.1, atr_period=14):
    """
    Detect Turtle Soup buy and sell signals

    Buy Signal: Price touches CRT low but closes above it
    Sell Signal: Price touches CRT high but closes below it

    Args:
        df: DataFrame with OHLC data
        crt_ranges: List of CRTRange objects
        atr_multiplier: ATR tolerance multiplier
        atr_period: ATR calculation period

    Returns:
        Dictionary with buy and sell signal indices
    """
    signals = {
        'buy': [],
        'sell': []
    }

    # Calculate ATR for tolerance
    atr = calculate_atr(df, atr_period)

    for idx in range(len(df)):
        tolerance = atr.iloc[idx] * atr_multiplier if not pd.isna(atr.iloc[idx]) else 0

        # Check each active CRT range
        for crt in crt_ranges:
            if not crt.is_active:
                continue

            if idx <= crt.detection_time_idx:
                continue

            row = df.iloc[idx]

            # Buy signal: low touches CRT low (with tolerance) and close > CRT low
            if not crt.buy_signaled:
                if row['low'] <= (crt.low + tolerance) and row['close'] > (crt.low - tolerance):
                    crt.buy_signaled = True
                    signals['buy'].append({
                        'idx': idx,
                        'price': row['close'],
                        'crt_level': crt.low
                    })

            # Sell signal: high touches CRT high (with tolerance) and close < CRT high
            if not crt.sell_signaled:
                if row['high'] >= (crt.high - tolerance) and row['close'] < (crt.high + tolerance):
                    crt.sell_signaled = True
                    signals['sell'].append({
                        'idx': idx,
                        'price': row['close'],
                        'crt_level': crt.high
                    })

            # Deactivate CRT if price breaks out
            if row['close'] > crt.high or row['close'] < crt.low:
                crt.is_active = False
                crt.end_idx = idx

    return signals


def filter_overlapping_crts(crt_ranges, enable_overlapping=False):
    """
    Filter CRT ranges to remove overlapping if not enabled

    Args:
        crt_ranges: List of CRTRange objects
        enable_overlapping: Whether to allow overlapping CRTs

    Returns:
        Filtered list of CRTRange objects
    """
    if enable_overlapping:
        return crt_ranges

    # Keep only the most recent non-overlapping CRT
    filtered = []
    for crt in crt_ranges:
        # Deactivate previous CRTs when a new one is detected
        if crt.is_active:
            for prev_crt in filtered:
                if prev_crt.is_active:
                    prev_crt.is_active = False
        filtered.append(crt)

    return filtered
