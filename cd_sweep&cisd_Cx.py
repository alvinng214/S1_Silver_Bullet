"""
cd_sweep&cisd_Cx - Python Translation

Detects Higher Timeframe (HTF) sweeps and CISD (Change in State of Delivery) signals.
Based on Pine Script by cdikici71

Key features:
- HTF box tracking (current and previous HTF candles)
- Sweep detection (price sweeps HTF high/low then closes back inside)
- CISD level detection (specific candlestick patterns at HTF extremes)
- CISD signal triggers when price crosses CISD level after a sweep
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class HTFCandle:
    """Higher Timeframe candle data"""
    start_idx: int
    end_idx: int
    open: float
    high: float
    low: float
    close: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@dataclass
class SweepBox:
    """Sweep box marking where price swept HTF level"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_high_sweep: bool  # True if swept high, False if swept low


@dataclass
class CISDLevel:
    """CISD (Change in State of Delivery) level"""
    idx: int
    price: float
    is_bullish: bool  # True for bull CISD, False for bear CISD


@dataclass
class CISDSignal:
    """CISD signal when price crosses level after sweep"""
    idx: int
    cisd_start_idx: int
    cisd_price: float
    is_bullish: bool


def get_htf_timeframe(ltf_minutes: int) -> int:
    """
    Auto-select HTF based on current timeframe.

    Args:
        ltf_minutes: Lower timeframe in minutes

    Returns:
        Higher timeframe in minutes
    """
    if ltf_minutes == 1:
        return 15
    elif ltf_minutes in [2, 3, 5]:
        return 60
    elif ltf_minutes == 15:
        return 240
    elif ltf_minutes == 30:
        return 720
    elif ltf_minutes == 60:
        return 1440  # 1 day
    elif ltf_minutes == 240:
        return 10080  # 1 week
    elif ltf_minutes == 1440:
        return 43200  # 1 month (30 days)
    else:
        return ltf_minutes * 4  # Default: 4x current timeframe


def detect_cd_sweep_cisd(
    df: pd.DataFrame,
    htf_minutes: int = None,
    show_htf_boxes: bool = True,
    show_sweep_boxes: bool = True,
    show_cisd: bool = True
) -> Dict:
    """
    Detect HTF sweeps and CISD signals.

    Args:
        df: DataFrame with OHLC data and datetime index
        htf_minutes: Higher timeframe in minutes (auto if None)
        show_htf_boxes: Show HTF candle boxes
        show_sweep_boxes: Show sweep boxes
        show_cisd: Show CISD levels and signals

    Returns:
        Dictionary with HTF candles, sweeps, CISD levels, and signals
    """
    n = len(df)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Auto-select HTF if not provided
    if htf_minutes is None:
        # Detect current timeframe (assume 5 min for this data)
        time_diff = (df.index[1] - df.index[0]).total_seconds() / 60
        htf_minutes = get_htf_timeframe(int(time_diff))

    print(f"Using HTF: {htf_minutes} minutes")

    # Resample to HTF
    df_htf = df.resample(f'{htf_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # Track HTF candle formation on LTF
    htf_candles = []
    sweep_boxes = []
    cisd_levels = []
    cisd_signals = []

    # Variables matching Pine Script
    o0, h0, l0, c0 = None, -np.inf, np.inf, None
    o1, h1, l1, c1 = None, None, None, None
    t0_idx, t1_idx = None, None
    h0_bar, l0_bar = None, None
    h_swept, l_swept = False, False

    bull_level, bear_level = np.inf, -np.inf
    bull_index, bear_index = -1, -1
    xcisd, ycisd = False, False

    # Map HTF periods to LTF
    htf_period_map = {}
    for i in range(len(df_htf)):
        htf_time = df_htf.index[i]
        # Find LTF bars within this HTF period
        if i < len(df_htf) - 1:
            next_htf_time = df_htf.index[i + 1]
            mask = (df.index >= htf_time) & (df.index < next_htf_time)
        else:
            mask = df.index >= htf_time

        ltf_indices = df.index[mask]
        if len(ltf_indices) > 0:
            start_idx = df.index.get_loc(ltf_indices[0])
            end_idx = df.index.get_loc(ltf_indices[-1])
            htf_period_map[i] = (start_idx, end_idx)

    # Process each bar
    for i in range(n):
        row = df.iloc[i]

        # Check if new HTF period starts
        current_htf_idx = None
        for htf_idx, (start, end) in htf_period_map.items():
            if start <= i <= end:
                current_htf_idx = htf_idx
                break

        if current_htf_idx is not None and (t0_idx is None or i == htf_period_map[current_htf_idx][0]):
            # New HTF period
            if t0_idx is not None:
                # Save previous HTF candle
                o1, h1, l1, c1 = o0, h0, l0, c0
                t1_idx = t0_idx

                if show_htf_boxes and t1_idx is not None:
                    start, end = htf_period_map[current_htf_idx - 1]
                    htf_candles.append(HTFCandle(
                        start_idx=start,
                        end_idx=end,
                        open=o1,
                        high=h1,
                        low=l1,
                        close=c1,
                        start_time=df.index[start],
                        end_time=df.index[end]
                    ))

            # Reset for new HTF period
            t0_idx = i
            o0 = row['open']
            h0 = row['high']
            l0 = row['low']
            c0 = row['close']
            h0_bar = i
            l0_bar = i
        else:
            # Update running H/L for current HTF period
            if row['high'] >= h0:
                h0 = row['high']
                h0_bar = i
            if row['low'] <= l0:
                l0 = row['low']
                l0_bar = i
            c0 = row['close']

        # Detect sweeps (only if we have previous HTF data)
        if h1 is not None and l1 is not None:
            # High sweep: h0 > h1 and current candle closes below h1
            h_swept = h0 > h1 and max(row['open'], row['close']) < h1

            # Low sweep: l0 < l1 and current candle closes above l1
            l_swept = l0 < l1 and min(row['open'], row['close']) > l1

            # Create sweep boxes
            if show_sweep_boxes and t0_idx is not None:
                if h_swept and row['open'] < h1:
                    # High sweep box from h1 to h0
                    sweep_boxes.append(SweepBox(
                        start_idx=t0_idx,
                        end_idx=i,
                        top=h0,
                        bottom=h1,
                        is_high_sweep=True
                    ))

                if l_swept and row['open'] > l1:
                    # Low sweep box from l0 to l1
                    sweep_boxes.append(SweepBox(
                        start_idx=t0_idx,
                        end_idx=i,
                        top=l1,
                        bottom=l0,
                        is_high_sweep=False
                    ))

        # CISD Detection (only if show_cisd and we have HTF data)
        if show_cisd and h1 is not None and l1 is not None:
            up = row['close'] > row['open']
            dw = row['close'] < row['open']
            eq = row['close'] == row['open']

            # Bull CISD: At new low (low == l0 and low < l1)
            if row['low'] == l0 and row['low'] < l1:
                # Simple case: down candle now, up candle before
                if i > 0:
                    prev = df.iloc[i-1]
                    prev_up = prev['close'] > prev['open']
                    prev_eq = prev['close'] == prev['open']

                    if (dw or eq) and (prev_up or prev_eq) and not (eq and prev_eq):
                        bull_level = row['open']
                        bull_index = i
                        cisd_levels.append(CISDLevel(idx=i, price=bull_level, is_bullish=True))
                    else:
                        # Complex case: look back for pattern
                        for j in range(2, min(11, i+1)):
                            if df.iloc[i-j]['low'] < row['low']:
                                break

                            prev_j = df.iloc[i-j]
                            prev_j1 = df.iloc[i-j+1]
                            up_j = prev_j['close'] > prev_j['open']
                            eq_j = prev_j['close'] == prev_j['open']
                            dw_j1 = prev_j1['close'] < prev_j1['open']

                            if (up_j or eq_j) and dw_j1:
                                bar = j - 1
                                bull_level = df.iloc[i-bar]['open']
                                bull_index = i - bar

                                # Find highest open among down candles
                                for k in range(bar, -1, -1):
                                    candle_k = df.iloc[i-k]
                                    dw_k = candle_k['close'] < candle_k['open']
                                    if candle_k['open'] > bull_level and dw_k:
                                        bull_level = candle_k['open']
                                        bull_index = i - k

                                # Adjust if current candle matters
                                if bull_level < row['open'] and not up:
                                    bull_level = row['open']
                                    bull_index = i
                                if bull_level < row['open'] and up:
                                    bull_level = row['high']
                                    bull_index = i

                                cisd_levels.append(CISDLevel(idx=bull_index, price=bull_level, is_bullish=True))
                                break

            # Bear CISD: At new high (high == h0 and high > h1)
            if row['high'] == h0 and row['high'] > h1:
                if i > 0:
                    prev = df.iloc[i-1]
                    prev_dw = prev['close'] < prev['open']
                    prev_eq = prev['close'] == prev['open']

                    if (up or eq) and (prev_dw or prev_eq) and not (eq and prev_eq):
                        bear_level = row['open']
                        bear_index = i
                        cisd_levels.append(CISDLevel(idx=i, price=bear_level, is_bullish=False))
                    else:
                        # Complex case
                        for j in range(2, min(11, i+1)):
                            if df.iloc[i-j]['high'] > row['high']:
                                break

                            prev_j = df.iloc[i-j]
                            prev_j1 = df.iloc[i-j+1]
                            dw_j = prev_j['close'] < prev_j['open']
                            eq_j = prev_j['close'] == prev_j['open']
                            up_j1 = prev_j1['close'] > prev_j1['open']

                            if (dw_j or eq_j) and up_j1:
                                bar = j - 1
                                bear_level = df.iloc[i-bar]['open']
                                bear_index = i - bar

                                # Find lowest open among up candles
                                for k in range(bar, -1, -1):
                                    candle_k = df.iloc[i-k]
                                    up_k = candle_k['close'] > candle_k['open']
                                    if candle_k['open'] < bear_level and up_k:
                                        bear_level = candle_k['open']
                                        bear_index = i - k

                                # Adjust if current candle matters
                                if bear_level > row['open'] and not dw:
                                    bear_level = row['open']
                                    bear_index = i
                                if bear_level > row['open'] and dw:
                                    bear_level = row['low']
                                    bear_index = i

                                cisd_levels.append(CISDLevel(idx=bear_index, price=bear_level, is_bullish=False))
                                break

            # Reset CISD flags
            if row['high'] >= (h0 if i > 0 and h0 is not None else row['high']):
                ycisd = False
            if row['low'] <= (l0 if i > 0 and l0 is not None else row['low']):
                xcisd = False

            # Check for CISD signal triggers
            if i > 0:
                prev_close = df.iloc[i-1]['close']

                # Bull CISD signal: close above bull_level after low sweep
                if (prev_close > bull_level and
                    (l_swept or (l1 <= l0 and l_swept)) and
                    not xcisd and
                    i - 1 >= bull_index and
                    bull_level < np.inf):

                    cisd_signals.append(CISDSignal(
                        idx=i-1,
                        cisd_start_idx=bull_index,
                        cisd_price=bull_level,
                        is_bullish=True
                    ))
                    bull_level = np.inf
                    xcisd = True

                # Bear CISD signal: close below bear_level after high sweep
                if (prev_close < bear_level and
                    (h_swept or (h1 >= h0 and h_swept)) and
                    not ycisd and
                    i - 1 >= bear_index and
                    bear_level > -np.inf):

                    cisd_signals.append(CISDSignal(
                        idx=i-1,
                        cisd_start_idx=bear_index,
                        cisd_price=bear_level,
                        is_bullish=False
                    ))
                    bear_level = -np.inf
                    ycisd = True

    return {
        'htf_candles': htf_candles,
        'sweep_boxes': sweep_boxes,
        'cisd_levels': cisd_levels,
        'cisd_signals': cisd_signals,
        'htf_minutes': htf_minutes
    }


if __name__ == '__main__':
    print("cd_sweep&cisd_Cx - Python Translation")
    print("=" * 70)
    print("Detects HTF sweeps and CISD (Change in State of Delivery) signals.")
    print("=" * 70)
