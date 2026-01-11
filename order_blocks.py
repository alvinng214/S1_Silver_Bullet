"""
Order Blocks (OB) Detection - Python translation of MirPapa ICT HTF OB

Order Blocks are zones where significant institutional orders were placed.
They represent the last candle before a strong move in the opposite direction.

Bullish Order Block: Last down candle before price reverses up
Bearish Order Block: Last up candle before price reverses down
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class OrderBlock:
    """Represents an Order Block zone"""
    timeframe: str
    is_bullish: bool
    top: float
    bottom: float
    start_idx: int
    end_idx: int  # Extended as price respects the zone
    created_at: int
    mitigated: bool = False

    def contains_price(self, price: float) -> bool:
        """Check if price is within the order block zone"""
        return self.bottom <= price <= self.top


def detect_order_blocks_simple(df, use_body=True, min_displacement=None):
    """
    Detect Order Blocks using simplified ICT logic.

    An Order Block forms when:
    1. Bullish OB: Price makes a down move, then strongly reverses up
       - Look for: low[1] < low[2] AND close breaks above high[1]
    2. Bearish OB: Price makes an up move, then strongly reverses down
       - Look for: high[1] > high[2] AND close breaks below low[1]

    Args:
        df: DataFrame with OHLC data and 'index' column
        use_body: If True, use open/close for box, else use high/low
        min_displacement: Minimum price move to confirm OB (optional)

    Returns:
        List of OrderBlock objects
    """
    order_blocks = []

    for i in range(3, len(df)):
        # Bullish Order Block: Last bearish candle before bullish reversal
        # Check if candle [i-1] made lower low than [i-2]
        if df['low'].iloc[i-1] < df['low'].iloc[i-2]:
            # Check if current candle breaks above the high of [i-1]
            if df['close'].iloc[i] > df['high'].iloc[i-1]:
                # Optional: Check for minimum displacement
                if min_displacement is None or (df['close'].iloc[i] - df['low'].iloc[i-1]) > min_displacement:
                    # The Order Block is candle [i-1]
                    if use_body:
                        top = max(df['open'].iloc[i-1], df['close'].iloc[i-1])
                        bottom = min(df['open'].iloc[i-1], df['close'].iloc[i-1])
                    else:
                        top = df['high'].iloc[i-1]
                        bottom = df['low'].iloc[i-1]

                    ob = OrderBlock(
                        timeframe="current",
                        is_bullish=True,
                        top=top,
                        bottom=bottom,
                        start_idx=int(df['index'].iloc[i-1]),
                        end_idx=int(df['index'].iloc[i]),
                        created_at=int(df['index'].iloc[i])
                    )
                    order_blocks.append(ob)

        # Bearish Order Block: Last bullish candle before bearish reversal
        # Check if candle [i-1] made higher high than [i-2]
        if df['high'].iloc[i-1] > df['high'].iloc[i-2]:
            # Check if current candle breaks below the low of [i-1]
            if df['close'].iloc[i] < df['low'].iloc[i-1]:
                # Optional: Check for minimum displacement
                if min_displacement is None or (df['high'].iloc[i-1] - df['close'].iloc[i]) > min_displacement:
                    # The Order Block is candle [i-1]
                    if use_body:
                        top = max(df['open'].iloc[i-1], df['close'].iloc[i-1])
                        bottom = min(df['open'].iloc[i-1], df['close'].iloc[i-1])
                    else:
                        top = df['high'].iloc[i-1]
                        bottom = df['low'].iloc[i-1]

                    ob = OrderBlock(
                        timeframe="current",
                        is_bullish=False,
                        top=top,
                        bottom=bottom,
                        start_idx=int(df['index'].iloc[i-1]),
                        end_idx=int(df['index'].iloc[i]),
                        created_at=int(df['index'].iloc[i])
                    )
                    order_blocks.append(ob)

    return order_blocks


def extend_order_blocks(order_blocks, df, max_extension=50):
    """
    Extend order blocks forward in time as long as price respects them.
    An OB is "alive" until price fully breaks through it.

    Args:
        order_blocks: List of OrderBlock objects
        df: DataFrame with OHLC data
        max_extension: Maximum bars to extend

    Returns:
        Updated list of OrderBlock objects
    """
    for ob in order_blocks:
        # Start from the candle after OB was created
        start_search = ob.created_at + 1
        end_search = min(start_search + max_extension, len(df))

        for i in range(start_search, end_search):
            if i >= len(df):
                break

            bar_idx = int(df['index'].iloc[i])
            close = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]

            # Check if OB is still valid (not fully mitigated)
            if ob.is_bullish:
                # Bullish OB is mitigated if close breaks below bottom
                if close < ob.bottom:
                    ob.mitigated = True
                    ob.end_idx = bar_idx
                    break
                # Extend if price touched the OB
                elif low <= ob.top:
                    ob.end_idx = bar_idx
            else:
                # Bearish OB is mitigated if close breaks above top
                if close > ob.top:
                    ob.mitigated = True
                    ob.end_idx = bar_idx
                    break
                # Extend if price touched the OB
                elif high >= ob.bottom:
                    ob.end_idx = bar_idx

    return order_blocks


def calculate_htf_order_blocks(df, timeframes_minutes={'15min': 15, '1H': 60}):
    """
    Calculate Order Blocks for multiple higher timeframes.

    Args:
        df: DataFrame with 5-minute OHLC data (with datetime index and 'index' column)
        timeframes_minutes: Dict of timeframe names to minutes

    Returns:
        Dictionary with order blocks for each timeframe
    """
    results = {}

    for tf_name, tf_minutes in timeframes_minutes.items():
        # Resample to HTF
        df_htf = df.resample(f'{tf_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'index': 'last'
        }).dropna()

        df_htf['index_htf'] = range(len(df_htf))
        df_htf['index'] = df_htf['index'].astype(int)

        # Detect OBs on HTF
        obs = detect_order_blocks_simple(df_htf, use_body=False)

        # Map HTF OB indices back to LTF
        for ob in obs:
            # Find the LTF index corresponding to HTF candle
            htf_candle_idx = ob.start_idx
            if htf_candle_idx < len(df_htf):
                ltf_idx = int(df_htf.iloc[htf_candle_idx]['index'])
                ob.start_idx = ltf_idx
                ob.created_at = ltf_idx

                # Find end index
                htf_end_idx = min(htf_candle_idx + 1, len(df_htf) - 1)
                ltf_end_idx = int(df_htf.iloc[htf_end_idx]['index'])
                ob.end_idx = ltf_end_idx

                ob.timeframe = tf_name

        # Extend OBs forward
        obs = extend_order_blocks(obs, df, max_extension=100)

        results[tf_name] = obs

    return results
