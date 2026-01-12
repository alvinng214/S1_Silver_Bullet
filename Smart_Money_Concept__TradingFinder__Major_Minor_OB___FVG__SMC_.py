"""
Fair Value Gap (FVG) Detection - Python translation of Smart Money Concept FVG

A Fair Value Gap is an imbalance or inefficiency in the market created by rapid price movement.

Bullish FVG: Gap between high[2] and low[0] when there's no overlap
Bearish FVG: Gap between low[2] and high[0] when there's no overlap

These gaps often act as magnets for price to return and "fill" them.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class FVG:
    """Represents a Fair Value Gap"""
    is_bullish: bool
    top: float
    bottom: float
    start_idx: int
    end_idx: int
    created_at: int
    mitigated: bool = False
    mitigation_idx: int = None

    def contains_price(self, price: float) -> bool:
        """Check if price is within the FVG zone"""
        return self.bottom <= price <= self.top

    def get_width(self) -> float:
        """Get the width of the FVG"""
        return self.top - self.bottom


def detect_fvg(df, filter_type='Defensive'):
    """
    Detect Fair Value Gaps in the price data.

    FVG Detection Logic:
    - Bullish FVG: high[i-2] < low[i] (there's a gap up)
    - Bearish FVG: low[i-2] > high[i] (there's a gap down)

    Args:
        df: DataFrame with OHLC data and 'index' column
        filter_type: Filter based on FVG width
            'Very Aggressive': No filter, show all FVGs
            'Aggressive': Filter FVGs smaller than 0.5 ATR
            'Defensive': Filter FVGs smaller than 0.7 ATR
            'Very Defensive': Filter FVGs smaller than 1.0 ATR

    Returns:
        List of FVG objects
    """
    fvgs = []

    # Calculate ATR for filtering if needed
    if filter_type != 'Very Aggressive':
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        atr = df['tr'].rolling(14).mean()

        # Set filter threshold based on type
        filter_multipliers = {
            'Aggressive': 0.5,
            'Defensive': 0.7,
            'Very Defensive': 1.0
        }
        filter_mult = filter_multipliers.get(filter_type, 0.7)
    else:
        atr = None
        filter_mult = 0

    for i in range(2, len(df)):
        # Bullish FVG: Gap between high[i-2] and low[i]
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            gap_top = df['low'].iloc[i]
            gap_bottom = df['high'].iloc[i-2]
            gap_width = gap_top - gap_bottom

            # Apply filter if needed
            if atr is not None:
                min_width = atr.iloc[i] * filter_mult
                if gap_width < min_width:
                    continue

            fvg = FVG(
                is_bullish=True,
                top=gap_top,
                bottom=gap_bottom,
                start_idx=int(df['index'].iloc[i-1]),
                end_idx=int(df['index'].iloc[i]),
                created_at=int(df['index'].iloc[i])
            )
            fvgs.append(fvg)

        # Bearish FVG: Gap between low[i-2] and high[i]
        elif df['low'].iloc[i-2] > df['high'].iloc[i]:
            gap_top = df['low'].iloc[i-2]
            gap_bottom = df['high'].iloc[i]
            gap_width = gap_top - gap_bottom

            # Apply filter if needed
            if atr is not None:
                min_width = atr.iloc[i] * filter_mult
                if gap_width < min_width:
                    continue

            fvg = FVG(
                is_bullish=False,
                top=gap_top,
                bottom=gap_bottom,
                start_idx=int(df['index'].iloc[i-1]),
                end_idx=int(df['index'].iloc[i]),
                created_at=int(df['index'].iloc[i])
            )
            fvgs.append(fvg)

    return fvgs


def extend_fvgs(fvgs, df, max_extension=100):
    """
    Extend FVGs forward and mark them as mitigated when filled.

    A FVG is mitigated when:
    - Bullish FVG: Price closes below the bottom of the gap
    - Bearish FVG: Price closes above the top of the gap

    Args:
        fvgs: List of FVG objects
        df: DataFrame with OHLC data
        max_extension: Maximum bars to extend

    Returns:
        Updated list of FVG objects
    """
    for fvg in fvgs:
        # Start from the bar after FVG was created
        start_search = fvg.created_at + 1
        end_search = min(start_search + max_extension, len(df))

        for i in range(start_search, end_search):
            if i >= len(df):
                break

            bar_idx = int(df['index'].iloc[i])
            close = df['close'].iloc[i]
            low = df['low'].iloc[i]
            high = df['high'].iloc[i]

            # Check if FVG is mitigated
            if fvg.is_bullish:
                # Bullish FVG is mitigated if price closes below bottom
                if close < fvg.bottom:
                    fvg.mitigated = True
                    fvg.mitigation_idx = bar_idx
                    fvg.end_idx = bar_idx
                    break
                # Extend if price is near or in the FVG
                elif low <= fvg.top:
                    fvg.end_idx = bar_idx
            else:
                # Bearish FVG is mitigated if price closes above top
                if close > fvg.top:
                    fvg.mitigated = True
                    fvg.mitigation_idx = bar_idx
                    fvg.end_idx = bar_idx
                    break
                # Extend if price is near or in the FVG
                elif high >= fvg.bottom:
                    fvg.end_idx = bar_idx

    return fvgs


def calculate_fvgs_for_chart(df, show_demand=True, show_supply=True, filter_type='Defensive'):
    """
    Calculate FVGs for display on chart.

    Args:
        df: DataFrame with OHLC data and 'index' column
        show_demand: Whether to show bullish FVGs
        show_supply: Whether to show bearish FVGs
        filter_type: FVG filter type

    Returns:
        Dictionary with bullish and bearish FVGs
    """
    # Detect all FVGs
    all_fvgs = detect_fvg(df, filter_type=filter_type)

    # Extend FVGs forward
    all_fvgs = extend_fvgs(all_fvgs, df, max_extension=150)

    # Split into bullish and bearish
    bullish_fvgs = [fvg for fvg in all_fvgs if fvg.is_bullish and show_demand]
    bearish_fvgs = [fvg for fvg in all_fvgs if not fvg.is_bullish and show_supply]

    return {
        'bullish': bullish_fvgs,
        'bearish': bearish_fvgs,
        'all': all_fvgs
    }
