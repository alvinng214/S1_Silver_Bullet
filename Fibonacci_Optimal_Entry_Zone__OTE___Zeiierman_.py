"""
Fibonacci Optimal Entry Zone [OTE] (Zeiierman) - Python Translation

Detects market structure changes and plots Fibonacci retracement levels
for optimal trade entry zones (OTE).

Features:
- Pivot-based market structure detection
- Change of Character (CHoCH) identification
- Fibonacci retracement levels (0.50, 0.618 by default)
- Golden Zone highlighting (between first two Fib levels)
- Swing tracking mode (updates levels in real-time)
- Bullish and bearish structure identification

License: CC BY-NC-SA 4.0
Original: Zeiierman
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Pivot:
    """Pivot high or low point"""
    index: int
    price: float
    is_high: bool  # True for high, False for low


@dataclass
class FibLevel:
    """Fibonacci retracement level"""
    level: float  # e.g., 0.50, 0.618
    price: float
    start_idx: int
    end_idx: int
    color: str
    is_bullish: bool  # True for bullish structure, False for bearish


@dataclass
class Structure:
    """Market structure with Fibonacci levels"""
    is_bullish: bool
    swing_high: Pivot
    swing_low: Pivot
    choch_idx: int  # Change of Character index
    choch_price: float
    fib_levels: List[FibLevel] = field(default_factory=list)
    golden_zone_top: Optional[float] = None
    golden_zone_bottom: Optional[float] = None


def detect_pivot_highs_lows(df: pd.DataFrame, period: int = 10) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Detect pivot highs and lows

    Args:
        df: DataFrame with OHLC data
        period: Period for pivot detection (bars on each side)

    Returns: (pivot_highs, pivot_lows)
    """
    pivot_highs = []
    pivot_lows = []

    for i in range(period, len(df) - period):
        # Check pivot high
        is_pivot_high = True
        center_high = df.iloc[i]['high']

        for j in range(i - period, i + period + 1):
            if j != i and df.iloc[j]['high'] >= center_high:
                is_pivot_high = False
                break

        if is_pivot_high:
            pivot_highs.append(Pivot(index=i, price=center_high, is_high=True))

        # Check pivot low
        is_pivot_low = True
        center_low = df.iloc[i]['low']

        for j in range(i - period, i + period + 1):
            if j != i and df.iloc[j]['low'] <= center_low:
                is_pivot_low = False
                break

        if is_pivot_low:
            pivot_lows.append(Pivot(index=i, price=center_low, is_high=False))

    return pivot_highs, pivot_lows


def calculate_fibonacci_level(level: float, high_price: float, low_price: float,
                              high_idx: int, low_idx: int) -> float:
    """
    Calculate Fibonacci retracement level

    Args:
        level: Fibonacci level (e.g., 0.50, 0.618)
        high_price: Swing high price
        low_price: Swing low price
        high_idx: Index of swing high
        low_idx: Index of swing low

    Returns: Price at Fibonacci level
    """
    if low_idx < high_idx:
        # Bullish structure: retracing from high down to low
        return high_price - (high_price - low_price) * level
    else:
        # Bearish structure: retracing from low up to high
        return low_price + (high_price - low_price) * level


def detect_fibonacci_ote(
    df: pd.DataFrame,
    pivot_period: int = 10,
    fib_levels: List[float] = None,
    fib_colors: List[str] = None,
    show_bullish: bool = True,
    show_bearish: bool = True,
    swing_tracker: bool = True
) -> dict:
    """
    Detect Fibonacci Optimal Entry Zones with market structure

    Args:
        df: DataFrame with OHLC data and datetime index
        pivot_period: Period for pivot detection
        fib_levels: List of Fibonacci levels to plot (default [0.50, 0.618])
        fib_colors: Colors for each Fibonacci level
        show_bullish: Show bullish structures
        show_bearish: Show bearish structures
        swing_tracker: Enable real-time swing tracking

    Returns: Dictionary with structures, pivots, and Fibonacci levels
    """
    if fib_levels is None:
        fib_levels = [0.50, 0.618]
    if fib_colors is None:
        fib_colors = ['#4CAF50', '#009688']

    print(f"Detecting Fibonacci OTE with pivot_period={pivot_period}")

    # Detect pivots
    pivot_highs, pivot_lows = detect_pivot_highs_lows(df, period=pivot_period)

    print(f"Detected {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")

    # Track market structure
    structures = []

    # State variables
    position = 0  # 0=neutral, >0=bullish count, <0=bearish count
    up_price = 0.0
    down_price = float('inf')
    up_idx = 0
    down_idx = 0

    swing_low = None
    swing_high = None
    iswing_low = 0
    iswing_high = 0

    # Process bars to track market structure
    for i in range(len(df)):
        row = df.iloc[i]

        # Update running high/low
        up_price = max(up_price, row['high'])
        down_price = min(down_price, row['low'])

        # Check for pivot high at this bar
        pivot_high_at_i = None
        for ph in pivot_highs:
            if ph.index == i:
                pivot_high_at_i = ph
                break

        if pivot_high_at_i and position <= 0:
            up_price = pivot_high_at_i.price
            up_idx = i

        # Check for pivot low at this bar
        pivot_low_at_i = None
        for pl in pivot_lows:
            if pl.index == i:
                pivot_low_at_i = pl
                break

        if pivot_low_at_i and position >= 0:
            down_price = pivot_low_at_i.price
            down_idx = i

        # Track structure changes
        # Bullish structure: higher high
        if up_price > (0 if i == 0 else df.iloc[i-1].get('_up_price', 0)):
            up_idx = i

            if position <= 0:
                # CHoCH to bullish
                if show_bullish:
                    # Create bullish structure
                    swing_high_pivot = Pivot(index=up_idx, price=up_price, is_high=True)
                    swing_low_pivot = Pivot(index=down_idx, price=down_price, is_high=False)

                    # Calculate Fibonacci levels
                    fib_level_objects = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fibonacci_level(
                            level, up_price, down_price, up_idx, down_idx
                        )
                        color = fib_colors[idx] if idx < len(fib_colors) else '#4CAF50'

                        fib_level_objects.append(FibLevel(
                            level=level,
                            price=fib_price,
                            start_idx=down_idx,
                            end_idx=i,
                            color=color,
                            is_bullish=True
                        ))

                    # Calculate golden zone (between first two levels)
                    golden_top = None
                    golden_bottom = None
                    if len(fib_level_objects) >= 2:
                        golden_top = max(fib_level_objects[0].price, fib_level_objects[1].price)
                        golden_bottom = min(fib_level_objects[0].price, fib_level_objects[1].price)

                    structure = Structure(
                        is_bullish=True,
                        swing_high=swing_high_pivot,
                        swing_low=swing_low_pivot,
                        choch_idx=i,
                        choch_price=up_price,
                        fib_levels=fib_level_objects,
                        golden_zone_top=golden_top,
                        golden_zone_bottom=golden_bottom
                    )
                    structures.append(structure)

                position = 1
                swing_low = down_price
                iswing_low = down_idx

            elif position >= 1:
                # Update existing bullish structure
                if show_bullish and len(structures) > 0 and structures[-1].is_bullish:
                    last_struct = structures[-1]

                    # Recalculate Fibonacci levels
                    if swing_tracker:
                        # Use current swing
                        base_low = down_price
                        base_low_idx = down_idx
                    else:
                        # Use original swing
                        base_low = swing_low
                        base_low_idx = iswing_low

                    fib_level_objects = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fibonacci_level(
                            level, up_price, base_low, up_idx, base_low_idx
                        )
                        color = fib_colors[idx] if idx < len(fib_colors) else '#4CAF50'

                        fib_level_objects.append(FibLevel(
                            level=level,
                            price=fib_price,
                            start_idx=base_low_idx,
                            end_idx=i,
                            color=color,
                            is_bullish=True
                        ))

                    # Update structure
                    last_struct.swing_high = Pivot(index=up_idx, price=up_price, is_high=True)
                    last_struct.fib_levels = fib_level_objects

                    if len(fib_level_objects) >= 2:
                        last_struct.golden_zone_top = max(fib_level_objects[0].price, fib_level_objects[1].price)
                        last_struct.golden_zone_bottom = min(fib_level_objects[0].price, fib_level_objects[1].price)

                position = position + 1 if position > 0 else 2

        # Bearish structure: lower low
        if down_price < (float('inf') if i == 0 else df.iloc[i-1].get('_down_price', float('inf'))):
            down_idx = i

            if position >= 0:
                # CHoCH to bearish
                if show_bearish:
                    # Create bearish structure
                    swing_high_pivot = Pivot(index=up_idx, price=up_price, is_high=True)
                    swing_low_pivot = Pivot(index=down_idx, price=down_price, is_high=False)

                    # Calculate Fibonacci levels (from low to high)
                    fib_level_objects = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fibonacci_level(
                            level, up_price, down_price, up_idx, down_idx
                        )
                        color = fib_colors[idx] if idx < len(fib_colors) else '#FF2222'

                        fib_level_objects.append(FibLevel(
                            level=level,
                            price=fib_price,
                            start_idx=up_idx,
                            end_idx=i,
                            color=color,
                            is_bullish=False
                        ))

                    # Calculate golden zone
                    golden_top = None
                    golden_bottom = None
                    if len(fib_level_objects) >= 2:
                        golden_top = max(fib_level_objects[0].price, fib_level_objects[1].price)
                        golden_bottom = min(fib_level_objects[0].price, fib_level_objects[1].price)

                    structure = Structure(
                        is_bullish=False,
                        swing_high=swing_high_pivot,
                        swing_low=swing_low_pivot,
                        choch_idx=i,
                        choch_price=down_price,
                        fib_levels=fib_level_objects,
                        golden_zone_top=golden_top,
                        golden_zone_bottom=golden_bottom
                    )
                    structures.append(structure)

                position = -1
                swing_high = up_price
                iswing_high = up_idx

            elif position <= -1:
                # Update existing bearish structure
                if show_bearish and len(structures) > 0 and not structures[-1].is_bullish:
                    last_struct = structures[-1]

                    # Recalculate Fibonacci levels
                    if swing_tracker:
                        base_high = up_price
                        base_high_idx = up_idx
                    else:
                        base_high = swing_high
                        base_high_idx = iswing_high

                    fib_level_objects = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fibonacci_level(
                            level, base_high, down_price, base_high_idx, down_idx
                        )
                        color = fib_colors[idx] if idx < len(fib_colors) else '#FF2222'

                        fib_level_objects.append(FibLevel(
                            level=level,
                            price=fib_price,
                            start_idx=base_high_idx,
                            end_idx=i,
                            color=color,
                            is_bullish=False
                        ))

                    # Update structure
                    last_struct.swing_low = Pivot(index=down_idx, price=down_price, is_high=False)
                    last_struct.fib_levels = fib_level_objects

                    if len(fib_level_objects) >= 2:
                        last_struct.golden_zone_top = max(fib_level_objects[0].price, fib_level_objects[1].price)
                        last_struct.golden_zone_bottom = min(fib_level_objects[0].price, fib_level_objects[1].price)

                position = position - 1 if position < 0 else -2

        # Store for next iteration
        df.at[df.index[i], '_up_price'] = up_price
        df.at[df.index[i], '_down_price'] = down_price

    print(f"Detected {len(structures)} market structures")
    bullish_count = len([s for s in structures if s.is_bullish])
    bearish_count = len([s for s in structures if not s.is_bullish])
    print(f"  - Bullish: {bullish_count}")
    print(f"  - Bearish: {bearish_count}")

    return {
        'structures': structures,
        'pivot_highs': pivot_highs,
        'pivot_lows': pivot_lows
    }


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    results = detect_fibonacci_ote(
        df,
        pivot_period=10,
        fib_levels=[0.50, 0.618],
        show_bullish=True,
        show_bearish=True,
        swing_tracker=True
    )

    print(f"\nResults Summary:")
    print(f"  Structures: {len(results['structures'])}")
    print(f"  Pivot Highs: {len(results['pivot_highs'])}")
    print(f"  Pivot Lows: {len(results['pivot_lows'])}")
