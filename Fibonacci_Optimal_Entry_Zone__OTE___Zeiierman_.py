"""
Fibonacci Optimal Entry Zone [OTE] (Zeiierman) - Accurate Python Translation

Detects market structure changes (CHoCH) and plots Fibonacci retracement levels
for optimal trade entry zones based on swing highs/lows.

License: CC BY-NC-SA 4.0
Original: Zeiierman
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class FibLine:
    """Fibonacci level line"""
    level: float
    price: float
    start_idx: int
    end_idx: int
    color: str


@dataclass
class Structure:
    """Market structure with CHoCH and Fibonacci levels"""
    is_bullish: bool
    choch_idx: int
    choch_price: float
    swing_high: float
    swing_low: float
    swing_high_idx: int
    swing_low_idx: int
    fib_lines: List[FibLine]
    trend_line_start_idx: int
    trend_line_start_price: float
    trend_line_end_idx: int
    trend_line_end_price: float
    active: bool = True


def detect_pivot_high_low(highs: np.ndarray, lows: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect pivot highs and lows

    Returns: (pivot_highs, pivot_lows) arrays with values at pivot indices, na elsewhere
    """
    n = len(highs)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)

    for i in range(period, n - period):
        # Check pivot high
        is_pivot_high = True
        for j in range(i - period, i + period + 1):
            if j != i and highs[j] >= highs[i]:
                is_pivot_high = False
                break

        if is_pivot_high:
            pivot_highs[i] = highs[i]

        # Check pivot low
        is_pivot_low = True
        for j in range(i - period, i + period + 1):
            if j != i and lows[j] <= lows[i]:
                is_pivot_low = False
                break

        if is_pivot_low:
            pivot_lows[i] = lows[i]

    return pivot_highs, pivot_lows


def calculate_fib_level(level: float, high: float, low: float, high_idx: int, low_idx: int) -> float:
    """
    Calculate Fibonacci retracement level

    Pine: fibb(v, h, l, ih, il)
    """
    if low_idx < high_idx:
        # Bullish structure: retracing from high to low
        return high - (high - low) * level
    elif low_idx > high_idx:
        # Bearish structure: retracing from low to high
        return low + (high - low) * level
    else:
        return np.nan


def detect_fibonacci_ote_accurate(
    df: pd.DataFrame,
    pivot_period: int = 10,
    fib_levels: List[float] = None,
    fib_colors: List[str] = None,
    show_bullish: bool = True,
    show_bearish: bool = True,
    swing_tracker: bool = True,
    show_old: bool = False,
    extend: bool = True
) -> Dict:
    """
    Accurate translation of Fibonacci OTE indicator

    Args:
        df: DataFrame with OHLC data
        pivot_period: Period for pivot detection
        fib_levels: Fibonacci levels to plot (default [0.50, 0.618])
        fib_colors: Colors for each level
        show_bullish: Show bullish structures
        show_bearish: Show bearish structures
        swing_tracker: Follow mode - update levels dynamically
        show_old: Keep old structures
        extend: Extend Fibonacci lines to current bar

    Returns: Dictionary with structures and tracking data
    """
    if fib_levels is None:
        fib_levels = [0.50, 0.618]
    if fib_colors is None:
        fib_colors = ['#4CAF50', '#009688']

    print(f"Detecting Fibonacci OTE (period={pivot_period}, levels={fib_levels})")

    n = len(df)
    highs = df['high'].values
    lows = df['low'].values

    # Detect pivots
    pivot_highs, pivot_lows = detect_pivot_high_low(highs, lows, pivot_period)

    # State variables (matching Pine Script)
    pos = 0  # Position: 0=neutral, >0=bullish, <0=bearish

    Up = 0.0  # Running maximum
    Dn = float('inf')  # Running minimum
    iUp = 0  # Index of Up
    iDn = 0  # Index of Dn

    swingLow = np.nan  # Stored swing low when CHoCH occurred
    swingHigh = np.nan  # Stored swing high when CHoCH occurred
    iswingLow = 0  # Index of swing low
    iswingHigh = 0  # Index of swing high

    structures = [] if show_old else None
    current_structure = None

    # Process each bar
    for i in range(n):
        # Update running Up/Dn (Pine: Up := math.max(Up[1], high))
        Up = max(Up, highs[i])
        Dn = min(Dn, lows[i])

        # Check for pivot high (Pine: if not na(pvtHi) and pos <= 0)
        if not np.isnan(pivot_highs[i]) and pos <= 0:
            Up = pivot_highs[i]

        # Check for pivot low (Pine: if not na(pvtLo) and pos >= 0)
        if not np.isnan(pivot_lows[i]) and pos >= 0:
            Dn = pivot_lows[i]

        # Get previous values
        prev_Up = df.iloc[i-1]['_Up'] if i > 0 and '_Up' in df.columns else 0.0
        prev_Dn = df.iloc[i-1]['_Dn'] if i > 0 and '_Dn' in df.columns else float('inf')
        prev_iUp = int(df.iloc[i-1]['_iUp']) if i > 0 and '_iUp' in df.columns else 0
        prev_iDn = int(df.iloc[i-1]['_iDn']) if i > 0 and '_iDn' in df.columns else 0

        # Structure detection: if Up > Up[1]
        if Up > prev_Up:
            iUp = i

            # CHoCH to bullish (Pine: if pos <= 0)
            if pos <= 0:
                if show_bullish:
                    # Clear old structure if not keeping history
                    if not show_old and current_structure is not None:
                        current_structure.active = False

                    # Create Fibonacci levels
                    fib_lines = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fib_level(level, Up, Dn, iUp, iDn)
                        color = fib_colors[idx] if idx < len(fib_colors) else '#4CAF50'

                        fib_lines.append(FibLine(
                            level=level,
                            price=fib_price,
                            start_idx=iDn,
                            end_idx=i,
                            color=color
                        ))

                    # Create structure
                    current_structure = Structure(
                        is_bullish=True,
                        choch_idx=i,
                        choch_price=prev_Up,
                        swing_high=Up,
                        swing_low=Dn,
                        swing_high_idx=iUp,
                        swing_low_idx=iDn,
                        fib_lines=fib_lines,
                        trend_line_start_idx=iDn,
                        trend_line_start_price=Dn,
                        trend_line_end_idx=i,
                        trend_line_end_price=Up,
                        active=True
                    )

                    if show_old:
                        structures.append(current_structure)

                pos = 1
                swingLow = Dn
                iswingLow = iDn

            # Update existing bullish structure (Pine: else if pos == 1 or pos > 1)
            elif pos >= 1 and show_bullish and current_structure is not None:
                # Recalculate Fibonacci levels
                base_low = Dn if swing_tracker else swingLow
                base_low_idx = iDn if swing_tracker else iswingLow

                for idx, level in enumerate(fib_levels):
                    fib_price = calculate_fib_level(level, Up, base_low, iUp, base_low_idx)
                    if idx < len(current_structure.fib_lines):
                        current_structure.fib_lines[idx].price = fib_price
                        current_structure.fib_lines[idx].start_idx = base_low_idx
                        current_structure.fib_lines[idx].end_idx = i

                # Update swing high
                current_structure.swing_high = Up
                current_structure.swing_high_idx = iUp

                # Update trend line
                if swing_tracker:
                    current_structure.trend_line_start_idx = iDn
                    current_structure.trend_line_start_price = Dn
                else:
                    current_structure.trend_line_start_idx = iswingLow
                    current_structure.trend_line_start_price = swingLow

                current_structure.trend_line_end_idx = i
                current_structure.trend_line_end_price = Up

                pos = 2 if pos == 1 else pos + 1

        # Reset iUp when Up decreases (Pine: else if Up < Up[1])
        elif Up < prev_Up:
            iUp = i - pivot_period

        # Structure detection: if Dn < Dn[1]
        if Dn < prev_Dn:
            iDn = i

            # CHoCH to bearish (Pine: if pos >= 0)
            if pos >= 0:
                if show_bearish:
                    # Clear old structure if not keeping history
                    if not show_old and current_structure is not None:
                        current_structure.active = False

                    # Create Fibonacci levels
                    fib_lines = []
                    for idx, level in enumerate(fib_levels):
                        fib_price = calculate_fib_level(level, Up, Dn, iUp, iDn)
                        color = fib_colors[idx] if idx < len(fib_colors) else '#FF2222'

                        fib_lines.append(FibLine(
                            level=level,
                            price=fib_price,
                            start_idx=iUp,
                            end_idx=i,
                            color=color
                        ))

                    # Create structure
                    current_structure = Structure(
                        is_bullish=False,
                        choch_idx=i,
                        choch_price=prev_Dn,
                        swing_high=Up,
                        swing_low=Dn,
                        swing_high_idx=iUp,
                        swing_low_idx=iDn,
                        fib_lines=fib_lines,
                        trend_line_start_idx=iUp,
                        trend_line_start_price=Up,
                        trend_line_end_idx=i,
                        trend_line_end_price=Dn,
                        active=True
                    )

                    if show_old:
                        structures.append(current_structure)

                pos = -1
                swingHigh = Up
                iswingHigh = iUp

            # Update existing bearish structure (Pine: else if pos == -1 or pos < -1)
            elif pos <= -1 and show_bearish and current_structure is not None:
                # Recalculate Fibonacci levels
                base_high = Up if swing_tracker else swingHigh
                base_high_idx = iUp if swing_tracker else iswingHigh

                for idx, level in enumerate(fib_levels):
                    fib_price = calculate_fib_level(level, base_high, Dn, base_high_idx, iDn)
                    if idx < len(current_structure.fib_lines):
                        current_structure.fib_lines[idx].price = fib_price
                        current_structure.fib_lines[idx].start_idx = base_high_idx
                        current_structure.fib_lines[idx].end_idx = i

                # Update swing low
                current_structure.swing_low = Dn
                current_structure.swing_low_idx = iDn

                # Update trend line
                if swing_tracker:
                    current_structure.trend_line_start_idx = iUp
                    current_structure.trend_line_start_price = Up
                else:
                    current_structure.trend_line_start_idx = iswingHigh
                    current_structure.trend_line_start_price = swingHigh

                current_structure.trend_line_end_idx = i
                current_structure.trend_line_end_price = Dn

                pos = -2 if pos == -1 else pos - 1

        # Reset iDn when Dn increases (Pine: else if Dn > Dn[1])
        elif Dn > prev_Dn:
            iDn = i - pivot_period

        # Store state for next iteration
        df.at[df.index[i], '_Up'] = Up
        df.at[df.index[i], '_Dn'] = Dn
        df.at[df.index[i], '_iUp'] = iUp
        df.at[df.index[i], '_iDn'] = iDn
        df.at[df.index[i], '_pos'] = pos

        # Extend Fibonacci lines to current bar if enabled
        if extend and current_structure is not None and current_structure.active:
            if (pos >= 0 and show_bullish and current_structure.is_bullish) or \
               (pos <= 0 and show_bearish and not current_structure.is_bullish):
                for fib_line in current_structure.fib_lines:
                    fib_line.end_idx = i

    # Collect all structures
    all_structures = []
    if show_old and structures is not None:
        all_structures = structures
    if current_structure is not None and current_structure.active:
        all_structures.append(current_structure)

    print(f"Detected {len(all_structures)} structures")
    bullish_count = len([s for s in all_structures if s.is_bullish])
    bearish_count = len([s for s in all_structures if not s.is_bullish])
    print(f"  - Bullish: {bullish_count}")
    print(f"  - Bearish: {bearish_count}")

    return {
        'structures': all_structures,
        'current_structure': current_structure,
        'pivot_highs': pivot_highs,
        'pivot_lows': pivot_lows
    }


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    results = detect_fibonacci_ote_accurate(
        df,
        pivot_period=10,
        fib_levels=[0.50, 0.618],
        show_bullish=True,
        show_bearish=True,
        swing_tracker=True,
        show_old=False,
        extend=True
    )

    print(f"\nResults:")
    print(f"  Total structures: {len(results['structures'])}")
