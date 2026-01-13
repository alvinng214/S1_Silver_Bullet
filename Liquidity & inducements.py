"""
Liquidity & Inducements Indicator
Translated from Pine Script by mickes

This indicator detects various types of liquidity and inducement zones:
- Grabs: Price wicks beyond pivot but closes back inside (liquidation grab)
- Sweeps: Price breaks through pivot and closes beyond it
- Equal Highs/Lows: Multiple pivots at similar price levels
- BSL/SSL: Buyside and Sellside Liquidity at untaken structure pivots
- Retracement Inducements: First pullback after break of structure
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


@dataclass
class Pivot:
    """Represents a pivot high or low"""
    price: float
    index: int
    type: int  # 1 for high, -1 for low


@dataclass
class LiquidityGrab:
    """Represents a liquidity grab (wick through pivot, close back inside)"""
    pivot: Pivot
    grab_index: int
    grab_price: float
    taken: bool = False
    invalidated: bool = False


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep (break through and close beyond pivot)"""
    pivot: Pivot
    sweep_index: int
    sweep_price: float
    is_bullish: bool  # True for bullish sweep (low swept), False for bearish


@dataclass
class EqualPivots:
    """Represents equal highs or equal lows (liquidity or inducement)"""
    pivot1: Pivot
    pivot2: Pivot
    is_liquidity: bool  # True for liquidity, False for inducement
    is_bullish: bool  # For inducement direction
    taken: bool = False


@dataclass
class ExternalLiquidity:
    """BSL (Buyside Liquidity) or SSL (Sellside Liquidity) at structure pivots"""
    pivot: Pivot
    is_buyside: bool  # True for BSL, False for SSL
    taken: bool = False


@dataclass
class RetracementInducement:
    """First retracement after break of structure"""
    pivot: Pivot
    is_bullish: bool  # True for bullish inducement, False for bearish
    start_idx: int
    invalidated: bool = False


@dataclass
class BigGrab:
    """Big liquidity grab with extended pivot lengths (typically from HTF)"""
    pivot: Pivot
    grab_index: int
    grab_price: float
    taken: bool = False
    invalidated: bool = False


@dataclass
class TurtleSoup:
    """Turtle Soup: Failed breakout pattern (price breaks pivot then reverses)"""
    pivot: Pivot
    break_index: int
    break_price: float
    reversal_index: int
    is_bullish: bool  # True for bullish turtle soup (broke high, reversed down)
    confirmed: bool = False  # Confirmed by CHoCH in opposite direction


def detect_pivots(df: pd.DataFrame, left_length: int = 5, right_length: int = 5) -> Tuple[List[Pivot], List[Pivot]]:
    """
    Detect pivot highs and lows

    Args:
        df: DataFrame with OHLC data
        left_length: Number of bars to the left
        right_length: Number of bars to the right

    Returns:
        Tuple of (pivot_highs, pivot_lows)
    """
    pivot_highs = []
    pivot_lows = []

    for i in range(left_length, len(df) - right_length):
        # Check for pivot high
        is_pivot_high = True
        for j in range(i - left_length, i + right_length + 1):
            if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                is_pivot_high = False
                break

        if is_pivot_high:
            pivot_highs.append(Pivot(
                price=df['high'].iloc[i],
                index=i,
                type=1
            ))

        # Check for pivot low
        is_pivot_low = True
        for j in range(i - left_length, i + right_length + 1):
            if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                is_pivot_low = False
                break

        if is_pivot_low:
            pivot_lows.append(Pivot(
                price=df['low'].iloc[i],
                index=i,
                type=-1
            ))

    return pivot_highs, pivot_lows


def detect_grabs(df: pd.DataFrame, pivots_high: List[Pivot], pivots_low: List[Pivot],
                 lookback: int = 5) -> List[LiquidityGrab]:
    """
    Detect liquidity grabs (wick through pivot, close back inside)

    Args:
        df: DataFrame with OHLC data
        pivots_high: List of pivot highs
        pivots_low: List of pivot lows
        lookback: Number of recent pivots to check

    Returns:
        List of LiquidityGrab objects
    """
    grabs = []

    # Check for bearish grabs (wick above pivot high, close back below)
    for pivot in pivots_high[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            high_price = df['high'].iloc[i]
            close_price = df['close'].iloc[i]

            # Wick goes above pivot but close is below
            if high_price >= pivot.price and close_price < pivot.price:
                grabs.append(LiquidityGrab(
                    pivot=pivot,
                    grab_index=i,
                    grab_price=high_price,
                    taken=True
                ))
                break
            # Invalidated if close goes above
            elif close_price > pivot.price:
                break

    # Check for bullish grabs (wick below pivot low, close back above)
    for pivot in pivots_low[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            low_price = df['low'].iloc[i]
            close_price = df['close'].iloc[i]

            # Wick goes below pivot but close is above
            if low_price <= pivot.price and close_price > pivot.price:
                grabs.append(LiquidityGrab(
                    pivot=pivot,
                    grab_index=i,
                    grab_price=low_price,
                    taken=True
                ))
                break
            # Invalidated if close goes below
            elif close_price < pivot.price:
                break

    return grabs


def detect_sweeps(df: pd.DataFrame, pivots_high: List[Pivot], pivots_low: List[Pivot],
                  lookback: int = 5) -> List[LiquiditySweep]:
    """
    Detect liquidity sweeps (break through and close beyond pivot)

    Args:
        df: DataFrame with OHLC data
        pivots_high: List of pivot highs
        pivots_low: List of pivot lows
        lookback: Number of recent pivots to check

    Returns:
        List of LiquiditySweep objects
    """
    sweeps = []

    # Check for bearish sweeps (high breaks through and close stays above)
    for pivot in pivots_high[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            high_price = df['high'].iloc[i]
            close_price = df['close'].iloc[i]

            # High goes above pivot and close stays above
            if high_price >= pivot.price and close_price >= pivot.price:
                sweeps.append(LiquiditySweep(
                    pivot=pivot,
                    sweep_index=i,
                    sweep_price=high_price,
                    is_bullish=False
                ))
                break

    # Check for bullish sweeps (low breaks through and close stays below)
    for pivot in pivots_low[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            low_price = df['low'].iloc[i]
            close_price = df['close'].iloc[i]

            # Low goes below pivot and close stays below
            if low_price <= pivot.price and close_price <= pivot.price:
                sweeps.append(LiquiditySweep(
                    pivot=pivot,
                    sweep_index=i,
                    sweep_price=low_price,
                    is_bullish=True
                ))
                break

    return sweeps


def detect_equal_pivots(pivots: List[Pivot], atr: float, atr_factor: float = 0.5,
                        lookback: int = 3, trend: int = 0) -> List[EqualPivots]:
    """
    Detect equal highs or equal lows

    Args:
        pivots: List of pivots (all highs or all lows)
        atr: Average True Range value
        atr_factor: ATR multiplier for equality threshold
        lookback: Number of pivots to look back
        trend: Current market trend (1 bullish, -1 bearish, 0 neutral)

    Returns:
        List of EqualPivots objects
    """
    equal_pivots = []
    threshold = atr * atr_factor

    if len(pivots) < 2:
        return equal_pivots

    for i in range(len(pivots) - 1):
        for j in range(i + 1, min(i + lookback + 1, len(pivots))):
            pivot1 = pivots[i]
            pivot2 = pivots[j]

            # Check if pivots are at equal levels (within ATR threshold)
            if abs(pivot1.price - pivot2.price) <= threshold:
                # Determine if it's liquidity or inducement based on trend
                is_liquidity = False
                is_bullish = False

                if pivot1.type == 1:  # Equal highs
                    # If trend is bearish, equal highs are inducement
                    if trend == -1:
                        is_liquidity = False
                        is_bullish = False
                    else:
                        is_liquidity = True
                else:  # Equal lows
                    # If trend is bullish, equal lows are inducement
                    if trend == 1:
                        is_liquidity = False
                        is_bullish = True
                    else:
                        is_liquidity = True

                equal_pivots.append(EqualPivots(
                    pivot1=pivot1,
                    pivot2=pivot2,
                    is_liquidity=is_liquidity,
                    is_bullish=is_bullish
                ))

    return equal_pivots


def detect_market_structure_trend(pivots_high: List[Pivot], pivots_low: List[Pivot], lookback: int = 3) -> int:
    """
    Detect current market structure trend based on recent pivots

    Args:
        pivots_high: List of pivot highs
        pivots_low: List of pivot lows
        lookback: Number of recent pivots to analyze

    Returns:
        1 for bullish trend (higher highs and higher lows)
        -1 for bearish trend (lower highs and lower lows)
        0 for neutral/choppy
    """
    if len(pivots_high) < 2 or len(pivots_low) < 2:
        return 0

    # Get recent pivots
    recent_highs = pivots_high[-min(lookback, len(pivots_high)):]
    recent_lows = pivots_low[-min(lookback, len(pivots_low)):]

    # Check for higher highs
    higher_highs = True
    for i in range(1, len(recent_highs)):
        if recent_highs[i].price <= recent_highs[i-1].price:
            higher_highs = False
            break

    # Check for higher lows
    higher_lows = True
    for i in range(1, len(recent_lows)):
        if recent_lows[i].price <= recent_lows[i-1].price:
            higher_lows = False
            break

    # Check for lower highs
    lower_highs = True
    for i in range(1, len(recent_highs)):
        if recent_highs[i].price >= recent_highs[i-1].price:
            lower_highs = False
            break

    # Check for lower lows
    lower_lows = True
    for i in range(1, len(recent_lows)):
        if recent_lows[i].price >= recent_lows[i-1].price:
            lower_lows = False
            break

    # Determine trend
    if higher_highs and higher_lows:
        return 1  # Bullish
    elif lower_highs and lower_lows:
        return -1  # Bearish
    else:
        return 0  # Neutral/choppy


def detect_external_liquidity(pivots_high: List[Pivot], pivots_low: List[Pivot],
                              df: pd.DataFrame, num_to_show: int = 1) -> Tuple[List[ExternalLiquidity], List[ExternalLiquidity]]:
    """
    Detect BSL (Buyside Liquidity) and SSL (Sellside Liquidity)
    These are untaken liquidity levels at market structure pivots

    Args:
        pivots_high: List of pivot highs
        pivots_low: List of pivot lows
        df: DataFrame with OHLC data
        num_to_show: Number of most recent liquidity levels to show

    Returns:
        Tuple of (BSL list, SSL list)
    """
    bsl_list = []  # Buyside liquidity
    ssl_list = []  # Sellside liquidity

    # Check pivot highs for BSL (untaken buyside liquidity)
    for pivot in pivots_high[-num_to_show:]:
        # Check if this pivot has been taken
        taken = False
        for i in range(pivot.index + 1, len(df)):
            if df['high'].iloc[i] >= pivot.price:
                taken = True
                break

        if not taken:
            bsl_list.append(ExternalLiquidity(
                pivot=pivot,
                is_buyside=True,
                taken=False
            ))

    # Check pivot lows for SSL (untaken sellside liquidity)
    for pivot in pivots_low[-num_to_show:]:
        # Check if this pivot has been taken
        taken = False
        for i in range(pivot.index + 1, len(df)):
            if df['low'].iloc[i] <= pivot.price:
                taken = True
                break

        if not taken:
            ssl_list.append(ExternalLiquidity(
                pivot=pivot,
                is_buyside=False,
                taken=False
            ))

    return bsl_list, ssl_list


def detect_big_grabs(df: pd.DataFrame, pivots_high: List[Pivot], pivots_low: List[Pivot],
                     pivot_left: int = 10, pivot_right: int = 10, lookback: int = 5) -> List[BigGrab]:
    """
    Detect big liquidity grabs using extended pivot lengths (typically for HTF analysis)

    Args:
        df: DataFrame with OHLC data
        pivots_high: List of pivot highs (can be from HTF)
        pivots_low: List of pivot lows (can be from HTF)
        pivot_left: Left pivot length (typically larger than regular grabs)
        pivot_right: Right pivot length
        lookback: Number of recent pivots to check

    Returns:
        List of BigGrab objects
    """
    big_grabs = []

    # Check for bearish big grabs (wick above pivot high, close back below)
    for pivot in pivots_high[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            high_price = df['high'].iloc[i]
            close_price = df['close'].iloc[i]

            # Wick goes above pivot but close is below
            if high_price >= pivot.price and close_price < pivot.price:
                big_grabs.append(BigGrab(
                    pivot=pivot,
                    grab_index=i,
                    grab_price=high_price,
                    taken=True
                ))
                break
            # Invalidated if close goes above
            elif close_price > pivot.price:
                break

    # Check for bullish big grabs (wick below pivot low, close back above)
    for pivot in pivots_low[-lookback:]:
        for i in range(pivot.index + 1, len(df)):
            low_price = df['low'].iloc[i]
            close_price = df['close'].iloc[i]

            # Wick goes below pivot but close is above
            if low_price <= pivot.price and close_price > pivot.price:
                big_grabs.append(BigGrab(
                    pivot=pivot,
                    grab_index=i,
                    grab_price=low_price,
                    taken=True
                ))
                break
            # Invalidated if close goes below
            elif close_price < pivot.price:
                break

    return big_grabs


def detect_turtle_soups(df: pd.DataFrame, pivots_high: List[Pivot], pivots_low: List[Pivot],
                        lookback: int = 5, require_confirmation: bool = True) -> List[TurtleSoup]:
    """
    Detect Turtle Soup patterns (failed breakouts)

    A turtle soup occurs when:
    - Price breaks above a pivot high but then reverses and closes below it (bearish)
    - Price breaks below a pivot low but then reverses and closes above it (bullish)

    Optionally requires confirmation via a change of character (CHoCH) in opposite direction

    Args:
        df: DataFrame with OHLC data
        pivots_high: List of pivot highs
        pivots_low: List of pivot lows
        lookback: Number of recent pivots to check
        require_confirmation: Whether to require CHoCH confirmation

    Returns:
        List of TurtleSoup objects
    """
    turtle_soups = []

    # Check for bearish turtle soups (broke high, reversed down)
    for pivot in pivots_high[-lookback:]:
        break_idx = None
        break_price = None

        for i in range(pivot.index + 1, len(df)):
            high_price = df['high'].iloc[i]
            close_price = df['close'].iloc[i]

            # Price breaks above pivot
            if high_price > pivot.price and break_idx is None:
                break_idx = i
                break_price = high_price

            # After break, check for reversal (close back below pivot)
            if break_idx is not None and i > break_idx:
                if close_price < pivot.price:
                    # Found turtle soup
                    turtle_soups.append(TurtleSoup(
                        pivot=pivot,
                        break_index=break_idx,
                        break_price=break_price,
                        reversal_index=i,
                        is_bullish=False,  # Bearish (broke high, reversed down)
                        confirmed=not require_confirmation  # If no confirmation required, mark as confirmed
                    ))
                    break
                # If price continues higher, not a turtle soup
                elif i > break_idx + 3:  # Give some bars for reversal
                    break

    # Check for bullish turtle soups (broke low, reversed up)
    for pivot in pivots_low[-lookback:]:
        break_idx = None
        break_price = None

        for i in range(pivot.index + 1, len(df)):
            low_price = df['low'].iloc[i]
            close_price = df['close'].iloc[i]

            # Price breaks below pivot
            if low_price < pivot.price and break_idx is None:
                break_idx = i
                break_price = low_price

            # After break, check for reversal (close back above pivot)
            if break_idx is not None and i > break_idx:
                if close_price > pivot.price:
                    # Found turtle soup
                    turtle_soups.append(TurtleSoup(
                        pivot=pivot,
                        break_index=break_idx,
                        break_price=break_price,
                        reversal_index=i,
                        is_bullish=True,  # Bullish (broke low, reversed up)
                        confirmed=not require_confirmation
                    ))
                    break
                # If price continues lower, not a turtle soup
                elif i > break_idx + 3:
                    break

    return turtle_soups


def calculate_liquidity_data(df: pd.DataFrame,
                             show_grabs: bool = True,
                             show_sweeps: bool = True,
                             show_equal_pivots: bool = True,
                             show_bsl_ssl: bool = True,
                             show_big_grabs: bool = False,
                             show_turtle_soups: bool = False,
                             pivot_left: int = 3,
                             pivot_right: int = 3,
                             big_grab_pivot_left: int = 10,
                             big_grab_pivot_right: int = 10,
                             turtle_soup_pivot_left: int = 1,
                             turtle_soup_pivot_right: int = 1,
                             lookback: int = 5) -> dict:
    """
    Calculate all liquidity and inducement data

    Args:
        df: DataFrame with OHLC data
        show_grabs: Show liquidity grabs
        show_sweeps: Show liquidity sweeps
        show_equal_pivots: Show equal highs/lows
        show_bsl_ssl: Show BSL/SSL
        show_big_grabs: Show big grabs (HTF grabs with larger pivot lengths)
        show_turtle_soups: Show turtle soup patterns (failed breakouts)
        pivot_left: Left pivot length for regular detection
        pivot_right: Right pivot length
        big_grab_pivot_left: Left pivot length for big grabs
        big_grab_pivot_right: Right pivot length for big grabs
        turtle_soup_pivot_left: Left pivot length for turtle soups
        turtle_soup_pivot_right: Right pivot length for turtle soups
        lookback: Lookback period

    Returns:
        Dictionary with all liquidity data
    """
    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    atr = df['tr'].rolling(window=14).mean().iloc[-1]

    # Detect pivots
    pivots_high, pivots_low = detect_pivots(df, pivot_left, pivot_right)

    # Detect market structure trend
    trend = detect_market_structure_trend(pivots_high, pivots_low, lookback=3)

    result = {
        'pivots_high': pivots_high,
        'pivots_low': pivots_low,
        'grabs': [],
        'sweeps': [],
        'equal_pivots': [],
        'bsl': [],
        'ssl': [],
        'big_grabs': [],
        'turtle_soups': [],
        'trend': trend  # Include trend in result
    }

    if show_grabs:
        result['grabs'] = detect_grabs(df, pivots_high, pivots_low, lookback)

    if show_sweeps:
        result['sweeps'] = detect_sweeps(df, pivots_high, pivots_low, lookback)

    if show_equal_pivots:
        # Pass actual trend instead of hardcoded 0
        equal_highs = detect_equal_pivots(pivots_high, atr, atr_factor=0.5, lookback=3, trend=trend)
        equal_lows = detect_equal_pivots(pivots_low, atr, atr_factor=0.5, lookback=3, trend=trend)
        result['equal_pivots'] = equal_highs + equal_lows

    if show_bsl_ssl:
        bsl, ssl = detect_external_liquidity(pivots_high, pivots_low, df, num_to_show=3)
        result['bsl'] = bsl
        result['ssl'] = ssl

    if show_big_grabs:
        # Detect big grabs with larger pivot lengths (typically for HTF)
        big_grab_highs, big_grab_lows = detect_pivots(df, big_grab_pivot_left, big_grab_pivot_right)
        result['big_grabs'] = detect_big_grabs(df, big_grab_highs, big_grab_lows,
                                                big_grab_pivot_left, big_grab_pivot_right, lookback)

    if show_turtle_soups:
        # Detect turtle soups with smaller pivot lengths
        turtle_soup_highs, turtle_soup_lows = detect_pivots(df, turtle_soup_pivot_left, turtle_soup_pivot_right)
        result['turtle_soups'] = detect_turtle_soups(df, turtle_soup_highs, turtle_soup_lows,
                                                      lookback=lookback, require_confirmation=True)

    return result
