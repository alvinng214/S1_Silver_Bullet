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


def calculate_liquidity_data(df: pd.DataFrame,
                             show_grabs: bool = True,
                             show_sweeps: bool = True,
                             show_equal_pivots: bool = True,
                             show_bsl_ssl: bool = True,
                             pivot_left: int = 3,
                             pivot_right: int = 3,
                             lookback: int = 5) -> dict:
    """
    Calculate all liquidity and inducement data

    Args:
        df: DataFrame with OHLC data
        show_grabs: Show liquidity grabs
        show_sweeps: Show liquidity sweeps
        show_equal_pivots: Show equal highs/lows
        show_bsl_ssl: Show BSL/SSL
        pivot_left: Left pivot length
        pivot_right: Right pivot length
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

    result = {
        'pivots_high': pivots_high,
        'pivots_low': pivots_low,
        'grabs': [],
        'sweeps': [],
        'equal_pivots': [],
        'bsl': [],
        'ssl': []
    }

    if show_grabs:
        result['grabs'] = detect_grabs(df, pivots_high, pivots_low, lookback)

    if show_sweeps:
        result['sweeps'] = detect_sweeps(df, pivots_high, pivots_low, lookback)

    if show_equal_pivots:
        equal_highs = detect_equal_pivots(pivots_high, atr, atr_factor=0.5, lookback=3, trend=0)
        equal_lows = detect_equal_pivots(pivots_low, atr, atr_factor=0.5, lookback=3, trend=0)
        result['equal_pivots'] = equal_highs + equal_lows

    if show_bsl_ssl:
        bsl, ssl = detect_external_liquidity(pivots_high, pivots_low, df, num_to_show=3)
        result['bsl'] = bsl
        result['ssl'] = ssl

    return result
