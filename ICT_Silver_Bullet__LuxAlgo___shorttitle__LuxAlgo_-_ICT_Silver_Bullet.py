"""
ICT Silver Bullet [LuxAlgo] - Python Translation
Detects FVGs during Silver Bullet sessions with strict filtering and target lines from swings

Features:
- Silver Bullet sessions: London (3-4 AM), AM (10-11 AM), PM (2-3 PM) NY time
- FVG detection with activation logic (requires retrace)
- Multiple filtering modes: All FVG, Trend-based, Strict, Super-Strict
- Market Structure Shift (MSS) detection
- Target lines from swing pivots
- Session-specific pivot tracking

License: CC BY-NC-SA 4.0
Original: LuxAlgo
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class Pivot:
    """Swing pivot (high or low)"""
    index: int
    price: float
    broken: bool = False


@dataclass
class FVG:
    """Fair Value Gap with activation logic"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_bullish: bool
    active: bool = False  # Activated by retrace
    current: bool = True  # Still in current session
    session_name: str = ""  # LN, AM, PM, or GN


@dataclass
class TargetLine:
    """Support/Resistance target line from pivot"""
    price: float
    source_idx: int
    is_resistance: bool  # True for highs, False for lows
    active: bool = True
    reached: bool = False


@dataclass
class SessionPivots:
    """Track pivots for a specific session"""
    swing_highs: List[Pivot] = field(default_factory=list)
    swing_lows: List[Pivot] = field(default_factory=list)
    min_pivot: float = 1e10
    max_pivot: float = 0.0
    target_highs: List[TargetLine] = field(default_factory=list)
    target_lows: List[TargetLine] = field(default_factory=list)


@dataclass
class ZigZagPoint:
    """Point in zig-zag trend detection"""
    index: int
    price: float
    direction: int  # 1 for high, -1 for low


def is_in_silver_bullet_session(dt: datetime, session_name: str = None) -> Tuple[bool, str]:
    """
    Check if datetime is in a Silver Bullet session (NY time)

    Returns: (is_in_session, session_name)
    """
    hour = dt.hour

    # London Open: 3-4 AM NY
    in_ln = hour == 3
    # AM Session: 10-11 AM NY
    in_am = hour == 10
    # PM Session: 2-3 PM NY (14:00-15:00)
    in_pm = hour == 14

    if session_name:
        if session_name == 'LN':
            return in_ln, 'LN'
        elif session_name == 'AM':
            return in_am, 'AM'
        elif session_name == 'PM':
            return in_pm, 'PM'

    if in_ln:
        return True, 'LN'
    elif in_am:
        return True, 'AM'
    elif in_pm:
        return True, 'PM'
    else:
        return False, None


def detect_pivot_highs_lows(df: pd.DataFrame, left: int = 5, right: int = 1) -> Tuple[List[dict], List[dict]]:
    """
    Detect pivot highs and lows

    Args:
        df: DataFrame with OHLC data
        left: Bars to the left for pivot detection
        right: Bars to the right for pivot detection

    Returns: (pivot_highs, pivot_lows) as lists of {'index': int, 'price': float}
    """
    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        # Check pivot high
        is_pivot_high = True
        center_high = df.iloc[i]['high']

        for j in range(i - left, i):
            if df.iloc[j]['high'] >= center_high:
                is_pivot_high = False
                break

        if is_pivot_high:
            for j in range(i + 1, i + right + 1):
                if df.iloc[j]['high'] >= center_high:
                    is_pivot_high = False
                    break

        if is_pivot_high:
            pivot_highs.append({'index': i, 'price': center_high})

        # Check pivot low
        is_pivot_low = True
        center_low = df.iloc[i]['low']

        for j in range(i - left, i):
            if df.iloc[j]['low'] <= center_low:
                is_pivot_low = False
                break

        if is_pivot_low:
            for j in range(i + 1, i + right + 1):
                if df.iloc[j]['low'] <= center_low:
                    is_pivot_low = False
                    break

        if is_pivot_low:
            pivot_lows.append({'index': i, 'price': center_low})

    return pivot_highs, pivot_lows


def detect_market_structure_trend(df: pd.DataFrame, zigzag_points: List[ZigZagPoint]) -> int:
    """
    Detect market structure trend based on zig-zag points

    Returns: 1 for bullish, -1 for bearish, 0 for neutral
    """
    if len(zigzag_points) < 3:
        return 0

    # Find last high and low in zig-zag
    last_high_idx = -1
    last_low_idx = -1

    for i in range(len(zigzag_points) - 1, -1, -1):
        if zigzag_points[i].direction == 1 and last_high_idx == -1:
            last_high_idx = i
        if zigzag_points[i].direction == -1 and last_low_idx == -1:
            last_low_idx = i

        if last_high_idx != -1 and last_low_idx != -1:
            break

    if last_high_idx == -1 or last_low_idx == -1:
        return 0

    current_price = df.iloc[-1]['close']
    last_high = zigzag_points[last_high_idx].price
    last_low = zigzag_points[last_low_idx].price

    # MSS Bullish: price breaks above last swing high
    if current_price > last_high:
        return 1
    # MSS Bearish: price breaks below last swing low
    elif current_price < last_low:
        return -1
    else:
        return 0


def detect_luxalgo_silver_bullet(
    df: pd.DataFrame,
    pivot_left: int = 5,
    pivot_right: int = 1,
    filter_mode: str = 'Super-Strict',
    extend_fvg: bool = True,
    target_mode: str = 'previous session (similar)',
    keep_lines: bool = True
) -> dict:
    """
    Detect ICT Silver Bullet signals with LuxAlgo logic

    Args:
        df: DataFrame with OHLC data and datetime index
        pivot_left: Left period for pivot detection
        pivot_right: Right period for pivot detection
        filter_mode: 'All FVG', 'Only FVG in the same direction of trend', 'Strict', 'Super-Strict'
        extend_fvg: Extend FVG boxes when active
        target_mode: 'previous session (any)' or 'previous session (similar)'
        keep_lines: Keep target lines between sessions (only in strict modes)

    Returns: Dictionary with sessions, FVGs, targets, and pivots
    """
    print(f"Detecting LuxAlgo Silver Bullet with filter_mode={filter_mode}")

    # Detect all pivot highs and lows first
    pivot_highs, pivot_lows = detect_pivot_highs_lows(df, left=pivot_left, right=pivot_right)

    print(f"Detected {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")

    # Initialize session trackers
    sessions = {
        'GN': SessionPivots(),  # General session (all Silver Bullet times combined)
        'LN': SessionPivots(),  # London Open
        'AM': SessionPivots(),  # AM Session
        'PM': SessionPivots()   # PM Session
    }

    # Track zig-zag for market structure
    zigzag_points: List[ZigZagPoint] = []
    last_direction = 0  # 1 for high, -1 for low

    # Track FVGs
    bullish_fvgs: List[FVG] = []
    bearish_fvgs: List[FVG] = []

    # Track session state
    current_session = None
    session_start_idx = None
    session_high = -float('inf')
    session_low = float('inf')

    # Market structure trend
    trend = 0  # 1 bullish, -1 bearish, 0 neutral

    # Filter settings
    use_trend = filter_mode != 'All FVG'
    strict = filter_mode == 'Strict'
    super_strict = filter_mode == 'Super-Strict'
    is_strict_mode = strict or super_strict

    # Process each bar
    for i in range(len(df)):
        dt = df.index[i]
        row = df.iloc[i]

        in_session, sess_name = is_in_silver_bullet_session(dt)

        # Track session starts and ends
        was_in_session = current_session is not None
        session_started = in_session and not was_in_session
        session_ended = not in_session and was_in_session

        if session_started:
            current_session = sess_name
            session_start_idx = i
            session_high = row['high']
            session_low = row['low']

            # Clear target lines if not keeping them
            if is_strict_mode and not keep_lines:
                for sess in sessions.values():
                    sess.target_highs.clear()
                    sess.target_lows.clear()

        # Update session high/low
        if in_session:
            session_high = max(session_high, row['high'])
            session_low = min(session_low, row['low'])

        # Update zig-zag with pivots
        for ph in [p for p in pivot_highs if p['index'] == i]:
            if last_direction != 1:
                zigzag_points.append(ZigZagPoint(i, ph['price'], 1))
                last_direction = 1
            else:
                # Update last high if higher
                if len(zigzag_points) > 0 and ph['price'] > zigzag_points[-1].price:
                    zigzag_points[-1] = ZigZagPoint(i, ph['price'], 1)

        for pl in [p for p in pivot_lows if p['index'] == i]:
            if last_direction != -1:
                zigzag_points.append(ZigZagPoint(i, pl['price'], -1))
                last_direction = -1
            else:
                # Update last low if lower
                if len(zigzag_points) > 0 and pl['price'] < zigzag_points[-1].price:
                    zigzag_points[-1] = ZigZagPoint(i, pl['price'], -1)

        # Update market structure trend
        if len(zigzag_points) >= 2:
            # Find relevant highs and lows
            high_idx = -1
            low_idx = -1

            for j in range(len(zigzag_points) - 1, max(-1, len(zigzag_points) - 4), -1):
                if zigzag_points[j].direction == 1 and high_idx == -1:
                    high_idx = j
                if zigzag_points[j].direction == -1 and low_idx == -1:
                    low_idx = j

            if high_idx != -1 and row['close'] > zigzag_points[high_idx].price and trend < 1:
                trend = 1  # MSS Bullish
            elif low_idx != -1 and row['close'] < zigzag_points[low_idx].price and trend > -1:
                trend = -1  # MSS Bearish

        # Detect FVGs during Silver Bullet sessions
        if in_session and i >= 2:
            # Bullish FVG: current low > high[2]
            if row['low'] > df.iloc[i-2]['high']:
                # Check trend filter
                create_fvg = True
                if use_trend and trend != 1:
                    create_fvg = False

                if create_fvg:
                    fvg = FVG(
                        start_idx=i-2,
                        end_idx=i,
                        top=row['low'],
                        bottom=df.iloc[i-2]['high'],
                        is_bullish=True,
                        active=False,
                        current=True,
                        session_name=sess_name
                    )
                    bullish_fvgs.append(fvg)

            # Bearish FVG: current high < low[2]
            if row['high'] < df.iloc[i-2]['low']:
                # Check trend filter
                create_fvg = True
                if use_trend and trend != -1:
                    create_fvg = False

                if create_fvg:
                    fvg = FVG(
                        start_idx=i-2,
                        end_idx=i,
                        top=df.iloc[i-2]['low'],
                        bottom=row['high'],
                        is_bullish=False,
                        active=False,
                        current=True,
                        session_name=sess_name
                    )
                    bearish_fvgs.append(fvg)

        # Update FVG activation status
        for fvg in bullish_fvgs:
            if not fvg.current:
                continue

            # Check if price retraced into FVG to activate it
            if not fvg.active:
                if row['low'] < fvg.top and row['close'] > fvg.bottom:
                    fvg.active = True
                    if extend_fvg:
                        fvg.end_idx = i

            # Extend FVG if active
            if fvg.active and extend_fvg and in_session:
                fvg.end_idx = i

            # Check for invalidation (close below bottom)
            if in_session and row['close'] < fvg.bottom:
                if super_strict:
                    fvg.current = False
                if is_strict_mode:
                    fvg.active = False

        for fvg in bearish_fvgs:
            if not fvg.current:
                continue

            # Check if price retraced into FVG to activate it
            if not fvg.active:
                if row['high'] > fvg.bottom and row['close'] < fvg.top:
                    fvg.active = True
                    if extend_fvg:
                        fvg.end_idx = i

            # Extend FVG if active
            if fvg.active and extend_fvg and in_session:
                fvg.end_idx = i

            # Check for invalidation (close above top)
            if in_session and row['close'] > fvg.top:
                if super_strict:
                    fvg.current = False
                if is_strict_mode:
                    fvg.active = False

        # Add session pivots
        for ph in [p for p in pivot_highs if p['index'] == i]:
            pivot = Pivot(index=i, price=ph['price'])

            if in_session:
                # Add to current session
                if sess_name == 'LN':
                    sessions['LN'].swing_highs.append(pivot)
                    sessions['LN'].max_pivot = max(sessions['LN'].max_pivot, ph['price'])
                elif sess_name == 'AM':
                    sessions['AM'].swing_highs.append(pivot)
                    sessions['AM'].max_pivot = max(sessions['AM'].max_pivot, ph['price'])
                elif sess_name == 'PM':
                    sessions['PM'].swing_highs.append(pivot)
                    sessions['PM'].max_pivot = max(sessions['PM'].max_pivot, ph['price'])

                # Also add to general session
                sessions['GN'].swing_highs.append(pivot)
                sessions['GN'].max_pivot = max(sessions['GN'].max_pivot, ph['price'])

        for pl in [p for p in pivot_lows if p['index'] == i]:
            pivot = Pivot(index=i, price=pl['price'])

            if in_session:
                # Add to current session
                if sess_name == 'LN':
                    sessions['LN'].swing_lows.append(pivot)
                    sessions['LN'].min_pivot = min(sessions['LN'].min_pivot, pl['price'])
                elif sess_name == 'AM':
                    sessions['AM'].swing_lows.append(pivot)
                    sessions['AM'].min_pivot = min(sessions['AM'].min_pivot, pl['price'])
                elif sess_name == 'PM':
                    sessions['PM'].swing_lows.append(pivot)
                    sessions['PM'].min_pivot = min(sessions['PM'].min_pivot, pl['price'])

                # Also add to general session
                sessions['GN'].swing_lows.append(pivot)
                sessions['GN'].min_pivot = min(sessions['GN'].min_pivot, pl['price'])

        # Handle session end
        if session_ended:
            # Determine which session data to use
            if target_mode == 'previous session (similar)':
                sess_key = current_session  # Use session-specific pivots
            else:
                sess_key = 'GN'  # Use all sessions combined

            sess_data = sessions[sess_key]

            # Create target lines from swing highs (resistance)
            for swing in sess_data.swing_highs:
                # Filter: only create target if pivot is above session high (strict modes)
                threshold = session_low if is_strict_mode else session_high

                if swing.price > threshold:
                    target = TargetLine(
                        price=swing.price,
                        source_idx=swing.index,
                        is_resistance=True,
                        active=True,
                        reached=False
                    )
                    sess_data.target_highs.append(target)

            # Create target lines from swing lows (support)
            for swing in sess_data.swing_lows:
                # Filter: only create target if pivot is below session low (strict modes)
                threshold = session_high if is_strict_mode else session_low

                if swing.price < threshold:
                    target = TargetLine(
                        price=swing.price,
                        source_idx=swing.index,
                        is_resistance=False,
                        active=True,
                        reached=False
                    )
                    sess_data.target_lows.append(target)

            # Clear session pivots for next session
            sess_data.swing_highs.clear()
            sess_data.swing_lows.clear()
            sess_data.min_pivot = 1e10
            sess_data.max_pivot = 0.0

            # Mark FVGs as no longer current
            for fvg in bullish_fvgs + bearish_fvgs:
                if fvg.current:
                    # Validate at end of session
                    if strict and fvg.is_bullish and row['close'] < fvg.bottom:
                        fvg.active = False
                    if super_strict and fvg.is_bullish and row['close'] < fvg.top:
                        fvg.active = False
                    if strict and not fvg.is_bullish and row['close'] > fvg.top:
                        fvg.active = False
                    if super_strict and not fvg.is_bullish and row['close'] > fvg.bottom:
                        fvg.active = False

                    # Make inactive FVGs invisible
                    if not fvg.active:
                        fvg.current = False

                    fvg.current = False

            current_session = None

        # Update target line status
        for sess in sessions.values():
            for target in sess.target_highs:
                if target.active and row['high'] > target.price:
                    target.active = False
                    target.reached = True

            for target in sess.target_lows:
                if target.active and row['low'] < target.price:
                    target.active = False
                    target.reached = True

    # Filter to active FVGs only
    active_bull_fvgs = [fvg for fvg in bullish_fvgs if fvg.active]
    active_bear_fvgs = [fvg for fvg in bearish_fvgs if fvg.active]

    print(f"Total FVGs: {len(bullish_fvgs)} bullish, {len(bearish_fvgs)} bearish")
    print(f"Active FVGs: {len(active_bull_fvgs)} bullish, {len(active_bear_fvgs)} bearish")

    # Collect all target lines
    all_targets = []
    for sess in sessions.values():
        all_targets.extend(sess.target_highs)
        all_targets.extend(sess.target_lows)

    active_targets = [t for t in all_targets if t.active]

    print(f"Active target lines: {len(active_targets)}")

    return {
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
        'active_bull_fvgs': active_bull_fvgs,
        'active_bear_fvgs': active_bear_fvgs,
        'sessions': sessions,
        'all_targets': all_targets,
        'active_targets': active_targets,
        'pivot_highs': pivot_highs,
        'pivot_lows': pivot_lows,
        'zigzag_points': zigzag_points
    }


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    results = detect_luxalgo_silver_bullet(
        df,
        pivot_left=5,
        pivot_right=1,
        filter_mode='Super-Strict',
        extend_fvg=True
    )

    print(f"\nResults:")
    print(f"  Total bullish FVGs: {len(results['bullish_fvgs'])}")
    print(f"  Total bearish FVGs: {len(results['bearish_fvgs'])}")
    print(f"  Active bullish FVGs: {len(results['active_bull_fvgs'])}")
    print(f"  Active bearish FVGs: {len(results['active_bear_fvgs'])}")
    print(f"  Active targets: {len(results['active_targets'])}")
