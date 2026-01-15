"""
Monday Range (Lines) - Python Translation

Tracks Monday's OHLC range and extends key levels through the week.
Based on Pine Script by u_20bf

Key features:
- Stores Monday OHLC data for multiple weeks
- Draws Monday High (MH), Monday Low (ML), and custom levels
- Tracks breakouts and reclaims of Monday range
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class Monday:
    """Stores Monday (first trading day of week) OHLC data"""
    wk_start_idx: int  # Index where week starts
    wk_end_idx: int    # Index where week ends
    wk_start_time: pd.Timestamp  # Timestamp of week start
    wk_end_time: pd.Timestamp    # Timestamp of week end
    open: float
    high: float
    low: float
    close: float

    def price_range(self) -> float:
        """Calculate Monday's price range"""
        return self.high - self.low


@dataclass
class LevelConfig:
    """Configuration for a range level"""
    enabled: bool
    label: str
    value: float  # Range multiplier (0.0=Low, 0.5=Mid, 1.0=High)
    color: str
    line_style: str
    line_width: int


@dataclass
class RangeLevel:
    """A horizontal level line from Monday range"""
    start_idx: int
    end_idx: int
    price: float
    label: str
    color: str
    line_style: str
    line_width: int
    week_key: int  # Which Monday this belongs to


@dataclass
class Breakout:
    """Breakout or reclaim event"""
    idx: int
    price: float
    is_high_break: bool  # True if high break, False if low break
    is_reclaim: bool     # True if reclaim, False if initial breakout
    week_key: int


def detect_monday_ranges(
    df: pd.DataFrame,
    max_weeks: int = 4,
    show_mh: bool = True,
    show_ml: bool = True,
    show_mo: bool = False,
    show_mc: bool = False,
    custom_levels: List[LevelConfig] = None,
    extension_type: str = 'end_of_week',  # 'end_of_week', 'current_bar', 'fixed_bars'
    fixed_bars_count: int = 5,
    track_breakouts: bool = True,
    track_reclaims: bool = True
) -> Dict:
    """
    Detect Monday ranges and calculate levels.

    Args:
        df: DataFrame with OHLC data and datetime index
        max_weeks: Number of weeks to display (most recent)
        show_mh: Show Monday High
        show_ml: Show Monday Low
        show_mo: Show Monday Open
        show_mc: Show Monday Close
        custom_levels: List of custom level configurations
        extension_type: How to extend lines ('end_of_week', 'current_bar', 'fixed_bars')
        fixed_bars_count: Number of daily bars if using fixed_bars
        track_breakouts: Track when price breaks Monday high/low
        track_reclaims: Track when price reclaims back into range

    Returns:
        Dictionary with levels and breakouts
    """
    if custom_levels is None:
        # Default: just show equilibrium (mid-range)
        custom_levels = [
            LevelConfig(enabled=True, label='EQ', value=0.5, color='#007FFF',
                       line_style='solid', line_width=1)
        ]

    n = len(df)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # Add day of week column (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek

    # Detect week boundaries (new week = Monday or first day after weekend)
    # Use week number to detect changes
    df['week_num'] = df.index.isocalendar().week
    df['year'] = df.index.year
    df['week_year'] = df['year'].astype(str) + '_' + df['week_num'].astype(str)

    # Find where week changes
    df['new_week'] = df['week_year'] != df['week_year'].shift(1)

    # Store Monday data for each week
    mondays = []
    monday_map = {}  # Map from week_key to Monday object

    current_week_key = None
    week_start_idx = None
    week_ohlc = {'open': None, 'high': -np.inf, 'low': np.inf, 'close': None}

    for i in range(n):
        row = df.iloc[i]

        # New week detected
        if row['new_week']:
            # Save previous week if exists
            if current_week_key is not None and week_start_idx is not None:
                # Find week end (last bar before new week, or last bar in data)
                week_end_idx = i - 1
                if week_end_idx < 0:
                    week_end_idx = 0

                monday = Monday(
                    wk_start_idx=week_start_idx,
                    wk_end_idx=week_end_idx,
                    wk_start_time=df.index[week_start_idx],
                    wk_end_time=df.index[week_end_idx],
                    open=week_ohlc['open'],
                    high=week_ohlc['high'],
                    low=week_ohlc['low'],
                    close=week_ohlc['close']
                )
                mondays.append(monday)
                monday_map[current_week_key] = monday

            # Start new week
            current_week_key = row['week_year']
            week_start_idx = i
            week_ohlc = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
        else:
            # Update running OHLC for first day of week (Monday)
            # We only want Monday's OHLC, so stop updating after first day
            if week_start_idx is not None:
                # Check if still on first day (within ~24 hours of week start)
                time_since_start = (df.index[i] - df.index[week_start_idx]).total_seconds()
                if time_since_start < 24 * 3600:  # Within 24 hours
                    week_ohlc['high'] = max(week_ohlc['high'], row['high'])
                    week_ohlc['low'] = min(week_ohlc['low'], row['low'])
                    week_ohlc['close'] = row['close']

    # Save last week
    if current_week_key is not None and week_start_idx is not None:
        monday = Monday(
            wk_start_idx=week_start_idx,
            wk_end_idx=n - 1,
            wk_start_time=df.index[week_start_idx],
            wk_end_time=df.index[n - 1],
            open=week_ohlc['open'],
            high=week_ohlc['high'],
            low=week_ohlc['low'],
            close=week_ohlc['close']
        )
        mondays.append(monday)
        monday_map[current_week_key] = monday

    # Keep only last max_weeks
    mondays = mondays[-max_weeks:] if len(mondays) > max_weeks else mondays

    # Generate levels for each Monday
    levels = []

    for monday in mondays:
        monday_range = monday.price_range()

        # Determine extension end
        if extension_type == 'end_of_week':
            extension_end_idx = monday.wk_end_idx
        elif extension_type == 'current_bar':
            extension_end_idx = n - 1
        else:  # fixed_bars
            # Calculate based on daily bars (approximate with 5-min bars)
            bars_per_day = 288  # 24 * 60 / 5 = 288 bars per day for 5-min data
            extension_end_idx = min(n - 1, monday.wk_start_idx + fixed_bars_count * bars_per_day)

        week_key = id(monday)

        # Monday High
        if show_mh:
            levels.append(RangeLevel(
                start_idx=monday.wk_start_idx,
                end_idx=extension_end_idx,
                price=monday.high,
                label='MH',
                color='#007FFF',
                line_style='solid',
                line_width=1,
                week_key=week_key
            ))

        # Monday Low
        if show_ml:
            levels.append(RangeLevel(
                start_idx=monday.wk_start_idx,
                end_idx=extension_end_idx,
                price=monday.low,
                label='ML',
                color='#007FFF',
                line_style='solid',
                line_width=1,
                week_key=week_key
            ))

        # Monday Open
        if show_mo:
            levels.append(RangeLevel(
                start_idx=monday.wk_start_idx,
                end_idx=extension_end_idx,
                price=monday.open,
                label='MO',
                color='#007FFF',
                line_style='solid',
                line_width=1,
                week_key=week_key
            ))

        # Monday Close
        if show_mc:
            levels.append(RangeLevel(
                start_idx=monday.wk_start_idx,
                end_idx=extension_end_idx,
                price=monday.close,
                label='MC',
                color='#007FFF',
                line_style='solid',
                line_width=1,
                week_key=week_key
            ))

        # Custom levels (e.g., equilibrium at 0.5)
        for level_config in custom_levels:
            if level_config.enabled:
                level_price = monday.low + (monday_range * level_config.value)
                levels.append(RangeLevel(
                    start_idx=monday.wk_start_idx,
                    end_idx=extension_end_idx,
                    price=level_price,
                    label=level_config.label,
                    color=level_config.color,
                    line_style=level_config.line_style,
                    line_width=level_config.line_width,
                    week_key=week_key
                ))

    # Track breakouts and reclaims
    breakouts = []

    if track_breakouts or track_reclaims:
        for monday in mondays:
            week_key = id(monday)

            # Only check bars after Monday completes (skip first day)
            # Approximate: skip first 288 bars (one day) after week start
            start_check_idx = monday.wk_start_idx + 288

            high_touched = False
            low_touched = False

            for i in range(start_check_idx, monday.wk_end_idx + 1):
                if i >= n:
                    break

                row = df.iloc[i]

                # Track if high or low was touched
                if row['high'] > monday.high or row['close'] > monday.high:
                    if not high_touched and track_breakouts:
                        # First time crossing above high
                        if i > 0 and df.iloc[i-1]['close'] <= monday.high and row['close'] > monday.high:
                            breakouts.append(Breakout(
                                idx=i,
                                price=row['high'],
                                is_high_break=True,
                                is_reclaim=False,
                                week_key=week_key
                            ))
                    high_touched = True

                if row['low'] < monday.low or row['close'] < monday.low:
                    if not low_touched and track_breakouts:
                        # First time crossing below low
                        if i > 0 and df.iloc[i-1]['close'] >= monday.low and row['close'] < monday.low:
                            breakouts.append(Breakout(
                                idx=i,
                                price=row['low'],
                                is_high_break=False,
                                is_reclaim=False,
                                week_key=week_key
                            ))
                    low_touched = True

                # Check for reclaims
                if track_reclaims:
                    # High reclaim: was above, now closes back below
                    if high_touched and row['close'] < monday.high:
                        if i > 0 and df.iloc[i-1]['close'] >= monday.high:
                            breakouts.append(Breakout(
                                idx=i,
                                price=row['high'],
                                is_high_break=True,
                                is_reclaim=True,
                                week_key=week_key
                            ))
                            high_touched = False  # Reset

                    # Low reclaim: was below, now closes back above
                    if low_touched and row['close'] > monday.low:
                        if i > 0 and df.iloc[i-1]['close'] <= monday.low:
                            breakouts.append(Breakout(
                                idx=i,
                                price=row['low'],
                                is_high_break=False,
                                is_reclaim=True,
                                week_key=week_key
                            ))
                            low_touched = False  # Reset

    return {
        'mondays': mondays,
        'levels': levels,
        'breakouts': breakouts
    }


if __name__ == '__main__':
    # Test with sample data
    print("Monday Range (Lines) - Python Translation")
    print("=" * 70)
    print("This module detects Monday OHLC ranges and extends levels through the week.")
    print("=" * 70)
