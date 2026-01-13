"""
ICT Customizable 50% Line & Daily/Asia/London/New York High/Low + True Day Open
Python implementation of ICT trading levels and killzone sessions
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, time


@dataclass
class SessionLevel:
    """Represents a session high or low level"""
    price: float
    start_idx: int
    end_idx: int
    session_name: str
    is_high: bool
    bar_of_extreme: int


@dataclass
class DailyLevel:
    """Represents daily high, low, and 50% level"""
    high: float
    low: float
    mid: float
    start_idx: int
    end_idx: int
    market_open_idx: int


def detect_daily_levels(df):
    """
    Detect daily high, low, and 50% levels
    Session starts at 17:00 ET (5 PM Eastern / 2 PM Pacific)

    Returns:
        List of DailyLevel objects
    """
    daily_levels = []

    # Convert to ET timezone for session detection
    df_et = df.copy()
    if df_et.index.tz is None:
        df_et.index = df_et.index.tz_localize('UTC')
    df_et.index = df_et.index.tz_convert('America/New_York')

    current_session_start = None
    current_high = None
    current_low = None
    current_high_idx = None
    current_low_idx = None
    market_open_idx = None

    for idx in range(len(df_et)):
        dt = df_et.index[idx]
        hour = dt.hour

        # New session starts at 17:00 ET
        if hour == 17 and (idx == 0 or df_et.index[idx-1].hour != 17):
            # Save previous session if exists
            if current_session_start is not None and current_high is not None:
                mid = (current_high + current_low) / 2
                daily_levels.append(DailyLevel(
                    high=current_high,
                    low=current_low,
                    mid=mid,
                    start_idx=current_session_start,
                    end_idx=idx - 1,
                    market_open_idx=market_open_idx
                ))

            # Start new session
            current_session_start = idx
            current_high = df_et.iloc[idx]['high']
            current_low = df_et.iloc[idx]['low']
            current_high_idx = idx
            current_low_idx = idx
            market_open_idx = idx

        # Update session extremes
        if current_session_start is not None:
            if df_et.iloc[idx]['high'] > current_high:
                current_high = df_et.iloc[idx]['high']
                current_high_idx = idx
            if df_et.iloc[idx]['low'] < current_low:
                current_low = df_et.iloc[idx]['low']
                current_low_idx = idx

    # Add final session
    if current_session_start is not None and current_high is not None:
        mid = (current_high + current_low) / 2
        daily_levels.append(DailyLevel(
            high=current_high,
            low=current_low,
            mid=mid,
            start_idx=current_session_start,
            end_idx=len(df_et) - 1,
            market_open_idx=market_open_idx
        ))

    return daily_levels


def detect_killzone_sessions(df):
    """
    Detect ICT Killzone session levels

    Sessions (in ET - America/New_York):
    - Asia: 20:00 - 00:00 (8 PM - 12 AM)
    - London: 02:00 - 05:00 (2 AM - 5 AM)
    - NY Full: 09:00 - 16:00 (9:30 AM - 4 PM)
    - NY AM: 09:00 - 11:00 (9:30 AM - 11 AM)
    - NY Lunch: 11:00 - 13:00 (11 AM - 1 PM)
    - NY PM: 13:00 - 15:00 (1 PM - 3 PM)

    Returns:
        Dictionary with session levels for each killzone
    """
    session_levels = {
        'asia_high': [],
        'asia_low': [],
        'london_high': [],
        'london_low': [],
        'ny_high': [],
        'ny_low': [],
        'ny_am_high': [],
        'ny_am_low': [],
        'ny_lunch_high': [],
        'ny_lunch_low': [],
        'ny_pm_high': [],
        'ny_pm_low': []
    }

    # Convert to ET timezone
    df_et = df.copy()
    if df_et.index.tz is None:
        df_et.index = df_et.index.tz_localize('UTC')
    df_et.index = df_et.index.tz_convert('America/New_York')

    # Define sessions with their time ranges
    sessions = {
        'asia': (20, 24, 'asia_high', 'asia_low'),  # 20:00-00:00 (spans midnight)
        'london': (2, 5, 'london_high', 'london_low'),  # 02:00-05:00
        'ny': (9, 16, 'ny_high', 'ny_low'),  # 09:00-16:00
        'ny_am': (9, 11, 'ny_am_high', 'ny_am_low'),  # 09:00-11:00
        'ny_lunch': (11, 13, 'ny_lunch_high', 'ny_lunch_low'),  # 11:00-13:00
        'ny_pm': (13, 15, 'ny_pm_high', 'ny_pm_low')  # 13:00-15:00
    }

    for session_name, (start_hour, end_hour, high_key, low_key) in sessions.items():
        in_session = False
        session_high = None
        session_low = None
        session_start_idx = None
        session_high_idx = None
        session_low_idx = None
        prev_day = None

        for idx in range(len(df_et)):
            dt = df_et.index[idx]
            hour = dt.hour
            current_day = dt.date()

            # Reset on new day
            if prev_day is not None and current_day != prev_day:
                if in_session:
                    # Save previous session
                    if session_high is not None:
                        session_levels[high_key].append(SessionLevel(
                            price=session_high,
                            start_idx=session_start_idx,
                            end_idx=idx - 1,
                            session_name=session_name.upper(),
                            is_high=True,
                            bar_of_extreme=session_high_idx
                        ))
                        session_levels[low_key].append(SessionLevel(
                            price=session_low,
                            start_idx=session_start_idx,
                            end_idx=idx - 1,
                            session_name=session_name.upper(),
                            is_high=False,
                            bar_of_extreme=session_low_idx
                        ))
                in_session = False
                session_high = None
                session_low = None

            prev_day = current_day

            # Check if we're in the session
            if session_name == 'asia':
                # Asia spans midnight: 20:00-23:59 or 00:00-00:00
                is_in = hour >= start_hour or hour < (end_hour % 24)
            else:
                is_in = start_hour <= hour < end_hour

            # Start new session
            if is_in and not in_session:
                session_start_idx = idx
                session_high = df_et.iloc[idx]['high']
                session_low = df_et.iloc[idx]['low']
                session_high_idx = idx
                session_low_idx = idx
                in_session = True

            # Update session extremes
            elif is_in and in_session:
                if df_et.iloc[idx]['high'] > session_high:
                    session_high = df_et.iloc[idx]['high']
                    session_high_idx = idx
                if df_et.iloc[idx]['low'] < session_low:
                    session_low = df_et.iloc[idx]['low']
                    session_low_idx = idx

            # End session
            elif not is_in and in_session:
                if session_high is not None:
                    session_levels[high_key].append(SessionLevel(
                        price=session_high,
                        start_idx=session_start_idx,
                        end_idx=idx - 1,
                        session_name=session_name.upper(),
                        is_high=True,
                        bar_of_extreme=session_high_idx
                    ))
                    session_levels[low_key].append(SessionLevel(
                        price=session_low,
                        start_idx=session_start_idx,
                        end_idx=idx - 1,
                        session_name=session_name.upper(),
                        is_high=False,
                        bar_of_extreme=session_low_idx
                    ))
                in_session = False
                session_high = None
                session_low = None

        # Save final session if active
        if in_session and session_high is not None:
            session_levels[high_key].append(SessionLevel(
                price=session_high,
                start_idx=session_start_idx,
                end_idx=len(df_et) - 1,
                session_name=session_name.upper(),
                is_high=True,
                bar_of_extreme=session_high_idx
            ))
            session_levels[low_key].append(SessionLevel(
                price=session_low,
                start_idx=session_start_idx,
                end_idx=len(df_et) - 1,
                session_name=session_name.upper(),
                is_high=False,
                bar_of_extreme=session_low_idx
            ))

    return session_levels
