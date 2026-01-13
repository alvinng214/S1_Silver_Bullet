"""
SW's Asia/London H/L's
Translated from Pine Script by steven_tyler22

Tracks and displays session highs and lows for:
- Asia Session (17:00-02:00 PT)
- London Session (00:00-09:00 PT)
- New York Session (05:30-14:00 PT)
- Previous Day High/Low (PDH/PDL) - 15:00-14:00 PT custom trading day
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta


@dataclass
class SessionLevel:
    """Represents a session high or low level"""
    price: float
    start_idx: int
    end_idx: int
    session_name: str
    is_high: bool
    bar_of_extreme: int  # Bar index where the extreme was set


def detect_session_levels(df, timezone_offset=0):
    """
    Detect Asia, London, and NY session highs/lows

    Args:
        df: DataFrame with datetime index and OHLC data
        timezone_offset: Hours offset from LA/PT time (default 0)

    Returns:
        Dictionary with session levels
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')

    # Session times in PT (Pacific Time / LA time)
    # Asia: 17:00 PT to 02:00 PT next day
    # London: 00:00 PT to 09:00 PT
    # NY: 05:30 PT to 14:00 PT

    session_levels = {
        'asia_high': [],
        'asia_low': [],
        'london_high': [],
        'london_low': [],
        'ny_high': [],
        'ny_low': []
    }

    # Track current session
    current_session = None
    session_start_idx = None
    session_high = None
    session_low = None
    session_high_bar = None
    session_low_bar = None

    for i in range(len(df)):
        dt = df.index[i]
        hour = dt.hour
        minute = dt.minute

        # Determine which session we're in
        in_asia = (hour >= 17) or (hour < 2)
        in_london = (0 <= hour < 9)
        in_ny = (hour == 5 and minute >= 30) or (6 <= hour < 14)

        # Determine session
        if in_asia:
            session = 'asia'
        elif in_london:
            session = 'london'
        elif in_ny:
            session = 'ny'
        else:
            session = None

        # Session change detection
        if session != current_session:
            # Freeze previous session if it exists
            if current_session is not None and session_high is not None:
                # Save the completed session
                session_levels[f'{current_session}_high'].append(SessionLevel(
                    price=session_high,
                    start_idx=session_start_idx,
                    end_idx=i,
                    session_name=current_session.upper(),
                    is_high=True,
                    bar_of_extreme=session_high_bar
                ))
                session_levels[f'{current_session}_low'].append(SessionLevel(
                    price=session_low,
                    start_idx=session_start_idx,
                    end_idx=i,
                    session_name=current_session.upper(),
                    is_high=False,
                    bar_of_extreme=session_low_bar
                ))

            # Start new session
            if session is not None:
                current_session = session
                session_start_idx = i
                session_high = df['high'].iloc[i]
                session_low = df['low'].iloc[i]
                session_high_bar = i
                session_low_bar = i

        # Update session extremes
        if session is not None and session == current_session:
            if df['high'].iloc[i] > session_high:
                session_high = df['high'].iloc[i]
                session_high_bar = i
            if df['low'].iloc[i] < session_low:
                session_low = df['low'].iloc[i]
                session_low_bar = i

    # Extend levels to current bar
    for key in session_levels:
        for level in session_levels[key]:
            level.end_idx = len(df) - 1

    return session_levels


def detect_pdh_pdl(df):
    """
    Detect Previous Day High and Low
    Custom trading day: 15:00 PT to 14:00 PT next day

    Args:
        df: DataFrame with datetime index and OHLC data

    Returns:
        Dictionary with PDH and PDL levels
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('datetime')

    pdh_pdl_levels = {
        'pdh': [],
        'pdl': [],
        'pd_mid': []
    }

    # Track current day session (15:00 to 14:00)
    current_day_high = None
    current_day_low = None
    current_day_high_bar = None
    current_day_low_bar = None
    prev_day_high = None
    prev_day_low = None
    prev_day_high_bar = None
    prev_day_low_bar = None
    session_start_idx = None

    for i in range(len(df)):
        dt = df.index[i]
        hour = dt.hour

        # Check if we're starting a new custom trading day (15:00 PT)
        if hour == 15 and (i == 0 or df.index[i-1].hour != 15):
            # Freeze previous day if exists
            if current_day_high is not None:
                prev_day_high = current_day_high
                prev_day_low = current_day_low
                prev_day_high_bar = current_day_high_bar
                prev_day_low_bar = current_day_low_bar

                # Save PDH/PDL
                if prev_day_high is not None:
                    pdh_pdl_levels['pdh'].append(SessionLevel(
                        price=prev_day_high,
                        start_idx=session_start_idx if session_start_idx else i,
                        end_idx=len(df) - 1,
                        session_name='PDH',
                        is_high=True,
                        bar_of_extreme=prev_day_high_bar
                    ))
                if prev_day_low is not None:
                    pdh_pdl_levels['pdl'].append(SessionLevel(
                        price=prev_day_low,
                        start_idx=session_start_idx if session_start_idx else i,
                        end_idx=len(df) - 1,
                        session_name='PDL',
                        is_high=False,
                        bar_of_extreme=prev_day_low_bar
                    ))
                if prev_day_high is not None and prev_day_low is not None:
                    pd_mid = (prev_day_high + prev_day_low) / 2
                    pdh_pdl_levels['pd_mid'].append(SessionLevel(
                        price=pd_mid,
                        start_idx=session_start_idx if session_start_idx else i,
                        end_idx=len(df) - 1,
                        session_name='PD MID',
                        is_high=False,
                        bar_of_extreme=min(prev_day_high_bar, prev_day_low_bar)
                    ))

            # Start new day
            session_start_idx = i
            current_day_high = df['high'].iloc[i]
            current_day_low = df['low'].iloc[i]
            current_day_high_bar = i
            current_day_low_bar = i

        # Update current day extremes (between 15:00 and 14:00)
        if current_day_high is not None:
            if df['high'].iloc[i] > current_day_high:
                current_day_high = df['high'].iloc[i]
                current_day_high_bar = i
            if df['low'].iloc[i] < current_day_low:
                current_day_low = df['low'].iloc[i]
                current_day_low_bar = i

    # Keep only the most recent PDH/PDL
    if pdh_pdl_levels['pdh']:
        pdh_pdl_levels['pdh'] = [pdh_pdl_levels['pdh'][-1]]
    if pdh_pdl_levels['pdl']:
        pdh_pdl_levels['pdl'] = [pdh_pdl_levels['pdl'][-1]]
    if pdh_pdl_levels['pd_mid']:
        pdh_pdl_levels['pd_mid'] = [pdh_pdl_levels['pd_mid'][-1]]

    return pdh_pdl_levels
