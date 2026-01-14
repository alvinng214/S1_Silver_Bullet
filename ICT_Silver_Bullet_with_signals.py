"""
ICT Silver Bullet with Signals
Translated from Pine Script by fikira

Detects Fair Value Gaps (FVGs) during ICT Silver Bullet sessions:
- London Open: 3-4 AM NY time
- AM Session: 10-11 AM NY time
- PM Session: 2-3 PM NY time

Also draws target lines from swing highs/lows to FVG retraces
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import time


@dataclass
class FVG:
    """Represents a Fair Value Gap"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_bullish: bool
    active: bool = False
    broken: bool = False
    current: bool = True
    formed_session: str = ""  # LN, AM, or PM
    targets: List['TargetLine'] = None

    def __post_init__(self):
        if self.targets is None:
            self.targets = []


@dataclass
class TargetLine:
    """Represents a support/resistance target line"""
    price: float
    source_idx: int
    source_type: str  # 'pivot_high', 'pivot_low', 'weekly', 'daily'
    active: bool = True
    reached: bool = False


@dataclass
class SilverBulletSession:
    """Represents a Silver Bullet session"""
    name: str
    start_time: time
    end_time: time
    start_idx: int
    end_idx: int
    fvgs: List[FVG]


def detect_pivots(df, left=10, right=1):
    """Detect pivot highs and lows"""
    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        # Check pivot high
        is_pivot_high = True
        for j in range(i - left, i + right + 1):
            if j != i and df['high'].iloc[j] >= df['high'].iloc[i]:
                is_pivot_high = False
                break

        if is_pivot_high:
            pivot_highs.append({'index': i, 'price': df['high'].iloc[i]})

        # Check pivot low
        is_pivot_low = True
        for j in range(i - left, i + right + 1):
            if j != i and df['low'].iloc[j] <= df['low'].iloc[i]:
                is_pivot_low = False
                break

        if is_pivot_low:
            pivot_lows.append({'index': i, 'price': df['low'].iloc[i]})

    return pivot_highs, pivot_lows


def detect_market_structure_trend(pivot_highs, pivot_lows, current_price):
    """Determine market structure trend (bullish/bearish)"""
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return 0

    # Check if price broke above recent pivot high (bullish MSS)
    for pivot in pivot_highs[-3:]:
        if current_price > pivot['price']:
            return 1

    # Check if price broke below recent pivot low (bearish MSS)
    for pivot in pivot_lows[-3:]:
        if current_price < pivot['price']:
            return -1

    return 0


def is_in_session(dt, session_name):
    """Check if datetime is in a Silver Bullet session"""
    # Convert to NY time (assuming input is already in correct timezone)
    hour = dt.hour
    minute = dt.minute

    if session_name == 'LN':  # London Open: 3-4 AM NY
        return hour == 3
    elif session_name == 'AM':  # AM Session: 10-11 AM NY
        return hour == 10
    elif session_name == 'PM':  # PM Session: 2-3 PM NY (14-15)
        return hour == 14

    return False


def detect_fvg_in_session(df, session_start_idx, session_end_idx, htf_minutes=15, trend=0, filter_by_trend=True):
    """
    Detect FVGs within a Silver Bullet session

    FVG occurs when:
    - Bullish FVG: candle[i].low > candle[i-2].high (gap up)
    - Bearish FVG: candle[i].high < candle[i-2].low (gap down)
    """
    fvgs = []

    # Resample to HTF if needed
    session_df = df.iloc[session_start_idx:session_end_idx + 1].copy()

    if htf_minutes > 5:  # Assuming base is 5min
        # Resample to HTF
        df_htf = session_df.resample(f'{htf_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        # Map HTF indices back to LTF
        htf_to_ltf = {}
        for htf_idx, htf_time in enumerate(df_htf.index):
            ltf_bars = session_df[session_df.index == htf_time]
            if len(ltf_bars) > 0:
                htf_to_ltf[htf_idx] = session_df.index.get_loc(ltf_bars.index[0]) + session_start_idx
    else:
        df_htf = session_df
        htf_to_ltf = {i: i + session_start_idx for i in range(len(df_htf))}

    # Detect FVGs
    for i in range(2, len(df_htf)):
        high_i = df_htf['high'].iloc[i]
        low_i = df_htf['low'].iloc[i]
        high_i2 = df_htf['high'].iloc[i-2]
        low_i2 = df_htf['low'].iloc[i-2]

        # Bullish FVG: current low > previous previous high
        if low_i > high_i2:
            if filter_by_trend and trend != 1:
                continue

            fvg = FVG(
                start_idx=htf_to_ltf.get(i-2, session_start_idx),
                end_idx=htf_to_ltf.get(i, session_start_idx),
                top=high_i2,
                bottom=low_i,
                is_bullish=True
            )
            fvgs.append(fvg)

        # Bearish FVG: current high < previous previous low
        elif high_i < low_i2:
            if filter_by_trend and trend != -1:
                continue

            fvg = FVG(
                start_idx=htf_to_ltf.get(i-2, session_start_idx),
                end_idx=htf_to_ltf.get(i, session_start_idx),
                top=low_i2,
                bottom=high_i,
                is_bullish=False
            )
            fvgs.append(fvg)

    return fvgs


def create_target_lines(pivot_highs, pivot_lows, fvg, current_idx, df, minimum_trade_framework=0):
    """Create target lines from pivots to FVG"""
    targets = []

    if fvg.is_bullish:
        # For bullish FVG, look for resistance targets (pivot highs) above FVG
        for pivot in pivot_highs:
            if pivot['index'] < fvg.start_idx and pivot['price'] > fvg.top + minimum_trade_framework:
                # Check if pivot hasn't been broken
                broken = False
                for i in range(pivot['index'], current_idx):
                    if i < len(df) and df['high'].iloc[i] > pivot['price']:
                        broken = True
                        break

                if not broken:
                    target = TargetLine(
                        price=pivot['price'],
                        source_idx=pivot['index'],
                        source_type='pivot_high'
                    )
                    targets.append(target)
    else:
        # For bearish FVG, look for support targets (pivot lows) below FVG
        for pivot in pivot_lows:
            if pivot['index'] < fvg.start_idx and pivot['price'] < fvg.bottom - minimum_trade_framework:
                # Check if pivot hasn't been broken
                broken = False
                for i in range(pivot['index'], current_idx):
                    if i < len(df) and df['low'].iloc[i] < pivot['price']:
                        broken = True
                        break

                if not broken:
                    target = TargetLine(
                        price=pivot['price'],
                        source_idx=pivot['index'],
                        source_type='pivot_low'
                    )
                    targets.append(target)

    return targets


def detect_silver_bullet_signals(df, htf_minutes=15, filter_by_trend=True, minimum_trade_framework=0):
    """
    Detect Silver Bullet sessions and FVGs

    Returns:
        Dictionary with sessions and signals
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df['time']))

    # Detect pivots
    pivot_highs, pivot_lows = detect_pivots(df, left=10, right=1)

    # Detect sessions
    sessions = []
    current_session = None

    for i in range(len(df)):
        dt = df.index[i]

        # Check which session we're in
        for session_name in ['LN', 'AM', 'PM']:
            if is_in_session(dt, session_name):
                if current_session is None or current_session.name != session_name:
                    # Starting new session
                    if current_session is not None:
                        current_session.end_idx = i - 1
                        sessions.append(current_session)

                    current_session = SilverBulletSession(
                        name=session_name,
                        start_time=dt.time(),
                        end_time=dt.time(),
                        start_idx=i,
                        end_idx=i,
                        fvgs=[]
                    )
                else:
                    current_session.end_idx = i
                    current_session.end_time = dt.time()
                break
        else:
            # Not in any session
            if current_session is not None:
                current_session.end_idx = i - 1
                sessions.append(current_session)
                current_session = None

    # Add last session if exists
    if current_session is not None:
        sessions.append(current_session)

    # Detect FVGs in each session
    for session in sessions:
        # Get trend at session end
        trend = detect_market_structure_trend(
            pivot_highs,
            pivot_lows,
            df['close'].iloc[session.end_idx]
        )

        # Detect FVGs
        fvgs = detect_fvg_in_session(
            df,
            session.start_idx,
            session.end_idx,
            htf_minutes=htf_minutes,
            trend=trend,
            filter_by_trend=filter_by_trend
        )

        session.fvgs = fvgs

        # Create target lines for FVGs that get activated
        for fvg in fvgs:
            # Check if price retraced into FVG after session
            for i in range(session.end_idx + 1, len(df)):
                if fvg.is_bullish and df['low'].iloc[i] <= fvg.top:
                    fvg.active = True
                    fvg.targets = create_target_lines(
                        pivot_highs, pivot_lows, fvg, i, df, minimum_trade_framework
                    )
                    break
                elif not fvg.is_bullish and df['high'].iloc[i] >= fvg.bottom:
                    fvg.active = True
                    fvg.targets = create_target_lines(
                        pivot_highs, pivot_lows, fvg, i, df, minimum_trade_framework
                    )
                    break

                # Check if FVG broken
                if fvg.is_bullish and df['close'].iloc[i] < fvg.bottom:
                    fvg.broken = True
                    fvg.current = False
                    break
                elif not fvg.is_bullish and df['close'].iloc[i] > fvg.top:
                    fvg.broken = True
                    fvg.current = False
                    break

    return {
        'sessions': sessions,
        'pivot_highs': pivot_highs,
        'pivot_lows': pivot_lows
    }
