"""
Smart Money Zones (FVG + OB) + MTF Trend Panel
Translated from Pine Script

This indicator combines:
1. Fair Value Gaps (FVG) detection
2. Order Blocks (OB) detection
3. Multi-Timeframe Trend Panel
4. Zone strength classification
5. Mitigation tracking
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class SMZone:
    """Smart Money Zone (FVG or OB)"""
    top: float
    bottom: float
    start_idx: int
    end_idx: int
    created_at: int
    is_fvg: bool  # True for FVG, False for OB
    is_bullish: bool
    strength: str  # WEAK, MEDIUM, STRONG, VERY STRONG
    is_live: bool = True
    mitigated_amount: float = 0.0
    mitigation_idx: int = None


def calculate_body_percent(open_val, high_val, low_val, close_val):
    """Calculate body percentage of candle"""
    candle_range = high_val - low_val
    body = abs(close_val - open_val)
    return body / candle_range if candle_range > 0 else 0.0


def get_zone_strength(zone_size, atr_val):
    """
    Classify zone strength based on size relative to ATR

    Args:
        zone_size: Size of the zone
        atr_val: Current ATR value

    Returns:
        Strength classification string
    """
    if atr_val <= 0:
        return "MEDIUM"

    ratio = zone_size / atr_val

    if ratio >= 2.0:
        return "VERY STRONG"
    elif ratio >= 1.5:
        return "STRONG"
    elif ratio >= 1.0:
        return "MEDIUM"
    else:
        return "WEAK"


def detect_smart_money_zones(df,
                             show_fvg=True,
                             show_ob=True,
                             max_zones=20,
                             min_body_pct=0.5,
                             ob_lookback=10,
                             atr_length=14,
                             use_trend_filter=True,
                             trend_ma_period=50):
    """
    Detect Smart Money Zones (FVG + OB) with trend filtering

    Args:
        df: DataFrame with OHLC data and 'index' column
        show_fvg: Show Fair Value Gaps
        show_ob: Show Order Blocks
        max_zones: Maximum zones per type to keep
        min_body_pct: Minimum impulse candle body percentage
        ob_lookback: Lookback period for Order Block detection
        atr_length: ATR calculation period
        use_trend_filter: Apply trend filter
        trend_ma_period: Trend MA period

    Returns:
        Dictionary with bullish and bearish zones (FVG and OB)
    """
    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_length).mean()

    # Calculate trend MA
    df['trend_ma'] = df['close'].rolling(window=trend_ma_period).mean()
    df['is_uptrend'] = df['close'] > df['trend_ma']
    df['is_downtrend'] = df['close'] < df['trend_ma']

    bull_fvgs = []
    bear_fvgs = []
    bull_obs = []
    bear_obs = []

    # Detect FVGs
    if show_fvg:
        for i in range(3, len(df)):
            atr_val = df['atr'].iloc[i]

            # Bullish FVG: low > high[2]
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                mid_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
                mid_body_pct = calculate_body_percent(
                    df['open'].iloc[i-1],
                    df['high'].iloc[i-1],
                    df['low'].iloc[i-1],
                    df['close'].iloc[i-1]
                )

                is_uptrend = df['is_uptrend'].iloc[i]

                if mid_bullish and mid_body_pct >= min_body_pct and (not use_trend_filter or is_uptrend):
                    top = df['low'].iloc[i]
                    bottom = df['high'].iloc[i-2]
                    gap_size = top - bottom
                    strength = get_zone_strength(gap_size, atr_val)

                    zone = SMZone(
                        top=top,
                        bottom=bottom,
                        start_idx=int(df['index'].iloc[i-2]),
                        end_idx=int(df['index'].iloc[i]),
                        created_at=int(df['index'].iloc[i]),
                        is_fvg=True,
                        is_bullish=True,
                        strength=strength
                    )
                    bull_fvgs.append(zone)

            # Bearish FVG: high < low[2]
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                mid_bearish = df['close'].iloc[i-1] < df['open'].iloc[i-1]
                mid_body_pct = calculate_body_percent(
                    df['open'].iloc[i-1],
                    df['high'].iloc[i-1],
                    df['low'].iloc[i-1],
                    df['close'].iloc[i-1]
                )

                is_downtrend = df['is_downtrend'].iloc[i]

                if mid_bearish and mid_body_pct >= min_body_pct and (not use_trend_filter or is_downtrend):
                    top = df['low'].iloc[i-2]
                    bottom = df['high'].iloc[i]
                    gap_size = top - bottom
                    strength = get_zone_strength(gap_size, atr_val)

                    zone = SMZone(
                        top=top,
                        bottom=bottom,
                        start_idx=int(df['index'].iloc[i-2]),
                        end_idx=int(df['index'].iloc[i]),
                        created_at=int(df['index'].iloc[i]),
                        is_fvg=True,
                        is_bullish=False,
                        strength=strength
                    )
                    bear_fvgs.append(zone)

    # Detect Order Blocks
    if show_ob:
        for i in range(ob_lookback + 1, len(df)):
            atr_val = df['atr'].iloc[i]

            # Bullish OB
            is_bull_break = df['close'].iloc[i] > df['close'].iloc[i-1] and \
                           (df['high'].iloc[i] - df['low'].iloc[i]) > atr_val * 1.2
            prev_bearish = df['close'].iloc[i-1] < df['open'].iloc[i-1]
            strong_up = df['close'].iloc[i] > df['open'].iloc[i] and \
                       calculate_body_percent(
                           df['open'].iloc[i],
                           df['high'].iloc[i],
                           df['low'].iloc[i],
                           df['close'].iloc[i]
                       ) >= min_body_pct

            is_uptrend = df['is_uptrend'].iloc[i]

            if is_bull_break and prev_bearish and strong_up and (not use_trend_filter or is_uptrend):
                top = max(df['open'].iloc[i-1], df['close'].iloc[i-1])
                bottom = df['low'].iloc[i-1]
                size = top - bottom
                strength = get_zone_strength(size, atr_val)

                zone = SMZone(
                    top=top,
                    bottom=bottom,
                    start_idx=int(df['index'].iloc[i-1]),
                    end_idx=int(df['index'].iloc[i]),
                    created_at=int(df['index'].iloc[i]),
                    is_fvg=False,
                    is_bullish=True,
                    strength=strength
                )
                bull_obs.append(zone)

            # Bearish OB
            is_bear_break = df['close'].iloc[i] < df['close'].iloc[i-1] and \
                           (df['high'].iloc[i] - df['low'].iloc[i]) > atr_val * 1.2
            prev_bullish = df['close'].iloc[i-1] > df['open'].iloc[i-1]
            strong_down = df['close'].iloc[i] < df['open'].iloc[i] and \
                         calculate_body_percent(
                             df['open'].iloc[i],
                             df['high'].iloc[i],
                             df['low'].iloc[i],
                             df['close'].iloc[i]
                         ) >= min_body_pct

            is_downtrend = df['is_downtrend'].iloc[i]

            if is_bear_break and prev_bullish and strong_down and (not use_trend_filter or is_downtrend):
                top = df['high'].iloc[i-1]
                bottom = min(df['open'].iloc[i-1], df['close'].iloc[i-1])
                size = top - bottom
                strength = get_zone_strength(size, atr_val)

                zone = SMZone(
                    top=top,
                    bottom=bottom,
                    start_idx=int(df['index'].iloc[i-1]),
                    end_idx=int(df['index'].iloc[i]),
                    created_at=int(df['index'].iloc[i]),
                    is_fvg=False,
                    is_bullish=False,
                    strength=strength
                )
                bear_obs.append(zone)

    # Apply max zones limit (keep most recent)
    bull_fvgs = bull_fvgs[-max_zones:] if len(bull_fvgs) > max_zones else bull_fvgs
    bear_fvgs = bear_fvgs[-max_zones:] if len(bear_fvgs) > max_zones else bear_fvgs
    bull_obs = bull_obs[-max_zones:] if len(bull_obs) > max_zones else bull_obs
    bear_obs = bear_obs[-max_zones:] if len(bear_obs) > max_zones else bear_obs

    # Extend zones forward and check mitigation
    all_zones = bull_fvgs + bear_fvgs + bull_obs + bear_obs
    for zone in all_zones:
        extend_and_mitigate_zone(zone, df)

    return {
        'bull_fvg': bull_fvgs,
        'bear_fvg': bear_fvgs,
        'bull_ob': bull_obs,
        'bear_ob': bear_obs,
        'trend_ma': df['trend_ma'].values
    }


def extend_and_mitigate_zone(zone, df):
    """
    Extend zone forward and check for mitigation

    Mitigation occurs when:
    - Bullish zone: Price revisits zone (low penetrates) with 50%+ penetration or close below bottom
    - Bearish zone: Price revisits zone (high penetrates) with 50%+ penetration or close above top
    """
    zone_size = zone.top - zone.bottom

    # Start checking from bar after zone was created
    start_check = zone.created_at + 1

    for i in range(start_check, len(df)):
        if i >= len(df):
            break

        bar_idx = int(df['index'].iloc[i])

        if zone.is_live:
            if zone.is_bullish:
                # Check if price comes down into zone
                if df['low'].iloc[i] <= zone.top and df['low'].iloc[i] >= zone.bottom:
                    penetration = zone.top - df['low'].iloc[i]
                    zone.mitigated_amount = (penetration / zone_size) * 100.0 if zone_size > 0 else 0

                # Mark as mitigated if 50%+ penetrated or closed below
                if zone.mitigated_amount >= 50.0 or df['close'].iloc[i] <= zone.bottom:
                    zone.is_live = False
                    zone.mitigation_idx = bar_idx
                    zone.end_idx = bar_idx
                    break
                elif df['low'].iloc[i] <= zone.top:
                    # Extend zone
                    zone.end_idx = bar_idx
            else:
                # Bearish zone: check if price comes up into zone
                if df['high'].iloc[i] >= zone.bottom and df['high'].iloc[i] <= zone.top:
                    penetration = df['high'].iloc[i] - zone.bottom
                    zone.mitigated_amount = (penetration / zone_size) * 100.0 if zone_size > 0 else 0

                # Mark as mitigated if 50%+ penetrated or closed above
                if zone.mitigated_amount >= 50.0 or df['close'].iloc[i] >= zone.top:
                    zone.is_live = False
                    zone.mitigation_idx = bar_idx
                    zone.end_idx = bar_idx
                    break
                elif df['high'].iloc[i] >= zone.bottom:
                    # Extend zone
                    zone.end_idx = bar_idx


def calculate_mtf_trends(df, timeframes_minutes, ma_period=50):
    """
    Calculate multi-timeframe trends

    Args:
        df: DataFrame with datetime index and OHLC data
        timeframes_minutes: Dict of timeframe names to minutes
        ma_period: Moving average period for trend

    Returns:
        Dictionary of timeframe trends (True = bullish, False = bearish)
    """
    trends = {}

    for tf_name, minutes in timeframes_minutes.items():
        # Resample to higher timeframe
        df_htf = df.resample(f'{minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        # Calculate trend
        ma = df_htf['close'].rolling(window=ma_period).mean()
        is_bullish = df_htf['close'].iloc[-1] > ma.iloc[-1] if len(ma) > 0 and not pd.isna(ma.iloc[-1]) else True

        trends[tf_name] = is_bullish

    return trends
