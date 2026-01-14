"""
Silver Bullet ICT Strategy [TradingFinder] 10-11 AM NY Time +FVG
TFlab Silver Bullet - Python Translation

Strategy:
1. Opening Range: 9:00-10:00 AM NY time - track high and low
2. Trading Time: 10:00-11:00 AM NY time
3. Break Detection: High or low break during trading time
4. CISD (Change in State of Delivery): Level from recent opposite candle
5. Order Blocks: Demand/Supply zones
6. FVGs: Fair Value Gaps for entry confirmation

License: MPL 2.0
Original: TFlab
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timedelta


@dataclass
class Session:
    """Opening range session data"""
    start_idx: int
    end_idx: int
    high: float
    low: float
    start_time: datetime


@dataclass
class OrderBlock:
    """Order Block (Demand/Supply zone)"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_demand: bool  # True for demand, False for supply
    mitigated: bool = False
    created_at: int = 0


@dataclass
class FVG:
    """Fair Value Gap"""
    start_idx: int
    end_idx: int
    top: float
    bottom: float
    is_bullish: bool
    mitigated: bool = False


@dataclass
class CISDLevel:
    """Change in State of Delivery Level"""
    price: float
    index: int
    is_bearish: bool  # True for bearish CISD (resistance), False for bullish (support)
    triggered: bool = False
    trigger_idx: Optional[int] = None


@dataclass
class SilverBulletSignal:
    """Complete Silver Bullet setup"""
    session: Session
    high_break: bool
    low_break: bool
    cisd_level: Optional[CISDLevel]
    order_block: Optional[OrderBlock]
    fvg: Optional[FVG]
    signal_type: str  # 'BULL' or 'BEAR'


def is_in_ny_opening_range(dt: datetime) -> bool:
    """Check if time is in NY opening range (9-10 AM)"""
    return dt.hour == 9


def is_in_ny_trading_time(dt: datetime) -> bool:
    """Check if time is in NY trading time (10-11 AM)"""
    return dt.hour == 10


def detect_fvg_simple(df: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[List[FVG], List[FVG]]:
    """
    Detect Fair Value Gaps in a range

    Returns: (bullish_fvgs, bearish_fvgs)
    """
    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(start_idx + 2, end_idx):
        if i >= len(df):
            break

        # Bullish FVG: current low > high[2]
        if df.iloc[i]['low'] > df.iloc[i-2]['high']:
            fvg = FVG(
                start_idx=i-2,
                end_idx=i,
                top=df.iloc[i]['low'],
                bottom=df.iloc[i-2]['high'],
                is_bullish=True,
                mitigated=False
            )
            bullish_fvgs.append(fvg)

        # Bearish FVG: current high < low[2]
        if df.iloc[i]['high'] < df.iloc[i-2]['low']:
            fvg = FVG(
                start_idx=i-2,
                end_idx=i,
                top=df.iloc[i-2]['low'],
                bottom=df.iloc[i]['high'],
                is_bullish=False,
                mitigated=False
            )
            bearish_fvgs.append(fvg)

    return bullish_fvgs, bearish_fvgs


def detect_cisd_level(df: pd.DataFrame, break_idx: int, is_high_break: bool, bar_back_check: int = 5) -> Optional[CISDLevel]:
    """
    Detect CISD (Change in State of Delivery) level

    For high break: Look back for bearish candle's open
    For low break: Look back for bullish candle's open
    """
    if break_idx < bar_back_check:
        return None

    for i in range(1, bar_back_check + 1):
        idx = break_idx - i
        if idx < 0:
            continue

        body = df.iloc[idx]['close'] - df.iloc[idx]['open']

        # High break → look for bearish candle (body < 0)
        if is_high_break and body < 0:
            # Use open of this or previous bearish candle (minimum)
            if i > 1 and idx > 0:
                body_prev = df.iloc[idx-1]['close'] - df.iloc[idx-1]['open']
                if body_prev < 0:
                    price = min(df.iloc[idx]['open'], df.iloc[idx-1]['open'])
                    source_idx = idx if df.iloc[idx]['open'] <= df.iloc[idx-1]['open'] else idx - 1
                else:
                    price = df.iloc[idx]['open']
                    source_idx = idx
            else:
                price = df.iloc[idx]['open']
                source_idx = idx

            return CISDLevel(
                price=price,
                index=source_idx,
                is_bearish=True,
                triggered=False
            )

        # Low break → look for bullish candle (body > 0)
        if not is_high_break and body > 0:
            # Use open of this or previous bullish candle (maximum)
            if i > 1 and idx > 0:
                body_prev = df.iloc[idx-1]['close'] - df.iloc[idx-1]['open']
                if body_prev > 0:
                    price = max(df.iloc[idx]['open'], df.iloc[idx-1]['open'])
                    source_idx = idx if df.iloc[idx]['open'] >= df.iloc[idx-1]['open'] else idx - 1
                else:
                    price = df.iloc[idx]['open']
                    source_idx = idx
            else:
                price = df.iloc[idx]['open']
                source_idx = idx

            return CISDLevel(
                price=price,
                index=source_idx,
                is_bearish=False,
                triggered=False
            )

    return None


def detect_order_block_simple(df: pd.DataFrame, cisd_level: CISDLevel, trading_start: int) -> Optional[OrderBlock]:
    """
    Simplified order block detection around CISD level

    For bearish: Use high of the candle at/near CISD
    For bullish: Use low of the candle at/near CISD
    """
    if cisd_level is None:
        return None

    idx = cisd_level.index
    if idx >= len(df):
        return None

    # Bearish Order Block (Supply)
    if cisd_level.is_bearish:
        # Find the high and create supply zone
        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']

        return OrderBlock(
            start_idx=idx,
            end_idx=idx + 60,  # Extend for 60 bars
            top=high,
            bottom=cisd_level.price,  # CISD price to high
            is_demand=False,
            mitigated=False,
            created_at=trading_start
        )

    # Bullish Order Block (Demand)
    else:
        # Find the low and create demand zone
        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']

        return OrderBlock(
            start_idx=idx,
            end_idx=idx + 60,  # Extend for 60 bars
            top=cisd_level.price,  # CISD price to low
            bottom=low,
            is_demand=True,
            mitigated=False,
            created_at=trading_start
        )


def detect_tradingfinder_silver_bullet(
    df: pd.DataFrame,
    bar_back_check: int = 5,
    show_order_blocks: bool = True,
    show_fvgs: bool = True
) -> dict:
    """
    Detect TradingFinder Silver Bullet setups for 10-11 AM NY Time

    Args:
        df: DataFrame with OHLC data and datetime index
        bar_back_check: Bars to look back for CISD level
        show_order_blocks: Include order blocks in results
        show_fvgs: Include FVGs in results

    Returns: Dictionary with sessions, signals, order blocks, FVGs, and CISD levels
    """
    print(f"Detecting TradingFinder Silver Bullet (10-11 AM NY Time)")

    sessions = []
    signals = []
    all_order_blocks = []
    all_fvgs = []
    all_cisd_levels = []

    # Track current session
    current_session = None
    session_start_idx = None
    session_high = -float('inf')
    session_low = float('inf')

    # Track trading time
    in_trading_time = False
    trading_start_idx = None
    high_break = False
    low_break = False
    high_break_idx = None
    low_break_idx = None

    for i in range(len(df)):
        dt = df.index[i]
        row = df.iloc[i]

        in_opening = is_in_ny_opening_range(dt)
        in_trading = is_in_ny_trading_time(dt)

        # === OPENING RANGE (9-10 AM) ===
        if in_opening:
            if current_session is None:
                # Start new session
                current_session = Session(
                    start_idx=i,
                    end_idx=i,
                    high=row['high'],
                    low=row['low'],
                    start_time=dt
                )
                session_start_idx = i
                session_high = row['high']
                session_low = row['low']
            else:
                # Update session high/low
                session_high = max(session_high, row['high'])
                session_low = min(session_low, row['low'])
                current_session.end_idx = i
                current_session.high = session_high
                current_session.low = session_low

        # === TRADING TIME (10-11 AM) ===
        if in_trading and current_session is not None:
            if not in_trading_time:
                # Start trading time
                in_trading_time = True
                trading_start_idx = i
                high_break = False
                low_break = False
                high_break_idx = None
                low_break_idx = None

            # Detect high break
            if not high_break and row['high'] > current_session.high:
                high_break = True
                high_break_idx = i

            # Detect low break
            if not low_break and row['low'] < current_session.low:
                low_break = True
                low_break_idx = i

        # === END OF TRADING TIME ===
        if not in_trading and not in_opening and in_trading_time:
            # Trading time ended, process the setup
            in_trading_time = False

            if current_session is not None:
                sessions.append(current_session)

                # Determine signal type
                cisd_level = None
                order_block = None
                fvg_list = []

                if high_break and not low_break:
                    # Bearish setup: High broke, look for sell
                    cisd_level = detect_cisd_level(df, high_break_idx, is_high_break=True, bar_back_check=bar_back_check)

                    if cisd_level:
                        all_cisd_levels.append(cisd_level)

                        # Detect order block
                        if show_order_blocks:
                            order_block = detect_order_block_simple(df, cisd_level, trading_start_idx)
                            if order_block:
                                all_order_blocks.append(order_block)

                        # Detect FVGs during trading time
                        if show_fvgs and trading_start_idx:
                            _, bearish_fvgs = detect_fvg_simple(df, trading_start_idx, i)
                            fvg_list = bearish_fvgs
                            all_fvgs.extend(bearish_fvgs)

                    signal = SilverBulletSignal(
                        session=current_session,
                        high_break=True,
                        low_break=False,
                        cisd_level=cisd_level,
                        order_block=order_block,
                        fvg=fvg_list[0] if len(fvg_list) > 0 else None,
                        signal_type='BEAR'
                    )
                    signals.append(signal)

                elif low_break and not high_break:
                    # Bullish setup: Low broke, look for buy
                    cisd_level = detect_cisd_level(df, low_break_idx, is_high_break=False, bar_back_check=bar_back_check)

                    if cisd_level:
                        all_cisd_levels.append(cisd_level)

                        # Detect order block
                        if show_order_blocks:
                            order_block = detect_order_block_simple(df, cisd_level, trading_start_idx)
                            if order_block:
                                all_order_blocks.append(order_block)

                        # Detect FVGs during trading time
                        if show_fvgs and trading_start_idx:
                            bullish_fvgs, _ = detect_fvg_simple(df, trading_start_idx, i)
                            fvg_list = bullish_fvgs
                            all_fvgs.extend(bullish_fvgs)

                    signal = SilverBulletSignal(
                        session=current_session,
                        high_break=False,
                        low_break=True,
                        cisd_level=cisd_level,
                        order_block=order_block,
                        fvg=fvg_list[0] if len(fvg_list) > 0 else None,
                        signal_type='BULL'
                    )
                    signals.append(signal)

            current_session = None

        # Check CISD level triggers
        for cisd in all_cisd_levels:
            if not cisd.triggered:
                # Bearish CISD: triggered when close <= level
                if cisd.is_bearish and row['close'] <= cisd.price:
                    cisd.triggered = True
                    cisd.trigger_idx = i
                # Bullish CISD: triggered when close >= level
                elif not cisd.is_bearish and row['close'] >= cisd.price:
                    cisd.triggered = True
                    cisd.trigger_idx = i

        # Check order block mitigation
        for ob in all_order_blocks:
            if not ob.mitigated and i > ob.created_at:
                # Demand OB: mitigated if price goes below bottom
                if ob.is_demand and row['low'] < ob.bottom:
                    ob.mitigated = True
                # Supply OB: mitigated if price goes above top
                elif not ob.is_demand and row['high'] > ob.top:
                    ob.mitigated = True

        # Check FVG mitigation
        for fvg in all_fvgs:
            if not fvg.mitigated and i > fvg.end_idx:
                # Bullish FVG: mitigated if price goes below bottom
                if fvg.is_bullish and row['low'] < fvg.bottom:
                    fvg.mitigated = True
                # Bearish FVG: mitigated if price goes above top
                elif not fvg.is_bullish and row['high'] > fvg.top:
                    fvg.mitigated = True

    print(f"Detected {len(sessions)} opening range sessions")
    print(f"Generated {len(signals)} Silver Bullet signals")
    print(f"  - Bullish: {len([s for s in signals if s.signal_type == 'BULL'])}")
    print(f"  - Bearish: {len([s for s in signals if s.signal_type == 'BEAR'])}")
    print(f"Order Blocks: {len(all_order_blocks)}")
    print(f"FVGs: {len(all_fvgs)}")
    print(f"CISD Levels: {len(all_cisd_levels)}")

    return {
        'sessions': sessions,
        'signals': signals,
        'order_blocks': all_order_blocks,
        'fvgs': all_fvgs,
        'cisd_levels': all_cisd_levels
    }


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv("PEPPERSTONE_XAUUSD, 5.csv")
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    results = detect_tradingfinder_silver_bullet(
        df,
        bar_back_check=5,
        show_order_blocks=True,
        show_fvgs=True
    )

    print(f"\nResults Summary:")
    print(f"  Sessions: {len(results['sessions'])}")
    print(f"  Signals: {len(results['signals'])}")
    print(f"  Order Blocks: {len(results['order_blocks'])}")
    print(f"  FVGs: {len(results['fvgs'])}")
    print(f"  CISD Levels: {len(results['cisd_levels'])}")
