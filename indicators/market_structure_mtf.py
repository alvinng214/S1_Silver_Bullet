"""
Market Structure MTF Trend Indicator for Backtrader
Translated from Pine Script: Market Structure MTF Trend [Pt]

This indicator identifies market structure shifts (CHoCH and BoS) based on pivot points
across multiple timeframes.

Key Concepts:
- CHoCH (Change of Character): Trend reversal when price breaks structure in opposite direction
- BoS (Break of Structure): Trend continuation when price breaks structure in same direction
- Uses candle CLOSES (not wicks) for structure breaks
"""

import backtrader as bt
import numpy as np


class MarketStructureIndicator(bt.Indicator):
    """
    Market Structure Indicator based on pivot highs and lows.

    Detects:
    - Bullish trend: Close crosses above last pivot high
    - Bearish trend: Close crosses below last pivot low
    - CHoCH: Change of Character (trend reversal)
    - BoS: Break of Structure (trend continuation)
    """

    lines = (
        'trend',          # Current trend: 1=bullish, -1=bearish, 0=undefined
        'choch',          # Change of Character signal: 1=bullish, -1=bearish, 0=none
        'bos',            # Break of Structure signal: 1=bullish, -1=bearish, 0=none
        'pivot_high',     # Last pivot high level
        'pivot_low',      # Last pivot low level
    )

    params = (
        ('pivot_strength', 15),  # Number of bars for pivot detection
    )

    def __init__(self):
        # State variables
        self.last_pivot_high = None
        self.last_pivot_low = None
        self.last_broken_high = None
        self.last_broken_low = None
        self.current_trend = 0  # 0=undefined, 1=bullish, -1=bearish

        # Store bar indices for pivot detection
        self.bar_count = 0

    def next(self):
        self.bar_count += 1

        # Initialize outputs
        self.lines.trend[0] = self.current_trend
        self.lines.choch[0] = 0
        self.lines.bos[0] = 0
        self.lines.pivot_high[0] = self.last_pivot_high if self.last_pivot_high else float('nan')
        self.lines.pivot_low[0] = self.last_pivot_low if self.last_pivot_low else float('nan')

        # Need enough bars for pivot detection
        if len(self) < self.params.pivot_strength * 2 + 1:
            return

        # Detect pivot high: current bar is higher than pivot_strength bars on each side
        pivot_high = self._detect_pivot_high()
        if pivot_high is not None:
            if self.current_trend == 1:  # Bullish trend
                # In bullish trend, track the highest pivot high
                if self.last_pivot_high is None:
                    self.last_pivot_high = pivot_high
                else:
                    self.last_pivot_high = max(pivot_high, self.last_pivot_high)
            else:
                # New pivot high regardless of trend
                self.last_pivot_high = pivot_high

        # Detect pivot low: current bar is lower than pivot_strength bars on each side
        pivot_low = self._detect_pivot_low()
        if pivot_low is not None:
            if self.current_trend == -1:  # Bearish trend
                # In bearish trend, track the lowest pivot low
                if self.last_pivot_low is None:
                    self.last_pivot_low = pivot_low
                else:
                    self.last_pivot_low = min(pivot_low, self.last_pivot_low)
            else:
                # New pivot low regardless of trend
                self.last_pivot_low = pivot_low

        # Update pivot level outputs
        if self.last_pivot_high is not None:
            self.lines.pivot_high[0] = self.last_pivot_high
        if self.last_pivot_low is not None:
            self.lines.pivot_low[0] = self.last_pivot_low

        # Check for structure breaks using CLOSE price
        close_price = self.data.close[0]

        # Bullish structure break: close crosses above last pivot high
        if self.last_pivot_high is not None and close_price > self.last_pivot_high:
            if self.data.close[-1] <= self.last_pivot_high:  # Confirmed crossover
                if self.current_trend == 1 and self.last_pivot_high != self.last_broken_high:
                    # BoS: Break in same direction (continuation)
                    self.lines.bos[0] = 1
                    self.last_broken_high = self.last_pivot_high
                elif self.current_trend != 1:
                    # CHoCH: Change of Character (reversal to bullish)
                    self.lines.choch[0] = 1
                    self.last_broken_high = self.last_pivot_high
                    self.last_broken_low = None

                self.current_trend = 1

        # Bearish structure break: close crosses below last pivot low
        if self.last_pivot_low is not None and close_price < self.last_pivot_low:
            if self.data.close[-1] >= self.last_pivot_low:  # Confirmed crossunder
                if self.current_trend == -1 and self.last_pivot_low != self.last_broken_low:
                    # BoS: Break in same direction (continuation)
                    self.lines.bos[0] = -1
                    self.last_broken_low = self.last_pivot_low
                elif self.current_trend != -1:
                    # CHoCH: Change of Character (reversal to bearish)
                    self.lines.choch[0] = -1
                    self.last_broken_low = self.last_pivot_low
                    self.last_broken_high = None

                self.current_trend = -1

        # Update trend output
        self.lines.trend[0] = self.current_trend

    def _detect_pivot_high(self):
        """
        Detect a pivot high at position [pivot_strength] bars ago.
        A pivot high is confirmed when the high at that position is higher than
        pivot_strength bars before and after it.
        """
        pivot_idx = self.params.pivot_strength

        # Get the high at pivot position
        pivot_high = self.data.high[-pivot_idx]

        # Check if it's higher than all bars before it (within pivot_strength range)
        for i in range(1, self.params.pivot_strength + 1):
            if self.data.high[-(pivot_idx - i)] >= pivot_high:
                return None

        # Check if it's higher than all bars after it (within pivot_strength range)
        for i in range(1, self.params.pivot_strength + 1):
            if self.data.high[-(pivot_idx + i)] >= pivot_high:
                return None

        return pivot_high

    def _detect_pivot_low(self):
        """
        Detect a pivot low at position [pivot_strength] bars ago.
        A pivot low is confirmed when the low at that position is lower than
        pivot_strength bars before and after it.
        """
        pivot_idx = self.params.pivot_strength

        # Get the low at pivot position
        pivot_low = self.data.low[-pivot_idx]

        # Check if it's lower than all bars before it (within pivot_strength range)
        for i in range(1, self.params.pivot_strength + 1):
            if self.data.low[-(pivot_idx - i)] <= pivot_low:
                return None

        # Check if it's lower than all bars after it (within pivot_strength range)
        for i in range(1, self.params.pivot_strength + 1):
            if self.data.low[-(pivot_idx + i)] <= pivot_low:
                return None

        return pivot_low


class MarketStructureMTF(bt.Indicator):
    """
    Multi-Timeframe Market Structure Indicator.

    Tracks market structure across multiple timeframes (similar to the original Pine Script).
    Each timeframe has its own trend, CHoCH, and BoS signals.
    """

    lines = (
        'trend_tf1', 'choch_tf1', 'bos_tf1',
        'trend_tf2', 'choch_tf2', 'bos_tf2',
        'trend_tf3', 'choch_tf3', 'bos_tf3',
        'trend_tf4', 'choch_tf4', 'bos_tf4',
    )

    params = (
        ('pivot_strength_tf1', 15),
        ('pivot_strength_tf2', 15),
        ('pivot_strength_tf3', 15),
        ('pivot_strength_tf4', 15),
    )

    def __init__(self):
        # For multi-timeframe support in backtrader, you would typically
        # resample data or use multiple data feeds
        # For now, we'll use the same data with different pivot strengths
        # as a simplified version

        # Create market structure indicators for different "timeframes"
        # Note: In a real implementation, you'd pass resampled data feeds
        self.ms_tf1 = MarketStructureIndicator(
            self.data,
            pivot_strength=self.params.pivot_strength_tf1
        )

        # Map outputs
        self.lines.trend_tf1 = self.ms_tf1.lines.trend
        self.lines.choch_tf1 = self.ms_tf1.lines.choch
        self.lines.bos_tf1 = self.ms_tf1.lines.bos

        # For TF2, TF3, TF4, you would add resampled data feeds
        # This is a simplified version showing the structure
