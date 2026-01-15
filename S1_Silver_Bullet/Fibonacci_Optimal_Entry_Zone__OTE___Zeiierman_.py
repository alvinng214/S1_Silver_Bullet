# -*- coding: utf-8 -*-
"""
Fibonacci Optimal Entry Zone [OTE] (Zeiierman)
Python translation from Pine Script

This indicator identifies market structure (CHoCH - Change of Character) and draws
Fibonacci retracement levels (0.50 and 0.618) to find optimal entry zones.

Original Pine Script by Zeiierman
Licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import mplfinance as mpf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FibonacciOTE:
    """
    Fibonacci Optimal Entry Zone indicator.

    Identifies swing highs/lows and market structure changes (CHoCH),
    then draws Fibonacci retracement levels for optimal trade entries.
    """

    def __init__(self,
                 structure_period: int = 10,
                 show_bullish: bool = True,
                 show_bearish: bool = True,
                 bullish_color: str = '#08ec32',
                 bearish_color: str = '#ff2222',
                 swing_tracker: bool = True,
                 show_swing_line: bool = True,
                 show_swing_labels: bool = True,
                 show_previous: bool = False,
                 extend_fibs: bool = True,
                 fill_golden_zone: bool = True,
                 golden_zone_color: str = '#009688',
                 fib_levels: list = None,
                 fib_colors: list = None):
        """
        Initialize the Fibonacci OTE indicator.

        Parameters:
        -----------
        structure_period : int
            Number of bars for pivot high/low calculation (default: 10)
        show_bullish : bool
            Show bullish market structure (default: True)
        show_bearish : bool
            Show bearish market structure (default: True)
        bullish_color : str
            Color for bullish structure lines and labels
        bearish_color : str
            Color for bearish structure lines and labels
        swing_tracker : bool
            Enable automatic tracking of recent swing points
        show_swing_line : bool
            Draw dotted lines connecting swing highs/lows
        show_swing_labels : bool
            Show price labels at swing highs/lows
        show_previous : bool
            Show previous Fibonacci levels
        extend_fibs : bool
            Extend Fibonacci levels to current bar
        fill_golden_zone : bool
            Fill the golden zone (0.5 - 0.618)
        golden_zone_color : str
            Color for the golden zone fill
        fib_levels : list
            Fibonacci retracement levels (default: [0.50, 0.618])
        fib_colors : list
            Colors for each Fibonacci level
        """
        self.structure_period = structure_period
        self.show_bullish = show_bullish
        self.show_bearish = show_bearish
        self.bullish_color = bullish_color
        self.bearish_color = bearish_color
        self.swing_tracker = swing_tracker
        self.show_swing_line = show_swing_line
        self.show_swing_labels = show_swing_labels
        self.show_previous = show_previous
        self.extend_fibs = extend_fibs
        self.fill_golden_zone = fill_golden_zone
        self.golden_zone_color = golden_zone_color
        self.fib_levels = fib_levels if fib_levels else [0.50, 0.618]
        self.fib_colors = fib_colors if fib_colors else ['#4CAF50', '#009688']

        # Internal state
        self.structures = []
        self.fib_zones = []

    def calculate_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pivot highs and lows.

        Parameters:
        -----------
        df : pd.DataFrame
            OHLC dataframe with 'high' and 'low' columns

        Returns:
        --------
        pd.DataFrame
            DataFrame with added pivot columns
        """
        df = df.copy()
        prd = self.structure_period

        # Calculate pivot high - highest point in window with equal values on both sides
        def is_pivot_high(window):
            if len(window) < 2 * prd + 1:
                return np.nan
            center = window[prd]
            # Check if center is the maximum
            if np.all(center >= window):
                return center
            return np.nan

        # Calculate pivot low - lowest point in window
        def is_pivot_low(window):
            if len(window) < 2 * prd + 1:
                return np.nan
            center = window[prd]
            # Check if center is the minimum
            if np.all(center <= window):
                return center
            return np.nan

        df['pivot_high'] = df['high'].rolling(window=2*prd+1, center=True).apply(
            is_pivot_high, raw=True
        )
        df['pivot_low'] = df['low'].rolling(window=2*prd+1, center=True).apply(
            is_pivot_low, raw=True
        )

        return df

    def calculate_fibonacci(self, high: float, low: float,
                           high_idx: int, low_idx: int, level: float) -> float:
        """
        Calculate Fibonacci retracement level.

        Parameters:
        -----------
        high : float
            Swing high price
        low : float
            Swing low price
        high_idx : int
            Index of swing high
        low_idx : int
            Index of swing low
        level : float
            Fibonacci level (e.g., 0.5, 0.618)

        Returns:
        --------
        float
            Price at the Fibonacci level
        """
        if low_idx < high_idx:
            # Bullish - retracement from high to low
            return high - (high - low) * level
        else:
            # Bearish - retracement from low to high
            return low + (high - low) * level

    def detect_structure(self, df: pd.DataFrame) -> dict:
        """
        Detect market structure and calculate Fibonacci zones.

        Parameters:
        -----------
        df : pd.DataFrame
            OHLC dataframe with pivot columns

        Returns:
        --------
        dict
            Dictionary containing structure data and Fibonacci zones
        """
        df = self.calculate_pivots(df)

        # Initialize tracking variables
        pos = 0  # Position: positive = bullish, negative = bearish
        up = df['high'].iloc[0] if len(df) > 0 else np.nan  # Current swing high
        dn = df['low'].iloc[0] if len(df) > 0 else np.nan   # Current swing low
        i_up = 0  # Index of current swing high
        i_dn = 0  # Index of current swing low

        swing_low = np.nan
        swing_high = np.nan
        i_swing_low = 0
        i_swing_high = 0

        structures = []
        fib_zones = []
        current_fib_zone = None

        for i in range(len(df)):
            bar = df.iloc[i]

            # Update running highs/lows
            up = max(up, bar['high']) if not np.isnan(up) else bar['high']
            dn = min(dn, bar['low']) if not np.isnan(dn) else bar['low']

            # Check for pivot high
            if not pd.isna(bar['pivot_high']):
                if pos <= 0:
                    up = bar['pivot_high']

            # Check for pivot low
            if not pd.isna(bar['pivot_low']):
                if pos >= 0:
                    dn = bar['pivot_low']

            # Bullish structure break (CHoCH to bullish)
            if i > 0 and up > df['high'].iloc[max(0, i-1):i+1].max() - (df['high'].iloc[i] if i > 0 else 0) + up:
                prev_up = df['high'].iloc[:i].max() if i > 0 else up
                if up > prev_up:
                    i_up = i

                    if pos <= 0:
                        # CHoCH - Change of Character to bullish
                        structures.append({
                            'type': 'CHoCH',
                            'direction': 'bullish',
                            'idx': i,
                            'price': up,
                            'start_idx': i_dn if i_dn > 0 else 0
                        })

                        # Create new Fibonacci zone
                        if self.show_bullish:
                            fib_zone = {
                                'direction': 'bullish',
                                'high': up,
                                'low': dn,
                                'high_idx': i,
                                'low_idx': i_dn if i_dn > 0 else 0,
                                'start_idx': i_dn if i_dn > 0 else 0,
                                'end_idx': i,
                                'levels': {}
                            }
                            for j, level in enumerate(self.fib_levels):
                                fib_val = self.calculate_fibonacci(up, dn, i, i_dn, level)
                                fib_zone['levels'][level] = {
                                    'price': fib_val,
                                    'color': self.fib_colors[j % len(self.fib_colors)]
                                }
                            current_fib_zone = fib_zone
                            fib_zones.append(fib_zone)

                        pos = 1
                        swing_low = dn
                        i_swing_low = i_dn

                    elif pos >= 1:
                        # Update existing bullish structure
                        if current_fib_zone and self.show_bullish:
                            if self.swing_tracker:
                                current_fib_zone['high'] = up
                                current_fib_zone['high_idx'] = i
                                current_fib_zone['end_idx'] = i
                                for j, level in enumerate(self.fib_levels):
                                    fib_val = self.calculate_fibonacci(up, dn, i, i_dn, level)
                                    current_fib_zone['levels'][level]['price'] = fib_val
                            else:
                                current_fib_zone['high'] = up
                                current_fib_zone['high_idx'] = i
                                for j, level in enumerate(self.fib_levels):
                                    fib_val = self.calculate_fibonacci(up, swing_low, i, i_swing_low, level)
                                    current_fib_zone['levels'][level]['price'] = fib_val
                        pos += 1

            # Bearish structure break (CHoCH to bearish)
            if i > 0:
                prev_dn = df['low'].iloc[:i].min() if i > 0 else dn
                if dn < prev_dn:
                    i_dn = i

                    if pos >= 0:
                        # CHoCH - Change of Character to bearish
                        structures.append({
                            'type': 'CHoCH',
                            'direction': 'bearish',
                            'idx': i,
                            'price': dn,
                            'start_idx': i_up if i_up > 0 else 0
                        })

                        # Create new Fibonacci zone
                        if self.show_bearish:
                            fib_zone = {
                                'direction': 'bearish',
                                'high': up,
                                'low': dn,
                                'high_idx': i_up if i_up > 0 else 0,
                                'low_idx': i,
                                'start_idx': i_up if i_up > 0 else 0,
                                'end_idx': i,
                                'levels': {}
                            }
                            for j, level in enumerate(self.fib_levels):
                                fib_val = self.calculate_fibonacci(up, dn, i_up, i, level)
                                fib_zone['levels'][level] = {
                                    'price': fib_val,
                                    'color': self.fib_colors[j % len(self.fib_colors)]
                                }
                            current_fib_zone = fib_zone
                            fib_zones.append(fib_zone)

                        pos = -1
                        swing_high = up
                        i_swing_high = i_up

                    elif pos <= -1:
                        # Update existing bearish structure
                        if current_fib_zone and self.show_bearish:
                            if self.swing_tracker:
                                current_fib_zone['low'] = dn
                                current_fib_zone['low_idx'] = i
                                current_fib_zone['end_idx'] = i
                                for j, level in enumerate(self.fib_levels):
                                    fib_val = self.calculate_fibonacci(up, dn, i_up, i, level)
                                    current_fib_zone['levels'][level]['price'] = fib_val
                            else:
                                current_fib_zone['low'] = dn
                                current_fib_zone['low_idx'] = i
                                for j, level in enumerate(self.fib_levels):
                                    fib_val = self.calculate_fibonacci(swing_high, dn, i_swing_high, i, level)
                                    current_fib_zone['levels'][level]['price'] = fib_val
                        pos -= 1

        self.structures = structures
        self.fib_zones = fib_zones

        return {
            'structures': structures,
            'fib_zones': fib_zones,
            'df': df
        }

    def plot(self, df: pd.DataFrame,
             last_n_bars: int = None,
             figsize: tuple = (16, 10),
             title: str = 'Fibonacci Optimal Entry Zone [OTE]',
             save_path: str = None,
             show_plot: bool = True) -> plt.Figure:
        """
        Plot candlestick chart with Fibonacci OTE indicator.

        Parameters:
        -----------
        df : pd.DataFrame
            OHLC dataframe
        last_n_bars : int
            Number of bars to display (None = all)
        figsize : tuple
            Figure size
        title : str
            Chart title
        save_path : str
            Path to save the chart (None = don't save)
        show_plot : bool
            Whether to display the plot

        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        # Detect structure and calculate Fibonacci zones
        result = self.detect_structure(df)
        df_with_pivots = result['df']

        # Slice data if needed
        if last_n_bars:
            df_plot = df_with_pivots.tail(last_n_bars).copy()
            start_idx = len(df_with_pivots) - last_n_bars
        else:
            df_plot = df_with_pivots.copy()
            start_idx = 0

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#131722')
        ax.set_facecolor('#131722')

        # Plot candlesticks
        for i in range(len(df_plot)):
            idx = i
            bar = df_plot.iloc[i]

            # Determine candle color
            if bar['close'] >= bar['open']:
                color = '#26a69a'  # Bullish - green
                body_color = '#26a69a'
            else:
                color = '#ef5350'  # Bearish - red
                body_color = '#ef5350'

            # Draw wick
            ax.plot([idx, idx], [bar['low'], bar['high']], color=color, linewidth=1)

            # Draw body
            body_bottom = min(bar['open'], bar['close'])
            body_height = abs(bar['close'] - bar['open'])
            if body_height < 0.01:
                body_height = 0.01
            rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                            facecolor=body_color, edgecolor=color, linewidth=1)
            ax.add_patch(rect)

        # Plot Fibonacci zones
        fib_zones_to_plot = self.fib_zones if self.show_previous else self.fib_zones[-1:] if self.fib_zones else []

        for zone in fib_zones_to_plot:
            zone_start = zone['start_idx'] - start_idx
            zone_end = zone['end_idx'] - start_idx

            if zone_end < 0:
                continue
            zone_start = max(0, zone_start)

            # Extend to current bar if enabled
            if self.extend_fibs:
                zone_end = len(df_plot) - 1

            # Draw swing line
            if self.show_swing_line:
                if zone['direction'] == 'bullish':
                    line_color = self.bullish_color
                    ax.plot([zone_start, zone_end],
                           [zone['low'], zone['high']],
                           color=line_color, linestyle=':', linewidth=2, alpha=0.7)
                else:
                    line_color = self.bearish_color
                    ax.plot([zone_start, zone_end],
                           [zone['high'], zone['low']],
                           color=line_color, linestyle=':', linewidth=2, alpha=0.7)

            # Draw Fibonacci levels
            level_prices = []
            for level, level_data in zone['levels'].items():
                price = level_data['price']
                color = level_data['color']
                level_prices.append(price)

                ax.hlines(y=price, xmin=zone_start, xmax=zone_end,
                         colors=color, linewidth=1.5, alpha=0.8)

                # Add level label
                ax.text(zone_end + 0.5, price, f'{level:.3f} ({price:.2f})',
                       color=color, fontsize=8, verticalalignment='center')

            # Fill golden zone
            if self.fill_golden_zone and len(level_prices) >= 2:
                ax.fill_between([zone_start, zone_end],
                               min(level_prices), max(level_prices),
                               color=self.golden_zone_color, alpha=0.2)

            # Draw swing labels
            if self.show_swing_labels:
                if zone['direction'] == 'bullish':
                    ax.annotate(f"{zone['low']:.2f}",
                               xy=(zone_start, zone['low']),
                               xytext=(zone_start, zone['low'] - (zone['high']-zone['low'])*0.05),
                               color='white', fontsize=8, ha='center')
                    ax.annotate(f"{zone['high']:.2f}",
                               xy=(zone_end, zone['high']),
                               xytext=(zone_end, zone['high'] + (zone['high']-zone['low'])*0.05),
                               color='white', fontsize=8, ha='center')
                else:
                    ax.annotate(f"{zone['high']:.2f}",
                               xy=(zone_start, zone['high']),
                               xytext=(zone_start, zone['high'] + (zone['high']-zone['low'])*0.05),
                               color='white', fontsize=8, ha='center')
                    ax.annotate(f"{zone['low']:.2f}",
                               xy=(zone_end, zone['low']),
                               xytext=(zone_end, zone['low'] - (zone['high']-zone['low'])*0.05),
                               color='white', fontsize=8, ha='center')

        # Plot CHoCH labels
        for structure in self.structures:
            struct_idx = structure['idx'] - start_idx
            if 0 <= struct_idx < len(df_plot):
                if structure['direction'] == 'bullish' and self.show_bullish:
                    ax.annotate('CHoCH', xy=(struct_idx, structure['price']),
                               xytext=(struct_idx, structure['price'] + (df_plot['high'].max()-df_plot['low'].min())*0.03),
                               color=self.bullish_color, fontsize=9, ha='center', fontweight='bold')
                elif structure['direction'] == 'bearish' and self.show_bearish:
                    ax.annotate('CHoCH', xy=(struct_idx, structure['price']),
                               xytext=(struct_idx, structure['price'] - (df_plot['high'].max()-df_plot['low'].min())*0.03),
                               color=self.bearish_color, fontsize=9, ha='center', fontweight='bold')

        # Styling
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Bar Index', color='white')
        ax.set_ylabel('Price', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.grid(True, alpha=0.2, color='gray')

        # Add legend
        legend_elements = []
        if self.show_bullish:
            legend_elements.append(mpatches.Patch(color=self.bullish_color, label='Bullish CHoCH'))
        if self.show_bearish:
            legend_elements.append(mpatches.Patch(color=self.bearish_color, label='Bearish CHoCH'))
        for i, level in enumerate(self.fib_levels):
            legend_elements.append(mpatches.Patch(
                color=self.fib_colors[i % len(self.fib_colors)],
                label=f'Fib {level}'
            ))
        if self.fill_golden_zone:
            legend_elements.append(mpatches.Patch(
                color=self.golden_zone_color, alpha=0.3,
                label='Golden Zone (OTE)'
            ))

        ax.legend(handles=legend_elements, loc='upper left',
                 facecolor='#1e222d', edgecolor='white', labelcolor='white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#131722', edgecolor='none')
            print(f"Chart saved to: {save_path}")

        if show_plot:
            plt.show()

        return fig

    def add_to_existing_chart(self, ax: plt.Axes, df: pd.DataFrame,
                              start_idx: int = 0, end_idx: int = None):
        """
        Add Fibonacci OTE indicator to an existing matplotlib axes.

        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axes to add indicator to
        df : pd.DataFrame
            OHLC dataframe
        start_idx : int
            Starting index for plotting
        end_idx : int
            Ending index for plotting
        """
        # Detect structure
        result = self.detect_structure(df)

        if end_idx is None:
            end_idx = len(df)

        # Plot Fibonacci zones
        fib_zones_to_plot = self.fib_zones if self.show_previous else self.fib_zones[-3:] if self.fib_zones else []

        for zone in fib_zones_to_plot:
            zone_start = zone['start_idx'] - start_idx
            zone_end = zone['end_idx'] - start_idx

            if zone_end < 0 or zone_start > (end_idx - start_idx):
                continue
            zone_start = max(0, zone_start)

            # Extend to current bar if enabled
            if self.extend_fibs:
                zone_end = end_idx - start_idx - 1

            # Draw swing line
            if self.show_swing_line:
                if zone['direction'] == 'bullish':
                    line_color = self.bullish_color
                    ax.plot([zone_start, zone_end],
                           [zone['low'], zone['high']],
                           color=line_color, linestyle=':', linewidth=2, alpha=0.7)
                else:
                    line_color = self.bearish_color
                    ax.plot([zone_start, zone_end],
                           [zone['high'], zone['low']],
                           color=line_color, linestyle=':', linewidth=2, alpha=0.7)

            # Draw Fibonacci levels
            level_prices = []
            for level, level_data in zone['levels'].items():
                price = level_data['price']
                color = level_data['color']
                level_prices.append(price)

                ax.hlines(y=price, xmin=zone_start, xmax=zone_end,
                         colors=color, linewidth=1.5, alpha=0.8)

                # Add level label
                ax.text(zone_end + 0.5, price, f'Fib {level:.3f}',
                       color=color, fontsize=7, verticalalignment='center')

            # Fill golden zone
            if self.fill_golden_zone and len(level_prices) >= 2:
                ax.fill_between([zone_start, zone_end],
                               min(level_prices), max(level_prices),
                               color=self.golden_zone_color, alpha=0.15)

        # Plot CHoCH labels
        for structure in self.structures:
            struct_idx = structure['idx'] - start_idx
            if 0 <= struct_idx < (end_idx - start_idx):
                if structure['direction'] == 'bullish' and self.show_bullish:
                    ax.annotate('CHoCH', xy=(struct_idx, structure['price']),
                               xytext=(struct_idx, structure['price']),
                               color=self.bullish_color, fontsize=7, ha='center',
                               fontweight='bold', alpha=0.8)
                elif structure['direction'] == 'bearish' and self.show_bearish:
                    ax.annotate('CHoCH', xy=(struct_idx, structure['price']),
                               xytext=(struct_idx, structure['price']),
                               color=self.bearish_color, fontsize=7, ha='center',
                               fontweight='bold', alpha=0.8)


def load_ohlc_data(csv_path: str) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.

    Parameters:
    -----------
    csv_path : str
        Path to CSV file

    Returns:
    --------
    pd.DataFrame
        OHLC dataframe with datetime index
    """
    df = pd.read_csv(csv_path)

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()

    # Handle different datetime column names
    datetime_cols = ['time', 'datetime', 'date', 'timestamp']
    for col in datetime_cols:
        if col in df.columns:
            df['datetime'] = pd.to_datetime(df[col])
            break

    # Ensure we have OHLC columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            # Try to find columns that might contain the data
            for c in df.columns:
                if col in c.lower():
                    df[col] = df[c]
                    break

    return df


def generate_fibonacci_ote_chart(csv_path: str,
                                  output_path: str,
                                  last_n_bars: int = 200,
                                  **kwargs) -> plt.Figure:
    """
    Generate Fibonacci OTE chart from CSV data.

    Parameters:
    -----------
    csv_path : str
        Path to OHLC CSV file
    output_path : str
        Path to save the chart
    last_n_bars : int
        Number of bars to display
    **kwargs
        Additional parameters for FibonacciOTE

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Load data
    df = load_ohlc_data(csv_path)

    # Create indicator
    fib_ote = FibonacciOTE(**kwargs)

    # Generate chart
    fig = fib_ote.plot(df,
                       last_n_bars=last_n_bars,
                       title='Fibonacci Optimal Entry Zone [OTE] (Zeiierman)',
                       save_path=output_path,
                       show_plot=False)

    return fig


def create_combined_chart(csv_path: str,
                          output_path: str,
                          last_n_bars: int = 200) -> plt.Figure:
    """
    Create combined market structure HTF chart with Fibonacci OTE.

    Parameters:
    -----------
    csv_path : str
        Path to OHLC CSV file
    output_path : str
        Path to save the chart
    last_n_bars : int
        Number of bars to display

    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Load data
    df = load_ohlc_data(csv_path)

    # Slice data
    if last_n_bars:
        df_plot = df.tail(last_n_bars).copy().reset_index(drop=True)
    else:
        df_plot = df.copy()

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14),
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#131722')
    ax1.set_facecolor('#131722')
    ax2.set_facecolor('#131722')

    # Plot candlesticks on main axis
    for i in range(len(df_plot)):
        bar = df_plot.iloc[i]

        if bar['close'] >= bar['open']:
            color = '#26a69a'
        else:
            color = '#ef5350'

        # Draw wick
        ax1.plot([i, i], [bar['low'], bar['high']], color=color, linewidth=1)

        # Draw body
        body_bottom = min(bar['open'], bar['close'])
        body_height = abs(bar['close'] - bar['open'])
        if body_height < 0.01:
            body_height = 0.01
        rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height,
                         facecolor=color, edgecolor=color, linewidth=1)
        ax1.add_patch(rect)

    # Add Fibonacci OTE indicator
    fib_ote = FibonacciOTE(
        structure_period=10,
        show_bullish=True,
        show_bearish=True,
        fill_golden_zone=True,
        show_previous=True
    )
    fib_ote.add_to_existing_chart(ax1, df,
                                   start_idx=len(df)-last_n_bars if last_n_bars else 0,
                                   end_idx=len(df))

    # Add Market Structure analysis (simplified HTF view)
    # Calculate simple market structure using higher timeframe logic
    htf_period = 20
    df_plot['htf_high'] = df_plot['high'].rolling(window=htf_period).max()
    df_plot['htf_low'] = df_plot['low'].rolling(window=htf_period).min()

    # Plot HTF structure on main chart
    ax1.plot(range(len(df_plot)), df_plot['htf_high'],
             color='#FFD700', linewidth=1, alpha=0.5, linestyle='--', label='HTF High')
    ax1.plot(range(len(df_plot)), df_plot['htf_low'],
             color='#FF6B6B', linewidth=1, alpha=0.5, linestyle='--', label='HTF Low')

    # Styling for main chart
    ax1.set_title('Combined Market Structure HTF Chart with Fibonacci OTE',
                  color='white', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', color='white')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['top'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['right'].set_color('white')
    ax1.grid(True, alpha=0.2, color='gray')

    # Create legend
    legend_elements = [
        mpatches.Patch(color='#08ec32', label='Bullish CHoCH'),
        mpatches.Patch(color='#ff2222', label='Bearish CHoCH'),
        mpatches.Patch(color='#4CAF50', label='Fib 0.50'),
        mpatches.Patch(color='#009688', label='Fib 0.618'),
        mpatches.Patch(color='#009688', alpha=0.3, label='Golden Zone (OTE)'),
        mpatches.Patch(color='#FFD700', alpha=0.5, label='HTF High'),
        mpatches.Patch(color='#FF6B6B', alpha=0.5, label='HTF Low'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left',
               facecolor='#1e222d', edgecolor='white', labelcolor='white')

    # Plot momentum/trend panel on secondary axis
    # Simple momentum calculation
    df_plot['momentum'] = df_plot['close'].diff(5)
    colors = ['#26a69a' if x >= 0 else '#ef5350' for x in df_plot['momentum'].fillna(0)]
    ax2.bar(range(len(df_plot)), df_plot['momentum'].fillna(0), color=colors, alpha=0.7)

    ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Bar Index', color='white')
    ax2.set_ylabel('Momentum', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['top'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['right'].set_color('white')
    ax2.grid(True, alpha=0.2, color='gray')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='#131722', edgecolor='none')
        print(f"Combined chart saved to: {output_path}")

    return fig


# Main execution
if __name__ == '__main__':
    import os

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Path to CSV file
    csv_path = os.path.join(parent_dir, 'PEPPERSTONE_XAUUSD, 5.csv')

    # Output paths
    fib_ote_chart_path = os.path.join(script_dir, 'Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.png')
    combined_chart_path = os.path.join(script_dir, 'combined_market_structure_htf_chart.png')

    print("Generating Fibonacci OTE Chart...")
    generate_fibonacci_ote_chart(
        csv_path=csv_path,
        output_path=fib_ote_chart_path,
        last_n_bars=200,
        structure_period=10,
        fill_golden_zone=True,
        show_previous=True
    )

    print("\nGenerating Combined Market Structure HTF Chart...")
    create_combined_chart(
        csv_path=csv_path,
        output_path=combined_chart_path,
        last_n_bars=200
    )

    print("\nDone!")
