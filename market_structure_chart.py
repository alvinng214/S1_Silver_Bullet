"""
Market Structure MTF Trend Chart Visualization
Python version of Pine Script: Market Structure MTF Trend [Pt]

This script displays:
1. Candlestick chart of OHLC data
2. Market structure trend indicators for multiple timeframes (as colored horizontal lines)
3. CHoCH (Change of Character) and BoS (Break of Structure) labels on the chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


def detect_pivot_high(highs, pivot_strength, index):
    """
    Detect pivot high at given index.
    A pivot high is confirmed when the high at that position is higher than
    pivot_strength bars before and after it.
    """
    if index < pivot_strength or index >= len(highs) - pivot_strength:
        return None

    pivot_high = highs[index]

    # Check if it's higher than all bars before it
    for i in range(1, pivot_strength + 1):
        if highs[index - i] >= pivot_high:
            return None

    # Check if it's higher than all bars after it
    for i in range(1, pivot_strength + 1):
        if highs[index + i] >= pivot_high:
            return None

    return pivot_high


def detect_pivot_low(lows, pivot_strength, index):
    """
    Detect pivot low at given index.
    A pivot low is confirmed when the low at that position is lower than
    pivot_strength bars before and after it.
    """
    if index < pivot_strength or index >= len(lows) - pivot_strength:
        return None

    pivot_low = lows[index]

    # Check if it's lower than all bars before it
    for i in range(1, pivot_strength + 1):
        if lows[index - i] <= pivot_low:
            return None

    # Check if it's lower than all bars after it
    for i in range(1, pivot_strength + 1):
        if lows[index + i] <= pivot_low:
            return None

    return pivot_low


def calculate_market_structure(df, pivot_strength=15):
    """
    Calculate market structure trend based on pivot points.

    Returns:
    - trend: array of trend values (1=bullish, -1=bearish, 0=undefined)
    - choch_signals: list of (index, direction, price) tuples
    - bos_signals: list of (index, direction, price) tuples
    - pivot_highs: dict of {index: price}
    - pivot_lows: dict of {index: price}
    """
    n = len(df)
    trend = np.zeros(n)

    # State variables
    last_pivot_high = None
    last_pivot_low = None
    last_broken_high = None
    last_broken_low = None
    current_trend = 0  # 0=undefined, 1=bullish, -1=bearish

    choch_signals = []
    bos_signals = []
    pivot_highs = {}
    pivot_lows = {}

    for i in range(n):
        # Detect pivots at position i - pivot_strength (with delay)
        if i >= pivot_strength * 2:
            pivot_idx = i - pivot_strength

            # Check for pivot high
            ph = detect_pivot_high(df['high'].values, pivot_strength, pivot_idx)
            if ph is not None:
                pivot_highs[pivot_idx] = ph
                if current_trend == 1:  # Bullish trend
                    # Track the highest pivot high
                    if last_pivot_high is None:
                        last_pivot_high = ph
                    else:
                        last_pivot_high = max(ph, last_pivot_high)
                else:
                    # New pivot high
                    last_pivot_high = ph

            # Check for pivot low
            pl = detect_pivot_low(df['low'].values, pivot_strength, pivot_idx)
            if pl is not None:
                pivot_lows[pivot_idx] = pl
                if current_trend == -1:  # Bearish trend
                    # Track the lowest pivot low
                    if last_pivot_low is None:
                        last_pivot_low = pl
                    else:
                        last_pivot_low = min(pl, last_pivot_low)
                else:
                    # New pivot low
                    last_pivot_low = pl

        # Check for structure breaks using CLOSE price
        close_price = df['close'].iloc[i]
        prev_close = df['close'].iloc[i-1] if i > 0 else close_price

        # Bullish structure break: close crosses above last pivot high
        if last_pivot_high is not None and close_price > last_pivot_high and prev_close <= last_pivot_high:
            if current_trend == 1 and last_pivot_high != last_broken_high:
                # BoS: Break in same direction (continuation)
                bos_signals.append((i, 1, close_price))
                last_broken_high = last_pivot_high
            elif current_trend != 1:
                # CHoCH: Change of Character (reversal to bullish)
                choch_signals.append((i, 1, close_price))
                last_broken_high = last_pivot_high
                last_broken_low = None

            current_trend = 1

        # Bearish structure break: close crosses below last pivot low
        elif last_pivot_low is not None and close_price < last_pivot_low and prev_close >= last_pivot_low:
            if current_trend == -1 and last_pivot_low != last_broken_low:
                # BoS: Break in same direction (continuation)
                bos_signals.append((i, -1, close_price))
                last_broken_low = last_pivot_low
            elif current_trend != -1:
                # CHoCH: Change of Character (reversal to bearish)
                choch_signals.append((i, -1, close_price))
                last_broken_low = last_pivot_low
                last_broken_high = None

            current_trend = -1

        trend[i] = current_trend

    return trend, choch_signals, bos_signals, pivot_highs, pivot_lows


def resample_to_timeframe(df, minutes):
    """
    Resample the dataframe to a higher timeframe.

    Args:
        df: DataFrame with time index and OHLC columns
        minutes: Number of minutes for the new timeframe

    Returns:
        Resampled DataFrame
    """
    df_resampled = df.resample(f'{minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    return df_resampled


def plot_market_structure_chart(csv_file, num_candles=200, pivot_strength=15):
    """
    Plot candlestick chart with market structure indicators.

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of recent candles to display
        pivot_strength: Pivot strength for structure detection
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Get last N candles
    df = df.tail(num_candles).copy()
    df['index'] = range(len(df))

    print(f"Analyzing last {len(df)} candles...")

    # Calculate market structure for current timeframe (5-minute based on filename)
    print("Calculating market structure for TF1 (15-min equivalent)...")
    trend_tf1, choch_tf1, bos_tf1, pivots_h_tf1, pivots_l_tf1 = calculate_market_structure(df, pivot_strength=5)

    print("Calculating market structure for TF2 (30-min equivalent)...")
    trend_tf2, choch_tf2, bos_tf2, pivots_h_tf2, pivots_l_tf2 = calculate_market_structure(df, pivot_strength=10)

    print("Calculating market structure for TF3 (1-hour equivalent)...")
    trend_tf3, choch_tf3, bos_tf3, pivots_h_tf3, pivots_l_tf3 = calculate_market_structure(df, pivot_strength=15)

    print("Calculating market structure for TF4 (4-hour equivalent)...")
    trend_tf4, choch_tf4, bos_tf4, pivots_h_tf4, pivots_l_tf4 = calculate_market_structure(df, pivot_strength=20)

    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)

    # Main chart (candlesticks)
    ax_main = fig.add_subplot(gs[0])

    # Market structure indicator panel
    ax_ms = fig.add_subplot(gs[1], sharex=ax_main)

    # ====================================
    # Plot Candlesticks
    # ====================================
    width = 0.8
    for idx in range(len(df)):
        row = df.iloc[idx]
        x_pos = idx
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']

        # Determine color
        if close_price >= open_price:
            color = 'blue'
            body_bottom = open_price
            body_height = close_price - open_price
        else:
            color = 'red'
            body_bottom = close_price
            body_height = open_price - close_price

        # Draw wick
        ax_main.plot([x_pos, x_pos], [low_price, high_price], color='black', linewidth=1, zorder=1)

        # Draw body
        rect = Rectangle(
            (x_pos - width/2, body_bottom),
            width,
            body_height,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=2
        )
        ax_main.add_patch(rect)

    # ====================================
    # Plot CHoCH and BoS labels on main chart
    # ====================================
    # TF3 signals (most relevant for display)
    for idx, direction, price in choch_tf3:
        if direction > 0:  # Bullish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price - 20),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        else:  # Bearish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price + 20),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    for idx, direction, price in bos_tf3:
        if direction > 0:  # Bullish BoS
            ax_main.annotate('BoS', xy=(idx, price), xytext=(idx, price - 15),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5),
                           fontsize=7, color='darkgreen', ha='center',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1, linestyle='--'))
        else:  # Bearish BoS
            ax_main.annotate('BoS', xy=(idx, price), xytext=(idx, price + 15),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.5),
                           fontsize=7, color='darkred', ha='center',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1, linestyle='--'))

    # ====================================
    # Plot Market Structure Trends
    # ====================================
    # Define colors for CHoCH and BoS
    choch_bull_color = 'darkgreen'
    choch_bear_color = 'darkred'
    bos_bull_color = 'green'
    bos_bear_color = 'red'

    # Plot TF1 (y=3)
    plot_trend_line(ax_ms, trend_tf1, choch_tf1, bos_tf1, 3,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Plot TF2 (y=2)
    plot_trend_line(ax_ms, trend_tf2, choch_tf2, bos_tf2, 2,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Plot TF3 (y=1)
    plot_trend_line(ax_ms, trend_tf3, choch_tf3, bos_tf3, 1,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Plot TF4 (y=0)
    plot_trend_line(ax_ms, trend_tf4, choch_tf4, bos_tf4, 0,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Add timeframe labels
    ax_ms.text(-5, 3, 'TF1\n(15m)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 2, 'TF2\n(30m)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 1, 'TF3\n(1H)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 0, 'TF4\n(4H)', ha='right', va='center', fontsize=9, fontweight='bold')

    # ====================================
    # Formatting
    # ====================================
    # X-axis labels
    step = max(1, len(df) // 10)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df.iloc[i].name.strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels([])  # Hide x labels on main chart

    ax_ms.set_xticks(tick_positions)
    ax_ms.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Main chart formatting
    ax_main.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax_main.set_title('XAUUSD Market Structure MTF Trend', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # Market structure panel formatting
    ax_ms.set_ylabel('Timeframes', fontsize=10, fontweight='bold')
    ax_ms.set_ylim(-0.5, 3.5)
    ax_ms.set_yticks([0, 1, 2, 3])
    ax_ms.set_yticklabels([])
    ax_ms.grid(True, alpha=0.2, axis='x')
    ax_ms.set_xlim(-10, len(df))

    # Legend
    legend_elements = [
        Line2D([0], [0], color=choch_bull_color, linewidth=8, label='CHoCH Bullish'),
        Line2D([0], [0], color=choch_bear_color, linewidth=8, label='CHoCH Bearish'),
        Line2D([0], [0], color=bos_bull_color, linewidth=8, label='BoS Bullish'),
        Line2D([0], [0], color=bos_bear_color, linewidth=8, label='BoS Bearish'),
    ]
    ax_ms.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=4)

    plt.tight_layout()

    # Save
    output_file = 'market_structure_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nChart saved as '{output_file}'")

    plt.show()


def plot_trend_line(ax, trend, choch_signals, bos_signals, y_level,
                   choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color):
    """
    Plot trend as colored horizontal line segments.
    """
    current_color = None
    segment_start = 0

    # Convert signals to sets for quick lookup
    choch_indices = {idx for idx, _, _ in choch_signals}
    bos_indices = {idx for idx, _, _ in bos_signals}
    choch_dict = {idx: direction for idx, direction, _ in choch_signals}
    bos_dict = {idx: direction for idx, direction, _ in bos_signals}

    for i in range(len(trend)):
        # Determine color for this bar
        if i in choch_indices:
            # CHoCH signal
            if choch_dict[i] > 0:
                new_color = choch_bull_color
            else:
                new_color = choch_bear_color
        elif i in bos_indices:
            # BoS signal
            if bos_dict[i] > 0:
                new_color = bos_bull_color
            else:
                new_color = bos_bear_color
        else:
            # No signal, maintain current color based on trend
            if trend[i] == 1:
                new_color = bos_bull_color if current_color == bos_bull_color or current_color == choch_bull_color else (current_color or bos_bull_color)
            elif trend[i] == -1:
                new_color = bos_bear_color if current_color == bos_bear_color or current_color == choch_bear_color else (current_color or bos_bear_color)
            else:
                new_color = current_color or 'gray'

        # If color changed, draw the previous segment
        if new_color != current_color and current_color is not None:
            ax.plot([segment_start, i], [y_level, y_level],
                   color=current_color, linewidth=10, solid_capstyle='butt')
            segment_start = i

        current_color = new_color

    # Draw final segment
    if current_color is not None:
        ax.plot([segment_start, len(trend)], [y_level, y_level],
               color=current_color, linewidth=10, solid_capstyle='butt')


if __name__ == '__main__':
    # Run the visualization
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_market_structure_chart(csv_file, num_candles=200, pivot_strength=15)
