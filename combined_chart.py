"""
Combined Market Structure MTF + HTF Bias Chart

This script combines:
1. Market Structure MTF Trend indicator (multi-timeframe trend panel)
2. HTF Bias indicator (HTF candles, sweeps, and bias)

Display:
- Main chart: Candlesticks with HTF levels and sweep markers
- Right side: Mini HTF candles (1H, 4H, Daily)
- Bottom panel: Market structure trend indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys
import os

# Import market structure functions
sys.path.append(os.path.dirname(__file__))
from market_structure_chart import (
    calculate_market_structure,
    plot_trend_line
)
from HTF_bias import calculate_htf_data, HTFCandle, Sweep


def plot_combined_chart(csv_file, num_candles=200, pivot_strength=15):
    """
    Plot combined chart with Market Structure MTF + HTF Bias indicators.

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

    # ========================================================================
    # CALCULATE MARKET STRUCTURE
    # ========================================================================
    print("Calculating market structure for multiple timeframes...")
    trend_tf1, choch_tf1, bos_tf1, pivots_h_tf1, pivots_l_tf1 = calculate_market_structure(df, pivot_strength=5)
    trend_tf2, choch_tf2, bos_tf2, pivots_h_tf2, pivots_l_tf2 = calculate_market_structure(df, pivot_strength=10)
    trend_tf3, choch_tf3, bos_tf3, pivots_h_tf3, pivots_l_tf3 = calculate_market_structure(df, pivot_strength=15)
    trend_tf4, choch_tf4, bos_tf4, pivots_h_tf4, pivots_l_tf4 = calculate_market_structure(df, pivot_strength=20)

    # ========================================================================
    # CALCULATE HTF DATA
    # ========================================================================
    print("Calculating HTF candles and sweeps...")
    htf_data = calculate_htf_data(df, timeframes_minutes={'1H': 60, '4H': 240, 'Daily': 1440})

    # ========================================================================
    # CREATE FIGURE WITH SUBPLOTS
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 10, height_ratios=[3, 1], hspace=0.1, wspace=0.3)

    # Main chart (candlesticks + HTF levels)
    ax_main = fig.add_subplot(gs[0, :8])  # Takes 8 columns

    # HTF candles mini chart
    ax_htf = fig.add_subplot(gs[0, 8:], sharey=ax_main)  # Takes 2 columns on the right

    # Market structure indicator panel
    ax_ms = fig.add_subplot(gs[1, :8], sharex=ax_main)  # Bottom panel

    # ========================================================================
    # PLOT MAIN CANDLESTICKS
    # ========================================================================
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

    # ========================================================================
    # PLOT HTF LEVELS AND SWEEPS
    # ========================================================================
    colors_htf = {'1H': 'orange', '4H': 'purple', 'Daily': 'brown'}
    alphas = {'1H': 0.3, '4H': 0.4, 'Daily': 0.5}

    for tf_name, data in htf_data.items():
        color = colors_htf.get(tf_name, 'gray')
        alpha = alphas.get(tf_name, 0.3)

        # Plot HTF high/low lines
        for candle in data['candles'][-10:]:  # Last 10 HTF candles
            # High line
            ax_main.axhline(y=candle.high, color=color, linestyle='--',
                          linewidth=1, alpha=alpha, zorder=0)
            # Low line
            ax_main.axhline(y=candle.low, color=color, linestyle='--',
                          linewidth=1, alpha=alpha, zorder=0)

        # Plot sweeps
        for sweep in data['sweeps']:
            if sweep.is_bullish:
                # Bullish sweep (swept high)
                ax_main.scatter(sweep.index, sweep.price, marker='^',
                              color='green', s=100, zorder=5, alpha=0.7,
                              label=f'{tf_name} Bull Sweep' if sweep == data['sweeps'][0] else '')
            else:
                # Bearish sweep (swept low)
                ax_main.scatter(sweep.index, sweep.price, marker='v',
                              color='red', s=100, zorder=5, alpha=0.7,
                              label=f'{tf_name} Bear Sweep' if sweep == data['sweeps'][0] else '')

    # ========================================================================
    # PLOT HTF MINI CANDLES (Right side)
    # ========================================================================
    htf_candle_width = 0.6
    y_min, y_max = ax_main.get_ylim()
    price_range = y_max - y_min

    x_offset = 0
    for tf_idx, (tf_name, data) in enumerate(htf_data.items()):
        candles = data['candles'][-8:]  # Show last 8 HTF candles
        bias = data['bias']

        for i, candle in enumerate(candles):
            x_pos = x_offset + i * (htf_candle_width + 0.2)

            # Normalize price to 0-1 range for display
            norm_open = (candle.open - y_min) / price_range
            norm_close = (candle.close - y_min) / price_range
            norm_high = (candle.high - y_min) / price_range
            norm_low = (candle.low - y_min) / price_range

            # Convert back to actual prices for display
            display_open = y_min + norm_open * price_range
            display_close = y_min + norm_close * price_range
            display_high = y_min + norm_high * price_range
            display_low = y_min + norm_low * price_range

            # Color
            if candle.is_bullish:
                color = 'green'
                body_bottom = display_open
                body_height = display_close - display_open
            else:
                color = 'red'
                body_bottom = display_close
                body_height = display_open - display_close

            # Draw wick
            ax_htf.plot([x_pos, x_pos], [display_low, display_high],
                       color='black', linewidth=1, zorder=1)

            # Draw body
            rect = Rectangle(
                (x_pos - htf_candle_width/2, body_bottom),
                htf_candle_width,
                body_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
            ax_htf.add_patch(rect)

        # Add timeframe label
        ax_htf.text(x_offset + 3, y_max + price_range * 0.02, tf_name,
                   fontsize=10, fontweight='bold', ha='center')

        # Add bias indicator
        bias_text = "↑" if bias == 1 else "↓" if bias == -1 else "→"
        bias_color = "green" if bias == 1 else "red" if bias == -1 else "gray"
        ax_htf.text(x_offset + 3, y_min - price_range * 0.02, bias_text,
                   fontsize=14, fontweight='bold', ha='center', color=bias_color)

        x_offset += len(candles) * (htf_candle_width + 0.2) + 2

    # ========================================================================
    # PLOT CHoCH AND BoS LABELS
    # ========================================================================
    for idx, direction, price in choch_tf3:
        if direction > 0:  # Bullish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price - 10),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        else:  # Bearish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price + 10),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # ========================================================================
    # PLOT MARKET STRUCTURE TRENDS
    # ========================================================================
    choch_bull_color = 'darkgreen'
    choch_bear_color = 'darkred'
    bos_bull_color = 'green'
    bos_bear_color = 'red'

    plot_trend_line(ax_ms, trend_tf1, choch_tf1, bos_tf1, 3,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf2, choch_tf2, bos_tf2, 2,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf3, choch_tf3, bos_tf3, 1,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf4, choch_tf4, bos_tf4, 0,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Add timeframe labels
    ax_ms.text(-5, 3, 'TF1\n(15m)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 2, 'TF2\n(30m)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 1, 'TF3\n(1H)', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 0, 'TF4\n(4H)', ha='right', va='center', fontsize=9, fontweight='bold')

    # ========================================================================
    # FORMATTING
    # ========================================================================
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
    ax_main.set_title('XAUUSD - Market Structure MTF + HTF Bias Analysis',
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(loc='upper left', fontsize=8)

    # HTF mini chart formatting
    ax_htf.set_ylabel('')
    ax_htf.set_title('HTF Candles', fontsize=10, fontweight='bold')
    ax_htf.set_xticks([])
    ax_htf.set_yticks([])
    ax_htf.grid(True, alpha=0.2)

    # Market structure panel formatting
    ax_ms.set_ylabel('Timeframes', fontsize=10, fontweight='bold')
    ax_ms.set_ylim(-0.5, 3.5)
    ax_ms.set_yticks([0, 1, 2, 3])
    ax_ms.set_yticklabels([])
    ax_ms.grid(True, alpha=0.2, axis='x')
    ax_ms.set_xlim(-10, len(df))

    # Market structure legend
    legend_elements = [
        Line2D([0], [0], color=choch_bull_color, linewidth=8, label='CHoCH Bullish'),
        Line2D([0], [0], color=choch_bear_color, linewidth=8, label='CHoCH Bearish'),
        Line2D([0], [0], color=bos_bull_color, linewidth=8, label='BoS Bullish'),
        Line2D([0], [0], color=bos_bear_color, linewidth=8, label='BoS Bearish'),
    ]
    ax_ms.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=4)

    plt.tight_layout()

    # Save
    output_file = 'combined_market_structure_htf_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCombined chart saved as '{output_file}'")

    plt.show()


if __name__ == '__main__':
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_combined_chart(csv_file, num_candles=200, pivot_strength=15)
