"""
CandelaCharts - HTF Sweeps Chart
Standalone candlestick chart showing HTF candles, sweeps, and bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load HTF Sweeps module
spec = importlib.util.spec_from_file_location(
    "htf_sweeps",
    "CandelaCharts - HTF Sweeps.py"
)
htf_sweeps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(htf_sweeps)

calculate_htf_data = htf_sweeps.calculate_htf_data


def plot_htf_sweeps_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with HTF candles, sweeps, and bias
    Shows 1H (blue), 4H (orange), and Daily (green) HTF levels

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Add index for calculations
    df['index'] = range(len(df))

    # Calculate HTF data (1H, 4H, Daily)
    print("Calculating HTF candles, sweeps, and bias...")
    htf_data = calculate_htf_data(
        df,
        timeframes_minutes={'1H': 60, '4H': 240, 'Daily': 1440}
    )

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure with main chart and HTF candle insets
    fig = plt.figure(figsize=(22, 14))

    # Main chart takes most of the space
    ax_main = plt.subplot2grid((20, 20), (0, 0), colspan=20, rowspan=17)

    # HTF candle mini-charts in top right
    ax_1h = plt.subplot2grid((20, 20), (0, 15), colspan=5, rowspan=4)
    ax_4h = plt.subplot2grid((20, 20), (4, 15), colspan=5, rowspan=4)
    ax_daily = plt.subplot2grid((20, 20), (8, 15), colspan=5, rowspan=4)

    # ========================================================================
    # PLOT MAIN CANDLESTICKS
    # ========================================================================
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
        color = 'blue' if row['close'] >= row['open'] else 'red'

        # Draw candle body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                         facecolor=color, edgecolor=color, alpha=0.8, zorder=2)
        ax_main.add_patch(rect)

        # Draw wicks
        ax_main.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5, zorder=1)

    # ========================================================================
    # PLOT HTF LEVELS AND SWEEPS
    # ========================================================================

    # Define colors for each timeframe
    tf_colors = {
        '1H': 'blue',
        '4H': 'orange',
        'Daily': 'green'
    }

    for tf_name, data in htf_data.items():
        color = tf_colors.get(tf_name, 'gray')

        # Plot HTF High/Low levels from recent candles
        recent_candles = data['candles'][-5:] if len(data['candles']) > 0 else []

        for candle in recent_candles:
            # Check if candle is in display range
            if candle.close_idx < start_idx:
                continue

            # Map indices to display range
            open_idx_display = max(0, candle.open_idx - start_idx)
            close_idx_display = min(len(df_display) - 1, candle.close_idx - start_idx)

            if close_idx_display < 0 or open_idx_display >= len(df_display):
                continue

            # Plot HTF High level
            ax_main.plot([open_idx_display, close_idx_display],
                        [candle.high, candle.high],
                        color=color, linewidth=1.5, linestyle='-',
                        alpha=0.7, zorder=3)

            # Plot HTF Low level
            ax_main.plot([open_idx_display, close_idx_display],
                        [candle.low, candle.low],
                        color=color, linewidth=1.5, linestyle='-',
                        alpha=0.7, zorder=3)

            # Add labels on the right
            ax_main.text(close_idx_display, candle.high, f' {tf_name} H',
                        fontsize=7, color=color, va='bottom', ha='left',
                        fontweight='bold', zorder=5)
            ax_main.text(close_idx_display, candle.low, f' {tf_name} L',
                        fontsize=7, color=color, va='top', ha='left',
                        fontweight='bold', zorder=5)

        # Plot sweeps
        for sweep in data['sweeps']:
            if sweep.index < start_idx:
                continue

            sweep_idx = sweep.index - start_idx
            if 0 <= sweep_idx < len(df_display):
                # Bullish sweep (swept high) - green triangle down
                if sweep.is_bullish:
                    ax_main.scatter(sweep_idx, sweep.price, marker='v', s=150,
                                  color='lime', edgecolors=color, linewidth=2,
                                  zorder=6, alpha=0.9)
                # Bearish sweep (swept low) - red triangle up
                else:
                    ax_main.scatter(sweep_idx, sweep.price, marker='^', s=150,
                                  color='red', edgecolors=color, linewidth=2,
                                  zorder=6, alpha=0.9)

        # Display bias
        bias = data['bias']
        bias_text = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}
        bias_color = {1: 'green', -1: 'red', 0: 'gray'}

        print(f"  {tf_name} Bias: {bias_text[bias]} ({len(data['candles'])} candles, {len(data['sweeps'])} sweeps)")

    # ========================================================================
    # PLOT HTF MINI-CANDLES (Top Right Insets)
    # ========================================================================

    def plot_mini_htf_candles(ax, candles, tf_name, color, num_to_show=10):
        """Plot mini HTF candles in inset chart"""
        recent = candles[-num_to_show:] if len(candles) > 0 else []

        for idx, candle in enumerate(recent):
            candle_color = 'green' if candle.is_bullish else 'red'

            # Draw candle body
            body_height = abs(candle.close - candle.open)
            body_bottom = min(candle.open, candle.close)
            rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                           facecolor=candle_color, edgecolor='black',
                           alpha=0.8, linewidth=0.5)
            ax.add_patch(rect)

            # Draw wicks
            ax.plot([idx, idx], [candle.low, candle.high],
                   color='black', linewidth=0.5)

        ax.set_xlim(-0.5, len(recent) - 0.5)
        if len(recent) > 0:
            y_min = min(c.low for c in recent)
            y_max = max(c.high for c in recent)
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

        ax.set_title(f'{tf_name} HTF Candles', fontsize=9, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Plot mini candles for each timeframe
    if '1H' in htf_data:
        plot_mini_htf_candles(ax_1h, htf_data['1H']['candles'], '1H', 'blue')
    if '4H' in htf_data:
        plot_mini_htf_candles(ax_4h, htf_data['4H']['candles'], '4H', 'orange')
    if 'Daily' in htf_data:
        plot_mini_htf_candles(ax_daily, htf_data['Daily']['candles'], 'Daily', 'green')

    # ========================================================================
    # FORMATTING
    # ========================================================================

    ax_main.set_xlim(-0.5, len(df_display) - 0.5)
    ax_main.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Price', fontsize=12, fontweight='bold')

    # Build title with bias information
    bias_info = []
    for tf_name in ['1H', '4H', 'Daily']:
        if tf_name in htf_data:
            bias = htf_data[tf_name]['bias']
            bias_text = {1: '↑', -1: '↓', 0: '→'}
            bias_info.append(f"{tf_name}{bias_text[bias]}")

    ax_main.set_title(f"CandelaCharts - HTF Sweeps | Bias: {' | '.join(bias_info)}",
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='1H HTF Levels'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='4H HTF Levels'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Daily HTF Levels'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                  markersize=10, label='Bearish Sweep', markeredgecolor='black', markeredgewidth=1),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='lime',
                  markersize=10, label='Bullish Sweep', markeredgecolor='black', markeredgewidth=1)
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    output_file = "CandelaCharts - HTF Sweeps.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nHTF Sweeps chart saved as '{output_file}'")
    print(f"\nHTF Summary:")
    for tf_name, data in htf_data.items():
        bias_text = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}
        print(f"  {tf_name}:")
        print(f"    - HTF Candles: {len(data['candles'])}")
        print(f"    - Sweeps: {len(data['sweeps'])}")
        print(f"    - Bias: {bias_text[data['bias']]}")


if __name__ == "__main__":
    plot_htf_sweeps_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
