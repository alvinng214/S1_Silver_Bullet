"""
Ighodalo Gold - CRT (Candles are Ranges Theory) Chart
Standalone candlestick chart showing CRT levels and Turtle Soup signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load CRT module
spec = importlib.util.spec_from_file_location(
    "crt_module",
    "Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.py"
)
crt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crt_module)

detect_crt_ranges = crt_module.detect_crt_ranges
detect_turtle_soup_signals = crt_module.detect_turtle_soup_signals
filter_overlapping_crts = crt_module.filter_overlapping_crts


def plot_crt_chart(csv_file, num_candles=500, lookback=20, timeframe_minutes=60):
    """
    Plot candlestick chart with CRT levels and Turtle Soup signals

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        lookback: CRT detection lookback period
        timeframe_minutes: Timeframe for CRT detection (60 = 1H)
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect CRT ranges on full dataset
    print(f"Detecting CRT ranges on {timeframe_minutes}min timeframe...")
    crt_ranges = detect_crt_ranges(df, lookback=lookback, timeframe_minutes=timeframe_minutes)
    print(f"Found {len(crt_ranges)} CRT ranges")

    # Filter to non-overlapping (optional)
    crt_ranges = filter_overlapping_crts(crt_ranges, enable_overlapping=False)

    # Detect Turtle Soup signals
    print("Detecting Turtle Soup signals...")
    signals = detect_turtle_soup_signals(df, crt_ranges, atr_multiplier=0.1, atr_period=14)
    print(f"Found {len(signals['buy'])} buy signals and {len(signals['sell'])} sell signals")

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(22, 14))

    # Plot candlesticks
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
        color = 'blue' if row['close'] >= row['open'] else 'red'

        # Draw candle body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                         facecolor=color, edgecolor=color, alpha=0.8, zorder=2)
        ax.add_patch(rect)

        # Draw wicks
        ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5, zorder=1)

    # Plot CRT ranges (only active ones or last 3)
    active_crts = [crt for crt in crt_ranges if crt.is_active or crt.end_idx >= start_idx][-3:]

    for crt in active_crts:
        # Map to display indices
        crt_start = max(0, crt.start_idx - start_idx)
        crt_end = min(len(df_display) - 1, crt.end_idx - start_idx)

        # CRT High (black solid line)
        ax.plot([crt_start, crt_end], [crt.high, crt.high],
               color='black', linewidth=1.5, linestyle='-', alpha=0.8, zorder=3)
        ax.text(crt_end, crt.high, ' CRTH', fontsize=8,
               color='black', va='bottom', ha='left', fontweight='bold', zorder=5)

        # CRT Low (black solid line)
        ax.plot([crt_start, crt_end], [crt.low, crt.low],
               color='black', linewidth=1.5, linestyle='-', alpha=0.8, zorder=3)
        ax.text(crt_end, crt.low, ' CRTL', fontsize=8,
               color='black', va='top', ha='left', fontweight='bold', zorder=5)

        # CRT Midpoint (gray dashed line)
        ax.plot([crt_start, crt_end], [crt.mid, crt.mid],
               color='gray', linewidth=1.2, linestyle='--', alpha=0.6, zorder=3)
        ax.text(crt_end, crt.mid, ' CRT 50%', fontsize=7,
               color='gray', va='center', ha='left', fontweight='bold', zorder=5)

    # Plot Turtle Soup signals
    # Buy signals (green triangle up)
    for signal in signals['buy']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='^', s=200,
                      color='green', edgecolors='black', linewidth=1.5, zorder=6, alpha=0.9)

    # Sell signals (red triangle down)
    for signal in signals['sell']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='v', s=200,
                      color='red', edgecolors='black', linewidth=1.5, zorder=6, alpha=0.9)

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(f"Ighodalo Gold - CRT (Candles are Ranges Theory) - {timeframe_minutes}min Timeframe",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=1.5, label='CRT High/Low'),
        plt.Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--', label='CRT 50%'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                  markersize=10, label='Turtle Soup Buy', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
                  markersize=10, label='Turtle Soup Sell', markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    output_file = "Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCRT chart saved as '{output_file}'")
    print(f"\nCRT Summary:")
    print(f"  - Total CRT Ranges Detected: {len(crt_ranges)}")
    print(f"  - Active CRT Ranges: {len([c for c in crt_ranges if c.is_active])}")
    print(f"  - Buy Signals: {len(signals['buy'])}")
    print(f"  - Sell Signals: {len(signals['sell'])}")


if __name__ == "__main__":
    plot_crt_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500, lookback=20, timeframe_minutes=60)
