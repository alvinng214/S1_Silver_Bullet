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


def plot_crt_chart(csv_file, num_candles=500, lookback=20):
    """
    Plot candlestick chart with CRT levels and Turtle Soup signals
    Shows both 1H CRT (blue) and 4H CRT (black)

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        lookback: CRT detection lookback period
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect CRT 1 - 1 Hour timeframe (blue)
    print("Detecting CRT 1 ranges on 60min (1H) timeframe...")
    crt1_ranges = detect_crt_ranges(df, lookback=lookback, timeframe_minutes=60)
    print(f"Found {len(crt1_ranges)} CRT 1 ranges")
    crt1_ranges = filter_overlapping_crts(crt1_ranges, enable_overlapping=False)

    # Detect CRT 2 - 4 Hour timeframe (black)
    print("Detecting CRT 2 ranges on 240min (4H) timeframe...")
    crt2_ranges = detect_crt_ranges(df, lookback=lookback, timeframe_minutes=240)
    print(f"Found {len(crt2_ranges)} CRT 2 ranges")
    crt2_ranges = filter_overlapping_crts(crt2_ranges, enable_overlapping=False)

    # Detect Turtle Soup signals for both timeframes
    print("Detecting Turtle Soup signals for CRT 1 (1H)...")
    signals_crt1 = detect_turtle_soup_signals(df, crt1_ranges, atr_multiplier=0.1, atr_period=14)
    print(f"CRT 1: {len(signals_crt1['buy'])} buy signals and {len(signals_crt1['sell'])} sell signals")

    print("Detecting Turtle Soup signals for CRT 2 (4H)...")
    signals_crt2 = detect_turtle_soup_signals(df, crt2_ranges, atr_multiplier=0.1, atr_period=14)
    print(f"CRT 2: {len(signals_crt2['buy'])} buy signals and {len(signals_crt2['sell'])} sell signals")

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

    # Plot CRT 1 ranges (1H - Blue) - show last 3
    recent_crt1 = crt1_ranges[-3:] if len(crt1_ranges) > 0 else []
    print(f"Displaying {len(recent_crt1)} CRT 1 (1H) ranges on chart")

    for crt in recent_crt1:
        # Map to display indices
        crt_start = max(0, crt.start_idx - start_idx)
        # If CRT is still active, extend to end of chart; otherwise use its end_idx
        if crt.is_active:
            crt_end = len(df_display) - 1
            line_alpha = 0.7
            line_style = '-'
        else:
            crt_end = min(len(df_display) - 1, crt.end_idx - start_idx)
            line_alpha = 0.4  # Faded for broken CRTs
            line_style = '--'  # Dashed for broken CRTs

        # Skip if CRT is completely outside visible range
        if crt_end < 0 or crt_start >= len(df_display):
            continue

        # CRT 1 High (blue)
        ax.plot([crt_start, crt_end], [crt.high, crt.high],
               color='blue', linewidth=1.5, linestyle=line_style, alpha=line_alpha, zorder=3)
        status = " (Active)" if crt.is_active else " (Broken)"
        ax.text(crt_end, crt.high, f' CRT1 H{status}', fontsize=8,
               color='blue', va='bottom', ha='left', fontweight='bold', zorder=5)

        # CRT 1 Low (blue)
        ax.plot([crt_start, crt_end], [crt.low, crt.low],
               color='blue', linewidth=1.5, linestyle=line_style, alpha=line_alpha, zorder=3)
        ax.text(crt_end, crt.low, f' CRT1 L{status}', fontsize=8,
               color='blue', va='top', ha='left', fontweight='bold', zorder=5)

        # CRT 1 Midpoint (light blue dashed)
        ax.plot([crt_start, crt_end], [crt.mid, crt.mid],
               color='lightblue', linewidth=1.2, linestyle='--', alpha=line_alpha * 0.7, zorder=2.8)
        ax.text(crt_end, crt.mid, ' CRT1 50%', fontsize=7,
               color='blue', va='center', ha='left', fontweight='bold', zorder=5)

    # Plot CRT 2 ranges (4H - Black) - show last 3
    recent_crt2 = crt2_ranges[-3:] if len(crt2_ranges) > 0 else []
    print(f"Displaying {len(recent_crt2)} CRT 2 (4H) ranges on chart")

    for crt in recent_crt2:
        # Map to display indices
        crt_start = max(0, crt.start_idx - start_idx)
        # If CRT is still active, extend to end of chart; otherwise use its end_idx
        if crt.is_active:
            crt_end = len(df_display) - 1
            line_alpha = 0.8
            line_style = '-'
        else:
            crt_end = min(len(df_display) - 1, crt.end_idx - start_idx)
            line_alpha = 0.5  # Faded for broken CRTs
            line_style = '--'  # Dashed for broken CRTs

        # Skip if CRT is completely outside visible range
        if crt_end < 0 or crt_start >= len(df_display):
            continue

        # CRT 2 High (black)
        ax.plot([crt_start, crt_end], [crt.high, crt.high],
               color='black', linewidth=2, linestyle=line_style, alpha=line_alpha, zorder=3.5)
        status = " (Active)" if crt.is_active else " (Broken)"
        ax.text(crt_end, crt.high, f' CRT2 H{status}', fontsize=8,
               color='black', va='bottom', ha='left', fontweight='bold', zorder=5)

        # CRT 2 Low (black)
        ax.plot([crt_start, crt_end], [crt.low, crt.low],
               color='black', linewidth=2, linestyle=line_style, alpha=line_alpha, zorder=3.5)
        ax.text(crt_end, crt.low, f' CRT2 L{status}', fontsize=8,
               color='black', va='top', ha='left', fontweight='bold', zorder=5)

        # CRT 2 Midpoint (gray dashed)
        ax.plot([crt_start, crt_end], [crt.mid, crt.mid],
               color='gray', linewidth=1.2, linestyle='--', alpha=line_alpha * 0.7, zorder=3.3)
        ax.text(crt_end, crt.mid, ' CRT2 50%', fontsize=7,
               color='gray', va='center', ha='left', fontweight='bold', zorder=5)

    # Plot Turtle Soup signals for CRT 1 (1H)
    # Buy signals (green triangle up)
    for signal in signals_crt1['buy']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='^', s=200,
                      color='green', edgecolors='blue', linewidth=2, zorder=6, alpha=0.9)

    # Sell signals (red triangle down)
    for signal in signals_crt1['sell']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='v', s=200,
                      color='red', edgecolors='blue', linewidth=2, zorder=6, alpha=0.9)

    # Plot Turtle Soup signals for CRT 2 (4H)
    # Buy signals (lime triangle up with black edge)
    for signal in signals_crt2['buy']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='^', s=250,
                      color='lime', edgecolors='black', linewidth=2.5, zorder=6.5, alpha=0.9)

    # Sell signals (magenta triangle down with black edge)
    for signal in signals_crt2['sell']:
        sig_idx = signal['idx'] - start_idx
        if 0 <= sig_idx < len(df_display):
            ax.scatter(sig_idx, signal['price'], marker='v', s=250,
                      color='magenta', edgecolors='black', linewidth=2.5, zorder=6.5, alpha=0.9)

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title("Ighodalo Gold - CRT (Candles are Ranges Theory) - CRT1 (1H Blue) + CRT2 (4H Black)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=1.5, label='CRT1 (1H) High/Low'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='CRT2 (4H) High/Low'),
        plt.Line2D([0], [0], color='lightblue', linewidth=1.2, linestyle='--', label='CRT1 50%'),
        plt.Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--', label='CRT2 50%'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                  markersize=10, label='CRT1 Buy', markeredgecolor='blue', markeredgewidth=2),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red',
                  markersize=10, label='CRT1 Sell', markeredgecolor='blue', markeredgewidth=2),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='lime',
                  markersize=11, label='CRT2 Buy', markeredgecolor='black', markeredgewidth=2),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='magenta',
                  markersize=11, label='CRT2 Sell', markeredgecolor='black', markeredgewidth=2)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    output_file = "Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nCRT chart saved as '{output_file}'")
    print(f"\nCRT Summary:")
    print(f"  CRT 1 (1H - Blue):")
    print(f"    - Total CRT1 Ranges Detected: {len(crt1_ranges)}")
    print(f"    - Active CRT1 Ranges: {len([c for c in crt1_ranges if c.is_active])}")
    print(f"    - CRT1 Buy Signals: {len(signals_crt1['buy'])}")
    print(f"    - CRT1 Sell Signals: {len(signals_crt1['sell'])}")
    print(f"  CRT 2 (4H - Black):")
    print(f"    - Total CRT2 Ranges Detected: {len(crt2_ranges)}")
    print(f"    - Active CRT2 Ranges: {len([c for c in crt2_ranges if c.is_active])}")
    print(f"    - CRT2 Buy Signals: {len(signals_crt2['buy'])}")
    print(f"    - CRT2 Sell Signals: {len(signals_crt2['sell'])}")


if __name__ == "__main__":
    plot_crt_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500, lookback=20)
