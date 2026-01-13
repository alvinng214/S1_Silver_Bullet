"""
SW's Asia/London H/L's Chart
Standalone candlestick chart showing session highs and lows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load session levels module
spec = importlib.util.spec_from_file_location(
    "session_levels",
    "SW's AsiaLondon HL's.py"
)
session_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(session_module)

detect_session_levels = session_module.detect_session_levels
detect_pdh_pdl = session_module.detect_pdh_pdl


def plot_session_levels_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with session highs/lows and PDH/PDL

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate session levels on full dataset
    print("Calculating Asia/London/NY session levels...")
    session_levels = detect_session_levels(df, timezone_offset=0)

    print("Calculating PDH/PDL...")
    pdh_pdl_levels = detect_pdh_pdl(df)

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot candlesticks first
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

    # Plot session levels
    # Asia Session - Purple
    for level in session_levels['asia_high']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#4a148c', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' Asia High', fontsize=8,
                   color='#4a148c', va='bottom', ha='left', fontweight='bold', zorder=5)

    for level in session_levels['asia_low']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#4a148c', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' Asia Low', fontsize=8,
                   color='#4a148c', va='top', ha='left', fontweight='bold', zorder=5)

    # London Session - Blue
    for level in session_levels['london_high']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#0c3299', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' London High', fontsize=8,
                   color='#0c3299', va='bottom', ha='left', fontweight='bold', zorder=5)

    for level in session_levels['london_low']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#0c3299', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' London Low', fontsize=8,
                   color='#0c3299', va='top', ha='left', fontweight='bold', zorder=5)

    # NY Session - Orange
    for level in session_levels['ny_high']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#fb7f1f', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' NY High', fontsize=8,
                   color='#fb7f1f', va='bottom', ha='left', fontweight='bold', zorder=5)

    for level in session_levels['ny_low']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#fb7f1f', linewidth=2, linestyle='-', alpha=0.8, zorder=3)
            # Add label
            ax.text(x_end, level.price, ' NY Low', fontsize=8,
                   color='#fb7f1f', va='top', ha='left', fontweight='bold', zorder=5)

    # PDH/PDL - Dark Green
    for level in pdh_pdl_levels['pdh']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#00332a', linewidth=2.5, linestyle='-', alpha=0.9, zorder=4)
            # Add label
            ax.text(x_end, level.price, ' PDH', fontsize=9,
                   color='#00332a', va='bottom', ha='left', fontweight='bold', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    for level in pdh_pdl_levels['pdl']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#00332a', linewidth=2.5, linestyle='-', alpha=0.9, zorder=4)
            # Add label
            ax.text(x_end, level.price, ' PDL', fontsize=9,
                   color='#00332a', va='top', ha='left', fontweight='bold', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    for level in pdh_pdl_levels['pd_mid']:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#00332a', linewidth=1.5, linestyle='--', alpha=0.7, zorder=4)
            # Add label
            ax.text(x_end, level.price, ' PD Mid', fontsize=8,
                   color='#00332a', va='center', ha='left', fontweight='bold', zorder=5)

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title("SW's Asia/London H/L's - Session Levels (Last 500 Candles)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#4a148c', alpha=0.8, label='Asia Session'),
        mpatches.Patch(facecolor='#0c3299', alpha=0.8, label='London Session'),
        mpatches.Patch(facecolor='#fb7f1f', alpha=0.8, label='NY Session'),
        mpatches.Patch(facecolor='#00332a', alpha=0.9, label='PDH/PDL/Mid')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    output_file = "SW's AsiaLondon HL's.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSession levels chart saved as '{output_file}'")
    print(f"\nSession Levels Summary:")
    print(f"  - Asia Highs: {len(session_levels['asia_high'])}")
    print(f"  - Asia Lows: {len(session_levels['asia_low'])}")
    print(f"  - London Highs: {len(session_levels['london_high'])}")
    print(f"  - London Lows: {len(session_levels['london_low'])}")
    print(f"  - NY Highs: {len(session_levels['ny_high'])}")
    print(f"  - NY Lows: {len(session_levels['ny_low'])}")
    print(f"  - PDH: {len(pdh_pdl_levels['pdh'])}")
    print(f"  - PDL: {len(pdh_pdl_levels['pdl'])}")
    print(f"  - PD Mid: {len(pdh_pdl_levels['pd_mid'])}")


if __name__ == "__main__":
    plot_session_levels_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
