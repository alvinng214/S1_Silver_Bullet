"""
cd_sweep&cisd_Cx - Candlestick Chart with HTF Sweeps and CISD

Visualizes HTF boxes, sweep areas, CISD levels, and signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import sys

# Import the cd_sweep&cisd detection module
# Note: Python doesn't allow & in module names, so we import as filename
import importlib.util
import os

spec = importlib.util.spec_from_file_location("cd_sweep_cisd", "cd_sweep&cisd_Cx.py")
cd_sweep_cisd_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd_sweep_cisd_module)
detect_cd_sweep_cisd = cd_sweep_cisd_module.detect_cd_sweep_cisd


def plot_cd_sweep_cisd_chart(csv_file, num_candles=500, htf_minutes=None):
    """
    Plot candlestick chart with cd_sweep&cisd indicator.

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of recent candles to display
        htf_minutes: Higher timeframe in minutes (auto if None)
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect cd_sweep&cisd on full dataset
    print("\nDetecting HTF sweeps and CISD signals...")
    cisd_data = detect_cd_sweep_cisd(
        df,
        htf_minutes=htf_minutes,
        show_htf_boxes=True,
        show_sweep_boxes=True,
        show_cisd=True
    )

    htf_candles = cisd_data['htf_candles']
    sweep_boxes = cisd_data['sweep_boxes']
    cisd_levels = cisd_data['cisd_levels']
    cisd_signals = cisd_data['cisd_signals']
    htf_mins = cisd_data['htf_minutes']

    print(f"HTF: {htf_mins} minutes")
    print(f"Detected {len(htf_candles)} HTF candles")
    print(f"Detected {len(sweep_boxes)} sweep boxes")
    print(f"Detected {len(cisd_levels)} CISD levels")
    print(f"Detected {len(cisd_signals)} CISD signals")

    # Filter to display range
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display['display_index'] = range(len(df_display))

    # Adjust HTF candles for display
    htf_candles_display = []
    for candle in htf_candles:
        if candle.end_idx >= start_idx:
            candle.start_idx = max(0, candle.start_idx - start_idx)
            candle.end_idx = min(len(df_display) - 1, candle.end_idx - start_idx)
            htf_candles_display.append(candle)

    # Adjust sweep boxes
    sweep_boxes_display = []
    for box in sweep_boxes:
        if box.end_idx >= start_idx:
            box.start_idx = max(0, box.start_idx - start_idx)
            box.end_idx = min(len(df_display) - 1, box.end_idx - start_idx)
            sweep_boxes_display.append(box)

    # Adjust CISD levels
    cisd_levels_display = []
    for level in cisd_levels:
        if level.idx >= start_idx:
            level.idx = level.idx - start_idx
            cisd_levels_display.append(level)

    # Adjust CISD signals
    cisd_signals_display = []
    for signal in cisd_signals:
        if signal.idx >= start_idx:
            signal.idx = signal.idx - start_idx
            signal.cisd_start_idx = max(0, signal.cisd_start_idx - start_idx)
            cisd_signals_display.append(signal)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot candlesticks
    print(f"\nPlotting {len(df_display)} candles...")
    width = 0.8
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
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
        ax.plot([x_pos, x_pos], [low_price, high_price], color='black', linewidth=1, zorder=1)

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
        ax.add_patch(rect)

    # Plot HTF boxes
    print("Plotting HTF boxes...")
    for candle in htf_candles_display[-10:]:  # Last 10 HTF candles
        if candle.end_idx < 0 or candle.start_idx >= len(df_display):
            continue

        # Determine color
        if candle.close > candle.open:
            border_color = '#089981'  # Teal
        elif candle.close < candle.open:
            border_color = '#F23645'  # Red
        else:
            border_color = '#1E3A8A'  # Navy (range)

        # Draw HTF box
        rect = Rectangle(
            (candle.start_idx - 0.5, candle.low),
            candle.end_idx - candle.start_idx + 1,
            candle.high - candle.low,
            facecolor='none',
            edgecolor=border_color,
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            zorder=3
        )
        ax.add_patch(rect)

    # Plot sweep boxes
    print("Plotting sweep boxes...")
    for box in sweep_boxes_display:
        if box.end_idx < 0 or box.start_idx >= len(df_display):
            continue

        # Sweep box (highlighted area)
        rect = Rectangle(
            (box.start_idx - 0.5, box.bottom),
            box.end_idx - box.start_idx + 1,
            box.top - box.bottom,
            facecolor='#ffeb3b',  # Yellow
            edgecolor='gray',
            linewidth=1,
            linestyle=':',
            alpha=0.2,
            zorder=2.5
        )
        ax.add_patch(rect)

    # Plot CISD levels
    print("Plotting CISD levels...")
    for level in cisd_levels_display:
        if 0 <= level.idx < len(df_display):
            # Draw CISD level line (short horizontal line)
            color = '#089981' if level.is_bullish else '#F23645'
            ax.plot(
                [level.idx, min(level.idx + 4, len(df_display) - 1)],
                [level.price, level.price],
                color=color,
                linewidth=2,
                linestyle=':',
                alpha=0.8,
                zorder=4
            )

    # Plot CISD signals
    print("Plotting CISD signals...")
    for signal in cisd_signals_display:
        if 0 <= signal.idx < len(df_display):
            # Draw box connecting CISD level to signal
            ax.plot(
                [signal.cisd_start_idx, signal.idx],
                [signal.cisd_price, signal.cisd_price],
                color='#089981' if signal.is_bullish else '#F23645',
                linewidth=1,
                alpha=0.6,
                zorder=3.5
            )

            # Add signal marker
            if signal.is_bullish:
                ax.scatter(
                    signal.idx,
                    df_display.iloc[signal.idx]['low'],
                    marker='^',
                    s=200,
                    color='#089981',
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=6,
                    alpha=0.9
                )
                # Add text label
                ax.text(
                    signal.idx,
                    df_display.iloc[signal.idx]['low'] - (df_display['high'].max() - df_display['low'].min()) * 0.01,
                    'CISD+',
                    fontsize=7,
                    color='#089981',
                    ha='center',
                    va='top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    zorder=7
                )
            else:
                ax.scatter(
                    signal.idx,
                    df_display.iloc[signal.idx]['high'],
                    marker='v',
                    s=200,
                    color='#F23645',
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=6,
                    alpha=0.9
                )
                # Add text label
                ax.text(
                    signal.idx,
                    df_display.iloc[signal.idx]['high'] + (df_display['high'].max() - df_display['low'].min()) * 0.01,
                    'CISD-',
                    fontsize=7,
                    color='#F23645',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                    zorder=7
                )

    # Formatting
    step = max(1, len(df_display) // 10)
    tick_positions = list(range(0, len(df_display), step))
    tick_labels = [df_display.iloc[i].name.strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(f'XAUUSD - cd_sweep&cisd_Cx (HTF: {htf_mins}min)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#089981', linewidth=2, linestyle='--', label='HTF Bull Box'),
        Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#F23645', linewidth=2, linestyle='--', label='HTF Bear Box'),
        Rectangle((0, 0), 1, 1, facecolor='#ffeb3b', edgecolor='gray', alpha=0.2, label='Sweep Area'),
        Line2D([0], [0], color='#089981', linewidth=2, linestyle=':', label='Bull CISD Level'),
        Line2D([0], [0], color='#F23645', linewidth=2, linestyle=':', label='Bear CISD Level'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#089981',
               markersize=10, label='Bull CISD Signal', markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#F23645',
               markersize=10, label='Bear CISD Signal', markeredgecolor='black', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = 'cd_sweep&cisd_Cx.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\ncd_sweep&cisd chart saved as '{output_file}'")

    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("CD_SWEEP&CISD SUMMARY")
    print("=" * 70)
    print(f"HTF Timeframe: {htf_mins} minutes")
    print(f"HTF Candles: {len(htf_candles)}")
    print(f"Sweep Boxes: {len(sweep_boxes)}")
    print(f"CISD Levels: {len(cisd_levels)}")
    print(f"  - Bullish: {sum(1 for l in cisd_levels if l.is_bullish)}")
    print(f"  - Bearish: {sum(1 for l in cisd_levels if not l.is_bullish)}")
    print(f"CISD Signals: {len(cisd_signals)}")
    print(f"  - Bullish: {sum(1 for s in cisd_signals if s.is_bullish)}")
    print(f"  - Bearish: {sum(1 for s in cisd_signals if not s.is_bullish)}")
    print("=" * 70)


if __name__ == '__main__':
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_cd_sweep_cisd_chart(csv_file, num_candles=500, htf_minutes=60)
