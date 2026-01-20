"""
Monday Range (Lines) - Candlestick Chart with Indicator

Visualizes Monday OHLC ranges extended through the week with breakout markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

# Import the Monday Range detection module
from Monday_Range__Lines_ import detect_monday_ranges, LevelConfig, RangeLevel


def plot_monday_range_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with Monday Range levels.

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of recent candles to display
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Monday ranges on full dataset
    print("\nDetecting Monday Range levels...")

    custom_levels = [
        LevelConfig(enabled=True, label='EQ', value=0.5, color='#00BFFF',
                   line_style='dashed', line_width=1)
    ]

    monday_data = detect_monday_ranges(
        df,
        max_weeks=4,
        show_mh=True,
        show_ml=True,
        show_mo=False,
        show_mc=False,
        custom_levels=custom_levels,
        extension_type='end_of_week',
        track_breakouts=True,
        track_reclaims=True
    )

    mondays = monday_data['mondays']
    levels = monday_data['levels']
    breakouts = monday_data['breakouts']

    print(f"Detected {len(mondays)} Monday ranges")
    print(f"Generated {len(levels)} level lines")
    print(f"Detected {len(breakouts)} breakout/reclaim events")

    # Filter to display range
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display['display_index'] = range(len(df_display))

    # Adjust level indices for display
    levels_display = []
    for level in levels:
        if level.end_idx >= start_idx:
            level_display = RangeLevel(
                start_idx=max(0, level.start_idx - start_idx),
                end_idx=min(len(df_display) - 1, level.end_idx - start_idx),
                price=level.price,
                label=level.label,
                color=level.color,
                line_style=level.line_style,
                line_width=level.line_width,
                week_key=level.week_key
            )
            levels_display.append(level_display)

    # Adjust breakout indices
    breakouts_display = []
    for breakout in breakouts:
        if breakout.idx >= start_idx:
            breakout.idx = breakout.idx - start_idx
            breakouts_display.append(breakout)

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

    # Plot Monday Range levels
    print("Plotting Monday Range levels...")
    for level in levels_display:
        # Convert line style
        linestyle_map = {
            'solid': '-',
            'dashed': '--',
            'dotted': ':'
        }
        linestyle = linestyle_map.get(level.line_style, '-')

        # Draw level line
        ax.plot(
            [level.start_idx, level.end_idx],
            [level.price, level.price],
            color=level.color,
            linewidth=level.line_width,
            linestyle=linestyle,
            alpha=0.7,
            zorder=3
        )

        # Add label at end
        if level.label:
            ax.text(
                level.end_idx + 2,
                level.price,
                level.label,
                fontsize=8,
                color=level.color,
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                zorder=5
            )

    # Plot breakout markers
    print("Plotting breakout markers...")
    for breakout in breakouts_display:
        if 0 <= breakout.idx < len(df_display):
            if breakout.is_reclaim:
                # Reclaim markers
                if breakout.is_high_break:
                    # High reclaim (closed back below after going above)
                    marker = 'v'
                    color = '#9B59B6'  # Purple
                    label_text = 'H-Reclaim'
                else:
                    # Low reclaim (closed back above after going below)
                    marker = '^'
                    color = '#2ECC71'  # Green
                    label_text = 'L-Reclaim'
            else:
                # Breakout markers
                if breakout.is_high_break:
                    # High breakout
                    marker = '^'
                    color = '#D4A574'  # Gold
                    label_text = 'H-Break'
                else:
                    # Low breakout
                    marker = 'v'
                    color = '#3498DB'  # Blue
                    label_text = 'L-Break'

            ax.scatter(
                breakout.idx,
                breakout.price,
                marker=marker,
                s=150,
                color=color,
                edgecolors='black',
                linewidth=1.5,
                zorder=6,
                alpha=0.9
            )

    # Formatting
    step = max(1, len(df_display) // 10)
    tick_positions = list(range(0, len(df_display), step))
    tick_labels = [df_display.iloc[i].name.strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title('XAUUSD - Monday Range (Lines)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#007FFF', linewidth=2, linestyle='-', label='Monday High (MH)'),
        Line2D([0], [0], color='#007FFF', linewidth=2, linestyle='-', label='Monday Low (ML)'),
        Line2D([0], [0], color='#00BFFF', linewidth=2, linestyle='--', label='Equilibrium (EQ)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#D4A574',
               markersize=10, label='High Break', markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#3498DB',
               markersize=10, label='Low Break', markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='#9B59B6',
               markersize=10, label='High Reclaim', markeredgecolor='black', linestyle='None'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ECC71',
               markersize=10, label='Low Reclaim', markeredgecolor='black', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save
    output_file = 'Monday_Range__Lines_.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nMonday Range chart saved as '{output_file}'")

    plt.show()

    # Summary
    print("\n" + "=" * 70)
    print("MONDAY RANGE SUMMARY")
    print("=" * 70)
    print(f"Monday ranges displayed: {len(mondays)}")
    print(f"Total levels: {len(levels)}")
    print(f"Breakouts/Reclaims: {len(breakouts)}")

    # Count by type
    high_breaks = sum(1 for b in breakouts if b.is_high_break and not b.is_reclaim)
    low_breaks = sum(1 for b in breakouts if not b.is_high_break and not b.is_reclaim)
    high_reclaims = sum(1 for b in breakouts if b.is_high_break and b.is_reclaim)
    low_reclaims = sum(1 for b in breakouts if not b.is_high_break and b.is_reclaim)

    print(f"  - High breaks: {high_breaks}")
    print(f"  - Low breaks: {low_breaks}")
    print(f"  - High reclaims: {high_reclaims}")
    print(f"  - Low reclaims: {low_reclaims}")
    print("=" * 70)


if __name__ == '__main__':
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_monday_range_chart(csv_file, num_candles=500)
