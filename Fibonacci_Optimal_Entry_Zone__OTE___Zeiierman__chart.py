"""
Fibonacci Optimal Entry Zone [OTE] Chart - Accurate Translation
Candlestick chart showing market structure with Fibonacci retracement levels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load Fibonacci OTE module
spec = importlib.util.spec_from_file_location(
    "fib_ote",
    "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py"
)
fib_ote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fib_ote)

detect_fibonacci_ote_accurate = fib_ote.detect_fibonacci_ote_accurate


def plot_fibonacci_ote_chart(
    csv_file,
    num_candles=500,
    pivot_period=10,
    fib_levels=None,
    show_bullish=True,
    show_bearish=True,
    swing_tracker=True,
    show_old=False,
    extend=True,
    show_golden_zone=True,
    show_swing_lines=True,
    show_swing_labels=True
):
    """
    Plot candlestick chart with accurate Fibonacci OTE indicator

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        pivot_period: Period for pivot detection
        fib_levels: List of Fibonacci levels to plot
        show_bullish: Show bullish structures
        show_bearish: Show bearish structures
        swing_tracker: Follow mode - update levels dynamically
        show_old: Keep old structures
        extend: Extend Fibonacci lines to current bar
        show_golden_zone: Fill golden zone between first two Fib levels
        show_swing_lines: Draw lines connecting swing highs/lows
        show_swing_labels: Show price labels at swings
    """
    if fib_levels is None:
        fib_levels = [0.50, 0.618]

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Fibonacci OTE
    print(f"\nDetecting Fibonacci OTE (accurate translation)...")
    fib_colors = ['#4CAF50', '#009688']

    results = detect_fibonacci_ote_accurate(
        df,
        pivot_period=pivot_period,
        fib_levels=fib_levels,
        fib_colors=fib_colors,
        show_bullish=show_bullish,
        show_bearish=show_bearish,
        swing_tracker=swing_tracker,
        show_old=show_old,
        extend=extend
    )

    structures = results['structures']
    pivot_highs = results['pivot_highs']
    pivot_lows = results['pivot_lows']

    print(f"\nVisualization Summary:")
    print(f"  Structures: {len(structures)}")

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 14))

    # ========================================================================
    # PLOT CANDLESTICKS
    # ========================================================================
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

    # ========================================================================
    # PLOT STRUCTURES WITH FIBONACCI LEVELS
    # ========================================================================
    for struct in structures:
        # Adjust indices for display
        swing_high_idx = struct.swing_high_idx - start_idx
        swing_low_idx = struct.swing_low_idx - start_idx
        choch_idx = struct.choch_idx - start_idx
        trend_start_idx = struct.trend_line_start_idx - start_idx
        trend_end_idx = struct.trend_line_end_idx - start_idx

        # Skip if completely outside visible range
        if swing_high_idx < -50 and swing_low_idx < -50:
            continue

        struct_color = '#08ec32' if struct.is_bullish else '#FF2222'

        # Draw CHoCH (Change of Character) marker
        if 0 <= choch_idx < len(df_display):
            choch_y = struct.choch_price
            if struct.is_bullish:
                ax.annotate(
                    'CHoCH',
                    xy=(choch_idx, choch_y),
                    xytext=(choch_idx, choch_y + 15),
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=struct_color, alpha=0.9),
                    arrowprops=dict(arrowstyle='-', color=struct_color, lw=2),
                    zorder=10
                )
                # Draw BoS line
                bos_start = max(0, choch_idx - 10)
                bos_end = min(len(df_display) - 1, choch_idx + 10)
                ax.axhline(y=choch_y, xmin=bos_start/len(df_display),
                          xmax=bos_end/len(df_display), color=struct_color,
                          linewidth=2, alpha=0.8, zorder=3)
            else:
                ax.annotate(
                    'CHoCH',
                    xy=(choch_idx, choch_y),
                    xytext=(choch_idx, choch_y - 15),
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=struct_color, alpha=0.9),
                    arrowprops=dict(arrowstyle='-', color=struct_color, lw=2),
                    zorder=10
                )
                # Draw BoS line
                bos_start = max(0, choch_idx - 10)
                bos_end = min(len(df_display) - 1, choch_idx + 10)
                ax.axhline(y=choch_y, xmin=bos_start/len(df_display),
                          xmax=bos_end/len(df_display), color=struct_color,
                          linewidth=2, alpha=0.8, zorder=3)

        # Draw trend line connecting swing points
        if show_swing_lines:
            if 0 <= trend_start_idx < len(df_display) or 0 <= trend_end_idx < len(df_display):
                line_start = max(0, trend_start_idx)
                line_end = min(len(df_display) - 1, trend_end_idx)

                ax.plot(
                    [line_start, line_end],
                    [struct.trend_line_start_price, struct.trend_line_end_price],
                    color=struct_color,
                    linewidth=2,
                    linestyle=':',
                    alpha=0.7,
                    zorder=1.5
                )

        # Draw swing point markers and labels
        if show_swing_labels:
            if 0 <= swing_high_idx < len(df_display):
                ax.scatter(swing_high_idx, struct.swing_high, marker='v',
                          s=100, color=struct_color, edgecolors='black', linewidth=1.5,
                          zorder=4, alpha=0.9)
                ax.text(swing_high_idx, struct.swing_high + 5, f'{struct.swing_high:.2f}',
                       fontsize=8, color=struct_color, ha='center', va='bottom',
                       fontweight='bold', zorder=5,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            if 0 <= swing_low_idx < len(df_display):
                ax.scatter(swing_low_idx, struct.swing_low, marker='^',
                          s=100, color=struct_color, edgecolors='black', linewidth=1.5,
                          zorder=4, alpha=0.9)
                ax.text(swing_low_idx, struct.swing_low - 5, f'{struct.swing_low:.2f}',
                       fontsize=8, color=struct_color, ha='center', va='top',
                       fontweight='bold', zorder=5,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Draw Fibonacci levels
        for fib_line in struct.fib_lines:
            fib_start = max(0, fib_line.start_idx - start_idx)
            fib_end = min(len(df_display) - 1, fib_line.end_idx - start_idx)

            if fib_end < 0:
                continue

            # Ensure we have valid indices
            fib_start = max(0, fib_start)
            fib_end = max(fib_start, fib_end)

            # Draw Fibonacci level line
            ax.plot(
                [fib_start, fib_end],
                [fib_line.price, fib_line.price],
                color=fib_line.color,
                linewidth=2.5,
                linestyle='-',
                alpha=0.9,
                zorder=2.5
            )

            # Add level label at the end
            if fib_end >= fib_start + 5:
                label_text = f'{fib_line.level:.3f}'
                ax.text(
                    fib_end - 3,
                    fib_line.price,
                    label_text,
                    fontsize=8,
                    color=fib_line.color,
                    va='center',
                    ha='right',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9),
                    zorder=5
                )

        # Fill golden zone
        if show_golden_zone and len(struct.fib_lines) >= 2:
            fib_top = max(struct.fib_lines[0].price, struct.fib_lines[1].price)
            fib_bottom = min(struct.fib_lines[0].price, struct.fib_lines[1].price)

            gz_start = max(0, min(struct.swing_high_idx, struct.swing_low_idx) - start_idx)
            gz_end = max(0, max(struct.fib_lines[0].end_idx, struct.fib_lines[1].end_idx) - start_idx)
            gz_end = min(len(df_display) - 1, gz_end)

            if gz_end > gz_start:
                golden_color = '#008080' if struct.is_bullish else '#FF6B6B'
                rect = Rectangle(
                    (gz_start - 0.5, fib_bottom),
                    gz_end - gz_start + 1,
                    fib_top - fib_bottom,
                    facecolor=golden_color,
                    alpha=0.2,
                    zorder=0.5
                )
                ax.add_patch(rect)

                # Add golden zone label
                mid_x = (gz_start + gz_end) / 2
                mid_y = (fib_top + fib_bottom) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    'Golden Zone\n(OTE)',
                    fontsize=10,
                    color=golden_color,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    alpha=0.8,
                    zorder=5
                )

    # ========================================================================
    # PLOT PIVOT POINTS (subset for reference)
    # ========================================================================
    for i in range(start_idx, min(len(df), start_idx + num_candles)):
        if not np.isnan(pivot_highs[i]):
            pivot_idx = i - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(pivot_idx, pivot_highs[i], marker='v', s=25,
                          color='gray', alpha=0.3, zorder=1)

        if not np.isnan(pivot_lows[i]):
            pivot_idx = i - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(pivot_idx, pivot_lows[i], marker='^', s=25,
                          color='gray', alpha=0.3, zorder=1)

    # ========================================================================
    # FORMATTING
    # ========================================================================
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')

    bullish_count = len([s for s in structures if s.is_bullish])
    bearish_count = len([s for s in structures if not s.is_bullish])

    title = (f"Fibonacci Optimal Entry Zone [OTE] (Zeiierman) - Accurate Translation\n"
            f"Period: {pivot_period} | {len(structures)} Structures "
            f"({bullish_count} Bullish, {bearish_count} Bearish) | "
            f"Swing Tracker: {'On' if swing_tracker else 'Off'}")
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#08ec32', alpha=0.9, label='Bullish CHoCH'),
        mpatches.Patch(facecolor='#FF2222', alpha=0.9, label='Bearish CHoCH'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2.5, label='Fib 0.50'),
        plt.Line2D([0], [0], color='#009688', linewidth=2.5, label='Fib 0.618'),
        mpatches.Patch(facecolor='#008080', alpha=0.2, label='Golden Zone (Bull)'),
        mpatches.Patch(facecolor='#FF6B6B', alpha=0.2, label='Golden Zone (Bear)'),
        plt.Line2D([0], [0], color='#08ec32', linewidth=2, linestyle=':', label='Swing Line'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#08ec32',
                  markeredgecolor='black', markersize=10, label='Swing High', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#08ec32',
                  markeredgecolor='black', markersize=10, label='Swing Low', linestyle='None'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                  markersize=5, label='Pivot High', alpha=0.3, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                  markersize=5, label='Pivot Low', alpha=0.3, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    output_file = "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nFibonacci OTE chart saved as '{output_file}'")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FIBONACCI OTE SUMMARY (ACCURATE TRANSLATION)")
    print(f"{'='*70}")
    print(f"Market Structures: {len(structures)}")
    print(f"  - Bullish: {bullish_count}")
    print(f"  - Bearish: {bearish_count}")
    print(f"Fibonacci Levels: {fib_levels}")
    print(f"Pivot Period: {pivot_period}")
    print(f"Swing Tracker: {'Enabled' if swing_tracker else 'Disabled'}")
    print(f"Show Old Structures: {'Yes' if show_old else 'No'}")
    print(f"Extend Lines: {'Yes' if extend else 'No'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    plot_fibonacci_ote_chart(
        "PEPPERSTONE_XAUUSD, 5.csv",
        num_candles=500,
        pivot_period=10,
        fib_levels=[0.50, 0.618],
        show_bullish=True,
        show_bearish=True,
        swing_tracker=True,
        show_old=False,
        extend=True,
        show_golden_zone=True,
        show_swing_lines=True,
        show_swing_labels=True
    )
