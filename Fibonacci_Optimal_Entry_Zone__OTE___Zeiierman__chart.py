"""
Fibonacci Optimal Entry Zone [OTE] Chart
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

detect_fibonacci_ote = fib_ote.detect_fibonacci_ote


def plot_fibonacci_ote_chart(
    csv_file,
    num_candles=500,
    pivot_period=10,
    fib_levels=None,
    show_bullish=True,
    show_bearish=True,
    swing_tracker=True,
    show_golden_zone=True,
    show_swing_lines=True
):
    """
    Plot candlestick chart with Fibonacci OTE indicator

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        pivot_period: Period for pivot detection
        fib_levels: List of Fibonacci levels to plot
        show_bullish: Show bullish structures
        show_bearish: Show bearish structures
        swing_tracker: Enable real-time swing tracking
        show_golden_zone: Fill golden zone between first two Fib levels
        show_swing_lines: Draw lines connecting swing highs/lows
    """
    if fib_levels is None:
        fib_levels = [0.50, 0.618]

    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Fibonacci OTE
    print(f"\nDetecting Fibonacci OTE...")
    fib_colors = ['#4CAF50', '#009688']  # Green shades for bullish

    results = detect_fibonacci_ote(
        df,
        pivot_period=pivot_period,
        fib_levels=fib_levels,
        fib_colors=fib_colors,
        show_bullish=show_bullish,
        show_bearish=show_bearish,
        swing_tracker=swing_tracker
    )

    structures = results['structures']
    pivot_highs = results['pivot_highs']
    pivot_lows = results['pivot_lows']

    print(f"\nVisualization Summary:")
    print(f"  Structures: {len(structures)}")
    print(f"  Pivot Highs: {len(pivot_highs)}")
    print(f"  Pivot Lows: {len(pivot_lows)}")

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
        # Skip if structure is before display range
        if struct.swing_high.index < start_idx and struct.swing_low.index < start_idx:
            continue

        swing_high_idx = max(0, struct.swing_high.index - start_idx)
        swing_low_idx = max(0, struct.swing_low.index - start_idx)
        choch_idx = max(0, struct.choch_idx - start_idx)

        if swing_high_idx >= len(df_display) or swing_low_idx >= len(df_display):
            continue

        # Choose colors
        struct_color = '#08ec32' if struct.is_bullish else '#FF2222'

        # Draw CHoCH (Change of Character) marker
        if 0 <= choch_idx < len(df_display):
            if struct.is_bullish:
                ax.annotate(
                    'CHoCH',
                    xy=(choch_idx, struct.choch_price),
                    xytext=(choch_idx, struct.choch_price + 10),
                    fontsize=8,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=struct_color, alpha=0.8),
                    arrowprops=dict(arrowstyle='-', color=struct_color, lw=1.5),
                    zorder=10
                )
                # Draw BoS line
                ax.axhline(y=struct.choch_price, xmin=choch_idx/len(df_display),
                          xmax=(choch_idx+20)/len(df_display), color=struct_color,
                          linewidth=2, alpha=0.7, zorder=3)
            else:
                ax.annotate(
                    'CHoCH',
                    xy=(choch_idx, struct.choch_price),
                    xytext=(choch_idx, struct.choch_price - 10),
                    fontsize=8,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=struct_color, alpha=0.8),
                    arrowprops=dict(arrowstyle='-', color=struct_color, lw=1.5),
                    zorder=10
                )
                # Draw BoS line
                ax.axhline(y=struct.choch_price, xmin=choch_idx/len(df_display),
                          xmax=(choch_idx+20)/len(df_display), color=struct_color,
                          linewidth=2, alpha=0.7, zorder=3)

        # Draw swing line connecting high and low
        if show_swing_lines and 0 <= swing_high_idx < len(df_display) and 0 <= swing_low_idx < len(df_display):
            ax.plot(
                [swing_low_idx, swing_high_idx] if struct.is_bullish else [swing_high_idx, swing_low_idx],
                [struct.swing_low.price, struct.swing_high.price],
                color=struct_color,
                linewidth=2,
                linestyle=':',
                alpha=0.6,
                zorder=1
            )

        # Draw swing labels
        if 0 <= swing_high_idx < len(df_display):
            ax.scatter(swing_high_idx, struct.swing_high.price, marker='v',
                      s=80, color=struct_color, edgecolors='black', linewidth=1,
                      zorder=4, alpha=0.8)
            ax.text(swing_high_idx, struct.swing_high.price + 3, f'{struct.swing_high.price:.2f}',
                   fontsize=7, color=struct_color, ha='center', va='bottom',
                   fontweight='bold', zorder=5)

        if 0 <= swing_low_idx < len(df_display):
            ax.scatter(swing_low_idx, struct.swing_low.price, marker='^',
                      s=80, color=struct_color, edgecolors='black', linewidth=1,
                      zorder=4, alpha=0.8)
            ax.text(swing_low_idx, struct.swing_low.price - 3, f'{struct.swing_low.price:.2f}',
                   fontsize=7, color=struct_color, ha='center', va='top',
                   fontweight='bold', zorder=5)

        # Draw Fibonacci levels
        for fib_level in struct.fib_levels:
            if fib_level.start_idx < start_idx:
                continue

            fib_start = max(0, fib_level.start_idx - start_idx)
            fib_end = min(len(df_display) - 1, fib_level.end_idx - start_idx)

            if fib_end < 0 or fib_start >= len(df_display):
                continue

            # Extend to end of chart
            fib_end = len(df_display) - 1

            # Draw Fibonacci level line
            ax.plot(
                [fib_start, fib_end],
                [fib_level.price, fib_level.price],
                color=fib_level.color,
                linewidth=2,
                linestyle='-',
                alpha=0.8,
                zorder=2
            )

            # Add level label
            label_text = f'{fib_level.level:.3f} ({fib_level.price:.2f})'
            ax.text(
                fib_end - 5,
                fib_level.price,
                label_text,
                fontsize=7,
                color=fib_level.color,
                va='center',
                ha='right',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                zorder=5
            )

        # Fill golden zone
        if show_golden_zone and struct.golden_zone_top and struct.golden_zone_bottom:
            # Find start of golden zone
            gz_start = max(0, min(struct.swing_high.index, struct.swing_low.index) - start_idx)
            gz_end = len(df_display) - 1

            if 0 <= gz_start < len(df_display):
                golden_color = '#008080' if struct.is_bullish else '#FF6B6B'
                rect = Rectangle(
                    (gz_start - 0.5, struct.golden_zone_bottom),
                    gz_end - gz_start + 1,
                    struct.golden_zone_top - struct.golden_zone_bottom,
                    facecolor=golden_color,
                    alpha=0.15,
                    zorder=0.5
                )
                ax.add_patch(rect)

                # Add golden zone label
                mid_x = (gz_start + gz_end) / 2
                mid_y = (struct.golden_zone_top + struct.golden_zone_bottom) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    'Golden Zone',
                    fontsize=9,
                    color=golden_color,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    alpha=0.7,
                    zorder=5
                )

    # ========================================================================
    # PLOT PIVOT POINTS (subset for clarity)
    # ========================================================================
    for pivot in pivot_highs[-50:]:
        if pivot.index >= start_idx:
            pivot_idx = pivot.index - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(pivot_idx, pivot.price, marker='v', s=30,
                          color='gray', alpha=0.3, zorder=1)

    for pivot in pivot_lows[-50:]:
        if pivot.index >= start_idx:
            pivot_idx = pivot.index - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(pivot_idx, pivot.price, marker='^', s=30,
                          color='gray', alpha=0.3, zorder=1)

    # ========================================================================
    # FORMATTING
    # ========================================================================
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')

    bullish_count = len([s for s in structures if s.is_bullish])
    bearish_count = len([s for s in structures if not s.is_bullish])

    title = (f"Fibonacci Optimal Entry Zone [OTE] (Zeiierman) | "
            f"Period: {pivot_period} | {len(structures)} Structures "
            f"({bullish_count} Bull, {bearish_count} Bear)")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#08ec32', alpha=0.8, label='Bullish CHoCH'),
        mpatches.Patch(facecolor='#FF2222', alpha=0.8, label='Bearish CHoCH'),
        plt.Line2D([0], [0], color='#4CAF50', linewidth=2, label=f'Fib 0.50'),
        plt.Line2D([0], [0], color='#009688', linewidth=2, label=f'Fib 0.618'),
        mpatches.Patch(facecolor='#008080', alpha=0.15, label='Golden Zone (Bull)'),
        mpatches.Patch(facecolor='#FF6B6B', alpha=0.15, label='Golden Zone (Bear)'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#08ec32',
                  markeredgecolor='black', markersize=8, label='Swing High', linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#08ec32',
                  markeredgecolor='black', markersize=8, label='Swing Low', linestyle='None'),
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
    print(f"FIBONACCI OTE SUMMARY")
    print(f"{'='*70}")
    print(f"Market Structures: {len(structures)}")
    print(f"  - Bullish: {bullish_count}")
    print(f"  - Bearish: {bearish_count}")
    print(f"Fibonacci Levels: {fib_levels}")
    print(f"Pivot Period: {pivot_period}")
    print(f"Swing Tracker: {'Enabled' if swing_tracker else 'Disabled'}")
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
        show_golden_zone=True,
        show_swing_lines=True
    )
