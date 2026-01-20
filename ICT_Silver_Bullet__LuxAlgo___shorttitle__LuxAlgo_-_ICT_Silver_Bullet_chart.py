"""
ICT Silver Bullet [LuxAlgo] Chart
Candlestick chart showing LuxAlgo Silver Bullet sessions, FVGs with activation, and target lines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load LuxAlgo Silver Bullet module
spec = importlib.util.spec_from_file_location(
    "luxalgo_silver_bullet",
    "ICT_Silver_Bullet__LuxAlgo___shorttitle__LuxAlgo_-_ICT_Silver_Bullet.py"
)
luxalgo_sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(luxalgo_sb)

detect_luxalgo_silver_bullet = luxalgo_sb.detect_luxalgo_silver_bullet
is_in_silver_bullet_session = luxalgo_sb.is_in_silver_bullet_session


def plot_luxalgo_silver_bullet_chart(
    csv_file,
    num_candles=500,
    pivot_left=5,
    pivot_right=1,
    filter_mode='Super-Strict',
    extend_fvg=True,
    target_mode='previous session (similar)',
    keep_lines=True,
    show_session_backgrounds=True
):
    """
    Plot candlestick chart with LuxAlgo ICT Silver Bullet indicator

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        pivot_left: Left period for pivot detection
        pivot_right: Right period for pivot detection
        filter_mode: 'All FVG', 'Only FVG in the same direction of trend', 'Strict', 'Super-Strict'
        extend_fvg: Extend FVG boxes when active
        target_mode: 'previous session (any)' or 'previous session (similar)'
        keep_lines: Keep target lines between sessions
        show_session_backgrounds: Show session background colors
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Silver Bullet signals
    print(f"\nDetecting LuxAlgo Silver Bullet signals...")
    results = detect_luxalgo_silver_bullet(
        df,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        filter_mode=filter_mode,
        extend_fvg=extend_fvg,
        target_mode=target_mode,
        keep_lines=keep_lines
    )

    bullish_fvgs = results['bullish_fvgs']
    bearish_fvgs = results['bearish_fvgs']
    active_bull_fvgs = results['active_bull_fvgs']
    active_bear_fvgs = results['active_bear_fvgs']
    all_targets = results['all_targets']
    active_targets = results['active_targets']
    pivot_highs = results['pivot_highs']
    pivot_lows = results['pivot_lows']

    print(f"\nVisualization Summary:")
    print(f"  Bullish FVGs: {len(active_bull_fvgs)} active / {len(bullish_fvgs)} total")
    print(f"  Bearish FVGs: {len(active_bear_fvgs)} active / {len(bearish_fvgs)} total")
    print(f"  Target lines: {len(active_targets)} active / {len(all_targets)} total")

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 14))

    # ========================================================================
    # PLOT SESSION BACKGROUNDS
    # ========================================================================
    if show_session_backgrounds:
        session_colors = {
            'LN': '#e8f5e9',  # Very light green for London
            'AM': '#e3f2fd',  # Very light blue for AM
            'PM': '#fff3e0'   # Very light orange for PM
        }

        # Track sessions in display range
        current_session = None
        session_start = None

        for idx in range(len(df_display)):
            row = df_display.iloc[idx]
            dt = row['datetime']

            in_session, sess_name = is_in_silver_bullet_session(dt)

            # Session started
            if in_session and current_session != sess_name:
                session_start = idx
                current_session = sess_name

            # Session ended
            if not in_session and current_session is not None:
                # Draw session background
                y_min = df_display['low'].min()
                y_max = df_display['high'].max()

                rect = Rectangle(
                    (session_start - 0.5, y_min),
                    idx - session_start,
                    y_max - y_min,
                    facecolor=session_colors.get(current_session, '#f5f5f5'),
                    alpha=0.2,
                    zorder=0
                )
                ax.add_patch(rect)

                # Add session label
                mid_x = session_start + (idx - session_start) / 2
                label_map = {
                    'LN': 'London Open\n3-4 AM NY',
                    'AM': 'AM Session\n10-11 AM NY',
                    'PM': 'PM Session\n2-3 PM NY'
                }
                ax.text(
                    mid_x,
                    y_max * 1.001,
                    label_map.get(current_session, current_session),
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    zorder=5
                )

                current_session = None
                session_start = None

        # Handle last session if still active
        if current_session is not None:
            y_min = df_display['low'].min()
            y_max = df_display['high'].max()

            rect = Rectangle(
                (session_start - 0.5, y_min),
                len(df_display) - session_start,
                y_max - y_min,
                facecolor=session_colors.get(current_session, '#f5f5f5'),
                alpha=0.2,
                zorder=0
            )
            ax.add_patch(rect)

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
    # PLOT FVGs (Fair Value Gaps)
    # ========================================================================
    # Plot all FVGs with different opacity for active vs inactive
    for fvg in bullish_fvgs:
        if fvg.end_idx < start_idx:
            continue

        fvg_start = max(0, fvg.start_idx - start_idx)
        fvg_end = min(len(df_display) - 1, fvg.end_idx - start_idx)

        if fvg_end < 0 or fvg_start >= len(df_display):
            continue

        # Choose color and alpha based on activation status
        if fvg.active:
            fvg_color = '#4dd0e1'  # Cyan - matches LuxAlgo
            alpha = 0.4
            edge_width = 1.5
        else:
            fvg_color = '#4dd0e1'
            alpha = 0.1
            edge_width = 0.5

        # Draw FVG box
        fvg_height = abs(fvg.top - fvg.bottom)
        rect = Rectangle(
            (fvg_start - 0.5, fvg.bottom),
            fvg_end - fvg_start + 1,
            fvg_height,
            facecolor=fvg_color,
            edgecolor=fvg_color,
            alpha=alpha,
            linewidth=edge_width,
            zorder=1.5
        )
        ax.add_patch(rect)

        # Add label for active FVGs
        if fvg.active:
            ax.text(
                fvg_end,
                (fvg.top + fvg.bottom) / 2,
                ' Bull FVG',
                fontsize=7,
                color=fvg_color,
                va='center',
                ha='left',
                fontweight='bold',
                zorder=5
            )

    for fvg in bearish_fvgs:
        if fvg.end_idx < start_idx:
            continue

        fvg_start = max(0, fvg.start_idx - start_idx)
        fvg_end = min(len(df_display) - 1, fvg.end_idx - start_idx)

        if fvg_end < 0 or fvg_start >= len(df_display):
            continue

        # Choose color and alpha based on activation status
        if fvg.active:
            fvg_color = '#ffc1b1'  # Salmon/coral - matches LuxAlgo
            alpha = 0.4
            edge_width = 1.5
        else:
            fvg_color = '#ffc1b1'
            alpha = 0.1
            edge_width = 0.5

        # Draw FVG box
        fvg_height = abs(fvg.top - fvg.bottom)
        rect = Rectangle(
            (fvg_start - 0.5, fvg.bottom),
            fvg_end - fvg_start + 1,
            fvg_height,
            facecolor=fvg_color,
            edgecolor=fvg_color,
            alpha=alpha,
            linewidth=edge_width,
            zorder=1.5
        )
        ax.add_patch(rect)

        # Add label for active FVGs
        if fvg.active:
            ax.text(
                fvg_end,
                (fvg.top + fvg.bottom) / 2,
                ' Bear FVG',
                fontsize=7,
                color=fvg_color,
                va='center',
                ha='left',
                fontweight='bold',
                zorder=5
            )

    # ========================================================================
    # PLOT TARGET LINES
    # ========================================================================
    # Only show active target lines
    for target in active_targets:
        if target.source_idx < start_idx:
            continue

        target_start = max(0, target.source_idx - start_idx)
        target_end = len(df_display) - 1

        # Resistance targets (from highs) - blue
        # Support targets (from lows) - red
        target_color = '#3e89fa' if target.is_resistance else '#b22833'

        ax.plot(
            [target_start, target_end],
            [target.price, target.price],
            color=target_color,
            linewidth=1.5,
            linestyle='--',
            alpha=0.7,
            zorder=3
        )

        # Add small dot at target hits
        # Check if target was reached in display range
        for i in range(target_start, len(df_display)):
            row = df_display.iloc[i]
            if target.is_resistance and row['high'] > target.price:
                ax.scatter(i, target.price, marker='o', s=30, color=target_color,
                          edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
                break
            elif not target.is_resistance and row['low'] < target.price:
                ax.scatter(i, target.price, marker='o', s=30, color=target_color,
                          edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
                break

    # ========================================================================
    # PLOT PIVOT POINTS (optional, show subset)
    # ========================================================================
    # Show last 30 pivots
    for pivot in pivot_highs[-30:]:
        if pivot['index'] >= start_idx:
            pivot_idx = pivot['index'] - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(
                    pivot_idx,
                    pivot['price'],
                    marker='v',
                    s=40,
                    color='gray',
                    alpha=0.4,
                    zorder=4
                )

    for pivot in pivot_lows[-30:]:
        if pivot['index'] >= start_idx:
            pivot_idx = pivot['index'] - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(
                    pivot_idx,
                    pivot['price'],
                    marker='^',
                    s=40,
                    color='gray',
                    alpha=0.4,
                    zorder=4
                )

    # ========================================================================
    # FORMATTING
    # ========================================================================
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')

    title = (f"ICT Silver Bullet [LuxAlgo] - Mode: {filter_mode} | "
            f"{len(active_bull_fvgs)} Bull FVGs | {len(active_bear_fvgs)} Bear FVGs | "
            f"{len(active_targets)} Targets")
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#e8f5e9', alpha=0.3, label='London Open (3-4 AM)'),
        mpatches.Patch(facecolor='#e3f2fd', alpha=0.3, label='AM Session (10-11 AM)'),
        mpatches.Patch(facecolor='#fff3e0', alpha=0.3, label='PM Session (2-3 PM)'),
        mpatches.Patch(facecolor='#4dd0e1', alpha=0.4, edgecolor='#4dd0e1',
                      linewidth=1.5, label='Bullish FVG (Active)'),
        mpatches.Patch(facecolor='#ffc1b1', alpha=0.4, edgecolor='#ffc1b1',
                      linewidth=1.5, label='Bearish FVG (Active)'),
        mpatches.Patch(facecolor='#4dd0e1', alpha=0.1, edgecolor='#4dd0e1',
                      linewidth=0.5, label='Bullish FVG (Inactive)'),
        mpatches.Patch(facecolor='#ffc1b1', alpha=0.1, edgecolor='#ffc1b1',
                      linewidth=0.5, label='Bearish FVG (Inactive)'),
        plt.Line2D([0], [0], color='#3e89fa', linewidth=1.5, linestyle='--',
                  label='Resistance Target'),
        plt.Line2D([0], [0], color='#b22833', linewidth=1.5, linestyle='--',
                  label='Support Target'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                  markersize=7, label='Pivot High', alpha=0.4, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                  markersize=7, label='Pivot Low', alpha=0.4, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3e89fa',
                  markeredgecolor='black', markersize=6, label='Target Hit', linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    output_file = "ICT_Silver_Bullet__LuxAlgo___shorttitle__LuxAlgo_-_ICT_Silver_Bullet.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nLuxAlgo Silver Bullet chart saved as '{output_file}'")

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Filter Mode: {filter_mode}")
    print(f"Total Bullish FVGs detected: {len(bullish_fvgs)}")
    print(f"  - Active: {len(active_bull_fvgs)}")
    print(f"  - Inactive: {len(bullish_fvgs) - len(active_bull_fvgs)}")
    print(f"Total Bearish FVGs detected: {len(bearish_fvgs)}")
    print(f"  - Active: {len(active_bear_fvgs)}")
    print(f"  - Inactive: {len(bearish_fvgs) - len(active_bear_fvgs)}")
    print(f"Total Target Lines: {len(all_targets)}")
    print(f"  - Active: {len(active_targets)}")
    print(f"  - Reached: {len([t for t in all_targets if t.reached])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    plot_luxalgo_silver_bullet_chart(
        "PEPPERSTONE_XAUUSD, 5.csv",
        num_candles=500,
        pivot_left=5,
        pivot_right=1,
        filter_mode='Super-Strict',
        extend_fvg=True,
        target_mode='previous session (similar)',
        keep_lines=True,
        show_session_backgrounds=True
    )
