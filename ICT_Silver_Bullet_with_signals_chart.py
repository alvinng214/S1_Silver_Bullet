"""
ICT Silver Bullet with Signals Chart
Standalone candlestick chart showing Silver Bullet sessions, FVGs, and target lines
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load ICT Silver Bullet module
spec = importlib.util.spec_from_file_location(
    "silver_bullet",
    "ICT_Silver_Bullet_with_signals.py"
)
silver_bullet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(silver_bullet)

detect_silver_bullet_signals = silver_bullet.detect_silver_bullet_signals


def plot_silver_bullet_chart(csv_file, num_candles=500, htf_minutes=15, filter_by_trend=True):
    """
    Plot candlestick chart with ICT Silver Bullet sessions, FVGs, and target lines

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        htf_minutes: HTF timeframe for FVG detection (default 15)
        filter_by_trend: Filter FVGs by market structure trend
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Silver Bullet signals
    print(f"Detecting Silver Bullet sessions and FVGs (HTF: {htf_minutes}min)...")
    results = detect_silver_bullet_signals(
        df,
        htf_minutes=htf_minutes,
        filter_by_trend=filter_by_trend,
        minimum_trade_framework=0
    )

    sessions = results['sessions']
    pivot_highs = results['pivot_highs']
    pivot_lows = results['pivot_lows']

    print(f"Found {len(sessions)} Silver Bullet sessions")

    # Count FVGs
    total_fvgs = sum(len(s.fvgs) for s in sessions)
    active_fvgs = sum(len([f for f in s.fvgs if f.active]) for s in sessions)
    print(f"Detected {total_fvgs} FVGs ({active_fvgs} active)")

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(22, 14))

    # ========================================================================
    # PLOT SESSION BACKGROUNDS
    # ========================================================================
    session_colors = {
        'LN': '#e3f2fd',  # Light blue for London
        'AM': '#fff3e0',  # Light orange for AM
        'PM': '#f3e5f5'   # Light purple for PM
    }

    session_names = {
        'LN': 'London Open (3-4 AM NY)',
        'AM': 'AM Session (10-11 AM NY)',
        'PM': 'PM Session (2-3 PM NY)'
    }

    for session in sessions:
        if session.end_idx < start_idx:
            continue

        sess_start = max(0, session.start_idx - start_idx)
        sess_end = min(len(df_display) - 1, session.end_idx - start_idx)

        if sess_end < 0 or sess_start >= len(df_display):
            continue

        # Get y-range for background
        y_min = df_display['low'].min()
        y_max = df_display['high'].max()

        # Draw session background
        rect = Rectangle(
            (sess_start - 0.5, y_min),
            sess_end - sess_start + 1,
            y_max - y_min,
            facecolor=session_colors.get(session.name, '#f5f5f5'),
            alpha=0.3,
            zorder=0
        )
        ax.add_patch(rect)

        # Add session label at top
        ax.text(
            sess_start + (sess_end - sess_start) / 2,
            y_max * 1.002,
            session_names.get(session.name, session.name),
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='bottom',
            zorder=5
        )

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
    for session in sessions:
        for fvg in session.fvgs:
            # Check if FVG is in visible range
            if fvg.end_idx < start_idx:
                continue

            fvg_start = max(0, fvg.start_idx - start_idx)
            # If active, extend to end of chart; otherwise use end_idx
            if fvg.active and not fvg.broken:
                fvg_end = len(df_display) - 1
            else:
                fvg_end = min(len(df_display) - 1, fvg.end_idx - start_idx + 20)

            if fvg_end < 0 or fvg_start >= len(df_display):
                continue

            # Choose color based on type and status
            if fvg.is_bullish:
                if fvg.active:
                    fvg_color = 'green'
                    alpha = 0.3
                else:
                    fvg_color = 'lightgreen'
                    alpha = 0.15
            else:
                if fvg.active:
                    fvg_color = 'red'
                    alpha = 0.3
                else:
                    fvg_color = 'lightcoral'
                    alpha = 0.15

            # Draw FVG box
            fvg_height = abs(fvg.top - fvg.bottom)
            rect = Rectangle(
                (fvg_start - 0.5, fvg.bottom),
                fvg_end - fvg_start + 1,
                fvg_height,
                facecolor=fvg_color,
                edgecolor=fvg_color,
                alpha=alpha,
                linewidth=1.5,
                zorder=1.5
            )
            ax.add_patch(rect)

            # Add FVG label
            label = f"{'Bull' if fvg.is_bullish else 'Bear'} FVG"
            if fvg.active:
                label += " (Active)"

            ax.text(
                fvg_end,
                (fvg.top + fvg.bottom) / 2,
                f' {label}',
                fontsize=7,
                color=fvg_color if fvg.active else 'gray',
                va='center',
                ha='left',
                fontweight='bold' if fvg.active else 'normal',
                zorder=5
            )

            # ========================================================================
            # PLOT TARGET LINES
            # ========================================================================
            if fvg.targets and fvg.active:
                for target in fvg.targets[:3]:  # Show max 3 targets per FVG
                    if not target.reached:
                        target_start = max(0, fvg.end_idx - start_idx)
                        target_end = len(df_display) - 1

                        target_color = 'darkgreen' if fvg.is_bullish else 'darkred'

                        ax.plot(
                            [target_start, target_end],
                            [target.price, target.price],
                            color=target_color,
                            linewidth=1.2,
                            linestyle='--',
                            alpha=0.6,
                            zorder=3
                        )

                        # Add target label
                        ax.text(
                            target_end,
                            target.price,
                            f' Target',
                            fontsize=6,
                            color=target_color,
                            va='center',
                            ha='left',
                            zorder=5
                        )

    # ========================================================================
    # PLOT PIVOT POINTS (optional, for reference)
    # ========================================================================
    # Show last 20 pivots
    for pivot in pivot_highs[-20:]:
        if pivot['index'] >= start_idx:
            pivot_idx = pivot['index'] - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(
                    pivot_idx,
                    pivot['price'],
                    marker='v',
                    s=50,
                    color='purple',
                    alpha=0.5,
                    zorder=4
                )

    for pivot in pivot_lows[-20:]:
        if pivot['index'] >= start_idx:
            pivot_idx = pivot['index'] - start_idx
            if 0 <= pivot_idx < len(df_display):
                ax.scatter(
                    pivot_idx,
                    pivot['price'],
                    marker='^',
                    s=50,
                    color='purple',
                    alpha=0.5,
                    zorder=4
                )

    # ========================================================================
    # FORMATTING
    # ========================================================================
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title(
        f"ICT Silver Bullet with Signals - {htf_minutes}min HTF | "
        f"{len(sessions)} Sessions | {total_fvgs} FVGs ({active_fvgs} Active)",
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#e3f2fd', alpha=0.5, label='London Open (3-4 AM)'),
        mpatches.Patch(facecolor='#fff3e0', alpha=0.5, label='AM Session (10-11 AM)'),
        mpatches.Patch(facecolor='#f3e5f5', alpha=0.5, label='PM Session (2-3 PM)'),
        mpatches.Patch(facecolor='green', alpha=0.3, edgecolor='green', label='Bullish FVG (Active)'),
        mpatches.Patch(facecolor='red', alpha=0.3, edgecolor='red', label='Bearish FVG (Active)'),
        plt.Line2D([0], [0], color='darkgreen', linewidth=1.2, linestyle='--', label='Bull Targets'),
        plt.Line2D([0], [0], color='darkred', linewidth=1.2, linestyle='--', label='Bear Targets'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='purple',
                  markersize=7, label='Pivot High', alpha=0.5, linestyle='None'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='purple',
                  markersize=7, label='Pivot Low', alpha=0.5, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    output_file = "ICT_Silver_Bullet_with_signals.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nICT Silver Bullet chart saved as '{output_file}'")
    print(f"\nSession Summary:")
    for session in sessions:
        print(f"  {session_names.get(session.name, session.name)}:")
        print(f"    - Start: {session.start_time}")
        print(f"    - FVGs: {len(session.fvgs)}")
        bullish_fvgs = [f for f in session.fvgs if f.is_bullish]
        bearish_fvgs = [f for f in session.fvgs if not f.is_bullish]
        active_fvgs_session = [f for f in session.fvgs if f.active]
        print(f"    - Bullish FVGs: {len(bullish_fvgs)}")
        print(f"    - Bearish FVGs: {len(bearish_fvgs)}")
        print(f"    - Active FVGs: {len(active_fvgs_session)}")


if __name__ == "__main__":
    plot_silver_bullet_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500, htf_minutes=15)
