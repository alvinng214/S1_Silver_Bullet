"""
Silver Bullet ICT Strategy [TradingFinder] Chart
Candlestick chart showing 10-11 AM NY Time Silver Bullet setups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load TradingFinder Silver Bullet module
spec = importlib.util.spec_from_file_location(
    "tradingfinder_sb",
    "Silver_Bullet_ICT_Strategy__TradingFinder__10-11_AM_NY_Time__FVG_TFlab_Silver_Bullet.py"
)
tradingfinder_sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tradingfinder_sb)

detect_tradingfinder_silver_bullet = tradingfinder_sb.detect_tradingfinder_silver_bullet
is_in_ny_opening_range = tradingfinder_sb.is_in_ny_opening_range
is_in_ny_trading_time = tradingfinder_sb.is_in_ny_trading_time


def plot_tradingfinder_silver_bullet_chart(
    csv_file,
    num_candles=500,
    bar_back_check=5,
    show_order_blocks=True,
    show_fvgs=True,
    show_sessions=True
):
    """
    Plot candlestick chart with TradingFinder Silver Bullet indicator

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
        bar_back_check: Bars to look back for CISD level
        show_order_blocks: Show order blocks
        show_fvgs: Show FVGs
        show_sessions: Show session backgrounds
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Detect Silver Bullet signals
    print(f"\nDetecting TradingFinder Silver Bullet signals...")
    results = detect_tradingfinder_silver_bullet(
        df,
        bar_back_check=bar_back_check,
        show_order_blocks=show_order_blocks,
        show_fvgs=show_fvgs
    )

    sessions = results['sessions']
    signals = results['signals']
    order_blocks = results['order_blocks']
    fvgs = results['fvgs']
    cisd_levels = results['cisd_levels']

    print(f"\nVisualization Summary:")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Signals: {len(signals)} ({len([s for s in signals if s.signal_type == 'BULL'])} BULL, {len([s for s in signals if s.signal_type == 'BEAR'])} BEAR)")
    print(f"  Order Blocks: {len(order_blocks)}")
    print(f"  FVGs: {len(fvgs)}")
    print(f"  CISD Levels: {len(cisd_levels)}")

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
    if show_sessions:
        for idx in range(len(df_display)):
            row = df_display.iloc[idx]
            dt = row['datetime']

            in_opening = is_in_ny_opening_range(dt)
            in_trading = is_in_ny_trading_time(dt)

            y_min = df_display['low'].min()
            y_max = df_display['high'].max()

            # Opening Range: 9-10 AM (light yellow)
            if in_opening:
                rect = Rectangle(
                    (idx - 0.5, y_min),
                    1,
                    y_max - y_min,
                    facecolor='#fff9c4',
                    alpha=0.3,
                    zorder=0
                )
                ax.add_patch(rect)

            # Trading Time: 10-11 AM (light green)
            if in_trading:
                rect = Rectangle(
                    (idx - 0.5, y_min),
                    1,
                    y_max - y_min,
                    facecolor='#c8e6c9',
                    alpha=0.3,
                    zorder=0
                )
                ax.add_patch(rect)

    # ========================================================================
    # PLOT OPENING RANGE LEVELS
    # ========================================================================
    for session in sessions:
        if session.end_idx < start_idx:
            continue

        sess_start = max(0, session.start_idx - start_idx)
        sess_end = min(len(df_display) - 1, session.end_idx - start_idx)

        if sess_end < 0 or sess_start >= len(df_display):
            continue

        # Plot high line (dotted black)
        ax.plot(
            [sess_start, sess_end + 60],  # Extend into trading time
            [session.high, session.high],
            color='black',
            linewidth=1.5,
            linestyle=':',
            alpha=0.7,
            zorder=3
        )

        # Plot low line (dotted black)
        ax.plot(
            [sess_start, sess_end + 60],
            [session.low, session.low],
            color='black',
            linewidth=1.5,
            linestyle=':',
            alpha=0.7,
            zorder=3
        )

        # Add labels
        mid_x = (sess_start + sess_end) / 2
        ax.text(
            mid_x,
            session.high,
            ' Range High',
            fontsize=7,
            color='black',
            va='bottom',
            ha='left',
            fontweight='bold',
            zorder=5
        )
        ax.text(
            mid_x,
            session.low,
            ' Range Low',
            fontsize=7,
            color='black',
            va='top',
            ha='left',
            fontweight='bold',
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
    # PLOT CISD LEVELS
    # ========================================================================
    for cisd in cisd_levels:
        if cisd.index < start_idx:
            continue

        cisd_idx = max(0, cisd.index - start_idx)
        cisd_end = len(df_display) - 1 if not cisd.triggered else min(len(df_display) - 1, cisd.trigger_idx - start_idx if cisd.trigger_idx else len(df_display) - 1)

        # Bearish CISD (resistance) - red dotted
        # Bullish CISD (support) - green dotted
        cisd_color = '#890505' if cisd.is_bearish else '#016004'

        ax.plot(
            [cisd_idx, cisd_end],
            [cisd.price, cisd.price],
            color=cisd_color,
            linewidth=2,
            linestyle=':',
            alpha=0.8,
            zorder=3.5
        )

        # Add CISD label
        label_text = 'CISD Level' if cisd.is_bearish else 'CISD Level'
        ax.text(
            cisd_idx + 2,
            cisd.price,
            f' {label_text}',
            fontsize=7,
            color=cisd_color,
            va='center',
            ha='left',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            zorder=5
        )

        # Mark trigger point
        if cisd.triggered and cisd.trigger_idx:
            trigger_idx = cisd.trigger_idx - start_idx
            if 0 <= trigger_idx < len(df_display):
                marker = 'v' if cisd.is_bearish else '^'
                ax.scatter(
                    trigger_idx,
                    cisd.price,
                    marker=marker,
                    s=120,
                    color=cisd_color,
                    edgecolors='black',
                    linewidth=1.5,
                    zorder=6,
                    alpha=0.9
                )

    # ========================================================================
    # PLOT ORDER BLOCKS
    # ========================================================================
    if show_order_blocks:
        for ob in order_blocks:
            if ob.end_idx < start_idx:
                continue

            ob_start = max(0, ob.start_idx - start_idx)
            ob_end = min(len(df_display) - 1, ob.end_idx - start_idx)

            if ob_end < 0 or ob_start >= len(df_display):
                continue

            # Demand OB (green), Supply OB (red)
            ob_color = '#17c001' if ob.is_demand else '#df0505'
            alpha = 0.2 if not ob.mitigated else 0.1

            ob_height = abs(ob.top - ob.bottom)
            rect = Rectangle(
                (ob_start - 0.5, ob.bottom),
                ob_end - ob_start + 1,
                ob_height,
                facecolor=ob_color,
                edgecolor=ob_color,
                alpha=alpha,
                linewidth=1.5,
                zorder=1
            )
            ax.add_patch(rect)

            # Add label
            if not ob.mitigated:
                label_text = 'Demand OB' if ob.is_demand else 'Supply OB'
                ax.text(
                    ob_end,
                    (ob.top + ob.bottom) / 2,
                    f' {label_text}',
                    fontsize=6,
                    color=ob_color,
                    va='center',
                    ha='left',
                    fontweight='bold',
                    zorder=5
                )

    # ========================================================================
    # PLOT FVGs
    # ========================================================================
    if show_fvgs:
        for fvg in fvgs:
            if fvg.end_idx < start_idx:
                continue

            fvg_start = max(0, fvg.start_idx - start_idx)
            fvg_end = min(len(df_display) - 1, fvg.end_idx - start_idx + 30)

            if fvg_end < 0 or fvg_start >= len(df_display):
                continue

            # Demand FVG (blue), Supply FVG (orange)
            fvg_color = '#2195f3' if fvg.is_bullish else '#ff9900'
            alpha = 0.3 if not fvg.mitigated else 0.1

            fvg_height = abs(fvg.top - fvg.bottom)
            rect = Rectangle(
                (fvg_start - 0.5, fvg.bottom),
                fvg_end - fvg_start + 1,
                fvg_height,
                facecolor=fvg_color,
                edgecolor=fvg_color,
                alpha=alpha,
                linewidth=1,
                linestyle='--',
                zorder=1.2
            )
            ax.add_patch(rect)

            # Add label for active FVGs
            if not fvg.mitigated:
                label_text = 'Demand FVG' if fvg.is_bullish else 'Supply FVG'
                ax.text(
                    fvg_end,
                    (fvg.top + fvg.bottom) / 2,
                    f' {label_text}',
                    fontsize=6,
                    color=fvg_color,
                    va='center',
                    ha='left',
                    zorder=5
                )

    # ========================================================================
    # PLOT SIGNAL MARKERS
    # ========================================================================
    for signal in signals:
        if signal.session.end_idx < start_idx:
            continue

        # Mark the break point
        if signal.high_break:
            # Bearish signal - mark with down arrow
            break_idx = signal.session.end_idx + 1 - start_idx
            if 0 <= break_idx < len(df_display):
                ax.annotate(
                    'BEAR',
                    xy=(break_idx, df_display.iloc[break_idx]['high']),
                    xytext=(break_idx, df_display.iloc[break_idx]['high'] + 20),
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    zorder=10
                )

        if signal.low_break:
            # Bullish signal - mark with up arrow
            break_idx = signal.session.end_idx + 1 - start_idx
            if 0 <= break_idx < len(df_display):
                ax.annotate(
                    'BULL',
                    xy=(break_idx, df_display.iloc[break_idx]['low']),
                    xytext=(break_idx, df_display.iloc[break_idx]['low'] - 20),
                    fontsize=9,
                    fontweight='bold',
                    color='white',
                    ha='center',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    zorder=10
                )

    # ========================================================================
    # FORMATTING
    # ========================================================================
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')

    title = (f"Silver Bullet ICT Strategy [TradingFinder] 10-11 AM NY Time | "
            f"{len(signals)} Signals | {len(order_blocks)} OBs | {len(fvgs)} FVGs")
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='#fff9c4', alpha=0.3, label='Opening Range (9-10 AM)'),
        mpatches.Patch(facecolor='#c8e6c9', alpha=0.3, label='Trading Time (10-11 AM)'),
        plt.Line2D([0], [0], color='black', linewidth=1.5, linestyle=':', label='Range High/Low'),
        plt.Line2D([0], [0], color='#890505', linewidth=2, linestyle=':', label='Bearish CISD Level'),
        plt.Line2D([0], [0], color='#016004', linewidth=2, linestyle=':', label='Bullish CISD Level'),
        mpatches.Patch(facecolor='#df0505', alpha=0.2, edgecolor='#df0505', linewidth=1.5, label='Supply OB'),
        mpatches.Patch(facecolor='#17c001', alpha=0.2, edgecolor='#17c001', linewidth=1.5, label='Demand OB'),
        mpatches.Patch(facecolor='#ff9900', alpha=0.3, edgecolor='#ff9900', linestyle='--', label='Supply FVG'),
        mpatches.Patch(facecolor='#2195f3', alpha=0.3, edgecolor='#2195f3', linestyle='--', label='Demand FVG'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#890505',
                  markeredgecolor='black', markersize=10, label='CISD Trigger', linestyle='None'),
        mpatches.Patch(facecolor='red', alpha=0.8, label='BEAR Signal'),
        mpatches.Patch(facecolor='green', alpha=0.8, label='BULL Signal')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    output_file = "Silver_Bullet_ICT_Strategy__TradingFinder__10-11_AM_NY_Time__FVG_TFlab_Silver_Bullet.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nTradingFinder Silver Bullet chart saved as '{output_file}'")

    # Print summary
    print(f"\n{'='*70}")
    print(f"TRADINGFINDER SILVER BULLET SUMMARY")
    print(f"{'='*70}")
    print(f"Opening Range Sessions: {len(sessions)}")
    print(f"Total Signals: {len(signals)}")
    print(f"  - Bullish (Low Break): {len([s for s in signals if s.signal_type == 'BULL'])}")
    print(f"  - Bearish (High Break): {len([s for s in signals if s.signal_type == 'BEAR'])}")
    print(f"CISD Levels: {len(cisd_levels)} ({len([c for c in cisd_levels if c.triggered])} triggered)")
    print(f"Order Blocks: {len(order_blocks)} ({len([ob for ob in order_blocks if not ob.mitigated])} active)")
    print(f"FVGs: {len(fvgs)} ({len([f for f in fvgs if not f.mitigated])} active)")
    print(f"{'='*70}")


if __name__ == "__main__":
    plot_tradingfinder_silver_bullet_chart(
        "PEPPERSTONE_XAUUSD, 5.csv",
        num_candles=500,
        bar_back_check=5,
        show_order_blocks=True,
        show_fvgs=True,
        show_sessions=True
    )
