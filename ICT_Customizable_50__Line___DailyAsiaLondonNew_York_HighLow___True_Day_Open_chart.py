"""
ICT Customizable 50% Line Chart
Standalone candlestick chart showing ICT daily levels and killzone sessions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load ICT module
spec = importlib.util.spec_from_file_location(
    "ict_levels",
    "ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.py"
)
ict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_module)

detect_daily_levels = ict_module.detect_daily_levels
detect_killzone_sessions = ict_module.detect_killzone_sessions


def plot_ict_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with ICT daily levels and killzone sessions

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate daily levels on full dataset
    print("Calculating daily high/low/50% levels...")
    daily_levels = detect_daily_levels(df)

    print("Calculating killzone session levels...")
    killzone_levels = detect_killzone_sessions(df)

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

    # Plot daily levels - last 2 sessions
    for daily in daily_levels[-2:]:
        if daily.end_idx >= start_idx:
            x_start = max(0, daily.start_idx - start_idx)
            x_end = min(len(df_display) - 1, daily.end_idx - start_idx)

            # Daily High (red)
            ax.plot([x_start, x_end], [daily.high, daily.high],
                   color='red', linewidth=1.5, linestyle=':', alpha=0.7, zorder=3)
            ax.text(x_end, daily.high, ' Daily High', fontsize=8,
                   color='red', va='bottom', ha='left', fontweight='bold', zorder=5)

            # Daily Low (red)
            ax.plot([x_start, x_end], [daily.low, daily.low],
                   color='red', linewidth=1.5, linestyle=':', alpha=0.7, zorder=3)
            ax.text(x_end, daily.low, ' Daily Low', fontsize=8,
                   color='red', va='top', ha='left', fontweight='bold', zorder=5)

            # 50% Line (cyan)
            ax.plot([x_start, x_end], [daily.mid, daily.mid],
                   color='cyan', linewidth=2, linestyle=':', alpha=0.8, zorder=3)
            ax.text(x_end, daily.mid, ' 50% Level', fontsize=9,
                   color='cyan', va='center', ha='left', fontweight='bold', zorder=5,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

            # Market Open vertical line
            if daily.market_open_idx >= start_idx:
                mo_x = daily.market_open_idx - start_idx
                ax.axvline(x=mo_x, color='white', linewidth=1, linestyle=':', alpha=0.5, zorder=2)
                ax.text(mo_x, daily.high + (daily.high - daily.low) * 0.02, 'Market Open',
                       fontsize=7, color='white', ha='center', va='bottom',
                       rotation=90, alpha=0.7, zorder=5)

    # Plot killzone sessions - last 2 of each
    # Asia - Blue
    for level in killzone_levels['asia_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='blue', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'Asia High ', fontsize=7,
                   color='blue', va='bottom', ha='right', fontweight='bold', zorder=5)

    for level in killzone_levels['asia_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='blue', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'Asia Low ', fontsize=7,
                   color='blue', va='top', ha='right', fontweight='bold', zorder=5)

    # London - Orange
    for level in killzone_levels['london_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#fd7200', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'London High ', fontsize=7,
                   color='#fd7200', va='bottom', ha='right', fontweight='bold', zorder=5)

    for level in killzone_levels['london_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#fd7200', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'London Low ', fontsize=7,
                   color='#fd7200', va='top', ha='right', fontweight='bold', zorder=5)

    # NY Full Session - Green
    for level in killzone_levels['ny_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#00ff96', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'NY High ', fontsize=7,
                   color='#00ff96', va='bottom', ha='right', fontweight='bold', zorder=5)

    for level in killzone_levels['ny_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='#00ff96', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)
            ax.text(x_end - 10, level.price, 'NY Low ', fontsize=7,
                   color='#00ff96', va='top', ha='right', fontweight='bold', zorder=5)

    # NY AM Killzone - Aqua
    for level in killzone_levels['ny_am_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='aqua', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    for level in killzone_levels['ny_am_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='aqua', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    # NY Lunch - Yellow
    for level in killzone_levels['ny_lunch_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='yellow', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    for level in killzone_levels['ny_lunch_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='yellow', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    # NY PM - Fuchsia
    for level in killzone_levels['ny_pm_high'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='fuchsia', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    for level in killzone_levels['ny_pm_low'][-2:]:
        if level.end_idx >= start_idx:
            x_start = max(0, level.start_idx - start_idx)
            x_end = min(len(df_display) - 1, level.end_idx - start_idx)
            ax.plot([x_start, x_end], [level.price, level.price],
                   color='fuchsia', linewidth=1.2, linestyle='-', alpha=0.5, zorder=2.5)

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax.set_title("ICT 50% Line + Daily/Killzone H/L (Last 2 Days)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='red', alpha=0.7, label='Daily High/Low'),
        mpatches.Patch(facecolor='cyan', alpha=0.8, label='50% Level'),
        mpatches.Patch(facecolor='blue', alpha=0.6, label='Asia Killzone'),
        mpatches.Patch(facecolor='#fd7200', alpha=0.6, label='London Killzone'),
        mpatches.Patch(facecolor='#00ff96', alpha=0.6, label='NY Full Session'),
        mpatches.Patch(facecolor='aqua', alpha=0.5, label='NY AM'),
        mpatches.Patch(facecolor='yellow', alpha=0.5, label='NY Lunch'),
        mpatches.Patch(facecolor='fuchsia', alpha=0.5, label='NY PM')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    output_file = "ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nICT levels chart saved as '{output_file}'")
    print(f"\nLevels Summary:")
    print(f"  - Daily Levels: {len(daily_levels)}")
    print(f"  - Asia Highs: {len(killzone_levels['asia_high'])}")
    print(f"  - London Highs: {len(killzone_levels['london_high'])}")
    print(f"  - NY Highs: {len(killzone_levels['ny_high'])}")
    print(f"  - NY AM Highs: {len(killzone_levels['ny_am_high'])}")
    print(f"  - NY Lunch Highs: {len(killzone_levels['ny_lunch_high'])}")
    print(f"  - NY PM Highs: {len(killzone_levels['ny_pm_high'])}")


if __name__ == "__main__":
    plot_ict_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
