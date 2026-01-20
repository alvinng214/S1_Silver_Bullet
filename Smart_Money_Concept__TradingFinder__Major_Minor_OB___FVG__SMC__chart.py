"""
Smart Money Concept Chart - Order Blocks + Fair Value Gaps
Standalone candlestick chart showing SMC indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load Smart Money Concept module
spec_smc = importlib.util.spec_from_file_location(
    "smart_money_concept",
    "Smart_Money_Concept__TradingFinder__Major_Minor_OB___FVG__SMC_.py"
)
smc_module = importlib.util.module_from_spec(spec_smc)
spec_smc.loader.exec_module(smc_module)
calculate_fvgs_for_chart = smc_module.calculate_fvgs_for_chart

# Load Order Blocks module
spec_ob = importlib.util.spec_from_file_location(
    "order_blocks",
    "MirPapa-ICT-HTF- FVG OB Threeple (EN).py"
)
ob_module = importlib.util.module_from_spec(spec_ob)
spec_ob.loader.exec_module(ob_module)
detect_order_blocks_simple = ob_module.detect_order_blocks_simple


def plot_smart_money_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with Smart Money Concept indicators

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"Loaded {len(df)} bars from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

    # Add index for calculations
    df['index'] = range(len(df))

    # Calculate indicators on full dataset
    print("Calculating Fair Value Gaps (FVG)...")
    fvg_data = calculate_fvgs_for_chart(df, show_demand=True, show_supply=True, filter_type='Defensive')

    print("Calculating Order Blocks...")
    all_obs = detect_order_blocks_simple(df, use_body=True)

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index(drop=True)

    print(f"Displaying last {len(df_display)} candles...")

    # Filter indicators to display range
    bullish_fvgs = [fvg for fvg in fvg_data['bullish']
                    if fvg.end_idx >= start_idx and fvg.start_idx < len(df)]
    bearish_fvgs = [fvg for fvg in fvg_data['bearish']
                    if fvg.end_idx >= start_idx and fvg.start_idx < len(df)]

    bullish_obs = [ob for ob in all_obs
                   if ob.is_bullish and ob.end_idx >= start_idx and ob.start_idx < len(df)]
    bearish_obs = [ob for ob in all_obs
                   if not ob.is_bullish and ob.end_idx >= start_idx and ob.start_idx < len(df)]

    # Adjust indices for display
    for fvg in bullish_fvgs:
        fvg.start_idx = max(0, fvg.start_idx - start_idx)
        fvg.end_idx = min(len(df_display), fvg.end_idx - start_idx)

    for fvg in bearish_fvgs:
        fvg.start_idx = max(0, fvg.start_idx - start_idx)
        fvg.end_idx = min(len(df_display), fvg.end_idx - start_idx)

    for ob in bullish_obs:
        ob.start_idx = max(0, ob.start_idx - start_idx)
        ob.end_idx = min(len(df_display), ob.end_idx - start_idx)

    for ob in bearish_obs:
        ob.start_idx = max(0, ob.start_idx - start_idx)
        ob.end_idx = min(len(df_display), ob.end_idx - start_idx)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot Order Blocks first (background)
    # Bullish OBs - green
    for ob in bullish_obs:
        if not ob.mitigated or (ob.end_idx - ob.start_idx) < 100:
            ob_rect = Rectangle(
                (ob.start_idx, ob.bottom),
                ob.end_idx - ob.start_idx,
                ob.top - ob.bottom,
                facecolor='green',
                edgecolor='darkgreen',
                alpha=0.2,
                linewidth=1.5,
                zorder=0.4
            )
            ax.add_patch(ob_rect)

    # Bearish OBs - red
    for ob in bearish_obs:
        if not ob.mitigated or (ob.end_idx - ob.start_idx) < 100:
            ob_rect = Rectangle(
                (ob.start_idx, ob.bottom),
                ob.end_idx - ob.start_idx,
                ob.top - ob.bottom,
                facecolor='red',
                edgecolor='darkred',
                alpha=0.2,
                linewidth=1.5,
                zorder=0.4
            )
            ax.add_patch(ob_rect)

    # Plot Fair Value Gaps
    # Bullish FVGs - cyan
    for fvg in bullish_fvgs:
        if not fvg.mitigated or (fvg.end_idx - fvg.start_idx) < 100:
            fvg_rect = Rectangle(
                (fvg.start_idx, fvg.bottom),
                fvg.end_idx - fvg.start_idx,
                fvg.top - fvg.bottom,
                facecolor='cyan',
                edgecolor='cyan',
                alpha=0.15,
                linewidth=1,
                linestyle='--',
                zorder=0.3
            )
            ax.add_patch(fvg_rect)

    # Bearish FVGs - orange
    for fvg in bearish_fvgs:
        if not fvg.mitigated or (fvg.end_idx - fvg.start_idx) < 100:
            fvg_rect = Rectangle(
                (fvg.start_idx, fvg.bottom),
                fvg.end_idx - fvg.start_idx,
                fvg.top - fvg.bottom,
                facecolor='orange',
                edgecolor='orange',
                alpha=0.15,
                linewidth=1,
                linestyle='--',
                zorder=0.3
            )
            ax.add_patch(fvg_rect)

    # Plot candlesticks on top
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

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.set_title(f'Smart Money Concept - Order Blocks + FVG (Last {num_candles} Candles)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.2, edgecolor='darkgreen', label='Bullish OB'),
        mpatches.Patch(facecolor='red', alpha=0.2, edgecolor='darkred', label='Bearish OB'),
        mpatches.Patch(facecolor='cyan', alpha=0.15, edgecolor='cyan', linestyle='--', label='Bullish FVG (Demand)'),
        mpatches.Patch(facecolor='orange', alpha=0.15, edgecolor='orange', linestyle='--', label='Bearish FVG (Supply)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    output_file = 'Smart_Money_Concept__TradingFinder__Major_Minor_OB___FVG__SMC_.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSmart Money Concept chart saved as '{output_file}'")
    print(f"  - Bullish OBs: {len(bullish_obs)}")
    print(f"  - Bearish OBs: {len(bearish_obs)}")
    print(f"  - Bullish FVGs: {len(bullish_fvgs)}")
    print(f"  - Bearish FVGs: {len(bearish_fvgs)}")


if __name__ == "__main__":
    plot_smart_money_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
