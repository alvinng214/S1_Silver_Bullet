"""
Smart Money Zones Chart - FVG + OB + MTF Trend Panel
Standalone candlestick chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load Smart Money Zones module
spec = importlib.util.spec_from_file_location(
    "smart_money_zones",
    "Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py"
)
smz_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smz_module)

detect_smart_money_zones = smz_module.detect_smart_money_zones
calculate_mtf_trends = smz_module.calculate_mtf_trends


def plot_smart_money_zones_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with Smart Money Zones + MTF Trend Panel

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of candles to display
    """
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['time'])
    df = df.set_index('datetime').sort_index()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Add index for calculations
    df['index'] = range(len(df))

    # Calculate Smart Money Zones on full dataset
    print("Calculating Smart Money Zones (FVG + OB)...")
    smz_data = detect_smart_money_zones(
        df,
        show_fvg=True,
        show_ob=True,
        max_zones=20,
        min_body_pct=0.5,
        ob_lookback=10,
        atr_length=14,
        use_trend_filter=True,
        trend_ma_period=50
    )

    # Calculate MTF Trends
    print("Calculating Multi-Timeframe Trends...")
    mtf_trends = calculate_mtf_trends(
        df,
        timeframes_minutes={'5m': 5, '15m': 15, '1H': 60, '4H': 240, '1D': 1440},
        ma_period=50
    )

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index()

    print(f"Displaying last {len(df_display)} candles...")

    # Filter zones to display range
    bull_fvgs = [z for z in smz_data['bull_fvg'] if z.end_idx >= start_idx]
    bear_fvgs = [z for z in smz_data['bear_fvg'] if z.end_idx >= start_idx]
    bull_obs = [z for z in smz_data['bull_ob'] if z.end_idx >= start_idx]
    bear_obs = [z for z in smz_data['bear_ob'] if z.end_idx >= start_idx]

    # Adjust indices for display
    for zone in bull_fvgs + bear_fvgs + bull_obs + bear_obs:
        zone.start_idx = max(0, zone.start_idx - start_idx)
        zone.end_idx = min(len(df_display), zone.end_idx - start_idx)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 10, height_ratios=[5, 1], hspace=0.15, wspace=0.3)

    # Main chart
    ax_main = fig.add_subplot(gs[0, :8])

    # MTF Trend Panel (right side)
    ax_panel = fig.add_subplot(gs[0, 8:])
    ax_panel.set_facecolor('#1a1a1a')  # Dark background
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis('off')

    # Trend MA panel (bottom)
    ax_trend = fig.add_subplot(gs[1, :8], sharex=ax_main)

    # Plot zones first (background)
    # Bullish FVGs - green
    for zone in bull_fvgs:
        alpha = 0.15 if zone.is_live else 0.05
        color = 'green'
        zone_rect = Rectangle(
            (zone.start_idx, zone.bottom),
            zone.end_idx - zone.start_idx,
            zone.top - zone.bottom,
            facecolor=color,
            edgecolor='darkgreen',
            alpha=alpha,
            linewidth=1,
            zorder=0.2
        )
        ax_main.add_patch(zone_rect)
        # Add label
        mid_x = (zone.start_idx + zone.end_idx) / 2
        ax_main.text(mid_x, zone.top, f'FVG {zone.strength}', fontsize=6,
                    color='green', ha='center', va='bottom', alpha=0.7, zorder=5)

    # Bearish FVGs - red
    for zone in bear_fvgs:
        alpha = 0.15 if zone.is_live else 0.05
        color = 'red'
        zone_rect = Rectangle(
            (zone.start_idx, zone.bottom),
            zone.end_idx - zone.start_idx,
            zone.top - zone.bottom,
            facecolor=color,
            edgecolor='darkred',
            alpha=alpha,
            linewidth=1,
            zorder=0.2
        )
        ax_main.add_patch(zone_rect)
        # Add label
        mid_x = (zone.start_idx + zone.end_idx) / 2
        ax_main.text(mid_x, zone.bottom, f'FVG {zone.strength}', fontsize=6,
                    color='red', ha='center', va='top', alpha=0.7, zorder=5)

    # Bullish OBs - blue
    for zone in bull_obs:
        alpha = 0.2 if zone.is_live else 0.05
        color = '#0088ff'
        zone_rect = Rectangle(
            (zone.start_idx, zone.bottom),
            zone.end_idx - zone.start_idx,
            zone.top - zone.bottom,
            facecolor=color,
            edgecolor='blue',
            alpha=alpha,
            linewidth=1.5,
            zorder=0.3
        )
        ax_main.add_patch(zone_rect)
        # Add label
        mid_x = (zone.start_idx + zone.end_idx) / 2
        ax_main.text(mid_x, zone.top, f'OB {zone.strength}', fontsize=6,
                    color='blue', ha='center', va='bottom', alpha=0.7, zorder=5)

    # Bearish OBs - orange
    for zone in bear_obs:
        alpha = 0.2 if zone.is_live else 0.05
        color = '#ff6600'
        zone_rect = Rectangle(
            (zone.start_idx, zone.bottom),
            zone.end_idx - zone.start_idx,
            zone.top - zone.bottom,
            facecolor=color,
            edgecolor='darkorange',
            alpha=alpha,
            linewidth=1.5,
            zorder=0.3
        )
        ax_main.add_patch(zone_rect)
        # Add label
        mid_x = (zone.start_idx + zone.end_idx) / 2
        ax_main.text(mid_x, zone.bottom, f'OB {zone.strength}', fontsize=6,
                    color='orange', ha='center', va='top', alpha=0.7, zorder=5)

    # Plot candlesticks
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
        color = 'blue' if row['close'] >= row['open'] else 'red'

        # Draw candle body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                         facecolor=color, edgecolor=color, alpha=0.8, zorder=2)
        ax_main.add_patch(rect)

        # Draw wicks
        ax_main.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5, zorder=1)

    # Plot Trend MA on bottom panel
    trend_ma_display = smz_data['trend_ma'][start_idx:]
    close_display = df_display['close'].values
    is_bullish = close_display > trend_ma_display

    colors = ['lime' if bull else 'red' for bull in is_bullish]
    ax_trend.bar(range(len(df_display)), [1]*len(df_display), color=colors, width=1, alpha=0.6)
    ax_trend.set_ylim(0, 1)
    ax_trend.set_xlim(-0.5, len(df_display) - 0.5)
    ax_trend.set_yticks([])
    ax_trend.set_xlabel('Candle Index', fontsize=10)
    ax_trend.set_title('Trend Filter (50 MA)', fontsize=10, fontweight='bold')
    ax_trend.grid(True, alpha=0.3, axis='x')

    # Create MTF Trend Panel
    ax_panel.text(0.5, 0.95, 'MTF TREND PANEL', ha='center', va='top',
                 fontsize=11, fontweight='bold', color='white',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#00bcd4', alpha=0.9))

    y_positions = np.linspace(0.80, 0.15, len(mtf_trends))

    for idx, (tf_name, is_bullish) in enumerate(mtf_trends.items()):
        y_pos = y_positions[idx]

        # Timeframe label with background box
        ax_panel.text(0.15, y_pos, tf_name, ha='center', va='center',
                     fontsize=12, fontweight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        # Trend status
        trend_text = "BULL" if is_bullish else "BEAR"
        trend_color = 'lime' if is_bullish else 'red'
        ax_panel.text(0.55, y_pos, trend_text, ha='center', va='center',
                     fontsize=11, fontweight='bold', color=trend_color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Status indicator circle (simple colored square)
        status_symbol = "●"
        ax_panel.text(0.85, y_pos, status_symbol, ha='center', va='center',
                     fontsize=16, color=trend_color, weight='bold')

    # Formatting
    ax_main.set_xlim(-0.5, len(df_display) - 0.5)
    ax_main.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax_main.set_title('Smart Money Zones (FVG + OB) + MTF Trend Panel - Last 500 Candles',
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.15, edgecolor='darkgreen', label='Bullish FVG'),
        mpatches.Patch(facecolor='red', alpha=0.15, edgecolor='darkred', label='Bearish FVG'),
        mpatches.Patch(facecolor='#0088ff', alpha=0.2, edgecolor='blue', label='Bullish OB'),
        mpatches.Patch(facecolor='#ff6600', alpha=0.2, edgecolor='darkorange', label='Bearish OB'),
        mpatches.Patch(facecolor='gray', alpha=0.05, label='Mitigated Zone')
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    output_file = 'Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSmart Money Zones chart saved as '{output_file}'")
    print(f"  - Bullish FVGs: {len(bull_fvgs)}")
    print(f"  - Bearish FVGs: {len(bear_fvgs)}")
    print(f"  - Bullish OBs: {len(bull_obs)}")
    print(f"  - Bearish OBs: {len(bear_obs)}")
    print(f"\nMTF Trends:")
    for tf_name, is_bullish in mtf_trends.items():
        status = "BULLISH 🟢" if is_bullish else "BEARISH 🔴"
        print(f"  - {tf_name}: {status}")


if __name__ == "__main__":
    plot_smart_money_zones_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
