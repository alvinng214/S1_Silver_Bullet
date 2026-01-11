"""
Combined Market Structure MTF + HTF Bias + Order Blocks Chart

This script combines:
1. Market Structure MTF Trend indicator (multi-timeframe trend panel)
2. HTF Bias indicator (HTF candles, sweeps, and bias)
3. Order Blocks (15min and 1H)

Display:
- Main chart: Candlesticks with HTF levels, sweep markers, and Order Blocks
- Right side: Mini HTF candles (1H, 4H, Daily)
- Bottom panel: Market structure trend indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys
import os

# Import market structure functions
sys.path.append(os.path.dirname(__file__))
from market_structure_chart import (
    calculate_market_structure,
    plot_trend_line
)
from HTF_bias import calculate_htf_data, HTFCandle, Sweep
from order_blocks import calculate_htf_order_blocks


def plot_combined_chart(csv_file, num_candles=200, pivot_strength=15):
    """
    Plot combined chart with Market Structure MTF + HTF Bias indicators.

    Args:
        csv_file: Path to CSV file with OHLC data
        num_candles: Number of recent candles to display
        pivot_strength: Pivot strength for structure detection
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Get last N candles
    df = df.tail(num_candles).copy()
    df['index'] = range(len(df))

    print(f"Analyzing last {len(df)} candles...")

    # ========================================================================
    # CALCULATE MARKET STRUCTURE FOR MULTIPLE TIMEFRAMES
    # ========================================================================
    print("Calculating market structure for multiple timeframes...")

    # TF1: 15min - Resample 5min data to 15min
    df_15m = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'index': 'last'
    }).dropna()
    df_15m['index_15m'] = range(len(df_15m))
    trend_tf1_15m, choch_tf1_15m, bos_tf1_15m, _, _ = calculate_market_structure(df_15m, pivot_strength=5)

    # Map 15min results back to 5min indices
    trend_tf1 = np.zeros(len(df))
    for i, row in df_15m.iterrows():
        ltf_idx = int(row['index'])
        if ltf_idx < len(df):
            tf_idx = int(row['index_15m'])
            if tf_idx < len(trend_tf1_15m):
                trend_tf1[ltf_idx] = trend_tf1_15m[tf_idx]
    # Forward fill
    for i in range(1, len(trend_tf1)):
        if trend_tf1[i] == 0:
            trend_tf1[i] = trend_tf1[i-1]

    # Map CHoCH and BoS signals
    choch_tf1 = []
    bos_tf1 = []
    for idx_15m, direction, price in choch_tf1_15m:
        if idx_15m < len(df_15m):
            ltf_idx = int(df_15m.iloc[idx_15m]['index'])
            if ltf_idx < len(df):
                choch_tf1.append((ltf_idx, direction, price))
    for idx_15m, direction, price in bos_tf1_15m:
        if idx_15m < len(df_15m):
            ltf_idx = int(df_15m.iloc[idx_15m]['index'])
            if ltf_idx < len(df):
                bos_tf1.append((ltf_idx, direction, price))

    # TF2: 1H - Resample to 1 hour
    df_1h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'index': 'last'
    }).dropna()
    df_1h['index_1h'] = range(len(df_1h))
    trend_tf2_1h, choch_tf2_1h, bos_tf2_1h, _, _ = calculate_market_structure(df_1h, pivot_strength=5)

    # Map 1H results back to 5min indices
    trend_tf2 = np.zeros(len(df))
    for i, row in df_1h.iterrows():
        ltf_idx = int(row['index'])
        if ltf_idx < len(df):
            tf_idx = int(row['index_1h'])
            if tf_idx < len(trend_tf2_1h):
                trend_tf2[ltf_idx] = trend_tf2_1h[tf_idx]
    for i in range(1, len(trend_tf2)):
        if trend_tf2[i] == 0:
            trend_tf2[i] = trend_tf2[i-1]

    choch_tf2 = []
    bos_tf2 = []
    for idx_1h, direction, price in choch_tf2_1h:
        if idx_1h < len(df_1h):
            ltf_idx = int(df_1h.iloc[idx_1h]['index'])
            if ltf_idx < len(df):
                choch_tf2.append((ltf_idx, direction, price))
    for idx_1h, direction, price in bos_tf2_1h:
        if idx_1h < len(df_1h):
            ltf_idx = int(df_1h.iloc[idx_1h]['index'])
            if ltf_idx < len(df):
                bos_tf2.append((ltf_idx, direction, price))

    # TF3: 4H - Resample to 4 hours
    df_4h = df.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'index': 'last'
    }).dropna()
    df_4h['index_4h'] = range(len(df_4h))
    trend_tf3_4h, choch_tf3_4h, bos_tf3_4h, _, _ = calculate_market_structure(df_4h, pivot_strength=3)

    # Map 4H results back to 5min indices
    trend_tf3 = np.zeros(len(df))
    for i, row in df_4h.iterrows():
        ltf_idx = int(row['index'])
        if ltf_idx < len(df):
            tf_idx = int(row['index_4h'])
            if tf_idx < len(trend_tf3_4h):
                trend_tf3[ltf_idx] = trend_tf3_4h[tf_idx]
    for i in range(1, len(trend_tf3)):
        if trend_tf3[i] == 0:
            trend_tf3[i] = trend_tf3[i-1]

    choch_tf3 = []
    bos_tf3 = []
    for idx_4h, direction, price in choch_tf3_4h:
        if idx_4h < len(df_4h):
            ltf_idx = int(df_4h.iloc[idx_4h]['index'])
            if ltf_idx < len(df):
                choch_tf3.append((ltf_idx, direction, price))
    for idx_4h, direction, price in bos_tf3_4h:
        if idx_4h < len(df_4h):
            ltf_idx = int(df_4h.iloc[idx_4h]['index'])
            if ltf_idx < len(df):
                bos_tf3.append((ltf_idx, direction, price))

    # TF4: Daily - Resample to 1 day
    df_daily = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'index': 'last'
    }).dropna()
    df_daily['index_daily'] = range(len(df_daily))
    trend_tf4_daily, choch_tf4_daily, bos_tf4_daily, _, _ = calculate_market_structure(df_daily, pivot_strength=2)

    # Map Daily results back to 5min indices
    trend_tf4 = np.zeros(len(df))
    for i, row in df_daily.iterrows():
        ltf_idx = int(row['index'])
        if ltf_idx < len(df):
            tf_idx = int(row['index_daily'])
            if tf_idx < len(trend_tf4_daily):
                trend_tf4[ltf_idx] = trend_tf4_daily[tf_idx]
    for i in range(1, len(trend_tf4)):
        if trend_tf4[i] == 0:
            trend_tf4[i] = trend_tf4[i-1]

    choch_tf4 = []
    bos_tf4 = []
    for idx_daily, direction, price in choch_tf4_daily:
        if idx_daily < len(df_daily):
            ltf_idx = int(df_daily.iloc[idx_daily]['index'])
            if ltf_idx < len(df):
                choch_tf4.append((ltf_idx, direction, price))
    for idx_daily, direction, price in bos_tf4_daily:
        if idx_daily < len(df_daily):
            ltf_idx = int(df_daily.iloc[idx_daily]['index'])
            if ltf_idx < len(df):
                bos_tf4.append((ltf_idx, direction, price))

    # ========================================================================
    # CALCULATE HTF DATA
    # ========================================================================
    print("Calculating HTF candles and sweeps...")
    htf_data = calculate_htf_data(df, timeframes_minutes={'1H': 60, '4H': 240, 'Daily': 1440})

    # ========================================================================
    # CALCULATE ORDER BLOCKS
    # ========================================================================
    print("Calculating Order Blocks for 15min and 1H...")
    order_blocks_data = calculate_htf_order_blocks(df, timeframes_minutes={'15min': 15, '1H': 60})

    # ========================================================================
    # CREATE FIGURE WITH SUBPLOTS
    # ========================================================================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 10, height_ratios=[3, 1], hspace=0.1, wspace=0.3)

    # Main chart (candlesticks + HTF levels)
    ax_main = fig.add_subplot(gs[0, :8])  # Takes 8 columns

    # HTF candles mini chart
    ax_htf = fig.add_subplot(gs[0, 8:], sharey=ax_main)  # Takes 2 columns on the right

    # Market structure indicator panel
    ax_ms = fig.add_subplot(gs[1, :8], sharex=ax_main)  # Bottom panel

    # ========================================================================
    # PLOT MAIN CANDLESTICKS
    # ========================================================================
    width = 0.8
    for idx in range(len(df)):
        row = df.iloc[idx]
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
        ax_main.plot([x_pos, x_pos], [low_price, high_price], color='black', linewidth=1, zorder=1)

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
        ax_main.add_patch(rect)

    # ========================================================================
    # PLOT HTF LEVELS AND SWEEPS
    # ========================================================================
    colors_htf = {'1H': 'orange', '4H': 'purple', 'Daily': 'brown'}
    alphas = {'1H': 0.3, '4H': 0.4, 'Daily': 0.5}

    for tf_name, data in htf_data.items():
        color = colors_htf.get(tf_name, 'gray')
        alpha = alphas.get(tf_name, 0.3)

        # Plot HTF high/low lines
        for candle in data['candles'][-10:]:  # Last 10 HTF candles
            # High line
            ax_main.axhline(y=candle.high, color=color, linestyle='--',
                          linewidth=1, alpha=alpha, zorder=0)
            # Low line
            ax_main.axhline(y=candle.low, color=color, linestyle='--',
                          linewidth=1, alpha=alpha, zorder=0)

        # Plot sweeps with different colors for each timeframe
        for sweep in data['sweeps']:
            if sweep.is_bullish:
                # Bullish sweep (swept high)
                if tf_name == '1H':
                    sweep_color = 'lime'  # Bright green for better visibility
                    label_text = '1H Bull Sweep'
                elif tf_name == '4H':
                    sweep_color = 'dodgerblue'  # Bright blue for better visibility
                    label_text = '4H Bull Sweep'
                else:  # Daily
                    sweep_color = 'darkgreen'
                    label_text = 'Daily Bull Sweep'

                ax_main.scatter(sweep.index, sweep.price, marker='^',
                              color=sweep_color, s=250, zorder=5, alpha=1.0, edgecolors='black', linewidth=2,
                              label=label_text if sweep == data['sweeps'][0] else '')
            else:
                # Bearish sweep (swept low)
                if tf_name == '1H':
                    sweep_color = 'magenta'  # Bright pink/magenta for better visibility
                    label_text = '1H Bear Sweep'
                elif tf_name == '4H':
                    sweep_color = 'crimson'  # Deep red for better visibility
                    label_text = '4H Bear Sweep'
                else:  # Daily
                    sweep_color = 'darkred'
                    label_text = 'Daily Bear Sweep'

                ax_main.scatter(sweep.index, sweep.price, marker='v',
                              color=sweep_color, s=250, zorder=5, alpha=1.0, edgecolors='black', linewidth=2,
                              label=label_text if sweep == data['sweeps'][0] else '')

    # ========================================================================
    # PLOT ORDER BLOCKS
    # ========================================================================
    ob_colors = {
        '15min': {'bull': 'green', 'bear': 'hotpink'},
        '1H': {'bull': 'blue', 'bear': 'red'}
    }
    ob_alpha = 0.2

    for tf_name, order_blocks in order_blocks_data.items():
        colors = ob_colors.get(tf_name, {'bull': 'gray', 'bear': 'gray'})

        for ob in order_blocks:
            # Only show unmitigated OBs or recently mitigated ones
            if not ob.mitigated or (ob.end_idx - ob.start_idx) < 200:
                color = colors['bull'] if ob.is_bullish else colors['bear']

                # Draw the Order Block as a rectangle
                ob_rect = Rectangle(
                    (ob.start_idx, ob.bottom),
                    ob.end_idx - ob.start_idx,
                    ob.top - ob.bottom,
                    facecolor=color,
                    edgecolor=color,
                    alpha=ob_alpha,
                    linewidth=1,
                    zorder=0.5
                )
                ax_main.add_patch(ob_rect)

    # Add OB legend entries
    ob_legend_elements = [
        mpatches.Patch(color=ob_colors['15min']['bull'], alpha=ob_alpha, label='15min Bull OB'),
        mpatches.Patch(color=ob_colors['15min']['bear'], alpha=ob_alpha, label='15min Bear OB'),
        mpatches.Patch(color=ob_colors['1H']['bull'], alpha=ob_alpha, label='1H Bull OB'),
        mpatches.Patch(color=ob_colors['1H']['bear'], alpha=ob_alpha, label='1H Bear OB'),
    ]

    # ========================================================================
    # PLOT HTF MINI CANDLES (Right side)
    # ========================================================================
    htf_candle_width = 0.6
    y_min, y_max = ax_main.get_ylim()
    price_range = y_max - y_min

    x_offset = 0
    for tf_idx, (tf_name, data) in enumerate(htf_data.items()):
        candles = data['candles'][-8:]  # Show last 8 HTF candles
        bias = data['bias']

        for i, candle in enumerate(candles):
            x_pos = x_offset + i * (htf_candle_width + 0.2)

            # Normalize price to 0-1 range for display
            norm_open = (candle.open - y_min) / price_range
            norm_close = (candle.close - y_min) / price_range
            norm_high = (candle.high - y_min) / price_range
            norm_low = (candle.low - y_min) / price_range

            # Convert back to actual prices for display
            display_open = y_min + norm_open * price_range
            display_close = y_min + norm_close * price_range
            display_high = y_min + norm_high * price_range
            display_low = y_min + norm_low * price_range

            # Color
            if candle.is_bullish:
                color = 'green'
                body_bottom = display_open
                body_height = display_close - display_open
            else:
                color = 'red'
                body_bottom = display_close
                body_height = display_open - display_close

            # Draw wick
            ax_htf.plot([x_pos, x_pos], [display_low, display_high],
                       color='black', linewidth=1, zorder=1)

            # Draw body
            rect = Rectangle(
                (x_pos - htf_candle_width/2, body_bottom),
                htf_candle_width,
                body_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
            ax_htf.add_patch(rect)

        # Add timeframe label
        ax_htf.text(x_offset + 3, y_max + price_range * 0.02, tf_name,
                   fontsize=10, fontweight='bold', ha='center')

        # Add bias indicator
        bias_text = "↑" if bias == 1 else "↓" if bias == -1 else "→"
        bias_color = "green" if bias == 1 else "red" if bias == -1 else "gray"
        ax_htf.text(x_offset + 3, y_min - price_range * 0.02, bias_text,
                   fontsize=14, fontweight='bold', ha='center', color=bias_color)

        x_offset += len(candles) * (htf_candle_width + 0.2) + 2

    # ========================================================================
    # PLOT CHoCH AND BoS LABELS
    # ========================================================================
    for idx, direction, price in choch_tf3:
        if direction > 0:  # Bullish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price - 10),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
        else:  # Bearish CHoCH
            ax_main.annotate('CHoCH', xy=(idx, price), xytext=(idx, price + 10),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                           fontsize=8, color='white', ha='center',
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # ========================================================================
    # PLOT MARKET STRUCTURE TRENDS
    # ========================================================================
    choch_bull_color = 'darkgreen'
    choch_bear_color = 'darkred'
    bos_bull_color = 'green'
    bos_bear_color = 'red'

    plot_trend_line(ax_ms, trend_tf1, choch_tf1, bos_tf1, 3,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf2, choch_tf2, bos_tf2, 2,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf3, choch_tf3, bos_tf3, 1,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf4, choch_tf4, bos_tf4, 0,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Add timeframe labels
    ax_ms.text(-5, 3, '15min', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 2, '1H', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 1, '4H', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 0, 'Daily', ha='right', va='center', fontsize=9, fontweight='bold')

    # ========================================================================
    # FORMATTING
    # ========================================================================
    # X-axis labels
    step = max(1, len(df) // 10)
    tick_positions = list(range(0, len(df), step))
    tick_labels = [df.iloc[i].name.strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels([])  # Hide x labels on main chart

    ax_ms.set_xticks(tick_positions)
    ax_ms.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Main chart formatting
    ax_main.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax_main.set_title('XAUUSD - Market Structure MTF + HTF Bias + Order Blocks',
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # Combine legend handles from sweeps and order blocks
    handles, labels = ax_main.get_legend_handles_labels()
    handles.extend(ob_legend_elements)
    ax_main.legend(handles=handles, loc='upper left', fontsize=7, ncol=2)

    # HTF mini chart formatting
    ax_htf.set_ylabel('')
    ax_htf.set_title('HTF Candles', fontsize=10, fontweight='bold')
    ax_htf.set_xticks([])
    ax_htf.set_yticks([])
    ax_htf.grid(True, alpha=0.2)

    # Market structure panel formatting
    ax_ms.set_ylabel('Timeframes', fontsize=10, fontweight='bold')
    ax_ms.set_ylim(-0.5, 3.5)
    ax_ms.set_yticks([0, 1, 2, 3])
    ax_ms.set_yticklabels([])
    ax_ms.grid(True, alpha=0.2, axis='x')
    ax_ms.set_xlim(-10, len(df))

    # Market structure legend
    legend_elements = [
        Line2D([0], [0], color=choch_bull_color, linewidth=8, label='CHoCH Bullish'),
        Line2D([0], [0], color=choch_bear_color, linewidth=8, label='CHoCH Bearish'),
        Line2D([0], [0], color=bos_bull_color, linewidth=8, label='BoS Bullish'),
        Line2D([0], [0], color=bos_bear_color, linewidth=8, label='BoS Bearish'),
    ]
    ax_ms.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=4)

    plt.tight_layout()

    # Save
    output_file = 'combined_market_structure_htf_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCombined chart saved as '{output_file}'")

    plt.show()


if __name__ == '__main__':
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_combined_chart(csv_file, num_candles=500, pivot_strength=15)
