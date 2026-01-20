"""
Standalone chart for Liquidity & Inducements indicator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import importlib.util

# Load Liquidity & inducements module
spec = importlib.util.spec_from_file_location("liquidity_inducements", "Liquidity & inducements.py")
liquidity_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(liquidity_module)

calculate_liquidity_data = liquidity_module.calculate_liquidity_data


def plot_liquidity_chart(csv_file, num_candles=500):
    """
    Plot candlestick chart with Liquidity & Inducements

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

    # Calculate liquidity data on full dataset with new features
    print("Calculating liquidity and inducements...")
    liquidity_data = calculate_liquidity_data(
        df,
        show_grabs=True,
        show_sweeps=True,
        show_equal_pivots=True,
        show_bsl_ssl=True,
        show_big_grabs=True,  # Enable Big Grabs
        show_turtle_soups=True,  # Enable Turtle Soups
        pivot_left=3,
        pivot_right=3,
        big_grab_pivot_left=10,  # Larger pivots for HTF grabs
        big_grab_pivot_right=10,
        turtle_soup_pivot_left=1,  # Smaller pivots for turtle soups
        turtle_soup_pivot_right=1,
        lookback=5
    )

    # Filter to last N candles for display
    start_idx = max(0, len(df) - num_candles)
    df_display = df.iloc[start_idx:].copy()
    df_display = df_display.reset_index(drop=True)

    print(f"Displaying last {len(df_display)} candles...")

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot candlesticks
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
        color = 'blue' if row['close'] >= row['open'] else 'red'

        # Draw candle body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['open'], row['close'])
        rect = Rectangle((idx - 0.4, body_bottom), 0.8, body_height,
                         facecolor=color, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)

        # Draw wicks
        ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=0.5)

    # Plot Grabs (filled zones with $$$ labels)
    for grab in liquidity_data['grabs']:
        if grab.grab_index >= start_idx and grab.pivot.index >= start_idx:
            pivot_idx = grab.pivot.index - start_idx
            grab_idx = grab.grab_index - start_idx

            if 0 <= pivot_idx < len(df_display) and 0 <= grab_idx < len(df_display):
                # Draw horizontal line
                ax.plot([pivot_idx, grab_idx], [grab.pivot.price, grab.pivot.price],
                       color='orange', linewidth=2, linestyle='--', alpha=0.7)

                # Draw filled zone
                if grab.pivot.type == 1:  # Bearish grab (above pivot high)
                    y_bottom = grab.pivot.price
                    y_top = grab.grab_price
                    label_y = grab.pivot.price
                    label_style = 'top'
                else:  # Bullish grab (below pivot low)
                    y_bottom = grab.grab_price
                    y_top = grab.pivot.price
                    label_y = grab.pivot.price
                    label_style = 'bottom'

                rect = Rectangle((pivot_idx, y_bottom), grab_idx - pivot_idx,
                               y_top - y_bottom, facecolor='orange', alpha=0.15)
                ax.add_patch(rect)

                # Add $$$ label
                mid_idx = (pivot_idx + grab_idx) / 2
                ax.text(mid_idx, label_y, '$$$', fontsize=10, color='orange',
                       fontweight='bold', ha='center', va=label_style)

    # Plot Sweeps ($ labels)
    for sweep in liquidity_data['sweeps']:
        if sweep.sweep_index >= start_idx and sweep.pivot.index >= start_idx:
            pivot_idx = sweep.pivot.index - start_idx
            sweep_idx = sweep.sweep_index - start_idx

            if 0 <= pivot_idx < len(df_display) and 0 <= sweep_idx < len(df_display):
                color = 'teal' if sweep.is_bullish else 'red'

                # Draw horizontal line
                ax.plot([pivot_idx, sweep_idx], [sweep.pivot.price, sweep.pivot.price],
                       color=color, linewidth=1.5, linestyle=':', alpha=0.8)

                # Add $ label
                mid_idx = (pivot_idx + sweep_idx) / 2
                label_y = sweep.pivot.price
                label_style = 'bottom' if sweep.is_bullish else 'top'
                ax.text(mid_idx, label_y, '$', fontsize=9, color=color,
                       fontweight='bold', ha='center', va=label_style)

    # Plot Equal Pivots ($$$ for liquidity, IDM for inducement)
    for eq_pivot in liquidity_data['equal_pivots']:
        if eq_pivot.pivot1.index >= start_idx and eq_pivot.pivot2.index >= start_idx:
            idx1 = eq_pivot.pivot1.index - start_idx
            idx2 = eq_pivot.pivot2.index - start_idx

            if 0 <= idx1 < len(df_display) and 0 <= idx2 < len(df_display):
                # Calculate average price
                avg_price = (eq_pivot.pivot1.price + eq_pivot.pivot2.price) / 2

                if eq_pivot.is_liquidity:
                    color = 'orange'
                    label = '$$$'
                else:
                    color = 'teal' if eq_pivot.is_bullish else 'red'
                    label = 'IDM'

                # Draw line connecting equal pivots
                ax.plot([idx1, idx2], [eq_pivot.pivot1.price, eq_pivot.pivot2.price],
                       color=color, linewidth=2, linestyle=':', alpha=0.7)

                # Add label
                mid_idx = (idx1 + idx2) / 2
                label_style = 'bottom' if eq_pivot.pivot1.type == -1 else 'top'
                ax.text(mid_idx, avg_price, label, fontsize=10, color=color,
                       fontweight='bold', ha='center', va=label_style)

    # Plot BSL (Buyside Liquidity)
    for bsl in liquidity_data['bsl']:
        if bsl.pivot.index >= start_idx:
            pivot_idx = bsl.pivot.index - start_idx

            if 0 <= pivot_idx < len(df_display):
                # Draw horizontal line extending to the right
                ax.axhline(y=bsl.pivot.price, xmin=pivot_idx/len(df_display), xmax=1,
                          color='teal', linewidth=1.5, linestyle='--', alpha=0.6)

                # Add label
                ax.text(len(df_display) - 5, bsl.pivot.price, 'BSL', fontsize=9,
                       color='teal', fontweight='bold', ha='right', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot SSL (Sellside Liquidity)
    for ssl in liquidity_data['ssl']:
        if ssl.pivot.index >= start_idx:
            pivot_idx = ssl.pivot.index - start_idx

            if 0 <= pivot_idx < len(df_display):
                # Draw horizontal line extending to the right
                ax.axhline(y=ssl.pivot.price, xmin=pivot_idx/len(df_display), xmax=1,
                          color='red', linewidth=1.5, linestyle='--', alpha=0.6)

                # Add label
                ax.text(len(df_display) - 5, ssl.pivot.price, 'SSL', fontsize=9,
                       color='red', fontweight='bold', ha='right', va='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Plot Big Grabs (larger zones with BIG $$$ labels)
    for big_grab in liquidity_data.get('big_grabs', []):
        if big_grab.grab_index >= start_idx and big_grab.pivot.index >= start_idx:
            pivot_idx = big_grab.pivot.index - start_idx
            grab_idx = big_grab.grab_index - start_idx

            if 0 <= pivot_idx < len(df_display) and 0 <= grab_idx < len(df_display):
                # Draw horizontal line
                ax.plot([pivot_idx, grab_idx], [big_grab.pivot.price, big_grab.pivot.price],
                       color='cyan', linewidth=3, linestyle='--', alpha=0.8)

                # Draw filled zone
                if big_grab.pivot.type == 1:  # Bearish big grab
                    y_bottom = big_grab.pivot.price
                    y_top = big_grab.grab_price
                    label_y = big_grab.pivot.price
                    label_style = 'top'
                else:  # Bullish big grab
                    y_bottom = big_grab.grab_price
                    y_top = big_grab.pivot.price
                    label_y = big_grab.pivot.price
                    label_style = 'bottom'

                rect = Rectangle((pivot_idx, y_bottom), grab_idx - pivot_idx,
                               y_top - y_bottom, facecolor='cyan', alpha=0.2)
                ax.add_patch(rect)

                # Add BIG $$$ label
                mid_idx = (pivot_idx + grab_idx) / 2
                ax.text(mid_idx, label_y, 'BIG $$$', fontsize=11, color='cyan',
                       fontweight='bold', ha='center', va=label_style)

    # Plot Turtle Soups (failed breakout patterns)
    for turtle_soup in liquidity_data.get('turtle_soups', []):
        if turtle_soup.break_index >= start_idx and turtle_soup.pivot.index >= start_idx:
            pivot_idx = turtle_soup.pivot.index - start_idx
            break_idx = turtle_soup.break_index - start_idx
            reversal_idx = turtle_soup.reversal_index - start_idx

            if 0 <= pivot_idx < len(df_display) and 0 <= break_idx < len(df_display):
                color = 'lime' if turtle_soup.is_bullish else 'magenta'

                # Draw line from pivot to break
                ax.plot([pivot_idx, break_idx], [turtle_soup.pivot.price, turtle_soup.pivot.price],
                       color=color, linewidth=2, linestyle='-.', alpha=0.7)

                # Draw arrow showing reversal
                if reversal_idx < len(df_display):
                    ax.annotate('', xy=(reversal_idx, turtle_soup.pivot.price),
                               xytext=(break_idx, turtle_soup.break_price),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))

                # Add TURTLE label
                mid_idx = (pivot_idx + break_idx) / 2
                label_style = 'bottom' if turtle_soup.is_bullish else 'top'
                ax.text(mid_idx, turtle_soup.pivot.price, 'TURTLE', fontsize=9, color=color,
                       fontweight='bold', ha='center', va=label_style)

    # Formatting
    ax.set_xlim(-0.5, len(df_display) - 0.5)
    ax.set_xlabel('Candle Index', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)

    # Display trend in title
    trend_text = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}
    trend = liquidity_data.get('trend', 0)
    ax.set_title(f'Liquidity & Inducements - Last {num_candles} Candles | Trend: {trend_text[trend]}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='orange', alpha=0.3, edgecolor='orange', label='Grabs ($$$)'),
        mpatches.Patch(facecolor='cyan', alpha=0.3, edgecolor='cyan', label='Big Grabs (BIG $$$)'),
        mpatches.Patch(facecolor='teal', alpha=0.5, label='Bullish Sweep ($)'),
        mpatches.Patch(facecolor='red', alpha=0.5, label='Bearish Sweep ($)'),
        mpatches.Patch(facecolor='orange', alpha=0.5, label='Equal Pivots - Liquidity ($$$)'),
        mpatches.Patch(facecolor='teal', alpha=0.5, label='Equal Pivots - Inducement (IDM)'),
        plt.Line2D([0], [0], color='lime', linewidth=2, linestyle='-.', label='Bullish Turtle Soup'),
        plt.Line2D([0], [0], color='magenta', linewidth=2, linestyle='-.', label='Bearish Turtle Soup'),
        plt.Line2D([0], [0], color='teal', linewidth=2, linestyle='--', label='BSL'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='SSL')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    output_file = 'liq_candlestick_chart.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nLiquidity & Inducements chart saved as '{output_file}'")
    print(f"  - Market Trend: {trend_text[trend]}")
    print(f"  - Grabs: {len(liquidity_data['grabs'])}")
    print(f"  - Big Grabs: {len(liquidity_data.get('big_grabs', []))}")
    print(f"  - Sweeps: {len(liquidity_data['sweeps'])}")
    print(f"  - Turtle Soups: {len(liquidity_data.get('turtle_soups', []))}")
    print(f"  - Equal Pivots: {len(liquidity_data['equal_pivots'])}")
    print(f"  - BSL: {len(liquidity_data['bsl'])}")
    print(f"  - SSL: {len(liquidity_data['ssl'])}")


if __name__ == "__main__":
    plot_liquidity_chart("PEPPERSTONE_XAUUSD, 5.csv", num_candles=500)
