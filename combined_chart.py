"""
Combined Market Structure MTF + HTF Bias + Order Blocks + FVG + Liquidity + Sessions + ICT + CRT + MTF Trends Chart

This script combines:
1. Market Structure MTF Trend indicator (multi-timeframe trend panel)
2. HTF Bias indicator (HTF candles, sweeps, and bias)
3. Order Blocks (15min and 1H)
4. Fair Value Gaps (FVG) - Demand and Supply
5. Liquidity & Inducements (Grabs, BSL/SSL)
6. Session Levels (Asia, London, NY, PDH/PDL)
7. ICT Levels (Daily 50% Line + Killzone Sessions)
8. CRT - Candles are Ranges Theory (1H CRT levels + Turtle Soup signals)
9. Multi-Timeframe Trends Panel (5m, 15m, 1H, 4H, 1D)

Display:
- Main chart: Candlesticks with HTF levels, sweep markers, Order Blocks, FVGs, Liquidity zones, Session levels, ICT levels, and CRT ranges
- Top right: Mini HTF candles (1H, 4H, Daily)
- Bottom left: Market structure trend indicators
- Bottom right: MTF Trends Panel (Smart Money Zones)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import sys
import os
import importlib.util

# Import market structure functions using importlib for filenames with special characters
sys.path.append(os.path.dirname(__file__))

# Load Market Structure MTF Trend [Pt].py
spec1 = importlib.util.spec_from_file_location("market_structure", "Market Structure MTF Trend [Pt].py")
market_structure = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(market_structure)
calculate_market_structure = market_structure.calculate_market_structure
plot_trend_line = market_structure.plot_trend_line

# Load CandelaCharts - HTF Sweeps.py
spec2 = importlib.util.spec_from_file_location("htf_sweeps", "CandelaCharts - HTF Sweeps.py")
htf_sweeps = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(htf_sweeps)
calculate_htf_data = htf_sweeps.calculate_htf_data
HTFCandle = htf_sweeps.HTFCandle
Sweep = htf_sweeps.Sweep

# Load MirPapa-ICT-HTF- FVG OB Threeple (EN).py
spec3 = importlib.util.spec_from_file_location("order_blocks", "MirPapa-ICT-HTF- FVG OB Threeple (EN).py")
order_blocks_module = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(order_blocks_module)
calculate_htf_order_blocks = order_blocks_module.calculate_htf_order_blocks

# Load Smart Money Concept [TradingFinder] Major Minor OB + FVG (SMC).py
spec4 = importlib.util.spec_from_file_location("fvg_detection", "Smart Money Concept [TradingFinder] Major Minor OB + FVG (SMC).py")
fvg_detection_module = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(fvg_detection_module)
calculate_fvgs_for_chart = fvg_detection_module.calculate_fvgs_for_chart

# Load Liquidity & inducements.py
spec5 = importlib.util.spec_from_file_location("liquidity_inducements", "Liquidity & inducements.py")
liquidity_module = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(liquidity_module)
calculate_liquidity_data = liquidity_module.calculate_liquidity_data

# Load Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py
spec6 = importlib.util.spec_from_file_location("smart_money_zones", "Smart_Money_Zones__FVG___OB____MTF_Trend_Panel.py")
smz_module = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(smz_module)
calculate_mtf_trends = smz_module.calculate_mtf_trends

# Load SW's AsiaLondon HL's.py
spec7 = importlib.util.spec_from_file_location("session_levels", "SW's AsiaLondon HL's.py")
session_levels_module = importlib.util.module_from_spec(spec7)
spec7.loader.exec_module(session_levels_module)
detect_session_levels = session_levels_module.detect_session_levels
detect_pdh_pdl = session_levels_module.detect_pdh_pdl

# Load ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.py
spec8 = importlib.util.spec_from_file_location("ict_levels", "ICT_Customizable_50__Line___DailyAsiaLondonNew_York_HighLow___True_Day_Open.py")
ict_levels_module = importlib.util.module_from_spec(spec8)
spec8.loader.exec_module(ict_levels_module)
detect_daily_levels = ict_levels_module.detect_daily_levels
detect_killzone_sessions = ict_levels_module.detect_killzone_sessions

# Load Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.py
spec9 = importlib.util.spec_from_file_location("crt_module", "Ighodalo_Gold_-_CRT__Candles_are_ranges_theory_.py")
crt_module = importlib.util.module_from_spec(spec9)
spec9.loader.exec_module(crt_module)
detect_crt_ranges = crt_module.detect_crt_ranges
detect_turtle_soup_signals = crt_module.detect_turtle_soup_signals
filter_overlapping_crts = crt_module.filter_overlapping_crts


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

    # Add index column for ALL data (needed for calculations)
    df['index'] = range(len(df))

    print(f"Analyzing with {len(df)} candles for MTF calculations...")

    # ========================================================================
    # CALCULATE MARKET STRUCTURE FOR MULTIPLE TIMEFRAMES
    # Use ALL data for proper MTF calculations
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
    # NOW FILTER TO LAST N CANDLES FOR DISPLAY
    # ========================================================================
    print(f"Filtering to last {num_candles} candles for display...")

    # Get the starting index for display
    start_idx = max(0, len(df) - num_candles)

    # Create display dataframe
    df_display = df.iloc[start_idx:].copy()
    df_display['display_index'] = range(len(df_display))

    # Filter trends to display range
    trend_tf1_display = trend_tf1[start_idx:]
    trend_tf2_display = trend_tf2[start_idx:]
    trend_tf3_display = trend_tf3[start_idx:]
    trend_tf4_display = trend_tf4[start_idx:]

    # Filter signals to display range
    choch_tf1_display = [(idx - start_idx, direction, price) for idx, direction, price in choch_tf1 if idx >= start_idx]
    bos_tf1_display = [(idx - start_idx, direction, price) for idx, direction, price in bos_tf1 if idx >= start_idx]

    choch_tf2_display = [(idx - start_idx, direction, price) for idx, direction, price in choch_tf2 if idx >= start_idx]
    bos_tf2_display = [(idx - start_idx, direction, price) for idx, direction, price in bos_tf2 if idx >= start_idx]

    choch_tf3_display = [(idx - start_idx, direction, price) for idx, direction, price in choch_tf3 if idx >= start_idx]
    bos_tf3_display = [(idx - start_idx, direction, price) for idx, direction, price in bos_tf3 if idx >= start_idx]

    choch_tf4_display = [(idx - start_idx, direction, price) for idx, direction, price in choch_tf4 if idx >= start_idx]
    bos_tf4_display = [(idx - start_idx, direction, price) for idx, direction, price in bos_tf4 if idx >= start_idx]

    # ========================================================================
    # CALCULATE HTF DATA on full dataset for proper context
    # ========================================================================
    print("Calculating HTF candles and sweeps on full dataset...")
    htf_data_full = calculate_htf_data(df, timeframes_minutes={'1H': 60, '4H': 240, 'Daily': 1440})

    # Filter HTF sweeps to display range
    htf_data = {}
    for tf_name, data in htf_data_full.items():
        htf_data[tf_name] = {
            'candles': data['candles'],  # Keep all candles for level drawing
            'sweeps': [s for s in data['sweeps'] if s.index >= start_idx],  # Filter sweeps to visible range
            'bias': data['bias']
        }
        # Adjust sweep indices for display
        for sweep in htf_data[tf_name]['sweeps']:
            sweep.index -= start_idx

    # ========================================================================
    # CALCULATE ORDER BLOCKS on full dataset
    # ========================================================================
    print("Calculating Order Blocks for 15min and 1H on full dataset...")
    order_blocks_data_full = calculate_htf_order_blocks(df, timeframes_minutes={'15min': 15, '1H': 60})

    # Filter OBs to those visible in display range
    order_blocks_data = {}
    for tf_name, obs in order_blocks_data_full.items():
        visible_obs = []
        for ob in obs:
            # Check if OB overlaps with display range
            if ob.end_idx >= start_idx and ob.start_idx < len(df):
                # Adjust indices for display
                ob_display = ob
                ob_display.start_idx = max(0, ob.start_idx - start_idx)
                ob_display.end_idx = min(len(df_display), ob.end_idx - start_idx)
                ob_display.created_at = ob.created_at - start_idx if ob.created_at >= start_idx else 0
                visible_obs.append(ob_display)
        order_blocks_data[tf_name] = visible_obs

    # ========================================================================
    # CALCULATE FAIR VALUE GAPS (FVG) on full dataset
    # ========================================================================
    print("Calculating Fair Value Gaps (FVG) on full dataset...")
    fvg_data_full = calculate_fvgs_for_chart(df, show_demand=True, show_supply=True, filter_type='Defensive')

    # Filter FVGs to those visible in display range
    fvg_data = {'bullish': [], 'bearish': [], 'all': []}
    for fvg in fvg_data_full['bullish']:
        if fvg.end_idx >= start_idx and fvg.start_idx < len(df):
            fvg.start_idx = max(0, fvg.start_idx - start_idx)
            fvg.end_idx = min(len(df_display), fvg.end_idx - start_idx)
            fvg.created_at = fvg.created_at - start_idx if fvg.created_at >= start_idx else 0
            fvg_data['bullish'].append(fvg)

    for fvg in fvg_data_full['bearish']:
        if fvg.end_idx >= start_idx and fvg.start_idx < len(df):
            fvg.start_idx = max(0, fvg.start_idx - start_idx)
            fvg.end_idx = min(len(df_display), fvg.end_idx - start_idx)
            fvg.created_at = fvg.created_at - start_idx if fvg.created_at >= start_idx else 0
            fvg_data['bearish'].append(fvg)

    # ========================================================================
    # CALCULATE LIQUIDITY & INDUCEMENTS on full dataset
    # ========================================================================
    print("Calculating Liquidity & Inducements on full dataset...")
    liquidity_data_full = calculate_liquidity_data(
        df,
        show_grabs=True,
        show_sweeps=True,
        show_equal_pivots=True,
        show_bsl_ssl=True,
        pivot_left=3,
        pivot_right=3,
        lookback=5
    )

    # Filter liquidity data to display range
    liquidity_data = {
        'grabs': [],
        'sweeps': [],
        'equal_pivots': [],
        'bsl': [],
        'ssl': []
    }

    # Filter grabs
    for grab in liquidity_data_full['grabs']:
        if grab.grab_index >= start_idx and grab.pivot.index >= start_idx:
            grab.pivot.index -= start_idx
            grab.grab_index -= start_idx
            liquidity_data['grabs'].append(grab)

    # Filter sweeps
    for sweep in liquidity_data_full['sweeps']:
        if sweep.sweep_index >= start_idx and sweep.pivot.index >= start_idx:
            sweep.pivot.index -= start_idx
            sweep.sweep_index -= start_idx
            liquidity_data['sweeps'].append(sweep)

    # Filter equal pivots
    for eq_pivot in liquidity_data_full['equal_pivots']:
        if eq_pivot.pivot1.index >= start_idx and eq_pivot.pivot2.index >= start_idx:
            eq_pivot.pivot1.index -= start_idx
            eq_pivot.pivot2.index -= start_idx
            liquidity_data['equal_pivots'].append(eq_pivot)

    # Filter BSL/SSL
    for bsl in liquidity_data_full['bsl']:
        if bsl.pivot.index >= start_idx:
            bsl.pivot.index -= start_idx
            liquidity_data['bsl'].append(bsl)

    for ssl in liquidity_data_full['ssl']:
        if ssl.pivot.index >= start_idx:
            ssl.pivot.index -= start_idx
            liquidity_data['ssl'].append(ssl)

    # ========================================================================
    # CALCULATE MTF TRENDS for Smart Money Zones panel
    # ========================================================================
    print("Calculating Multi-Timeframe Trends...")
    mtf_trends = calculate_mtf_trends(
        df,
        timeframes_minutes={'5m': 5, '15m': 15, '1H': 60, '4H': 240, '1D': 1440},
        ma_period=50
    )

    # ========================================================================
    # CALCULATE SESSION LEVELS (Asia, London, NY, PDH/PDL)
    # ========================================================================
    print("Calculating Asia/London/NY session levels...")
    session_levels_full = detect_session_levels(df, timezone_offset=0)

    print("Calculating PDH/PDL...")
    pdh_pdl_levels_full = detect_pdh_pdl(df)

    # Filter session levels to display range
    session_levels = {}
    for key, levels in session_levels_full.items():
        session_levels[key] = [level for level in levels if level.end_idx >= start_idx]
        # Adjust indices for display
        for level in session_levels[key]:
            level.start_idx = max(0, level.start_idx - start_idx)
            level.end_idx = min(len(df_display), level.end_idx - start_idx)

    pdh_pdl_levels = {}
    for key, levels in pdh_pdl_levels_full.items():
        pdh_pdl_levels[key] = [level for level in levels if level.end_idx >= start_idx]
        # Adjust indices for display
        for level in pdh_pdl_levels[key]:
            level.start_idx = max(0, level.start_idx - start_idx)
            level.end_idx = min(len(df_display), level.end_idx - start_idx)

    # ========================================================================
    # CALCULATE ICT LEVELS (Daily High/Low/50% + Killzones)
    # ========================================================================
    print("Calculating ICT daily levels (50% line)...")
    daily_levels_full = detect_daily_levels(df)

    print("Calculating ICT killzone sessions...")
    killzone_levels_full = detect_killzone_sessions(df)

    # Filter ICT daily levels to display range
    daily_levels = [level for level in daily_levels_full if level.end_idx >= start_idx]
    for level in daily_levels:
        level.start_idx = max(0, level.start_idx - start_idx)
        level.end_idx = min(len(df_display), level.end_idx - start_idx)
        if level.market_open_idx >= start_idx:
            level.market_open_idx = level.market_open_idx - start_idx

    # Filter killzone levels to display range
    killzone_levels = {}
    for key, levels in killzone_levels_full.items():
        killzone_levels[key] = [level for level in levels if level.end_idx >= start_idx]
        # Adjust indices for display
        for level in killzone_levels[key]:
            level.start_idx = max(0, level.start_idx - start_idx)
            level.end_idx = min(len(df_display), level.end_idx - start_idx)

    # ========================================================================
    # CALCULATE CRT RANGES (Candles are Ranges Theory)
    # ========================================================================
    print("Detecting CRT ranges (1H timeframe)...")
    crt_ranges_full = detect_crt_ranges(df, lookback=20, timeframe_minutes=60)
    crt_ranges_full = filter_overlapping_crts(crt_ranges_full, enable_overlapping=False)

    print("Detecting Turtle Soup signals...")
    turtle_soup_signals = detect_turtle_soup_signals(df, crt_ranges_full, atr_multiplier=0.1, atr_period=14)

    # Filter CRT ranges to display range (only last 2 active or recent)
    crt_ranges = [crt for crt in crt_ranges_full if crt.is_active or crt.end_idx >= start_idx][-2:]
    for crt in crt_ranges:
        crt.start_idx = max(0, crt.start_idx - start_idx)
        crt.end_idx = min(len(df_display), crt.end_idx - start_idx)

    # Filter signals to display range
    buy_signals = [s for s in turtle_soup_signals['buy'] if s['idx'] >= start_idx]
    sell_signals = [s for s in turtle_soup_signals['sell'] if s['idx'] >= start_idx]
    for sig in buy_signals:
        sig['idx'] = sig['idx'] - start_idx
    for sig in sell_signals:
        sig['idx'] = sig['idx'] - start_idx

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

    # MTF Trend Panel
    ax_mtf = fig.add_subplot(gs[1, 8:])  # Bottom right corner
    ax_mtf.set_facecolor('#1a1a1a')  # Dark background
    ax_mtf.set_xlim(0, 1)
    ax_mtf.set_ylim(0, 1)
    ax_mtf.axis('off')

    # ========================================================================
    # PLOT MAIN CANDLESTICKS (using display data)
    # ========================================================================
    width = 0.8
    for idx in range(len(df_display)):
        row = df_display.iloc[idx]
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
    # PLOT FAIR VALUE GAPS (FVG)
    # ========================================================================
    # Colors matching Smart Money Concept Pine Script
    fvg_bull_color = 'cyan'  # Demand FVG - light blue/cyan
    fvg_bear_color = 'orange'  # Supply FVG - orange
    fvg_alpha = 0.15  # More transparent than OBs

    # Plot Bullish FVGs (Demand)
    for fvg in fvg_data['bullish']:
        # Only show unmitigated or recently mitigated FVGs
        if not fvg.mitigated or (fvg.end_idx - fvg.start_idx) < 150:
            fvg_rect = Rectangle(
                (fvg.start_idx, fvg.bottom),
                fvg.end_idx - fvg.start_idx,
                fvg.top - fvg.bottom,
                facecolor=fvg_bull_color,
                edgecolor=fvg_bull_color,
                alpha=fvg_alpha,
                linewidth=0.5,
                linestyle='--',
                zorder=0.3
            )
            ax_main.add_patch(fvg_rect)

    # Plot Bearish FVGs (Supply)
    for fvg in fvg_data['bearish']:
        # Only show unmitigated or recently mitigated FVGs
        if not fvg.mitigated or (fvg.end_idx - fvg.start_idx) < 150:
            fvg_rect = Rectangle(
                (fvg.start_idx, fvg.bottom),
                fvg.end_idx - fvg.start_idx,
                fvg.top - fvg.bottom,
                facecolor=fvg_bear_color,
                edgecolor=fvg_bear_color,
                alpha=fvg_alpha,
                linewidth=0.5,
                linestyle='--',
                zorder=0.3
            )
            ax_main.add_patch(fvg_rect)

    # Add FVG legend entries
    fvg_legend_elements = [
        mpatches.Patch(facecolor=fvg_bull_color, edgecolor=fvg_bull_color,
                      alpha=fvg_alpha, linestyle='--', label='Demand FVG'),
        mpatches.Patch(facecolor=fvg_bear_color, edgecolor=fvg_bear_color,
                      alpha=fvg_alpha, linestyle='--', label='Supply FVG'),
    ]

    # ========================================================================
    # PLOT LIQUIDITY & INDUCEMENTS
    # ========================================================================
    # Plot Grabs (filled zones with $$$ labels)
    for grab in liquidity_data['grabs']:
        pivot_idx = grab.pivot.index
        grab_idx = grab.grab_index

        if 0 <= pivot_idx < len(df_display) and 0 <= grab_idx < len(df_display):
            # Draw horizontal line
            ax_main.plot([pivot_idx, grab_idx], [grab.pivot.price, grab.pivot.price],
                        color='orange', linewidth=1.5, linestyle=':', alpha=0.6, zorder=2)

            # Draw filled zone
            if grab.pivot.type == 1:  # Bearish grab (above pivot high)
                y_bottom = grab.pivot.price
                y_top = grab.grab_price
                label_y = grab.pivot.price
                va = 'top'
            else:  # Bullish grab (below pivot low)
                y_bottom = grab.grab_price
                y_top = grab.pivot.price
                label_y = grab.pivot.price
                va = 'bottom'

            grab_rect = Rectangle((pivot_idx, y_bottom), grab_idx - pivot_idx,
                                 y_top - y_bottom, facecolor='orange', alpha=0.1, zorder=0.5)
            ax_main.add_patch(grab_rect)

            # Add $$$ label
            mid_idx = (pivot_idx + grab_idx) / 2
            ax_main.text(mid_idx, label_y, '$$$', fontsize=8, color='orange',
                        fontweight='bold', ha='center', va=va, zorder=5)

    # Plot Sweeps ($ labels with lines)
    for sweep in liquidity_data['sweeps']:
        pivot_idx = sweep.pivot.index
        sweep_idx = sweep.sweep_index

        if 0 <= pivot_idx < len(df_display) and 0 <= sweep_idx < len(df_display):
            color = 'teal' if sweep.is_bullish else 'red'

            # Draw horizontal line
            ax_main.plot([pivot_idx, sweep_idx], [sweep.pivot.price, sweep.pivot.price],
                        color=color, linewidth=1, linestyle=':', alpha=0.6, zorder=2)

            # Add $ label
            mid_idx = (pivot_idx + sweep_idx) / 2
            label_y = sweep.pivot.price
            va = 'bottom' if sweep.is_bullish else 'top'
            ax_main.text(mid_idx, label_y, '$', fontsize=7, color=color,
                        fontweight='bold', ha='center', va=va, zorder=5)

    # Plot Equal Pivots ($$$ for liquidity, IDM for inducement)
    # Only show a subset to avoid cluttering
    equal_pivots_to_show = [ep for ep in liquidity_data['equal_pivots']
                           if not ep.is_liquidity or ep.pivot1.index > len(df_display) - 200][:10]

    for eq_pivot in equal_pivots_to_show:
        idx1 = eq_pivot.pivot1.index
        idx2 = eq_pivot.pivot2.index

        if 0 <= idx1 < len(df_display) and 0 <= idx2 < len(df_display):
            # Calculate average price
            avg_price = (eq_pivot.pivot1.price + eq_pivot.pivot2.price) / 2

            if eq_pivot.is_liquidity:
                color = 'orange'
                label = '$$$'
                alpha = 0.4
            else:
                color = 'teal' if eq_pivot.is_bullish else 'red'
                label = 'IDM'
                alpha = 0.5

            # Draw line connecting equal pivots
            ax_main.plot([idx1, idx2], [eq_pivot.pivot1.price, eq_pivot.pivot2.price],
                        color=color, linewidth=1.5, linestyle=':', alpha=alpha, zorder=1)

            # Add label (smaller and less prominent)
            mid_idx = (idx1 + idx2) / 2
            va = 'bottom' if eq_pivot.pivot1.type == -1 else 'top'
            ax_main.text(mid_idx, avg_price, label, fontsize=6, color=color,
                        fontweight='bold', ha='center', va=va, alpha=0.8, zorder=5)

    # Plot BSL (Buyside Liquidity) - extend to the right
    for bsl in liquidity_data['bsl']:
        pivot_idx = bsl.pivot.index
        if 0 <= pivot_idx < len(df_display):
            ax_main.axhline(y=bsl.pivot.price, xmin=pivot_idx/len(df_display), xmax=1,
                          color='teal', linewidth=1.2, linestyle='--', alpha=0.5, zorder=1)
            ax_main.text(len(df_display) - 10, bsl.pivot.price, 'BSL', fontsize=7,
                        color='teal', fontweight='bold', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6), zorder=5)

    # Plot SSL (Sellside Liquidity) - extend to the right
    for ssl in liquidity_data['ssl']:
        pivot_idx = ssl.pivot.index
        if 0 <= pivot_idx < len(df_display):
            ax_main.axhline(y=ssl.pivot.price, xmin=pivot_idx/len(df_display), xmax=1,
                          color='red', linewidth=1.2, linestyle='--', alpha=0.5, zorder=1)
            ax_main.text(len(df_display) - 10, ssl.pivot.price, 'SSL', fontsize=7,
                        color='red', fontweight='bold', ha='right', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6), zorder=5)

    # Add liquidity legend entries
    liquidity_legend_elements = [
        mpatches.Patch(facecolor='orange', alpha=0.15, edgecolor='orange',
                      linestyle=':', label='Grabs ($$$)'),
        plt.Line2D([0], [0], color='teal', linewidth=1, linestyle=':', label='Bullish Sweep ($)'),
        plt.Line2D([0], [0], color='red', linewidth=1, linestyle=':', label='Bearish Sweep ($)'),
        plt.Line2D([0], [0], color='orange', linewidth=1.5, linestyle=':', label='Equal Pivots ($$$)'),
        plt.Line2D([0], [0], color='teal', linewidth=1.5, linestyle=':', label='Inducement (IDM)'),
        plt.Line2D([0], [0], color='teal', linewidth=1.5, linestyle='--', label='BSL'),
        plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='--', label='SSL')
    ]

    # ========================================================================
    # PLOT SESSION LEVELS (Asia, London, NY, PDH/PDL)
    # ========================================================================
    # Only plot most recent session levels to avoid clutter
    # Asia Session - Purple
    for level in session_levels.get('asia_high', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#4a148c', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    for level in session_levels.get('asia_low', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#4a148c', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    # London Session - Blue
    for level in session_levels.get('london_high', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#0c3299', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    for level in session_levels.get('london_low', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#0c3299', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    # NY Session - Orange
    for level in session_levels.get('ny_high', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#fb7f1f', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    for level in session_levels.get('ny_low', [])[-2:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#fb7f1f', linewidth=1.5, linestyle='-', alpha=0.6, zorder=3)

    # PDH/PDL - Dark Green (most prominent)
    for level in pdh_pdl_levels.get('pdh', []):
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#00332a', linewidth=2, linestyle='-', alpha=0.8, zorder=4)
        # Add small label
        ax_main.text(level.end_idx - 5, level.price, 'PDH ', fontsize=6,
                    color='#00332a', va='bottom', ha='right', fontweight='bold', zorder=5)

    for level in pdh_pdl_levels.get('pdl', []):
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#00332a', linewidth=2, linestyle='-', alpha=0.8, zorder=4)
        # Add small label
        ax_main.text(level.end_idx - 5, level.price, 'PDL ', fontsize=6,
                    color='#00332a', va='top', ha='right', fontweight='bold', zorder=5)

    for level in pdh_pdl_levels.get('pd_mid', []):
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#00332a', linewidth=1, linestyle='--', alpha=0.6, zorder=4)

    # Add session levels legend entries
    session_legend_elements = [
        plt.Line2D([0], [0], color='#4a148c', linewidth=1.5, label='Asia Session'),
        plt.Line2D([0], [0], color='#0c3299', linewidth=1.5, label='London Session'),
        plt.Line2D([0], [0], color='#fb7f1f', linewidth=1.5, label='NY Session'),
        plt.Line2D([0], [0], color='#00332a', linewidth=2, label='PDH/PDL')
    ]

    # ========================================================================
    # PLOT ICT LEVELS (Daily 50% Line + Killzone Sessions)
    # ========================================================================
    # Plot daily 50% levels (last 1 for clarity)
    for daily in daily_levels[-1:]:
        # 50% Line (cyan, most prominent)
        ax_main.plot([daily.start_idx, daily.end_idx], [daily.mid, daily.mid],
                    color='cyan', linewidth=1.8, linestyle=':', alpha=0.7, zorder=3.5)
        ax_main.text(daily.end_idx - 10, daily.mid, '50% ', fontsize=7,
                    color='cyan', va='center', ha='right', fontweight='bold', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

        # Daily High (lighter red to avoid confusion with existing red)
        ax_main.plot([daily.start_idx, daily.end_idx], [daily.high, daily.high],
                    color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.4, zorder=2.8)

        # Daily Low (lighter red)
        ax_main.plot([daily.start_idx, daily.end_idx], [daily.low, daily.low],
                    color='#ff6b6b', linewidth=1, linestyle=':', alpha=0.4, zorder=2.8)

    # Plot ICT killzone levels (only last 1 session of each for clarity)
    # These are more subtle than the main session levels
    # Asia Killzone - light blue
    for level in killzone_levels.get('asia_high', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#6ba3ff', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    for level in killzone_levels.get('asia_low', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#6ba3ff', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    # London Killzone - light orange
    for level in killzone_levels.get('london_high', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#ffb366', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    for level in killzone_levels.get('london_low', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#ffb366', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    # NY Killzone - light green
    for level in killzone_levels.get('ny_high', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#66ffb3', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    for level in killzone_levels.get('ny_low', [])[-1:]:
        ax_main.plot([level.start_idx, level.end_idx], [level.price, level.price],
                    color='#66ffb3', linewidth=1, linestyle='-', alpha=0.3, zorder=2.5)

    # Add ICT levels legend entries (simpler to avoid clutter)
    ict_legend_elements = [
        plt.Line2D([0], [0], color='cyan', linewidth=1.8, linestyle=':', label='ICT 50% Line'),
        plt.Line2D([0], [0], color='#6ba3ff', linewidth=1, alpha=0.3, label='ICT Killzones')
    ]

    # ========================================================================
    # PLOT CRT RANGES (Candles are Ranges Theory)
    # ========================================================================
    # Plot CRT high/low/mid levels (subtle styling to avoid clutter)
    for crt in crt_ranges:
        # CRT High (dark gray)
        ax_main.plot([crt.start_idx, crt.end_idx], [crt.high, crt.high],
                    color='#2d2d2d', linewidth=1.3, linestyle='-', alpha=0.6, zorder=3.2)
        ax_main.text(crt.end_idx - 5, crt.high, 'CRTH ', fontsize=6,
                    color='#2d2d2d', va='bottom', ha='right', fontweight='bold', zorder=5)

        # CRT Low (dark gray)
        ax_main.plot([crt.start_idx, crt.end_idx], [crt.low, crt.low],
                    color='#2d2d2d', linewidth=1.3, linestyle='-', alpha=0.6, zorder=3.2)
        ax_main.text(crt.end_idx - 5, crt.low, 'CRTL ', fontsize=6,
                    color='#2d2d2d', va='top', ha='right', fontweight='bold', zorder=5)

        # CRT Midpoint (light gray dashed)
        ax_main.plot([crt.start_idx, crt.end_idx], [crt.mid, crt.mid],
                    color='#808080', linewidth=1, linestyle='--', alpha=0.4, zorder=3.1)

    # Plot Turtle Soup signals (if any in visible range)
    for sig in buy_signals:
        if 0 <= sig['idx'] < len(df_display):
            ax_main.scatter(sig['idx'], sig['price'], marker='^', s=120,
                          color='lime', edgecolors='black', linewidth=1, zorder=6, alpha=0.8)

    for sig in sell_signals:
        if 0 <= sig['idx'] < len(df_display):
            ax_main.scatter(sig['idx'], sig['price'], marker='v', s=120,
                          color='magenta', edgecolors='black', linewidth=1, zorder=6, alpha=0.8)

    # Add CRT legend entries
    crt_legend_elements = [
        plt.Line2D([0], [0], color='#2d2d2d', linewidth=1.3, label='CRT H/L'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='lime',
                  markersize=8, label='Turtle Soup Buy', markeredgecolor='black', linestyle='None'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='magenta',
                  markersize=8, label='Turtle Soup Sell', markeredgecolor='black', linestyle='None')
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
    # PLOT CHoCH AND BoS LABELS (using display data)
    # ========================================================================
    for idx, direction, price in choch_tf3_display:
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
    # PLOT MARKET STRUCTURE TRENDS (using display data)
    # ========================================================================
    choch_bull_color = 'darkgreen'
    choch_bear_color = 'darkred'
    bos_bull_color = 'green'
    bos_bear_color = 'red'

    plot_trend_line(ax_ms, trend_tf1_display, choch_tf1_display, bos_tf1_display, 3,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf2_display, choch_tf2_display, bos_tf2_display, 2,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf3_display, choch_tf3_display, bos_tf3_display, 1,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)
    plot_trend_line(ax_ms, trend_tf4_display, choch_tf4_display, bos_tf4_display, 0,
                    choch_bull_color, choch_bear_color, bos_bull_color, bos_bear_color)

    # Add timeframe labels
    ax_ms.text(-5, 3, '15min', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 2, '1H', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 1, '4H', ha='right', va='center', fontsize=9, fontweight='bold')
    ax_ms.text(-5, 0, 'Daily', ha='right', va='center', fontsize=9, fontweight='bold')

    # ========================================================================
    # FORMATTING
    # ========================================================================
    # X-axis labels (using display data)
    step = max(1, len(df_display) // 10)
    tick_positions = list(range(0, len(df_display), step))
    tick_labels = [df_display.iloc[i].name.strftime('%Y-%m-%d %H:%M') for i in tick_positions]
    ax_main.set_xticks(tick_positions)
    ax_main.set_xticklabels([])  # Hide x labels on main chart

    ax_ms.set_xticks(tick_positions)
    ax_ms.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Main chart formatting
    ax_main.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax_main.set_title('XAUUSD - Market Structure + HTF + OB + FVG + Liquidity + Sessions + ICT + CRT + MTF Trends',
                     fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='--')

    # Combine legend handles from sweeps, order blocks, FVGs, liquidity, session levels, ICT, and CRT
    handles, labels = ax_main.get_legend_handles_labels()
    handles.extend(ob_legend_elements)
    handles.extend(fvg_legend_elements)
    handles.extend(liquidity_legend_elements)
    handles.extend(session_legend_elements)
    handles.extend(ict_legend_elements)
    handles.extend(crt_legend_elements)
    ax_main.legend(handles=handles, loc='upper left', fontsize=5.5, ncol=3)

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
    ax_ms.set_xlim(-10, len(df_display))

    # Market structure legend
    legend_elements = [
        Line2D([0], [0], color=choch_bull_color, linewidth=8, label='CHoCH Bullish'),
        Line2D([0], [0], color=choch_bear_color, linewidth=8, label='CHoCH Bearish'),
        Line2D([0], [0], color=bos_bull_color, linewidth=8, label='BoS Bullish'),
        Line2D([0], [0], color=bos_bear_color, linewidth=8, label='BoS Bearish'),
    ]
    ax_ms.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=4)

    # ========================================================================
    # PLOT MTF TREND PANEL
    # ========================================================================
    # Title
    ax_mtf.text(0.5, 0.95, 'MTF TRENDS', ha='center', va='top',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#00bcd4', alpha=0.9))

    # Plot each timeframe
    y_positions = np.linspace(0.78, 0.15, len(mtf_trends))

    for idx, (tf_name, is_bullish) in enumerate(mtf_trends.items()):
        y_pos = y_positions[idx]

        # Timeframe label with white background box for visibility
        ax_mtf.text(0.18, y_pos, tf_name, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

        # Trend status with color
        trend_text = "BULL" if is_bullish else "BEAR"
        trend_color = 'lime' if is_bullish else 'red'
        ax_mtf.text(0.58, y_pos, trend_text, ha='center', va='center',
                   fontsize=8, fontweight='bold', color=trend_color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Status indicator (circle)
        status_symbol = "●"
        ax_mtf.text(0.88, y_pos, status_symbol, ha='center', va='center',
                   fontsize=12, color=trend_color, weight='bold')

    plt.tight_layout()

    # Save
    output_file = 'combined_market_structure_htf_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCombined chart saved as '{output_file}'")

    plt.show()


if __name__ == '__main__':
    csv_file = 'PEPPERSTONE_XAUUSD, 5.csv'
    plot_combined_chart(csv_file, num_candles=500, pivot_strength=15)
