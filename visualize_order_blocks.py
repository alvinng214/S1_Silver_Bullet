"""
Visualize all 1H bullish and bearish order blocks on a candlestick chart.

Uses the PEPPERSTONE_XAUUSD 5-minute CSV data, resamples to 1H,
applies the MTF Order Block Finder logic at each bar to detect all
order blocks, then renders them on an interactive Plotly candlestick chart.
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Import the OB finder from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

ob_module = import_module("MTF Order Block Finder")
OBZone = ob_module.OBZone
OBSettings = ob_module.OBSettings


def find_all_order_blocks(df_1h: pd.DataFrame, settings: OBSettings) -> list:
    """
    Slide through the 1H OHLC data and detect order blocks at every bar
    using the same logic as the Pine Script MTF Order Block Finder.

    The original compute function only looks at the tail of the data.
    Here we iterate over every possible window to collect all OBs.
    """
    ob_period = settings.ob_period + 1  # matches the +1 in compute
    ob_shift = settings.ob_shift
    ob_search = settings.ob_search

    if ob_search >= ob_period:
        ob_search = ob_period
    if ob_shift >= ob_period:
        ob_shift = ob_period

    o = df_1h["open"].values
    h = df_1h["high"].values
    l = df_1h["low"].values
    c = df_1h["close"].values
    times = df_1h.index

    all_zones = []
    seen = set()

    for end in range(ob_period, len(df_1h)):
        # Build arrays in Pine style (index 0 = most recent)
        co = list(reversed(o[end - ob_period: end + 1]))
        ch = list(reversed(h[end - ob_period: end + 1]))
        cl = list(reversed(l[end - ob_period: end + 1]))
        cc = list(reversed(c[end - ob_period: end + 1]))
        ct = list(reversed(times[end - ob_period: end + 1]))

        # Relative move check
        if cc[ob_period] == 0:
            continue
        relmove = abs(cc[ob_period] - cc[1]) / cc[ob_period] * 100 > settings.threshold
        if co[ob_period] == 0:
            continue
        doji_candle = abs(cc[ob_period] - co[ob_period]) / co[ob_period] * 100 > settings.doji

        bullish_ob = cc[ob_period] < co[ob_period]
        bearish_ob = cc[ob_period] > co[ob_period]

        upcandles = 0
        downcandles = 0
        for i in range(1, ob_period):
            base = co[i]
            if base == cc[i] and base == ch[i] and base == cl[i]:
                continue
            if base == 0:
                continue
            if abs(100 * (cc[i] - co[i]) / co[i]) < settings.fuzzy:
                upcandles += 1
                downcandles += 1
                continue
            if cc[i] > co[i]:
                upcandles += 1
            elif cc[i] < co[i]:
                downcandles += 1

        if not (doji_candle and relmove):
            continue

        # Bullish OB
        if bullish_ob and (upcandles == (ob_period - 1)):
            selector_shift = ob_shift
            if settings.ob_selector == "OHLC":
                idx = ob_period - selector_shift
                bar_dir = np.sign(cc[idx] - co[idx])
                ob_high = cc[idx] if bar_dir == 1 else co[idx]
                ob_low = cl[idx]
            elif settings.ob_selector == "High/Low":
                idx = ob_period - selector_shift
                ob_high = ch[idx]
                ob_low = cl[idx]
            else:  # Context
                ob_high, ob_low, selector_shift = _low_wick_search_local(
                    ob_period, ob_search, co, cl, cc
                )
                idx = ob_period - selector_shift

            source_time = ct[ob_period - selector_shift]
            key = ("bull", str(source_time), ob_high, ob_low)
            if key not in seen:
                seen.add(key)
                all_zones.append(OBZone(
                    direction="bull",
                    source_time=pd.to_datetime(source_time),
                    high=float(ob_high),
                    low=float(ob_low),
                    avg=float((ob_high + ob_low) / 2),
                    selector_shift=selector_shift,
                ))

        # Bearish OB
        if bearish_ob and (downcandles == (ob_period - 1)):
            selector_shift = ob_shift
            if settings.ob_selector == "OHLC":
                idx = ob_period - selector_shift
                bar_dir = np.sign(cc[idx] - co[idx])
                ob_low = co[idx] if bar_dir == 1 else cc[idx]
                ob_high = ch[idx]
            elif settings.ob_selector == "High/Low":
                idx = ob_period - selector_shift
                ob_high = ch[idx]
                ob_low = cl[idx]
            else:  # Context
                ob_high, ob_low, selector_shift = _high_wick_search_local(
                    ob_period, ob_search, co, ch, cc
                )
                idx = ob_period - selector_shift

            source_time = ct[ob_period - selector_shift]
            key = ("bear", str(source_time), ob_high, ob_low)
            if key not in seen:
                seen.add(key)
                all_zones.append(OBZone(
                    direction="bear",
                    source_time=pd.to_datetime(source_time),
                    high=float(ob_high),
                    low=float(ob_low),
                    avg=float((ob_high + ob_low) / 2),
                    selector_shift=selector_shift,
                ))

    return all_zones


def _low_wick_search_local(start_index, length, o, l, c):
    wick_h = np.nan
    wick_l = np.nan
    index = 0
    for i in range(length + 1):
        if i > 0 and l[start_index - i] > wick_l:
            continue
        bar_dir = np.sign(c[start_index - i] - o[start_index - i])
        wick_h = c[start_index - i] if bar_dir == 1 else o[start_index - i]
        wick_l = l[start_index - i]
        index = i
    return wick_h, wick_l, index


def _high_wick_search_local(start_index, length, o, h, c):
    wick_h = np.nan
    wick_l = np.nan
    index = 0
    for i in range(length + 1):
        if i > 0 and h[start_index - i] < wick_h:
            continue
        bar_dir = np.sign(c[start_index - i] - o[start_index - i])
        wick_l = o[start_index - i] if bar_dir == 1 else c[start_index - i]
        wick_h = h[start_index - i]
        index = i
    return wick_h, wick_l, index


def main():
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "PEPPERSTONE_XAUUSD, 5.csv",
    )
    print(f"Loading 5-min data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
    df = df[["open", "high", "low", "close"]]
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    # Resample to 1H
    df_1h = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    print(f"1H candles: {len(df_1h)} (from {df_1h.index[0]} to {df_1h.index[-1]})")

    # Find all order blocks on 1H data using default settings (resolution=""
    # means use the chart timeframe directly, i.e. 1H)
    settings = OBSettings(
        resolution="",
        ob_period=5,
        threshold=0.3,
        doji=0.05,
        fuzzy=0.01,
        ob_shift=1,
        ob_selector="OHLC",
        ob_search=2,
        bull_channels=9999,
        bear_channels=9999,
    )

    zones = find_all_order_blocks(df_1h, settings)
    bull_zones = [z for z in zones if z.direction == "bull"]
    bear_zones = [z for z in zones if z.direction == "bear"]
    print(f"Found {len(bull_zones)} bullish OBs and {len(bear_zones)} bearish OBs")

    # --- Build the Plotly chart ---
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_1h.index,
        open=df_1h["open"],
        high=df_1h["high"],
        low=df_1h["low"],
        close=df_1h["close"],
        name="XAUUSD 1H",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Determine how far to extend each OB box to the right
    last_time = df_1h.index[-1]

    # Draw bullish OBs as green semi-transparent rectangles
    for z in bull_zones:
        fig.add_shape(
            type="rect",
            x0=z.source_time,
            x1=last_time,
            y0=z.low,
            y1=z.high,
            fillcolor="rgba(38, 166, 154, 0.15)",
            line=dict(color="rgba(38, 166, 154, 0.6)", width=1),
            layer="below",
        )

    # Draw bearish OBs as red semi-transparent rectangles
    for z in bear_zones:
        fig.add_shape(
            type="rect",
            x0=z.source_time,
            x1=last_time,
            y0=z.low,
            y1=z.high,
            fillcolor="rgba(239, 83, 80, 0.15)",
            line=dict(color="rgba(239, 83, 80, 0.6)", width=1),
            layer="below",
        )

    # Add invisible scatter traces for the legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="rgba(38, 166, 154, 0.4)", symbol="square"),
        name=f"Bullish OB ({len(bull_zones)})",
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="rgba(239, 83, 80, 0.4)", symbol="square"),
        name=f"Bearish OB ({len(bear_zones)})",
    ))

    fig.update_layout(
        title="XAUUSD 1H — Order Blocks (MTF Order Block Finder)",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800,
        width=1600,
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
        ),
    )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "order_blocks_1h_chart.html",
    )
    fig.write_html(out_path)
    print(f"Interactive chart saved to: {out_path}")

    # Also save a static PNG
    png_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "order_blocks_1h_chart.png",
    )
    try:
        fig.write_image(png_path, width=1600, height=800, scale=2)
        print(f"Static PNG saved to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG (kaleido may be missing): {e}")


if __name__ == "__main__":
    main()
