"""
Plot XAUUSD candlestick chart with MTF FVG boxes.

Uses the MTF FVG x2 [MK] module's data structures and resampling helpers
to detect and display:
- 1H Bullish FVG (green boxes)
- 4H Bullish FVG (blue boxes)
- 1H Bearish FVG (pink boxes)
- 4H Bearish FVG (red boxes)

FVG boxes start at their creation candle and end when price mitigates them
(or extend to the right edge if still active).
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# Import the MTF FVG module (has spaces in filename)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mtf_fvg", os.path.join(os.path.dirname(__file__), "MTF FVG x2 [MK].py")
)
mtf_fvg = importlib.util.module_from_spec(spec)
sys.modules["mtf_fvg"] = mtf_fvg
spec.loader.exec_module(mtf_fvg)

# ── Load CSV ─────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "PEPPERSTONE_XAUUSD, 5.csv")
df_raw = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
df_raw.index = df_raw.index.tz_localize(None) if df_raw.index.tz is None else df_raw.index.tz_convert(None)

# Keep only OHLC columns
df = df_raw[["open", "high", "low", "close"]].copy()

# ── Detect FVGs on resampled HTF data ────────────────────────────────────
# A classic FVG uses 3 consecutive HTF candles:
#   Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
#   Bearish FVG: candle[i-2].low  > candle[i].high (gap down)

def detect_fvgs_with_mitigation(df_5min, rule, tf_label):
    """Detect FVGs and compute their mitigation end time using 1H bars."""
    htf = df_5min.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()

    # Also resample to 1H for mitigation tracking (chart resolution)
    htf_1h = df_5min.resample("1h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()

    fvgs = []  # list of dicts with box info + start_time + end_time

    for i in range(2, len(htf)):
        h_prev2 = htf["high"].iloc[i - 2]
        l_curr = htf["low"].iloc[i]
        l_prev2 = htf["low"].iloc[i - 2]
        h_curr = htf["high"].iloc[i]
        ts = htf.index[i]

        direction = None
        top = bottom = None

        # Bullish FVG
        if l_curr > h_prev2:
            direction = "bull"
            top = l_curr
            bottom = h_prev2

        # Bearish FVG
        if h_curr < l_prev2:
            direction = "bear"
            top = l_prev2
            bottom = h_curr

        if direction is None:
            continue

        # Find mitigation time: when does price fill through the zone?
        subsequent = htf_1h.loc[htf_1h.index > ts]
        end_time = None
        for j in range(len(subsequent)):
            if direction == "bull":
                # Mitigated when price low penetrates below the bottom of zone
                if subsequent["low"].iloc[j] <= bottom:
                    end_time = subsequent.index[j]
                    break
            else:
                # Mitigated when price high penetrates above the top of zone
                if subsequent["high"].iloc[j] >= top:
                    end_time = subsequent.index[j]
                    break

        fvgs.append({
            "direction": direction,
            "tf": tf_label,
            "top": top,
            "bottom": bottom,
            "start_time": ts,
            "end_time": end_time,  # None means still active
            "mitigated": end_time is not None,
        })

    return fvgs, htf


fvgs_1h, htf_1h = detect_fvgs_with_mitigation(df, "1h", "1H")
fvgs_4h, htf_4h = detect_fvgs_with_mitigation(df, "4h", "4H")

bull_1h = [f for f in fvgs_1h if f["direction"] == "bull"]
bear_1h = [f for f in fvgs_1h if f["direction"] == "bear"]
bull_4h = [f for f in fvgs_4h if f["direction"] == "bull"]
bear_4h = [f for f in fvgs_4h if f["direction"] == "bear"]

print(f"1H Bullish FVGs: {len(bull_1h)} ({sum(1 for f in bull_1h if not f['mitigated'])} active)")
print(f"1H Bearish FVGs: {len(bear_1h)} ({sum(1 for f in bear_1h if not f['mitigated'])} active)")
print(f"4H Bullish FVGs: {len(bull_4h)} ({sum(1 for f in bull_4h if not f['mitigated'])} active)")
print(f"4H Bearish FVGs: {len(bear_4h)} ({sum(1 for f in bear_4h if not f['mitigated'])} active)")
print(f"Total FVGs: {len(fvgs_1h) + len(fvgs_4h)}")

# ── Candlestick plotting on 1H timeframe ─────────────────────────────────
df_plot = htf_1h.copy()

fig, ax = plt.subplots(figsize=(36, 14))

# Plot candlesticks
dates = mdates.date2num(df_plot.index.to_pydatetime())
width = 0.028

for i in range(len(df_plot)):
    o = df_plot["open"].iloc[i]
    c = df_plot["close"].iloc[i]
    h = df_plot["high"].iloc[i]
    l = df_plot["low"].iloc[i]
    d = dates[i]

    color = "#26a69a" if c >= o else "#ef5350"
    body_bottom = min(o, c)
    body_height = abs(c - o)
    if body_height < 0.01:
        body_height = 0.01
    ax.bar(d, body_height, bottom=body_bottom, width=width, color=color,
           edgecolor=color, linewidth=0.5, zorder=5)
    ax.plot([d, d], [l, body_bottom], color=color, linewidth=0.7, zorder=5)
    ax.plot([d, d], [min(o, c) + body_height, h], color=color, linewidth=0.7, zorder=5)

# ── Draw FVG boxes ───────────────────────────────────────────────────────
x_right_edge = dates[-1] + 1.0  # extend active boxes 1 day past last bar

def draw_fvg_boxes(fvg_list, facecolor, edgecolor, alpha, label_text, zorder=2):
    """Draw FVG boxes from creation time to mitigation (or chart end)."""
    drawn_label = False
    for fvg in fvg_list:
        top = max(fvg["top"], fvg["bottom"])
        bottom = min(fvg["top"], fvg["bottom"])
        height = top - bottom
        if height < 0.01:
            continue

        x_start = mdates.date2num(fvg["start_time"].to_pydatetime())
        if fvg["end_time"] is not None:
            x_end = mdates.date2num(fvg["end_time"].to_pydatetime())
        else:
            x_end = x_right_edge

        lbl = label_text if not drawn_label else None
        a = alpha if not fvg["mitigated"] else alpha * 0.5  # dimmer for mitigated
        rect = mpatches.Rectangle(
            (x_start, bottom), x_end - x_start, height,
            linewidth=0.6, edgecolor=edgecolor, facecolor=facecolor,
            alpha=a, zorder=zorder, label=lbl
        )
        ax.add_patch(rect)
        drawn_label = True

# Colors:  1H Bull=green, 4H Bull=blue, 1H Bear=pink, 4H Bear=red
draw_fvg_boxes(bull_1h, "#00c853", "#00e676", 0.22, "1H Bullish FVG (Green)", zorder=2)
draw_fvg_boxes(bull_4h, "#2979ff", "#448aff", 0.28, "4H Bullish FVG (Blue)", zorder=3)
draw_fvg_boxes(bear_1h, "#ff80ab", "#ff4081", 0.22, "1H Bearish FVG (Pink)", zorder=2)
draw_fvg_boxes(bear_4h, "#d50000", "#ff1744", 0.28, "4H Bearish FVG (Red)", zorder=3)

# ── Formatting ───────────────────────────────────────────────────────────
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
fig.autofmt_xdate()

ax.set_title("XAUUSD — MTF Fair Value Gaps: 1H (Green/Pink) & 4H (Blue/Red)",
             fontsize=16, fontweight="bold")
ax.set_ylabel("Price (USD)", fontsize=12)
ax.set_xlabel("Date/Time", fontsize=12)

legend = ax.legend(loc="upper left", fontsize=11, framealpha=0.85,
                   facecolor="#2a2a3e", edgecolor="#555")
for text in legend.get_texts():
    text.set_color("white")

ax.grid(True, alpha=0.15, linestyle="--", color="#555")
ax.set_facecolor("#1e1e2f")
fig.patch.set_facecolor("#15151e")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
for spine in ax.spines.values():
    spine.set_color("#444")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "mtf_fvg_xauusd.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved to: {output_path}")
