"""Generate an XAUUSD chart with 1H and 4H Order Blocks visualized.

Color scheme:
  - 1H Bullish OB: green boxes
  - 4H Bullish OB: blue boxes
  - 1H Bearish OB: pink boxes
  - 4H Bearish OB: red boxes
"""

import sys
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

# Import the order block detection logic
sys.path.insert(0, os.path.dirname(__file__))
from importlib import import_module

ob_module = import_module("Order Blocks & Imbalance MTF")
build_order_blocks = ob_module.build_order_blocks
OBSettings = ob_module.OBSettings

# ── Load data ──────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "PEPPERSTONE_XAUUSD, 5.csv")
raw = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
raw.columns = [c.strip().lower() for c in raw.columns]

# Keep only OHLC columns
df = raw[["open", "high", "low", "close"]].copy()
df = df.sort_index()

# ── Detect order blocks at 1H and 4H ──────────────────────────────────────
settings_1h = OBSettings(
    timeframe="1H",
    fvg_threshold=0.5,
    mitigation_type="Wick",
    show_bull=True,
    show_bear=True,
    use_smart_view=False,
    visible_limit=20,
    extend_active=False,
)

settings_4h = OBSettings(
    timeframe="4H",
    fvg_threshold=0.5,
    mitigation_type="Wick",
    show_bull=True,
    show_bear=True,
    use_smart_view=False,
    visible_limit=20,
    extend_active=False,
)

zones_1h = build_order_blocks(df, settings_1h)
zones_4h = build_order_blocks(df, settings_4h)

print(f"1H zones found: {len(zones_1h)}  (bull: {sum(z.is_bullish for z in zones_1h)}, bear: {sum(not z.is_bullish for z in zones_1h)})")
print(f"4H zones found: {len(zones_4h)}  (bull: {sum(z.is_bullish for z in zones_4h)}, bear: {sum(not z.is_bullish for z in zones_4h)})")

# ── Build the chart ────────────────────────────────────────────────────────
# Resample to 1H candles for a cleaner chart
df_1h = df.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
print(f"1H candles: {len(df_1h)}")

fig, ax = plt.subplots(figsize=(28, 12))

# Draw candlesticks on 1H data
width = pd.Timedelta(minutes=40)
width_num = mdates.date2num(df_1h.index[0] + width) - mdates.date2num(df_1h.index[0])
thin_width = width_num * 0.15

for ts, row in df_1h.iterrows():
    ts_num = mdates.date2num(ts)
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    color = "#26a69a" if c >= o else "#ef5350"
    # Wick
    ax.plot([ts_num, ts_num], [l, h], color=color, linewidth=0.8)
    # Body
    body_bottom = min(o, c)
    body_height = abs(c - o)
    if body_height < 0.01:
        body_height = 0.01
    rect = mpatches.FancyBboxPatch(
        (ts_num - width_num / 2, body_bottom),
        width_num, body_height,
        boxstyle="square,pad=0",
        facecolor=color, edgecolor=color, linewidth=0.5,
    )
    ax.add_patch(rect)

# ── Draw order block zones ─────────────────────────────────────────────────
def draw_zones(zones, bull_color, bear_color, bull_alpha, bear_alpha, label_prefix, zorder_base):
    for zone in zones:
        if not zone.visible:
            continue
        is_bull = zone.is_bullish
        color = bull_color if is_bull else bear_color
        alpha = bull_alpha if is_bull else bear_alpha

        left = mdates.date2num(zone.left_time)
        right = mdates.date2num(zone.right_time)
        bottom = zone.bottom
        height = zone.top - zone.bottom

        rect = mpatches.FancyBboxPatch(
            (left, bottom), right - left, height,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor=color,
            alpha=alpha, linewidth=1.2,
            zorder=zorder_base,
        )
        ax.add_patch(rect)

# 4H zones drawn first (behind), 1H zones on top
draw_zones(zones_4h, bull_color="#1565C0", bear_color="#C62828",
           bull_alpha=0.28, bear_alpha=0.28, label_prefix="4H", zorder_base=1)
draw_zones(zones_1h, bull_color="#2E7D32", bear_color="#E91E9A",
           bull_alpha=0.25, bear_alpha=0.25, label_prefix="1H", zorder_base=2)

# ── Formatting ─────────────────────────────────────────────────────────────
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=4))
fig.autofmt_xdate()

ax.set_ylabel("XAUUSD Price", fontsize=12)
ax.set_title("XAUUSD – Order Blocks (1H & 4H)", fontsize=16, fontweight="bold")
ax.grid(True, alpha=0.2)

# Legend
legend_patches = [
    mpatches.Patch(facecolor="#2E7D32", alpha=0.4, label="1H Bullish OB (Green)"),
    mpatches.Patch(facecolor="#1565C0", alpha=0.4, label="4H Bullish OB (Blue)"),
    mpatches.Patch(facecolor="#E91E9A", alpha=0.4, label="1H Bearish OB (Pink)"),
    mpatches.Patch(facecolor="#C62828", alpha=0.4, label="4H Bearish OB (Red)"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=11)

ax.autoscale_view()
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "xauusd_order_blocks.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved chart to {out_path}")
