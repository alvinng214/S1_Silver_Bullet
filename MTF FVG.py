"""Python translation of Pine Script `MTF FVG` (© pmk07).

Source: S1_Silver_Bullet/MTF FVG.txt

Three-bar Fair Value Gap detection across multiple HTFs with partial-test
("changelvl/changecolor") and full-mitigation tracking.

Detection (`find_box`, evaluated at every HTF bar's `barstate.isconfirmed`):
    Pine:
        x := low[2] >= high  ? -1
           : low    >= high[2] ? +1
           : 0
        bull -> top=low,    bottom=high[2]
        bear -> top=low[2], bottom=high
    The 3-bar pattern is bars [N-2, N-1, N]; bar N-1 contains the gap.

Publishing (`timeframe.change(tf)`):
    The detection for HTF bar N is exposed to the LTF on the FIRST LTF bar
    of HTF bar N+1 (Pine `request.security(..., lookahead=off)` semantics).
    box.left = `time - t*60000*2` -> the open time of HTF bar N-2.

Mitigation (`control_box`, runs on EVERY LTF bar):
    Bull box deleted (mitigated) when LTF.low  < box.bottom
    Bear box deleted (mitigated) when LTF.high > box.top
    Otherwise, partial test:
        bull: when LTF.low  < box.top    -> box.top    = LTF.low  (changelvl)
        bear: when LTF.high > box.bottom -> box.bottom = LTF.high (changelvl)
    `changecolor` flips the box to the "tested" colour after a partial test.
    `extend_r` keeps the box's right edge tracking the latest LTF bar.

Pine input parity:
    changelvl    (bool, default true)
    changecolor  (bool, default true)
    extend_r     (bool, default true)
    plotLabel    (bool, default false)
    BullColor / BearColor / BullColorTested / BearColorTested -> kept as
        descriptive constants; this module returns structured data, not
        TradingView drawing primitives.
    Timeframe toggles (default values match Pine):
        1=F, 3=F, 5=F, 15=T, 30=F, 45=F, 60=T, 120=F, 180=F, 240=T, D=T, W=T

Optional extension: a `enable_monthly` toggle (defaults OFF) is provided so
the equity-research pipeline can keep its prior monthly FVG output. Monthly
is NOT in the original Pine source; toggle it explicitly when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Settings + result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FVGSettings:
    # General (Pine `group='General'`)
    changelvl: bool = True
    changecolor: bool = True
    extend_r: bool = True

    # Style (cosmetic in TV; preserved for input parity)
    plot_label: bool = False
    bull_color: str = "rgba(0,255,0,0.10)"
    bear_color: str = "rgba(255,0,0,0.10)"
    bull_color_tested: str = "rgba(128,128,128,0.10)"
    bear_color_tested: str = "rgba(128,128,128,0.10)"

    # Timeframe toggles (Pine defaults)
    enable_1m: bool = False
    enable_3m: bool = False
    enable_5m: bool = False
    enable_15m: bool = True
    enable_30m: bool = False
    enable_45m: bool = False
    enable_60m: bool = True
    enable_120m: bool = False
    enable_180m: bool = False
    enable_240m: bool = True
    enable_daily: bool = True
    enable_weekly: bool = True
    # Extension toggle (NOT in original Pine source).
    enable_monthly: bool = False


@dataclass
class FVGBox:
    direction: str             # "bull" or "bear"
    timeframe: str             # "1","3","5","15","30","45","60","120","180","240","D","W","M"
    top: float
    bottom: float
    created_time: pd.Timestamp        # Pine `_time` — open of HTF bar N-2
    publish_time: pd.Timestamp        # first LTF bar of HTF bar N+1 (Pine `time` at create_box)
    right_time: pd.Timestamp          # latest LTF bar processed (extend.right)
    tested: bool = False              # flipped True on first partial test
    mitigated: bool = False           # True after Pine `box.delete`
    mitigated_time: Optional[pd.Timestamp] = None


# Pine TF code -> (settings attr, pandas resample rule, Pine `t` minutes)
_TF_TABLE: List[Tuple[str, str, str, int]] = [
    ("1",   "enable_1m",     "1min",   1),
    ("3",   "enable_3m",     "3min",   3),
    ("5",   "enable_5m",     "5min",   5),
    ("15",  "enable_15m",    "15min",  15),
    ("30",  "enable_30m",    "30min",  30),
    ("45",  "enable_45m",    "45min",  45),
    ("60",  "enable_60m",    "60min",  60),
    ("120", "enable_120m",   "120min", 120),
    ("180", "enable_180m",   "180min", 180),
    ("240", "enable_240m",   "240min", 240),
    ("D",   "enable_daily",  "1D",     1440),
    ("W",   "enable_weekly", "W-MON",  10080),
    ("M",   "enable_monthly","MS",     43200),  # extension; not in Pine source
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.lower() for c in df.columns if isinstance(c, str)}
    df = df.rename(columns=cols)
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    # `1D` and `MS` are inherently left-labelled in pandas; the others need
    # explicit left/closed-left so the bar timestamp = HTF bar's open time.
    if rule in {"1D", "MS"}:
        return df.resample(rule).agg(agg).dropna()
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


def _enabled_tfs(settings: FVGSettings) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for code, attr, rule, minutes in _TF_TABLE:
        if getattr(settings, attr, False):
            out.append((code, rule, minutes))
    return out


def _detect_fvg(htf_df: pd.DataFrame, htf_idx: int) -> Optional[Tuple[str, float, float]]:
    """Pine `find_box` at HTF bar N (returns None when x == 0).

    Returns ``(direction, top, bottom)``.
    """
    if htf_idx < 2:
        return None
    low_n = float(htf_df["low"].iloc[htf_idx])
    high_n = float(htf_df["high"].iloc[htf_idx])
    low_n2 = float(htf_df["low"].iloc[htf_idx - 2])
    high_n2 = float(htf_df["high"].iloc[htf_idx - 2])
    # Pine: x := low[2] >= high ? -1 : low >= high[2] ? 1 : 0
    if low_n2 >= high_n:
        return ("bear", low_n2, high_n)
    if low_n >= high_n2:
        return ("bull", low_n, high_n2)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_mtf_fvg(
    df: pd.DataFrame,
    settings: Optional[FVGSettings] = None,
) -> Dict[str, object]:
    """Run MTF FVG bar-by-bar over the LTF dataframe.

    Returns ``{"boxes": [...], "active_boxes": [...], "mitigated_boxes": [...]}``
    where each list element is an :class:`FVGBox`.
    """
    if settings is None:
        settings = FVGSettings()
    df = _prepare(df)

    enabled = _enabled_tfs(settings)
    if not enabled:
        return {"boxes": [], "active_boxes": [], "mitigated_boxes": []}

    # Resample once per HTF; precompute LTF -> HTF index lookup.
    htf_frames: Dict[str, pd.DataFrame] = {code: _resample(df, rule) for code, rule, _ in enabled}
    ltf_index = df.index
    ltf_to_htf: Dict[str, np.ndarray] = {}
    for code, _rule, _ in enabled:
        htf_idx = htf_frames[code].index
        # Position of the HTF bar containing each LTF timestamp:
        # the largest HTF index <= LTF timestamp.
        positions = np.searchsorted(htf_idx, ltf_index.values, side="right") - 1
        ltf_to_htf[code] = positions

    # Precompute every potential FVG per HTF (detection depends only on HTF data).
    detections: Dict[str, Dict[int, Tuple[str, float, float]]] = {}
    for code, _rule, _ in enabled:
        htf_df = htf_frames[code]
        d_map: Dict[int, Tuple[str, float, float]] = {}
        for hi in range(2, len(htf_df)):
            res = _detect_fvg(htf_df, hi)
            if res is not None:
                d_map[hi] = res
        detections[code] = d_map

    boxes: List[FVGBox] = []
    last_htf_pos: Dict[str, int] = {code: -1 for code, _, _ in enabled}

    for li in range(len(ltf_index)):
        ts = ltf_index[li]
        ltf_low = float(df["low"].iloc[li])
        ltf_high = float(df["high"].iloc[li])

        # 1) `timeframe.change(tf)` — fires on the FIRST LTF bar of a new HTF.
        #    Publish the FVG (if any) for the just-confirmed HTF bar.
        for code, _rule, _ in enabled:
            cur_htf_pos = int(ltf_to_htf[code][li])
            prev_htf_pos = last_htf_pos[code]
            if cur_htf_pos != prev_htf_pos and prev_htf_pos >= 0:
                det = detections[code].get(prev_htf_pos)
                if det is not None:
                    direction, top, bottom = det
                    htf_df = htf_frames[code]
                    # Pine: _time = time - t*60000*2 -> open of HTF[N-2].
                    left_idx = max(0, prev_htf_pos - 2)
                    boxes.append(FVGBox(
                        direction=direction,
                        timeframe=code,
                        top=top,
                        bottom=bottom,
                        created_time=htf_df.index[left_idx],
                        publish_time=ts,
                        right_time=ts,
                    ))
            last_htf_pos[code] = cur_htf_pos

        # 2) `control_box` — runs on every LTF bar for every box.
        for box in boxes:
            if box.mitigated:
                continue
            if box.publish_time > ts:
                continue
            if box.direction == "bull":
                if ltf_low < box.bottom:
                    box.mitigated = True
                    box.mitigated_time = ts
                    continue
                if ltf_low < box.top:
                    if settings.changelvl:
                        box.top = ltf_low
                    if settings.changecolor:
                        box.tested = True
                if settings.extend_r:
                    box.right_time = ts
            else:  # bear
                if ltf_high > box.top:
                    box.mitigated = True
                    box.mitigated_time = ts
                    continue
                if ltf_high > box.bottom:
                    if settings.changelvl:
                        box.bottom = ltf_high
                    if settings.changecolor:
                        box.tested = True
                if settings.extend_r:
                    box.right_time = ts

    active = [b for b in boxes if not b.mitigated]
    mitigated = [b for b in boxes if b.mitigated]
    return {"boxes": boxes, "active_boxes": active, "mitigated_boxes": mitigated}


__all__ = ["FVGSettings", "FVGBox", "compute_mtf_fvg"]
