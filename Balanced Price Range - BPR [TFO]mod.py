"""Python translation of Pine Script `BPR [TFO] - Modified` (© tradeforopp).

Source: S1_Silver_Bullet/Balanced Price Range - BPR [TFO]mod.txt

A Balanced Price Range forms when an opposing 3-bar Fair Value Gap appears
within ``bars_since`` bars of an earlier FVG, and the two FVGs OVERLAP. The
overlap region itself is the BPR zone.

Pine algorithm (bullish case):
    new_fvg_bearish = low[2] - high > 0
    new_fvg_bullish = low - high[2] > 0
    bull_num_since  = ta.barssince(new_fvg_bearish)         // N
    bull_bpr_cond_1 = new_fvg_bullish AND N <= bars_since
    bull_bpr_cond_2 = high[N] + low[N+2] + high[2] + low
                       > max(low[N+2], low) - min(high[N], high[2])
    bull_combined_low  = max(high[N], high[2])               // overlap bottom
    bull_combined_high = min(low[N+2], low)                  // overlap top
    bull_result        = cond_1 AND cond_2 AND only_clean AND
                         (combined_high - combined_low >= bpr_threshold)

Drawing:
    `if bull_result[1]` — the box is created on the bar AFTER the result
    fires, using the PRIOR bar's combined edges. Box left = bar where the
    middle (gap) bar of the bear FVG sat (index = t - N - 2).

Mitigation (`extend_right=true`):
    bull box mitigated when close < bpr_bottom (and mirror for bear)
    Box `right` extends each bar until mitigation, then freezes.

Inputs (Pine defaults):
    bpr_threshold   = 0
    bars_since      = 10
    extend_right    = true
    only_clean_bpr  = false
    delete_old_bpr  = false
    show_midline    = true
    max_bpr_count   = 50
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Settings + result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BPRSettings:
    bpr_threshold: float = 0.0
    bars_since: int = 10
    extend_right: bool = True
    only_clean_bpr: bool = False
    delete_old_bpr: bool = False
    show_midline: bool = True
    max_bpr_count: int = 50


@dataclass
class BPRBox:
    direction: str                       # "bull" or "bear"
    top: float
    bottom: float
    midline: float                        # (top + bottom) / 2
    left_time: pd.Timestamp               # bar where bear/bull FVG's middle gap bar sat
    right_time: pd.Timestamp              # extends until mitigation
    created_time: pd.Timestamp            # bar t (where Pine `if result[1]` fires)
    bull_num_since: int                   # N = barssince(opposing FVG) at create-1
    mitigated: bool = False
    mitigated_time: Optional[pd.Timestamp] = None


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


def _barssince(flags: List[bool]) -> List[int]:
    """Pine `ta.barssince`: -1 (proxy for `na`) until the first True; then
    0 on True bars, increments on False bars after.
    """
    out: List[int] = [-1] * len(flags)
    last = -1  # -1 = never seen
    for i, v in enumerate(flags):
        if v:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_bpr(df: pd.DataFrame, settings: Optional[BPRSettings] = None) -> Dict[str, object]:
    """Run BPR [TFO] - Modified across the bars and return all BPR boxes.

    Returns ``{"boxes": [...], "active_boxes": [...], "mitigated_boxes": [...]}``.
    """
    if settings is None:
        settings = BPRSettings()
    df = _prepare(df)
    n = len(df)
    if n < 3:
        return {"boxes": [], "active_boxes": [], "mitigated_boxes": []}

    H = df["high"].astype(float).tolist()
    L = df["low"].astype(float).tolist()
    C = df["close"].astype(float).tolist()
    idx = df.index

    # Helper for safe past-bar lookup; when out of range, mirror Pine na via NaN.
    def Hp(i: int) -> float:
        return H[i] if 0 <= i < n else float("nan")

    def Lp(i: int) -> float:
        return L[i] if 0 <= i < n else float("nan")

    # 1) Per-bar FVG flags (using Pine's [2] -> i-2 convention).
    new_fvg_bearish = [False] * n
    new_fvg_bullish = [False] * n
    for i in range(n):
        if i >= 2:
            new_fvg_bearish[i] = (Lp(i - 2) - Hp(i)) > 0
            new_fvg_bullish[i] = (Lp(i) - Hp(i - 2)) > 0

    # 2) Per-bar `barssince` for opposing FVGs.
    bull_num_since = _barssince(new_fvg_bearish)  # bars since last bear FVG
    bear_num_since = _barssince(new_fvg_bullish)  # bars since last bull FVG

    # 3) Per-bar candidate combined edges and result flags.
    bull_combined_low: List[Optional[float]] = [None] * n
    bull_combined_high: List[Optional[float]] = [None] * n
    bull_result = [False] * n

    bear_combined_low: List[Optional[float]] = [None] * n
    bear_combined_high: List[Optional[float]] = [None] * n
    bear_result = [False] * n

    for i in range(n):
        # ---- Bull BPR ----
        N = bull_num_since[i]
        cond_1 = new_fvg_bullish[i] and (N != -1) and (N <= settings.bars_since)
        if cond_1:
            # Pine: high[N] + low[N+2] + high[2] + low > max(low[N+2], low) - min(high[N], high[2])
            hN = Hp(i - N) if N >= 0 else float("nan")
            lN2 = Lp(i - (N + 2)) if (N + 2) >= 0 else float("nan")
            h2 = Hp(i - 2)
            l0 = Lp(i)
            cond_2 = (hN + lN2 + h2 + l0) > (max(lN2, l0) - min(hN, h2))
            if cond_2:
                comb_low = max(hN, h2)              # overlap bottom
                comb_high = min(lN2, l0)            # overlap top
                # Pine `only_clean_bpr` test
                cond_3 = True
                if settings.only_clean_bpr:
                    for h in range(2, N + 1):       # `for h = 2 to N`
                        if Hp(i - h) > comb_low:
                            cond_3 = False
                            break
                if cond_3 and (comb_high - comb_low >= settings.bpr_threshold):
                    bull_combined_low[i] = comb_low
                    bull_combined_high[i] = comb_high
                    bull_result[i] = True

        # ---- Bear BPR ----
        M = bear_num_since[i]
        bcond_1 = new_fvg_bearish[i] and (M != -1) and (M <= settings.bars_since)
        if bcond_1:
            hM = Hp(i - M) if M >= 0 else float("nan")
            lM2 = Lp(i - (M + 2)) if (M + 2) >= 0 else float("nan")
            h2 = Hp(i - 2)
            l0 = Lp(i)
            bcond_2 = (hM + lM2 + h2 + l0) > (max(lM2, l0) - min(hM, h2))
            if bcond_2:
                # Pine: bear_combined_low  = max(high[M+2], high)
                #       bear_combined_high = min(low[M],   low[2])
                hM2 = Hp(i - (M + 2))
                h0 = Hp(i)
                lM = Lp(i - M) if M >= 0 else float("nan")
                l2 = Lp(i - 2)
                comb_low = max(hM2, h0)
                comb_high = min(lM, l2)
                bcond_3 = True
                if settings.only_clean_bpr:
                    for h in range(2, M + 1):
                        if Lp(i - h) < comb_high:
                            bcond_3 = False
                            break
                if bcond_3 and (comb_high - comb_low >= settings.bpr_threshold):
                    bear_combined_low[i] = comb_low
                    bear_combined_high[i] = comb_high
                    bear_result[i] = True

    # 4) Walk bars; create boxes on bar t when result[t-1] was True.
    bull_boxes: List[BPRBox] = []
    bear_boxes: List[BPRBox] = []
    all_boxes: List[BPRBox] = []

    for i in range(n):
        ts = idx[i]
        # Box creation uses bar t-1's stored values.
        if i >= 1 and bull_result[i - 1]:
            N1 = bull_num_since[i - 1]
            ch = bull_combined_high[i - 1]
            cl = bull_combined_low[i - 1]
            assert ch is not None and cl is not None
            # Pine: left = bar_index - bull_num_since[1] - 2 = i - N1 - 2 (here `i` plays bar_index)
            left_idx = max(0, i - N1 - 2)
            box = BPRBox(
                direction="bull",
                top=ch, bottom=cl,
                midline=(ch + cl) / 2,
                left_time=idx[left_idx],
                right_time=ts,                  # initial; mitigation step extends/freezes
                created_time=ts,
                bull_num_since=N1,
            )
            bull_boxes.append(box)
            all_boxes.append(box)
            # Cap size (Pine `array.unshift` + `array.pop` if oversized — i.e. drop oldest).
            if len(bull_boxes) > settings.max_bpr_count:
                bull_boxes.pop(0)

        if i >= 1 and bear_result[i - 1]:
            M1 = bear_num_since[i - 1]
            ch = bear_combined_high[i - 1]
            cl = bear_combined_low[i - 1]
            assert ch is not None and cl is not None
            left_idx = max(0, i - M1 - 2)
            box = BPRBox(
                direction="bear",
                top=ch, bottom=cl,
                midline=(ch + cl) / 2,
                left_time=idx[left_idx],
                right_time=ts,
                created_time=ts,
                bull_num_since=M1,
            )
            bear_boxes.append(box)
            all_boxes.append(box)
            if len(bear_boxes) > settings.max_bpr_count:
                bear_boxes.pop(0)

        # Mitigation: extend or freeze on every bar.
        if settings.extend_right:
            close_i = C[i]
            for box in bull_boxes:
                if box.mitigated:
                    continue
                if box.created_time > ts:           # not created yet
                    continue
                if close_i < box.bottom:
                    box.mitigated = True
                    box.mitigated_time = ts
                    box.right_time = ts
                else:
                    box.right_time = ts
            for box in bear_boxes:
                if box.mitigated:
                    continue
                if box.created_time > ts:
                    continue
                if close_i > box.top:
                    box.mitigated = True
                    box.mitigated_time = ts
                    box.right_time = ts
                else:
                    box.right_time = ts

    active = [b for b in all_boxes if not b.mitigated]
    mitigated = [b for b in all_boxes if b.mitigated]
    return {"boxes": all_boxes, "active_boxes": active, "mitigated_boxes": mitigated}


__all__ = ["BPRSettings", "BPRBox", "compute_bpr"]
