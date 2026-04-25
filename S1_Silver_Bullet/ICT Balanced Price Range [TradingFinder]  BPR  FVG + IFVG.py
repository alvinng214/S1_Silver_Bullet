"""ICT Balanced Price Range [TradingFinder]  BPR  FVG + IFVG — Python translation.

Faithful translation of TradingFinder's Pine Script v5 indicator, which composes
three of their published public libraries:

  1. ``FVGDetectorLibrary`` v3               — FVG detection with ATR-based 4-level filter
  2. ``OrderBlockDrawing_TradingFinder`` v4  — FVG mitigation + Breaker Block
                                                (= IFVG: when price closes through
                                                the far edge of a mitigated FVG, it
                                                flips bias and becomes an IFVG)
  3. ``OrderBlockOverlappingDrawing`` v1     — BPR = intersection of a newly-formed
                                                FVG and the live opposite-bias IFVG

In the TradingFinder wiring, a bullish BPR is the overlap between a NEW bullish
(demand) FVG and the live DEMAND IFVG (= an earlier supply FVG that was mitigated
and flipped). This is the canonical ICT BPR: both sides of the market have left
order flow in the same price zone.

Public entry point:

    run_bpr_indicator(df, fvg_filter='On', fvg_filter_type='Defensive',
                      fvg_validity_period=500,
                      mitigation_level_bpr='Proximal',
                      mitigation_level_ifvg='Proximal') -> BPRIndicatorOutput

``df`` must have columns ['open', 'high', 'low', 'close'] on a DatetimeIndex.

The output preserves all per-bar state series (boolean triggers, zone levels,
mitigation edges) so a forensic audit against the Pine source is straightforward.

Pine → Python conventions used in this translation:

  - Pine ``var`` (persist across bars) → Python local in the bar loop, hoisted
    above the loop.
  - Pine ``[1]`` (prior bar value) → a ``*_prev`` snapshot taken at the start of
    each bar, before any updates.
  - Pine ``ta.atr(55)`` = Wilder RMA of True Range, seeded with SMA of the first
    55 TRs (see ``_wilder_rma``). NOT ``rolling(55).mean()``.
  - Pine evaluation order within one bar matters — the two ``OBDrawing`` calls
    write the IFVG levels that the two ``OBOverlappingDrawing`` calls read on the
    same bar.
  - Pine library typo: the Demand branch of ``OBOverlappingDrawing`` references
    ``TriggerConditionOin`` (vs ``TriggerConditionOrigin`` in the Supply branch).
    Pine v5 would not compile with an undeclared identifier, so we treat this as
    a copy-paste artifact and use ``TriggerConditionOrigin`` in both branches.

For conventions on proximal/distal: for a Demand zone, price approaches from
above, so Proximal = top edge (= current bar's ``low`` on FVG formation), Distal
= bottom edge (= ``high[2]``). For a Supply zone, mirror: Proximal = bottom,
Distal = top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pine-equivalent math primitives
# ---------------------------------------------------------------------------
def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Equivalent of Pine ``ta.tr(true)``: max(h-l, |h-c[1]|, |l-c[1]|).

    On bar 0 (no prior close) Pine's ``ta.tr(true)`` returns ``high - low``;
    we match that via ``fillna`` on the shifted-close components.
    """
    prev_close = close.shift(1)
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _wilder_rma(series: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.rma(x, length)``: Wilder smoothing seeded with SMA.

    Pine's ta.rma:
      - first valid at index (length - 1), seeded as SMA of the first ``length`` values;
      - subsequent: rma[i] = rma[i-1] * (length-1)/length + x[i]/length.
    """
    vals = series.to_numpy(dtype=float)
    out = np.full_like(vals, np.nan, dtype=float)
    n = len(vals)
    if n < length:
        return pd.Series(out, index=series.index)
    seed = np.nanmean(vals[:length])
    if np.isnan(seed):
        return pd.Series(out, index=series.index)
    out[length - 1] = seed
    alpha = 1.0 / length
    for i in range(length, n):
        if np.isnan(vals[i]):
            out[i] = out[i - 1]
        else:
            out[i] = out[i - 1] * (1.0 - alpha) + vals[i] * alpha
    return pd.Series(out, index=series.index)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 55) -> pd.Series:
    return _wilder_rma(_true_range(high, low, close), length)


# ---------------------------------------------------------------------------
# Result dataclasses — one per library, plus a top-level bundle
# ---------------------------------------------------------------------------
@dataclass
class FVGSeries:
    """FVGDetectorLibrary.FVGDetector output, forward-filled across bars.

    ``d_condition`` / ``s_condition`` are per-bar *edge* triggers (true only
    on the bar a new FVG fires). The level series (``d_distal`` / ``d_proximal``
    / ``d_bar_idx`` and supply counterparts) are forward-filled because Pine's
    ``var`` keeps the most-recent values after a trigger.
    """
    d_condition: pd.Series
    d_distal: pd.Series
    d_proximal: pd.Series
    d_bar_idx: pd.Series
    s_condition: pd.Series
    s_distal: pd.Series
    s_proximal: pd.Series
    s_bar_idx: pd.Series


@dataclass
class OBDrawingSeries:
    """Output of a single ``OBDrawing`` call (either Demand or Supply side).

    ``fvg_live`` and ``fvg_mitigated_alert`` describe the primary zone (the FVG
    being tracked). ``ifvg_*`` describes the Breaker Block (= IFVG) that spawns
    when the primary zone is broken through its distal.

    For ``OBDrawing('Demand')``: primary = demand FVG, ifvg = supply IFVG.
    For ``OBDrawing('Supply')``: primary = supply FVG, ifvg = demand IFVG.
    """
    fvg_live: pd.Series
    fvg_mitigated_alert: pd.Series
    ifvg_trigger: pd.Series              # Check_BB false → true edge
    ifvg_live: pd.Series
    ifvg_mitigated_alert: pd.Series
    ifvg_proximal: pd.Series              # ProximalPrice_BB (forward-filled)
    ifvg_distal: pd.Series                # DistalPrice_BB
    ifvg_bar_idx: pd.Series               # Index_BB


@dataclass
class BPRSeries:
    """Output of a single ``OBOverlappingDrawing`` call (one BPR direction)."""
    trigger: pd.Series                    # single-bar edge: BPR zone just formed
    live: pd.Series                       # BPR zone still live (Check)
    mitigated_alert: pd.Series
    proximal: pd.Series
    distal: pd.Series
    bar_idx: pd.Series                    # Index_Curr captured at trigger


@dataclass
class BPRIndicatorOutput:
    fvg: FVGSeries
    obd_demand: OBDrawingSeries            # tracks demand FVG + supply IFVG
    obd_supply: OBDrawingSeries            # tracks supply FVG + demand IFVG
    bpr_bullish: BPRSeries                 # from OBOverlappingDrawing('Demand', ...)
    bpr_bearish: BPRSeries                 # from OBOverlappingDrawing('Supply', ...)
    atr: pd.Series


# ---------------------------------------------------------------------------
# Filter-level helpers — translate the ATR / body-ratio ladder literally
# ---------------------------------------------------------------------------
_VALID_FILTERS = {"Very Aggressive", "Aggressive", "Defensive", "Very Defensive"}
_VALID_MITIGATION = {"Proximal", "Distal", "50 % OB"}


def _safe_body_ratio(o: float, c: float, h: float, l: float) -> float:
    rng = h - l
    if rng == 0 or np.isnan(rng):
        return 0.0
    return abs((c - o) / rng)


def _demand_filter(
    filter_on: str,
    filter_type: str,
    o: float, h: float, l: float, c: float,
    o1: float, h1: float, l1: float, c1: float,
    o2: float, h2: float, l2: float, c2: float,
    atr_i: float,
) -> bool:
    """Demand (bullish) FVG filter ladder — mirrors FVGDetectorLibrary lines 35-55."""
    base = (l > h2)
    if not base:
        return False
    if filter_on == "Off":
        return True
    # filter_on == 'On' below — if filter_type is unknown, Pine's switch falls
    # through and no branch returns true → we return False.
    if filter_type == "Very Aggressive":
        return h > h1
    atr_valid = (not np.isnan(atr_i))
    if filter_type == "Aggressive":
        return atr_valid and ((h1 - l1) >= 1.0 * atr_i) and (h > h1)
    if filter_type == "Defensive":
        if not atr_valid:
            return False
        mid_bullish = (c2 - o2 > 0) and (c1 - o1 > 0)
        br_1 = _safe_body_ratio(o1, c1, h1, l1)
        return (
            ((h1 - l1) >= 1.5 * atr_i)
            and (h > h1)
            and (mid_bullish or (br_1 > 0.7))
        )
    if filter_type == "Very Defensive":
        if not atr_valid:
            return False
        mid_bullish = (c2 - o2 > 0) and (c1 - o1 > 0)
        br_1 = _safe_body_ratio(o1, c1, h1, l1)
        br_2 = _safe_body_ratio(o2, c2, h2, l2)
        br_c = _safe_body_ratio(o, c, h, l)
        return (
            ((h1 - l1) >= 1.5 * atr_i)
            and (h > h1)
            and mid_bullish and (br_1 > 0.7)
            and (br_2 > 0.35) and (br_c > 0.35)
        )
    return False


def _supply_filter(
    filter_on: str,
    filter_type: str,
    o: float, h: float, l: float, c: float,
    o1: float, h1: float, l1: float, c1: float,
    o2: float, h2: float, l2: float, c2: float,
    atr_i: float,
) -> bool:
    """Supply (bearish) FVG filter ladder — mirrors FVGDetectorLibrary lines 57-77."""
    base = (h < l2)
    if not base:
        return False
    if filter_on == "Off":
        return True
    if filter_type == "Very Aggressive":
        return l1 > l
    atr_valid = (not np.isnan(atr_i))
    if filter_type == "Aggressive":
        return atr_valid and ((h1 - l1) >= 1.0 * atr_i) and (l1 > l)
    if filter_type == "Defensive":
        if not atr_valid:
            return False
        mid_bearish = (c2 - o2 < 0) and (c1 - o1 < 0)
        br_1 = _safe_body_ratio(o1, c1, h1, l1)
        return (
            ((h1 - l1) >= 1.5 * atr_i)
            and (l1 > l)
            and (mid_bearish or (br_1 > 0.7))
        )
    if filter_type == "Very Defensive":
        if not atr_valid:
            return False
        mid_bearish = (c2 - o2 < 0) and (c1 - o1 < 0)
        br_1 = _safe_body_ratio(o1, c1, h1, l1)
        br_2 = _safe_body_ratio(o2, c2, h2, l2)
        br_c = _safe_body_ratio(o, c, h, l)
        return (
            ((h1 - l1) >= 1.5 * atr_i)
            and (l1 > l)
            and mid_bearish and (br_1 > 0.7)
            and (br_2 > 0.35) and (br_c > 0.35)
        )
    return False


def _resolve_ml(level: str, proximal: float, distal: float) -> float:
    """Pine switch: 'Proximal' → proximal, 'Distal' → distal, '50 % OB' → midpoint."""
    if level == "Proximal":
        return proximal
    if level == "Distal":
        return distal
    if level == "50 % OB":
        return (proximal + distal) / 2.0
    return proximal  # defensive fallback


# ---------------------------------------------------------------------------
# Main indicator — single bar loop preserving Pine evaluation order
# ---------------------------------------------------------------------------
def run_bpr_indicator(
    df: pd.DataFrame,
    *,
    fvg_filter: str = "On",
    fvg_filter_type: str = "Defensive",
    fvg_validity_period: int = 500,
    mitigation_level_bpr: str = "Proximal",
    mitigation_level_ifvg: str = "Proximal",
) -> BPRIndicatorOutput:
    """Run the full BPR / FVG + IFVG indicator on OHLC bars.

    Defaults match the Pine inputs (FVGVaP=500, MLBPR='Proximal',
    MLIFVG='Proximal', PFVGFilter=True, PFVGFilterType='Defensive').
    """
    if fvg_filter not in ("On", "Off"):
        raise ValueError(f"fvg_filter must be 'On' or 'Off', got {fvg_filter!r}")
    if fvg_filter == "On" and fvg_filter_type not in _VALID_FILTERS:
        raise ValueError(f"fvg_filter_type must be one of {_VALID_FILTERS}, got {fvg_filter_type!r}")
    if mitigation_level_bpr not in _VALID_MITIGATION:
        raise ValueError(f"mitigation_level_bpr must be one of {_VALID_MITIGATION}")
    if mitigation_level_ifvg not in _VALID_MITIGATION:
        raise ValueError(f"mitigation_level_ifvg must be one of {_VALID_MITIGATION}")
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"df missing required column {col!r}")

    n = len(df)
    atr = _atr(df["high"], df["low"], df["close"], 55)

    opn = df["open"].to_numpy(dtype=float)
    hgh = df["high"].to_numpy(dtype=float)
    low_ = df["low"].to_numpy(dtype=float)
    cls = df["close"].to_numpy(dtype=float)
    atr_np = atr.to_numpy(dtype=float)

    # ---------- FVGDetector state (shared) ----------
    DDFVG = 0.0; DPFVG = 0.0; BarDFVG = 0
    SDFVG = 0.0; SPFVG = 0.0; BarSFVG = 0

    # Per-bar output buffers for FVGDetector
    d_cond_buf = np.zeros(n, dtype=bool)
    d_distal_buf = np.zeros(n, dtype=float)
    d_proximal_buf = np.zeros(n, dtype=float)
    d_bar_idx_buf = np.zeros(n, dtype=np.int64)
    s_cond_buf = np.zeros(n, dtype=bool)
    s_distal_buf = np.zeros(n, dtype=float)
    s_proximal_buf = np.zeros(n, dtype=float)
    s_bar_idx_buf = np.zeros(n, dtype=np.int64)

    # ---------- OBDrawing('Demand') state ----------
    # Primary zone tracks demand FVG; BB tracks supply IFVG (flipped demand FVG).
    d_DistalPrice = 0.0
    d_ProximalPrice = 0.0
    d_Check = True            # Pine init value — zone "live" until first real FVG
    d_ProximalPrice_BB = 0.0
    d_DistalPrice_BB = 0.0
    d_Check_BB = False
    d_Index_BB = 0
    d_CBB_hist = [False, False, False, False]  # [CBB, CBB[1], CBB[2], CBB[3]] — we rotate

    d_fvg_live_buf = np.zeros(n, dtype=bool)
    d_fvg_mit_buf = np.zeros(n, dtype=bool)
    d_ifvg_trig_buf = np.zeros(n, dtype=bool)
    d_ifvg_live_buf = np.zeros(n, dtype=bool)
    d_ifvg_mit_buf = np.zeros(n, dtype=bool)
    d_ifvg_prox_buf = np.zeros(n, dtype=float)
    d_ifvg_dist_buf = np.zeros(n, dtype=float)
    d_ifvg_idx_buf = np.zeros(n, dtype=np.int64)

    # ---------- OBDrawing('Supply') state ----------
    s_DistalPrice = 0.0
    s_ProximalPrice = 0.0
    s_Check = True
    s_ProximalPrice_BB = 0.0
    s_DistalPrice_BB = 0.0
    s_Check_BB = False
    s_Index_BB = 0
    s_CBB_hist = [False, False, False, False]

    s_fvg_live_buf = np.zeros(n, dtype=bool)
    s_fvg_mit_buf = np.zeros(n, dtype=bool)
    s_ifvg_trig_buf = np.zeros(n, dtype=bool)
    s_ifvg_live_buf = np.zeros(n, dtype=bool)
    s_ifvg_mit_buf = np.zeros(n, dtype=bool)
    s_ifvg_prox_buf = np.zeros(n, dtype=float)
    s_ifvg_dist_buf = np.zeros(n, dtype=float)
    s_ifvg_idx_buf = np.zeros(n, dtype=np.int64)

    # ---------- OBOverlappingDrawing('Demand') — bullish BPR ----------
    # Reads: TriggerConditionOrigin = d_cond, _Pre = demand IFVG (from supply OBDrawing),
    # _Curr = new demand FVG levels.
    bpr_d_ProximalPrice = 0.0
    bpr_d_DistalPrice = 0.0
    bpr_d_Check = True
    bpr_d_Bar = 0  # last Index_Curr for which we processed an overlap

    bpr_d_trig_buf = np.zeros(n, dtype=bool)
    bpr_d_live_buf = np.zeros(n, dtype=bool)
    bpr_d_mit_buf = np.zeros(n, dtype=bool)
    bpr_d_prox_buf = np.zeros(n, dtype=float)
    bpr_d_dist_buf = np.zeros(n, dtype=float)
    bpr_d_idx_buf = np.zeros(n, dtype=np.int64)

    # ---------- OBOverlappingDrawing('Supply') — bearish BPR ----------
    bpr_s_ProximalPrice = 0.0
    bpr_s_DistalPrice = 0.0
    bpr_s_Check = True
    bpr_s_Bar = 0

    bpr_s_trig_buf = np.zeros(n, dtype=bool)
    bpr_s_live_buf = np.zeros(n, dtype=bool)
    bpr_s_mit_buf = np.zeros(n, dtype=bool)
    bpr_s_prox_buf = np.zeros(n, dtype=float)
    bpr_s_dist_buf = np.zeros(n, dtype=float)
    bpr_s_idx_buf = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # =====================================================================
        # 1. FVGDetector (library #1)
        # =====================================================================
        d_cond = False
        s_cond = False
        if i >= 2:
            atr_i = atr_np[i]
            d_cond = _demand_filter(
                fvg_filter, fvg_filter_type,
                opn[i], hgh[i], low_[i], cls[i],
                opn[i - 1], hgh[i - 1], low_[i - 1], cls[i - 1],
                opn[i - 2], hgh[i - 2], low_[i - 2], cls[i - 2],
                atr_i,
            )
            s_cond = _supply_filter(
                fvg_filter, fvg_filter_type,
                opn[i], hgh[i], low_[i], cls[i],
                opn[i - 1], hgh[i - 1], low_[i - 1], cls[i - 1],
                opn[i - 2], hgh[i - 2], low_[i - 2], cls[i - 2],
                atr_i,
            )

        if d_cond:
            DDFVG = hgh[i - 2]
            DPFVG = low_[i]
            BarDFVG = i
        if s_cond:
            SDFVG = low_[i - 2]
            SPFVG = hgh[i]
            BarSFVG = i

        d_cond_buf[i] = d_cond
        d_distal_buf[i] = DDFVG
        d_proximal_buf[i] = DPFVG
        d_bar_idx_buf[i] = BarDFVG
        s_cond_buf[i] = s_cond
        s_distal_buf[i] = SDFVG
        s_proximal_buf[i] = SPFVG
        s_bar_idx_buf[i] = BarSFVG

        # =====================================================================
        # 2. OBDrawing('Demand')  — tracks demand FVG, spawns supply IFVG
        # =====================================================================
        d_Check_prev = d_Check
        d_Check_BB_prev = d_Check_BB

        # Update zone levels on new trigger
        if d_cond:
            d_DistalPrice = DDFVG
            d_ProximalPrice = DPFVG

        # Mitigation level for the primary zone + for the BB zone
        d_ML = _resolve_ml(mitigation_level_ifvg, d_ProximalPrice, d_DistalPrice)
        d_ML_BB = _resolve_ml(mitigation_level_ifvg, d_ProximalPrice_BB, d_DistalPrice_BB)

        # Primary zone demotion: low punched proximal/ML, or aged out
        if (low_[i] < d_ML) or ((i - BarDFVG) >= fvg_validity_period):
            d_Check = False
        # New trigger refreshes zone to live (overrides the demotion above if same bar)
        if d_cond:
            d_Check = True

        # CBB edge: zone just went live→mitigated this bar (using values computed above)
        d_CBB_current = (d_Check_prev is True) and (d_Check is False)

        # BB promotion: within a 4-bar window after a CBB, on first close past distal
        window_active = d_CBB_current or d_CBB_hist[0] or d_CBB_hist[1] or d_CBB_hist[2]
        if window_active and (not d_Check_BB):
            if cls[i] < d_DistalPrice:
                d_Index_BB = i
                d_Check_BB = True
                # Swap levels: old proximal → new distal, old distal → new proximal
                d_DistalPrice_BB = d_ProximalPrice
                d_ProximalPrice_BB = d_DistalPrice
                # Recompute ML_BB now that the levels just updated
                d_ML_BB = _resolve_ml(mitigation_level_ifvg, d_ProximalPrice_BB, d_DistalPrice_BB)

        # BB demotion: high punched BB's ML, aged out, or another FVG mitigation started
        if (
            (hgh[i] > d_ML_BB)
            or ((i - d_Index_BB) >= fvg_validity_period)
            or (d_Check_prev is True and d_Check is False)
        ):
            d_Check_BB = False

        # Mitigation alerts (edge flags)
        d_mit_alert = (d_Check_prev is True) and (d_Check is False)
        d_mit_alert_bb = (
            (d_Check_BB_prev is True) and (d_Check_BB is False) and (not d_mit_alert)
        )
        d_ifvg_trig_edge = (d_Check_BB_prev is False) and (d_Check_BB is True)

        # Rotate CBB history: [curr, prev, prev2, prev3]
        d_CBB_hist = [d_CBB_current, d_CBB_hist[0], d_CBB_hist[1], d_CBB_hist[2]]

        d_fvg_live_buf[i] = d_Check
        d_fvg_mit_buf[i] = d_mit_alert
        d_ifvg_trig_buf[i] = d_ifvg_trig_edge
        d_ifvg_live_buf[i] = d_Check_BB
        d_ifvg_mit_buf[i] = d_mit_alert_bb
        d_ifvg_prox_buf[i] = d_ProximalPrice_BB
        d_ifvg_dist_buf[i] = d_DistalPrice_BB
        d_ifvg_idx_buf[i] = d_Index_BB

        # =====================================================================
        # 3. OBDrawing('Supply') — tracks supply FVG, spawns demand IFVG
        # =====================================================================
        s_Check_prev = s_Check
        s_Check_BB_prev = s_Check_BB

        if s_cond:
            s_DistalPrice = SDFVG
            s_ProximalPrice = SPFVG

        s_ML = _resolve_ml(mitigation_level_ifvg, s_ProximalPrice, s_DistalPrice)
        s_ML_BB = _resolve_ml(mitigation_level_ifvg, s_ProximalPrice_BB, s_DistalPrice_BB)

        # Supply zone demotion: high punched proximal/ML, or aged out
        if (hgh[i] > s_ML) or ((i - BarSFVG) >= fvg_validity_period):
            s_Check = False
        if s_cond:
            s_Check = True

        s_CBB_current = (s_Check_prev is True) and (s_Check is False)

        # Supply BB promotion: on first close above distal, within 4-bar window
        window_active = s_CBB_current or s_CBB_hist[0] or s_CBB_hist[1] or s_CBB_hist[2]
        if window_active and (not s_Check_BB):
            if cls[i] > s_DistalPrice:
                s_Index_BB = i
                s_Check_BB = True
                s_DistalPrice_BB = s_ProximalPrice
                s_ProximalPrice_BB = s_DistalPrice
                s_ML_BB = _resolve_ml(mitigation_level_ifvg, s_ProximalPrice_BB, s_DistalPrice_BB)

        # Demand BB demotion: low punched ML, aged out, or another supply mitigation started
        if (
            (low_[i] < s_ML_BB)
            or ((i - s_Index_BB) >= fvg_validity_period)
            or (s_Check_prev is True and s_Check is False)
        ):
            s_Check_BB = False

        s_mit_alert = (s_Check_prev is True) and (s_Check is False)
        s_mit_alert_bb = (
            (s_Check_BB_prev is True) and (s_Check_BB is False) and (not s_mit_alert)
        )
        s_ifvg_trig_edge = (s_Check_BB_prev is False) and (s_Check_BB is True)

        s_CBB_hist = [s_CBB_current, s_CBB_hist[0], s_CBB_hist[1], s_CBB_hist[2]]

        s_fvg_live_buf[i] = s_Check
        s_fvg_mit_buf[i] = s_mit_alert
        s_ifvg_trig_buf[i] = s_ifvg_trig_edge
        s_ifvg_live_buf[i] = s_Check_BB
        s_ifvg_mit_buf[i] = s_mit_alert_bb
        s_ifvg_prox_buf[i] = s_ProximalPrice_BB
        s_ifvg_dist_buf[i] = s_DistalPrice_BB
        s_ifvg_idx_buf[i] = s_Index_BB

        # =====================================================================
        # 4. OBOverlappingDrawing('Demand') — BULLISH BPR
        #    _Pre = demand IFVG (from Supply OBDrawing, just written above)
        #    _Curr = new demand FVG
        # =====================================================================
        bpr_d_Check_prev = bpr_d_Check

        # Capture "is there a new demand FVG this bar and we haven't already processed it?"
        bpr_d_trigger = False
        if d_cond and (bpr_d_Bar != BarDFVG):
            bpr_d_Bar = BarDFVG
            # _Pre = demand IFVG (from Supply OBDrawing):  s_ProximalPrice_BB, s_DistalPrice_BB
            # _Curr = new demand FVG:                      DDFVG, DPFVG
            pp_pre = s_ProximalPrice_BB
            dp_pre = s_DistalPrice_BB
            pp_curr = DPFVG  # proximal of bullish FVG = top
            dp_curr = DDFVG  # distal = bottom

            # Three cases for demand overlap (direct translation of Pine):
            if (pp_curr >= pp_pre) and (dp_curr <= pp_pre) and (dp_curr >= dp_pre):
                bpr_d_ProximalPrice = pp_pre
                bpr_d_DistalPrice = dp_curr
                bpr_d_trigger = True
            elif (pp_curr <= pp_pre) and (dp_pre <= pp_curr) and (dp_curr <= dp_pre):
                bpr_d_ProximalPrice = pp_curr
                bpr_d_DistalPrice = dp_pre
                bpr_d_trigger = True
            elif (pp_curr <= pp_pre) and (dp_pre <= pp_curr):
                bpr_d_ProximalPrice = pp_curr
                bpr_d_DistalPrice = dp_curr
                bpr_d_trigger = True
            # else: no overlap — leave Trigger=false (matches Pine's explicit else)

        # Mitigation level for BPR
        bpr_d_ML = _resolve_ml(mitigation_level_bpr, bpr_d_ProximalPrice, bpr_d_DistalPrice)

        # Demand BPR demotion
        if (low_[i] < bpr_d_ML) or ((i - BarDFVG) >= fvg_validity_period):
            bpr_d_Check = False
        # New BPR trigger refreshes
        if bpr_d_trigger:
            bpr_d_Check = True

        bpr_d_mit_alert = (bpr_d_Check_prev is True) and (bpr_d_Check is False)

        bpr_d_trig_buf[i] = bpr_d_trigger
        bpr_d_live_buf[i] = bpr_d_Check
        bpr_d_mit_buf[i] = bpr_d_mit_alert
        bpr_d_prox_buf[i] = bpr_d_ProximalPrice
        bpr_d_dist_buf[i] = bpr_d_DistalPrice
        bpr_d_idx_buf[i] = BarDFVG

        # =====================================================================
        # 5. OBOverlappingDrawing('Supply') — BEARISH BPR
        #    _Pre = supply IFVG (from Demand OBDrawing):  d_ProximalPrice_BB, d_DistalPrice_BB
        #    _Curr = new supply FVG
        # =====================================================================
        bpr_s_Check_prev = bpr_s_Check

        bpr_s_trigger = False
        if s_cond and (bpr_s_Bar != BarSFVG):
            bpr_s_Bar = BarSFVG
            pp_pre = d_ProximalPrice_BB
            dp_pre = d_DistalPrice_BB
            pp_curr = SPFVG  # proximal of bearish FVG = bottom
            dp_curr = SDFVG  # distal = top

            # For Supply, Distal > Proximal (top > bottom)
            if (dp_curr >= dp_pre) and (dp_pre >= pp_curr) and (pp_pre <= pp_curr):
                bpr_s_ProximalPrice = pp_curr
                bpr_s_DistalPrice = dp_pre
                bpr_s_trigger = True
            elif (dp_pre >= dp_curr) and (pp_pre <= dp_curr) and (pp_pre >= pp_curr):
                bpr_s_ProximalPrice = pp_pre
                bpr_s_DistalPrice = dp_curr
                bpr_s_trigger = True
            elif (dp_pre >= dp_curr) and (pp_curr >= pp_pre):
                bpr_s_ProximalPrice = pp_curr
                bpr_s_DistalPrice = dp_curr
                bpr_s_trigger = True

        bpr_s_ML = _resolve_ml(mitigation_level_bpr, bpr_s_ProximalPrice, bpr_s_DistalPrice)

        if (hgh[i] > bpr_s_ML) or ((i - BarSFVG) >= fvg_validity_period):
            bpr_s_Check = False
        if bpr_s_trigger:
            bpr_s_Check = True

        bpr_s_mit_alert = (bpr_s_Check_prev is True) and (bpr_s_Check is False)

        bpr_s_trig_buf[i] = bpr_s_trigger
        bpr_s_live_buf[i] = bpr_s_Check
        bpr_s_mit_buf[i] = bpr_s_mit_alert
        bpr_s_prox_buf[i] = bpr_s_ProximalPrice
        bpr_s_dist_buf[i] = bpr_s_DistalPrice
        bpr_s_idx_buf[i] = BarSFVG

    # ---------- Pack into dataclasses ----------
    idx = df.index

    fvg = FVGSeries(
        d_condition=pd.Series(d_cond_buf, index=idx),
        d_distal=pd.Series(d_distal_buf, index=idx),
        d_proximal=pd.Series(d_proximal_buf, index=idx),
        d_bar_idx=pd.Series(d_bar_idx_buf, index=idx),
        s_condition=pd.Series(s_cond_buf, index=idx),
        s_distal=pd.Series(s_distal_buf, index=idx),
        s_proximal=pd.Series(s_proximal_buf, index=idx),
        s_bar_idx=pd.Series(s_bar_idx_buf, index=idx),
    )

    obd_demand = OBDrawingSeries(
        fvg_live=pd.Series(d_fvg_live_buf, index=idx),
        fvg_mitigated_alert=pd.Series(d_fvg_mit_buf, index=idx),
        ifvg_trigger=pd.Series(d_ifvg_trig_buf, index=idx),
        ifvg_live=pd.Series(d_ifvg_live_buf, index=idx),
        ifvg_mitigated_alert=pd.Series(d_ifvg_mit_buf, index=idx),
        ifvg_proximal=pd.Series(d_ifvg_prox_buf, index=idx),
        ifvg_distal=pd.Series(d_ifvg_dist_buf, index=idx),
        ifvg_bar_idx=pd.Series(d_ifvg_idx_buf, index=idx),
    )

    obd_supply = OBDrawingSeries(
        fvg_live=pd.Series(s_fvg_live_buf, index=idx),
        fvg_mitigated_alert=pd.Series(s_fvg_mit_buf, index=idx),
        ifvg_trigger=pd.Series(s_ifvg_trig_buf, index=idx),
        ifvg_live=pd.Series(s_ifvg_live_buf, index=idx),
        ifvg_mitigated_alert=pd.Series(s_ifvg_mit_buf, index=idx),
        ifvg_proximal=pd.Series(s_ifvg_prox_buf, index=idx),
        ifvg_distal=pd.Series(s_ifvg_dist_buf, index=idx),
        ifvg_bar_idx=pd.Series(s_ifvg_idx_buf, index=idx),
    )

    bpr_bullish = BPRSeries(
        trigger=pd.Series(bpr_d_trig_buf, index=idx),
        live=pd.Series(bpr_d_live_buf, index=idx),
        mitigated_alert=pd.Series(bpr_d_mit_buf, index=idx),
        proximal=pd.Series(bpr_d_prox_buf, index=idx),
        distal=pd.Series(bpr_d_dist_buf, index=idx),
        bar_idx=pd.Series(bpr_d_idx_buf, index=idx),
    )

    bpr_bearish = BPRSeries(
        trigger=pd.Series(bpr_s_trig_buf, index=idx),
        live=pd.Series(bpr_s_live_buf, index=idx),
        mitigated_alert=pd.Series(bpr_s_mit_buf, index=idx),
        proximal=pd.Series(bpr_s_prox_buf, index=idx),
        distal=pd.Series(bpr_s_dist_buf, index=idx),
        bar_idx=pd.Series(bpr_s_idx_buf, index=idx),
    )

    return BPRIndicatorOutput(
        fvg=fvg,
        obd_demand=obd_demand,
        obd_supply=obd_supply,
        bpr_bullish=bpr_bullish,
        bpr_bearish=bpr_bearish,
        atr=atr,
    )


__all__ = [
    "run_bpr_indicator",
    "BPRIndicatorOutput",
    "BPRSeries",
    "OBDrawingSeries",
    "FVGSeries",
]
