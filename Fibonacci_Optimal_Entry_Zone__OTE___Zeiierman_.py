"""Fibonacci Optimal Entry Zone [OTE] (Zeiierman).

Faithful Python translation of the Zeiierman Pine v6 indicator
(``Fibonacci Optimal Entry Zone [OTE] (Zeiierman).txt``).

The Pine source is a single-timeframe state machine. This module mirrors
every bar-by-bar computation exactly (Pine behaviour is sacred here; the
analyst-facing multi-timeframe wrapper lives in ``tools/ict_ote.py``).

State variables mirroring the Pine ``var`` declarations:

- ``Up`` / ``Dn``: running max / min of ``high`` / ``low`` since the last
  pivot reset, initialised to NaN (Pine ``var Up = float(na)``).
- ``iUp`` / ``iDn``: anchor bar indices for the Fibonacci geometry.
- ``pos``: regime counter.
    *  0 = neutral
    * +1 = first bullish CHoCH; +2 = first bullish continuation;
      +3, +4, ... = further bullish continuations
    * -1 = first bearish CHoCH; -2, -3, ... = bearish continuations
- ``swingLow`` / ``swingHigh`` (+ their indices): anchor values captured
  at the moment of the CHoCH flip, used when ``follow=False``.

Per bar (in order, as in Pine):

1. Snapshot ``prev_up = Up``, ``prev_dn = Dn``, ``prev_i_up = iUp``,
   ``prev_i_dn = iDn``, ``prev_pos = pos``.
2. ``Up := math.max(Up[1], high)`` — na-propagating max (see
   :func:`_pine_max`).
3. ``Dn := math.min(Dn[1], low)`` — symmetric.
4. ``pvtHi = ta.pivothigh(high, prd, prd)`` and ``pvtLo = ta.pivotlow(...)``
   evaluated on bar ``b``, returning ``high[b-prd]`` / ``low[b-prd]`` iff
   that bar is strictly greater/less than every other bar in the
   ``[b - 2*prd, b]`` window.
5. If ``pvtHi`` confirmed and ``prev_pos <= 0``: ``Up := pvtHi``.
6. If ``pvtLo`` confirmed and ``prev_pos >= 0``: ``Dn := pvtLo``.
7. **Bullish branch** (``if Up > Up[1] ... elif Up < Up[1] ...``): sets
   ``iUp`` and, on a structural transition, mutates ``pos`` / emits a
   CHoCH or continuation event.
8. **Bearish branch** (``if Dn < Dn[1] ... elif Dn > Dn[1] ...``) —
   executes AFTER the bullish branch and sees the possibly-updated
   ``pos`` (this matches Pine's flat ``if`` blocks, not nested).

Pine fib geometry:

    fibb(v, h, l, ih, il) =
        h - (h - l) * v      if il < ih   (low formed first)
        l + (h - l) * v      if il > ih   (high formed first)
        NaN                  if il == ih  (degenerate)

Active OTE zone orientation matches the Pine draw:

- Bullish: ``fibb(v, Up, Dn, iUp, iDn)`` — retrace from the anchor
  high toward the anchor low.
- Bearish: ``fibb(v, Up, Dn, iUp, iDn)`` in ``follow=True`` mode, or
  ``fibb(v, Dn, swingHigh, iDn, iswingHigh)`` in ``follow=False`` mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


_NA_IDX = -1  # sentinel for Pine ``int(na)`` anchor indices


# ---------------------------------------------------------------------------
# Settings — mirror Pine's ``input.*`` declarations, defaults preserved.
# ---------------------------------------------------------------------------
@dataclass
class OTESettings:
    """Pine input parameters, defaults identical to Pine v6 source."""

    prd: int = 10                         # input.int(10, minval=1)
    levels: List[float] = field(
        default_factory=lambda: [0.5, 0.618]
    )                                     # inputLevels filtered by levelsBool
    follow: bool = True                   # input.bool(true, 'Swing tracker')
    show_old: bool = False                # input.bool(false, 'Previous')
    extend: bool = True                   # input.bool(true, 'Extend')
    enable_bull: bool = True              # input.bool(true, 'Bullish Structure')
    enable_bear: bool = True              # input.bool(true, 'Bearish Structure')

    def __post_init__(self) -> None:
        if self.prd < 1:
            raise ValueError("prd must be >= 1 (Pine input.int minval=1)")
        self.levels = [float(v) for v in self.levels]


# ---------------------------------------------------------------------------
# Event / state dataclasses — return shape of :func:`calculate_ote`.
# ---------------------------------------------------------------------------
@dataclass
class OTEEvent:
    """One CHoCH or continuation emitted by the Pine state machine."""

    bar: int                              # current iteration index b
    center_bar: int                       # round((iUp[1] + b) / 2) — Pine label anchor
    price: float                          # Up[1] (bullish) / Dn[1] (bearish) at emission
    kind: str                             # "choch" | "continuation"
    direction: str                        # "bullish" | "bearish"
    pos_after: int                        # pos immediately after this event
    anchor_up: float                      # Up at emission (after pivot override)
    anchor_dn: float                      # Dn at emission
    anchor_i_up: int                      # iUp at emission
    anchor_i_dn: int                      # iDn at emission
    fib_values: List[float]               # fib level prices for the (just-drawn/updated) zone


@dataclass
class OTEBarState:
    """Per-bar snapshot (mirrors Pine's live variables at bar close)."""

    bar: int
    up: float
    dn: float
    i_up: int
    i_dn: int
    pos: int
    swing_low: float
    swing_high: float
    active_direction: str                 # "bullish" | "bearish" | "neutral"
    fib_prices: List[float]               # NaN-filled list of length len(settings.levels)


@dataclass
class OTEActiveZone:
    """Snapshot of the currently-active OTE Fibonacci zone at the last bar."""

    direction: str                        # "bullish" | "bearish" | "neutral"
    anchor_up: float
    anchor_dn: float
    anchor_i_up: int
    anchor_i_dn: int
    fib_levels: List[float]               # input level values (0.5, 0.618, ...)
    fib_prices: List[float]               # corresponding prices (NaN when degenerate)
    zone_top: float                       # max(fib_prices) (ignoring NaN) or NaN
    zone_bottom: float                    # min(fib_prices) or NaN


@dataclass
class OTEOutputs:
    settings: OTESettings
    index: pd.DatetimeIndex
    events: List[OTEEvent]
    states: List[OTEBarState]
    active_zone: OTEActiveZone


# ---------------------------------------------------------------------------
# Pine helpers.
# ---------------------------------------------------------------------------
def _pine_max(a: float, b: float) -> float:
    """Pine's ``math.max`` with na propagation (na in → na out).

    The OTE indicator relies on this behaviour so that ``Up`` stays NaN
    until the first pivot confirms; a plain ``max(nan, x) == x`` would
    trigger spurious CHoCH events on bar 1.
    """
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a if a >= b else b


def _pine_min(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a if a <= b else b


def _ta_pivot_high(highs: np.ndarray, b: int, prd: int) -> float:
    """Pine ``ta.pivothigh(high, prd, prd)`` evaluated on bar ``b``.

    Returns ``highs[b - prd]`` iff that bar is strictly greater than
    every other bar in ``[b - 2*prd, b]``; otherwise NaN. Mirrors Pine's
    strict inequality on both sides (ties disqualify the pivot).
    """
    cand = b - prd
    if cand < prd or cand + prd > len(highs) - 1:
        return np.nan
    pivot_val = highs[cand]
    for j in range(cand - prd, cand + prd + 1):
        if j == cand:
            continue
        if highs[j] >= pivot_val:
            return np.nan
    return float(pivot_val)


def _ta_pivot_low(lows: np.ndarray, b: int, prd: int) -> float:
    cand = b - prd
    if cand < prd or cand + prd > len(lows) - 1:
        return np.nan
    pivot_val = lows[cand]
    for j in range(cand - prd, cand + prd + 1):
        if j == cand:
            continue
        if lows[j] <= pivot_val:
            return np.nan
    return float(pivot_val)


def _fibb(v: float, h: float, l: float, ih: int, il: int) -> float:
    """Pine's ``fibb`` retrace helper.

    * ``il < ih``: low formed first  → retrace from h downward:  h - (h - l) * v
    * ``il > ih``: high formed first → retrace from l upward:    l + (h - l) * v
    * ``il == ih`` or any NaN anchor → NaN (degenerate).
    """
    if np.isnan(h) or np.isnan(l) or ih == _NA_IDX or il == _NA_IDX:
        return np.nan
    if il < ih:
        return h - (h - l) * v
    if il > ih:
        return l + (h - l) * v
    return np.nan


def _zone_bounds(fib_prices: List[float]) -> tuple[float, float]:
    """Return (top, bottom) of an OTE zone, ignoring NaN levels."""
    arr = np.asarray([p for p in fib_prices if not np.isnan(p)], dtype=float)
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.max()), float(arr.min())


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def calculate_ote(
    df: pd.DataFrame,
    settings: Optional[OTESettings] = None,
) -> OTEOutputs:
    """Run the faithful Pine state machine over an OHLC(V) DataFrame.

    Parameters
    ----------
    df
        Timestamp-indexed DataFrame with at least ``high`` and ``low``
        columns.
    settings
        :class:`OTESettings` (defaults to Pine defaults when omitted).

    Returns
    -------
    :class:`OTEOutputs`
        * ``events``: every CHoCH / continuation in chronological order.
        * ``states``: per-bar ``OTEBarState`` (useful for back-audit).
        * ``active_zone``: snapshot of the latest OTE zone at the final bar.
    """
    if settings is None:
        settings = OTESettings()
    prd = settings.prd
    fib_in_levels = settings.levels

    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("df must have 'high' and 'low' columns")

    highs = np.asarray(df["high"].to_numpy(), dtype=float)
    lows = np.asarray(df["low"].to_numpy(), dtype=float)
    n = len(df)

    # Live state (mirror Pine ``var`` declarations).
    up = np.nan
    dn = np.nan
    i_up = _NA_IDX
    i_dn = _NA_IDX
    pos = 0

    swing_low = np.nan
    swing_high = np.nan
    i_swing_low = _NA_IDX
    i_swing_high = _NA_IDX

    events: List[OTEEvent] = []
    states: List[OTEBarState] = []

    # Fib-price tracking that mirrors Pine's ``flevels`` array + showOld.
    # Pine behaviour:
    #   - On CHoCH: clear flevels (if not showOld) and create fresh level lines.
    #   - On continuation (same direction): UpdateLine — i.e. same logical
    #     zone, new fib values.
    # Active zone mirror:
    current_direction = "neutral"
    current_fib_prices: List[float] = [np.nan for _ in fib_in_levels]
    current_anchor_up = np.nan
    current_anchor_dn = np.nan
    current_anchor_i_up = _NA_IDX
    current_anchor_i_dn = _NA_IDX

    for b in range(n):
        # ---- 1. snapshot prior-bar state (Pine [1]) ----
        prev_up = up
        prev_dn = dn
        prev_i_up = i_up
        prev_i_dn = i_dn
        prev_pos = pos

        # ---- 2-3. running max/min ----
        up = _pine_max(prev_up, highs[b])
        dn = _pine_min(prev_dn, lows[b])

        # ---- 4-6. pivot detection + conditional override ----
        pvt_hi = _ta_pivot_high(highs, b, prd)
        pvt_lo = _ta_pivot_low(lows, b, prd)
        if not np.isnan(pvt_hi) and prev_pos <= 0:
            up = pvt_hi
        if not np.isnan(pvt_lo) and prev_pos >= 0:
            dn = pvt_lo

        # ---- 7. Bullish branch ----
        up_gt_prev = (
            (not np.isnan(up))
            and (not np.isnan(prev_up))
            and (up > prev_up)
        )
        up_lt_prev = (
            (not np.isnan(up))
            and (not np.isnan(prev_up))
            and (up < prev_up)
        )

        if up_gt_prev:
            i_up = b
            # Pine: centerBull = math.round(math.avg(iUp[1], b))
            # iUp[1] is the prior-bar value of iUp (captured pre-mutation).
            center = (
                int(round((prev_i_up + b) / 2))
                if prev_i_up != _NA_IDX
                else b
            )

            if prev_pos <= 0:
                # ---------- bullish CHoCH ----------
                if settings.enable_bull:
                    fvals = [
                        _fibb(l, up, dn, i_up, i_dn) for l in fib_in_levels
                    ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_up),
                            kind="choch",
                            direction="bullish",
                            pos_after=1,
                            anchor_up=float(up),
                            anchor_dn=float(dn) if not np.isnan(dn) else np.nan,
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    # Pine: if not showOld → delete(trend) + delete all flevels,
                    # then flevels.clear(); recreate. On show_old=True, old
                    # zones are kept on the chart — but our adapter's
                    # ``active_zone`` always reflects the latest zone.
                    current_direction = "bullish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up)
                    current_anchor_dn = float(dn) if not np.isnan(dn) else np.nan
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = i_dn
                pos = 1
                swing_low = dn
                i_swing_low = i_dn

            elif prev_pos == 1:
                # ---------- first bullish continuation ----------
                if settings.enable_bull:
                    target_low = dn if settings.follow else swing_low
                    target_i = i_dn if settings.follow else i_swing_low
                    fvals = [
                        _fibb(l, up, target_low, i_up, target_i)
                        for l in fib_in_levels
                    ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_up),
                            kind="continuation",
                            direction="bullish",
                            pos_after=2,
                            anchor_up=float(up),
                            anchor_dn=float(dn) if not np.isnan(dn) else np.nan,
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    current_direction = "bullish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up)
                    current_anchor_dn = (
                        float(target_low) if not np.isnan(target_low) else np.nan
                    )
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = target_i
                pos = 2

            elif prev_pos > 1:
                # ---------- further bullish continuation ----------
                if settings.enable_bull:
                    target_low = dn if settings.follow else swing_low
                    target_i = i_dn if settings.follow else i_swing_low
                    fvals = [
                        _fibb(l, up, target_low, i_up, target_i)
                        for l in fib_in_levels
                    ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_up),
                            kind="continuation",
                            direction="bullish",
                            pos_after=prev_pos + 1,
                            anchor_up=float(up),
                            anchor_dn=float(dn) if not np.isnan(dn) else np.nan,
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    current_direction = "bullish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up)
                    current_anchor_dn = (
                        float(target_low) if not np.isnan(target_low) else np.nan
                    )
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = target_i
                pos = prev_pos + 1

        elif up_lt_prev:
            # Pine: iUp := b - prd — relocate the anchor to the actual pivot bar
            # (this happens only in a bearish regime when a lower pivot high
            # overrides the running max).
            i_up = b - prd

        # ---- 8. Bearish branch (reads the POSSIBLY-UPDATED pos) ----
        dn_lt_prev = (
            (not np.isnan(dn))
            and (not np.isnan(prev_dn))
            and (dn < prev_dn)
        )
        dn_gt_prev = (
            (not np.isnan(dn))
            and (not np.isnan(prev_dn))
            and (dn > prev_dn)
        )

        if dn_lt_prev:
            i_dn = b
            # Pine: centerBear = math.round(math.avg(iDn[1], b))
            center = (
                int(round((prev_i_dn + b) / 2))
                if prev_i_dn != _NA_IDX
                else b
            )

            if pos >= 0:
                # ---------- bearish CHoCH ----------
                if settings.enable_bear:
                    # Pine: fibb(l, Dn, Up, iDn, iUp) — h=Dn, l=Up, ih=iDn, il=iUp.
                    # iUp < iDn (bearish flip), so _fibb returns Dn + (Up-Dn)*v
                    # (retrace from the anchor low upward toward the high).
                    fvals = [
                        _fibb(l, dn, up, i_dn, i_up) for l in fib_in_levels
                    ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_dn),
                            kind="choch",
                            direction="bearish",
                            pos_after=-1,
                            anchor_up=float(up) if not np.isnan(up) else np.nan,
                            anchor_dn=float(dn),
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    current_direction = "bearish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up) if not np.isnan(up) else np.nan
                    current_anchor_dn = float(dn)
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = i_dn
                pos = -1
                swing_high = up
                i_swing_high = i_up

            elif pos == -1:
                # ---------- first bearish continuation ----------
                if settings.enable_bear:
                    # Pine: val = follow ? fibb(l, Up, Dn, iUp, iDn)
                    #                    : fibb(l, Dn, swingHigh, iDn, iswingHigh)
                    if settings.follow:
                        fvals = [
                            _fibb(l, up, dn, i_up, i_dn) for l in fib_in_levels
                        ]
                    else:
                        fvals = [
                            _fibb(l, dn, swing_high, i_dn, i_swing_high)
                            for l in fib_in_levels
                        ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_dn),
                            kind="continuation",
                            direction="bearish",
                            pos_after=-2,
                            anchor_up=float(up) if not np.isnan(up) else np.nan,
                            anchor_dn=float(dn),
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    current_direction = "bearish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up) if not np.isnan(up) else np.nan
                    current_anchor_dn = float(dn)
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = i_dn
                pos = -2

            elif pos < -1:
                # ---------- further bearish continuation ----------
                if settings.enable_bear:
                    if settings.follow:
                        fvals = [
                            _fibb(l, up, dn, i_up, i_dn) for l in fib_in_levels
                        ]
                    else:
                        fvals = [
                            _fibb(l, dn, swing_high, i_dn, i_swing_high)
                            for l in fib_in_levels
                        ]
                    events.append(
                        OTEEvent(
                            bar=b,
                            center_bar=center,
                            price=float(prev_dn),
                            kind="continuation",
                            direction="bearish",
                            pos_after=pos - 1,
                            anchor_up=float(up) if not np.isnan(up) else np.nan,
                            anchor_dn=float(dn),
                            anchor_i_up=i_up,
                            anchor_i_dn=i_dn,
                            fib_values=list(fvals),
                        )
                    )
                    current_direction = "bearish"
                    current_fib_prices = list(fvals)
                    current_anchor_up = float(up) if not np.isnan(up) else np.nan
                    current_anchor_dn = float(dn)
                    current_anchor_i_up = i_up
                    current_anchor_i_dn = i_dn
                pos = pos - 1

        elif dn_gt_prev:
            # Pine: iDn := b - prd (symmetric to the up_lt_prev housekeeping)
            i_dn = b - prd

        # ---- record per-bar state ----
        states.append(
            OTEBarState(
                bar=b,
                up=float(up) if not np.isnan(up) else np.nan,
                dn=float(dn) if not np.isnan(dn) else np.nan,
                i_up=i_up,
                i_dn=i_dn,
                pos=pos,
                swing_low=float(swing_low) if not np.isnan(swing_low) else np.nan,
                swing_high=float(swing_high) if not np.isnan(swing_high) else np.nan,
                active_direction=current_direction,
                fib_prices=list(current_fib_prices),
            )
        )

    zone_top, zone_bottom = _zone_bounds(current_fib_prices)
    active = OTEActiveZone(
        direction=current_direction,
        anchor_up=current_anchor_up,
        anchor_dn=current_anchor_dn,
        anchor_i_up=current_anchor_i_up,
        anchor_i_dn=current_anchor_i_dn,
        fib_levels=list(fib_in_levels),
        fib_prices=list(current_fib_prices),
        zone_top=zone_top,
        zone_bottom=zone_bottom,
    )

    return OTEOutputs(
        settings=settings,
        index=pd.DatetimeIndex(df.index),
        events=events,
        states=states,
        active_zone=active,
    )
