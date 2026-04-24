"""Python translation of `ICT HTF MSS & Liquidity (fadi)` Pine Script v6.

Source: `S1_Silver_Bullet/ICT HTF MSS & Liquidity (fadi).txt` (© fadizeidan,
MPL-2.0).

Detection flow — preserved 1:1 with the Pine source
---------------------------------------------------
The Pine script runs on an LTF chart and queries an HTF via
``request.security(…, lookahead = barmerge.lookahead_on)``. Each time a new
HTF bar starts (``ta.change(time(HTF))``) it builds a rolling window of the
six most-recently-completed HTF highs/lows and runs ``findST``:

    _h = highs[1] > highs[SkipEQ(2)]  AND  highs[1] > highs[0]
    _l =  lows[1] <  lows[SkipEQ(2)]  AND   lows[1] <  lows[0]

(highs[0] = last closed HTF bar, highs[1] = the HTF bar before that, …).
A positive check creates a new Short-Term pivot (STH / STL) at
``(time=times[1], price=highs[1]|lows[1])``. The newly-added pivot then
triggers ``FindIT`` and ``FindLT`` which look for 3-pivot fractals inside the
STH/ITH (and STL/ITL) sequences to promote a pivot from ST → IT → LT.

Every LTF bar ``CheckClaimed`` iterates the pivots of the currently-selected
tier (Pine uses `if not na(pivot.ln)` to restrict to the drawn tier):

    * first time ``close`` pushes beyond ``pivot.price`` → *claimed* (the
      liquidity is taken).
    * once claimed, the first time ``close`` pushes back through
      ``pivot.price`` → *reclaimed* (the MSS / "Break of Structure back into
      the taken liquidity"). All older, still-claimed-but-not-reclaimed
      same-side pivots are cascade-marked reclaimed to prevent double
      triggering.

The Pine source also defines ``CheckSetup`` (BOS + FVG within 10 HTF bars of
the reclaim) but **never calls it** from the main block — it is dead code in
this version. This translation mirrors that: ``CheckSetup`` is implemented
so the parameters / detection would match if it were activated, but is not
invoked from :func:`compute_htf_mss_liquidity`.

Architectural notes specific to the Python port (agreed with the user)
----------------------------------------------------------------------
* Caller supplies a *single* LTF DataFrame plus an ``htf`` string (e.g.
  ``"1W"``). The HTF series is resampled internally with
  ``label="left", closed="left"`` so each HTF bar's value is visible on the
  first LTF bar of that period — this mirrors Pine's ``lookahead = on``.
* Claim / reclaim are evaluated on **LTF** closes so that HTF liquidity can
  be taken mid-period exactly as in Pine.
* The Pine setting ``level`` (Short / Intermediate / Long Term) is kept as
  a parameter. The pipeline adapter passes ``_claim_all_tiers=True`` to
  track claim / reclaim for *every* pivot in one pass and filters per-tier
  downstream — lossless and avoids three redundant passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Settings + data types
# ---------------------------------------------------------------------------
_LEVEL_PREFIX = {
    "Short Term": "ST",
    "Intermediate Term": "IT",
    "Long Term": "LT",
}


@dataclass
class FadiSettings:
    """Mirror of the Pine inputs that affect detection."""

    htf: str = "15"  # Pine numeric string = minutes; we also accept "1W", "1M"
    level: str = "Short Term"  # "Short Term" | "Intermediate Term" | "Long Term"
    extend: int = 10  # kept for API parity; no longer cosmetic-only if future use
    max_lines: int = 50  # cap on "active" lines kept for the selected tier


@dataclass
class Pivot:
    """One HTF pivot. Matches Pine's ``type Pivot`` with cosmetic fields dropped."""

    time: pd.Timestamp
    price: float
    is_high: bool
    is_low: bool
    lbl_text: str  # current tier tag: "STH"/"STL"/"ITH"/"ITL"/"LTH"/"LTL"
    tier: str = "ST"  # max-achieved tier: "ST" | "IT" | "LT"
    claimed: bool = False
    reclaimed: bool = False
    claim_time: Optional[pd.Timestamp] = None
    reclaim_time: Optional[pd.Timestamp] = None


@dataclass
class FadiResult:
    pivots: List[Pivot] = field(default_factory=list)
    htf_index: Optional[pd.DatetimeIndex] = None  # for debugging / verification
    level: str = "Short Term"  # echoed back


# ---------------------------------------------------------------------------
# HTF resampling (shared convention with MTF FVG x2)
# ---------------------------------------------------------------------------
def _tf_to_rule(tf: str) -> str:
    """Translate Pine-style ``tf`` to a pandas resample rule.

    Accepts:
        * pure digits (Pine treats this as minutes, e.g. ``"15"`` → 15min)
        * trailing unit tokens ``H``, ``D``, ``W``, ``M``
        * the convenience forms ``1D``, ``1W``, ``1M`` already in use by the
          project.
    """
    tf = tf.strip()
    if tf.isdigit():
        return f"{int(tf)}min"
    tf_up = tf.upper()
    if tf_up.endswith("MIN"):
        return tf_up.lower()
    if tf_up.endswith("H"):
        return tf_up.replace("H", "h")
    if tf_up in ("1D", "D"):
        return "1D"
    if tf_up.endswith("D"):
        return tf_up
    if tf_up in ("W", "1W"):
        return "W"
    if tf_up in ("M", "1M"):
        return "MS"
    return tf_up


def _resample_htf(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Left-labelled OHLC resample (matches Pine's ``lookahead_on``)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if rule in ("1D", "MS"):
        return df.resample(rule).agg(agg).dropna()
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


# ---------------------------------------------------------------------------
# Pine helpers — SkipEQHigh / SkipEQLow / SkipEQPivot (translated verbatim)
# ---------------------------------------------------------------------------
def _skip_eq_highs(highs: List[float], idx: int) -> int:
    """Pine: while highs[i] == highs[i-1] and i < 5 → i++.

    The return value is clamped to ``len(highs) - 1`` so callers never index
    out of range when the window is smaller than Pine's 6-element array
    (e.g. early in the series). This matches Pine's NA-on-missing-history
    behaviour: an out-of-range index would be NA in Pine, and NA comparisons
    return False there — equivalent to stopping SkipEQ at the end of the
    available window.
    """
    i = idx
    while i < 5 and i - 1 >= 0 and i < len(highs) and highs[i] == highs[i - 1]:
        i += 1
    return min(i, max(0, len(highs) - 1))


def _skip_eq_lows(lows: List[float], idx: int) -> int:
    i = idx
    while i < 5 and i - 1 >= 0 and i < len(lows) and lows[i] == lows[i - 1]:
        i += 1
    return min(i, max(0, len(lows) - 1))


def _skip_eq_pivot(pivots: List[Pivot], idx: int) -> int:
    """Pine's buggy ``p.size() < i`` condition is preserved verbatim.

    The operands are swapped vs. what a normal bounds check would be; under
    realistic array sizes the loop exits on the first iteration, so this is
    effectively a no-op. We keep it for exact parity with the Pine source.
    """
    i = idx
    while (
        i - 1 >= 0
        and i < len(pivots)
        and pivots[i].price == pivots[i - 1].price
        and pivots[i].lbl_text == pivots[i - 1].lbl_text
        and len(pivots) < i  # <-- Pine's ``p.size() < i`` — intentionally odd
    ):
        i += 1
    return i


# ---------------------------------------------------------------------------
# Internal state — one ``MarketStructure`` per run
# ---------------------------------------------------------------------------
@dataclass
class _Term:
    ST: List[Pivot] = field(default_factory=list)
    IT: List[Pivot] = field(default_factory=list)
    LT: List[Pivot] = field(default_factory=list)
    STH: List[Pivot] = field(default_factory=list)
    ITH: List[Pivot] = field(default_factory=list)
    STL: List[Pivot] = field(default_factory=list)
    ITL: List[Pivot] = field(default_factory=list)


def _add_liquidity_noop(term: _Term, pivot: Pivot, level: str, max_lines: int) -> None:
    """Mirror of Pine's ``AddLiquidity``.

    Pine's implementation manages a line object for the selected tier only.
    We don't draw; instead we *mark* which pivots would have a line (Pine
    later uses this as the filter inside ``CheckClaimed``). The ``tier``
    field on each pivot already gives us this information, so this function
    is intentionally minimal — kept for call-site parity with the Pine source.
    """
    # The Pine body also enforces a `max_lines` cap by popping the oldest
    # line via `lines.pop()`. For claim-tracking fidelity we don't need to
    # forget pivots; we keep the cap available as a post-filter in the
    # adapter if needed.
    del term, pivot, level, max_lines  # intentional no-op


def _find_it(term: _Term, level: str, max_lines: int) -> None:
    """Pine's ``FindIT`` — promote STH / STL middle-of-3 to ITH / ITL."""
    if len(term.STH) > 2:
        h1 = term.STH[0]
        h2 = term.STH[1]
        h3 = term.STH[_skip_eq_pivot(term.STH, 2)]
        if h2.price > h3.price and h2.price > h1.price and h2.lbl_text == "STH":
            h2.lbl_text = "ITH"
            h2.tier = "IT"
            term.ITH.insert(0, h2)
            term.IT.insert(0, h2)
            _add_liquidity_noop(term, h2, level, max_lines)

    if len(term.STL) > 2:
        l1 = term.STL[0]
        l2 = term.STL[1]
        l3 = term.STL[_skip_eq_pivot(term.STL, 2)]
        if l2.price < l3.price and l2.price < l1.price and l2.lbl_text == "STL":
            l2.lbl_text = "ITL"
            l2.tier = "IT"
            term.ITL.insert(0, l2)
            term.IT.insert(0, l2)
            _add_liquidity_noop(term, l2, level, max_lines)


def _find_lt(term: _Term, level: str, max_lines: int) -> None:
    """Pine's ``FindLT`` — promote ITH / ITL middle-of-3 to LTH / LTL."""
    if len(term.ITH) > 2:
        h1 = term.ITH[0]
        h2 = term.ITH[1]
        h3 = term.ITH[_skip_eq_pivot(term.ITH, 2)]
        if h2.price > h3.price and h2.price > h1.price and h2.lbl_text == "ITH":
            h2.lbl_text = "LTH"
            h2.tier = "LT"
            term.LT.insert(0, h2)
            _add_liquidity_noop(term, h2, level, max_lines)

    if len(term.ITL) > 2:
        l1 = term.ITL[0]
        l2 = term.ITL[1]
        l3 = term.ITL[_skip_eq_pivot(term.ITL, 2)]
        if l2.price < l3.price and l2.price < l1.price and l2.lbl_text == "ITL":
            l2.lbl_text = "LTL"
            l2.tier = "LT"
            term.LT.insert(0, l2)
            _add_liquidity_noop(term, l2, level, max_lines)


def _add_pivot(
    term: _Term,
    price: float,
    time: pd.Timestamp,
    is_high: bool,
    is_low: bool,
    lbl: str,
    level: str,
    max_lines: int,
    max_buffer: int,
) -> None:
    """Pine's ``Add`` — push new ST pivot and run FindIT/FindLT promotions."""
    # ``isNew`` guard: only add if newer than the most-recent ST pivot.
    is_new = True
    if term.ST:
        prev = term.ST[0]
        is_new = prev.time < time
    if not is_new:
        return

    pivot = Pivot(
        time=time,
        price=price,
        is_high=is_high,
        is_low=is_low,
        lbl_text=lbl,
        tier="ST",
    )
    term.ST.insert(0, pivot)
    if is_high:
        term.STH.insert(0, pivot)
    else:
        term.STL.insert(0, pivot)

    _add_liquidity_noop(term, pivot, level, max_lines)
    _find_it(term, level, max_lines)
    _find_lt(term, level, max_lines)

    # Pine: ``if MS.ST.size() > MAX_BUFFER: pop + delete line``. We keep it
    # for parity; dropping the oldest pivot also keeps claim-tracking bounded.
    if len(term.ST) > max_buffer:
        term.ST.pop()


def _find_st(
    term: _Term,
    highs_window: List[float],
    lows_window: List[float],
    times_window: List[pd.Timestamp],
    level: str,
    max_lines: int,
    max_buffer: int,
) -> None:
    """Pine's ``findST`` — 3-HTF-bar fractal on the window built this HTF bar."""
    if len(highs_window) < 3:
        return
    _h = (
        highs_window[1] > highs_window[_skip_eq_highs(highs_window, 2)]
        and highs_window[1] > highs_window[0]
    )
    _l = (
        lows_window[1] < lows_window[_skip_eq_lows(lows_window, 2)]
        and lows_window[1] < lows_window[0]
    )

    if _h:
        _add_pivot(
            term,
            price=highs_window[1],
            time=times_window[1],
            is_high=True,
            is_low=False,
            lbl="STH",
            level=level,
            max_lines=max_lines,
            max_buffer=max_buffer,
        )
    if _l:
        _add_pivot(
            term,
            price=lows_window[1],
            time=times_window[1],
            is_high=False,
            is_low=True,
            lbl="STL",
            level=level,
            max_lines=max_lines,
            max_buffer=max_buffer,
        )


def _pivot_matches_level(pivot: Pivot, level: str) -> bool:
    """Pine: ``if not na(pivot.ln)`` — lines only exist on the selected tier."""
    prefix = _LEVEL_PREFIX.get(level, "ST")
    return pivot.lbl_text.startswith(prefix)


def _check_claimed(
    term: _Term,
    close_i: float,
    time_i: pd.Timestamp,
    level: str,
    claim_all_tiers: bool,
) -> None:
    """Pine's ``CheckClaimed`` — evaluate claim / reclaim on every LTF bar."""
    for pivot in term.ST:
        if not claim_all_tiers and not _pivot_matches_level(pivot, level):
            continue

        if not pivot.claimed:
            if pivot.is_high and close_i > pivot.price:
                pivot.claimed = True
                pivot.claim_time = time_i
            elif pivot.is_low and close_i < pivot.price:
                pivot.claimed = True
                pivot.claim_time = time_i

        if pivot.claimed and not pivot.reclaimed:
            triggered = False
            if pivot.is_high and close_i < pivot.price:
                pivot.reclaimed = True
                triggered = True
            elif pivot.is_low and close_i > pivot.price:
                pivot.reclaimed = True
                triggered = True

            if triggered:
                pivot.reclaim_time = time_i
                # Pine cascade: mark every older same-side claimed-but-not-
                # reclaimed pivot as reclaimed (no timestamps on the cascade).
                for p in term.ST:
                    if p is pivot:
                        continue
                    same_side = (pivot.is_high and p.is_high) or (
                        pivot.is_low and p.is_low
                    )
                    if same_side and p.claimed and not p.reclaimed:
                        p.reclaimed = True


# ---------------------------------------------------------------------------
# Optional: CheckSetup — translated for completeness; NOT invoked (dead code
# in the Pine source). Left here in case the user wants to enable it later.
# ---------------------------------------------------------------------------
@dataclass
class _Setup:
    pivot: Optional[Pivot] = None
    has_bos: bool = False
    has_fvg: bool = False
    bos_level: float = 0.0
    bos_time: Optional[pd.Timestamp] = None
    fvg_open: float = 0.0
    fvg_close: float = 0.0
    fvg_time: Optional[pd.Timestamp] = None
    active: bool = False


def _check_setup(setup: _Setup, df: pd.DataFrame, current_idx: int) -> None:
    """Faithful translation of Pine's ``CheckSetup``.

    Not called from :func:`compute_htf_mss_liquidity`; see module docstring.
    """
    if setup.active or setup.pivot is None:
        return
    pivot = setup.pivot
    reclaim_idx = df.index.get_indexer([pivot.reclaim_time or pivot.time])[0]
    if reclaim_idx < 0:
        return
    if not setup.has_bos:
        lookback = 10
        for offset in range(0, lookback + 1):
            j = reclaim_idx + offset
            if j + 2 >= len(df):
                break
            if pivot.is_high:
                lj1 = float(df["low"].iat[j + 1])
                lj = float(df["low"].iat[j])
                lj2 = float(df["low"].iat[j + 2])
                if lj1 < lj and lj1 < lj2 and lj1 < pivot.price:
                    setup.has_bos = True
                    setup.bos_level = lj1
                    setup.bos_time = df.index[j + 1]
                    break
            else:
                hj1 = float(df["high"].iat[j + 1])
                hj = float(df["high"].iat[j])
                hj2 = float(df["high"].iat[j + 2])
                if hj1 > hj and hj1 > hj2 and hj1 > pivot.price:
                    setup.has_bos = True
                    setup.bos_level = hj1
                    setup.bos_time = df.index[j + 1]
                    break
    if not setup.has_fvg and current_idx >= 2:
        if pivot.is_high:
            if float(df["high"].iat[current_idx]) < float(df["low"].iat[current_idx - 2]):
                setup.has_fvg = True
                setup.fvg_open = float(df["low"].iat[current_idx - 2])
                setup.fvg_close = float(df["high"].iat[current_idx])
                setup.fvg_time = df.index[current_idx - 2]
        else:
            if float(df["low"].iat[current_idx]) > float(df["high"].iat[current_idx - 2]):
                setup.has_fvg = True
                setup.fvg_open = float(df["high"].iat[current_idx - 2])
                setup.fvg_close = float(df["low"].iat[current_idx])
                setup.fvg_time = df.index[current_idx - 2]
    if setup.has_bos and setup.has_fvg:
        close_i = float(df["close"].iat[current_idx])
        setup.active = (
            (close_i < setup.bos_level)
            if pivot.is_high
            else (close_i > setup.bos_level)
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
_MAX_BUFFER = 500  # Pine's ``MAX_BUFFER`` literal


def compute_htf_mss_liquidity(
    df: pd.DataFrame,
    settings: Optional[FadiSettings] = None,
    *,
    _claim_all_tiers: bool = False,
) -> FadiResult:
    """Run the fadi HTF MSS & Liquidity detector on an LTF OHLC DataFrame.

    Parameters
    ----------
    df
        LTF bars (DatetimeIndex, ``high``, ``low``, ``close`` columns).
    settings
        :class:`FadiSettings` instance; defaults mirror the Pine inputs.
    _claim_all_tiers
        *Internal.* When True, claim / reclaim is tracked for every pivot
        regardless of ``settings.level`` — used by the adapter to emit all
        three tiers in one pass. The Pine-faithful default is False.

    Raises
    ------
    ValueError
        When LTF is not strictly finer than HTF (mirrors Pine's
        ``if helper.Validtimeframe(HTF)`` guard).
    """
    s = settings or FadiSettings()
    if s.level not in _LEVEL_PREFIX:
        raise ValueError(f"level must be one of {sorted(_LEVEL_PREFIX)}")
    if len(df) < 2:
        return FadiResult(level=s.level)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    rule = _tf_to_rule(s.htf)
    htf_df = _resample_htf(df, rule)
    if len(htf_df) < 4:  # we need >= 3 completed HTF bars + 1 current
        return FadiResult(level=s.level, htf_index=htf_df.index)

    # Validtimeframe: the LTF bar cadence must be strictly finer than HTF.
    ltf_sec = _estimate_bar_seconds(df.index)
    htf_sec = _estimate_bar_seconds(htf_df.index)
    if ltf_sec is not None and htf_sec is not None and ltf_sec >= htf_sec:
        raise ValueError(
            f"LTF (~{ltf_sec}s) is not strictly finer than HTF (~{htf_sec}s); "
            "fadi's `Validtimeframe` guard rejects this."
        )

    term = _Term()

    # For each LTF bar, figure out which HTF bar it belongs to.
    htf_starts = htf_df.index.values.astype("datetime64[ns]")
    ltf_times = df.index.values.astype("datetime64[ns]")
    period_idx = np.searchsorted(htf_starts, ltf_times, side="right") - 1  # -1 if before first HTF

    prev_period = -2
    for i in range(len(df)):
        k = int(period_idx[i])
        if k >= 3:
            is_first_of_period = k != prev_period
            if is_first_of_period:
                # Build the 6-bar window of most-recently-completed HTF bars
                # (Pine: h = high[1]..h5 = high[6], highest index oldest).
                highs_w = [float(htf_df["high"].iat[k - j]) for j in range(1, 7) if k - j >= 0]
                lows_w = [float(htf_df["low"].iat[k - j]) for j in range(1, 7) if k - j >= 0]
                times_w = [htf_df.index[k - j] for j in range(1, 7) if k - j >= 0]
                _find_st(term, highs_w, lows_w, times_w, s.level, s.max_lines, _MAX_BUFFER)
            prev_period = k

        close_i = float(df["close"].iat[i])
        t_i = df.index[i]
        _check_claimed(term, close_i, t_i, s.level, _claim_all_tiers)

    # Collect pivots — preserve chronological order for downstream consumers.
    all_pivots = sorted(term.ST, key=lambda p: p.time)
    return FadiResult(pivots=all_pivots, htf_index=htf_df.index, level=s.level)


def _estimate_bar_seconds(index: pd.DatetimeIndex) -> Optional[int]:
    if len(index) < 2:
        return None
    deltas = np.diff(index.values.astype("datetime64[s]").astype(np.int64))
    if len(deltas) == 0:
        return None
    # Use the median to be robust to holidays / weekend gaps.
    return int(np.median(deltas))


__all__ = [
    "FadiSettings",
    "Pivot",
    "FadiResult",
    "compute_htf_mss_liquidity",
]
