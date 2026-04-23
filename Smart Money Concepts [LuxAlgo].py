"""Python translation of LuxAlgo's "Smart Money Concepts [LuxAlgo]" Pine v5
indicator.

Mirrors the DETECTION logic for the current-timeframe SMC primitives:

    - Swing market structure (BoS / CHoCH) at a configurable pivot length.
    - Internal market structure at a fixed pivot length of 5.
    - Swing and internal order blocks (stored on structure breaks).
    - Order block mitigation / invalidation (Close or High/Low source).
    - Fair Value Gaps (current-timeframe only; HTF FVG via request.security is
      intentionally out of scope).
    - Equal Highs / Equal Lows (EQH / EQL).
    - Trailing swing extremes + premium / equilibrium / discount zones.

Drawing-layer constructs in the Pine source (box.new, line.new, label.new,
plotcandle, alertcondition) have no Python counterpart and are NOT ported;
what matters for analysis is the *detected events*, which this module
surfaces via an ``SMCResult`` dataclass.

Pine -> Python mapping cheatsheet:

    var             -> persistent state on the analyzer object / closure
    array.push      -> list.append
    array.unshift   -> list.insert(0, value)   # prepend
    array.pop       -> list.pop()              # removes LAST
    array.shift     -> list.pop(0)             # removes FIRST
    ta.highest(N)   -> max(high[i-N+1 : i+1])  # includes current bar
    ta.lowest(N)    -> min(low[i-N+1  : i+1])
    ta.atr(N)       -> RMA of true range, length N (SMA-seeded)
    ta.cum(x)       -> cumulative sum of x (NaN treated as 0)
    ta.change(x)    -> x - x[1]  (fires "new" on first non-equal sample)
    ta.crossover    -> prev_a <= prev_b AND a > b
    ta.crossunder   -> prev_a >= prev_b AND a < b

The source-to-Python correspondence is annotated inline at each function
head so the translation can be audited against the Pine script line by line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Constants (mirror Pine #DEFINES)
# -----------------------------------------------------------------------------
BULLISH_LEG = 1
BEARISH_LEG = 0

BULLISH = +1
BEARISH = -1

# Option strings (kept verbatim for CLI / JSON interoperability).
ATR_FILTER = "Atr"
RANGE_FILTER = "Cumulative Mean Range"

MITIGATION_CLOSE = "Close"
MITIGATION_HIGHLOW = "High/Low"

MODE_HISTORICAL = "Historical"
MODE_PRESENT = "Present"

TAG_BOS = "BOS"
TAG_CHOCH = "CHoCH"


# -----------------------------------------------------------------------------
# Settings (1:1 with the input.* calls in the Pine source)
# -----------------------------------------------------------------------------
@dataclass
class SMCSettings:
    # ---- real-time structures ----
    show_internal_structure: bool = True
    internal_filter_confluence: bool = False
    show_swing_structure: bool = True
    swings_length: int = 50  # Pine minval = 10
    show_high_low_swings: bool = True

    # ---- order blocks ----
    show_internal_order_blocks: bool = True
    internal_order_blocks_size: int = 5  # Pine 1..20
    show_swing_order_blocks: bool = False
    swing_order_blocks_size: int = 5  # Pine 1..20
    order_block_filter: str = ATR_FILTER
    order_block_mitigation: str = MITIGATION_HIGHLOW

    # ---- equal highs / lows ----
    show_equal_highs_lows: bool = True
    equal_highs_lows_length: int = 3  # Pine minval = 1
    equal_highs_lows_threshold: float = 0.1  # Pine [0, 0.5]

    # ---- fair value gaps ----
    show_fair_value_gaps: bool = False
    fair_value_gaps_auto_threshold: bool = True
    fair_value_gaps_timeframe: str = ""  # "" = current TF (only supported mode)
    fair_value_gaps_extend: int = 1  # purely cosmetic; kept for parity

    # ---- premium / discount ----
    show_premium_discount_zones: bool = False

    # ---- misc (kept for parity; not used in detection) ----
    mode: str = MODE_HISTORICAL

    def __post_init__(self) -> None:
        self.swings_length = max(10, int(self.swings_length))
        self.internal_order_blocks_size = max(
            1, min(20, int(self.internal_order_blocks_size))
        )
        self.swing_order_blocks_size = max(
            1, min(20, int(self.swing_order_blocks_size))
        )
        self.equal_highs_lows_length = max(1, int(self.equal_highs_lows_length))
        self.equal_highs_lows_threshold = max(
            0.0, min(0.5, float(self.equal_highs_lows_threshold))
        )
        if self.order_block_filter not in (ATR_FILTER, RANGE_FILTER):
            self.order_block_filter = ATR_FILTER
        if self.order_block_mitigation not in (MITIGATION_CLOSE, MITIGATION_HIGHLOW):
            self.order_block_mitigation = MITIGATION_HIGHLOW
        if self.mode not in (MODE_HISTORICAL, MODE_PRESENT):
            self.mode = MODE_HISTORICAL


# -----------------------------------------------------------------------------
# Stateful types (mirror the Pine UDTs)
# -----------------------------------------------------------------------------
@dataclass
class Pivot:
    """Mirrors Pine 'type pivot' — a swing/internal/equal pivot point."""

    current_level: float = float("nan")
    last_level: float = float("nan")
    crossed: bool = False
    bar_time: Optional[pd.Timestamp] = None
    bar_index: int = -1


@dataclass
class TrailingExtremes:
    """Mirrors Pine 'type trailingExtremes'."""

    top: float = float("nan")
    bottom: float = float("nan")
    bar_time: Optional[pd.Timestamp] = None
    bar_index: int = -1
    last_top_time: Optional[pd.Timestamp] = None
    last_bottom_time: Optional[pd.Timestamp] = None


@dataclass
class OrderBlock:
    """Mirrors Pine 'type orderBlock' + mitigation tracking."""

    bar_high: float
    bar_low: float
    bar_time: pd.Timestamp
    bar_index: int
    bias: int  # +1 BULLISH or -1 BEARISH
    scope: str = "internal"  # "internal" or "swing"
    # Pine removes mitigated OBs from the active list. We remove them too,
    # but surface a copy to the consumer via SMCResult.mitigated_*_order_blocks
    # so the analyst still has an audit trail.
    mitigated: bool = False
    mitigated_time: Optional[pd.Timestamp] = None


@dataclass
class FairValueGap:
    """Mirrors Pine 'type fairValueGap' (minus the drawing box fields)."""

    top: float
    bottom: float
    bias: int
    created_time: pd.Timestamp
    created_index: int
    filled: bool = False
    filled_time: Optional[pd.Timestamp] = None


@dataclass
class StructureEvent:
    """A BoS or CHoCH event emitted by displayStructure()."""

    bar_time: pd.Timestamp
    bar_index: int
    kind: str  # "BOS" or "CHoCH"
    direction: int  # +1 BULLISH / -1 BEARISH
    scope: str  # "swing" or "internal"
    pivot_level: float
    pivot_time: Optional[pd.Timestamp] = None


@dataclass
class EqualEvent:
    """An EQH / EQL event emitted when a pivot matches the previous level."""

    first_time: pd.Timestamp
    second_time: pd.Timestamp
    level: float
    kind: str  # "EQH" or "EQL"


@dataclass
class SMCResult:
    bars_analyzed: int
    # --- detected events ---
    structure_events: List[StructureEvent] = field(default_factory=list)
    equal_events: List[EqualEvent] = field(default_factory=list)
    fair_value_gaps: List[FairValueGap] = field(default_factory=list)
    # --- order blocks ---
    swing_order_blocks: List[OrderBlock] = field(default_factory=list)
    internal_order_blocks: List[OrderBlock] = field(default_factory=list)
    mitigated_swing_order_blocks: List[OrderBlock] = field(default_factory=list)
    mitigated_internal_order_blocks: List[OrderBlock] = field(default_factory=list)
    # --- running state (as of the final bar) ---
    swing_trend: int = 0  # +1 / -1 / 0 (unconfirmed)
    internal_trend: int = 0
    last_swing_high: Pivot = field(default_factory=Pivot)
    last_swing_low: Pivot = field(default_factory=Pivot)
    last_internal_high: Pivot = field(default_factory=Pivot)
    last_internal_low: Pivot = field(default_factory=Pivot)
    trailing_top: float = float("nan")
    trailing_bottom: float = float("nan")
    trailing_last_top_time: Optional[pd.Timestamp] = None
    trailing_last_bottom_time: Optional[pd.Timestamp] = None
    # --- premium / equilibrium / discount zones (derived from trailing) ---
    premium_discount: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Numeric helpers (ta.atr, ta.tr, ta.rma, ta.cum)
# -----------------------------------------------------------------------------
def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Pine ta.tr(true): first bar TR is high - low (handles NaN prev close)."""
    n = len(high)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    if n < 2:
        return tr
    prev_close = close[:-1]
    tr[1:] = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - prev_close),
            np.abs(low[1:] - prev_close),
        ]
    )
    return tr


def _rma(series: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.rma: SMA-seeded Wilder's smoothing."""
    n = len(series)
    out = np.full(n, np.nan)
    if n < length:
        return out
    out[length - 1] = float(np.mean(series[:length]))
    for i in range(length, n):
        out[i] = (out[i - 1] * (length - 1) + series[i]) / length
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    return _rma(_true_range(high, low, close), length)


# -----------------------------------------------------------------------------
# Leg detection (Pine: leg(size))
# -----------------------------------------------------------------------------
def _leg_for_bar(
    high: np.ndarray,
    low: np.ndarray,
    size: int,
    i: int,
    prev_leg: int,
) -> int:
    """Pine equivalent of ``leg(size)`` evaluated at bar ``i``.

        newLegHigh = high[size] > ta.highest(size)
        newLegLow  = low[size]  < ta.lowest(size)

    ``ta.highest(size)`` is the max over the last ``size`` bars INCLUDING
    the current bar; ``high[size]`` is the bar ``size`` steps back.

    Returns the new leg value, or ``prev_leg`` if unchanged.
    """
    if i < size:
        return prev_leg
    # high[i - size + 1 : i + 1] spans size bars up to and including bar i.
    highest = np.max(high[i - size + 1 : i + 1])
    lowest = np.min(low[i - size + 1 : i + 1])
    candidate_high = high[i - size]
    candidate_low = low[i - size]
    if candidate_high > highest:
        return BEARISH_LEG
    if candidate_low < lowest:
        return BULLISH_LEG
    return prev_leg


# -----------------------------------------------------------------------------
# Order block helpers (Pine: storeOrdeBlock, deleteOrderBlocks)
# -----------------------------------------------------------------------------
def _store_order_block(
    obs: List[OrderBlock],
    pivot: Pivot,
    bias: int,
    parsed_highs: List[float],
    parsed_lows: List[float],
    times: List[pd.Timestamp],
    current_bar_index: int,
    scope: str,
) -> None:
    """Port of Pine ``storeOrdeBlock``.

        if bias == BEARISH
            a_rray      := parsedHighs.slice(p_ivot.barIndex, bar_index)
            parsedIndex := p_ivot.barIndex + a_rray.indexof(a_rray.max())
        else
            a_rray      := parsedLows.slice(p_ivot.barIndex, bar_index)
            parsedIndex := p_ivot.barIndex + a_rray.indexof(a_rray.min())
        orderBlock := orderBlock.new(parsedHighs[parsedIndex],
                                     parsedLows[parsedIndex],
                                     times[parsedIndex], bias)
        if orderBlocks.size() >= 100
            orderBlocks.pop()         # removes LAST (oldest)
        orderBlocks.unshift(o_rderBlock)  # prepends (newest at index 0)

    Note: ``array.slice(from, to)`` is [from, to) and ``array.indexof`` returns
    the FIRST occurrence -> matches ``np.argmax`` / ``np.argmin``.
    """
    pivot_idx = pivot.bar_index
    if pivot_idx < 0 or pivot_idx >= current_bar_index:
        return
    slice_highs = parsed_highs[pivot_idx:current_bar_index]
    slice_lows = parsed_lows[pivot_idx:current_bar_index]
    if not slice_highs:
        return
    if bias == BEARISH:
        rel = int(np.argmax(np.asarray(slice_highs, dtype=float)))
    else:
        rel = int(np.argmin(np.asarray(slice_lows, dtype=float)))
    parsed_index = pivot_idx + rel
    ob = OrderBlock(
        bar_high=float(parsed_highs[parsed_index]),
        bar_low=float(parsed_lows[parsed_index]),
        bar_time=times[parsed_index],
        bar_index=parsed_index,
        bias=bias,
        scope=scope,
    )
    if len(obs) >= 100:
        obs.pop()  # remove oldest (Pine array.pop removes LAST element)
    obs.insert(0, ob)  # newest goes to the front (Pine array.unshift)


def _delete_mitigated_obs(
    obs: List[OrderBlock],
    mitigated_archive: List[OrderBlock],
    close: float,
    high: float,
    low: float,
    mitigation: str,
    current_time: pd.Timestamp,
) -> None:
    """Port of Pine ``deleteOrderBlocks``.

        bearishOrderBlockMitigationSource = mitigation == CLOSE ? close : high
        bullishOrderBlockMitigationSource = mitigation == CLOSE ? close : low

        for eachOrderBlock in orderBlocks
            if bearishSource > eachOrderBlock.barHigh and bias == BEARISH:
                remove
            else if bullishSource < eachOrderBlock.barLow and bias == BULLISH:
                remove

    We remove from the active list (Pine semantics) and archive the victim so
    the analyst can audit mitigations after the fact.
    """
    bearish_src = close if mitigation == MITIGATION_CLOSE else high
    bullish_src = close if mitigation == MITIGATION_CLOSE else low
    survivors: List[OrderBlock] = []
    for ob in obs:
        if ob.bias == BEARISH and bearish_src > ob.bar_high:
            ob.mitigated = True
            ob.mitigated_time = current_time
            mitigated_archive.append(ob)
            continue
        if ob.bias == BULLISH and bullish_src < ob.bar_low:
            ob.mitigated = True
            ob.mitigated_time = current_time
            mitigated_archive.append(ob)
            continue
        survivors.append(ob)
    obs[:] = survivors


# -----------------------------------------------------------------------------
# Main analyzer (Pine execution flow, linearised into a bar-by-bar loop)
# -----------------------------------------------------------------------------
def analyze_smart_money(df: pd.DataFrame, settings: Optional[SMCSettings] = None) -> SMCResult:
    """Run the full Smart Money Concepts [LuxAlgo] detection over an OHLC
    DataFrame and return the accumulated state / events.

    The DataFrame MUST have a monotonic timestamp index and the columns
    ``open``, ``high``, ``low``, ``close``. ``volume`` is accepted but
    unused. Column casing and order don't matter as long as the names match.
    """
    settings = settings or SMCSettings()
    if df.empty:
        return SMCResult(bars_analyzed=0)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    open_ = df["open"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    times: List[pd.Timestamp] = list(df.index)
    n = len(df)

    # -------------------------------------------------------------------------
    # Volatility measure -> parsed highs/lows.
    # -------------------------------------------------------------------------
    atr200 = _atr(high, low, close, 200)
    if settings.order_block_filter == ATR_FILTER:
        volatility = atr200
    else:
        tr = _true_range(high, low, close)
        cum_tr = np.cumsum(tr)
        # Pine uses bar_index (0-indexed) as the divisor -> divide-by-zero on
        # bar 0. We use (i + 1) to keep the series finite and well-defined;
        # the first-bar value is just TR[0] either way, so the behaviour
        # matches after bar 1.
        divisor = np.arange(1, n + 1, dtype=float)
        volatility = cum_tr / divisor

    # NaN in `volatility` (the early ATR warm-up) yields False on ``>=``,
    # which matches Pine's behaviour: high_volatility stays False and
    # parsed_high / parsed_low collapse to the raw high / low.
    with np.errstate(invalid="ignore"):
        high_volatility = (high - low) >= (2.0 * np.nan_to_num(volatility, nan=np.inf))
    parsed_high_series = np.where(high_volatility, low, high)
    parsed_low_series = np.where(high_volatility, high, low)

    # -------------------------------------------------------------------------
    # FVG threshold: ta.cum(|barDeltaPercent|) / bar_index * 2
    # -------------------------------------------------------------------------
    bar_delta = np.zeros(n)
    if n >= 2:
        prev_open = open_[:-1]
        prev_close_arr = close[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = (prev_close_arr - prev_open) / (prev_open * 100.0)
        delta = np.where(np.isfinite(delta), delta, 0.0)
        bar_delta[1:] = delta
    cum_abs_delta = np.cumsum(np.abs(bar_delta))

    # -------------------------------------------------------------------------
    # State (Pine var declarations)
    # -------------------------------------------------------------------------
    swing_high = Pivot()
    swing_low = Pivot()
    internal_high = Pivot()
    internal_low = Pivot()
    equal_high = Pivot()
    equal_low = Pivot()

    swing_trend = 0
    internal_trend = 0
    trailing = TrailingExtremes()

    swing_obs: List[OrderBlock] = []
    internal_obs: List[OrderBlock] = []
    mitigated_swing_obs: List[OrderBlock] = []
    mitigated_internal_obs: List[OrderBlock] = []

    fvgs: List[FairValueGap] = []
    all_fvgs: List[FairValueGap] = []  # archive, including filled ones

    structure_events: List[StructureEvent] = []
    equal_events: List[EqualEvent] = []

    parsed_highs: List[float] = []
    parsed_lows: List[float] = []

    # leg state (var int leg) — separate instance per size
    swing_leg = 0
    internal_leg = 0
    eqhl_leg = 0

    # For crossover/crossunder: prev_close from the preceding bar.
    prev_close = float("nan")

    # -------------------------------------------------------------------------
    # Main bar-by-bar loop
    # -------------------------------------------------------------------------
    for i in range(n):
        t = times[i]
        c_i = close[i]
        h_i = high[i]
        l_i = low[i]
        o_i = open_[i]

        # Snapshot pivot levels BEFORE getCurrentStructure runs so that the
        # crossover check below can use ``level[1]`` semantics (value at end of
        # previous bar, same as Pine).
        snap_swing_high_level = swing_high.current_level
        snap_swing_low_level = swing_low.current_level
        snap_internal_high_level = internal_high.current_level
        snap_internal_low_level = internal_low.current_level
        snap_prev_close = prev_close if i > 0 else c_i

        # ---- parsedHighs.push / parsedLows.push ----
        parsed_highs.append(float(parsed_high_series[i]))
        parsed_lows.append(float(parsed_low_series[i]))

        # ---- updateTrailingExtremes() ----
        #   trailing.top = math.max(high, trailing.top)
        #   trailing.bottom = math.min(low, trailing.bottom)
        if np.isnan(trailing.top) or h_i > trailing.top:
            trailing.top = h_i
            trailing.last_top_time = t
        if np.isnan(trailing.bottom) or l_i < trailing.bottom:
            trailing.bottom = l_i
            trailing.last_bottom_time = t

        # ---- deleteFairValueGaps() (invalidates existing FVGs) ----
        #   if low < fvg.bottom and bias == BULLISH: delete
        #   if high > fvg.top and bias == BEARISH:    delete
        if fvgs:
            survivors: List[FairValueGap] = []
            for fvg in fvgs:
                if fvg.bias == BULLISH and l_i < fvg.bottom:
                    fvg.filled = True
                    fvg.filled_time = t
                    continue
                if fvg.bias == BEARISH and h_i > fvg.top:
                    fvg.filled = True
                    fvg.filled_time = t
                    continue
                survivors.append(fvg)
            fvgs = survivors

        # ---- getCurrentStructure(swingsLength, equalHighLow=False, internal=False) ----
        prev_swing_leg = swing_leg
        swing_leg = _leg_for_bar(high, low, settings.swings_length, i, prev_swing_leg)
        if swing_leg != prev_swing_leg:
            size = settings.swings_length
            if swing_leg == BULLISH_LEG:
                # new bullish leg -> swing LOW formed at bar i - size
                swing_low.last_level = swing_low.current_level
                swing_low.current_level = float(low[i - size])
                swing_low.crossed = False
                swing_low.bar_time = times[i - size]
                swing_low.bar_index = i - size
                # Pine: trailing snap to confirmed swing low
                trailing.bottom = swing_low.current_level
                trailing.bar_time = swing_low.bar_time
                trailing.bar_index = swing_low.bar_index
                trailing.last_bottom_time = swing_low.bar_time
            else:  # BEARISH_LEG -> swing HIGH formed
                swing_high.last_level = swing_high.current_level
                swing_high.current_level = float(high[i - size])
                swing_high.crossed = False
                swing_high.bar_time = times[i - size]
                swing_high.bar_index = i - size
                trailing.top = swing_high.current_level
                trailing.bar_time = swing_high.bar_time
                trailing.bar_index = swing_high.bar_index
                trailing.last_top_time = swing_high.bar_time

        # ---- getCurrentStructure(5, equalHighLow=False, internal=True) ----
        prev_internal_leg = internal_leg
        internal_leg = _leg_for_bar(high, low, 5, i, prev_internal_leg)
        if internal_leg != prev_internal_leg:
            size = 5
            if internal_leg == BULLISH_LEG:
                internal_low.last_level = internal_low.current_level
                internal_low.current_level = float(low[i - size])
                internal_low.crossed = False
                internal_low.bar_time = times[i - size]
                internal_low.bar_index = i - size
            else:
                internal_high.last_level = internal_high.current_level
                internal_high.current_level = float(high[i - size])
                internal_high.crossed = False
                internal_high.bar_time = times[i - size]
                internal_high.bar_index = i - size

        # ---- getCurrentStructure(equalHighsLowsLength, equalHighLow=True) ----
        if settings.show_equal_highs_lows:
            prev_eqhl_leg = eqhl_leg
            eqhl_leg = _leg_for_bar(high, low, settings.equal_highs_lows_length, i, prev_eqhl_leg)
            if eqhl_leg != prev_eqhl_leg:
                size = settings.equal_highs_lows_length
                atr_i = atr200[i] if not np.isnan(atr200[i]) else 0.0
                thresh = settings.equal_highs_lows_threshold * atr_i
                if eqhl_leg == BULLISH_LEG:  # new low candidate
                    candidate = float(low[i - size])
                    if (
                        not np.isnan(equal_low.current_level)
                        and abs(equal_low.current_level - candidate) < thresh
                    ):
                        equal_events.append(
                            EqualEvent(
                                first_time=equal_low.bar_time,
                                second_time=times[i - size],
                                level=candidate,
                                kind="EQL",
                            )
                        )
                    equal_low.last_level = equal_low.current_level
                    equal_low.current_level = candidate
                    equal_low.crossed = False
                    equal_low.bar_time = times[i - size]
                    equal_low.bar_index = i - size
                else:  # BEARISH_LEG -> new high candidate
                    candidate = float(high[i - size])
                    if (
                        not np.isnan(equal_high.current_level)
                        and abs(equal_high.current_level - candidate) < thresh
                    ):
                        equal_events.append(
                            EqualEvent(
                                first_time=equal_high.bar_time,
                                second_time=times[i - size],
                                level=candidate,
                                kind="EQH",
                            )
                        )
                    equal_high.last_level = equal_high.current_level
                    equal_high.current_level = candidate
                    equal_high.crossed = False
                    equal_high.bar_time = times[i - size]
                    equal_high.bar_index = i - size

        # ---- displayStructure(internal=True) ----
        if settings.internal_filter_confluence:
            hi_minus_body = h_i - max(c_i, o_i)
            body_minus_lo = min(c_i, o_i) - l_i
            bullish_bar = hi_minus_body > body_minus_lo
            bearish_bar = hi_minus_body < body_minus_lo
        else:
            bullish_bar = True
            bearish_bar = True

        # internal bullish crossover
        if (
            not np.isnan(internal_high.current_level)
            and not internal_high.crossed
            and (internal_high.current_level != swing_high.current_level)
            and bullish_bar
            and snap_prev_close <= snap_internal_high_level
            and c_i > internal_high.current_level
        ):
            tag = TAG_CHOCH if internal_trend == BEARISH else TAG_BOS
            structure_events.append(
                StructureEvent(
                    bar_time=t,
                    bar_index=i,
                    kind=tag,
                    direction=BULLISH,
                    scope="internal",
                    pivot_level=internal_high.current_level,
                    pivot_time=internal_high.bar_time,
                )
            )
            internal_high.crossed = True
            internal_trend = BULLISH
            if settings.show_internal_order_blocks:
                _store_order_block(
                    internal_obs,
                    internal_high,
                    BULLISH,
                    parsed_highs,
                    parsed_lows,
                    times,
                    i,
                    scope="internal",
                )

        # internal bearish crossunder
        if (
            not np.isnan(internal_low.current_level)
            and not internal_low.crossed
            and (internal_low.current_level != swing_low.current_level)
            and bearish_bar
            and snap_prev_close >= snap_internal_low_level
            and c_i < internal_low.current_level
        ):
            tag = TAG_CHOCH if internal_trend == BULLISH else TAG_BOS
            structure_events.append(
                StructureEvent(
                    bar_time=t,
                    bar_index=i,
                    kind=tag,
                    direction=BEARISH,
                    scope="internal",
                    pivot_level=internal_low.current_level,
                    pivot_time=internal_low.bar_time,
                )
            )
            internal_low.crossed = True
            internal_trend = BEARISH
            if settings.show_internal_order_blocks:
                _store_order_block(
                    internal_obs,
                    internal_low,
                    BEARISH,
                    parsed_highs,
                    parsed_lows,
                    times,
                    i,
                    scope="internal",
                )

        # ---- displayStructure(internal=False) ----
        # swing bullish crossover (no extraCondition beyond the level check)
        if (
            not np.isnan(swing_high.current_level)
            and not swing_high.crossed
            and snap_prev_close <= snap_swing_high_level
            and c_i > swing_high.current_level
        ):
            tag = TAG_CHOCH if swing_trend == BEARISH else TAG_BOS
            structure_events.append(
                StructureEvent(
                    bar_time=t,
                    bar_index=i,
                    kind=tag,
                    direction=BULLISH,
                    scope="swing",
                    pivot_level=swing_high.current_level,
                    pivot_time=swing_high.bar_time,
                )
            )
            swing_high.crossed = True
            swing_trend = BULLISH
            if settings.show_swing_order_blocks:
                _store_order_block(
                    swing_obs,
                    swing_high,
                    BULLISH,
                    parsed_highs,
                    parsed_lows,
                    times,
                    i,
                    scope="swing",
                )

        # swing bearish crossunder
        if (
            not np.isnan(swing_low.current_level)
            and not swing_low.crossed
            and snap_prev_close >= snap_swing_low_level
            and c_i < swing_low.current_level
        ):
            tag = TAG_CHOCH if swing_trend == BULLISH else TAG_BOS
            structure_events.append(
                StructureEvent(
                    bar_time=t,
                    bar_index=i,
                    kind=tag,
                    direction=BEARISH,
                    scope="swing",
                    pivot_level=swing_low.current_level,
                    pivot_time=swing_low.bar_time,
                )
            )
            swing_low.crossed = True
            swing_trend = BEARISH
            if settings.show_swing_order_blocks:
                _store_order_block(
                    swing_obs,
                    swing_low,
                    BEARISH,
                    parsed_highs,
                    parsed_lows,
                    times,
                    i,
                    scope="swing",
                )

        # ---- deleteOrderBlocks() for both scopes ----
        if settings.show_internal_order_blocks and internal_obs:
            _delete_mitigated_obs(
                internal_obs,
                mitigated_internal_obs,
                c_i,
                h_i,
                l_i,
                settings.order_block_mitigation,
                t,
            )
        if settings.show_swing_order_blocks and swing_obs:
            _delete_mitigated_obs(
                swing_obs,
                mitigated_swing_obs,
                c_i,
                h_i,
                l_i,
                settings.order_block_mitigation,
                t,
            )

        # ---- drawFairValueGaps() (current-timeframe only) ----
        # Pine request.security(sym, "", [close[1], open[1], time[1], high[0],
        # low[0], time[0], high[2], low[2]], lookahead=barmerge.lookahead_on)
        # For tf="" this is equivalent to the raw bar series — no resampling.
        # bullishFVG = currentLow > last2High and lastClose > last2High
        #              and barDeltaPercent > threshold and newTimeframe
        # newTimeframe is true on every bar of the chart TF.
        if settings.show_fair_value_gaps and i >= 2:
            last_close = close[i - 1]
            last_open = open_[i - 1]
            if last_open != 0.0:
                bdp = (last_close - last_open) / (last_open * 100.0)
            else:
                bdp = 0.0
            if settings.fair_value_gaps_auto_threshold:
                # bar_index starts at 0; use (i + 1) to avoid /0 on bar 0.
                threshold = cum_abs_delta[i] / (i + 1) * 2.0
            else:
                threshold = 0.0

            high_i_minus_2 = high[i - 2]
            low_i_minus_2 = low[i - 2]

            if l_i > high_i_minus_2 and last_close > high_i_minus_2 and bdp > threshold:
                fvg = FairValueGap(
                    top=float(l_i),
                    bottom=float(high_i_minus_2),
                    bias=BULLISH,
                    created_time=t,
                    created_index=i,
                )
                fvgs.insert(0, fvg)
                all_fvgs.append(fvg)
            if h_i < low_i_minus_2 and last_close < low_i_minus_2 and (-bdp) > threshold:
                fvg = FairValueGap(
                    top=float(h_i),
                    bottom=float(low_i_minus_2),
                    bias=BEARISH,
                    created_time=t,
                    created_index=i,
                )
                fvgs.insert(0, fvg)
                all_fvgs.append(fvg)

        prev_close = c_i

    # -------------------------------------------------------------------------
    # Premium / Equilibrium / Discount zones (computed from trailing extremes)
    #   Pine drawPremiumDiscountZones():
    #     Premium:     top = trailing.top,
    #                  bottom = 0.95*top + 0.05*bottom
    #     Equilibrium: top = 0.525*top + 0.475*bottom,
    #                  bottom = 0.525*bottom + 0.475*top
    #     Discount:    top = 0.95*bottom + 0.05*top,
    #                  bottom = trailing.bottom
    # -------------------------------------------------------------------------
    premium_discount: Dict[str, float] = {}
    if not np.isnan(trailing.top) and not np.isnan(trailing.bottom):
        t_top = float(trailing.top)
        t_bot = float(trailing.bottom)
        premium_discount = {
            "premium_top": t_top,
            "premium_bottom": 0.95 * t_top + 0.05 * t_bot,
            "equilibrium_top": 0.525 * t_top + 0.475 * t_bot,
            "equilibrium_bottom": 0.525 * t_bot + 0.475 * t_top,
            "discount_top": 0.95 * t_bot + 0.05 * t_top,
            "discount_bottom": t_bot,
        }

    # Surface only the requested number of OBs per scope (Pine's
    # internalOrderBlocksSizeInput / swingOrderBlocksSizeInput), but keep them
    # in newest-first order (OB at index 0 is the most recent, matching Pine's
    # use of ``array.unshift``).
    visible_internal = internal_obs[: settings.internal_order_blocks_size]
    visible_swing = swing_obs[: settings.swing_order_blocks_size]

    return SMCResult(
        bars_analyzed=n,
        structure_events=structure_events,
        equal_events=equal_events,
        fair_value_gaps=list(all_fvgs),
        swing_order_blocks=visible_swing,
        internal_order_blocks=visible_internal,
        mitigated_swing_order_blocks=mitigated_swing_obs,
        mitigated_internal_order_blocks=mitigated_internal_obs,
        swing_trend=swing_trend,
        internal_trend=internal_trend,
        last_swing_high=swing_high,
        last_swing_low=swing_low,
        last_internal_high=internal_high,
        last_internal_low=internal_low,
        trailing_top=float(trailing.top),
        trailing_bottom=float(trailing.bottom),
        trailing_last_top_time=trailing.last_top_time,
        trailing_last_bottom_time=trailing.last_bottom_time,
        premium_discount=premium_discount,
    )


__all__ = [
    "BULLISH",
    "BEARISH",
    "SMCSettings",
    "SMCResult",
    "Pivot",
    "TrailingExtremes",
    "OrderBlock",
    "FairValueGap",
    "StructureEvent",
    "EqualEvent",
    "analyze_smart_money",
]
