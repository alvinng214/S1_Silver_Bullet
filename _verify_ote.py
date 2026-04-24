"""Verification probes for ``Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py``.

Synthetic-only tests that pin the Pine state machine's observable behaviour:

1. ``_fibb`` unit test: bullish vs bearish retrace geometry and degenerate
   (``il == ih``, NaN anchor) cases.
2. No events fire before the first pivot can possibly confirm (``b < 2*prd``).
   This catches the legacy "``up > highs[b-1]``" bug, which fires a false
   CHoCH on bar 1 as soon as any price makes a new relative high.
3. Pure uptrend: one bullish CHoCH followed by strictly bullish continuations;
   no bearish events; pos marches 1 → 2 → 3 → ...
4. Pure downtrend: symmetric to (3) with bearish events and negative pos.
5. Bull → bear flip: bullish CHoCH, then a decisive break below establishes
   a bearish CHoCH; pos sign flips negative.
6. Active-zone integrity: final ``active_zone`` matches the last event's
   ``fib_values`` and the last bar state's ``fib_prices``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "Fibonacci_Optimal_Entry_Zone__OTE___Zeiierman_.py")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ote = _load("ote_zeiierman", _SRC)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  OK  {msg}")


def _df_from_hl(highs, lows) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(highs), freq="B")
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes}, index=idx
    )


# ---------------------------------------------------------------------------
# Probe 1 — _fibb unit
# ---------------------------------------------------------------------------
def probe_fibb_unit() -> None:
    print("\n[probe 1] _fibb geometry")
    # Bullish context: low formed first (il < ih), retrace from h downward.
    _assert(
        math.isclose(ote._fibb(0.5, 100.0, 90.0, 10, 5), 95.0),
        "bullish 0.5 retrace: 100 - (100-90)*0.5 = 95",
    )
    _assert(
        math.isclose(ote._fibb(0.618, 100.0, 90.0, 10, 5), 100.0 - 10.0 * 0.618),
        "bullish 0.618 retrace: 100 - 6.18 = 93.82",
    )
    # Bearish context: high formed first (il > ih), retrace from l upward.
    _assert(
        math.isclose(ote._fibb(0.5, 100.0, 90.0, 5, 10), 95.0),
        "bearish-style 0.5 (h=100, l=90, ih<il): 90 + 10*0.5 = 95",
    )
    _assert(
        math.isclose(ote._fibb(0.618, 100.0, 90.0, 5, 10), 90.0 + 10.0 * 0.618),
        "bearish-style 0.618: 90 + 6.18 = 96.18",
    )
    # Degenerate: il == ih
    _assert(
        math.isnan(ote._fibb(0.5, 100.0, 90.0, 5, 5)),
        "il == ih → NaN (degenerate)",
    )
    # NaN anchor indices
    _assert(
        math.isnan(ote._fibb(0.5, 100.0, 90.0, ote._NA_IDX, 5)),
        "anchor_i_high == _NA_IDX → NaN",
    )
    # NaN h
    _assert(
        math.isnan(ote._fibb(0.5, float("nan"), 90.0, 10, 5)),
        "NaN h → NaN",
    )


# ---------------------------------------------------------------------------
# Probe 2 — no premature events (the legacy shift-by-one bug fix)
# ---------------------------------------------------------------------------
def probe_no_premature_events() -> None:
    print("\n[probe 2] no events before 2*prd bars")
    prd = 5
    # Small spike on bar 1 that WOULD trigger the legacy "up > highs[b-1]"
    # bug as a phantom CHoCH, but that cannot be a confirmed Pine pivot.
    highs = [100.0, 101.0] + [100.0] * 20
    lows = [99.0] * 22
    df = _df_from_hl(highs, lows)

    out = ote.calculate_ote(df, ote.OTESettings(prd=prd))
    early = [e for e in out.events if e.bar < 2 * prd]
    _assert(
        len(early) == 0,
        f"no events with bar < 2*prd ({2*prd}); got {[e.bar for e in out.events]}",
    )


# ---------------------------------------------------------------------------
# Probe 3 — pure uptrend
# ---------------------------------------------------------------------------
def probe_pure_uptrend() -> None:
    print("\n[probe 3] zigzag uptrend → bullish events only, pos strictly climbs")
    prd = 3
    # A monotonic ramp produces no pivots (each new bar prints the new high),
    # which is correct Pine behaviour but useless for exercising the state
    # machine. Build a zigzag uptrend: every ~5 bars pulls back 2 bars,
    # seeding both pivot highs and pivot lows, then breaks out to a new HH.
    rng = np.random.default_rng(42)
    highs = []
    lows = []
    level = 100.0
    for cycle in range(12):
        # 5-bar rally
        for _ in range(5):
            level += rng.uniform(1.0, 2.0)
            highs.append(level)
            lows.append(level - 0.5)
        # 2-bar pullback
        for _ in range(2):
            level -= rng.uniform(0.5, 1.5)
            highs.append(level + 0.3)
            lows.append(level)
    df = _df_from_hl(highs, lows)

    out = ote.calculate_ote(df, ote.OTESettings(prd=prd))

    directions = [e.direction for e in out.events]
    _assert(
        all(d == "bullish" for d in directions),
        f"all events are bullish (got {set(directions)})",
    )
    _assert(len(out.events) >= 1, "at least one bullish event fired")
    # First bullish event must be a CHoCH.
    _assert(out.events[0].kind == "choch", "first bullish event is a CHoCH")
    # pos_after should be 1, then 2, then 3, ... monotonically non-decreasing.
    pos_values = [e.pos_after for e in out.events]
    _assert(pos_values[0] == 1, "first bullish CHoCH sets pos_after=1")
    _assert(
        all(b - a >= 1 for a, b in zip(pos_values, pos_values[1:])),
        "pos_after strictly climbs across consecutive bullish events",
    )


# ---------------------------------------------------------------------------
# Probe 4 — pure downtrend
# ---------------------------------------------------------------------------
def probe_pure_downtrend() -> None:
    print("\n[probe 4] zigzag downtrend → bearish events only, pos strictly drops")
    prd = 3
    rng = np.random.default_rng(43)
    highs = []
    lows = []
    level = 140.0
    for cycle in range(12):
        # 5-bar slide
        for _ in range(5):
            level -= rng.uniform(1.0, 2.0)
            highs.append(level + 0.5)
            lows.append(level)
        # 2-bar bounce
        for _ in range(2):
            level += rng.uniform(0.5, 1.5)
            highs.append(level)
            lows.append(level - 0.3)
    df = _df_from_hl(highs, lows)

    out = ote.calculate_ote(df, ote.OTESettings(prd=prd))

    directions = [e.direction for e in out.events]
    _assert(
        all(d == "bearish" for d in directions),
        f"all events are bearish (got {set(directions)})",
    )
    _assert(len(out.events) >= 1, "at least one bearish event fired")
    _assert(out.events[0].kind == "choch", "first bearish event is a CHoCH")
    pos_values = [e.pos_after for e in out.events]
    _assert(pos_values[0] == -1, "first bearish CHoCH sets pos_after=-1")
    _assert(
        all(b - a <= -1 for a, b in zip(pos_values, pos_values[1:])),
        "pos_after strictly drops across consecutive bearish events",
    )


# ---------------------------------------------------------------------------
# Probe 5 — bull → bear flip
# ---------------------------------------------------------------------------
def probe_flip() -> None:
    print("\n[probe 5] bull-then-bear: regime flips, sign of pos flips negative")
    prd = 3
    # First half: zigzag uptrend. Second half: steep drop.
    rng = np.random.default_rng(44)
    highs, lows = [], []
    level = 100.0
    for _ in range(8):
        for _ in range(5):
            level += rng.uniform(1.0, 2.0)
            highs.append(level)
            lows.append(level - 0.5)
        for _ in range(2):
            level -= rng.uniform(0.5, 1.2)
            highs.append(level + 0.3)
            lows.append(level)
    # Then a decisive break down, well below all recent pivot lows.
    for _ in range(8):
        for _ in range(5):
            level -= rng.uniform(1.5, 2.5)
            highs.append(level + 0.5)
            lows.append(level)
        for _ in range(2):
            level += rng.uniform(0.3, 1.0)
            highs.append(level)
            lows.append(level - 0.3)
    df = _df_from_hl(highs, lows)

    out = ote.calculate_ote(df, ote.OTESettings(prd=prd))
    _assert(len(out.events) >= 2, "at least two events (bullish then bearish)")
    first_bear_idx = next(
        (i for i, e in enumerate(out.events) if e.direction == "bearish"),
        -1,
    )
    _assert(
        first_bear_idx > 0,
        "a bearish event exists AFTER at least one bullish event",
    )
    first_bear = out.events[first_bear_idx]
    _assert(
        first_bear.kind == "choch",
        "the first bearish event after a bullish run is a CHoCH",
    )
    _assert(
        first_bear.pos_after == -1,
        "first bearish CHoCH resets pos_after to -1",
    )


# ---------------------------------------------------------------------------
# Probe 6 — active_zone consistency
# ---------------------------------------------------------------------------
def probe_active_zone_consistency() -> None:
    print("\n[probe 6] active_zone matches last event + last bar state")
    prd = 3
    rng = np.random.default_rng(45)
    highs, lows = [], []
    level = 100.0
    for _ in range(12):
        for _ in range(5):
            level += rng.uniform(1.0, 2.0)
            highs.append(level)
            lows.append(level - 0.5)
        for _ in range(2):
            level -= rng.uniform(0.5, 1.2)
            highs.append(level + 0.3)
            lows.append(level)
    df = _df_from_hl(highs, lows)

    out = ote.calculate_ote(df, ote.OTESettings(prd=prd))
    _assert(len(out.events) >= 1, "at least one event (required for this probe)")
    last_evt = out.events[-1]
    _assert(
        out.active_zone.direction == last_evt.direction,
        f"active_zone.direction ({out.active_zone.direction}) matches last event ({last_evt.direction})",
    )
    # Last bar state's fib_prices must equal the active zone.
    last_state = out.states[-1]
    _assert(
        last_state.fib_prices == out.active_zone.fib_prices,
        "last bar state.fib_prices mirrors active_zone.fib_prices",
    )
    # zone_top / zone_bottom are consistent with fib_prices
    finite = [p for p in out.active_zone.fib_prices if not math.isnan(p)]
    if finite:
        _assert(
            math.isclose(out.active_zone.zone_top, max(finite)),
            "zone_top = max(fib_prices)",
        )
        _assert(
            math.isclose(out.active_zone.zone_bottom, min(finite)),
            "zone_bottom = min(fib_prices)",
        )


def main() -> None:
    probe_fibb_unit()
    probe_no_premature_events()
    probe_pure_uptrend()
    probe_pure_downtrend()
    probe_flip()
    probe_active_zone_consistency()
    print("\nAll OTE probes passed.")


if __name__ == "__main__":
    main()
