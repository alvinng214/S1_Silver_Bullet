"""Verification probes for the HTF-publishing timing fix in
``Market Structure MTF Trend [Pt].py``.

Two probes:

1. Synthetic BoS week: build a daily OHLC series where a weekly pivot high
   is broken by the close on a specific Friday. Assert that the
   ``bos_edge=True`` for the ``1W`` timeframe lands on that Friday, NOT on
   the following Monday (the old shift(1) behaviour).

2. Synthetic HTF mapping unit test: feed a controlled LTF/HTF index pair to
   ``_confirmed_htf_positions`` and assert the expected per-LTF-bar HTF
   positions exactly, including the "defer to j-1" and "publish j" cases.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_THIS_DIR, "Market Structure MTF Trend [Pt].py")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ms = _load("ms_ptv", _SRC)


def _assert_eq(name: str, got, want) -> None:
    if got != want:
        raise AssertionError(f"[{name}] expected {want!r}, got {got!r}")
    print(f"  OK  {name}: {got!r}")


# ---------------------------------------------------------------------------
# Probe 1 — confirmed-bar mapping on hand-built indices
# ---------------------------------------------------------------------------
def probe_confirmed_htf_positions() -> None:
    print("\n[probe 1] _confirmed_htf_positions synthetic")

    # LTF: daily bars Mon-Fri across 3 consecutive weeks (15 bars).
    ltf = pd.DatetimeIndex(
        [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",  # week 0
            "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",  # week 1
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",  # week 2
        ]
    )
    # HTF: weekly bars labelled on the Mondays (left-labelled resample).
    htf = pd.DatetimeIndex(["2024-01-01", "2024-01-08", "2024-01-15"])

    got = ms._confirmed_htf_positions(ltf, htf).tolist()
    # Expected:
    #  - Mon..Thu of week 0 (indices 0..3) -> still inside week 0 -> j-1 = -1
    #  - Fri of week 0 (index 4)           -> next LTF = Mon of week 1 ->
    #                                         in later HTF bar -> publish j=0
    #  - Mon..Thu of week 1 (indices 5..8) -> still inside week 1 -> j-1 = 0
    #  - Fri of week 1 (index 9)           -> next LTF in week 2 -> publish j=1
    #  - Mon..Thu of week 2 (indices 10..13)-> still inside week 2 -> j-1 = 1
    #  - Fri of week 2 (index 14, LAST)    -> live-edge extrapolates via last
    #                                         diff (1 day) -> Sat -> still
    #                                         week 2 -> j-1 = 1
    want = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    _assert_eq("confirmed_pos[Mon w0]", got[0], want[0])
    _assert_eq("confirmed_pos[Fri w0 = week 0 CONFIRM]", got[4], want[4])
    _assert_eq("confirmed_pos[Mon w1]", got[5], want[5])
    _assert_eq("confirmed_pos[Fri w1 = week 1 CONFIRM]", got[9], want[9])
    _assert_eq("confirmed_pos[Mon w2]", got[10], want[10])
    _assert_eq("confirmed_pos[Fri w2 = live edge, still forming]", got[14], want[14])
    _assert_eq("full vector", got, want)


# ---------------------------------------------------------------------------
# Probe 2 — end-to-end BoS event lands on the correct LTF day
# ---------------------------------------------------------------------------
def _build_daily_series_with_weekly_bos() -> pd.DataFrame:
    """Daily OHLC where a weekly pivot high gets broken on a specific Friday.

    We build 30 weeks of tame sideways action followed by a "break week"
    where Friday's close is above the highest weekly high of the prior
    ~30 weeks. With weekly pivot_len = 3 (short, for tractability), the
    last confirmed weekly pivot high will sit well below Friday's close of
    the break week, so the weekly MS state flips bos=True on THAT weekly
    bar, and our HTF->LTF mapping must publish it on that week's Friday.
    """
    weeks = 30
    rng = pd.bdate_range("2023-01-02", periods=weeks * 5, freq="B")

    # Tame oscillation: closes drift between 100 and 105 with a decaying
    # amplitude, with a single prominent pivot high in week 10.
    closes = []
    for w in range(weeks):
        base = 100.0 + 0.05 * w
        for d in range(5):
            if w == 10 and d == 2:  # mid-week high (the weekly pivot H)
                closes.append(base + 4.5)
            elif w == 10 and d == 3:
                closes.append(base + 3.8)
            elif w == 10 and d == 4:
                closes.append(base + 3.6)
            else:
                closes.append(base + (d - 2) * 0.15)

    closes = np.array(closes, dtype=float)
    # Pad rectangular OHLC off the close.
    opens = closes - 0.2
    highs = closes + 0.3
    lows = closes - 0.3

    # Week 28 break: Friday's close pushes well above the week-10 pivot
    # high (~105 + 4.5 = 109.5). We set Fri of week 28 to 115.
    break_week_idx = 28
    fri_bar_idx = break_week_idx * 5 + 4
    closes[fri_bar_idx] = 115.0
    highs[fri_bar_idx] = 115.5
    opens[fri_bar_idx] = 110.0

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=rng,
    )
    return df


def probe_weekly_bos_timing() -> None:
    print("\n[probe 2] end-to-end weekly break timing on synthetic daily series")
    df = _build_daily_series_with_weekly_bos()

    # Compute MS on daily, requesting weekly HTF (10080min) with pivot_len=3
    # so the pivot confirms early enough.
    outputs = ms.calculate_market_structure_mtf(
        df,
        timeframes=("1440", "10080", "10080", "10080"),  # 1D, 1W (x3 placeholders)
        pivot_strengths=(3, 3, 3, 3),
        is_lower_tf=(False, False, False, False),
    )

    tf_weekly = outputs.tf2  # index 1 is the first weekly TF

    # The initial current_trend is False, so the first break of a pivot high
    # is a CHoCH (trend-flip), not a BoS. The timing-fix question is the
    # same either way: the event must land on the LTF bar whose close
    # coincides with the HTF bar's close.
    bos_edge = tf_weekly.bos_edge.astype(bool)
    bull_choch = tf_weekly.bullish_choch.astype(bool)

    break_events: list = []
    for ts, v in bos_edge.items():
        if bool(v):
            break_events.append(("BoS", ts))
    for ts, v in bull_choch.items():
        if bool(v):
            break_events.append(("bullCHoCH", ts))
    break_events.sort(key=lambda r: r[1])

    break_week_idx = 28
    expected_fri = df.index[break_week_idx * 5 + 4]
    next_mon = df.index[break_week_idx * 5 + 5] if (break_week_idx * 5 + 5) < len(df) else None

    print(f"  expected Fri (breakout close)  = {expected_fri.date()}")
    if next_mon is not None:
        print(f"  next Mon (was the OLD buggy)   = {next_mon.date()}")
    print(
        "  break events on weekly TF      = "
        f"{[(k, t.date().isoformat()) for k, t in break_events]}"
    )

    # Filter to bullish events on or after the expected Friday — the first
    # one must land EXACTLY on the breakout Friday, not the following Monday.
    on_or_after = [e for e in break_events if e[1] >= expected_fri]
    if not on_or_after:
        raise AssertionError(
            f"No BoS or bullish CHoCH on or after {expected_fri}. "
            f"All events: {break_events}"
        )
    kind, first = on_or_after[0]
    if first.date() != expected_fri.date():
        raise AssertionError(
            f"{kind} landed on {first.date()} (expected {expected_fri.date()}). "
            f"Looks like the old shift(1) behaviour resurfaced."
        )
    print(f"  OK  first breakout event ({kind}) lands on the confirming Friday")


def main() -> None:
    probe_confirmed_htf_positions()
    probe_weekly_bos_timing()
    print("\nAll market-structure timing probes passed.")


if __name__ == "__main__":
    main()
