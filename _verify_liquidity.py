"""Synthetic probes for the three liquidity translations.

Goal: verify each Python module's detection logic reproduces the
Pine-source behaviour on small, hand-designed DataFrames where the expected
outcome is unambiguous.

Run: .venv/bin/python S1_Silver_Bullet/_verify_liquidity.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass on Python 3.14 walks sys.modules
    spec.loader.exec_module(mod)
    return mod


ualgo = _load("ualgo", "Liquidity Sweeps [UAlgo].py")
target = _load("target", "SMC Target Liquidity V.35 (Manual Distance Control).py")
fadi = _load("fadi", "ICT HTF MSS & Liquidity (fadi).py")


def _ohlc(prices_high, prices_low, prices_close=None, start="2024-01-02", freq="B"):
    idx = pd.date_range(start, periods=len(prices_high), freq=freq)
    if prices_close is None:
        prices_close = [(h + l) / 2 for h, l in zip(prices_high, prices_low)]
    df = pd.DataFrame(
        {
            "open": prices_close,
            "high": prices_high,
            "low": prices_low,
            "close": prices_close,
        },
        index=idx,
    )
    return df


# =========================================================================
# 1. Liquidity Sweeps [UAlgo]
# =========================================================================
def test_ualgo_basic_buy_sweep():
    """Bar 10 is a strict pivot high. Later a bar wicks above AND closes below."""
    # pivot_period=3 for speed. Need >= 2*3+1 = 7 bars minimum.
    # Construct: small values, then a clear pivot high at index 5, confirmed at index 8.
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 13, 22, 11, 10, 9]  # pivot at idx 5 (h=20)
    lows = [h - 2 for h in highs]
    closes = [h - 0.5 for h in highs]
    # Sweep expected at idx 10 (high=22 > pivot=20, close=21.5 > pivot=20?) need close < 20.
    closes[10] = 19.0
    df = _ohlc(highs, lows, closes)

    res = ualgo.compute_liquidity_sweeps(df, pivot_period=3, max_lines=3)
    kinds = [e.kind for e in res.events]
    assert "buy_liquidity_sweep" in kinds, f"expected buy_liquidity_sweep in {kinds}"
    sweep = next(e for e in res.events if e.kind == "buy_liquidity_sweep")
    assert abs(sweep.pivot_price - 20) < 1e-9
    assert sweep.event_price == 22
    print("OK  UAlgo: buy_liquidity_sweep detected at pivot=20, event_high=22")


def test_ualgo_breakout_invalidation():
    """Pivot high gets taken out by a close > pivot (no wick-back) → breakout_up event, no sweep."""
    highs = [10, 11, 12, 13, 14, 20, 14, 13, 12, 13, 22, 22, 22]
    lows = [h - 2 for h in highs]
    closes = [h - 0.5 for h in highs]
    closes[10] = 21.5  # CLOSE above pivot 20 → breakout_up
    df = _ohlc(highs, lows, closes)

    res = ualgo.compute_liquidity_sweeps(df, pivot_period=3, max_lines=3)
    kinds = [e.kind for e in res.events]
    assert "buy_liquidity_sweep" not in kinds
    assert "breakout_up" in kinds
    print("OK  UAlgo: breakout_up (close > pivot) → no sweep emitted, invalidation only")


def test_ualgo_sell_sweep():
    """Mirror: pivot low swept by wick-below + close-above."""
    lows = [20, 19, 18, 17, 16, 10, 16, 17, 18, 17, 8, 16, 17]
    highs = [l + 2 for l in lows]
    closes = [l + 0.5 for l in lows]
    closes[10] = 11.0  # low=8 < pivot=10 AND close=11 > 10 → sell sweep
    df = _ohlc(highs, lows, closes)

    res = ualgo.compute_liquidity_sweeps(df, pivot_period=3, max_lines=3)
    kinds = [e.kind for e in res.events]
    assert "sell_liquidity_sweep" in kinds, kinds
    sweep = next(e for e in res.events if e.kind == "sell_liquidity_sweep")
    assert sweep.pivot_price == 10
    assert sweep.event_price == 8
    print("OK  UAlgo: sell_liquidity_sweep detected at pivot=10, event_low=8")


def test_ualgo_max_lines():
    """When > max_lines pivots pile up, the oldest is dropped."""
    np.random.seed(1)
    n = 60
    highs = np.linspace(10, 30, n) + np.random.uniform(0, 1, n)
    # manually seed 5 pivots
    for i in (10, 20, 30, 40, 50):
        highs[i] += 5
    lows = highs - 2
    closes = highs - 0.5
    df = _ohlc(list(highs), list(lows), list(closes))

    res = ualgo.compute_liquidity_sweeps(df, pivot_period=3, max_lines=2)
    # Expect at most 2 lines remaining at any time → at end, <= 2 active.
    assert len(res.active_resistance) <= 2
    print(f"OK  UAlgo: active_resistance={len(res.active_resistance)} ≤ max_lines=2")


# =========================================================================
# 2. SMC Target Liquidity V.35
# =========================================================================
def test_target_sfp_support():
    """Support line: close[prev]<lvl then close[cur]>lvl → SFP."""
    # prd=3 → need >= 2*prd+2=8 bars.
    lows = [20, 19, 18, 10, 18, 19, 20, 15, 12, 14]  # pivot at idx 3 (low=10)
    highs = [l + 2 for l in lows]
    closes = [l + 0.5 for l in lows]
    # Confirm bar: idx 6. After that, at idx 7 close=9 < 10, idx 8 close=14 > 10 → SFP?
    # Wait: SFP needs close[i]>lvl AND close[i-1]<lvl. So at i=8, close=14>10 AND close[7]=9<10 → SFP.
    closes[7] = 9.0  # prev close below
    closes[8] = 14.0  # current close above
    df = _ohlc(highs, lows, closes)

    s = target.SMCTargetSettings(prd=3, max_active_lines=5)
    res = target.compute_smc_target_liquidity(df, s)
    kinds = [(e.resolution, e.side) for e in res.events]
    assert ("SFP", "support") in kinds, kinds
    print("OK  Target: SFP (support) detected after close[prev]<lvl and close[cur]>lvl")


def test_target_mss_support():
    """Support line: close[cur]<lvl AND close[prev]<lvl AND close[cur]<close[prev] → MSS."""
    lows = [20, 19, 18, 10, 18, 19, 20, 12, 9, 8]  # pivot at 3 (low=10), confirm at 6
    highs = [l + 2 for l in lows]
    closes = [l + 0.5 for l in lows]
    # At i=7: close=9 < lvl=10. At i=8: close=8 < 10 AND close<close[7]=9 → MSS at i=8.
    closes[7] = 9.0
    closes[8] = 8.0
    df = _ohlc(highs, lows, closes)

    s = target.SMCTargetSettings(prd=3, max_active_lines=5)
    res = target.compute_smc_target_liquidity(df, s)
    kinds = [(e.resolution, e.side) for e in res.events]
    # Could be X first if low<=lvl and close>lvl at i=7 — check priority.
    assert ("MSS", "support") in kinds, kinds
    print("OK  Target: MSS (support) detected on second bar of breakdown")


def test_target_x_support():
    """Support line: low<=lvl AND close>lvl on the same bar → X."""
    lows = [20, 19, 18, 10, 18, 19, 20, 15, 9, 15]  # pivot at 3 (low=10)
    highs = [l + 2 for l in lows]
    closes = [l + 0.5 for l in lows]
    closes[8] = 11.0  # close > 10 while low = 9 < 10 → X
    df = _ohlc(highs, lows, closes)

    s = target.SMCTargetSettings(prd=3, max_active_lines=5)
    res = target.compute_smc_target_liquidity(df, s)
    kinds = [(e.resolution, e.side) for e in res.events]
    assert ("X", "support") in kinds, kinds
    print("OK  Target: X (wick-only sweep on support) detected")


def test_target_sessions_daily_is_other():
    """Daily bars: the short-circuit forces every session tag to 'other'.

    On intraday Pine charts ``time(tf, sess, tz)`` is meaningful; on daily
    bars it leaks the midnight-UTC → UTC-4 offset and would wrongly tag
    every pivot 'asia'. The Python module detects daily-or-slower cadence
    and returns 'other' uniformly — this is the agreed-upon Q2 default.
    """
    np.random.seed(2)
    n = 30
    highs = 20 + np.random.uniform(0, 1, n)
    lows = 15 + np.random.uniform(0, 1, n)
    lows[10] = 10.0
    closes = (highs + lows) / 2
    df = _ohlc(list(highs), list(lows), list(closes))

    s = target.SMCTargetSettings(prd=3, tz_manual="UTC-4")
    res = target.compute_smc_target_liquidity(df, s)
    all_sessions = {lvl.session for lvl in res.active_buy_lines + res.active_sell_lines}
    all_sessions |= {e.session for e in res.events}
    assert all_sessions <= {"other"}, f"expected only 'other' on daily, got {all_sessions}"
    print("OK  Target: sessions all resolve to 'other' on daily bars (short-circuit)")


def test_target_sessions_intraday():
    """On hourly bars with NY timezone, pivots at 14:30 should map to ny_pm."""
    idx = pd.date_range("2024-01-02 08:00", periods=40, freq="30min")
    highs = np.linspace(20, 30, 40) + np.random.uniform(0, 0.5, 40)
    lows = highs - 2
    # Inject a pivot low at index 11 (= 13:30 local UTC → 09:30 ET, will be in NY AM session)
    lows[11] = 10.0
    closes = (highs + lows) / 2
    df = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes}, index=idx
    )

    s = target.SMCTargetSettings(prd=3, tz_manual="UTC-4")  # ET-ish
    res = target.compute_smc_target_liquidity(df, s)
    seen = {lvl.session for lvl in res.active_buy_lines + res.active_sell_lines}
    seen |= {e.session for e in res.events}
    # Must see at least one non-other session given the time window.
    assert seen - {"other"}, f"expected at least one intraday session tag, got {seen}"
    print(f"OK  Target: intraday session tagging produced {sorted(seen)}")


# =========================================================================
# 3. ICT HTF MSS & Liquidity (fadi)
# =========================================================================
def test_fadi_pivot_detection():
    """On weekly HTF, detect the middle of 3 HTF bars as a pivot."""
    # Build ~60 daily bars spanning ~12 weeks. Engineer weekly highs such that
    # week 6 is the highest, week 5 and 7 lower → a simple STH pivot in week 6.
    idx = pd.date_range("2024-01-01", periods=90, freq="B")
    high = np.full(90, 50.0)
    low = np.full(90, 40.0)
    close = np.full(90, 45.0)
    # assign weekly high pattern: bumps at week 6 (~indices 25-29) and flatness elsewhere
    for i in range(len(idx)):
        wk = (idx[i] - idx[0]).days // 7
        if wk == 6:
            high[i] = 70.0
        elif wk == 10:
            high[i] = 60.0
    low = high - 10
    close = (high + low) / 2
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )

    s = fadi.FadiSettings(htf="1W", level="Short Term", max_lines=50)
    res = fadi.compute_htf_mss_liquidity(df, s)
    pivots = res.pivots
    highs = [p for p in pivots if p.is_high]
    assert any(abs(p.price - 70.0) < 1e-6 for p in highs), [p.price for p in highs]
    print(f"OK  fadi: detected STH at price 70.0 ({len(highs)} high pivots total)")


def test_fadi_claim_and_reclaim():
    """After a pivot is formed, first LTF close beyond → claim. Close back through → reclaim."""
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    high = np.full(120, 50.0)
    low = np.full(120, 40.0)
    close = np.full(120, 45.0)
    for i in range(len(idx)):
        wk = (idx[i] - idx[0]).days // 7
        if wk == 3:
            high[i] = 80.0  # pivot high in week 3
    low = high - 10
    close = (high + low) / 2
    # Later (after pivot confirmed): have close > 80 to claim, then close < 80 to reclaim.
    # Week 3 is roughly days 15-19. Pivot is detected at week 5 (after week 3 becomes
    # the middle of weeks 2,3,4). So from week 6 onward we can trigger claim/reclaim.
    # Days 50+: jam close up to 85 (claim), then day 60: close drops back to 70 (reclaim).
    for i in range(50, 55):
        high[i] = 90.0
        close[i] = 85.0
        low[i] = 80.0
    for i in range(60, 65):
        high[i] = 80.0
        close[i] = 70.0
        low[i] = 65.0
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close}, index=idx
    )

    s = fadi.FadiSettings(htf="1W", level="Short Term", max_lines=50)
    res = fadi.compute_htf_mss_liquidity(df, s, _claim_all_tiers=True)
    pivot_80 = next((p for p in res.pivots if p.is_high and abs(p.price - 80) < 1e-6), None)
    assert pivot_80 is not None, [p.price for p in res.pivots if p.is_high]
    assert pivot_80.claimed, f"expected claimed; got {pivot_80}"
    assert pivot_80.reclaimed, f"expected reclaimed; got {pivot_80}"
    assert pivot_80.claim_time < pivot_80.reclaim_time
    print(
        f"OK  fadi: pivot@80 claimed at {pivot_80.claim_time.date()}, "
        f"reclaimed at {pivot_80.reclaim_time.date()}"
    )


def test_fadi_validtf_rejects_equal_tf():
    """When LTF cadence >= HTF cadence, the guard rejects the run."""
    idx = pd.date_range("2024-01-01", periods=30, freq="W")
    n = len(idx)
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 130, n),
            "high": np.linspace(100, 130, n) + 5,
            "low": np.linspace(100, 130, n) - 5,
            "close": np.linspace(100, 130, n),
        },
        index=idx,
    )
    s = fadi.FadiSettings(htf="1W")
    try:
        fadi.compute_htf_mss_liquidity(df, s)
    except ValueError as e:
        print(f"OK  fadi: Validtimeframe guard rejected equal-TF case: {e}")
        return
    raise AssertionError("expected ValueError for LTF >= HTF")


# =========================================================================
# Run all
# =========================================================================
if __name__ == "__main__":
    tests = [
        ("UAlgo buy sweep", test_ualgo_basic_buy_sweep),
        ("UAlgo breakout (no sweep)", test_ualgo_breakout_invalidation),
        ("UAlgo sell sweep", test_ualgo_sell_sweep),
        ("UAlgo max_lines cap", test_ualgo_max_lines),
        ("Target SFP support", test_target_sfp_support),
        ("Target MSS support", test_target_mss_support),
        ("Target X support", test_target_x_support),
        ("Target sessions on daily", test_target_sessions_daily_is_other),
        ("Target sessions intraday", test_target_sessions_intraday),
        ("fadi pivot detection", test_fadi_pivot_detection),
        ("fadi claim/reclaim", test_fadi_claim_and_reclaim),
        ("fadi Validtimeframe guard", test_fadi_validtf_rejects_equal_tf),
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERR   {name}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"{len(failed)} FAILED")
        for n, msg in failed:
            print(f"  - {n}: {msg}")
        sys.exit(1)
    print(f"All {len(tests)} probes passed")
