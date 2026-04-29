"""cd_sweep&cisd_Cx — Python port of cdikici71's Pine v5 indicator.

Mirrors the *detection* logic of `cd_sweep&cisd_Cx.txt`:
  - HTF candle state machine (o0/h0/l0/c0/t0 + history o1..o3/h1..h3/l1..l3/c1..c3).
  - High/low sweeps within the running HTF candle (h_swept, l_swept), plus the
    sticky h_swept1 / l_swept1 captured at HTF rollover.
  - Bullish & bearish CISD level discovery — both the engulf-style branch and
    the lookback-search branch (i = 2..10), including the j-loop highest-bearish-open
    refinement and the tail-edge open/high overrides.
  - xbull / xbear triggers with xcisd / ycisd one-shot guards, including the
    "faded" cisd box case (level erased but not armed).
  - Independent HTF bias state machine on the htfbias timeframe.

Visual elements (boxes, candle minicharts, key-level boxes, screener tables,
SMT labels, multi-symbol request.security alerts, remaining-time clock) are out
of scope; this port emits per-bar detection state so it can be diffed against
the Pine boxes/triggers in TradingView.

Usage
-----
Library:
    import importlib.util, pathlib
    p = pathlib.Path('cd_sweep&cisd_Cx.py')
    spec = importlib.util.spec_from_file_location('cd_sweep_cisd_Cx', p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = mod.analyze(ltf_bars, htf_bars=htf_bars, htfbias_bars=htfbias_bars)

CLI (paths to JSON files containing list[dict] of bars):
    python3 'cd_sweep&cisd_Cx.py' --ltf ltf.json --htf htf.json [--htfbias bias.json] [--summary]

Each bar dict needs keys: time (ms epoch), open, high, low, close.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# HTF rollover detection
# ---------------------------------------------------------------------------

def _change_mask(ltf_times: Sequence[int], htf_times: Sequence[int]) -> list[bool]:
    """Return list[bool]: True at LTF bars where the containing HTF bucket index increments.

    Mirrors Pine's `timeframe.change(htf)` using a bisect-based bucket lookup
    against the supplied HTF open-time series. Bar 0 is forced True so the
    HTF state machine initialises (Pine does an implicit init via `var`).
    """
    out: list[bool] = []
    last_bucket = -1
    for t in ltf_times:
        idx = bisect.bisect_right(htf_times, t) - 1 if htf_times else -1
        out.append(idx != last_bucket)
        last_bucket = idx
    if out:
        out[0] = True
    return out


def _f(bars: list[dict[str, Any]], key: str) -> list[float]:
    return [float(b[key]) for b in bars]


def _i(bars: list[dict[str, Any]], key: str) -> list[int]:
    return [int(b[key]) for b in bars]


# ---------------------------------------------------------------------------
# Core analyser
# ---------------------------------------------------------------------------

def analyze(
    ltf_bars: list[dict[str, Any]],
    *,
    htf_bars: list[dict[str, Any]] | None = None,
    htfbias_bars: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the cd_sweep&cisd state machine over an LTF series.

    Returns a dict with:
      - n_bars: int
      - events: {"xbull": [...], "xbear": [...]}  (trigger events with anchor + level)
      - bars: per-bar list of detection state (h_swept, l_swept, bull_level,
        bear_level, xbull, xbear, xcisd, ycisd, bias, plus h0/l0/o0/c0/h1/l1/o1/c1)
    """
    n = len(ltf_bars)
    if n == 0:
        return {"n_bars": 0, "events": {"xbull": [], "xbear": []}, "bars": []}

    times = _i(ltf_bars, "time")
    opens = _f(ltf_bars, "open")
    highs = _f(ltf_bars, "high")
    lows = _f(ltf_bars, "low")
    closes = _f(ltf_bars, "close")

    htf_times = _i(htf_bars, "time") if htf_bars else []
    bias_times = _i(htfbias_bars, "time") if htfbias_bars else htf_times

    htf_change = _change_mask(times, htf_times)
    bias_change = _change_mask(times, bias_times)

    up = [closes[i] > opens[i] for i in range(n)]
    dw = [closes[i] < opens[i] for i in range(n)]
    eq = [closes[i] == opens[i] for i in range(n)]

    # Per-bar arrays (what Pine `var`s would expose via `[N]`).
    o0 = [opens[0]] * n; h0 = [highs[0]] * n; l0 = [lows[0]] * n
    c0 = [closes[0]] * n; t0 = [times[0]] * n
    h0bar = [0] * n; l0bar = [0] * n; h0t = [times[0]] * n; l0t = [times[0]] * n
    o1 = [opens[0]] * n; h1 = [highs[0]] * n; l1 = [lows[0]] * n
    c1 = [closes[0]] * n; t1 = [times[0]] * n
    h2 = [highs[0]] * n; l2 = [lows[0]] * n; o2 = [opens[0]] * n; c2 = [closes[0]] * n
    h3 = [highs[0]] * n; l3 = [lows[0]] * n; o3 = [opens[0]] * n; c3 = [closes[0]] * n

    h_swept_a = [False] * n; l_swept_a = [False] * n
    h_swept1_a = [False] * n; l_swept1_a = [False] * n

    bull_level_a = [highs[0]] * n; bear_level_a = [lows[0]] * n
    bull_index_a = [0] * n; bear_index_a = [0] * n
    xcisd_a = [False] * n; ycisd_a = [False] * n
    xbull_a = [False] * n; xbear_a = [False] * n

    # Rolling scalars
    o0_, h0_, l0_, c0_, t0_ = opens[0], highs[0], lows[0], closes[0], times[0]
    h0bar_, l0bar_, h0t_, l0t_ = 0, 0, times[0], times[0]
    o1_, h1_, l1_, c1_, t1_ = opens[0], highs[0], lows[0], closes[0], times[0]
    h2_, l2_, o2_, c2_ = highs[0], lows[0], opens[0], closes[0]
    h3_, l3_, o3_, c3_ = highs[0], lows[0], opens[0], closes[0]
    h_swept1_, l_swept1_ = False, False

    bull_level = highs[0]
    bear_level = lows[0]
    bull_index = 0
    bear_index = 0
    xcisd = False
    ycisd = False

    events_bull: list[dict[str, Any]] = []
    events_bear: list[dict[str, Any]] = []

    for i in range(n):
        # Snapshot prev-bar state (Pine `[1]` reads).
        prev_o0, prev_h0, prev_l0, prev_t0 = o0_, h0_, l0_, t0_
        prev_o1, prev_h1, prev_l1, prev_c1 = o1_, h1_, l1_, c1_
        prev_h2, prev_l2, prev_o2, prev_c2 = h2_, l2_, o2_, c2_
        prev_h0bar, prev_l0bar = h0bar_, l0bar_
        prev_h_swept = h_swept_a[i - 1] if i > 0 else False
        prev_l_swept = l_swept_a[i - 1] if i > 0 else False
        prev_close = closes[i - 1] if i > 0 else closes[i]

        # ----- Pine lines 102-106: timeframe.change(htf) → reset accumulators -----
        if htf_change[i]:
            t0_ = times[i]
            o0_ = opens[i]
            h0_ = highs[i]
            l0_ = lows[i]
            h0bar_ = i; l0bar_ = i
            h0t_ = times[i]; l0t_ = times[i]

        # ----- lines 107-115: roll high/low high-water marks; c0 := close -----
        if highs[i] >= h0_:
            h0_ = highs[i]; h0bar_ = i; h0t_ = times[i]
        if lows[i] <= l0_:
            l0_ = lows[i]; l0bar_ = i; l0t_ = times[i]
        c0_ = closes[i]

        # ----- lines 116-132: timeframe.change(htf) → shift HTF history down -----
        if htf_change[i]:
            o1_ = prev_o0
            h1_ = prev_h0
            l1_ = prev_l0
            t1_ = prev_t0
            c1_ = prev_close
            h2_ = prev_h1
            l2_ = prev_l1
            o2_ = prev_o1
            c2_ = prev_c1
            h3_ = prev_h2
            l3_ = prev_l2
            o3_ = prev_o2
            c3_ = prev_c2

        # ----- lines 205-206: live sweeps -----
        h_swept_now = (h0_ > h1_) and (max(o0_, c0_) < h1_)
        l_swept_now = (l0_ < l1_) and (min(o0_, c0_) > l1_)

        # ----- lines 208-210: sticky h_swept1 / l_swept1 captured at change -----
        if htf_change[i]:
            h_swept1_ = prev_h_swept
            l_swept1_ = prev_l_swept

        # ----- lines 254-281: bullish CISD level discovery -----
        if lows[i] == l0_ and lows[i] < l1_:
            cur_dw_or_eq = dw[i] or eq[i]
            prev_up_or_eq = (up[i - 1] or eq[i - 1]) if i > 0 else False
            both_eq = eq[i] and (eq[i - 1] if i > 0 else False)
            if cur_dw_or_eq and prev_up_or_eq and not both_eq:
                # engulf-style: down-after-up (no equal-equal pair).
                bull_level = opens[i]
                bull_index = i
            else:
                # lookback search, k = 2..10 bars back.
                for k in range(2, 11):
                    if i - k < 0:
                        break
                    if lows[i - k] < lows[i]:
                        break
                    flip_ok = ((up[i - k] or eq[i - k]) and dw[i - k + 1])
                    if flip_ok:
                        bar = k - 1
                        bull_level = opens[i - bar]
                        bull_index = i - bar
                        # Walk bars from `bar`-back down to current; raise to highest bearish open.
                        for j in range(bar, -1, -1):
                            jj = i - j
                            if jj < 0:
                                continue
                            if opens[jj] > bull_level and dw[jj]:
                                bull_level = opens[jj]
                                bull_index = jj
                        # Tail-edge corrections (Pine lines 273-278).
                        if bull_level < opens[i] and not (closes[i] > opens[i]):
                            bull_level = opens[i]
                            bull_index = i
                        if bull_level < opens[i] and (closes[i] > opens[i]):
                            bull_level = highs[i]
                            bull_index = i
                        break

        # ----- lines 283-311: bearish CISD level discovery -----
        if highs[i] == h0_ and highs[i] > h1_:
            cur_up_or_eq = up[i] or eq[i]
            prev_dw_or_eq = (dw[i - 1] or eq[i - 1]) if i > 0 else False
            both_eq = eq[i] and (eq[i - 1] if i > 0 else False)
            if cur_up_or_eq and prev_dw_or_eq and not both_eq:
                bear_level = opens[i]
                bear_index = i
            else:
                for k in range(2, 11):
                    if i - k < 0:
                        break
                    if highs[i - k] > highs[i]:
                        break
                    flip_ok = ((dw[i - k] or eq[i - k]) and up[i - k + 1])
                    if flip_ok:
                        ybar = k - 1
                        bear_level = opens[i - ybar]
                        bear_index = i - ybar
                        for j in range(ybar, -1, -1):
                            jj = i - j
                            if jj < 0:
                                continue
                            if opens[jj] < bear_level and up[jj]:
                                bear_level = opens[jj]
                                bear_index = jj
                        if bear_level > opens[i] and not (closes[i] < opens[i]):
                            bear_level = opens[i]
                            bear_index = i
                        if bear_level > opens[i] and (closes[i] < opens[i]):
                            bear_level = lows[i]
                            bear_index = i
                        break

        # ----- lines 313-316: ycisd / xcisd reset on new HTF extreme -----
        if highs[i] >= prev_h0:
            ycisd = False
        if lows[i] <= prev_l0:
            xcisd = False

        # ----- lines 318-339: xbull trigger + faded-cisd case -----
        xbull = False
        xbear = False
        if i > 0:
            cond_l = prev_l_swept or (l1_ <= l0_ and l_swept1_)
            cond_h = prev_h_swept or (h1_ >= h0_ and h_swept1_)
            if (closes[i - 1] > bull_level and cond_l and not xcisd
                    and (i - 1) >= bull_index):
                xbull = True
                events_bull.append({
                    "trigger_bar": i,
                    "trigger_time": times[i],
                    "anchor_bar": bull_index,
                    "anchor_time": times[bull_index],
                    "level": bull_level,
                })
                bull_level = 1_000_000.0
                xcisd = True
            elif (closes[i - 1] > bull_level and not cond_l and not xcisd
                    and (i - 1) >= bull_index):
                # Faded box: level erased, xcisd NOT set.
                bull_level = 1_000_000.0

            # ----- lines 341-364: xbear trigger + faded-cisd case -----
            if (closes[i - 1] < bear_level and cond_h and not ycisd
                    and (i - 1) >= bear_index):
                xbear = True
                events_bear.append({
                    "trigger_bar": i,
                    "trigger_time": times[i],
                    "anchor_bar": bear_index,
                    "anchor_time": times[bear_index],
                    "level": bear_level,
                })
                bear_level = 0.0
                ycisd = True
            elif (closes[i - 1] < bear_level and not cond_h and not ycisd
                    and (i - 1) >= bear_index):
                bear_level = 0.0

        # Persist per-bar state.
        o0[i], h0[i], l0[i], c0[i], t0[i] = o0_, h0_, l0_, c0_, t0_
        o1[i], h1[i], l1[i], c1[i], t1[i] = o1_, h1_, l1_, c1_, t1_
        h2[i], l2[i], o2[i], c2[i] = h2_, l2_, o2_, c2_
        h3[i], l3[i], o3[i], c3[i] = h3_, l3_, o3_, c3_
        h0bar[i], l0bar[i], h0t[i], l0t[i] = h0bar_, l0bar_, h0t_, l0t_
        h_swept_a[i] = h_swept_now
        l_swept_a[i] = l_swept_now
        h_swept1_a[i] = h_swept1_
        l_swept1_a[i] = l_swept1_
        bull_level_a[i] = bull_level
        bear_level_a[i] = bear_level
        bull_index_a[i] = bull_index
        bear_index_a[i] = bear_index
        xcisd_a[i] = xcisd
        ycisd_a[i] = ycisd
        xbull_a[i] = xbull
        xbear_a[i] = xbear

    # ----- HTF Bias state machine (Pine lines 561-621) -----
    bias_a = [0] * n

    bo0_, bh0_, bl0_, bc0_, bt0_ = opens[0], highs[0], lows[0], closes[0], times[0]
    bo1_, bh1_, bl1_, bc1_, bt1_ = opens[0], highs[0], lows[0], closes[0], times[0]
    bh2_, bl2_, bo2_, bc2_ = highs[0], lows[0], opens[0], closes[0]
    bias = 0

    for i in range(n):
        prev_bo0, prev_bh0, prev_bl0, prev_bt0 = bo0_, bh0_, bl0_, bt0_
        prev_bh1, prev_bl1, prev_bo1, prev_bc1 = bh1_, bl1_, bo1_, bc1_
        prev_close = closes[i - 1] if i > 0 else closes[i]

        if bias_change[i]:
            bt0_ = times[i]
            bo0_ = opens[i]
            bh0_ = highs[i]
            bl0_ = lows[i]
        if highs[i] >= bh0_:
            bh0_ = highs[i]
        if lows[i] <= bl0_:
            bl0_ = lows[i]
        bc0_ = closes[i]

        if bias_change[i]:
            bo1_ = prev_bo0
            bh1_ = prev_bh0
            bl1_ = prev_bl0
            bt1_ = prev_bt0
            bc1_ = prev_close
            bh2_ = prev_bh1
            bl2_ = prev_bl1
            bo2_ = prev_bo1
            bc2_ = prev_bc1

            bias = 0
            if bc1_ > bh2_:
                bias = 1
            if bc1_ < bl2_:
                bias = -1
            if bc1_ < bh2_ and bc1_ > bl2_ and bh1_ > bh2_ and bl1_ > bl2_:
                bias = -1
            if bc1_ > bl2_ and bc1_ < bh2_ and bh1_ < bh2_ and bl1_ < bl2_:
                bias = 1
            if bh1_ <= bh2_ and bl1_ >= bl2_:
                bias = 1 if bc2_ > bo2_ else -1

        bias_a[i] = bias

    bars_out = []
    for i in range(n):
        bars_out.append({
            "i": i,
            "time": times[i],
            "h_swept": h_swept_a[i],
            "l_swept": l_swept_a[i],
            "h_swept1": h_swept1_a[i],
            "l_swept1": l_swept1_a[i],
            "h0": h0[i], "l0": l0[i], "o0": o0[i], "c0": c0[i],
            "h1": h1[i], "l1": l1[i], "o1": o1[i], "c1": c1[i],
            "bull_level": bull_level_a[i],
            "bear_level": bear_level_a[i],
            "bull_index": bull_index_a[i],
            "bear_index": bear_index_a[i],
            "xbull": xbull_a[i],
            "xbear": xbear_a[i],
            "xcisd": xcisd_a[i],
            "ycisd": ycisd_a[i],
            "bias": bias_a[i],
        })

    return {
        "n_bars": n,
        "events": {"xbull": events_bull, "xbear": events_bear},
        "bars": bars_out,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ltf", required=True)
    ap.add_argument("--htf", required=True)
    ap.add_argument("--htfbias", default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    def _load(p: str) -> list[dict[str, Any]]:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "bars" in data:
            return data["bars"]
        return data

    ltf = _load(args.ltf)
    htf_b = _load(args.htf)
    bias_b = _load(args.htfbias) if args.htfbias else htf_b

    out = analyze(ltf, htf_bars=htf_b, htfbias_bars=bias_b)
    if args.summary:
        out = {"n_bars": out["n_bars"], "events": out["events"]}
    json.dump(out, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
