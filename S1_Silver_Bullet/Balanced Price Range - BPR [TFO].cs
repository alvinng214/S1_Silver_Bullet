// =============================================================================
// Balanced Price Range - BPR [TFO].cs — C# cTrader port
// Original Pine Script: © tradeforopp (MPL 2.0)
// =============================================================================
// Architecture note:
//   Last-bar-only + ClearObjects + redraw pattern (matches Liquidity Sweeps
//   [UAlgo].cs and the TradingFinder BPR port). Reason: cTrader's Chart.Draw*
//   methods do NOT update objects in-place when called repeatedly with the same
//   name — calling DrawRectangle each tick creates duplicates.
//
//   Fix: on every Calculate call, clear all owned objects, then simulate Pine's
//   bar-by-bar state machine from bar 0 to the last bar, tracking active and
//   historical (frozen) boxes, then draw every survivor in a single pass.
//
// Pine logic faithfully reproduced:
//   FVG (3-bar)        : bear → low[i-2] - high[i] > 0;  bull → low[i] - high[i-2] > 0
//   Bull BPR cond_1    : new_fvg_bullish AND bars_since_last_bear_fvg <= bars_since
//   Bull BPR cond_2    : sum-of-4-prices > max-low - min-high (mirrored verbatim
//                        — algebraically near-tautological; the real overlap test
//                        is the threshold check on (combined_high - combined_low))
//   Bull combined_low  : max(high[bull_num_since],     high[2])  — top of bear FVG ∪ top of bull FVG (lower edges)
//   Bull combined_high : min(low [bull_num_since+2],   low [0])  — upper edges
//   Bull cond_3        : if only_clean_bpr → no high in [2..bull_num_since] may close above combined_low
//   Bull result        : cond_1 AND cond_2 AND cond_3 AND (combined_high - combined_low >= threshold)
//   Drawing            : on bull_result[1] → box from (bar_index - bull_num_since - 1, combined_high[1])
//                        to (bar_index + 1, combined_low[1])
//   Box tracking       : low > bottom → extend right;  low < bottom → invalidate
//                        (delete if delete_old_bpr=true, else freeze)
//   Bearish BPR        : symmetric mirror (high replaces low; box top instead of bottom)
//
// Faithful behavioral parity:
//   - bull_num_since uses CURRENT bar's value at draw time (Pine quirk:
//     `box.new(bar_index - bull_num_since - 1, ...)` — bull_num_since is NOT [1])
//   - combined_high[1] / combined_low[1] use prev-bar values (the bar where the
//     result fired)
//   - bar_index + 1 right edge points one bar into the future; we synthesise the
//     time via period-extrapolation when no future bar exists
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class BalancedPriceRangeTfo : Indicator
    {
        private const string Prefix = "BPR_TFO_";

        // ── Parameters (Pine input order preserved) ───────────────────────────

        [Parameter("BPR Threshold", DefaultValue = 0.0, Step = 0.25)]
        public double BprThreshold { get; set; }

        [Parameter("Bars to Look Back for BPR", DefaultValue = 10, MinValue = 1, MaxValue = 500)]
        public int BarsSince { get; set; }

        [Parameter("Only Clean BPR", DefaultValue = false)]
        public bool OnlyCleanBpr { get; set; }

        [Parameter("Delete Old BPR", DefaultValue = false)]
        public bool DeleteOldBpr { get; set; }

        // Pine: color.new(color.red,   70) → alpha ≈ 0x4D, RGB = #FF0000
        [Parameter("Bearish BPR Color", DefaultValue = "#4DFF0000")]
        public Color BearishBprColor { get; set; }

        // Pine: color.new(color.green, 70) → alpha ≈ 0x4D, RGB = #008000 (Pine green, NOT #00FF00)
        [Parameter("Bullish BPR Color", DefaultValue = "#4D008000")]
        public Color BullishBprColor { get; set; }

        // Diagnostic: when true, every drawn box is emitted to the cTrader log
        // in the exact format the TradingView MCP parity script expects:
        //   BPR_DUMP idx=<n> L=<leftBar> R=<rightBar> TOP=<top> BOT=<bottom> COLOR=<0|1>
        // COLOR=0 = bullish, COLOR=1 = bearish (matches TV's borderColor/bgColor scheme).
        [Parameter("Dump Boxes To Log (parity check)", DefaultValue = false)]
        public bool DumpBoxesToLog { get; set; }

        // Diagnostic: when true, every bar where newBull/newBear/bullResult/bearResult
        // is true is dumped to the log with full OHLC + flags. Lets you cross-check
        // cTrader's bar data against a CSV / TradingView feed, bar-by-bar:
        //   BPR_BAR i=<idx> t=<utc-iso> O=<o> H=<h> L=<l> C=<c>  newB=<0|1> newU=<0|1>  brZ=<0|1> blZ=<0|1>  cLo=<x> cHi=<x>  bullNS=<n> bearNS=<n>
        // (newB=newBear, newU=newBull, brZ=bearResult, blZ=bullResult)
        [Parameter("Dump Bar Diagnostics (FVG/BPR per-bar)", DefaultValue = false)]
        public bool DumpBarDiagnostics { get; set; }

        // ── Internal types ────────────────────────────────────────────────────

        private sealed class BoxRecord
        {
            public int     LeftBar;
            public int     RightBar;
            public double  Top;
            public double  Bottom;
            public Color   Color;
            public bool    IsBull;
        }

        // ── Private fields ────────────────────────────────────────────────────

        private int _objId;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize() { /* no per-bar state needed */ }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1) return;

            ClearOwned();
            _objId = 0;

            int N = Bars.Count;
            if (N < 4) return;

            // ── Step 1: pre-compute per-bar FVG flags ─────────────────────────
            var newBear = new bool[N];
            var newBull = new bool[N];
            for (int i = 2; i < N; i++)
            {
                newBear[i] = Bars.LowPrices [i - 2] - Bars.HighPrices[i] > 0.0;
                newBull[i] = Bars.LowPrices [i]     - Bars.HighPrices[i - 2] > 0.0;
            }

            // ── Step 2: ta.barssince — int? mirrors Pine's na ─────────────────
            var bullNumSince = new int?[N]; // bars since last new_fvg_bearish
            var bearNumSince = new int?[N]; // bars since last new_fvg_bullish
            for (int i = 0; i < N; i++)
            {
                if (newBear[i])                  bullNumSince[i] = 0;
                else if (i > 0 && bullNumSince[i - 1].HasValue) bullNumSince[i] = bullNumSince[i - 1].Value + 1;

                if (newBull[i])                  bearNumSince[i] = 0;
                else if (i > 0 && bearNumSince[i - 1].HasValue) bearNumSince[i] = bearNumSince[i - 1].Value + 1;
            }

            // ── Step 3: per-bar BPR state ─────────────────────────────────────
            var bullResult       = new bool  [N];
            var bullCombinedLow  = new double[N];
            var bullCombinedHigh = new double[N];

            var bearResult       = new bool  [N];
            var bearCombinedLow  = new double[N];
            var bearCombinedHigh = new double[N];

            for (int i = 0; i < N; i++)
            {
                ComputeBull(i, newBull, bullNumSince, bullResult, bullCombinedLow, bullCombinedHigh);
                ComputeBear(i, newBear, bearNumSince, bearResult, bearCombinedLow, bearCombinedHigh);
            }

            // ── Step 3b: per-bar diagnostic dump ─────────────────────────────
            if (DumpBarDiagnostics)
            {
                for (int i = 0; i < N; i++)
                {
                    if (!(newBear[i] || newBull[i] || bullResult[i] || bearResult[i])) continue;
                    string bullNs = bullNumSince[i].HasValue ? bullNumSince[i].Value.ToString(CultureInfo.InvariantCulture) : "na";
                    string bearNs = bearNumSince[i].HasValue ? bearNumSince[i].Value.ToString(CultureInfo.InvariantCulture) : "na";
                    Print(string.Format(CultureInfo.InvariantCulture,
                        "BPR_BAR i={0} t={1:yyyy-MM-ddTHH:mm:ssZ} O={2} H={3} L={4} C={5}  newB={6} newU={7}  brZ={8} blZ={9}  bullCLo={10} bullCHi={11} bearCLo={12} bearCHi={13}  bullNS={14} bearNS={15}",
                        i, Bars.OpenTimes[i],
                        Bars.OpenPrices[i], Bars.HighPrices[i], Bars.LowPrices[i], Bars.ClosePrices[i],
                        newBear[i] ? 1 : 0, newBull[i] ? 1 : 0,
                        bearResult[i] ? 1 : 0, bullResult[i] ? 1 : 0,
                        bullCombinedLow[i], bullCombinedHigh[i],
                        bearCombinedLow[i], bearCombinedHigh[i],
                        bullNs, bearNs));
                }
            }

            // ── Step 4: simulate Pine's drawing/tracking state machine ────────
            var history = new List<BoxRecord>();
            BoxRecord activeBull = null;
            BoxRecord activeBear = null;

            for (int i = 0; i < N; i++)
            {
                // ── Bullish BPR creation (Pine: if bull_result[1]) ────────────
                if (i >= 1 && bullResult[i - 1])
                {
                    int bns = bullNumSince[i] ?? -1;     // Pine quirk: uses CURRENT bar's bull_num_since
                    if (bns >= 0)
                    {
                        // Pine: if delete_old_bpr and not na(box_bullish) → box.delete(box_bullish)
                        if (activeBull != null)
                        {
                            if (DeleteOldBpr) { /* drop silently */ }
                            else              { history.Add(activeBull); }
                            activeBull = null;
                        }
                        activeBull = new BoxRecord
                        {
                            LeftBar  = i - bns - 1,
                            RightBar = i + 1,
                            Top      = bullCombinedHigh[i - 1],
                            Bottom   = bullCombinedLow [i - 1],
                            Color    = BullishBprColor,
                            IsBull   = true
                        };
                    }
                }

                // ── Bullish BPR tracking (Pine post-creation block) ───────────
                if (activeBull != null)
                {
                    double lo = Bars.LowPrices[i];
                    if (lo > activeBull.Bottom)
                    {
                        activeBull.RightBar = i + 1;
                    }
                    else if (lo < activeBull.Bottom)
                    {
                        if (DeleteOldBpr) { /* delete silently */ }
                        else              { history.Add(activeBull); }
                        activeBull = null;
                    }
                    // lo == bottom exactly → no change (Pine's strict < / > semantics)
                }

                // ── Bearish BPR creation ──────────────────────────────────────
                if (i >= 1 && bearResult[i - 1])
                {
                    int bns = bearNumSince[i] ?? -1;
                    if (bns >= 0)
                    {
                        if (activeBear != null)
                        {
                            if (DeleteOldBpr) { }
                            else              { history.Add(activeBear); }
                            activeBear = null;
                        }
                        activeBear = new BoxRecord
                        {
                            LeftBar  = i - bns - 1,
                            RightBar = i + 1,
                            Top      = bearCombinedHigh[i - 1],
                            Bottom   = bearCombinedLow [i - 1],
                            Color    = BearishBprColor,
                            IsBull   = false
                        };
                    }
                }

                // ── Bearish BPR tracking ──────────────────────────────────────
                if (activeBear != null)
                {
                    double hi = Bars.HighPrices[i];
                    if (hi < activeBear.Top)
                    {
                        activeBear.RightBar = i + 1;
                    }
                    else if (hi > activeBear.Top)
                    {
                        if (DeleteOldBpr) { }
                        else              { history.Add(activeBear); }
                        activeBear = null;
                    }
                }
            }

            // ── Step 5: render ────────────────────────────────────────────────
            int dumpIdx = 0;
            foreach (BoxRecord b in history) { DrawBox(b); MaybeDump(b, dumpIdx++); }
            if (activeBull != null) { DrawBox(activeBull); MaybeDump(activeBull, dumpIdx++); }
            if (activeBear != null) { DrawBox(activeBear); MaybeDump(activeBear, dumpIdx++); }

            if (DumpBoxesToLog)
                Print(string.Format(CultureInfo.InvariantCulture,
                    "BPR_DUMP_TOTAL count={0} barCount={1} lastBarUtc={2:yyyy-MM-ddTHH:mm:ssZ}",
                    dumpIdx, Bars.Count, Bars.OpenTimes[Bars.Count - 1]));
        }

        private void MaybeDump(BoxRecord b, int idx)
        {
            if (!DumpBoxesToLog) return;
            // Format mirrors the TV-side parity output exactly.
            // COLOR: 0 = bullish, 1 = bearish.
            Print(string.Format(CultureInfo.InvariantCulture,
                "BPR_DUMP idx={0} L={1} R={2} TOP={3} BOT={4} COLOR={5}",
                idx, b.LeftBar, b.RightBar, b.Top, b.Bottom, b.IsBull ? 0 : 1));
        }

        // ── Bullish per-bar computation ───────────────────────────────────────
        private void ComputeBull(
            int i, bool[] newBull, int?[] bullNumSince,
            bool[] result, double[] combinedLow, double[] combinedHigh)
        {
            if (!newBull[i])                       return;
            int? bnsOpt = bullNumSince[i];
            if (!bnsOpt.HasValue)                  return;
            int bns = bnsOpt.Value;
            if (bns > BarsSince)                   return;

            int  iA = i - bns;       // bear FVG bar (bar of new_fvg_bearish)
            int  iB = i - bns - 2;   // 2 bars before the bear FVG bar
            int  iC = i - 2;
            if (iA < 0 || iB < 0 || iC < 0)        return;

            double hiA = Bars.HighPrices[iA];
            double loB = Bars.LowPrices [iB];
            double hiC = Bars.HighPrices[iC];
            double lo0 = Bars.LowPrices [i];

            // Pine cond_2 — mirrored verbatim (algebraically near-tautological)
            double sum  = hiA + loB + hiC + lo0;
            double diff = Math.Max(loB, lo0) - Math.Min(hiA, hiC);
            bool   cond2 = sum > diff;
            if (!cond2)                            return;

            double cLow  = Math.Max(hiA, hiC);
            double cHigh = Math.Min(loB, lo0);

            // cond_3: only_clean_bpr → no high[h] for h in [2..bns] may exceed cLow
            if (OnlyCleanBpr)
            {
                for (int h = 2; h <= bns; h++)
                {
                    int j = i - h;
                    if (j < 0) continue;
                    if (Bars.HighPrices[j] > cLow) return;
                }
            }

            if (cHigh - cLow < BprThreshold)       return;

            result      [i] = true;
            combinedLow [i] = cLow;
            combinedHigh[i] = cHigh;
        }

        // ── Bearish per-bar computation ───────────────────────────────────────
        private void ComputeBear(
            int i, bool[] newBear, int?[] bearNumSince,
            bool[] result, double[] combinedLow, double[] combinedHigh)
        {
            if (!newBear[i])                       return;
            int? bnsOpt = bearNumSince[i];
            if (!bnsOpt.HasValue)                  return;
            int bns = bnsOpt.Value;
            if (bns > BarsSince)                   return;

            int  iA = i - bns;       // bull FVG bar (bar of new_fvg_bullish)
            int  iB = i - bns - 2;   // bull FVG's "high[2]" — bar 2 before iA
            int  iC = i - 2;
            if (iA < 0 || iB < 0 || iC < 0)        return;

            double hiA  = Bars.HighPrices[iA];   // high[bear_num_since]
            double loA  = Bars.LowPrices [iA];   // low [bear_num_since]
            double hiB  = Bars.HighPrices[iB];   // high[bear_num_since + 2]
            double loC  = Bars.LowPrices [iC];   // low [2]
            double hi0  = Bars.HighPrices[i];    // high (current)

            // Pine cond_2 (note: Pine uses high[bear_num_since] + low[bear_num_since+2] + high[2] + low
            //                — verbatim mirror, even though combined_low/high pull different bars)
            double low2  = Bars.LowPrices [iB];                    // low [bear_num_since + 2]
            double low0  = Bars.LowPrices [i];                     // low (current)
            double high2 = Bars.HighPrices[iC];                    // high[2]
            double sum   = hiA + low2 + high2 + low0;
            double diff  = Math.Max(low2, low0) - Math.Min(hiA, high2);
            bool   cond2 = sum > diff;
            if (!cond2)                            return;

            // bear_combined_low  = max(high[bear_num_since + 2], high)
            // bear_combined_high = min(low [bear_num_since],     low [2])
            double cLow  = Math.Max(hiB, hi0);
            double cHigh = Math.Min(loA, loC);

            // cond_3: only_clean_bpr → no low[h] for h in [2..bns] may dip below cHigh
            if (OnlyCleanBpr)
            {
                for (int h = 2; h <= bns; h++)
                {
                    int j = i - h;
                    if (j < 0) continue;
                    if (Bars.LowPrices[j] < cHigh) return;
                }
            }

            if (cHigh - cLow < BprThreshold)       return;

            result      [i] = true;
            combinedLow [i] = cLow;
            combinedHigh[i] = cHigh;
        }

        // ── Drawing ───────────────────────────────────────────────────────────
        private void DrawBox(BoxRecord b)
        {
            DateTime t1 = BarTime(b.LeftBar);
            DateTime t2 = BarTime(b.RightBar);
            string name = Prefix + (b.IsBull ? "bull_" : "bear_") + NextId();
            var rect = Chart.DrawRectangle(name, t1, b.Top, t2, b.Bottom, b.Color, 1, LineStyle.Solid);
            rect.IsFilled = true;
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        // Map a (possibly-future) bar index to a chart time.
        // For barIdx == Bars.Count we extrapolate one period ahead using the
        // last known bar spacing — this lets us honour Pine's `bar_index + 1`
        // right-edge convention even on the last bar.
        private DateTime BarTime(int barIdx)
        {
            if (barIdx < 0) barIdx = 0;
            int last = Bars.Count - 1;
            if (barIdx <= last) return Bars.OpenTimes[barIdx];

            if (last >= 1)
            {
                TimeSpan step = Bars.OpenTimes[last] - Bars.OpenTimes[last - 1];
                long ticks = step.Ticks * (barIdx - last);
                return Bars.OpenTimes[last] + TimeSpan.FromTicks(ticks);
            }
            return Bars.OpenTimes[last];
        }

        private string NextId()
            => (_objId++).ToString(CultureInfo.InvariantCulture);

        private void ClearOwned()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(obj.Name);
            foreach (var n in names)
                Chart.RemoveObject(n);
        }
    }
}
