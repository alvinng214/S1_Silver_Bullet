// =============================================================================
// Balanced Price Range - BPR [TFO]mod.cs — C# cTrader port
// Original Pine Script: © tradeforopp (MPL 2.0)
// Modded: extend_right + 50% midline + max_bpr_count cap
// =============================================================================
// Architecture: last-bar-only + ClearObjects + redraw (mirrors the original
//   BPR [TFO] port and Liquidity Sweeps [UAlgo]).
//
// Differences vs. the original BPR [TFO]:
//   1. New input `extend_right` (default true) — when true, maintains a list
//      of BPRs and extends each box's right edge until price *closes* through
//      the zone (close < bottom for bull, close > top for bear). On mitigation
//      the right edge freezes at `bar_index` (NOT `bar_index + 1`).
//   2. New inputs `show_midline` (default true) + `midline_color` — draws a
//      dashed horizontal line at (top+bottom)/2 spanning the box.
//   3. New input `max_bpr_count` (1..100, default 50) — caps the active list;
//      oldest popped on overflow (Pine: array.unshift to front, array.pop from
//      back).
//   4. Box left edge formula changed from `bar_index - bull_num_since - 1`
//      (current-bar bns) to `bar_index - bull_num_since[1] - 2` (prev-bar bns).
//      Equal in the normal case; differs when a new bear FVG happens on the
//      same bar as the draw.
//   5. Mitigation criterion in extend_right mode is `close vs zone edge`,
//      NOT `low/high vs zone edge` as in the original (which still applies in
//      non-extend mode).
//
// Pine state mapping (C# fields):
//   Pine                            C#
//   ----------------------------    -------------------------------------
//   array<BPRData> bullish_bprs   ↔ List<BoxRecord> _activeBull (extend mode)
//   array<BPRData> bearish_bprs   ↔ List<BoxRecord> _activeBear (extend mode)
//   var box box_bullish           ↔ BoxRecord _singleBull       (non-extend)
//   var box box_bearish           ↔ BoxRecord _singleBear       (non-extend)
//   bpr_data.isMitigated          ↔ BoxRecord.IsMitigated
//   line midline                  ↔ BoxRecord.HasMidline + (Top+Bottom)/2
//
// Faithful behavioral parity:
//   - Pine's `array.unshift` (front) + `array.pop` (back) preserved
//   - `for i = array.size-1 to 0` iteration in reverse to allow mid-loop remove
//   - Midline drawn only when `show_midline = true` AT CREATION (Pine quirk:
//     toggling show_midline mid-run does not retroactively add midlines)
//   - alertcondition() has no automatic cTrader equivalent — not implemented
//     (matches Liquidity Sweeps [UAlgo] convention)
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class BalancedPriceRangeTfoMod : Indicator
    {
        private const string Prefix = "BPR_TFO_MOD_";

        // ── Parameters (Pine input order preserved) ───────────────────────────

        [Parameter("BPR Threshold", DefaultValue = 0.0, Step = 0.25)]
        public double BprThreshold { get; set; }

        [Parameter("Bars to Look Back for BPR", DefaultValue = 10, MinValue = 1, MaxValue = 500)]
        public int BarsSince { get; set; }

        [Parameter("Extend Right Until Mitigated", DefaultValue = true)]
        public bool ExtendRight { get; set; }

        [Parameter("Only Clean BPR", DefaultValue = false)]
        public bool OnlyCleanBpr { get; set; }

        [Parameter("Delete Old BPR", DefaultValue = false)]
        public bool DeleteOldBpr { get; set; }

        [Parameter("Bearish BPR Color", DefaultValue = "#4DFF0000")]
        public Color BearishBprColor { get; set; }

        [Parameter("Bullish BPR Color", DefaultValue = "#4D008000")]
        public Color BullishBprColor { get; set; }

        // Pine: color.new(color.black, 0) → fully opaque black
        [Parameter("50% Line Color", DefaultValue = "#FF000000")]
        public Color MidlineColor { get; set; }

        [Parameter("Show 50% Line", DefaultValue = true)]
        public bool ShowMidline { get; set; }

        [Parameter("Max BPR Display Count", DefaultValue = 50, MinValue = 1, MaxValue = 100)]
        public int MaxBprCount { get; set; }

        // Diagnostic parity-check dump — same format as the original [TFO] port:
        //   BPR_DUMP idx=<n> L=<leftBar> R=<rightBar> TOP=<top> BOT=<bottom> COLOR=<0|1>
        // COLOR: 0 = bullish, 1 = bearish.
        [Parameter("Dump Boxes To Log (parity check)", DefaultValue = false)]
        public bool DumpBoxesToLog { get; set; }

        // Diagnostic: dumps every bar where newBull/newBear/bullResult/bearResult
        // is true with full OHLC + flags. Use to cross-check cTrader's bar feed
        // against a CSV / TradingView feed when a BPR is missing or extra.
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
            public bool    HasMidline;
            public bool    IsMitigated;
        }

        // ── Private fields ────────────────────────────────────────────────────

        private int _objId;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize() { }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1) return;

            ClearOwned();
            _objId = 0;

            int N = Bars.Count;
            if (N < 4) return;

            // ── Step 1: per-bar FVG flags ─────────────────────────────────────
            var newBear = new bool[N];
            var newBull = new bool[N];
            for (int i = 2; i < N; i++)
            {
                newBear[i] = Bars.LowPrices [i - 2] - Bars.HighPrices[i] > 0.0;
                newBull[i] = Bars.LowPrices [i]     - Bars.HighPrices[i - 2] > 0.0;
            }

            // ── Step 2: ta.barssince ──────────────────────────────────────────
            var bullNumSince = new int?[N];
            var bearNumSince = new int?[N];
            for (int i = 0; i < N; i++)
            {
                if (newBear[i])                                           bullNumSince[i] = 0;
                else if (i > 0 && bullNumSince[i - 1].HasValue)           bullNumSince[i] = bullNumSince[i - 1].Value + 1;

                if (newBull[i])                                           bearNumSince[i] = 0;
                else if (i > 0 && bearNumSince[i - 1].HasValue)           bearNumSince[i] = bearNumSince[i - 1].Value + 1;
            }

            // ── Step 3: per-bar BPR result + combined edges ───────────────────
            var bullResult       = new bool  [N];
            var bullCombinedLow  = new double[N];
            var bullCombinedHigh = new double[N];

            var bearResult       = new bool  [N];
            var bearCombinedLow  = new double[N];
            var bearCombinedHigh = new double[N];

            for (int i = 0; i < N; i++)
            {
                ComputeBull(i, bullNumSince, bullResult, bullCombinedLow, bullCombinedHigh);
                ComputeBear(i, bearNumSince, bearResult, bearCombinedLow, bearCombinedHigh);
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

            // ── Step 4: simulate Pine's bar-by-bar drawing/tracking ───────────
            var bullList    = new List<BoxRecord>();   // newest at index 0 (Pine unshift)
            var bearList    = new List<BoxRecord>();
            var historyBull = new List<BoxRecord>();   // for non-extend mode
            var historyBear = new List<BoxRecord>();
            BoxRecord singleBull = null;
            BoxRecord singleBear = null;

            for (int i = 0; i < N; i++)
            {
                // ── Bullish BPR creation (Pine: if bull_result[1]) ────────────
                if (i >= 1 && bullResult[i - 1])
                {
                    int? bnsPrev = bullNumSince[i - 1];               // Pine: bull_num_since[1]
                    if (bnsPrev.HasValue)
                    {
                        int leftBar = i - bnsPrev.Value - 2;
                        if (leftBar < 0) leftBar = 0;
                        var rec = new BoxRecord
                        {
                            LeftBar     = leftBar,
                            RightBar    = i + 1,
                            Top         = bullCombinedHigh[i - 1],
                            Bottom      = bullCombinedLow [i - 1],
                            Color       = BullishBprColor,
                            IsBull      = true,
                            HasMidline  = ShowMidline,
                            IsMitigated = false
                        };

                        if (ExtendRight)
                        {
                            bullList.Insert(0, rec);                  // unshift
                            if (bullList.Count > MaxBprCount)
                                bullList.RemoveAt(bullList.Count - 1);// pop oldest
                        }
                        else
                        {
                            // Original behavior: replace single tracked box
                            if (singleBull != null)
                            {
                                if (DeleteOldBpr) { /* drop */ }
                                else              { historyBull.Add(singleBull); }
                            }
                            singleBull = rec;
                        }
                    }
                }

                // ── Bullish BPR per-bar update ────────────────────────────────
                if (ExtendRight)
                {
                    // Pine: for i = array.size-1 to 0 (reverse for safe removal)
                    for (int k = bullList.Count - 1; k >= 0; k--)
                    {
                        var bpr = bullList[k];
                        if (bpr.IsMitigated) continue;

                        double cl = Bars.ClosePrices[i];
                        if (cl < bpr.Bottom)
                        {
                            // Mitigated — freeze right at bar_index (NOT +1)
                            bpr.IsMitigated = true;
                            bpr.RightBar    = i;
                            if (DeleteOldBpr) bullList.RemoveAt(k);
                        }
                        else
                        {
                            bpr.RightBar = i + 1;
                        }
                    }
                }
                else if (singleBull != null)
                {
                    // Original mode — uses LOW vs bottom (not close)
                    double lo = Bars.LowPrices[i];
                    if (lo > singleBull.Bottom)
                    {
                        singleBull.RightBar = i + 1;
                    }
                    else if (lo < singleBull.Bottom)
                    {
                        if (DeleteOldBpr) { /* drop */ }
                        else              { historyBull.Add(singleBull); }
                        singleBull = null;
                    }
                    // lo == bottom → no change
                }

                // ── Bearish BPR creation ──────────────────────────────────────
                if (i >= 1 && bearResult[i - 1])
                {
                    int? bnsPrev = bearNumSince[i - 1];
                    if (bnsPrev.HasValue)
                    {
                        int leftBar = i - bnsPrev.Value - 2;
                        if (leftBar < 0) leftBar = 0;
                        var rec = new BoxRecord
                        {
                            LeftBar     = leftBar,
                            RightBar    = i + 1,
                            Top         = bearCombinedHigh[i - 1],
                            Bottom      = bearCombinedLow [i - 1],
                            Color       = BearishBprColor,
                            IsBull      = false,
                            HasMidline  = ShowMidline,
                            IsMitigated = false
                        };

                        if (ExtendRight)
                        {
                            bearList.Insert(0, rec);
                            if (bearList.Count > MaxBprCount)
                                bearList.RemoveAt(bearList.Count - 1);
                        }
                        else
                        {
                            if (singleBear != null)
                            {
                                if (DeleteOldBpr) { /* drop */ }
                                else              { historyBear.Add(singleBear); }
                            }
                            singleBear = rec;
                        }
                    }
                }

                // ── Bearish BPR per-bar update ────────────────────────────────
                if (ExtendRight)
                {
                    for (int k = bearList.Count - 1; k >= 0; k--)
                    {
                        var bpr = bearList[k];
                        if (bpr.IsMitigated) continue;

                        double cl = Bars.ClosePrices[i];
                        if (cl > bpr.Top)
                        {
                            bpr.IsMitigated = true;
                            bpr.RightBar    = i;
                            if (DeleteOldBpr) bearList.RemoveAt(k);
                        }
                        else
                        {
                            bpr.RightBar = i + 1;
                        }
                    }
                }
                else if (singleBear != null)
                {
                    double hi = Bars.HighPrices[i];
                    if (hi < singleBear.Top)
                    {
                        singleBear.RightBar = i + 1;
                    }
                    else if (hi > singleBear.Top)
                    {
                        if (DeleteOldBpr) { /* drop */ }
                        else              { historyBear.Add(singleBear); }
                        singleBear = null;
                    }
                }
            }

            // ── Step 5: render ────────────────────────────────────────────────
            int dumpIdx = 0;
            if (ExtendRight)
            {
                foreach (var b in bullList) { DrawBox(b); MaybeDump(b, dumpIdx++); }
                foreach (var b in bearList) { DrawBox(b); MaybeDump(b, dumpIdx++); }
            }
            else
            {
                foreach (var b in historyBull) { DrawBox(b); MaybeDump(b, dumpIdx++); }
                foreach (var b in historyBear) { DrawBox(b); MaybeDump(b, dumpIdx++); }
                if (singleBull != null) { DrawBox(singleBull); MaybeDump(singleBull, dumpIdx++); }
                if (singleBear != null) { DrawBox(singleBear); MaybeDump(singleBear, dumpIdx++); }
            }

            if (DumpBoxesToLog)
                Print(string.Format(CultureInfo.InvariantCulture,
                    "BPR_DUMP_TOTAL count={0} barCount={1} lastBarUtc={2:yyyy-MM-ddTHH:mm:ssZ} extendRight={3}",
                    dumpIdx, Bars.Count, Bars.OpenTimes[Bars.Count - 1], ExtendRight));
        }

        // ── Bullish per-bar BPR computation ───────────────────────────────────
        private void ComputeBull(
            int i, int?[] bullNumSince,
            bool[] result, double[] combinedLow, double[] combinedHigh)
        {
            // new_fvg_bullish at bar i
            if (i < 2) return;
            bool nfBull = Bars.LowPrices[i] - Bars.HighPrices[i - 2] > 0.0;
            if (!nfBull) return;

            int? bnsOpt = bullNumSince[i];
            if (!bnsOpt.HasValue) return;
            int bns = bnsOpt.Value;
            if (bns > BarsSince) return;

            int iA = i - bns;
            int iB = i - bns - 2;
            int iC = i - 2;
            if (iA < 0 || iB < 0 || iC < 0) return;

            double hiA = Bars.HighPrices[iA];
            double loB = Bars.LowPrices [iB];
            double hiC = Bars.HighPrices[iC];
            double lo0 = Bars.LowPrices [i];

            // Pine cond_2 (mirrored verbatim)
            double sum  = hiA + loB + hiC + lo0;
            double diff = Math.Max(loB, lo0) - Math.Min(hiA, hiC);
            if (!(sum > diff)) return;

            double cLow  = Math.Max(hiA, hiC);
            double cHigh = Math.Min(loB, lo0);

            if (OnlyCleanBpr)
            {
                for (int h = 2; h <= bns; h++)
                {
                    int j = i - h;
                    if (j < 0) continue;
                    if (Bars.HighPrices[j] > cLow) return;
                }
            }

            if (cHigh - cLow < BprThreshold) return;

            result      [i] = true;
            combinedLow [i] = cLow;
            combinedHigh[i] = cHigh;
        }

        // ── Bearish per-bar BPR computation ───────────────────────────────────
        private void ComputeBear(
            int i, int?[] bearNumSince,
            bool[] result, double[] combinedLow, double[] combinedHigh)
        {
            if (i < 2) return;
            bool nfBear = Bars.LowPrices[i - 2] - Bars.HighPrices[i] > 0.0;
            if (!nfBear) return;

            int? bnsOpt = bearNumSince[i];
            if (!bnsOpt.HasValue) return;
            int bns = bnsOpt.Value;
            if (bns > BarsSince) return;

            int iA = i - bns;
            int iB = i - bns - 2;
            int iC = i - 2;
            if (iA < 0 || iB < 0 || iC < 0) return;

            double hiA  = Bars.HighPrices[iA];
            double loA  = Bars.LowPrices [iA];
            double hiB  = Bars.HighPrices[iB];
            double loB  = Bars.LowPrices [iB];
            double loC  = Bars.LowPrices [iC];
            double hiC  = Bars.HighPrices[iC];
            double hi0  = Bars.HighPrices[i];
            double lo0  = Bars.LowPrices [i];

            // Pine cond_2 (mirrored verbatim — same expression as bull)
            double sum  = hiA + loB + hiC + lo0;
            double diff = Math.Max(loB, lo0) - Math.Min(hiA, hiC);
            if (!(sum > diff)) return;

            // bear_combined_low  = max(high[bear_num_since + 2], high)
            // bear_combined_high = min(low [bear_num_since],     low [2])
            double cLow  = Math.Max(hiB, hi0);
            double cHigh = Math.Min(loA, loC);

            if (OnlyCleanBpr)
            {
                for (int h = 2; h <= bns; h++)
                {
                    int j = i - h;
                    if (j < 0) continue;
                    if (Bars.LowPrices[j] < cHigh) return;
                }
            }

            if (cHigh - cLow < BprThreshold) return;

            result      [i] = true;
            combinedLow [i] = cLow;
            combinedHigh[i] = cHigh;
        }

        // ── Drawing ───────────────────────────────────────────────────────────
        private void DrawBox(BoxRecord b)
        {
            DateTime t1 = BarTime(b.LeftBar);
            DateTime t2 = BarTime(b.RightBar);
            string side = b.IsBull ? "bull_" : "bear_";

            string boxName = Prefix + side + "box_" + NextId();
            var rect = Chart.DrawRectangle(boxName, t1, b.Top, t2, b.Bottom, b.Color, 1, LineStyle.Solid);
            rect.IsFilled = true;

            if (b.HasMidline)
            {
                double mid = (b.Top + b.Bottom) / 2.0;
                string lineName = Prefix + side + "mid_" + NextId();
                Chart.DrawTrendLine(lineName, t1, mid, t2, mid, MidlineColor, 1, LineStyle.Lines);
            }
        }

        private void MaybeDump(BoxRecord b, int idx)
        {
            if (!DumpBoxesToLog) return;
            Print(string.Format(CultureInfo.InvariantCulture,
                "BPR_DUMP idx={0} L={1} R={2} TOP={3} BOT={4} COLOR={5} MIT={6}",
                idx, b.LeftBar, b.RightBar, b.Top, b.Bottom, b.IsBull ? 0 : 1, b.IsMitigated ? 1 : 0));
        }

        // ── Helpers ───────────────────────────────────────────────────────────

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
