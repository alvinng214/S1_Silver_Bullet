// =============================================================================
// Liquidity Sweeps [UAlgo].cs — C# cTrader port
// Original Pine Script: © UAlgo (MPL 2.0)
// =============================================================================
// Architecture note:
//   Uses the "last-bar-only + ClearObjects + redraw" pattern (same as ICT Sessions).
//   Reason: cTrader's Chart.DrawTrendLine() creates a NEW object each call even
//   when the same name is passed — it does NOT update x2 in-place. The original
//   incremental approach (calling DrawTrendLine every bar to extend x2) produced
//   thousands of overlapping objects, causing rendering failure / nothing visible.
//
//   Fix: on every Calculate call only process the last bar. Simulate Pine's full
//   stateful logic by scanning bars 0..N-1, then draw the final state in one pass.
//
// Pine logic faithfully reproduced:
//   Pivot detection : ta.pivothigh/low with strict inequality (no ties)
//   S/R cap         : oldest removed when count > maxLine (Pine array.remove index 0)
//   Resistance sweep: high > level AND close < level  → diagonal line + label
//   Resistance break: close > level                   → silent removal
//   Support sweep   : low  < level AND close > level  → diagonal line + label
//   Support break   : close < level                   → silent removal
//   Sweep line      : diagonal from (pivot origin bar, pivot price) → (sweep bar, wick)
//   ATR             : Wilder's RMA ≈ cTrader Exponential, period 14 — label Y-offset only
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class LiquiditySweepsUAlgo : Indicator
    {
        private const string Prefix = "LSUA_";

        // Internal state for one S/R level (used only during the per-tick simulation)
        private struct ActiveLevel
        {
            public double   Price;
            public DateTime OriginUtc; // UTC open time of the pivot bar
            public ActiveLevel(double price, DateTime origin)
            { Price = price; OriginUtc = origin; }
        }

        // ── Parameters ────────────────────────────────────────────────────────

        [Parameter("Pivot Length", Group = "S/R Settings", DefaultValue = 20, MinValue = 3, MaxValue = 100)]
        public int PivotPeriod { get; set; }

        [Parameter("Maximum Lines", Group = "S/R Settings", DefaultValue = 3, MinValue = 1, MaxValue = 100)]
        public int MaxLines { get; set; }

        // Pine: color.rgb(255,82,82, transp=53) → alpha = round((1-0.53)*255) = 0x78
        [Parameter("Resistance Color", Group = "Colors", DefaultValue = "#78FF5252")]
        public Color ResistanceColor { get; set; }

        // Pine: color.rgb(76,175,79, transp=32) → alpha = round((1-0.32)*255) = 0xAD
        [Parameter("Support Color", Group = "Colors", DefaultValue = "#AD4CAF4F")]
        public Color SupportColor { get; set; }

        // Pine allows 0 but cTrader minimum is 1
        [Parameter("Line Width", Group = "Colors", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int LineWidth { get; set; }

        // Pine: hardcoded color.purple for buy sweep line
        [Parameter("Buy Sweep Color", Group = "Sweep", DefaultValue = "#FF800080")]
        public Color BuySweepColor { get; set; }

        // Pine: hardcoded color.blue for sell sweep line
        [Parameter("Sell Sweep Color", Group = "Sweep", DefaultValue = "#FF0000FF")]
        public Color SellSweepColor { get; set; }

        // Pine: hardcoded width=1 for sweep lines
        [Parameter("Sweep Line Width", Group = "Sweep", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int SweepLineWidth { get; set; }

        // Pine: label color=color.purple, textcolor=color.white for buy sweep
        [Parameter("Buy Label Color", Group = "Labels", DefaultValue = "#FF800080")]
        public Color BuyLabelColor { get; set; }

        // Pine: label color=na (default blue), textcolor=color.white for sell sweep
        [Parameter("Sell Label Color", Group = "Labels", DefaultValue = "#FFFFFFFF")]
        public Color SellLabelColor { get; set; }

        // ── Private fields ────────────────────────────────────────────────────

        private AverageTrueRange _atr;
        private int              _objId;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            // Pine: atrOffset = ta.atr(14). Wilder's RMA ≈ Exponential in cTrader.
            // Used only for label Y-offset — minor approximation is acceptable.
            _atr = Indicators.AverageTrueRange(14, MovingAverageType.Exponential);
        }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            // Last-bar-only pattern: clear everything and rebuild each tick.
            // This avoids the cTrader object duplication issue caused by calling
            // DrawTrendLine with an existing name (does NOT update x2 in-place).
            if (index != Bars.Count - 1) return;

            ClearObjects();
            _objId = 0;

            int minStart = PivotPeriod * 2;
            if (index < minStart) return;

            // Active S/R levels — simulated from bar 0 forward (exactly as Pine does)
            var activeRes = new List<ActiveLevel>();
            var activeSup = new List<ActiveLevel>();

            for (int i = minStart; i <= index; i++)
            {
                DateTime utcBar = Bars.OpenTimes[i];
                double   hi     = Bars.HighPrices[i];
                double   lo     = Bars.LowPrices[i];
                double   cl     = Bars.ClosePrices[i];
                bool     closed = i < Bars.Count - 1; // forming bar not yet closed

                // ATR for label offset — Pine: atrOffset at current bar
                double atr = (i >= 14 && !double.IsNaN(_atr.Result[i]))
                    ? _atr.Result[i] : 0.0;

                // ── New pivot high ────────────────────────────────────────────
                // Pine: if ph → add to resistanceArray, cap to maxLine
                if (closed && IsPivotHigh(i))
                {
                    int      pb = i - PivotPeriod;
                    activeRes.Add(new ActiveLevel(Bars.HighPrices[pb], Bars.OpenTimes[pb]));
                    // Pine: if array.size > maxLine → delete+remove oldest (index 0)
                    while (activeRes.Count > MaxLines)
                        activeRes.RemoveAt(0);
                }

                // ── New pivot low ─────────────────────────────────────────────
                if (closed && IsPivotLow(i))
                {
                    int      pb = i - PivotPeriod;
                    activeSup.Add(new ActiveLevel(Bars.LowPrices[pb], Bars.OpenTimes[pb]));
                    while (activeSup.Count > MaxLines)
                        activeSup.RemoveAt(0);
                }

                // Sweep/break only on closed bars (Pine fires on bar close)
                if (!closed) continue;

                // ── Resistance: sweep / break ─────────────────────────────────
                // Pine: for i = array.size-1 to 0 (reverse — safe with removal)
                for (int r = activeRes.Count - 1; r >= 0; r--)
                {
                    ActiveLevel lvl = activeRes[r];

                    if (hi > lvl.Price && cl < lvl.Price)
                    {
                        // Pine: high > highPrice AND close < highPrice
                        // → wick swept above level, close rejected below
                        // → diagonal line: (origin, level price) → (this bar, high)
                        Chart.DrawTrendLine(Prefix + "swp_" + NextId(),
                            lvl.OriginUtc, lvl.Price, utcBar, hi,
                            BuySweepColor, SweepLineWidth, LineStyle.Solid);
                        // Pine: label.new(bar_index, highPrice + atrOffset, "Buy Liquidity Sweep")
                        Chart.DrawText(Prefix + "lbl_" + NextId(),
                            "Buy Liquidity Sweep",
                            utcBar, lvl.Price + atr, BuyLabelColor);
                        activeRes.RemoveAt(r);
                    }
                    else if (cl > lvl.Price)
                    {
                        // Pine: close > highPrice → level broken, remove silently
                        activeRes.RemoveAt(r);
                    }
                }

                // ── Support: sweep / break ────────────────────────────────────
                for (int s = activeSup.Count - 1; s >= 0; s--)
                {
                    ActiveLevel lvl = activeSup[s];

                    if (lo < lvl.Price && cl > lvl.Price)
                    {
                        // Pine: low < lowPrice AND close > lowPrice
                        // → wick swept below level, close rejected above
                        // → diagonal line: (origin, level price) → (this bar, low)
                        Chart.DrawTrendLine(Prefix + "swp_" + NextId(),
                            lvl.OriginUtc, lvl.Price, utcBar, lo,
                            SellSweepColor, SweepLineWidth, LineStyle.Solid);
                        // Pine: label.new(bar_index, lowPrice - atrOffset, "Sell Liquidity Sweep")
                        Chart.DrawText(Prefix + "lbl_" + NextId(),
                            "Sell Liquidity Sweep",
                            utcBar, lvl.Price - atr, SellLabelColor);
                        activeSup.RemoveAt(s);
                    }
                    else if (cl < lvl.Price)
                    {
                        // Pine: close < lowPrice → level broken, remove silently
                        activeSup.RemoveAt(s);
                    }
                }
            }

            // ── Draw surviving active S/R lines ───────────────────────────────
            // Extend from pivot origin to the current (last) bar.
            // Pine: line.set_x2(line, bar_index) extends to current bar each tick.
            DateTime utcNow = Bars.OpenTimes[index];

            foreach (ActiveLevel lvl in activeRes)
                Chart.DrawTrendLine(Prefix + "res_" + NextId(),
                    lvl.OriginUtc, lvl.Price, utcNow, lvl.Price,
                    ResistanceColor, LineWidth, LineStyle.Solid);

            foreach (ActiveLevel lvl in activeSup)
                Chart.DrawTrendLine(Prefix + "sup_" + NextId(),
                    lvl.OriginUtc, lvl.Price, utcNow, lvl.Price,
                    SupportColor, LineWidth, LineStyle.Solid);
        }

        // ── Pivot helpers ─────────────────────────────────────────────────────

        // Pine: ta.pivothigh(high, pivotPeriod, pivotPeriod)
        // Center bar = index - pivotPeriod.
        // Fires when center is STRICTLY highest in [center-P .. center+P].
        // Strict: any bar with high >= center disqualifies (no ties allowed).
        private bool IsPivotHigh(int index)
        {
            int    c  = index - PivotPeriod;
            double ph = Bars.HighPrices[c];
            int    L  = c - PivotPeriod;
            if (L < 0) return false;

            for (int j = L; j <= index; j++)
            {
                if (j == c) continue;
                if (Bars.HighPrices[j] >= ph) return false;
            }
            return true;
        }

        // Pine: ta.pivotlow(low, pivotPeriod, pivotPeriod)
        // Fires when center is STRICTLY lowest in [center-P .. center+P].
        private bool IsPivotLow(int index)
        {
            int    c  = index - PivotPeriod;
            double pl = Bars.LowPrices[c];
            int    L  = c - PivotPeriod;
            if (L < 0) return false;

            for (int j = L; j <= index; j++)
            {
                if (j == c) continue;
                if (Bars.LowPrices[j] <= pl) return false;
            }
            return true;
        }

        // ── Utility ───────────────────────────────────────────────────────────

        private string NextId()
            => (_objId++).ToString(CultureInfo.InvariantCulture);

        private void ClearObjects()
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
