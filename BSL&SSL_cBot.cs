// BSL and SSL.cs  (Modified for cBot use)
// - Adds bot-readable output series: BSLLevel / SSLLevel
// - Exposes CurrentBSL / CurrentSSL
// - Fixes LineStyle mapping (no DotsVerySparse in cTrader)

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class BSL_SSL : Indicator
    {
        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "Market structure")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "Market structure")]
        public int PivotRight { get; set; }

        [Parameter("Show (Pools)", DefaultValue = 1, MinValue = 1, Group = "Buyside & sellside liquidity")]
        public int ShowPools { get; set; }

        public enum LiquidityLineStyle
        {
            Solid,
            Dots,
            Dashes
        }

        [Parameter("Line style", DefaultValue = LiquidityLineStyle.Dots, Group = "Display")]
        public LiquidityLineStyle LineStyleParam { get; set; }

        [Parameter("Buyside Color", DefaultValue = "Teal", Group = "Display")]
        public string BuysideColorName { get; set; }

        [Parameter("Sellside Color", DefaultValue = "Red", Group = "Display")]
        public string SellsideColorName { get; set; }

        [Parameter("Line Thickness", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = "Display")]
        public int LineThickness { get; set; }

        [Parameter("Label Offset (pips)", DefaultValue = 3, MinValue = 0, Group = "Display")]
        public double LabelOffsetPips { get; set; }

        // Bot-readable outputs (also show as discontinuous lines)
        [Output("BSLLevel", LineColor = "Teal", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BSLLevel { get; set; }

        [Output("SSLLevel", LineColor = "Red", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries SSLLevel { get; set; }

        // Latest active pool prices for bots
        public double CurrentBSL { get; private set; } = double.NaN;
        public double CurrentSSL { get; private set; } = double.NaN;

        private struct Pivot
        {
            public double Price;
            public int BarIndex;
            public int Type; // 1 = pivot high (BSL), -1 = pivot low (SSL)
        }

        private class ExternalLiquidity
        {
            public double Price;
            public ChartTrendLine Line;
            public ChartText Label;
            public bool Hidden;
            public int PivotIndex;
            public bool IsBuyside;
        }

        private readonly LinkedList<Pivot> _pivots = new LinkedList<Pivot>();   // newest first
        private const int MaxPivotsToKeep = 30;

        private readonly LinkedList<ExternalLiquidity> _buysidePools = new LinkedList<ExternalLiquidity>();
        private readonly LinkedList<ExternalLiquidity> _sellsidePools = new LinkedList<ExternalLiquidity>();

        private int _lastProcessedIndex = -1;

        private Color _buysideColor;
        private Color _sellsideColor;

        protected override void Initialize()
        {
            _buysideColor = ParseColorOrDefault(BuysideColorName, Color.Teal);
            _sellsideColor = ParseColorOrDefault(SellsideColorName, Color.Red);
        }

        public override void Calculate(int index)
        {
            if (index <= _lastProcessedIndex)
                return;

            _lastProcessedIndex = index;

            // Need enough bars to confirm a pivot
            int pivotIndex = index - PivotRight;
            if (pivotIndex <= 0)
            {
                BSLLevel[index] = CurrentBSL;
                SSLLevel[index] = CurrentSSL;
                return;
            }

            DetectAndStoreConfirmedPivots(index);
            AddExternalLiquidityFromPivot(index);

            // Optional: implement mitigation/invalidations later if you want parity with your Pine behavior
            // ClearMitigated();

            ApplyShowRules();

            // Update bot-readable values
            CurrentBSL = GetActivePoolPrice(_buysidePools);
            CurrentSSL = GetActivePoolPrice(_sellsidePools);

            BSLLevel[index] = CurrentBSL;
            SSLLevel[index] = CurrentSSL;
        }

        private void DetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd = pivotIndex + PivotRight;

            if (leftStart < 0 || rightEnd >= Bars.Count)
                return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow = Bars.LowPrices[pivotIndex];

            bool isPivotHigh = IsPivotHigh(candidateHigh, leftStart, rightEnd);
            bool isPivotLow = IsPivotLow(candidateLow, leftStart, rightEnd);

            if (isPivotHigh)
                UnshiftPivot(new Pivot { Price = candidateHigh, BarIndex = pivotIndex, Type = 1 });

            if (isPivotLow)
                UnshiftPivot(new Pivot { Price = candidateLow, BarIndex = pivotIndex, Type = -1 });
        }

        // Non-strict: candidate must equal window max/min (ties allowed)
        private bool IsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                max = Math.Max(max, Bars.HighPrices[i]);

            return candidate >= max - Symbol.TickSize * 0.0001;
        }

        private bool IsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                min = Math.Min(min, Bars.LowPrices[i]);

            return candidate <= min + Symbol.TickSize * 0.0001;
        }

        private void UnshiftPivot(Pivot pivot)
        {
            _pivots.AddFirst(pivot);

            while (_pivots.Count > MaxPivotsToKeep)
                _pivots.RemoveLast();
        }

        private void AddExternalLiquidityFromPivot(int currentIndex)
        {
            int confirmedPivotIndex = currentIndex - PivotRight;

            // Add pools for pivots that just got confirmed on this bar
            foreach (var pivot in _pivots)
            {
                if (pivot.BarIndex != confirmedPivotIndex)
                    continue;

                if (pivot.Type == 1)
                    AddExternalLiquidity(_buysidePools, pivot, isBuyside: true);
                else if (pivot.Type == -1)
                    AddExternalLiquidity(_sellsidePools, pivot, isBuyside: false);
            }
        }

        private void AddExternalLiquidity(LinkedList<ExternalLiquidity> poolList, Pivot pivot, bool isBuyside)
        {
            // Hide existing pools in the same list so only the newest is active
            foreach (var existing in poolList)
            {
                if (!existing.Hidden)
                    SetPoolHidden(existing, true);
            }

            string labelText = isBuyside ? "Buyside liquidity" : "Sellside liquidity";
            string idBase = isBuyside ? "BSL" : "SSL";
            string uid = $"{idBase}_{SymbolName}_{TimeFrame}_{pivot.BarIndex}_{pivot.Price:F5}";

            var color = isBuyside ? _buysideColor : _sellsideColor;

            var line = Chart.DrawTrendLine(uid + "_LINE",
                pivot.BarIndex, pivot.Price,
                Bars.Count - 1, pivot.Price,
                color,
                LineThickness,
                MapLineStyle(LineStyleParam));

            line.ExtendToInfinity = true;

            double offset = LabelOffsetPips * Symbol.PipSize;
            double labelY = isBuyside ? pivot.Price + offset : pivot.Price - offset;

            var label = Chart.DrawText(uid + "_LABEL",
                labelText,
                Bars.Count - 1,
                labelY,
                color);

            var liq = new ExternalLiquidity
            {
                Price = pivot.Price,
                Line = line,
                Label = label,
                Hidden = false,
                PivotIndex = pivot.BarIndex,
                IsBuyside = isBuyside
            };

            poolList.AddFirst(liq);
            SetPoolHidden(liq, false);
        }

        private void SetPoolHidden(ExternalLiquidity liq, bool hidden)
        {
            liq.Hidden = hidden;

            if (liq.Line != null)
                liq.Line.Color = hidden ? Color.Transparent : (liq.IsBuyside ? _buysideColor : _sellsideColor);

            if (liq.Label != null)
                liq.Label.IsHidden = hidden;
        }

        private void ApplyShowRules()
        {
            // Current implementation:
            // - Newest pool is visible; older pools are hidden (per list)
            // - ShowPools can be expanded later for additional modes
            // If you want ShowPools==0 to hide everything:
            if (ShowPools <= 0)
            {
                foreach (var b in _buysidePools)
                    SetPoolHidden(b, true);

                foreach (var s in _sellsidePools)
                    SetPoolHidden(s, true);
            }
        }

        private LineStyle MapLineStyle(LiquidityLineStyle style)
        {
            switch (style)
            {
                case LiquidityLineStyle.Dashes:
                    // cTrader does not have a true "Dashes" line style.
                    // Use the sparsest dots available across most versions.
                    return LineStyle.DotsVeryRare; // if your build lacks this, change to LineStyle.DotsRare

                case LiquidityLineStyle.Dots:
                    return LineStyle.Dots;

                default:
                    return LineStyle.Solid;
            }
        }

        private Color ParseColorOrDefault(string name, Color fallback)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(name))
                    return fallback;

                return Color.FromName(name);
            }
            catch
            {
                return fallback;
            }
        }

        private double GetActivePoolPrice(LinkedList<ExternalLiquidity> pools)
        {
            foreach (var p in pools)
            {
                if (!p.Hidden)
                    return p.Price;
            }
            return double.NaN;
        }
    }
}
