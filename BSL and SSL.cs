// BSL&SSL.cs
// Mirrors "BSL & SSL" portion of Liquidity & inducements.txt
// Incorporates pivot logic consistent with PriceAction.Pivot(structure): ta.pivothigh/ta.pivotlow (non-strict max/min).

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

        [Parameter("Buyside Color", DefaultValue = "Teal", Group = "Buyside & sellside liquidity")]
        public string BuysideColorName { get; set; }

        [Parameter("Sellside Color", DefaultValue = "Red", Group = "Buyside & sellside liquidity")]
        public string SellsideColorName { get; set; }

        [Parameter("Label Y Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Group = "Display")]
        public double LabelOffsetPips { get; set; }

        [Output("Current BSL", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries CurrentBSL { get; set; }

        [Output("Current SSL", LineColor = "Transparent", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries CurrentSSL { get; set; }

        private sealed class Pivot
        {
            public double Price;
            public int BarIndex;   // pivot bar index (equivalent to bar_index - RightLength at confirmation)
            public int Type;       // 1 = high (BSL), -1 = low (SSL)
        }

        private sealed class ExternalLiquidity
        {
            public double Price;
            public ChartTrendLine Line;
            public ChartText Label;
            public bool Hidden;
            public int PivotIndex;
        }

        private readonly LinkedList<Pivot> _pivots = new LinkedList<Pivot>();   // newest first
        private const int MaxPivotsToKeep = 10;

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

            int pivotIndex = index - PivotRight;
            if (pivotIndex <= 0)
                return;

            DetectAndStoreConfirmedPivots(index);
            AddExternalLiquidityFromNewPivot(index);
            ClearMitigated();
            ApplyShowRules();
            UpdateOutputLevels(index);
        }

        private void UpdateOutputLevels(int index)
        {
            CurrentBSL[index] = _buysidePools.First != null ? _buysidePools.First.Value.Price : double.NaN;
            CurrentSSL[index] = _sellsidePools.First != null ? _sellsidePools.First.Value.Price : double.NaN;
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

        // Non-strict: candidate must equal window max/min (ties allowed), matching ta.pivothigh/low behavior needed for your data.
        private bool IsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];

            return candidate == max;
        }

        private bool IsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];

            return candidate == min;
        }

        private void UnshiftPivot(Pivot p)
        {
            // defensive: avoid exact duplicates
            if (_pivots.First != null &&
                _pivots.First.Value.BarIndex == p.BarIndex &&
                _pivots.First.Value.Type == p.Type &&
                Math.Abs(_pivots.First.Value.Price - p.Price) < Symbol.PipSize * 0.1)
                return;

            _pivots.AddFirst(p);

            while (_pivots.Count > MaxPivotsToKeep)
                _pivots.RemoveLast();
        }

        private void AddExternalLiquidityFromNewPivot(int currentIndex)
        {
            int confirmedPivotIndex = currentIndex - PivotRight;

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
            // Hide existing pools (Pine sets older pools to na color/textcolor when new pivot comes in)
            foreach (var existing in poolList)
            {
                if (!existing.Hidden)
                    SetPoolHidden(existing, true);
            }

            string labelText = isBuyside ? "Buyside liquidity" : "Sellside liquidity";
            string idBase = isBuyside ? "BSL" : "SSL";
            string uid = $"{idBase}_{SymbolName}_{TimeFrame}_{pivot.BarIndex}_{pivot.Price:F5}";

            var line = Chart.DrawTrendLine(uid + "_LINE",
                pivot.BarIndex, pivot.Price,
                Bars.Count - 1, pivot.Price,
                Color.Transparent,
                1,
                MapLineStyle(LineStyleParam));

            line.ExtendToInfinity = true;

            double offset = LabelOffsetPips * Symbol.PipSize;
            double labelY = isBuyside ? pivot.Price + offset : pivot.Price - offset;

            var label = Chart.DrawText(uid + "_LABEL",
                labelText,
                pivot.BarIndex,
                labelY,
                Color.Transparent);

            var ex = new ExternalLiquidity
            {
                Price = pivot.Price,
                PivotIndex = pivot.BarIndex,
                Line = line,
                Label = label,
                Hidden = true
            };

            poolList.AddFirst(ex);
        }

        private void ClearMitigated()
        {
            // Sellside mitigated: Low <= SSL
            var node = _sellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                var ssl = node.Value;

                if (Bars.LowPrices[_lastProcessedIndex] <= ssl.Price)
                {
                    DeletePool(ssl);
                    _sellsidePools.Remove(node);
                }
                node = next;
            }

            // Buyside mitigated: High >= BSL
            node = _buysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                var bsl = node.Value;

                if (Bars.HighPrices[_lastProcessedIndex] >= bsl.Price)
                {
                    DeletePool(bsl);
                    _buysidePools.Remove(node);
                }
                node = next;
            }
        }

        private void ApplyShowRules()
        {
            ApplyShowRulesToList(_buysidePools, ShowPools, _buysideColor);
            ApplyShowRulesToList(_sellsidePools, ShowPools, _sellsideColor);
        }

        private void ApplyShowRulesToList(LinkedList<ExternalLiquidity> pools, int show, Color c)
        {
            int i = 0;
            foreach (var p in pools)
            {
                i++;
                bool shouldShow = i <= show;
                SetPoolHidden(p, !shouldShow);

                if (shouldShow)
                {
                    if (p.Line != null) p.Line.Color = c;
                    if (p.Label != null) p.Label.Color = c;
                }
            }
        }

        private void SetPoolHidden(ExternalLiquidity p, bool hidden)
        {
            p.Hidden = hidden;

            if (p.Line != null)
                p.Line.IsHidden = hidden;

            if (p.Label != null)
                p.Label.IsHidden = hidden;
        }

        private void DeletePool(ExternalLiquidity p)
        {
            if (p.Label != null)
                Chart.RemoveObject(p.Label.Name);

            if (p.Line != null)
                Chart.RemoveObject(p.Line.Name);
        }

        private static LineStyle MapLineStyle(LiquidityLineStyle s)
        {
            switch (s)
            {
                case LiquidityLineStyle.Solid:
                    return LineStyle.Solid;
                case LiquidityLineStyle.Dashes:
                    return LineStyle.Lines;
                case LiquidityLineStyle.Dots:
                default:
                    return LineStyle.Dots;
            }
        }

        private static Color ParseColorOrDefault(string name, Color fallback)
        {
            if (string.IsNullOrWhiteSpace(name))
                return fallback;

            try { return Color.FromName(name.Trim()); }
            catch { return fallback; }
        }
    }
}
