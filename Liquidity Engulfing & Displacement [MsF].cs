using System;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class LiquidityEngulfingDisplacementMsF : Indicator
    {
        public enum DisplacementType
        {
            OpenToClose,
            HighToLow
        }

        [Parameter("Show H1 LEC", DefaultValue = true, Group = "LEC setting")]
        public bool ShowH1 { get; set; }

        [Parameter("Show H4 LEC", DefaultValue = false, Group = "LEC setting")]
        public bool ShowH4 { get; set; }

        [Parameter("Show Current LEC", DefaultValue = false, Group = "LEC setting")]
        public bool ShowCurrent { get; set; }

        [Parameter("Apply Stop Hunt Wick Filter", DefaultValue = true, Group = "LEC setting")]
        public bool FilterLiquidity { get; set; }

        [Parameter("Apply Close Filter", DefaultValue = true, Group = "LEC setting")]
        public bool FilterClose { get; set; }

        [Parameter("Require FVG", DefaultValue = true, Group = "Displacement setting")]
        public bool RequireFvg { get; set; }

        [Parameter("Displacement Type", DefaultValue = DisplacementType.OpenToClose, Group = "Displacement setting")]
        public DisplacementType DispType { get; set; }

        [Parameter("Displacement Length", DefaultValue = 100, MinValue = 1, Group = "Displacement setting")]
        public int StdLen { get; set; }

        [Parameter("Displacement Strength", DefaultValue = 2, MinValue = 0, Group = "Displacement setting")]
        public int StdX { get; set; }

        [Parameter("Bar Color", DefaultValue = "Yellow", Group = "Displacement setting")]
        public Color DispColor { get; set; }

        [Parameter("Bull LEC Color", DefaultValue = "Aqua", Group = "Style")]
        public Color BullColor { get; set; }

        [Parameter("Bear LEC Color", DefaultValue = "Red", Group = "Style")]
        public Color BearColor { get; set; }

        [Parameter("Marker Offset (pips)", DefaultValue = 5.0, MinValue = 0.0, Group = "Style")]
        public double MarkerOffsetPips { get; set; }

        private Bars _h1Bars;
        private Bars _h4Bars;

        protected override void Initialize()
        {
            _h1Bars = MarketData.GetBars(TimeFrame.Hour);
            _h4Bars = MarketData.GetBars(TimeFrame.Hour4);
        }

        public override void Calculate(int index)
        {
            DrawLecSignals(index);
            ApplyDisplacementColor(index);
        }

        private void DrawLecSignals(int index)
        {
            DrawMappedLec(index, _h1Bars, "H1", ShowH1);
            DrawMappedLec(index, _h4Bars, "H4", ShowH4);
            DrawMappedLec(index, Bars, "CUR", ShowCurrent);
        }

        private void DrawMappedLec(int chartIndex, Bars sourceBars, string tfKey, bool isEnabled)
        {
            string bullName = "LEC_BULL_" + tfKey + "_" + chartIndex;
            string bearName = "LEC_BEAR_" + tfKey + "_" + chartIndex;

            if (!isEnabled || sourceBars == null || sourceBars.Count < 2 || chartIndex < 1)
            {
                Chart.RemoveObject(bullName);
                Chart.RemoveObject(bearName);
                return;
            }

            int currentMappedIndex = FindBarIndexAtOrBefore(sourceBars, Bars.OpenTimes[chartIndex]);
            int prevMappedIndex = FindBarIndexAtOrBefore(sourceBars, Bars.OpenTimes[chartIndex - 1]);

            bool bullNow, bearNow;
            bool bullPrev, bearPrev;

            EvaluateLec(sourceBars, currentMappedIndex, out bullNow, out bearNow);
            EvaluateLec(sourceBars, prevMappedIndex, out bullPrev, out bearPrev);

            bool bullSignal = bullNow && !bullPrev;
            bool bearSignal = bearNow && !bearPrev;

            if (bullSignal)
            {
                double y = Bars.LowPrices[chartIndex] - MarkerOffsetPips * Symbol.PipSize;
                Chart.DrawIcon(bullName, ChartIconType.UpTriangle, Bars.OpenTimes[chartIndex], y, BullColor);
            }
            else
            {
                Chart.RemoveObject(bullName);
            }

            if (bearSignal)
            {
                double y = Bars.HighPrices[chartIndex] + MarkerOffsetPips * Symbol.PipSize;
                Chart.DrawIcon(bearName, ChartIconType.DownTriangle, Bars.OpenTimes[chartIndex], y, BearColor);
            }
            else
            {
                Chart.RemoveObject(bearName);
            }
        }

        private void EvaluateLec(Bars b, int index, out bool bullEngulf, out bool bearEngulf)
        {
            bullEngulf = false;
            bearEngulf = false;

            if (b == null || index < 1 || index >= b.Count)
                return;

            double priorOpen = b.OpenPrices[index - 1];
            double priorClose = b.ClosePrices[index - 1];
            double currentOpen = b.OpenPrices[index];
            double currentClose = b.ClosePrices[index];

            bullEngulf = (currentOpen <= priorClose) && (currentOpen < priorOpen) && (currentClose > priorOpen);
            bearEngulf = (currentOpen >= priorClose) && (currentOpen > priorOpen) && (currentClose < priorOpen);

            if (FilterLiquidity)
            {
                bullEngulf = bullEngulf && b.LowPrices[index] <= b.LowPrices[index - 1];
                bearEngulf = bearEngulf && b.HighPrices[index] >= b.HighPrices[index - 1];
            }

            if (FilterClose)
            {
                bullEngulf = bullEngulf && b.ClosePrices[index] >= b.HighPrices[index - 1];
                bearEngulf = bearEngulf && b.ClosePrices[index] <= b.LowPrices[index - 1];
            }
        }

        private void ApplyDisplacementColor(int index)
        {
            if (RequireFvg)
            {
                if (index < 2)
                    return;

                bool displacement = IsDisplacementWithFvg(index);

                int colorIndex = index - 1;
                if (colorIndex < 0)
                    return;

                if (displacement)
                    SetBarColor(colorIndex, DispColor);
                else
                    ResetBarColor(colorIndex);
            }
            else
            {
                bool displacement = IsDisplacementNoFvg(index);

                if (displacement)
                    SetBarColor(index, DispColor);
                else
                    ResetBarColor(index);
            }
        }

        private bool IsDisplacementWithFvg(int index)
        {
            if (index < 2)
                return false;

            double prevRange = GetCandleRange(index - 1);
            double prevStd = GetStdDev(index - 1);

            if (double.IsNaN(prevStd))
                return false;

            bool fvg;
            if (Bars.ClosePrices[index - 1] > Bars.OpenPrices[index - 1])
                fvg = Bars.HighPrices[index - 2] < Bars.LowPrices[index];
            else
                fvg = Bars.LowPrices[index - 2] > Bars.HighPrices[index];

            return prevRange > prevStd && fvg;
        }

        private bool IsDisplacementNoFvg(int index)
        {
            double range = GetCandleRange(index);
            double std = GetStdDev(index);

            if (double.IsNaN(std))
                return false;

            return range > std;
        }

        private double GetCandleRange(int index)
        {
            if (index < 0 || index >= Bars.Count)
                return double.NaN;

            if (DispType == DisplacementType.OpenToClose)
                return Math.Abs(Bars.OpenPrices[index] - Bars.ClosePrices[index]);

            return Bars.HighPrices[index] - Bars.LowPrices[index];
        }

        private double GetStdDev(int index)
        {
            if (index < 0 || StdLen <= 0)
                return double.NaN;

            int start = index - StdLen + 1;
            if (start < 0)
                return double.NaN;

            double sum = 0.0;
            for (int i = start; i <= index; i++)
                sum += GetCandleRange(i);

            double mean = sum / StdLen;

            double varianceSum = 0.0;
            for (int i = start; i <= index; i++)
            {
                double diff = GetCandleRange(i) - mean;
                varianceSum += diff * diff;
            }

            double variance = varianceSum / StdLen;
            return Math.Sqrt(variance) * StdX;
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            if (bars == null || bars.Count == 0)
                return -1;

            int idx = bars.OpenTimes.GetIndexByTime(time);
            if (idx >= 0)
                return idx;

            for (int i = bars.Count - 1; i >= 0; i--)
            {
                if (bars.OpenTimes[i] <= time)
                    return i;
            }

            return -1;
        }

        private void SetBarColor(int index, Color color)
        {
            if (index < 0 || index >= Bars.Count)
                return;

            Chart.SetBarColor(index, color);
        }

        private void ResetBarColor(int index)
        {
            if (index < 0 || index >= Bars.Count)
                return;

            Chart.ResetBarColor(index);
        }
    }
}
