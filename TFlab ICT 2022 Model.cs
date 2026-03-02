using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class TFlabICT2022Model : Indicator
    {
        private const string Prefix = "TF2022_";
        private const int MaxObjects = 500;

        private sealed class SwingPoint
        {
            public double Price;
            public DateTime Time;
            public int Index;
            public bool Permit = true;
            public int AoiIndex = int.MaxValue;
        }

        private sealed class FvgZone
        {
            public int Index;
            public double Distal;
            public double Proximal;
            public bool IsDemand;
        }

        // Logical Setting
        [Parameter("Swing Period", Group = "Logical Setting", DefaultValue = 50, MinValue = 1)]
        public int SwingPeriod { get; set; }

        [Parameter("Max Swing Back Method", Group = "Logical Setting", DefaultValue = "All")]
        public string MaxSwingBackMethod { get; set; }

        [Parameter("Max Swing Back", Group = "Logical Setting", DefaultValue = 100, MinValue = 1)]
        public int MaxSwingBack { get; set; }

        [Parameter("FVG Length", Group = "Logical Setting", DefaultValue = 120, MinValue = 1)]
        public int FvgLength { get; set; }

        [Parameter("MSS Length", Group = "Logical Setting", DefaultValue = 80, MinValue = 1)]
        public int MssLength { get; set; }

        // FVG Logical Setting
        [Parameter("FVG Filter", Group = "FVG Logical Setting", DefaultValue = false)]
        public bool PfvgFilter { get; set; }

        [Parameter("FVG Filter Type", Group = "FVG Logical Setting", DefaultValue = "Defensive")]
        public string PfvgFilterType { get; set; }

        [Parameter("Mitigation Level FVG", Group = "FVG Logical Setting", DefaultValue = "Proximal")]
        public string Mlfvg { get; set; }

        // FVG Display Setting
        [Parameter("Demand FVG", Group = "FVG Display Setting", DefaultValue = true)]
        public bool DemandFvgShow { get; set; }

        [Parameter("Demand Color", Group = "FVG Display Setting", DefaultValue = "#6C03A803")]
        public Color DemandFvgColor { get; set; }

        [Parameter("Supply FVG", Group = "FVG Display Setting", DefaultValue = true)]
        public bool SupplyFvgShow { get; set; }

        [Parameter("Supply Color", Group = "FVG Display Setting", DefaultValue = "#70C20D00")]
        public Color SupplyFvgColor { get; set; }

        // Display Setting - Liquidity Sweep
        [Parameter("Show All LvL", Group = "Display Setting", DefaultValue = true)]
        public bool AShow { get; set; }

        [Parameter("Show High LvL", Group = "Display Setting", DefaultValue = true)]
        public bool HShow { get; set; }

        [Parameter("Show Low LvL", Group = "Display Setting", DefaultValue = true)]
        public bool LShow { get; set; }

        [Parameter("High Name", Group = "Display Setting", DefaultValue = "#870505")]
        public Color HNColor { get; set; }

        [Parameter("Low Name", Group = "Display Setting", DefaultValue = "#014F07")]
        public Color LNColor { get; set; }

        [Parameter("High Line", Group = "Display Setting", DefaultValue = "#B80000B9")]
        public Color HLColor { get; set; }

        [Parameter("Low Line", Group = "Display Setting", DefaultValue = "#A7069206")]
        public Color LLColor { get; set; }

        // Display Setting - MSS
        [Parameter("Show All MSS", Group = "Display Setting", DefaultValue = true)]
        public bool AShowMss { get; set; }

        [Parameter("Show High MSS", Group = "Display Setting", DefaultValue = true)]
        public bool HShowMss { get; set; }

        [Parameter("Show Low MSS", Group = "Display Setting", DefaultValue = true)]
        public bool LShowMss { get; set; }

        [Parameter("High MSS Name", Group = "Display Setting", DefaultValue = "#870505")]
        public Color HNColorMss { get; set; }

        [Parameter("Low MSS Name", Group = "Display Setting", DefaultValue = "#014F07")]
        public Color LNColorMss { get; set; }

        [Parameter("High MSS Line", Group = "Display Setting", DefaultValue = "#B80000B9")]
        public Color HLColorMss { get; set; }

        [Parameter("Low MSS Line", Group = "Display Setting", DefaultValue = "#A7069206")]
        public Color LLColorMss { get; set; }

        // Alert
        [Parameter("Alert", Group = "Alert", DefaultValue = "On")]
        public string AlertMode { get; set; }

        [Parameter("Alert Name", Group = "Alert", DefaultValue = "2022 Model [TradingFinder]")]
        public string AlertName { get; set; }

        [Parameter("Message Frequency", Group = "Alert", DefaultValue = "Once Per Bar")]
        public string Frequency { get; set; }

        [Parameter("Show Alert time by Time Zone", Group = "Alert", DefaultValue = "UTC")]
        public string AlertTimeZone { get; set; }

        [Parameter("Long Position Message", Group = "Alert", DefaultValue = "Long Signal Position Based on 2022 Model")]
        public string MessageBull { get; set; }

        [Parameter("Short Position Message", Group = "Alert", DefaultValue = "Short Signal Position Based on 2022 Model")]
        public string MessageBear { get; set; }

        private readonly List<SwingPoint> _swingHighs = new List<SwingPoint>();
        private readonly List<SwingPoint> _swingLows = new List<SwingPoint>();
        private readonly List<FvgZone> _bearFvgs = new List<FvgZone>();
        private readonly List<FvgZone> _bullFvgs = new List<FvgZone>();

        protected override void Initialize()
        {
        }

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1)
                return;

            ClearObjects();
            _swingHighs.Clear();
            _swingLows.Clear();
            _bearFvgs.Clear();
            _bullFvgs.Clear();

            int maxBack = ResolveMaxSwingBack();
            int lastHighSweepIndex = -1;
            int lastLowSweepIndex = -1;
            double lastHighSweepPrice = 0;
            double lastLowSweepPrice = 0;

            int mssBearStartIndex = -1;
            double mssBearLevel = 0;
            bool permitHReset = false;

            int mssBullStartIndex = -1;
            double mssBullLevel = 0;
            bool permitLReset = false;

            bool prevPermitHReset = false;
            bool prevPermitLReset = false;

            int lastPivotHighIndex = -1;
            double lastPivotHighPrice = 0;
            int lastPivotLowIndex = -1;
            double lastPivotLowPrice = 0;

            var atr = Indicators.AverageTrueRange(55, MovingAverageType.Simple);

            for (int i = 2; i < Bars.Count - 2; i++)
            {
                bool hAlert = false;
                bool lAlert = false;

                // MSS pivot detection equivalent to pivothigh(2,1), pivotlow(2,1)
                if (IsPivotHigh(i - 1, 2, 1))
                {
                    lastPivotHighIndex = i - 1;
                    lastPivotHighPrice = Bars.HighPrices[lastPivotHighIndex];
                }
                if (IsPivotLow(i - 1, 2, 1))
                {
                    lastPivotLowIndex = i - 1;
                    lastPivotLowPrice = Bars.LowPrices[lastPivotLowIndex];
                }

                // major swing pivot detection equivalent ta.pivothigh(SwingPeriod, SwingPeriod)
                int pivotIndex = i - SwingPeriod;
                if (pivotIndex >= SwingPeriod && pivotIndex < Bars.Count - SwingPeriod)
                {
                    if (IsPivotHigh(pivotIndex, SwingPeriod, SwingPeriod))
                    {
                        _swingHighs.Add(new SwingPoint
                        {
                            Index = pivotIndex,
                            Time = Bars.OpenTimes[pivotIndex],
                            Price = Bars.HighPrices[pivotIndex]
                        });
                    }
                    if (IsPivotLow(pivotIndex, SwingPeriod, SwingPeriod))
                    {
                        _swingLows.Add(new SwingPoint
                        {
                            Index = pivotIndex,
                            Time = Bars.OpenTimes[pivotIndex],
                            Price = Bars.LowPrices[pivotIndex]
                        });
                    }
                }

                // FVG detection (local equivalent of imported FVG library)
                bool demandFvg = false;
                bool supplyFvg = false;
                FvgZone demandZone = null;
                FvgZone supplyZone = null;
                DetectFvg(i, out demandFvg, out demandZone, out supplyFvg, out supplyZone);

                // Liquidity sweep high
                int hs = Math.Min(maxBack, _swingHighs.Count);
                for (int k = 1; k <= hs; k++)
                {
                    var swing = _swingHighs[_swingHighs.Count - k];
                    if (!swing.Permit)
                        continue;

                    if (i <= swing.AoiIndex && Bars.HighPrices[i] > swing.Price && Bars.ClosePrices[i] < swing.Price && swing.AoiIndex == int.MaxValue)
                    {
                        swing.Permit = false;
                        hAlert = true;
                        lastHighSweepIndex = swing.Index;
                        lastHighSweepPrice = swing.Price;
                        DrawLiquiditySweep(true, i, swing, atr.Result[i]);
                    }

                    if (Bars.ClosePrices[i] > swing.Price && swing.AoiIndex == int.MaxValue)
                        swing.AoiIndex = i;
                }

                // Liquidity sweep low
                int ls = Math.Min(maxBack, _swingLows.Count);
                for (int k = 1; k <= ls; k++)
                {
                    var swing = _swingLows[_swingLows.Count - k];
                    if (!swing.Permit)
                        continue;

                    if (i <= swing.AoiIndex && Bars.LowPrices[i] < swing.Price && Bars.ClosePrices[i] > swing.Price && swing.AoiIndex == int.MaxValue)
                    {
                        swing.Permit = false;
                        lAlert = true;
                        lastLowSweepIndex = swing.Index;
                        lastLowSweepPrice = swing.Price;
                        DrawLiquiditySweep(false, i, swing, atr.Result[i]);
                    }

                    if (Bars.ClosePrices[i] < swing.Price && swing.AoiIndex == int.MaxValue)
                        swing.AoiIndex = i;
                }

                // reset permits analogous to Pine H_Permit/L_Permit
                if (!hAlert)
                    ResetPermits(_swingHighs);
                if (!lAlert)
                    ResetPermits(_swingLows);

                // MSS set after high sweep
                if (hAlert && lastPivotLowIndex >= 0)
                {
                    permitHReset = true;
                    mssBearStartIndex = lastPivotLowIndex;
                    mssBearLevel = lastPivotLowPrice;
                    DrawMssCandidate(true, i, mssBearStartIndex, mssBearLevel, atr.Result[i]);
                }

                // MSS set after low sweep
                if (lAlert && lastPivotHighIndex >= 0)
                {
                    permitLReset = true;
                    mssBullStartIndex = lastPivotHighIndex;
                    mssBullLevel = lastPivotHighPrice;
                    DrawMssCandidate(false, i, mssBullStartIndex, mssBullLevel, atr.Result[i]);
                }

                // Maintain FVG stacks while MSS pending
                if (permitHReset && supplyFvg && supplyZone != null)
                    _bearFvgs.Add(supplyZone);
                if (_bearFvgs.Count > 0 && Bars.HighPrices[i] > _bearFvgs[_bearFvgs.Count - 1].Distal)
                    _bearFvgs.RemoveAt(_bearFvgs.Count - 1);

                if (permitLReset && demandFvg && demandZone != null)
                    _bullFvgs.Add(demandZone);
                if (_bullFvgs.Count > 0 && Bars.LowPrices[i] < _bullFvgs[_bullFvgs.Count - 1].Distal)
                    _bullFvgs.RemoveAt(_bullFvgs.Count - 1);

                bool bearTrigger = false;
                bool bullTrigger = false;
                bool fvgBearTrigger = false;
                bool fvgBullTrigger = false;

                // confirm bearish MSS
                if (permitHReset && mssBearStartIndex >= 0)
                {
                    if (Bars.ClosePrices[Math.Max(i - 1, 0)] >= mssBearLevel && (Math.Max(i - 1, 0) - mssBearStartIndex) <= MssLength)
                        ExtendMssLine(true, i + 1, mssBearStartIndex, mssBearLevel);

                    if (Bars.ClosePrices[Math.Max(i - 1, 0)] <= mssBearLevel && (Math.Max(i - 1, 0) - mssBearStartIndex) <= MssLength)
                    {
                        permitHReset = false;
                        bearTrigger = true;
                        ColorizeHighStructures(i, mssBearStartIndex, mssBearLevel, atr.Result[i]);
                        fvgBearTrigger = _bearFvgs.Count > 0;
                    }
                }

                // confirm bullish MSS
                if (permitLReset && mssBullStartIndex >= 0)
                {
                    if (Bars.ClosePrices[i] <= mssBullLevel && (i - mssBullStartIndex) <= MssLength)
                        ExtendMssLine(false, i + 1, mssBullStartIndex, mssBullLevel);

                    if (Bars.ClosePrices[i] >= mssBullLevel && (i - mssBullStartIndex) <= MssLength)
                    {
                        permitLReset = false;
                        bullTrigger = true;
                        ColorizeLowStructures(i, mssBullStartIndex, mssBullLevel, atr.Result[i]);
                        fvgBullTrigger = _bullFvgs.Count > 0;
                    }
                }

                if (prevPermitHReset && !permitHReset)
                    bearTrigger = true;
                if (prevPermitLReset && !permitLReset)
                    bullTrigger = true;

                // Draw FVG OB equivalent
                if (fvgBullTrigger && _bullFvgs.Count > 0)
                {
                    var zone = _bullFvgs[_bullFvgs.Count - 1];
                    if (i - zone.Index < 500)
                    {
                        DrawFvgBlock(zone, i, true);
                        TriggerAlert(true, i);
                    }
                }

                if (fvgBearTrigger && _bearFvgs.Count > 0)
                {
                    var zone = _bearFvgs[_bearFvgs.Count - 1];
                    if (i - zone.Index < 500)
                    {
                        DrawFvgBlock(zone, i, false);
                        TriggerAlert(false, i);
                    }
                }

                // reset on new sweeps
                if (hAlert)
                    _bearFvgs.Clear();
                if (lAlert)
                    _bullFvgs.Clear();

                prevPermitHReset = permitHReset;
                prevPermitLReset = permitLReset;
            }
        }

        private void DetectFvg(int i, out bool demandFvg, out FvgZone demandZone, out bool supplyFvg, out FvgZone supplyZone)
        {
            demandFvg = false;
            supplyFvg = false;
            demandZone = null;
            supplyZone = null;

            if (i < 2)
                return;

            // demand FVG
            if (Bars.LowPrices[i] > Bars.HighPrices[i - 2])
            {
                double distal = Bars.HighPrices[i - 2];
                double proximal = Bars.LowPrices[i];
                double width = proximal - distal;
                if (!PfvgFilter || PassesFvgFilter(width, i))
                {
                    demandFvg = true;
                    demandZone = new FvgZone { Index = i, Distal = distal, Proximal = proximal, IsDemand = true };
                }
            }

            // supply FVG
            if (Bars.HighPrices[i] < Bars.LowPrices[i - 2])
            {
                double distal = Bars.LowPrices[i - 2];
                double proximal = Bars.HighPrices[i];
                double width = distal - proximal;
                if (!PfvgFilter || PassesFvgFilter(width, i))
                {
                    supplyFvg = true;
                    supplyZone = new FvgZone { Index = i, Distal = distal, Proximal = proximal, IsDemand = false };
                }
            }
        }

        private bool PassesFvgFilter(double width, int i)
        {
            double atr = Math.Max(1e-8, Indicators.AverageTrueRange(14, MovingAverageType.Simple).Result[i]);
            double ratio = width / atr;
            switch (PfvgFilterType)
            {
                case "Very Aggressive": return ratio <= 2.5;
                case "Aggressive": return ratio <= 1.8;
                case "Defensive": return ratio <= 1.2;
                case "Very Defensive": return ratio <= 0.8;
                default: return true;
            }
        }

        private void DrawLiquiditySweep(bool isHigh, int i, SwingPoint swing, double atr)
        {
            if (!AShow)
                return;
            if (isHigh && !HShow)
                return;
            if (!isHigh && !LShow)
                return;

            string side = isHigh ? "H" : "L";
            Color levelColor = isHigh ? HLColor : LLColor;
            Color nameColor = isHigh ? HNColor : LNColor;
            // TradingView parity requested by user:
            // - bearish sweep label (high-side sweep) below the candle
            // - bullish sweep label (low-side sweep) above the candle
            double yText = isHigh ? Bars.LowPrices[i] - 0.35 * atr : Bars.HighPrices[i] + 0.35 * atr;

            Chart.DrawTrendLine(Prefix + "LS_LINE_" + side + "_" + i,
                Bars.OpenTimes[swing.Index], swing.Price,
                Bars.OpenTimes[i], swing.Price,
                levelColor, 1, LineStyle.DotsRare);

            Chart.DrawText(Prefix + "LS_NAME_" + side + "_" + i,
                "Liquidity Sweep",
                Bars.OpenTimes[i], yText,
                nameColor);
        }

        private void DrawMssCandidate(bool bearish, int i, int startIndex, double level, double atr)
        {
            if (!AShowMss)
                return;
            if (bearish && !HShowMss)
                return;
            if (!bearish && !LShowMss)
                return;

            string side = bearish ? "H" : "L";
            Chart.DrawTrendLine(Prefix + "MSS_LINE_" + side,
                Bars.OpenTimes[startIndex], level,
                Bars.OpenTimes[i], level,
                bearish ? HLColorMss : LLColorMss, 1, LineStyle.Solid);

            double labelY = bearish ? Bars.LowPrices[startIndex] - 0.3 * atr : Bars.HighPrices[startIndex] + 0.6 * atr;
            Chart.DrawText(Prefix + "MSS_NAME_" + side,
                "MSS",
                Bars.OpenTimes[Math.Min(startIndex + 1, Bars.Count - 1)],
                labelY,
                bearish ? HNColorMss : LNColorMss);
        }

        private void ExtendMssLine(bool bearish, int toIndex, int startIndex, double level)
        {
            string side = bearish ? "H" : "L";
            Chart.DrawTrendLine(Prefix + "MSS_LINE_" + side,
                Bars.OpenTimes[startIndex], level,
                Bars.OpenTimes[Math.Min(toIndex, Bars.Count - 1)], level,
                bearish ? HLColorMss : LLColorMss, 1, LineStyle.Solid);
        }

        private void ColorizeHighStructures(int i, int startIndex, double level, double atr)
        {
            if (AShow && HShow)
            {
                Chart.DrawTrendLine(Prefix + "LS_CONF_H_" + i,
                    Bars.OpenTimes[startIndex], level,
                    Bars.OpenTimes[i], level,
                    HLColor, 1, LineStyle.DotsRare);
            }

            if (AShowMss && HShowMss)
            {
                Chart.DrawTrendLine(Prefix + "MSS_CONF_H_" + i,
                    Bars.OpenTimes[startIndex], level,
                    Bars.OpenTimes[i], level,
                    HLColorMss, 1, LineStyle.Solid);
                Chart.DrawText(Prefix + "MSS_CONF_H_TXT_" + i, "MSS", Bars.OpenTimes[i], Bars.HighPrices[i] + 0.25 * atr, HNColorMss);
                Chart.DrawIcon(Prefix + "MSS_TRI_H_" + i, ChartIconType.DownTriangle, Bars.OpenTimes[i], Bars.HighPrices[i] + 0.6 * atr, HNColorMss);
            }
        }

        private void ColorizeLowStructures(int i, int startIndex, double level, double atr)
        {
            if (AShow && LShow)
            {
                Chart.DrawTrendLine(Prefix + "LS_CONF_L_" + i,
                    Bars.OpenTimes[startIndex], level,
                    Bars.OpenTimes[i], level,
                    LLColor, 1, LineStyle.DotsRare);
            }

            if (AShowMss && LShowMss)
            {
                Chart.DrawTrendLine(Prefix + "MSS_CONF_L_" + i,
                    Bars.OpenTimes[startIndex], level,
                    Bars.OpenTimes[i], level,
                    LLColorMss, 1, LineStyle.Solid);
                Chart.DrawText(Prefix + "MSS_CONF_L_TXT_" + i, "MSS", Bars.OpenTimes[i], Bars.LowPrices[i] - 0.25 * atr, LNColorMss);
                Chart.DrawIcon(Prefix + "MSS_TRI_L_" + i, ChartIconType.UpTriangle, Bars.OpenTimes[i], Bars.LowPrices[i] - 0.6 * atr, LNColorMss);
            }
        }

        private void DrawFvgBlock(FvgZone zone, int i, bool bullish)
        {
            if (bullish && !DemandFvgShow)
                return;
            if (!bullish && !SupplyFvgShow)
                return;

            int endIndex = Math.Min(Bars.Count - 1, zone.Index + FvgLength);
            double top = Math.Max(zone.Distal, zone.Proximal);
            double bottom = Math.Min(zone.Distal, zone.Proximal);

            var box = Chart.DrawRectangle(Prefix + "FVG_" + (bullish ? "BULL_" : "BEAR_") + i,
                Bars.OpenTimes[zone.Index], top,
                Bars.OpenTimes[endIndex], bottom,
                bullish ? DemandFvgColor : SupplyFvgColor);
            box.IsFilled = true;

            double mitigation = ResolveMitigationLevel(zone);
            Chart.DrawTrendLine(Prefix + "FVG_MIT_" + (bullish ? "B_" : "S_") + i,
                Bars.OpenTimes[zone.Index], mitigation,
                Bars.OpenTimes[endIndex], mitigation,
                bullish ? DemandFvgColor : SupplyFvgColor,
                1, LineStyle.Solid);
        }

        private double ResolveMitigationLevel(FvgZone zone)
        {
            switch (Mlfvg)
            {
                case "Distal":
                    return zone.Distal;
                case "50 % OB":
                    return (zone.Distal + zone.Proximal) * 0.5;
                case "Proximal":
                default:
                    return zone.Proximal;
            }
        }

        private void TriggerAlert(bool bullish, int i)
        {
            if (!string.Equals(AlertMode, "On", StringComparison.OrdinalIgnoreCase))
                return;

            string msg = bullish ? MessageBull : MessageBear;
            Print("{0} | {1} | {2} | {3}", AlertName, bullish ? "BULL" : "BEAR", Bars.OpenTimes[i], msg);
        }

        private bool IsPivotHigh(int pivotIndex, int left, int right)
        {
            if (pivotIndex - left < 0 || pivotIndex + right >= Bars.Count)
                return false;
            double p = Bars.HighPrices[pivotIndex];
            for (int j = pivotIndex - left; j <= pivotIndex + right; j++)
            {
                if (j == pivotIndex)
                    continue;
                if (Bars.HighPrices[j] >= p)
                    return false;
            }
            return true;
        }

        private bool IsPivotLow(int pivotIndex, int left, int right)
        {
            if (pivotIndex - left < 0 || pivotIndex + right >= Bars.Count)
                return false;
            double p = Bars.LowPrices[pivotIndex];
            for (int j = pivotIndex - left; j <= pivotIndex + right; j++)
            {
                if (j == pivotIndex)
                    continue;
                if (Bars.LowPrices[j] <= p)
                    return false;
            }
            return true;
        }

        private int ResolveMaxSwingBack()
        {
            if (string.Equals(MaxSwingBackMethod, "Custom", StringComparison.OrdinalIgnoreCase))
                return Math.Max(1, MaxSwingBack);
            return 100000;
        }

        private void ResetPermits(List<SwingPoint> list)
        {
            int cap = Math.Min(list.Count, 2000);
            for (int i = list.Count - cap; i < list.Count; i++)
            {
                if (i >= 0 && !list[i].Permit)
                    list[i].Permit = true;
            }
        }

        private void ClearObjects()
        {
            var toDelete = new List<string>();
            foreach (var obj in Chart.Objects)
            {
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    toDelete.Add(obj.Name);
            }

            if (toDelete.Count > MaxObjects)
                toDelete.Sort(StringComparer.Ordinal);

            foreach (var name in toDelete)
                Chart.RemoveObject(name);
        }
    }
}
