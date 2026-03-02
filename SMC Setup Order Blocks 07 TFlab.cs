using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMCSetupOrderBlocks07TFlab : Indicator
    {
        private const string Prefix = "SMC07_";

        private sealed class PivotNode
        {
            public string Type;
            public double Value;
            public int Index;
        }

        [Parameter("Pivot Period", DefaultValue = 2, MinValue = 2)]
        public int PP { get; set; }

        [Parameter("Order Block Validity Period (Bar)", Group = "Global Setting", DefaultValue = 500, MinValue = 10, MaxValue = 4998)]
        public int OBVaP { get; set; }

        [Parameter("Mitigation Level OB", DefaultValue = "Proximal")]
        public string MLOB { get; set; }

        [Parameter("Order Block Refine", Group = "Order Blocks Refinement", DefaultValue = true)]
        public bool Refine { get; set; }

        [Parameter("Refine Method", Group = "Order Blocks Refinement", DefaultValue = "Defensive")]
        public string RefineMe { get; set; }

        private readonly List<PivotNode> _array = new List<PivotNode>();

        protected override void Initialize()
        {
        }

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1)
                return;

            ClearObjects();
            _array.Clear();

            int beObi = -1;
            int buObi = -1;

            for (int i = PP; i < Bars.Count - PP; i++)
            {
                bool highPivot = IsPivotHigh(i, PP);
                bool lowPivot = IsPivotLow(i, PP);

                double highValue = Bars.HighPrices[i];
                double lowValue = Bars.LowPrices[i];
                int highIndex = i;
                int lowIndex = i;

                if (highPivot && lowPivot)
                {
                    ProcessBothPivots(i, highValue, lowValue, highIndex, lowIndex);
                }
                else if (highPivot)
                {
                    ProcessHighPivot(highValue, highIndex, lowValue, lowIndex);
                }
                else if (lowPivot)
                {
                    ProcessLowPivot(lowValue, lowIndex, highValue, highIndex);
                }

                if (_array.Count > 8)
                {
                    var p0 = _array[_array.Count - 1];
                    var p1 = _array[_array.Count - 2];
                    var p2 = _array[_array.Count - 3];
                    var p3 = _array[_array.Count - 4];
                    var p4 = _array[_array.Count - 5];
                    var p5 = _array[_array.Count - 6];
                    var p6 = _array[_array.Count - 7];

                    bool bearSetup = p0.Type == "LL" && p1.Type == "LH" && p2.Type == "HL" && p3.Type == "LH" && p4.Type == "LL" && p5.Type == "LH" && p6.Type == "LL"
                                     && p0.Value > p4.Value && p0.Index == i;

                    bool bullSetup = p0.Type == "HH" && p1.Type == "HL" && p2.Type == "LH" && p3.Type == "HL" && p4.Type == "HH" && p5.Type == "HL" && p6.Type == "HH"
                                     && p0.Value < p4.Value && p0.Index == i;

                    if (bearSetup)
                    {
                        beObi = p5.Index;
                        DrawSignal(false, i);
                        DrawOrderBlock(false, beObi, i);
                    }

                    if (bullSetup)
                    {
                        buObi = p5.Index;
                        DrawSignal(true, i);
                        DrawOrderBlock(true, buObi, i);
                    }
                }
            }
        }

        private void ProcessBothPivots(int i, double highValue, double lowValue, int highIndex, int lowIndex)
        {
            if (_array.Count == 0)
                return;

            var last = _array[_array.Count - 1].Type;

            if (last == "L" || last == "LL")
            {
                if (lowValue < _array[_array.Count - 1].Value)
                {
                    RemoveLast();
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                }
                else
                {
                    Push(BuildHighType(highValue), highValue, highIndex);
                }
            }
            else if (last == "H" || last == "HH")
            {
                if (highValue > _array[_array.Count - 1].Value)
                {
                    RemoveLast();
                    Push(BuildHighType(highValue), highValue, highIndex);
                }
                else
                {
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                }
            }
            else if (last == "LH")
            {
                if (highValue < _array[_array.Count - 1].Value)
                {
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                }
                else if (highValue > _array[_array.Count - 1].Value)
                {
                    if (Bars.ClosePrices[i] < _array[_array.Count - 1].Value)
                    {
                        RemoveLast();
                        Push(BuildHighType(highValue), highValue, highIndex);
                    }
                    else if (Bars.ClosePrices[i] > _array[_array.Count - 1].Value)
                    {
                        Push(BuildLowType(lowValue), lowValue, lowIndex);
                    }
                }
            }
            else if (last == "HL")
            {
                if (lowValue > _array[_array.Count - 1].Value)
                {
                    Push(BuildHighType(highValue), highValue, highIndex);
                }
                else if (lowValue < _array[_array.Count - 1].Value)
                {
                    if (Bars.ClosePrices[i] > _array[_array.Count - 1].Value)
                    {
                        RemoveLast();
                        Push(BuildLowType(lowValue), lowValue, lowIndex);
                    }
                    else if (Bars.ClosePrices[i] < _array[_array.Count - 1].Value)
                    {
                        Push(BuildHighType(highValue), highValue, highIndex);
                    }
                }
            }
        }

        private void ProcessHighPivot(double highValue, int highIndex, double lowValue, int lowIndex)
        {
            if (_array.Count == 0)
            {
                Push("H", highValue, highIndex);
                return;
            }

            var last = _array[_array.Count - 1].Type;
            if (last == "L" || last == "HL" || last == "LL")
            {
                if (_array[_array.Count - 1].Value < highValue)
                    Push(BuildHighType(highValue), highValue, highIndex);
                else
                {
                    RemoveLast();
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                }
            }
            else if (last == "H" || last == "HH" || last == "LH")
            {
                if (_array[_array.Count - 1].Value < highValue)
                {
                    RemoveLast();
                    Push(BuildHighType(highValue), highValue, highIndex);
                }
            }
        }

        private void ProcessLowPivot(double lowValue, int lowIndex, double highValue, int highIndex)
        {
            if (_array.Count == 0)
            {
                Push("L", lowValue, lowIndex);
                return;
            }

            var last = _array[_array.Count - 1].Type;
            if (last == "H" || last == "HH" || last == "LH")
            {
                if (lowValue < _array[_array.Count - 1].Value)
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                else
                {
                    RemoveLast();
                    Push(BuildHighType(highValue), highValue, highIndex);
                }
            }
            else if (last == "L" || last == "HL" || last == "LL")
            {
                if (_array[_array.Count - 1].Value > lowValue)
                {
                    RemoveLast();
                    Push(BuildLowType(lowValue), lowValue, lowIndex);
                }
            }
        }

        private string BuildHighType(double highValue)
        {
            if (_array.Count > 2)
                return _array[_array.Count - 2].Value < highValue ? "HH" : "LH";
            return "H";
        }

        private string BuildLowType(double lowValue)
        {
            if (_array.Count > 2)
                return _array[_array.Count - 2].Value < lowValue ? "HL" : "LL";
            return "L";
        }

        private void Push(string type, double value, int index)
        {
            _array.Add(new PivotNode { Type = type, Value = value, Index = index });
        }

        private void RemoveLast()
        {
            if (_array.Count > 0)
                _array.RemoveAt(_array.Count - 1);
        }

        private void DrawSignal(bool bullish, int i)
        {
            string tag = bullish ? "BULL" : "BEAR";
            var icon = bullish ? ChartIconType.DownArrow : ChartIconType.UpArrow;
            var color = bullish ? Color.FromRgb(47, 156, 38) : Color.FromRgb(179, 2, 2);
            double y = bullish ? Bars.HighPrices[i] : Bars.LowPrices[i];
            Chart.DrawIcon(Prefix + "SIG_" + tag + "_" + i, icon, Bars.OpenTimes[i], y, color);
            Chart.DrawText(Prefix + "TXT_" + tag + "_" + i,
                bullish ? "Bull Setup07" : "Bear Setup07",
                Bars.OpenTimes[i],
                bullish ? y + Symbol.PipSize * 8 : y - Symbol.PipSize * 8,
                color);
        }

        private void DrawOrderBlock(bool bullish, int originIndex, int triggerIndex)
        {
            if (originIndex < 0 || originIndex >= Bars.Count)
                return;

            double candleHigh = Bars.HighPrices[originIndex];
            double candleLow = Bars.LowPrices[originIndex];
            double bodyHigh = Math.Max(Bars.OpenPrices[originIndex], Bars.ClosePrices[originIndex]);
            double bodyLow = Math.Min(Bars.OpenPrices[originIndex], Bars.ClosePrices[originIndex]);

            double distal;
            double proximal;

            if (bullish)
            {
                distal = candleLow;
                proximal = bodyHigh;
            }
            else
            {
                distal = candleHigh;
                proximal = bodyLow;
            }

            if (Refine)
            {
                if (RefineMe == "Defensive")
                {
                    if (bullish)
                        proximal = Math.Min(proximal, bodyLow + (bodyHigh - bodyLow) * 0.5);
                    else
                        proximal = Math.Max(proximal, bodyHigh - (bodyHigh - bodyLow) * 0.5);
                }
                else if (RefineMe == "Aggressive")
                {
                    if (bullish)
                        proximal = bodyHigh;
                    else
                        proximal = bodyLow;
                }
            }

            int endIndex = Math.Min(Bars.Count - 1, originIndex + OBVaP);
            double top = Math.Max(distal, proximal);
            double bottom = Math.Min(distal, proximal);

            var fill = bullish ? Color.FromArgb(0x67, 0x31, 0x9B, 0x2D) : Color.FromArgb(0x54, 0x52, 0x52, 0xFF);
            var box = Chart.DrawRectangle(Prefix + "OB_" + (bullish ? "D_" : "S_") + triggerIndex,
                Bars.OpenTimes[originIndex], top,
                Bars.OpenTimes[endIndex], bottom,
                fill);
            box.IsFilled = true;

            double mitigation = GetMitigation(distal, proximal);
            Chart.DrawTrendLine(Prefix + "MIT_" + (bullish ? "D_" : "S_") + triggerIndex,
                Bars.OpenTimes[originIndex], mitigation,
                Bars.OpenTimes[endIndex], mitigation,
                fill,
                1,
                LineStyle.Solid);
        }

        private double GetMitigation(double distal, double proximal)
        {
            switch (MLOB)
            {
                case "Distal":
                    return distal;
                case "50 % OB":
                    return (distal + proximal) * 0.5;
                case "Proximal":
                default:
                    return proximal;
            }
        }

        private bool IsPivotHigh(int i, int p)
        {
            if (i - p < 0 || i + p >= Bars.Count)
                return false;
            double pivot = Bars.HighPrices[i];
            for (int j = i - p; j <= i + p; j++)
            {
                if (j == i)
                    continue;
                if (Bars.HighPrices[j] >= pivot)
                    return false;
            }
            return true;
        }

        private bool IsPivotLow(int i, int p)
        {
            if (i - p < 0 || i + p >= Bars.Count)
                return false;
            double pivot = Bars.LowPrices[i];
            for (int j = i - p; j <= i + p; j++)
            {
                if (j == i)
                    continue;
                if (Bars.LowPrices[j] <= pivot)
                    return false;
            }
            return true;
        }

        private void ClearObjects()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
            {
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(obj.Name);
            }
            foreach (var name in names)
                Chart.RemoveObject(name);
        }
    }
}
