using System;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVGRealtime : Indicator
    {
        [Parameter("Pip Size Multiplier", DefaultValue = 1.0)]
        public double PipSizeManual { get; set; }

        [Parameter("FVG Search Lookback (Bars)", DefaultValue = 15, MinValue = 1)]
        public int IFVG_GapBars { get; set; }

        [Parameter("Min FVG Size (Pips/Points)", DefaultValue = 0.0)]
        public double MinFVG_Pips { get; set; }

        [Parameter("FVG Epsilon (Price Units)", DefaultValue = 0.0)]
        public double FVG_EpsPoints { get; set; }

        [Parameter("Show IFVG Zones", DefaultValue = true)]
        public bool ShowZones { get; set; }

        [Parameter("Zone Color Buy", DefaultValue = "#062625")]
        public Color ZoneColorBuy { get; set; }

        [Parameter("Zone Color Sell", DefaultValue = "#3E003E")]
        public Color ZoneColorSell { get; set; }

        [Parameter("MA Period", DefaultValue = 21, MinValue = 1)]
        public int MA_Period { get; set; }

        [Parameter("MA Type", DefaultValue = "EMA")]
        public string MA_Kind { get; set; }

        private const string BuySignalColorHex = "#008080"; // teal
        private const string SellSignalColorHex = "#800000"; // maroon

        private IndicatorDataSeries _maSeries;
        private IndicatorDataSeries _emaSeries;
        private int _lastAlertBar = -1;
        private int _lastAlertDir;

        protected override void Initialize()
        {
            _maSeries = CreateDataSeries();
            _emaSeries = CreateDataSeries();
        }

        public override void Calculate(int index)
        {
            // Pine source computes TickSize from PipSizeManual even though it is not used downstream.
            _ = Symbol.TickSize * PipSizeManual;
            var minSizeValue = MinFVG_Pips * (Symbol.TickSize * 10.0);
            var calcEps = FVG_EpsPoints;

            var maVal = CalculateMa(index);
            var signalDir = 0;

            RemoveRealtimeZoneObjects(index);

            for (var i = 1; i <= IFVG_GapBars; i++)
            {
                var fvgType = DetectFvg(index, i, calcEps);
                if (fvgType == 0)
                    continue;

                if (fvgType == 1)
                {
                    var gapLow = Bars.HighPrices[index - (i + 2)];
                    var gapHigh = Bars.LowPrices[index - i];

                    if ((gapHigh - gapLow) >= minSizeValue)
                    {
                        var alreadyBroken = false;
                        if (i > 1)
                        {
                            for (var k = i - 1; k >= 1; k--)
                            {
                                if (Bars.ClosePrices[index - k] < gapLow)
                                {
                                    alreadyBroken = true;
                                    break;
                                }
                            }
                        }

                        if (!alreadyBroken && Bars.ClosePrices[index] < gapLow)
                        {
                            var maCondition = !double.IsNaN(maVal) && !double.IsNaN(_maSeries[index - 1]) && maVal < _maSeries[index - 1] && Bars.ClosePrices[index] < maVal;
                            if (maCondition)
                            {
                                signalDir = -1;
                                if (ShowZones)
                                    DrawIfvgZone(index, i, gapHigh, gapLow, false);
                                break;
                            }
                        }
                    }
                }
                else if (fvgType == -1)
                {
                    var gapLow2 = Bars.HighPrices[index - i];
                    var gapHigh2 = Bars.LowPrices[index - (i + 2)];

                    if ((gapHigh2 - gapLow2) >= minSizeValue)
                    {
                        var alreadyBroken2 = false;
                        if (i > 1)
                        {
                            for (var k = i - 1; k >= 1; k--)
                            {
                                if (Bars.ClosePrices[index - k] > gapHigh2)
                                {
                                    alreadyBroken2 = true;
                                    break;
                                }
                            }
                        }

                        if (!alreadyBroken2 && Bars.ClosePrices[index] > gapHigh2)
                        {
                            var maCondition2 = !double.IsNaN(maVal) && !double.IsNaN(_maSeries[index - 1]) && maVal > _maSeries[index - 1] && Bars.ClosePrices[index] > maVal;
                            if (maCondition2)
                            {
                                signalDir = 1;
                                if (ShowZones)
                                    DrawIfvgZone(index, i, gapHigh2, gapLow2, true);
                                break;
                            }
                        }
                    }
                }
            }

            DrawSignals(index, signalDir);
            EmitAlerts(index, signalDir);
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 >= currentIndex)
                return 0;

            var h2 = Bars.HighPrices[currentIndex - (idx + 2)];
            var l2 = Bars.LowPrices[currentIndex - (idx + 2)];
            var lt = Bars.LowPrices[currentIndex - idx];
            var ht = Bars.HighPrices[currentIndex - idx];

            if (lt > h2 - epsVal)
                return 1;

            if (ht < l2 + epsVal)
                return -1;

            return 0;
        }

        private double CalculateMa(int index)
        {
            if (string.Equals(MA_Kind, "SMA", StringComparison.Ordinal))
            {
                if (index < MA_Period - 1)
                {
                    _maSeries[index] = double.NaN;
                    return _maSeries[index];
                }

                var sum = 0.0;
                for (var i = index - MA_Period + 1; i <= index; i++)
                    sum += Bars.ClosePrices[i];

                _maSeries[index] = sum / MA_Period;
                return _maSeries[index];
            }

            var alpha = 2.0 / (MA_Period + 1.0);
            if (index == 0)
            {
                _emaSeries[index] = Bars.ClosePrices[index];
                _maSeries[index] = _emaSeries[index];
                return _maSeries[index];
            }

            _emaSeries[index] = (alpha * Bars.ClosePrices[index]) + ((1.0 - alpha) * _emaSeries[index - 1]);
            _maSeries[index] = _emaSeries[index];
            return _maSeries[index];
        }

        private void DrawIfvgZone(int index, int i, double top, double bottom, bool isBuy)
        {
            var leftIndex = index - (i + 2);
            var zoneId = $"ifvg_zone_{index}_{i}_{(isBuy ? "buy" : "sell")}";
            var baseColor = isBuy ? ZoneColorBuy : ZoneColorSell;
            var zoneColor = Color.FromArgb(153, baseColor.R, baseColor.G, baseColor.B);

            var rect = Chart.DrawRectangle(zoneId, leftIndex, top, index, bottom, zoneColor);
            rect.IsFilled = true;
            rect.IsInteractive = false;
        }

        private void RemoveRealtimeZoneObjects(int index)
        {
            for (var i = 1; i <= IFVG_GapBars; i++)
            {
                Chart.RemoveObject($"ifvg_zone_{index}_{i}_buy");
                Chart.RemoveObject($"ifvg_zone_{index}_{i}_sell");
            }
        }

        private void DrawSignals(int index, int signalDir)
        {
            var buyId = $"ifvg_buy_{index}";
            var sellId = $"ifvg_sell_{index}";

            if (signalDir == 1)
                Chart.DrawIcon(buyId, ChartIconType.UpTriangle, index, Bars.LowPrices[index], Color.FromHex(BuySignalColorHex));
            else
                Chart.RemoveObject(buyId);

            if (signalDir == -1)
                Chart.DrawIcon(sellId, ChartIconType.DownTriangle, index, Bars.HighPrices[index], Color.FromHex(SellSignalColorHex));
            else
                Chart.RemoveObject(sellId);
        }

        private void EmitAlerts(int index, int signalDir)
        {
            if (index != Bars.Count - 1)
                return;

            if (signalDir != 0 && (_lastAlertBar != index || _lastAlertDir != signalDir))
            {
                if (signalDir == 1)
                    Print("IFVG Buy Signal (Realtime)");
                else
                    Print("IFVG Sell Signal (Realtime)");

                Print("IFVG Signal Triggered");
                _lastAlertBar = index;
                _lastAlertDir = signalDir;
            }
        }
    }
}
