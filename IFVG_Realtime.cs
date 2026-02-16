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
        public int FvgGapBars { get; set; }

        [Parameter("Min FVG Size (Pips/Points)", DefaultValue = 0.0)]
        public double MinFvgPips { get; set; }

        [Parameter("FVG Epsilon (Price Units)", DefaultValue = 0.0)]
        public double FvgEpsilonPoints { get; set; }

        [Parameter("Show IFVG Zones", DefaultValue = true)]
        public bool ShowZones { get; set; }

        [Parameter("Zone Color Buy", DefaultValue = "#062625")]
        public Color ZoneColorBuy { get; set; }

        [Parameter("Zone Color Sell", DefaultValue = "#3E003E")]
        public Color ZoneColorSell { get; set; }

        [Parameter("MA Period", DefaultValue = 21, MinValue = 1)]
        public int MaPeriod { get; set; }

        [Parameter("MA Type", DefaultValue = "EMA")]
        public string MaType { get; set; }

        private const string BuySignalColorHex = "#008080"; // teal
        private const string SellSignalColorHex = "#800000"; // maroon
        private const double PipsToPointsMultiplier = 10.0;
        private const string BuyAlertMessage = "IFVG Buy Signal (Realtime) | IFVG Signal Triggered";
        private const string SellAlertMessage = "IFVG Sell Signal (Realtime) | IFVG Signal Triggered";

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
            var minSizeValue = MinFvgPips * (Symbol.TickSize * PipsToPointsMultiplier);
            var calcEps = FvgEpsilonPoints;

            var maVal = CalculateMa(index);
            var signalDir = 0;

            RemoveRealtimeZoneObjects(index);

            for (var i = 1; i <= FvgGapBars; i++)
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
            if (string.Equals(MaType, "SMA", StringComparison.Ordinal))
            {
                if (index < MaPeriod - 1)
                {
                    _maSeries[index] = double.NaN;
                    return _maSeries[index];
                }

                var sum = 0.0;
                for (var i = index - MaPeriod + 1; i <= index; i++)
                    sum += Bars.ClosePrices[i];

                _maSeries[index] = sum / MaPeriod;
                return _maSeries[index];
            }

            var alpha = 2.0 / (MaPeriod + 1.0);
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
            for (var i = 1; i <= FvgGapBars; i++)
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
                Print(signalDir == 1 ? BuyAlertMessage : SellAlertMessage);
                _lastAlertBar = index;
                _lastAlertDir = signalDir;
            }
        }
    }
}
