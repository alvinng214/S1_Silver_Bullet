using System;
using cAlgo.API;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVGRealtime : Indicator
    {
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
        private const string BuyAlertMessage = "IFVG Buy Signal (Realtime) | IFVG Signal Triggered";
        private const string SellAlertMessage = "IFVG Sell Signal (Realtime) | IFVG Signal Triggered";

        private IndicatorDataSeries _maSeries;
        private SimpleMovingAverage _sma;
        private ExponentialMovingAverage _ema;
        private int _lastAlertBar = -1;
        private int _lastAlertDir;

        protected override void Initialize()
        {
            _maSeries = CreateDataSeries();
            _sma = Indicators.SimpleMovingAverage(Bars.ClosePrices, MaPeriod);
            _ema = Indicators.ExponentialMovingAverage(Bars.ClosePrices, MaPeriod);
        }

        public override void Calculate(int index)
        {
            var minSizeValue = MinFvgPips * Symbol.PipSize;
            var calcEps = FvgEpsilonPoints;

            var maVal = CalculateMa(index);
            var signalDir = 0;

            RemoveRealtimeZoneObjects(index);

            for (var i = 1; i <= FvgGapBars; i++)
            {
                var fvgType = DetectFvg(index, i, calcEps);
                if (fvgType == 0)
                    continue;

                if (TryProcessFvgCandidate(index, i, fvgType, minSizeValue, maVal, out signalDir))
                    break;
            }

            DrawSignals(index, signalDir);
            EmitAlerts(index, signalDir);
        }

        private bool TryProcessFvgCandidate(int index, int i, int fvgType, double minSizeValue, double maVal, out int signalDir)
        {
            signalDir = 0;

            var isBearishGap = fvgType == 1;
            var gapLow = isBearishGap ? Bars.HighPrices[index - (i + 2)] : Bars.HighPrices[index - i];
            var gapHigh = isBearishGap ? Bars.LowPrices[index - i] : Bars.LowPrices[index - (i + 2)];

            if ((gapHigh - gapLow) < minSizeValue)
                return false;

            var alreadyBroken = false;
            if (i > 1)
            {
                for (var k = i - 1; k >= 1; k--)
                {
                    var close = Bars.ClosePrices[index - k];
                    if ((isBearishGap && close < gapLow) || (!isBearishGap && close > gapHigh))
                    {
                        alreadyBroken = true;
                        break;
                    }
                }
            }

            if (alreadyBroken)
                return false;

            var breakout = isBearishGap ? Bars.ClosePrices[index] < gapLow : Bars.ClosePrices[index] > gapHigh;
            if (!breakout)
                return false;

            var maReady = !double.IsNaN(maVal) && !double.IsNaN(_maSeries[index - 1]);
            var maCondition = isBearishGap
                ? maReady && maVal < _maSeries[index - 1] && Bars.ClosePrices[index] < maVal
                : maReady && maVal > _maSeries[index - 1] && Bars.ClosePrices[index] > maVal;

            if (!maCondition)
                return false;

            signalDir = isBearishGap ? -1 : 1;
            if (ShowZones)
                DrawIfvgZone(index, i, gapHigh, gapLow, !isBearishGap);

            return true;
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 > currentIndex)
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
            if (string.Equals(MaType, "SMA", StringComparison.OrdinalIgnoreCase))
                _maSeries[index] = _sma.Result[index];
            else
                _maSeries[index] = _ema.Result[index];

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

            if (signalDir == -1)
                Chart.DrawIcon(sellId, ChartIconType.DownTriangle, index, Bars.HighPrices[index], Color.FromHex(SellSignalColorHex));
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
