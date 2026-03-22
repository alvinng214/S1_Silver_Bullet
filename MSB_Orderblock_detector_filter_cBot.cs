using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MSB_Orderblock_detector_filter_cBot : Robot
    {
        private sealed class ObRecord
        {
            public double Max;
            public double Min;
            public bool IsBull;
            public DateTime DetectionTime;
            public int DetectionChartIndex;
        }

        private sealed class SignalState
        {
            public double Point;
            public bool IsBull;
            public bool Entry;
            public int Index;
            public DateTime Time;
        }

        private sealed class StructureZone
        {
            public string Label;
            public double Top;
            public double Bottom;
            public int CreatedIndex;
        }

        [Parameter("SL Lookback Bars", Group = "Risk", DefaultValue = 5, MinValue = 1)]
        public int StopLossLookbackBars { get; set; }

        [Parameter("Risk %", Group = "Risk", DefaultValue = 1.0, MinValue = 0.01, Step = 0.01)]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", Group = "Risk", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1)]
        public double RiskRewardRatio { get; set; }

        [Parameter("Use Chart Timeframe", Group = "Timeframe", DefaultValue = true)]
        public bool UseChartTimeframe { get; set; }

        [Parameter("Time-Frame Order-Block", Group = "Timeframe", DefaultValue = "Hour")]
        public TimeFrame InputTimeFrame { get; set; }

        [Parameter("Min dist", Group = "Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDist { get; set; }

        [Parameter("Use Heikin-Ashi", Group = "Signals", DefaultValue = false)]
        public bool UseHeikinAshi { get; set; }

        [Parameter("Instance Name", Group = "General", DefaultValue = "MSB_Orderblock_detector_filter_cBot")]
        public string InstanceName { get; set; }

        [Parameter("Enable Filter 1 (Bu-OB / Be-OB Touch)", Group = "MSB Filters", DefaultValue = true)]
        public bool EnableFilter1 { get; set; }

        [Parameter("Filter 1 Lookback Bars", Group = "MSB Filters", DefaultValue = 10, MinValue = 1, MaxValue = 500)]
        public int Filter1LookbackBars { get; set; }

        [Parameter("Enable Filter 2 (Bu-BB/Bu-MB / Be-BB/Be-MB Touch)", Group = "MSB Filters", DefaultValue = true)]
        public bool EnableFilter2 { get; set; }

        [Parameter("Filter 2 Lookback Bars", Group = "MSB Filters", DefaultValue = 10, MinValue = 1, MaxValue = 500)]
        public int Filter2LookbackBars { get; set; }

        [Parameter("ZigZag Length", Group = "MSB Settings", DefaultValue = 9, MinValue = 1)]
        public int ZigZagLen { get; set; }

        [Parameter("Fib Factor for breakout confirmation", Group = "MSB Settings", DefaultValue = 0.33, MinValue = 0.0, MaxValue = 1.0, Step = 0.01)]
        public double FibFactor { get; set; }

        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new List<ObRecord>();
        private SignalState _signal = NewEmptySignal();

        private int _lastDetectedObSourceIndex = -1;
        private readonly List<double> _haSourceOpen = new List<double>();
        private readonly List<double> _haSourceClose = new List<double>();

        private int _trend = 1;
        private int _market = 1;
        private int _lastToUpBar = -1;
        private int _lastToDownBar = -1;
        private double _lastMsbL0 = double.NaN;
        private double _lastMsbH0 = double.NaN;
        private int _lastProcessedMsbIndex = -1;

        private readonly List<double> _highPoints = new List<double>();
        private readonly List<int> _highIndices = new List<int>();
        private readonly List<double> _lowPoints = new List<double>();
        private readonly List<int> _lowIndices = new List<int>();

        private readonly List<StructureZone> _bullishObZones = new List<StructureZone>();
        private readonly List<StructureZone> _bearishObZones = new List<StructureZone>();
        private readonly List<StructureZone> _bullishBbMbZones = new List<StructureZone>();
        private readonly List<StructureZone> _bearishBbMbZones = new List<StructureZone>();

        protected override void OnStart()
        {
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            SeedHistoricalState();
        }

        protected override void OnBar()
        {
            var index = Bars.Count - 2;
            if (index < 2)
                return;

            UpdateMsbState(index);

            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
            if (sourceIndex < 2)
                return;

            EnsureHeikinAshiSource(sourceIndex);
            DetectOrderBlock(index, sourceIndex);

            var sHigh = _sourceBars.HighPrices[sourceIndex];
            var sLow = _sourceBars.LowPrices[sourceIndex];
            var sClose = _sourceBars.ClosePrices[sourceIndex];

            HandleMitigationOb(index, sLow, sHigh);

            var candleDir = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var cond = 0;

            var signalClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            if (signalClose > _signal.Point && _signal.IsBull && candleDir == 1 && !_signal.Entry)
            {
                _signal.Entry = true;
                cond = 1;
            }

            if (signalClose < _signal.Point && !_signal.IsBull && candleDir == -1 && !_signal.Entry)
            {
                _signal.Entry = true;
                cond = -1;
            }

            if (cond != 0)
            {
                if (PassesMsbFilters(index, cond))
                    ExecuteSignalTrade(index, cond);
                else
                    Print("Trade blocked by MSB filters at bar {0}. Direction={1}", index, cond > 0 ? "Long" : "Short");
            }
        }

        private void SeedHistoricalState()
        {
            var chartMaxIndex = Bars.Count - 2;
            if (chartMaxIndex < 2)
                return;

            for (var index = 2; index <= chartMaxIndex; index++)
            {
                UpdateMsbState(index);

                var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
                if (sourceIndex < 2)
                    continue;

                EnsureHeikinAshiSource(sourceIndex);
                DetectOrderBlock(index, sourceIndex);

                var sHigh = _sourceBars.HighPrices[sourceIndex];
                var sLow = _sourceBars.LowPrices[sourceIndex];
                var sClose = _sourceBars.ClosePrices[sourceIndex];
                HandleMitigationOb(index, sLow, sHigh);

                var candleDir = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
                var signalClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;

                if (signalClose > _signal.Point && _signal.IsBull && candleDir == 1 && !_signal.Entry)
                    _signal.Entry = true;

                if (signalClose < _signal.Point && !_signal.IsBull && candleDir == -1 && !_signal.Entry)
                    _signal.Entry = true;
            }
        }

        private void ExecuteSignalTrade(int signalBarIndex, int direction)
        {
            var tradeType = direction > 0 ? TradeType.Buy : TradeType.Sell;
            var stopLossPrice = direction > 0
                ? GetLowestLowBeforeSignal(signalBarIndex, StopLossLookbackBars)
                : GetHighestHighBeforeSignal(signalBarIndex, StopLossLookbackBars);

            if (double.IsNaN(stopLossPrice))
                return;

            var entryPrice = direction > 0 ? Symbol.Ask : Symbol.Bid;
            var stopLossPips = direction > 0
                ? (entryPrice - stopLossPrice) / Symbol.PipSize
                : (stopLossPrice - entryPrice) / Symbol.PipSize;

            if (stopLossPips <= 0)
            {
                Print("Skipped trade. Invalid SL distance. Direction={0}, Entry={1}, SL={2}", direction, entryPrice, stopLossPrice);
                return;
            }

            var riskAmount = Account.Balance * (RiskPercent / 100.0);
            var rawVolumeInUnits = riskAmount / (stopLossPips * Symbol.PipValue);
            var volumeInUnits = Symbol.NormalizeVolumeInUnits(rawVolumeInUnits, RoundingMode.Down);

            if (volumeInUnits < Symbol.VolumeInUnitsMin)
                volumeInUnits = Symbol.VolumeInUnitsMin;
            if (volumeInUnits > Symbol.VolumeInUnitsMax)
                volumeInUnits = Symbol.VolumeInUnitsMax;

            var takeProfitPips = stopLossPips * RiskRewardRatio;
            if (takeProfitPips <= 0)
            {
                Print("Skipped trade. Invalid TP distance. R:R={0}", RiskRewardRatio);
                return;
            }

            ExecuteMarketOrder(tradeType, SymbolName, volumeInUnits, InstanceName, stopLossPips, takeProfitPips);
        }

        private bool PassesMsbFilters(int signalBarIndex, int direction)
        {
            if (direction > 0)
            {
                if (EnableFilter1 && !HasRecentTouch(_bullishObZones, signalBarIndex, Filter1LookbackBars))
                    return false;

                if (EnableFilter2 && !HasRecentTouch(_bullishBbMbZones, signalBarIndex, Filter2LookbackBars))
                    return false;
            }
            else
            {
                if (EnableFilter1 && !HasRecentTouch(_bearishObZones, signalBarIndex, Filter1LookbackBars))
                    return false;

                if (EnableFilter2 && !HasRecentTouch(_bearishBbMbZones, signalBarIndex, Filter2LookbackBars))
                    return false;
            }

            return true;
        }

        private bool HasRecentTouch(List<StructureZone> zones, int signalBarIndex, int lookbackBars)
        {
            if (zones.Count == 0)
                return false;

            var from = Math.Max(0, signalBarIndex - lookbackBars + 1);
            for (var barIndex = signalBarIndex; barIndex >= from; barIndex--)
            {
                var low = Bars.LowPrices[barIndex];
                var high = Bars.HighPrices[barIndex];

                for (var i = zones.Count - 1; i >= 0; i--)
                {
                    var zone = zones[i];
                    if (zone.CreatedIndex > barIndex)
                        continue;

                    if (low <= zone.Top && high >= zone.Bottom)
                        return true;
                }
            }

            return false;
        }

        private double GetLowestLowBeforeSignal(int signalBarIndex, int lookbackBars)
        {
            var from = Math.Max(0, signalBarIndex - lookbackBars + 1);
            var to = signalBarIndex;
            if (to < from)
                return double.NaN;

            var lowest = double.MaxValue;
            for (var i = from; i <= to; i++)
                lowest = Math.Min(lowest, Bars.LowPrices[i]);
            return lowest;
        }

        private double GetHighestHighBeforeSignal(int signalBarIndex, int lookbackBars)
        {
            var from = Math.Max(0, signalBarIndex - lookbackBars + 1);
            var to = signalBarIndex;
            if (to < from)
                return double.NaN;

            var highest = double.MinValue;
            for (var i = from; i <= to; i++)
                highest = Math.Max(highest, Bars.HighPrices[i]);
            return highest;
        }

        private void DetectOrderBlock(int chartIndex, int sourceIndex)
        {
            if (sourceIndex == _lastDetectedObSourceIndex)
                return;

            var candleDir = _sourceBars.ClosePrices[sourceIndex] > _sourceBars.OpenPrices[sourceIndex] ? 1 : -1;
            var candleDirPrev = _sourceBars.ClosePrices[sourceIndex - 1] > _sourceBars.OpenPrices[sourceIndex - 1] ? 1 : -1;

            var detected = false;
            var isBull = false;
            var max = 0.0;
            var min = 0.0;

            if (candleDir == 1 && candleDirPrev == -1 && _sourceBars.HighPrices[sourceIndex] > _sourceBars.HighPrices[sourceIndex - 1])
            {
                detected = true;
                isBull = true;
                max = _sourceBars.HighPrices[sourceIndex - 1];
                min = _sourceBars.LowPrices[sourceIndex - 1];
            }

            if (candleDir == -1 && candleDirPrev == 1 && _sourceBars.LowPrices[sourceIndex] < _sourceBars.LowPrices[sourceIndex - 1])
            {
                detected = true;
                isBull = false;
                max = _sourceBars.HighPrices[sourceIndex - 1];
                min = _sourceBars.LowPrices[sourceIndex - 1];
            }

            if (!detected)
                return;

            _obRecords.Insert(0, new ObRecord
            {
                Max = max,
                Min = min,
                IsBull = isBull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex],
                DetectionChartIndex = chartIndex
            });

            _lastDetectedObSourceIndex = sourceIndex;
        }

        private void HandleMitigationOb(int index, double sLow, double sHigh)
        {
            for (var i = _obRecords.Count - 1; i >= 0; i--)
            {
                var r = _obRecords[i];
                var now = Bars.OpenTimes[index];

                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        _obRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(index, r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        _obRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(index, r.Min, false);
                    }
                }
            }
        }

        private void UpdateMsbState(int index)
        {
            if (index <= _lastProcessedMsbIndex || index < ZigZagLen)
                return;

            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];

            var highest = GetHighest(index, ZigZagLen);
            var lowest = GetLowest(index, ZigZagLen);
            var toUp = high >= highest;
            var toDown = low <= lowest;

            var lastTrendUpSince = _lastToUpBar >= 0 ? index - 1 - _lastToUpBar : index;
            var lastTrendDownSince = _lastToDownBar >= 0 ? index - 1 - _lastToDownBar : index;

            var lowWindow = Math.Max(lastTrendUpSince, 1);
            var lowVal = GetLowest(index, lowWindow);
            var lowIndex = FindLastLowMatch(index, lowVal, lowWindow);

            var highWindow = Math.Max(lastTrendDownSince, 1);
            var highVal = GetHighest(index, highWindow);
            var highIndex = FindLastHighMatch(index, highVal, highWindow);

            var previousTrend = _trend;
            if (_trend == 1 && toDown)
                _trend = -1;
            else if (_trend == -1 && toUp)
                _trend = 1;

            var trendChanged = _trend != previousTrend;
            if (trendChanged)
            {
                if (_trend == 1)
                {
                    _lowPoints.Add(lowVal);
                    _lowIndices.Add(lowIndex);
                }
                else
                {
                    _highPoints.Add(highVal);
                    _highIndices.Add(highIndex);
                }

                if (_highPoints.Count > 5)
                {
                    _highPoints.RemoveAt(0);
                    _highIndices.RemoveAt(0);
                }

                if (_lowPoints.Count > 5)
                {
                    _lowPoints.RemoveAt(0);
                    _lowIndices.RemoveAt(0);
                }
            }

            var h0 = _highPoints.Count >= 1 ? _highPoints[_highPoints.Count - 1] : double.NaN;
            var h0i = _highIndices.Count >= 1 ? _highIndices[_highIndices.Count - 1] : -1;
            var h1 = _highPoints.Count >= 2 ? _highPoints[_highPoints.Count - 2] : double.NaN;
            var h1i = _highIndices.Count >= 2 ? _highIndices[_highIndices.Count - 2] : -1;
            var l0 = _lowPoints.Count >= 1 ? _lowPoints[_lowPoints.Count - 1] : double.NaN;
            var l0i = _lowIndices.Count >= 1 ? _lowIndices[_lowIndices.Count - 1] : -1;
            var l1 = _lowPoints.Count >= 2 ? _lowPoints[_lowPoints.Count - 2] : double.NaN;
            var l1i = _lowIndices.Count >= 2 ? _lowIndices[_lowIndices.Count - 2] : -1;

            var previousMarket = _market;
            var allPivotsAvailable = !double.IsNaN(l0) && !double.IsNaN(l1) && !double.IsNaN(h0) && !double.IsNaN(h1);
            var guardBlocked = (!double.IsNaN(_lastMsbL0) && l0 == _lastMsbL0) || (!double.IsNaN(_lastMsbH0) && h0 == _lastMsbH0);

            if (allPivotsAvailable && !guardBlocked)
            {
                if (_market == 1 && l0 < l1 && l0 < l1 - Math.Abs(h0 - l1) * FibFactor)
                    _market = -1;
                else if (_market == -1 && h0 > h1 && h0 > h1 + Math.Abs(h1 - l0) * FibFactor)
                    _market = 1;
            }

            var marketChanged = _market != previousMarket;
            if (marketChanged)
            {
                _lastMsbL0 = l0;
                _lastMsbH0 = h0;

                if (_market == 1)
                    OnBullishMsb(index, h0i, h1i, l0, l0i, l1, l1i);
                else
                    OnBearishMsb(index, h0, h0i, h1, h1i, l0i, l1i);
            }

            PruneBrokenZones(index);

            if (toUp)
                _lastToUpBar = index;
            if (toDown)
                _lastToDownBar = index;

            _lastProcessedMsbIndex = index;
        }

        private void OnBullishMsb(int index, int h0i, int h1i, double l0, int l0i, double l1, int l1i)
        {
            var buObBar = FindLastCandle(h1i, l0i, true, index);
            if (buObBar >= 0)
                _bullishObZones.Add(new StructureZone { Label = "Bu-OB", Top = Bars.HighPrices[buObBar], Bottom = Bars.LowPrices[buObBar], CreatedIndex = index });

            var buBbBar = FindLastCandle(l1i - ZigZagLen, h1i, false, index);
            if (buBbBar >= 0)
            {
                var label = l0 < l1 ? "Bu-BB" : "Bu-MB";
                _bullishBbMbZones.Add(new StructureZone { Label = label, Top = Bars.HighPrices[buBbBar], Bottom = Bars.LowPrices[buBbBar], CreatedIndex = index });
            }
        }

        private void OnBearishMsb(int index, double h0, int h0i, double h1, int h1i, int l0i, int l1i)
        {
            var beObBar = FindLastCandle(l1i, h0i, false, index);
            if (beObBar >= 0)
                _bearishObZones.Add(new StructureZone { Label = "Be-OB", Top = Bars.HighPrices[beObBar], Bottom = Bars.LowPrices[beObBar], CreatedIndex = index });

            var beBbBar = FindLastCandle(h1i - ZigZagLen, l1i, true, index);
            if (beBbBar >= 0)
            {
                var label = h0 > h1 ? "Be-BB" : "Be-MB";
                _bearishBbMbZones.Add(new StructureZone { Label = label, Top = Bars.HighPrices[beBbBar], Bottom = Bars.LowPrices[beBbBar], CreatedIndex = index });
            }
        }

        private void PruneBrokenZones(int index)
        {
            var close = Bars.ClosePrices[index];
            RemoveBrokenBullishZones(_bullishObZones, close);
            RemoveBrokenBullishZones(_bullishBbMbZones, close);
            RemoveBrokenBearishZones(_bearishObZones, close);
            RemoveBrokenBearishZones(_bearishBbMbZones, close);
        }

        private static void RemoveBrokenBullishZones(List<StructureZone> zones, double close)
        {
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                if (close < zones[i].Bottom)
                    zones.RemoveAt(i);
            }
        }

        private static void RemoveBrokenBearishZones(List<StructureZone> zones, double close)
        {
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                if (close > zones[i].Top)
                    zones.RemoveAt(i);
            }
        }

        private int FindLastCandle(int fromBar, int toBar, bool bearish, int maxBar)
        {
            var start = Math.Max(0, Math.Min(fromBar, toBar));
            var end = Math.Min(maxBar, Math.Max(fromBar, toBar));
            var result = -1;

            for (var i = start; i <= end; i++)
            {
                var o = Bars.OpenPrices[i];
                var c = Bars.ClosePrices[i];
                var match = bearish ? o > c : o < c;
                if (match)
                    result = i;
            }

            return result;
        }

        private double GetHighest(int index, int length)
        {
            var max = double.MinValue;
            for (var i = index; i >= Math.Max(0, index - length + 1); i--)
                max = Math.Max(max, Bars.HighPrices[i]);
            return max;
        }

        private double GetLowest(int index, int length)
        {
            var min = double.MaxValue;
            for (var i = index; i >= Math.Max(0, index - length + 1); i--)
                min = Math.Min(min, Bars.LowPrices[i]);
            return min;
        }

        private int FindLastLowMatch(int index, double value, int window)
        {
            for (var i = index; i >= Math.Max(0, index - window + 1); i--)
            {
                if (Bars.LowPrices[i] == value)
                    return i;
            }

            return index;
        }

        private int FindLastHighMatch(int index, double value, int window)
        {
            for (var i = index; i >= Math.Max(0, index - window + 1); i--)
            {
                if (Bars.HighPrices[i] == value)
                    return i;
            }

            return index;
        }

        private SignalState NewSignal(int index, double point, bool isBull)
        {
            return new SignalState
            {
                Point = point,
                IsBull = isBull,
                Entry = false,
                Index = index,
                Time = Bars.OpenTimes[index]
            };
        }

        private static SignalState NewEmptySignal()
        {
            return new SignalState
            {
                Point = double.NaN,
                Entry = false
            };
        }

        private void EnsureHeikinAshiSource(int sourceIndex)
        {
            while (_haSourceClose.Count <= sourceIndex)
            {
                var i = _haSourceClose.Count;
                var close = (_sourceBars.OpenPrices[i] + _sourceBars.HighPrices[i] + _sourceBars.LowPrices[i] + _sourceBars.ClosePrices[i]) / 4.0;
                var open = i == 0
                    ? (_sourceBars.OpenPrices[i] + _sourceBars.ClosePrices[i]) / 2.0
                    : (_haSourceOpen[i - 1] + _haSourceClose[i - 1]) / 2.0;
                _haSourceOpen.Add(open);
                _haSourceClose.Add(close);
            }
        }

        private static int FindBarIndexAtOrBefore(Bars bars, DateTime t)
        {
            var times = bars.OpenTimes;
            var left = 0;
            var right = times.Count - 1;
            var ans = -1;

            while (left <= right)
            {
                var mid = (left + right) / 2;
                if (times[mid] <= t)
                {
                    ans = mid;
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }

            return ans;
        }
    }
}
