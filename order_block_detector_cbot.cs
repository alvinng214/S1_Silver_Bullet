using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OrderBlockDetectorCbot : Robot
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

        [Parameter("Instance Name", Group = "General", DefaultValue = "order_block_detector_cbot")]
        public string InstanceName { get; set; }

        private OrderBlockDetector _indicator;
        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new List<ObRecord>();
        private SignalState _signal = NewEmptySignal();

        private int _lastDetectedObSourceIndex = -1;
        private readonly List<double> _haSourceOpen = new List<double>();
        private readonly List<double> _haSourceClose = new List<double>();

        protected override void OnStart()
        {
            _indicator = Indicators.GetIndicator<OrderBlockDetector>();
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
        }

        protected override void OnBar()
        {
            var index = Bars.Count - 2;
            if (index < 2)
                return;

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
                ExecuteSignalTrade(index, cond);
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

            bool detected = false;
            bool isBull = false;
            double max = 0;
            double min = 0;

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
