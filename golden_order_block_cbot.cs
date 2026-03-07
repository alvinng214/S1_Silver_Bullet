using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class GoldenOrderBlockDetectorCbot : Robot
    {
        [Parameter("Structure Period", DefaultValue = 10, MinValue = 1, Group = "Structure")]
        public int prd { get; set; }

        [Parameter("Bullish Structure", DefaultValue = true, Group = "Structure")]
        public bool bull { get; set; }

        [Parameter("Bullish Color", DefaultValue = "#08EC32", Group = "Structure")]
        public Color bull2 { get; set; }

        [Parameter("Bearish Structure", DefaultValue = true, Group = "Structure")]
        public bool bear { get; set; }

        [Parameter("Bearish Color", DefaultValue = "#FF2222", Group = "Structure")]
        public Color bear2 { get; set; }

        [Parameter("BoS Width", DefaultValue = 1, MinValue = 1, MaxValue = 10, Group = "Structure")]
        public int s_width { get; set; }

        [Parameter("Swing tracker", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool follow { get; set; }

        [Parameter("Swing Line", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool swingline { get; set; }

        [Parameter("Swing Line Width", DefaultValue = 2, MinValue = 1, MaxValue = 10, Group = "Fibonacci Mode")]
        public int swline_width { get; set; }

        [Parameter("Swing Labels", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool swinglab { get; set; }

        [Parameter("Previous", DefaultValue = false, Group = "Fibonacci")]
        public bool showOld { get; set; }

        [Parameter("Extend", DefaultValue = true, Group = "Fibonacci")]
        public bool extend { get; set; }

        [Parameter("Fill Golden Zone", DefaultValue = false, Group = "Fibonacci")]
        public bool golden { get; set; }

        [Parameter("Bullish Golden Zone Color", DefaultValue = "#9900FF00", Group = "Fibonacci")]
        public Color bullGoldZone { get; set; }

        [Parameter("Bearish Golden Zone Color", DefaultValue = "#99FF0000", Group = "Fibonacci")]
        public Color bearGoldZone { get; set; }

        [Parameter("L1 Enabled", DefaultValue = true, Group = "Levels")]
        public bool level1Enabled { get; set; }

        [Parameter("L1 Value", DefaultValue = 0.618, Group = "Levels")]
        public double level1Value { get; set; }

        [Parameter("L1 Color", DefaultValue = "#4CAF50", Group = "Levels")]
        public Color level1Color { get; set; }

        [Parameter("L2 Enabled", DefaultValue = true, Group = "Levels")]
        public bool level2Enabled { get; set; }

        [Parameter("L2 Value", DefaultValue = 0.786, Group = "Levels")]
        public double level2Value { get; set; }

        [Parameter("L2 Color", DefaultValue = "#009688", Group = "Levels")]
        public Color level2Color { get; set; }

        [Parameter("Fibb Width", DefaultValue = 1, MinValue = 1, MaxValue = 10, Group = "Levels")]
        public int fibb_width { get; set; }

        [Parameter("Use Chart Timeframe", Group = "OB Timeframe", DefaultValue = true)]
        public bool UseChartTimeframe { get; set; }

        [Parameter("Time-Frame Order-Block", Group = "OB Timeframe", DefaultValue = "Hour")]
        public TimeFrame InputTimeFrame { get; set; }

        [Parameter("Line width Liquidated", Group = "OB Display", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int LineWidthLiquidated { get; set; }

        [Parameter("Transparency", Group = "OB Display", DefaultValue = 80, MinValue = 1, MaxValue = 100)]
        public int Transparency { get; set; }

        [Parameter("Color Bull", Group = "OB Display", DefaultValue = "Green")]
        public Color ColorBull { get; set; }

        [Parameter("Color Bear", Group = "OB Display", DefaultValue = "Red")]
        public Color ColorBear { get; set; }

        [Parameter("Color FVG Bull", Group = "OB Display", DefaultValue = "Blue")]
        public Color ColorFvgBull { get; set; }

        [Parameter("Color FVG Bear", Group = "OB Display", DefaultValue = "Orange")]
        public Color ColorFvgBear { get; set; }

        [Parameter("Show Order-Blocks", Group = "OB Display", DefaultValue = true)]
        public bool ShowOb { get; set; }

        [Parameter("Show Fair-Value-Gaps", Group = "OB Display", DefaultValue = true)]
        public bool ShowFvg { get; set; }

        [Parameter("Show Signals Order-Block", Group = "OB Display", DefaultValue = true)]
        public bool ShowSignalsOb { get; set; }

        [Parameter("Show Signals FVG", Group = "OB Display", DefaultValue = true)]
        public bool ShowSignalsFvg { get; set; }

        [Parameter("Min dist", Group = "OB Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDist { get; set; }

        [Parameter("Min dist FVG", Group = "OB Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", Group = "OB Signals", DefaultValue = false)]
        public bool UseHeikinAshi { get; set; }

        [Parameter("Signal Offset (pips)", Group = "OB Signals", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1)]
        public double SignalOffsetPips { get; set; }

        [Parameter("Risk %", Group = "Risk", DefaultValue = 1.0, MinValue = 0.01, Step = 0.01)]
        public double RiskPercent { get; set; }

        [Parameter("Instance Name", Group = "General", DefaultValue = "golden_order_block_detector_cbot")]
        public string InstanceName { get; set; }

        private sealed class ObRecord
        {
            public double Max;
            public double Min;
            public bool IsBull;
            public DateTime DetectionTime;
            public int DetectionChartIndex;
        }

        private sealed class FvgRecord
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
        }

        private int _pos;
        private double _Up = double.NaN;
        private double _Dn = double.NaN;
        private int _iUp = int.MinValue;
        private int _iDn = int.MinValue;

        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new();
        private readonly List<FvgRecord> _fvgRecords = new();
        private SignalState _signal = NewEmptySignal();
        private SignalState _signalFvg = NewEmptySignal();
        private readonly List<double> _haSourceOpen = new();
        private readonly List<double> _haSourceClose = new();
        private int _lastDetectedObSourceIndex = -1;
        private int _lastDetectedFvgSourceIndex = -1;

        protected override void OnStart()
        {
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
        }

        protected override void OnBar()
        {
            var signalIndex = Bars.Count - 2;
            if (signalIndex < 2)
                return;

            UpdateStructureState(signalIndex);

            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[signalIndex]);
            if (sourceIndex < 2)
                return;

            EnsureHeikinAshiSource(sourceIndex);
            DetectOrderBlock(signalIndex, sourceIndex);
            DetectFvg(signalIndex, sourceIndex);

            var sHigh = _sourceBars.HighPrices[sourceIndex];
            var sLow = _sourceBars.LowPrices[sourceIndex];
            var sClose = _sourceBars.ClosePrices[sourceIndex];

            HandleMitigationOb(signalIndex, sLow, sHigh);
            HandleMitigationFvg(signalIndex, sLow, sHigh);

            var candleDir = Bars.ClosePrices[signalIndex] > Bars.OpenPrices[signalIndex] ? 1 : -1;
            var cond = 0;
            var condFvg = 0;

            var signalClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            if (!double.IsNaN(_signal.Point) && signalClose > _signal.Point && _signal.IsBull && candleDir == 1 && !_signal.Entry)
            {
                _signal.Entry = true;
                cond = 1;
            }
            if (!double.IsNaN(_signal.Point) && signalClose < _signal.Point && !_signal.IsBull && candleDir == -1 && !_signal.Entry)
            {
                _signal.Entry = true;
                cond = -1;
            }

            var fvgClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            if (!double.IsNaN(_signalFvg.Point) && fvgClose > _signalFvg.Point && _signalFvg.IsBull && candleDir == 1 && !_signalFvg.Entry)
            {
                _signalFvg.Entry = true;
                condFvg = 1;
            }
            if (!double.IsNaN(_signalFvg.Point) && fvgClose < _signalFvg.Point && !_signalFvg.IsBull && candleDir == -1 && !_signalFvg.Entry)
            {
                _signalFvg.Entry = true;
                condFvg = -1;
            }

            ApplyGoldenZoneFilter(signalIndex, ref cond);
            ApplyGoldenZoneFilter(signalIndex, ref condFvg);

            if (cond == 1 || condFvg == 1)
                ExecuteDirectionalTrade(TradeType.Buy);

            if (cond == -1 || condFvg == -1)
                ExecuteDirectionalTrade(TradeType.Sell);
        }

        private void ExecuteDirectionalTrade(TradeType tradeType)
        {
            if (!TryGetLatestLevel0And1(out var level0, out var level1))
                return;

            if (tradeType == TradeType.Buy && !(level0 < level1))
                return;
            if (tradeType == TradeType.Sell && !(level0 > level1))
                return;

            var entryPrice = tradeType == TradeType.Buy ? Symbol.Ask : Symbol.Bid;
            var stopLossPips = tradeType == TradeType.Buy
                ? (entryPrice - level0) / Symbol.PipSize
                : (level0 - entryPrice) / Symbol.PipSize;

            if (stopLossPips <= 0)
                return;

            var riskAmount = Account.Balance * (RiskPercent / 100.0);
            var rawVolume = riskAmount / (stopLossPips * Symbol.PipValue);
            var volumeInUnits = Symbol.NormalizeVolumeInUnits(rawVolume, RoundingMode.Down);

            if (volumeInUnits < Symbol.VolumeInUnitsMin)
                volumeInUnits = Symbol.VolumeInUnitsMin;
            if (volumeInUnits > Symbol.VolumeInUnitsMax)
                volumeInUnits = Symbol.VolumeInUnitsMax;

            ExecuteMarketOrder(tradeType, SymbolName, volumeInUnits, InstanceName, stopLossPips, null);
        }

        private void ApplyGoldenZoneFilter(int index, ref int cond)
        {
            if (cond == 0)
                return;

            if (!TryGetLatestLevel0And1(out var level0, out var level1))
            {
                cond = 0;
                return;
            }

            var top = Math.Max(level0, level1);
            var bot = Math.Min(level0, level1);
            var isBullishZone = _pos > 0 || level1 > level0;
            var isBearishZone = _pos < 0 || level1 < level0;

            var price = Bars.ClosePrices[index];
            var isInsideZone = price >= bot && price <= top;
            if (!isInsideZone || (cond == 1 && !isBullishZone) || (cond == -1 && !isBearishZone))
                cond = 0;
        }

        private bool TryGetLatestLevel0And1(out double level0, out double level1)
        {
            level0 = double.NaN;
            level1 = double.NaN;
            if (_iUp == int.MinValue || _iDn == int.MinValue || double.IsNaN(_Up) || double.IsNaN(_Dn))
                return false;

            level0 = Fibb(0.0, _Up, _Dn, _iUp, _iDn);
            level1 = Fibb(level1Value, _Up, _Dn, _iUp, _iDn);
            return !double.IsNaN(level0) && !double.IsNaN(level1);
        }

        private void UpdateStructureState(int index)
        {
            if (index < prd * 2 + 2)
                return;

            var upPrev = _Up;
            var dnPrev = _Dn;

            var up = double.IsNaN(upPrev) ? double.NaN : Math.Max(upPrev, Bars.HighPrices[index]);
            var dn = double.IsNaN(dnPrev) ? double.NaN : Math.Min(dnPrev, Bars.LowPrices[index]);

            if (TryPivotHigh(index, prd, out _, out var ph) && _pos <= 0)
                up = ph;

            if (TryPivotLow(index, prd, out _, out var pl) && _pos >= 0)
                dn = pl;

            _Up = up;
            _Dn = dn;

            if (!double.IsNaN(_Up) && !double.IsNaN(upPrev) && _Up > upPrev)
            {
                _iUp = index;
                _pos = _pos <= 0 ? 1 : _pos + 1;
            }
            else if (!double.IsNaN(_Up) && !double.IsNaN(upPrev) && _Up < upPrev)
            {
                _iUp = index - prd;
            }

            if (!double.IsNaN(_Dn) && !double.IsNaN(dnPrev) && _Dn < dnPrev)
            {
                _iDn = index;
                _pos = _pos >= 0 ? -1 : _pos - 1;
            }
            else if (!double.IsNaN(_Dn) && !double.IsNaN(dnPrev) && _Dn > dnPrev)
            {
                _iDn = index - prd;
            }
        }

        private void DetectOrderBlock(int index, int sourceIndex)
        {
            if (!ShowOb || sourceIndex == _lastDetectedObSourceIndex)
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
                DetectionChartIndex = index
            });

            _lastDetectedObSourceIndex = sourceIndex;
        }

        private void DetectFvg(int index, int sourceIndex)
        {
            if (!ShowFvg || sourceIndex == _lastDetectedFvgSourceIndex)
                return;

            var detected = false;
            var isBull = false;
            var max = 0.0;
            var min = 0.0;

            if (_sourceBars.LowPrices[sourceIndex] > _sourceBars.HighPrices[sourceIndex - 2])
            {
                detected = true;
                isBull = true;
                max = _sourceBars.LowPrices[sourceIndex];
                min = _sourceBars.HighPrices[sourceIndex - 2];
            }

            if (_sourceBars.LowPrices[sourceIndex - 2] > _sourceBars.HighPrices[sourceIndex])
            {
                detected = true;
                isBull = false;
                max = _sourceBars.LowPrices[sourceIndex - 2];
                min = _sourceBars.HighPrices[sourceIndex];
            }

            if (!detected)
                return;

            _fvgRecords.Insert(0, new FvgRecord
            {
                Max = max,
                Min = min,
                IsBull = isBull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex],
                DetectionChartIndex = index
            });

            _lastDetectedFvgSourceIndex = sourceIndex;
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
                            _signal = NewSignal(r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        _obRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(r.Min, false);
                    }
                }
            }
        }

        private void HandleMitigationFvg(int index, double sLow, double sHigh)
        {
            for (var i = _fvgRecords.Count - 1; i >= 0; i--)
            {
                var r = _fvgRecords[i];
                var now = Bars.OpenTimes[index];

                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        _fvgRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        _fvgRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(r.Min, false);
                    }
                }
            }
        }

        private static SignalState NewEmptySignal() => new SignalState { Point = double.NaN, Entry = false };

        private static SignalState NewSignal(double point, bool isBull)
        {
            return new SignalState { Point = point, IsBull = isBull, Entry = false };
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

        private bool TryPivotHigh(int index, int p, out int pivotIndex, out double pivotHigh)
        {
            pivotIndex = index - p;
            pivotHigh = double.NaN;

            if (pivotIndex - p < 0) return false;
            if (pivotIndex + p > index) return false;

            var ph = Bars.HighPrices[pivotIndex];
            for (var i = pivotIndex - p; i <= pivotIndex + p; i++)
            {
                if (i == pivotIndex) continue;
                if (Bars.HighPrices[i] >= ph) return false;
            }

            pivotHigh = ph;
            return true;
        }

        private bool TryPivotLow(int index, int p, out int pivotIndex, out double pivotLow)
        {
            pivotIndex = index - p;
            pivotLow = double.NaN;

            if (pivotIndex - p < 0) return false;
            if (pivotIndex + p > index) return false;

            var pl = Bars.LowPrices[pivotIndex];
            for (var i = pivotIndex - p; i <= pivotIndex + p; i++)
            {
                if (i == pivotIndex) continue;
                if (Bars.LowPrices[i] <= pl) return false;
            }

            pivotLow = pl;
            return true;
        }

        private static double Fibb(double v, double h, double l, int ih, int il)
        {
            if (ih == int.MinValue || il == int.MinValue)
                return double.NaN;

            if (il < ih) return h - (h - l) * v;
            if (il > ih) return l + (h - l) * v;
            return double.NaN;
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
