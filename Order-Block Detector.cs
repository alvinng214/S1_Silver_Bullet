using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class OrderBlockDetector : Indicator
    {
        private sealed class ObRecord
        {
            public double Max;
            public double Min;
            public bool IsBull;
            public DateTime DetectionTime;
            public int DetectionChartIndex;
            public string BoxId;
        }

        private sealed class FvgRecord
        {
            public double Max;
            public double Min;
            public bool IsBull;
            public DateTime DetectionTime;
            public int DetectionChartIndex;
            public string BoxId;
        }

        private sealed class SignalState
        {
            public double CandleOpen;
            public double CandleHigh;
            public double CandleLow;
            public double CandleClose;
            public double Point;
            public bool IsBull;
            public bool Entry;
            public int Index;
            public DateTime Time;
        }

        [Parameter("Use Chart Timeframe", Group = "Timeframe", DefaultValue = true)]
        public bool UseChartTimeframe { get; set; }

        [Parameter("Time-Frame Order-Block", Group = "Timeframe", DefaultValue = "Hour")]
        public TimeFrame InputTimeFrame { get; set; }

        [Parameter("Line width Liquidated", Group = "Display", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int LineWidthLiquidated { get; set; }

        [Parameter("Transparency", Group = "Display", DefaultValue = 80, MinValue = 1, MaxValue = 100)]
        public int Transparency { get; set; }

        [Parameter("Color Bull", Group = "Display", DefaultValue = "Green")]
        public Color ColorBull { get; set; }

        [Parameter("Color Bear", Group = "Display", DefaultValue = "Red")]
        public Color ColorBear { get; set; }

        [Parameter("Color FVG Bull", Group = "Display", DefaultValue = "Blue")]
        public Color ColorFvgBull { get; set; }

        [Parameter("Color FVG Bear", Group = "Display", DefaultValue = "Orange")]
        public Color ColorFvgBear { get; set; }

        [Parameter("Show Order-Blocks", Group = "Display", DefaultValue = true)]
        public bool ShowOb { get; set; }

        [Parameter("Show Fair-Value-Gaps", Group = "Display", DefaultValue = true)]
        public bool ShowFvg { get; set; }

        [Parameter("Show Signals Order-Block", Group = "Display", DefaultValue = true)]
        public bool ShowSignalsOb { get; set; }

        [Parameter("Show Signals FVG", Group = "Display", DefaultValue = true)]
        public bool ShowSignalsFvg { get; set; }

        // ── Signal outputs consumed by cBots ────────────────────────────────
        // 1.0 on the bar where the signal fires, 0.0 otherwise.
        // LongSignal  fires when an OB or FVG bull entry condition is met.
        // ShortSignal fires when an OB or FVG bear entry condition is met.
        [Output("Long Signal", LineColor = "Lime", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries LongSignal { get; set; }

        [Output("Short Signal", LineColor = "Red", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries ShortSignal { get; set; }

        [Parameter("Min dist", Group = "Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDist { get; set; }

        [Parameter("Min dist FVG", Group = "Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", Group = "Signals", DefaultValue = false)]
        public bool UseHeikinAshi { get; set; }

        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new List<ObRecord>();
        private readonly List<FvgRecord> _fvgRecords = new List<FvgRecord>();

        private SignalState _signal = NewEmptySignal();
        private SignalState _signalFvg = NewEmptySignal();

        private int _lastDetectedObSourceIndex = -1;
        private int _lastDetectedFvgSourceIndex = -1;
        private int _shapeId;

        private readonly List<double> _haSourceOpen = new List<double>();
        private readonly List<double> _haSourceClose = new List<double>();

        protected override void Initialize()
        {
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
            {
                LongSignal[index]  = double.NaN;
                ShortSignal[index] = double.NaN;
                return;
            }

            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
            if (sourceIndex < 2)
                return;

            EnsureHeikinAshiSource(sourceIndex);

            UpdateBoxes(index);

            DetectOrderBlock(index, sourceIndex);
            DetectFvg(index, sourceIndex);

            var sHigh = _sourceBars.HighPrices[sourceIndex];
            var sLow = _sourceBars.LowPrices[sourceIndex];
            var sClose = _sourceBars.ClosePrices[sourceIndex];

            HandleMitigationOb(index, sLow, sHigh);
            HandleMitigationFvg(index, sLow, sHigh);

            var candleDir = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var cond = 0;
            var condFvg = 0;

            var signalClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose; // mirrors Pine assignment using fvg_close for OB section
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

            var fvgClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            if (fvgClose > _signalFvg.Point && _signalFvg.IsBull && candleDir == 1 && !_signalFvg.Entry)
            {
                _signalFvg.Entry = true;
                condFvg = 1;
            }

            if (fvgClose < _signalFvg.Point && !_signalFvg.IsBull && candleDir == -1 && !_signalFvg.Entry)
            {
                _signalFvg.Entry = true;
                condFvg = -1;
            }

            DrawSignals(index, cond, condFvg);

            // Expose signal state for cBot consumption (mirrors ICT_01 pattern).
            // Long  = OB bull entry OR FVG bull entry on this bar.
            // Short = OB bear entry OR FVG bear entry on this bar.
            LongSignal[index]  = (cond == 1  || condFvg == 1)  ? 1.0 : double.NaN;
            ShortSignal[index] = (cond == -1 || condFvg == -1) ? 1.0 : double.NaN;
        }

        private void DetectOrderBlock(int index, int sourceIndex)
        {
            if (!ShowOb || sourceIndex == _lastDetectedObSourceIndex)
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

            var id = $"ob_box_{_sourceBars.OpenTimes[sourceIndex].Ticks}";
            var boxColor = Color.FromArgb((int)Math.Round(255.0 * (100 - Transparency) / 100.0), isBull ? ColorBull : ColorBear);
            var box = Chart.DrawRectangle(id, Bars.OpenTimes[Math.Max(0, index - 1)], max, Bars.OpenTimes[index], min, boxColor);
            box.IsFilled = true;
            box.Color = boxColor;

            _obRecords.Insert(0, new ObRecord
            {
                Max = max,
                Min = min,
                IsBull = isBull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex],
                DetectionChartIndex = index,
                BoxId = id
            });

            _lastDetectedObSourceIndex = sourceIndex;
        }

        private void DetectFvg(int index, int sourceIndex)
        {
            if (!ShowFvg || sourceIndex == _lastDetectedFvgSourceIndex)
                return;

            bool detected = false;
            bool isBull = false;
            double max = 0;
            double min = 0;

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

            var id = $"fvg_box_{_sourceBars.OpenTimes[sourceIndex].Ticks}";
            var boxColor = Color.FromArgb((int)Math.Round(255.0 * (100 - Transparency) / 100.0), isBull ? ColorFvgBull : ColorFvgBear);
            var box = Chart.DrawRectangle(id, Bars.OpenTimes[Math.Max(0, index - 1)], max, Bars.OpenTimes[index], min, boxColor);
            box.IsFilled = true;
            box.Color = boxColor;

            _fvgRecords.Insert(0, new FvgRecord
            {
                Max = max,
                Min = min,
                IsBull = isBull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex],
                DetectionChartIndex = index,
                BoxId = id
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
                        DrawLiquidationLine($"ob_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Max, ColorBull);
                        Chart.RemoveObject(r.BoxId);
                        _obRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(index, r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        DrawLiquidationLine($"ob_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Min, ColorBear);
                        Chart.RemoveObject(r.BoxId);
                        _obRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(index, r.Min, false);
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
                        DrawLiquidationLine($"fvg_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Max, ColorFvgBull);
                        Chart.RemoveObject(r.BoxId);
                        _fvgRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(index, r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        DrawLiquidationLine($"fvg_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Min, ColorFvgBear);
                        Chart.RemoveObject(r.BoxId);
                        _fvgRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(index, r.Min, false);
                    }
                }
            }
        }

        private SignalState NewSignal(int index, double point, bool isBull)
        {
            return new SignalState
            {
                CandleOpen = Bars.OpenPrices[index],
                CandleHigh = Bars.HighPrices[index],
                CandleLow = Bars.LowPrices[index],
                CandleClose = Bars.ClosePrices[index],
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

        private void DrawSignals(int index, int cond, int condFvg)
        {
            if (ShowSignalsOb && cond == 1)
                DrawSignalIcon($"ob_buy_{index}_{_shapeId++}", ChartIconType.UpArrow, index, Bars.LowPrices[index], ColorBull);
            if (ShowSignalsOb && cond == -1)
                DrawSignalIcon($"ob_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], ColorBear);

            if (ShowSignalsFvg && condFvg == 1)
                DrawSignalIcon($"fvg_buy_{index}_{_shapeId++}", ChartIconType.UpArrow, index, Bars.LowPrices[index], ColorFvgBull);
            if (ShowSignalsFvg && condFvg == -1)
                DrawSignalIcon($"fvg_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], ColorFvgBear);
        }

        private void DrawSignalIcon(string id, ChartIconType type, int index, double y, Color color)
        {
            Chart.DrawIcon(id, type, Bars.OpenTimes[index], y, color);
        }

        private void DrawLiquidationLine(string id, DateTime from, DateTime to, double price, Color color)
        {
            var line = Chart.DrawTrendLine(id, from, price, to, price, color, LineWidthLiquidated, LineStyle.LinesDots);
            line.ExtendToInfinity = false;
        }

        private void UpdateBoxes(int index)
        {
            var rightTime = Bars.OpenTimes[index];
            foreach (var r in _obRecords)
            {
                if (Chart.FindObject(r.BoxId) is ChartRectangle rect)
                    rect.Time2 = rightTime;
            }

            foreach (var r in _fvgRecords)
            {
                if (Chart.FindObject(r.BoxId) is ChartRectangle rect)
                    rect.Time2 = rightTime;
            }
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
