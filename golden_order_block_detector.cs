using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class GoldenOrderBlockDetector : Indicator
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

        private sealed class SignalState
        {
            public double Point;
            public bool IsBull;
            public bool Entry;
            public int Index;
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

        [Parameter("Show Order-Blocks", Group = "Display", DefaultValue = true)]
        public bool ShowOb { get; set; }

        [Parameter("Show Signals", Group = "Display", DefaultValue = true)]
        public bool ShowSignals { get; set; }

        [Parameter("Min dist", Group = "Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDist { get; set; }

        [Parameter("Signal Offset (pips)", Group = "Signals", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1)]
        public double SignalOffsetPips { get; set; }

        [Parameter("Structure Period", Group = "OTE", DefaultValue = 10, MinValue = 1)]
        public int StructurePeriod { get; set; }

        [Parameter("Golden Level 1", Group = "OTE", DefaultValue = 0.618)]
        public double GoldenLevel1 { get; set; }

        [Parameter("Golden Level 2", Group = "OTE", DefaultValue = 0.786)]
        public double GoldenLevel2 { get; set; }

        [Parameter("Show Golden Zone", Group = "OTE", DefaultValue = true)]
        public bool ShowGoldenZone { get; set; }

        [Parameter("Bullish Golden Zone Color", Group = "OTE", DefaultValue = "#9900FF00")]
        public Color BullGoldZone { get; set; }

        [Parameter("Bearish Golden Zone Color", Group = "OTE", DefaultValue = "#99FF0000")]
        public Color BearGoldZone { get; set; }

        [Output("Long Signal", LineColor = "Lime", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries LongSignal { get; set; }

        [Output("Short Signal", LineColor = "Red", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries ShortSignal { get; set; }

        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new List<ObRecord>();
        private SignalState _signal = NewEmptySignal();

        private int _lastDetectedObSourceIndex = -1;
        private int _shapeId;

        // OTE state
        private int _pos;
        private double _up = double.NaN;
        private double _dn = double.NaN;
        private int _iUp = int.MinValue;
        private int _iDn = int.MinValue;
        private ChartRectangle _goldRect;
        private string _goldRectName;
        private int _drawId;

        protected override void Initialize()
        {
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
        }

        public override void Calculate(int index)
        {
            LongSignal[index] = double.NaN;
            ShortSignal[index] = double.NaN;

            if (index < Math.Max(2, StructurePeriod * 2 + 2))
                return;

            UpdateOteState(index);
            DrawGoldenZone(index);

            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
            if (sourceIndex < 1)
                return;

            UpdateBoxes(index);
            DetectOrderBlock(index, sourceIndex);

            var sHigh = _sourceBars.HighPrices[sourceIndex];
            var sLow = _sourceBars.LowPrices[sourceIndex];
            HandleMitigationOb(index, sLow, sHigh);

            var candleDir = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var cond = 0;
            var signalClose = _sourceBars.ClosePrices[sourceIndex];

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
                var inBullZone = IsPriceInGoldenZone(index, bullishOnly: true);
                var inBearZone = IsPriceInGoldenZone(index, bullishOnly: false);

                if ((cond == 1 && !inBullZone) || (cond == -1 && !inBearZone))
                    cond = 0;
            }

            DrawSignals(index, cond);
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

        private void UpdateOteState(int index)
        {
            var upPrev = _up;
            var dnPrev = _dn;

            if (double.IsNaN(_up)) _up = Bars.HighPrices[index - 1];
            if (double.IsNaN(_dn)) _dn = Bars.LowPrices[index - 1];

            var up = Math.Max(_up, Bars.HighPrices[index]);
            var dn = Math.Min(_dn, Bars.LowPrices[index]);

            if (TryPivotHigh(index, StructurePeriod, out _, out var ph) && _pos <= 0)
                up = ph;
            if (TryPivotLow(index, StructurePeriod, out _, out var pl) && _pos >= 0)
                dn = pl;

            _up = up;
            _dn = dn;

            if (!double.IsNaN(upPrev) && _up > upPrev)
            {
                _iUp = index;
                _pos = _pos <= 0 ? 1 : _pos + 1;
            }
            else if (!double.IsNaN(upPrev) && _up < upPrev)
            {
                _iUp = index - StructurePeriod;
            }

            if (!double.IsNaN(dnPrev) && _dn < dnPrev)
            {
                _iDn = index;
                _pos = _pos >= 0 ? -1 : _pos - 1;
            }
            else if (!double.IsNaN(dnPrev) && _dn > dnPrev)
            {
                _iDn = index - StructurePeriod;
            }
        }

        private bool IsPriceInGoldenZone(int index, bool bullishOnly)
        {
            if (!TryGetGoldenZone(index, out var top, out var bot, out var bullish, out var bearish))
                return false;

            if (bullishOnly && !bullish)
                return false;
            if (!bullishOnly && !bearish)
                return false;

            var close = Bars.ClosePrices[index];
            return close >= bot && close <= top;
        }

        private void DrawGoldenZone(int index)
        {
            if (!ShowGoldenZone)
            {
                RemoveGoldenFill();
                return;
            }

            if (!TryGetGoldenZone(index, out var top, out var bot, out var bullish, out var bearish))
            {
                RemoveGoldenFill();
                return;
            }

            var leftIndex = Math.Max(_iUp, _iDn);
            if (leftIndex == int.MinValue)
                return;

            var left = Bars.OpenTimes[leftIndex];
            var right = Bars.OpenTimes[index];
            var src = bullish ? BullGoldZone : BearGoldZone;
            var fill = Color.FromArgb(153, src.R, src.G, src.B);

            if (_goldRect == null)
            {
                _goldRectName = $"gold_zone_{_drawId++}";
                _goldRect = Chart.DrawRectangle(_goldRectName, left, top, right, bot, fill);
            }
            else
            {
                _goldRect.Time1 = left;
                _goldRect.Time2 = right;
                _goldRect.Y1 = top;
                _goldRect.Y2 = bot;
            }

            _goldRect.IsFilled = true;
            _goldRect.Color = fill;
            _goldRect.Thickness = 0;
            _goldRect.IsInteractive = false;
        }

        private bool TryGetGoldenZone(int index, out double top, out double bot, out bool bullish, out bool bearish)
        {
            top = bot = double.NaN;
            bullish = bearish = false;

            if (_iUp == int.MinValue || _iDn == int.MinValue || double.IsNaN(_up) || double.IsNaN(_dn) || _pos == 0)
                return false;

            var z1 = Fibb(GoldenLevel1, _up, _dn, _iUp, _iDn);
            var z2 = Fibb(GoldenLevel2, _up, _dn, _iUp, _iDn);
            if (double.IsNaN(z1) || double.IsNaN(z2))
                return false;

            top = Math.Max(z1, z2);
            bot = Math.Min(z1, z2);
            bullish = _pos > 0;
            bearish = _pos < 0;
            return true;
        }

        private double Fibb(double v, double h, double l, int ih, int il)
        {
            if (ih == int.MinValue || il == int.MinValue)
                return double.NaN;

            if (il < ih) return h - (h - l) * v;
            if (il > ih) return l + (h - l) * v;
            return double.NaN;
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

        private void RemoveGoldenFill()
        {
            if (_goldRect != null && !string.IsNullOrEmpty(_goldRectName))
            {
                Chart.RemoveObject(_goldRectName);
                _goldRect = null;
                _goldRectName = null;
            }
        }

        private static SignalState NewEmptySignal() => new SignalState { Point = double.NaN, Entry = false, Index = -1 };

        private SignalState NewSignal(int index, double point, bool isBull)
        {
            return new SignalState
            {
                Point = point,
                IsBull = isBull,
                Entry = false,
                Index = index
            };
        }

        private void DrawSignals(int index, int cond)
        {
            var offset = SignalOffsetPips * Symbol.PipSize;

            if (ShowSignals && cond == 1)
            {
                Chart.DrawIcon($"buy_{index}_{_shapeId++}", ChartIconType.UpArrow, Bars.OpenTimes[index], Bars.LowPrices[index] - offset, ColorBull);
                LongSignal[index] = Bars.LowPrices[index] - offset;
            }

            if (ShowSignals && cond == -1)
            {
                Chart.DrawIcon($"sell_{index}_{_shapeId++}", ChartIconType.DownArrow, Bars.OpenTimes[index], Bars.HighPrices[index] + offset, ColorBear);
                ShortSignal[index] = Bars.HighPrices[index] + offset;
            }
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
