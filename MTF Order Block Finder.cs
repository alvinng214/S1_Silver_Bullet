using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MTFOrderBlockFinder : Indicator
    {
        public enum ZoneDrawingStyle
        {
            BOTH,
            BOX,
            LINE
        }

        public enum OrderBlockSourceSelector
        {
            HighLow,
            OHLC,
            Context
        }

        private sealed class ZoneLineSet
        {
            public string AvgId;
            public string HighId;
            public string LowId;
            public ChartTrendLine Avg;
            public ChartTrendLine High;
            public ChartTrendLine Low;
        }

        private sealed class ZoneBox
        {
            public string Id;
            public ChartRectangle Box;
        }

        [Parameter("Use Chart Timeframe (like empty Pine TF)", Group = "Basic Configuration", DefaultValue = true)]
        public bool UseChartTimeframe { get; set; }

        [Parameter("Order Block Timeframe", Group = "Basic Configuration", DefaultValue = "Hour")]
        public TimeFrame Resolution { get; set; }

        [Parameter("Order Block Required Length", Group = "Basic Configuration", DefaultValue = 5, MinValue = 1, MaxValue = 20)]
        public int ObPeriodInput { get; set; }

        [Parameter("Order Block Required Move %", Group = "Basic Configuration", DefaultValue = 0.3, MinValue = 0, Step = 0.05)]
        public double Threshold { get; set; }

        [Parameter("Bullish Zones to Show", Group = "Basic Configuration", DefaultValue = 4, MinValue = 0)]
        public int BullChannels { get; set; }

        [Parameter("Bearish Zones to Show", Group = "Basic Configuration", DefaultValue = 4, MinValue = 0)]
        public int BearChannels { get; set; }

        [Parameter("1st Candle Filter", Group = "Advanced Configuration", DefaultValue = 0.05, MinValue = 0, Step = 0.005)]
        public double Doji { get; set; }

        [Parameter("2nd+ Candle Filter", Group = "Advanced Configuration", DefaultValue = 0.01, MinValue = 0, Step = 0.005)]
        public double Fuzzy { get; set; }

        [Parameter("Order Block Draw Distance %", Group = "Advanced Configuration", DefaultValue = 0.0, MinValue = 0, Step = 0.5)]
        public double NearPrice { get; set; }

        [Parameter("Zone Drawing Style", Group = "Style and Colors", DefaultValue = ZoneDrawingStyle.BOX)]
        public ZoneDrawingStyle StyleInput { get; set; }

        [Parameter("Bullish Zone Color", Group = "Style and Colors", DefaultValue = "#5900FF00")]
        public Color BullColor { get; set; }

        [Parameter("Bearish Zone Color", Group = "Style and Colors", DefaultValue = "#59FF0000")]
        public Color BearColor { get; set; }

        [Parameter("Line Width for Zone's High/Low", Group = "Style and Colors", DefaultValue = 1, MinValue = 0)]
        public int LineWidthHL { get; set; }

        [Parameter("Line Width for Zone's Avg", Group = "Style and Colors", DefaultValue = 1, MinValue = 0)]
        public int LineWidthAvg { get; set; }

        [Parameter("Order Block Candle Shift", Group = "Experimental Settings", DefaultValue = 1, MinValue = 0)]
        public int ObShiftInput { get; set; }

        [Parameter("Order Block Source Selector", Group = "Experimental Settings", DefaultValue = OrderBlockSourceSelector.OHLC)]
        public OrderBlockSourceSelector ObSelectorInput { get; set; }

        [Parameter("Order Block Context Search Length", Group = "Experimental Settings", DefaultValue = 2, MinValue = 0, MaxValue = 20)]
        public int ObSearchInput { get; set; }

        [Output("New Order Block", LineColor = "Yellow", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries AlertAny { get; set; }
        [Output("New Bullish Order Block", LineColor = "Lime", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries AlertBull { get; set; }
        [Output("New Bearish Order Block", LineColor = "Red", PlotType = PlotType.DiscontinuousLine)]
        public IndicatorDataSeries AlertBear { get; set; }

        private readonly List<ZoneLineSet> _bullLines = new List<ZoneLineSet>();
        private readonly List<ZoneLineSet> _bearLines = new List<ZoneLineSet>();
        private readonly List<ZoneBox> _bullBoxes = new List<ZoneBox>();
        private readonly List<ZoneBox> _bearBoxes = new List<ZoneBox>();

        private Bars _srcBars;
        private int _id;
        private int _obPeriod;
        private int _obShift;
        private int _obSearch;
        private int _lastProcessedHtfIndex = -1;

        protected override void Initialize()
        {
            _obPeriod = ObPeriodInput + 1;
            _obShift = Math.Min(ObShiftInput, _obPeriod - 1);
            _obSearch = Math.Min(ObSearchInput, _obPeriod - 1);

            var selectedTf = UseChartTimeframe ? Bars.TimeFrame : Resolution;
            var chartSec = TimeframeToSec(Bars.TimeFrame);
            var resSec = TimeframeToSec(selectedTf);
            var effectiveTf = (resSec > 0 && resSec < chartSec) ? Bars.TimeFrame : selectedTf;
            _srcBars = effectiveTf == Bars.TimeFrame ? Bars : MarketData.GetBars(effectiveTf);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            bool obBull = false;
            bool obBear = false;
            double obBullHigh = double.NaN, obBullLow = double.NaN, obBullAvg = double.NaN;
            double obBearHigh = double.NaN, obBearLow = double.NaN, obBearAvg = double.NaN;

            var htfIndex = FindBarIndexAtOrBefore(_srcBars, Bars.OpenTimes[index]);
            var minSize = htfIndex + 1;
            var warmup = _obPeriod;
            var htfHasNewData = htfIndex != _lastProcessedHtfIndex;

            if (NearPrice > 0 && index == Bars.Count - 1)
            {
                if (StyleInput == ZoneDrawingStyle.LINE || StyleInput == ZoneDrawingStyle.BOTH)
                {
                    RemoveDistantLines(_bullLines, Bars.OpenPrices[index], NearPrice);
                    RemoveDistantLines(_bearLines, Bars.OpenPrices[index], NearPrice);
                }
                if (StyleInput == ZoneDrawingStyle.BOX || StyleInput == ZoneDrawingStyle.BOTH)
                {
                    RemoveDistantBoxes(_bullBoxes, Bars.OpenPrices[index], NearPrice);
                    RemoveDistantBoxes(_bearBoxes, Bars.OpenPrices[index], NearPrice);
                }
            }

            if (minSize > warmup && htfHasNewData)
            {
                var i0 = htfIndex - 1;
                var iob = htfIndex - _obPeriod;

                var relMove = 100.0 * (Math.Abs(_srcBars.ClosePrices[iob] - _srcBars.ClosePrices[i0]) / _srcBars.ClosePrices[iob]) > Threshold;
                var dojiCandle = 100.0 * Math.Abs(_srcBars.ClosePrices[iob] - _srcBars.OpenPrices[iob]) / _srcBars.OpenPrices[iob] > Doji;

                var bullishOB = _srcBars.ClosePrices[iob] < _srcBars.OpenPrices[iob];
                var bearishOB = _srcBars.ClosePrices[iob] > _srcBars.OpenPrices[iob];

                int up = 0, down = 0;
                for (int i = 1; i <= _obPeriod - 1; i++)
                {
                    var bi = htfIndex - i;
                    if (IsZeroCandle(bi))
                        continue;

                    var tClose = _srcBars.ClosePrices[bi];
                    var tOpen = _srcBars.OpenPrices[bi];
                    if (Math.Abs(100.0 * (tClose - tOpen) / tOpen) < Fuzzy)
                    {
                        up++;
                        down++;
                        continue;
                    }
                    if (tClose > tOpen)
                        up++;
                    else if (tClose < tOpen)
                        down++;
                }

                if (dojiCandle && relMove)
                {
                    obBull = bullishOB && up == (_obPeriod - 1);
                    obBear = bearishOB && down == (_obPeriod - 1);

                    int selectorShift = _obShift;
                    if (obBull)
                    {
                        if (ObSelectorInput == OrderBlockSourceSelector.Context)
                        {
                            var r = LowWickSearch(htfIndex - _obPeriod, _obSearch);
                            selectorShift = r.idx;
                            obBullHigh = r.wickH;
                            obBullLow = r.wickL;
                        }
                        else if (ObSelectorInput == OrderBlockSourceSelector.HighLow)
                        {
                            var k = htfIndex - (_obPeriod - selectorShift);
                            obBullHigh = _srcBars.HighPrices[k];
                            obBullLow = _srcBars.LowPrices[k];
                        }
                        else
                        {
                            var r = LowWickSearch(htfIndex - (_obPeriod - selectorShift), 0);
                            obBullHigh = r.wickH;
                            obBullLow = r.wickL;
                        }
                        obBullAvg = (obBullHigh + obBullLow) / 2.0;
                    }

                    if (obBear)
                    {
                        if (ObSelectorInput == OrderBlockSourceSelector.Context)
                        {
                            var r = HighWickSearch(htfIndex - _obPeriod, _obSearch);
                            selectorShift = r.idx;
                            obBearHigh = r.wickH;
                            obBearLow = r.wickL;
                        }
                        else if (ObSelectorInput == OrderBlockSourceSelector.HighLow)
                        {
                            var k = htfIndex - (_obPeriod - selectorShift);
                            obBearHigh = _srcBars.HighPrices[k];
                            obBearLow = _srcBars.LowPrices[k];
                        }
                        else
                        {
                            var r = HighWickSearch(htfIndex - (_obPeriod - selectorShift), 0);
                            obBearHigh = r.wickH;
                            obBearLow = r.wickL;
                        }
                        obBearAvg = (obBearHigh + obBearLow) / 2.0;
                    }

                    var tfSec = TimeframeToSec(Bars.TimeFrame);
                    var resSec = TimeframeToSec(_srcBars.TimeFrame);
                    var syncTime = _srcBars.OpenTimes[Math.Max(0, htfIndex - 1)];

                    if (StyleInput == ZoneDrawingStyle.LINE || StyleInput == ZoneDrawingStyle.BOTH)
                    {
                        if (obBull && BullChannels > 0)
                        {
                            if (_bullLines.Count == BullChannels)
                                RemoveOldestLine(_bullLines);
                            _bullLines.Add(CreateLineSet(obBullHigh, obBullLow, obBullAvg, syncTime, BullColor));
                        }
                        if (obBear && BearChannels > 0)
                        {
                            if (_bearLines.Count == BearChannels)
                                RemoveOldestLine(_bearLines);
                            _bearLines.Add(CreateLineSet(obBearHigh, obBearLow, obBearAvg, syncTime, BearColor));
                        }
                    }

                    if (StyleInput == ZoneDrawingStyle.BOX || StyleInput == ZoneDrawingStyle.BOTH)
                    {
                        if (obBull && BullChannels > 0)
                        {
                            if (_bullBoxes.Count == BullChannels)
                                RemoveOldestBox(_bullBoxes);
                            _bullBoxes.Add(CreateBox(obBullHigh, obBullLow, syncTime, BullColor));
                        }
                        if (obBear && BearChannels > 0)
                        {
                            if (_bearBoxes.Count == BearChannels)
                                RemoveOldestBox(_bearBoxes);
                            _bearBoxes.Add(CreateBox(obBearHigh, obBearLow, syncTime, BearColor));
                        }
                    }

                    var syncMult = tfSec == 0 ? 1 : (resSec / Math.Max(1, tfSec));
                    var os = -(_obPeriod - selectorShift) * (Resolution == Bars.TimeFrame ? 1 : syncMult);
                    var srcIndexA = htfIndex - (_obPeriod - selectorShift);
                    var srcIndexB = htfIndex - (_obPeriod - selectorShift - 1);
                    if (srcIndexA >= 0 && srcIndexB >= 0 && srcIndexA < _srcBars.Count && srcIndexB < _srcBars.Count)
                    {
                        var startTime = _srcBars.OpenTimes[srcIndexA];
                        var endTime = _srcBars.OpenTimes[srcIndexB];
                        var avgTime = startTime.AddTicks((endTime - startTime).Ticks / 2);

                        if (obBull)
                            DrawCustomMarkers(true, obBullHigh, obBullLow, obBullAvg, os, startTime, endTime, avgTime);
                        if (obBear)
                            DrawCustomMarkers(false, obBearHigh, obBearLow, obBearAvg, os, startTime, endTime, avgTime);
                    }
                }
            }

            _lastProcessedHtfIndex = htfIndex;

            AlertBull[index] = obBull ? 1 : double.NaN;
            AlertBear[index] = obBear ? 1 : double.NaN;
            AlertAny[index] = (obBull || obBear) ? 1 : double.NaN;
        }

        private int TimeframeToSec(TimeFrame tf)
        {
            return (int)Math.Round(tf.ToTimeSpan().TotalSeconds);
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var i = bars.OpenTimes.GetIndexByTime(time);
            if (i >= 0)
                return i;
            for (int j = bars.Count - 1; j >= 0; j--)
                if (bars.OpenTimes[j] <= time)
                    return j;
            return -1;
        }

        private bool IsZeroCandle(int i)
        {
            var o = _srcBars.OpenPrices[i];
            return o == _srcBars.ClosePrices[i] && o == _srcBars.HighPrices[i] && o == _srcBars.LowPrices[i];
        }

        private (double wickH, double wickL, int idx) LowWickSearch(int start, int len)
        {
            double wickH = double.NaN, wickL = double.NaN;
            int idx = 0;
            for (int i = 0; i <= len; i++)
            {
                var bi = start + i;
                if (bi < 0 || bi >= _srcBars.Count)
                    continue;
                if (!double.IsNaN(wickL) && _srcBars.LowPrices[bi] > wickL && i > 0)
                    continue;
                var dir = Math.Sign(_srcBars.ClosePrices[bi] - _srcBars.OpenPrices[bi]);
                wickH = dir == 1 ? _srcBars.ClosePrices[bi] : _srcBars.OpenPrices[bi];
                wickL = _srcBars.LowPrices[bi];
                idx = i;
            }
            return (wickH, wickL, idx);
        }

        private (double wickH, double wickL, int idx) HighWickSearch(int start, int len)
        {
            double wickH = double.NaN, wickL = double.NaN;
            int idx = 0;
            for (int i = 0; i <= len; i++)
            {
                var bi = start + i;
                if (bi < 0 || bi >= _srcBars.Count)
                    continue;
                if (!double.IsNaN(wickH) && _srcBars.HighPrices[bi] < wickH && i > 0)
                    continue;
                var dir = Math.Sign(_srcBars.ClosePrices[bi] - _srcBars.OpenPrices[bi]);
                wickL = dir == 1 ? _srcBars.OpenPrices[bi] : _srcBars.ClosePrices[bi];
                wickH = _srcBars.HighPrices[bi];
                idx = i;
            }
            return (wickH, wickL, idx);
        }

        private ZoneLineSet CreateLineSet(double h, double l, double avg, DateTime t, Color c)
        {
            var set = new ZoneLineSet();
            set.AvgId = "ob_la_" + _id++;
            set.HighId = "ob_lh_" + _id++;
            set.LowId = "ob_ll_" + _id++;
            var t2 = t.AddSeconds(12);
            var cAvg = LineWidthAvg > 0 ? c : Color.FromArgb(0, c);
            var cHl = LineWidthHL > 0 ? c : Color.FromArgb(0, c);
            set.Avg = Chart.DrawTrendLine(set.AvgId, t, avg, t2, avg, cAvg, Math.Max(LineWidthAvg, 1), LineStyle.DotsRare);
            set.High = Chart.DrawTrendLine(set.HighId, t, h, t2, h, cHl, Math.Max(LineWidthHL, 1), LineStyle.Solid);
            set.Low = Chart.DrawTrendLine(set.LowId, t, l, t2, l, cHl, Math.Max(LineWidthHL, 1), LineStyle.Solid);
            set.Avg.ExtendToInfinity = true;
            set.High.ExtendToInfinity = true;
            set.Low.ExtendToInfinity = true;
            return set;
        }

        private ZoneBox CreateBox(double h, double l, DateTime t, Color c)
        {
            var b = new ZoneBox();
            b.Id = "ob_b_" + _id++;
            b.Box = Chart.DrawRectangle(b.Id, t, h, t.AddSeconds(12), l, c, 0, LineStyle.Solid);
            b.Box.IsFilled = true;
            b.Box.Color = c;
            return b;
        }

        private void DrawCustomMarkers(bool bull, double h, double l, double avg, int os, DateTime start, DateTime end, DateTime avgTime)
        {
            var c = bull ? Opaque(BullColor) : Opaque(BearColor);
            var text = bull ? "Bull OB" : "Bear OB";
            var baseId = "ob_m_" + _id++;

            if (Resolution == Bars.TimeFrame)
            {
                var i = Bars.Count - 1;
                var x = Math.Max(0, i + os);
                Chart.DrawText(baseId + "_lbl", text, Bars.OpenTimes[x], bull ? l : h, c);
                var x2 = Math.Max(0, Math.Min(Bars.Count - 1, i + os - 1));
                Chart.DrawTrendLine(baseId + "_h", Bars.OpenTimes[x], h, Bars.OpenTimes[x2], h, c, 2, LineStyle.Solid);
                Chart.DrawTrendLine(baseId + "_l", Bars.OpenTimes[x], l, Bars.OpenTimes[x2], l, c, 2, LineStyle.Solid);
                Chart.DrawTrendLine(baseId + "_a", Bars.OpenTimes[x], avg, Bars.OpenTimes[x2], avg, c, 1, LineStyle.Solid);
            }
            else
            {
                Chart.DrawText(baseId + "_lbl", text, avgTime, bull ? l : h, c);
                Chart.DrawTrendLine(baseId + "_h", start, h, end, h, c, 2, LineStyle.Solid);
                Chart.DrawTrendLine(baseId + "_l", start, l, end, l, c, 2, LineStyle.Solid);
                Chart.DrawTrendLine(baseId + "_a", start, avg, end, avg, c, 1, LineStyle.Solid);
            }
        }

        private Color Opaque(Color c)
        {
            return Color.FromArgb(255, c.R, c.G, c.B);
        }

        private void RemoveDistantLines(List<ZoneLineSet> arr, double source, double percent)
        {
            var limit = source * percent / 100.0;
            for (int i = arr.Count - 1; i >= 0; i--)
            {
                var top = arr[i].High.Y1;
                var bottom = arr[i].Low.Y1;
                if (Math.Abs(top - source) > limit && Math.Abs(bottom - source) > limit)
                {
                    Chart.RemoveObject(arr[i].AvgId);
                    Chart.RemoveObject(arr[i].HighId);
                    Chart.RemoveObject(arr[i].LowId);
                    arr.RemoveAt(i);
                }
            }
        }

        private void RemoveDistantBoxes(List<ZoneBox> arr, double source, double percent)
        {
            var limit = source * percent / 100.0;
            for (int i = arr.Count - 1; i >= 0; i--)
            {
                var top = arr[i].Box.Y1;
                var bottom = arr[i].Box.Y2;
                if (Math.Abs(top - source) > limit && Math.Abs(bottom - source) > limit)
                {
                    Chart.RemoveObject(arr[i].Id);
                    arr.RemoveAt(i);
                }
            }
        }

        private void RemoveOldestLine(List<ZoneLineSet> arr)
        {
            if (arr.Count == 0) return;
            var z = arr[0];
            Chart.RemoveObject(z.AvgId);
            Chart.RemoveObject(z.HighId);
            Chart.RemoveObject(z.LowId);
            arr.RemoveAt(0);
        }

        private void RemoveOldestBox(List<ZoneBox> arr)
        {
            if (arr.Count == 0) return;
            Chart.RemoveObject(arr[0].Id);
            arr.RemoveAt(0);
        }
    }
}
