// Fibonacci Optimal Entry Zone [OTE] (Zeiierman).cs
// FIX: ChartRectangle in your cTrader build has NO FillColor property.
//      Golden zone fill is implemented as a filled rectangle using .Color and .IsFilled=true.

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class GoldenOrderBlockDetector : Indicator
    {
        // =========================
        // Parameters (mirrors Pine)
        // =========================

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

        // Fibonacci Mode
        [Parameter("Swing tracker", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool follow { get; set; }

        [Parameter("Swing Line", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool swingline { get; set; }

        [Parameter("Swing Line Width", DefaultValue = 2, MinValue = 1, MaxValue = 10, Group = "Fibonacci Mode")]
        public int swline_width { get; set; }

        [Parameter("Swing Labels", DefaultValue = true, Group = "Fibonacci Mode")]
        public bool swinglab { get; set; }

        // Fibonacci
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

        // Levels
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

        // =========================
        // Order-Block Detector types
        // =========================
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
            public double Point;
            public bool IsBull;
            public bool Entry;
        }

        // Timeframe
        [Parameter("Use Chart Timeframe", Group = "OB Timeframe", DefaultValue = true)]
        public bool UseChartTimeframe { get; set; }

        [Parameter("Time-Frame Order-Block", Group = "OB Timeframe", DefaultValue = "Hour")]
        public TimeFrame InputTimeFrame { get; set; }

        // Display
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

        // Signals
        [Parameter("Min dist", Group = "OB Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDist { get; set; }

        [Parameter("Min dist FVG", Group = "OB Signals", DefaultValue = 1, MinValue = 1)]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", Group = "OB Signals", DefaultValue = false)]
        public bool UseHeikinAshi { get; set; }

        [Parameter("Signal Offset (pips)", Group = "OB Signals", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1)]
        public double SignalOffsetPips { get; set; }

        [Output("Long Signal", LineColor = "Lime", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries LongSignal { get; set; }

        [Output("Short Signal", LineColor = "Red", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries ShortSignal { get; set; }

        // =========================
        // Internal State (Pine vars)
        // =========================

        private int _pos; // 0 neutral; >0 bullish ladder; <0 bearish ladder

        private double _Up = double.NaN;
        private double _Dn = double.NaN;

        private int _iUp = int.MinValue;
        private int _iDn = int.MinValue;

        private double _swingLow = double.NaN;
        private double _swingHigh = double.NaN;

        private int _iSwingLow = int.MinValue;
        private int _iSwingHigh = int.MinValue;

        private readonly List<double> _levels = new();
        private readonly List<Color> _colors = new();

        private readonly List<ChartTrendLine> _flevels = new();
        private readonly List<string> _flevelNames = new();

        private ChartTrendLine _trend;
        private string _trendName;

        // Golden zone fill approximation
        private ChartRectangle _goldRect;
        private string _goldRectName;

        // Swing labels
        private string _startLabelName;
        private string _endUpLabelName;
        private string _endDnLabelName;

        private int _id;

        // OB state
        private Bars _sourceBars;
        private readonly List<ObRecord> _obRecords = new();
        private readonly List<FvgRecord> _fvgRecords = new();
        private SignalState _signal = NewEmptySignal();
        private SignalState _signalFvg = NewEmptySignal();
        private readonly List<double> _haSourceOpen = new();
        private readonly List<double> _haSourceClose = new();
        private int _lastDetectedObSourceIndex = -1;
        private int _lastDetectedFvgSourceIndex = -1;
        private int _shapeId;

        protected override void Initialize()
        {
            RebuildLevels();
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
        }

        public override void Calculate(int index)
        {
            LongSignal[index] = double.NaN;
            ShortSignal[index] = double.NaN;

            if (index >= prd * 2 + 2)
            {
            double UpPrev = _Up;
            double DnPrev = _Dn;
            int iUpPrev = _iUp;
            int iDnPrev = _iDn;

            double Up = double.IsNaN(UpPrev) ? double.NaN : Math.Max(UpPrev, Bars.HighPrices[index]);
            double Dn = double.IsNaN(DnPrev) ? double.NaN : Math.Min(DnPrev, Bars.LowPrices[index]);

            if (TryPivotHigh(index, prd, out _, out var ph) && _pos <= 0)
                Up = ph;

            if (TryPivotLow(index, prd, out _, out var pl) && _pos >= 0)
                Dn = pl;

            _Up = Up;
            _Dn = Dn;

            // ---------- Bullish ----------
            if (!double.IsNaN(_Up) && !double.IsNaN(UpPrev) && _Up > UpPrev)
            {
                _iUp = index;
                int? centerBull = CenterIndexOrNa(iUpPrev, index);

                if (_pos <= 0)
                {
                    ClearActiveSet(deleteDrawings: !showOld);

                    if (bull)
                    {
                        if (centerBull.HasValue)
                            DrawChoch(centerBull.Value, UpPrev, "CHoCH", bull2, labelDown: true);

                        if (iUpPrev != int.MinValue)
                            DrawBos(iUpPrev, index, UpPrev, bull2);

                        if (_iDn != int.MinValue && !double.IsNaN(_Dn))
                            BuildFibLevelsForCurrentLeg(isBull: true, index: index);

                        if (swingline && _iDn != int.MinValue && !double.IsNaN(_Dn))
                            CreateSwingTrend(x1: _iDn, y1: _Dn, x2: index, y2: _Up, c: bull2);

                        if (swinglab)
                        {
                            RemoveIfExists(_endUpLabelName);
                            _endUpLabelName = NextName("endu");
                            DrawPriceLabel(_endUpLabelName, index, _Up, downStyle: true);
                        }
                    }

                    _pos = 1;
                    _swingLow = _Dn;
                    _iSwingLow = _iDn;
                }
                else if (_pos == 1)
                {
                    if (bull)
                        UpdateFibAndSwing(isBull: true, index: index);

                    _pos = 2;
                }
                else
                {
                    if (bull)
                    {
                        UpdateFibAndSwing(isBull: true, index: index);

                        if (swinglab && (follow ? _iDn : _iSwingLow) != int.MinValue)
                        {
                            int sx = follow ? _iDn : _iSwingLow;
                            double sy = follow ? _Dn : _swingLow;

                            RemoveIfExists(_startLabelName);
                            _startLabelName = NextName("start");
                            DrawPriceLabel(_startLabelName, sx, sy, downStyle: false);
                        }
                    }

                    _pos = _pos + 1;
                }
            }
            else if (!double.IsNaN(_Up) && !double.IsNaN(UpPrev) && _Up < UpPrev)
            {
                _iUp = index - prd;
            }

            // ---------- Bearish ----------
            if (!double.IsNaN(_Dn) && !double.IsNaN(DnPrev) && _Dn < DnPrev)
            {
                _iDn = index;
                int? centerBear = CenterIndexOrNa(iDnPrev, index);

                if (_pos >= 0)
                {
                    ClearActiveSet(deleteDrawings: !showOld);

                    if (bear)
                    {
                        if (centerBear.HasValue)
                            DrawChoch(centerBear.Value, DnPrev, "CHoCH", bear2, labelDown: false);

                        if (iDnPrev != int.MinValue)
                            DrawBos(iDnPrev, index, DnPrev, bear2);

                        if (_iUp != int.MinValue && !double.IsNaN(_Up))
                            BuildFibLevelsForCurrentLeg(isBull: false, index: index);

                        if (swingline && _iUp != int.MinValue && !double.IsNaN(_Up))
                            CreateSwingTrend(x1: _iUp, y1: _Up, x2: index, y2: _Dn, c: bear2);

                        if (swinglab)
                        {
                            RemoveIfExists(_endDnLabelName);
                            _endDnLabelName = NextName("endd");
                            DrawPriceLabel(_endDnLabelName, index, _Dn, downStyle: false);
                        }
                    }

                    _pos = -1;
                    _swingHigh = _Up;
                    _iSwingHigh = _iUp;
                }
                else if (_pos == -1)
                {
                    if (bear)
                        UpdateFibAndSwing(isBull: false, index: index);

                    _pos = -2;
                }
                else
                {
                    if (bear)
                    {
                        UpdateFibAndSwing(isBull: false, index: index);

                        if (swinglab && (follow ? _iUp : _iSwingHigh) != int.MinValue)
                        {
                            int sx = follow ? _iUp : _iSwingHigh;
                            double sy = follow ? _Up : _swingHigh;

                            RemoveIfExists(_startLabelName);
                            _startLabelName = NextName("start");
                            DrawPriceLabel(_startLabelName, sx, sy, downStyle: true);
                        }
                    }

                    _pos = _pos - 1;
                }
            }
            else if (!double.IsNaN(_Dn) && !double.IsNaN(DnPrev) && _Dn > DnPrev)
            {
                _iDn = index - prd;
            }

            // Extend
            if (extend)
            {
                DateTime t2 = TimeOf(index);

                for (int i = 0; i < _flevels.Count; i++)
                    _flevels[i].Time2 = t2;

                if (_trend != null)
                    _trend.Time2 = t2;

                if (_goldRect != null)
                    _goldRect.Time2 = t2;
            }

            // Golden fill (fixed for your API)
            if (golden && _flevels.Count > 1)
                UpdateGoldenFill(index);
            else
                RemoveGoldenFill();
            }

            ProcessOrderBlockDetector(index);
        }

        // =========================
        // Levels build (Pine: levels.push, colors.unshift)
        // =========================
        private void RebuildLevels()
        {
            _levels.Clear();
            _colors.Clear();

            if (level1Enabled)
            {
                _levels.Add(level1Value);
                _colors.Insert(0, level1Color);
            }

            if (level2Enabled)
            {
                _levels.Add(level2Value);
                _colors.Insert(0, level2Color);
            }
        }

        // =========================
        // Pivot logic (strict)
        // =========================
        private bool TryPivotHigh(int index, int p, out int pivotIndex, out double pivotHigh)
        {
            pivotIndex = index - p;
            pivotHigh = double.NaN;

            if (pivotIndex - p < 0) return false;
            if (pivotIndex + p > index) return false;

            double ph = Bars.HighPrices[pivotIndex];
            for (int i = pivotIndex - p; i <= pivotIndex + p; i++)
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

            double pl = Bars.LowPrices[pivotIndex];
            for (int i = pivotIndex - p; i <= pivotIndex + p; i++)
            {
                if (i == pivotIndex) continue;
                if (Bars.LowPrices[i] <= pl) return false;
            }

            pivotLow = pl;
            return true;
        }

        // =========================
        // Pine fibb()
        // =========================
        private double Fibb(double v, double h, double l, int ih, int il)
        {
            if (ih == int.MinValue || il == int.MinValue)
                return double.NaN;

            if (il < ih) return h - (h - l) * v;
            if (il > ih) return l + (h - l) * v;

            return double.NaN;
        }

        private int? CenterIndexOrNa(int prevIndex, int currIndex)
        {
            if (prevIndex == int.MinValue)
                return null;

            double avg = (prevIndex + currIndex) / 2.0;
            return (int)Math.Round(avg, MidpointRounding.AwayFromZero);
        }

        // =========================
        // Fib lines
        // =========================
        private void BuildFibLevelsForCurrentLeg(bool isBull, int index)
        {
            _flevels.Clear();
            _flevelNames.Clear();

            for (int i = 0; i < _levels.Count; i++)
            {
                double level = _levels[i];
                Color col = _colors[i];

                double val = isBull
                    ? Fibb(level, _Up, _Dn, _iUp, _iDn)
                    : Fibb(level, _Dn, _Up, _iDn, _iUp);

                if (double.IsNaN(val))
                    continue;

                int x1 = isBull ? _iDn : _iUp;
                var ln = DrawHorizontalLine(NextName("fib"), x1, index, val, col, fibb_width);

                _flevels.Insert(0, ln);
                _flevelNames.Insert(0, ln.Name);
            }
        }

        private void UpdateFibAndSwing(bool isBull, int index)
        {
            if (_flevels.Count == 0)
                return;

            for (int i = 0; i < _levels.Count && i < _flevels.Count; i++)
            {
                double level = _levels[i];

                double val;
                int x1;

                if (isBull)
                {
                    if (follow)
                    {
                        val = Fibb(level, _Up, _Dn, _iUp, _iDn);
                        x1 = _iDn;
                    }
                    else
                    {
                        val = Fibb(level, _Up, _swingLow, _iUp, _iSwingLow);
                        x1 = _iSwingLow;
                    }
                }
                else
                {
                    if (follow)
                    {
                        val = Fibb(level, _Up, _Dn, _iUp, _iDn);
                        x1 = _iUp;
                    }
                    else
                    {
                        val = Fibb(level, _Dn, _swingHigh, _iDn, _iSwingHigh);
                        x1 = _iSwingHigh;
                    }
                }

                if (double.IsNaN(val) || x1 == int.MinValue)
                    continue;

                UpdateHorizontalLine(_flevels[i], x1, index, val);
            }

            if (_trend != null)
            {
                if (isBull)
                {
                    if (follow && _iDn != int.MinValue && !double.IsNaN(_Dn))
                        UpdateTrendXY1(_trend, _iDn, _Dn);

                    UpdateTrendXY2(_trend, index, _Up);
                }
                else
                {
                    if (follow && _iUp != int.MinValue && !double.IsNaN(_Up))
                        UpdateTrendXY1(_trend, _iUp, _Up);

                    UpdateTrendXY2(_trend, index, _Dn);
                }
            }

            if (swinglab)
            {
                if (isBull)
                {
                    RemoveIfExists(_endUpLabelName);
                    _endUpLabelName = NextName("endu");
                    DrawPriceLabel(_endUpLabelName, index, _Up, downStyle: true);
                }
                else
                {
                    RemoveIfExists(_endDnLabelName);
                    _endDnLabelName = NextName("endd");
                    DrawPriceLabel(_endDnLabelName, index, _Dn, downStyle: false);
                }
            }
        }

        // =========================
        // Drawing helpers
        // =========================
        private string NextName(string prefix) => $"ote_zei_{prefix}_{_id++}";

        private DateTime TimeOf(int idx)
        {
            if (idx < 0) idx = 0;
            if (idx >= Bars.Count) idx = Bars.Count - 1;
            return Bars.OpenTimes[idx];
        }

        private void DrawChoch(int xIndex, double y, string text, Color col, bool labelDown)
        {
            var name = NextName("choch");
            var t = Chart.DrawText(name, text, TimeOf(xIndex), y, col);
            t.IsInteractive = false;
            t.HorizontalAlignment = HorizontalAlignment.Center;
            t.VerticalAlignment = labelDown ? VerticalAlignment.Top : VerticalAlignment.Bottom;
        }

        private void DrawBos(int x1, int x2, double y, Color col)
        {
            var ln = DrawHorizontalLine(NextName("bos"), x1, x2, y, col, s_width);
            ln.LineStyle = LineStyle.Solid;
        }

        private void CreateSwingTrend(int x1, double y1, int x2, double y2, Color c)
        {
            _trendName = NextName("trend");
            _trend = Chart.DrawTrendLine(_trendName, TimeOf(x1), y1, TimeOf(x2), y2, c);
            _trend.IsInteractive = false;
            _trend.Thickness = swline_width;
            _trend.LineStyle = LineStyle.Dots;
        }

        private void DrawPriceLabel(string name, int xIndex, double y, bool downStyle)
        {
            var col = Chart.ColorSettings.ForegroundColor;
            var txt = y.ToString("0.#####", CultureInfo.InvariantCulture);

            var t = Chart.DrawText(name, txt, TimeOf(xIndex), y, col);
            t.IsInteractive = false;
            t.HorizontalAlignment = HorizontalAlignment.Center;
            t.VerticalAlignment = downStyle ? VerticalAlignment.Top : VerticalAlignment.Bottom;
        }

        private ChartTrendLine DrawHorizontalLine(string name, int x1, int x2, double y, Color col, int width)
        {
            var ln = Chart.DrawTrendLine(name, TimeOf(x1), y, TimeOf(x2), y, col);
            ln.IsInteractive = false;
            ln.Thickness = width;
            ln.LineStyle = LineStyle.Solid;
            return ln;
        }

        private void UpdateHorizontalLine(ChartTrendLine ln, int x1, int x2, double y)
        {
            if (ln == null) return;
            ln.Time1 = TimeOf(x1);
            ln.Time2 = TimeOf(x2);
            ln.Y1 = y;
            ln.Y2 = y;
        }

        private void UpdateTrendXY1(ChartTrendLine ln, int x1, double y1)
        {
            if (ln == null) return;
            ln.Time1 = TimeOf(x1);
            ln.Y1 = y1;
        }

        private void UpdateTrendXY2(ChartTrendLine ln, int x2, double y2)
        {
            if (ln == null) return;
            ln.Time2 = TimeOf(x2);
            ln.Y2 = y2;
        }

        // =========================
        // Golden zone fill (FIXED: no FillColor)
        // =========================
        private void UpdateGoldenFill(int index)
        {
            if (!TryGetGoldenZoneBounds(out var top, out var bot, out var isBullishZone, out var isBearishZone))
            {
                RemoveGoldenFill();
                return;
            }

            var sourceColor = isBullishZone ? bullGoldZone : bearGoldZone;
            var fillColor = Color.FromArgb(153, sourceColor.R, sourceColor.G, sourceColor.B);

            var a = _flevels[0];
            var b = _flevels[1];
            DateTime left = a.Time1 > b.Time1 ? a.Time1 : b.Time1;
            DateTime right = TimeOf(index);

            if (_goldRect == null)
            {
                _goldRectName = NextName("gold");
                _goldRect = Chart.DrawRectangle(_goldRectName, left, top, right, bot, fillColor);
                _goldRect.IsInteractive = false;
                _goldRect.IsFilled = true;
                _goldRect.Color = fillColor;
                _goldRect.Thickness = 0;
            }
            else
            {
                _goldRect.Time1 = left;
                _goldRect.Time2 = right;
                _goldRect.Y1 = top;
                _goldRect.Y2 = bot;

                _goldRect.IsFilled = true;
                _goldRect.Color = fillColor;
                _goldRect.Thickness = 0;
            }
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

        // =========================
        // Combined OB + FVG logic (with OTE golden-zone signal filter)
        // =========================
        private void ProcessOrderBlockDetector(int index)
        {
            if (index < 2)
                return;

            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
            if (sourceIndex < 2)
                return;

            EnsureHeikinAshiSource(sourceIndex);
            UpdateObFvgBoxes(index);

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

            ApplyGoldenZoneFilter(index, ref cond);
            ApplyGoldenZoneFilter(index, ref condFvg);

            DrawObSignals(index, cond, condFvg);
        }

        private void ApplyGoldenZoneFilter(int index, ref int cond)
        {
            if (cond == 0)
                return;

            if (!TryGetGoldenZoneBounds(out var top, out var bot, out var isBullishZone, out var isBearishZone))
            {
                cond = 0;
                return;
            }

            var price = Bars.ClosePrices[index];
            var isInsideZone = price >= bot && price <= top;

            if (!isInsideZone || (cond == 1 && !isBullishZone) || (cond == -1 && !isBearishZone))
                cond = 0;
        }

        private bool TryGetGoldenZoneBounds(out double top, out double bot, out bool isBullishZone, out bool isBearishZone)
        {
            top = bot = double.NaN;
            isBullishZone = false;
            isBearishZone = false;

            if (_flevels.Count <= 1 || _flevels[0] == null || _flevels[1] == null)
                return false;

            var level0 = _flevels[0];
            var level1 = _flevels[1];
            top = Math.Max(level0.Y1, level1.Y1);
            bot = Math.Min(level0.Y1, level1.Y1);

            isBullishZone = _pos > 0;
            isBearishZone = _pos < 0;

            if (!isBullishZone && !isBearishZone)
            {
                isBullishZone = level1.Y1 > level0.Y1;
                isBearishZone = level1.Y1 < level0.Y1;
            }

            return isBullishZone || isBearishZone;
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

            var id = $"gobd_ob_box_{_sourceBars.OpenTimes[sourceIndex].Ticks}";
            var boxColor = Color.FromArgb((int)Math.Round(255.0 * (100 - Transparency) / 100.0), isBull ? ColorBull : ColorBear);
            var box = Chart.DrawRectangle(id, Bars.OpenTimes[Math.Max(0, index - 1)], max, Bars.OpenTimes[index], min, boxColor);
            box.IsFilled = true;
            box.Color = boxColor;
            box.IsInteractive = false;

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

            var id = $"gobd_fvg_box_{_sourceBars.OpenTimes[sourceIndex].Ticks}";
            var boxColor = Color.FromArgb((int)Math.Round(255.0 * (100 - Transparency) / 100.0), isBull ? ColorFvgBull : ColorFvgBear);
            var box = Chart.DrawRectangle(id, Bars.OpenTimes[Math.Max(0, index - 1)], max, Bars.OpenTimes[index], min, boxColor);
            box.IsFilled = true;
            box.Color = boxColor;
            box.IsInteractive = false;

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
            for (int i = _obRecords.Count - 1; i >= 0; i--)
            {
                var r = _obRecords[i];
                var now = Bars.OpenTimes[index];

                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        DrawObLiquidationLine($"gobd_ob_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Max, ColorBull);
                        Chart.RemoveObject(r.BoxId);
                        _obRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        DrawObLiquidationLine($"gobd_ob_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Min, ColorBear);
                        Chart.RemoveObject(r.BoxId);
                        _obRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDist < index)
                            _signal = NewSignal(r.Min, false);
                    }
                }
            }
        }

        private void HandleMitigationFvg(int index, double sLow, double sHigh)
        {
            for (int i = _fvgRecords.Count - 1; i >= 0; i--)
            {
                var r = _fvgRecords[i];
                var now = Bars.OpenTimes[index];

                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        DrawObLiquidationLine($"gobd_fvg_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Max, ColorFvgBull);
                        Chart.RemoveObject(r.BoxId);
                        _fvgRecords.RemoveAt(i);

                        if (r.DetectionChartIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(r.Max, true);
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        DrawObLiquidationLine($"gobd_fvg_line_{r.DetectionTime.Ticks}_{i}", r.DetectionTime, now, r.Min, ColorFvgBear);
                        Chart.RemoveObject(r.BoxId);
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
            return new SignalState
            {
                Point = point,
                IsBull = isBull,
                Entry = false
            };
        }

        private void DrawObSignals(int index, int cond, int condFvg)
        {
            var offset = SignalOffsetPips * Symbol.PipSize;

            if (ShowSignalsOb && cond == 1)
                DrawSignalIcon($"gobd_ob_buy_{index}_{_shapeId++}", ChartIconType.UpArrow, index, Bars.LowPrices[index], ColorBull, -offset);
            if (ShowSignalsOb && cond == -1)
                DrawSignalIcon($"gobd_ob_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], ColorBear, offset);

            if (ShowSignalsFvg && condFvg == 1)
                DrawSignalIcon($"gobd_fvg_buy_{index}_{_shapeId++}", ChartIconType.UpArrow, index, Bars.LowPrices[index], ColorFvgBull, -offset);
            if (ShowSignalsFvg && condFvg == -1)
                DrawSignalIcon($"gobd_fvg_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], ColorFvgBear, offset);

            if (cond == 1 || condFvg == 1)
                LongSignal[index] = Bars.LowPrices[index] - offset;

            if (cond == -1 || condFvg == -1)
                ShortSignal[index] = Bars.HighPrices[index] + offset;
        }

        private void DrawSignalIcon(string id, ChartIconType type, int index, double y, Color color, double delta)
        {
            Chart.DrawIcon(id, type, Bars.OpenTimes[index], y + delta, color);
        }

        private void DrawObLiquidationLine(string id, DateTime from, DateTime to, double price, Color color)
        {
            var line = Chart.DrawTrendLine(id, from, price, to, price, color, LineWidthLiquidated, LineStyle.LinesDots);
            line.ExtendToInfinity = false;
        }

        private void UpdateObFvgBoxes(int index)
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

        // =========================
        // Cleanup helpers
        // =========================
        private void ClearActiveSet(bool deleteDrawings)
        {
            if (deleteDrawings)
            {
                if (!string.IsNullOrEmpty(_trendName))
                    Chart.RemoveObject(_trendName);

                for (int i = 0; i < _flevelNames.Count; i++)
                {
                    if (!string.IsNullOrEmpty(_flevelNames[i]))
                        Chart.RemoveObject(_flevelNames[i]);
                }

                RemoveGoldenFill();
            }
            else
            {
                // Keep previously drawn golden rectangles when Previous=true.
                _goldRect = null;
                _goldRectName = null;
            }

            _trend = null;
            _trendName = null;

            _flevels.Clear();
            _flevelNames.Clear();
        }

        private void RemoveIfExists(string name)
        {
            if (!string.IsNullOrEmpty(name))
                Chart.RemoveObject(name);
        }
    }
}
