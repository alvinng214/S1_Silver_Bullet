// Fibonacci Optimal Entry Zone [OTE] (Zeiierman).cs
// cTrader indicator translation from TradingView Pine Script:
// "Fibonacci Optimal Entry Zone [OTE] (Zeiierman)"
//
// Notes / limitations (cTrader vs Pine):
// - TradingView label styles (label_up/label_down bubble/arrow) are approximated with Chart.DrawText alignment.
// - TradingView linefill between two lines is approximated with a filled ChartRectangle between those levels.
// - This implementation mirrors the pivot/state machine + fib level updates as closely as possible.

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class FibonacciOptimalEntryZoneOTEZeiierman : Indicator
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

        // Pine: color.new(color.teal, 50)
        [Parameter("Golden Zone Color", DefaultValue = "#80009688", Group = "Fibonacci")]
        public Color goldZone { get; set; }

        // Levels (Pine has 2 configurable levels)
        [Parameter("L1 Enabled", DefaultValue = true, Group = "Levels")]
        public bool level1Enabled { get; set; }

        [Parameter("L1 Value", DefaultValue = 0.50, Group = "Levels")]
        public double level1Value { get; set; }

        [Parameter("L1 Color", DefaultValue = "#4CAF50", Group = "Levels")]
        public Color level1Color { get; set; }

        [Parameter("L2 Enabled", DefaultValue = true, Group = "Levels")]
        public bool level2Enabled { get; set; }

        [Parameter("L2 Value", DefaultValue = 0.618, Group = "Levels")]
        public double level2Value { get; set; }

        [Parameter("L2 Color", DefaultValue = "#009688", Group = "Levels")]
        public Color level2Color { get; set; }

        [Parameter("Fibb Width", DefaultValue = 1, MinValue = 1, MaxValue = 10, Group = "Levels")]
        public int fibb_width { get; set; }

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

        private readonly List<ChartTrendLine> _flevels = new();      // active fib levels (order mirrors Pine usage)
        private readonly List<string> _flevelNames = new();          // names for deletion when showOld == false

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

        protected override void Initialize()
        {
            RebuildLevels();
        }

        public override void Calculate(int index)
        {
            if (index < prd * 2 + 2)
                return;

            // Cache previous series values
            double UpPrev = _Up;
            double DnPrev = _Dn;
            int iUpPrev = _iUp;
            int iDnPrev = _iDn;

            // Up := math.max(Up[1], high)
            // Dn := math.min(Dn[1], low)
            double Up = double.IsNaN(UpPrev) ? double.NaN : Math.Max(UpPrev, Bars.HighPrices[index]);
            double Dn = double.IsNaN(DnPrev) ? double.NaN : Math.Min(DnPrev, Bars.LowPrices[index]);

            // Pivot detection (ta.pivothigh/low) affects Up/Dn only when pos gate matches
            if (TryPivotHigh(index, prd, out _, out var ph) && _pos <= 0)
                Up = ph;

            if (TryPivotLow(index, prd, out _, out var pl) && _pos >= 0)
                Dn = pl;

            _Up = Up;
            _Dn = Dn;

            // =========================
            // Bullish structure block
            // =========================
            if (!double.IsNaN(_Up) && !double.IsNaN(UpPrev) && _Up > UpPrev)
            {
                _iUp = index;
                int? centerBull = CenterIndexOrNa(iUpPrev, index);

                if (_pos <= 0)
                {
                    // New bullish leg
                    ClearActiveSet(deleteDrawings: !showOld);

                    if (bull)
                    {
                        // CHoCH label at centerBull, Up[1]
                        if (centerBull.HasValue)
                            DrawChoch(centerBull.Value, UpPrev, "CHoCH", bull2, labelDown: true);

                        // BoS line at Up[1] from iUp[1] to b
                        if (iUpPrev != int.MinValue)
                            DrawBos(iUpPrev, index, UpPrev, bull2);

                        // Fib levels (anchored by iDn < iUp)
                        if (_iDn != int.MinValue && !double.IsNaN(_Dn))
                        {
                            BuildFibLevelsForCurrentLeg(isBull: true, index: index);
                        }

                        // Swing trendline
                        if (swingline && _iDn != int.MinValue && !double.IsNaN(_Dn))
                            CreateSwingTrend(isBull: true, x1: _iDn, y1: _Dn, x2: index, y2: _Up, c: bull2);

                        // End label
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
                    // First continuation update
                    if (bull)
                        UpdateFibAndSwing(isBull: true, index: index);

                    _pos = 2;
                }
                else
                {
                    // Further continuation
                    if (bull)
                    {
                        UpdateFibAndSwing(isBull: true, index: index);

                        // Start label (at iDn)
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
                // iUp := b - prd
                _iUp = index - prd;
            }

            // =========================
            // Bearish structure block
            // =========================
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
                        {
                            BuildFibLevelsForCurrentLeg(isBull: false, index: index);
                        }

                        if (swingline && _iUp != int.MinValue && !double.IsNaN(_Up))
                            CreateSwingTrend(isBull: false, x1: _iUp, y1: _Up, x2: index, y2: _Dn, c: bear2);

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

            // Extend lines if enabled (Pine l.set_x2 each bar)
            if (extend && _flevels.Count > 0)
            {
                var t2 = TimeOf(index);
                for (int i = 0; i < _flevels.Count; i++)
                    _flevels[i].Time2 = t2;

                if (_trend != null)
                    _trend.Time2 = t2;

                if (_goldRect != null)
                    _goldRect.Time2 = t2;
            }

            // Golden zone fill
            if (golden && _flevels.Count > 1)
                UpdateGoldenFill(index);
            else
                RemoveGoldenFill();
        }

        // =========================
        // Level list build (Pine: levels.push, colors.unshift)
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
        private bool TryPivotHigh(int index, int prd, out int pivotIndex, out double pivotHigh)
        {
            pivotIndex = index - prd;
            pivotHigh = double.NaN;

            if (pivotIndex - prd < 0) return false;
            if (pivotIndex + prd > index) return false;

            double ph = Bars.HighPrices[pivotIndex];
            for (int i = pivotIndex - prd; i <= pivotIndex + prd; i++)
            {
                if (i == pivotIndex) continue;
                if (Bars.HighPrices[i] >= ph) return false;
            }

            pivotHigh = ph;
            return true;
        }

        private bool TryPivotLow(int index, int prd, out int pivotIndex, out double pivotLow)
        {
            pivotIndex = index - prd;
            pivotLow = double.NaN;

            if (pivotIndex - prd < 0) return false;
            if (pivotIndex + prd > index) return false;

            double pl = Bars.LowPrices[pivotIndex];
            for (int i = pivotIndex - prd; i <= pivotIndex + prd; i++)
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
        // Build fib levels for current leg
        // =========================
        private void BuildFibLevelsForCurrentLeg(bool isBull, int index)
        {
            // In Pine, lines are created with flevels.unshift(...) while iterating levels in order.
            // That means the final list order matches "reverse insertion".
            // Here we replicate by inserting at 0 each time.

            _flevels.Clear();
            _flevelNames.Clear();

            for (int i = 0; i < _levels.Count; i++)
            {
                double level = _levels[i];
                Color col = _colors[i];

                double val;

                if (isBull)
                    val = Fibb(level, _Up, _Dn, _iUp, _iDn);
                else
                    val = Fibb(level, _Dn, _Up, _iDn, _iUp);

                if (double.IsNaN(val))
                    continue;

                int x1 = isBull ? _iDn : _iUp;
                int x2 = index;

                var ln = DrawHorizontalLine(NextName("fib"), x1, x2, val, col, fibb_width);

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
        // Draw helpers
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
            // Pine uses line style solid by default for structure lines; keep Solid.
            ln.LineStyle = LineStyle.Solid;
        }

        private void CreateSwingTrend(bool isBull, int x1, double y1, int x2, double y2, Color c)
        {
            _trendName = NextName("trend");
            _trend = Chart.DrawTrendLine(_trendName, TimeOf(x1), y1, TimeOf(x2), y2, c);
            _trend.IsInteractive = false;
            _trend.Thickness = swline_width;
            _trend.LineStyle = LineStyle.Dots; // Pine uses dotted for swing line
        }

        private void DrawPriceLabel(string name, int xIndex, double y, bool downStyle)
        {
            // Pine uses chart.fg_color; cTrader closest is ForegroundColor
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
        // Golden zone fill
        // =========================
        private void UpdateGoldenFill(int index)
        {
            var a = _flevels[0];
            var b = _flevels[1];
            if (a == null || b == null)
                return;

            double yA = a.Y1;
            double yB = b.Y1;
            double top = Math.Max(yA, yB);
            double bot = Math.Min(yA, yB);

            // Use the later of the two start times as left edge (close to Pine’s linefill semantics)
            DateTime left = a.Time1 > b.Time1 ? a.Time1 : b.Time1;
            DateTime right = TimeOf(index);

            if (_goldRect == null)
            {
                _goldRectName = NextName("gold");
                _goldRect = Chart.DrawRectangle(_goldRectName, left, top, right, bot, Color.Transparent);
                _goldRect.IsInteractive = false;
                _goldRect.IsFilled = true;
                _goldRect.FillColor = goldZone;
                _goldRect.Thickness = 0;
                _goldRect.Color = Color.Transparent;
            }
            else
            {
                _goldRect.Time1 = left;
                _goldRect.Time2 = right;
                _goldRect.Y1 = top;
                _goldRect.Y2 = bot;
                _goldRect.FillColor = goldZone;
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
                // Keep drawings (showOld == true) but detach references
                RemoveGoldenFill();
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
