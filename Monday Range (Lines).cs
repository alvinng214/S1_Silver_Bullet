// Monday Range (Lines).cs
// cTrader indicator conversion of:
// S1_Silver_Bullet/Monday Range (Lines).txt (Pine Script v6)
//
// Parity intent:
// - Mirror Monday range storage/drawing logic and UI toggles as closely as possible in cTrader.
// - Preserve breakout/reclaim detection logic and marker controls.
//
// Platform limitations (documented):
// - TradingView alertcondition() cannot be created from cTrader indicators.
// - TradingView label styles/sizes are approximated with cTrader icons and font sizes.

using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MondayRangeLines : Indicator
    {
        private const int MaxMondaysArraySize = 52;

        private const string GroupDisplay = "Display";
        private const string GroupExtension = "Line Extension";
        private const string GroupMainLevels = "Monday High, Low, Open & Close levels";
        private const string GroupCustomLevels = "Custom levels";
        private const string GroupLabels = "Labels";
        private const string GroupAlerts = "Alerts";

        public enum ExtensionMode
        {
            EndOfWeek,
            CurrentBar,
            FixedBars
        }

        public enum LineStyleOption
        {
            Solid,
            Dotted,
            Dashed
        }

        public enum TvSize
        {
            Auto,
            Tiny,
            Small,
            Normal,
            Large,
            Huge
        }

        public enum MarkerStyle
        {
            LabelDown,
            LabelUp,
            TriangleDown,
            TriangleUp
        }

        private sealed class Monday
        {
            public DateTime WeekStart;
            public DateTime WeekEnd;
            public double Open;
            public double High;
            public double Low;
            public double Close;

            public double Range => High - Low;
        }

        private sealed class RangeEvent
        {
            public DateTime BarTime;
            public double Price;
            public long WeekKey;
        }

        private sealed class LevelConfig
        {
            public bool Enabled;
            public string Text;
            public double Value;
            public Color Color;
            public LineStyleOption LineStyle;
            public int Width;
        }

        [Parameter("Number of weeks of ranges to display", DefaultValue = 4, MinValue = 1, MaxValue = MaxMondaysArraySize, Group = GroupDisplay)]
        public int MaxMondays { get; set; }

        [Parameter("Line extension", DefaultValue = ExtensionMode.EndOfWeek, Group = GroupExtension)]
        public ExtensionMode ExtensionType { get; set; }

        [Parameter("Fixed daily bars count", DefaultValue = 5, MinValue = 1, MaxValue = 50, Group = GroupExtension)]
        public int FixedBarsCount { get; set; }

        [Parameter("MH Enabled", DefaultValue = true, Group = GroupMainLevels)]
        public bool UseMh { get; set; }
        [Parameter("MH Text", DefaultValue = "MH", Group = GroupMainLevels)]
        public string MhText { get; set; }
        [Parameter("MH Color", DefaultValue = "#007FFF", Group = GroupMainLevels)]
        public Color MhColor { get; set; }
        [Parameter("MH Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupMainLevels)]
        public LineStyleOption MhStyle { get; set; }
        [Parameter("MH Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupMainLevels)]
        public int MhWidth { get; set; }

        [Parameter("ML Enabled", DefaultValue = true, Group = GroupMainLevels)]
        public bool UseMl { get; set; }
        [Parameter("ML Text", DefaultValue = "ML", Group = GroupMainLevels)]
        public string MlText { get; set; }
        [Parameter("ML Color", DefaultValue = "#007FFF", Group = GroupMainLevels)]
        public Color MlColor { get; set; }
        [Parameter("ML Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupMainLevels)]
        public LineStyleOption MlStyle { get; set; }
        [Parameter("ML Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupMainLevels)]
        public int MlWidth { get; set; }

        [Parameter("MO Enabled", DefaultValue = false, Group = GroupMainLevels)]
        public bool UseMo { get; set; }
        [Parameter("MO Text", DefaultValue = "MO", Group = GroupMainLevels)]
        public string MoText { get; set; }
        [Parameter("MO Color", DefaultValue = "#007FFF", Group = GroupMainLevels)]
        public Color MoColor { get; set; }
        [Parameter("MO Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupMainLevels)]
        public LineStyleOption MoStyle { get; set; }
        [Parameter("MO Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupMainLevels)]
        public int MoWidth { get; set; }

        [Parameter("MC Enabled", DefaultValue = false, Group = GroupMainLevels)]
        public bool UseMc { get; set; }
        [Parameter("MC Text", DefaultValue = "MC", Group = GroupMainLevels)]
        public string McText { get; set; }
        [Parameter("MC Color", DefaultValue = "#007FFF", Group = GroupMainLevels)]
        public Color McColor { get; set; }
        [Parameter("MC Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupMainLevels)]
        public LineStyleOption McStyle { get; set; }
        [Parameter("MC Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupMainLevels)]
        public int McWidth { get; set; }

        [Parameter("#1 Enabled", DefaultValue = true, Group = GroupCustomLevels)]
        public bool UseHz1 { get; set; }
        [Parameter("#1 Text", DefaultValue = "EQ", Group = GroupCustomLevels)]
        public string Hz1Text { get; set; }
        [Parameter("#1 Value", DefaultValue = 0.5, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz1Value { get; set; }
        [Parameter("#1 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz1Color { get; set; }
        [Parameter("#1 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz1Style { get; set; }
        [Parameter("#1 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz1Width { get; set; }

        [Parameter("#2 Enabled", DefaultValue = false, Group = GroupCustomLevels)]
        public bool UseHz2 { get; set; }
        [Parameter("#2 Text", DefaultValue = "", Group = GroupCustomLevels)]
        public string Hz2Text { get; set; }
        [Parameter("#2 Value", DefaultValue = 0.0, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz2Value { get; set; }
        [Parameter("#2 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz2Color { get; set; }
        [Parameter("#2 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz2Style { get; set; }
        [Parameter("#2 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz2Width { get; set; }

        [Parameter("#3 Enabled", DefaultValue = false, Group = GroupCustomLevels)]
        public bool UseHz3 { get; set; }
        [Parameter("#3 Text", DefaultValue = "", Group = GroupCustomLevels)]
        public string Hz3Text { get; set; }
        [Parameter("#3 Value", DefaultValue = 0.0, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz3Value { get; set; }
        [Parameter("#3 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz3Color { get; set; }
        [Parameter("#3 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz3Style { get; set; }
        [Parameter("#3 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz3Width { get; set; }

        [Parameter("#4 Enabled", DefaultValue = false, Group = GroupCustomLevels)]
        public bool UseHz4 { get; set; }
        [Parameter("#4 Text", DefaultValue = "", Group = GroupCustomLevels)]
        public string Hz4Text { get; set; }
        [Parameter("#4 Value", DefaultValue = 0.0, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz4Value { get; set; }
        [Parameter("#4 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz4Color { get; set; }
        [Parameter("#4 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz4Style { get; set; }
        [Parameter("#4 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz4Width { get; set; }

        [Parameter("#5 Enabled", DefaultValue = false, Group = GroupCustomLevels)]
        public bool UseHz5 { get; set; }
        [Parameter("#5 Text", DefaultValue = "", Group = GroupCustomLevels)]
        public string Hz5Text { get; set; }
        [Parameter("#5 Value", DefaultValue = 0.0, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz5Value { get; set; }
        [Parameter("#5 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz5Color { get; set; }
        [Parameter("#5 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz5Style { get; set; }
        [Parameter("#5 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz5Width { get; set; }

        [Parameter("#6 Enabled", DefaultValue = false, Group = GroupCustomLevels)]
        public bool UseHz6 { get; set; }
        [Parameter("#6 Text", DefaultValue = "", Group = GroupCustomLevels)]
        public string Hz6Text { get; set; }
        [Parameter("#6 Value", DefaultValue = 0.0, Step = 0.01, Group = GroupCustomLevels)]
        public double Hz6Value { get; set; }
        [Parameter("#6 Color", DefaultValue = "#007FFF", Group = GroupCustomLevels)]
        public Color Hz6Color { get; set; }
        [Parameter("#6 Line Style", DefaultValue = LineStyleOption.Solid, Group = GroupCustomLevels)]
        public LineStyleOption Hz6Style { get; set; }
        [Parameter("#6 Width", DefaultValue = 1, MinValue = 1, MaxValue = 5, Group = GroupCustomLevels)]
        public int Hz6Width { get; set; }

        [Parameter("Text Colour", DefaultValue = "Gray", Group = GroupLabels)]
        public Color TextColor { get; set; }
        [Parameter("Label Size", DefaultValue = TvSize.Small, Group = GroupLabels)]
        public TvSize LabelSize { get; set; }

        [Parameter("Enable alerts", DefaultValue = true, Group = GroupAlerts)]
        public bool EnableAlerts { get; set; }

        [Parameter("Show breakout markers on chart", DefaultValue = false, Group = GroupAlerts)]
        public bool ShowBreakoutLabels { get; set; }
        [Parameter("High break", DefaultValue = "#D4A574", Group = GroupAlerts)]
        public Color HighBreakColor { get; set; }
        [Parameter("High break style", DefaultValue = MarkerStyle.LabelDown, Group = GroupAlerts)]
        public MarkerStyle HighBreakStyle { get; set; }
        [Parameter("High break label size", DefaultValue = TvSize.Small, Group = GroupAlerts)]
        public TvSize HighBreakLabelSize { get; set; }

        [Parameter("Low break", DefaultValue = "#3498DB", Group = GroupAlerts)]
        public Color LowBreakColor { get; set; }
        [Parameter("Low break style", DefaultValue = MarkerStyle.LabelUp, Group = GroupAlerts)]
        public MarkerStyle LowBreakStyle { get; set; }
        [Parameter("Low break label size", DefaultValue = TvSize.Small, Group = GroupAlerts)]
        public TvSize LowBreakLabelSize { get; set; }

        [Parameter("Show reclaim markers on chart", DefaultValue = false, Group = GroupAlerts)]
        public bool ShowReclaimLabels { get; set; }
        [Parameter("High reclaim", DefaultValue = "#9B59B6", Group = GroupAlerts)]
        public Color HighReclaimColor { get; set; }
        [Parameter("High reclaim style", DefaultValue = MarkerStyle.LabelDown, Group = GroupAlerts)]
        public MarkerStyle HighReclaimStyle { get; set; }
        [Parameter("High reclaim label size", DefaultValue = TvSize.Small, Group = GroupAlerts)]
        public TvSize HighReclaimLabelSize { get; set; }

        [Parameter("Low reclaim", DefaultValue = "#2ECC71", Group = GroupAlerts)]
        public Color LowReclaimColor { get; set; }
        [Parameter("Low reclaim style", DefaultValue = MarkerStyle.LabelUp, Group = GroupAlerts)]
        public MarkerStyle LowReclaimStyle { get; set; }
        [Parameter("Low reclaim label size", DefaultValue = TvSize.Small, Group = GroupAlerts)]
        public TvSize LowReclaimLabelSize { get; set; }

        private Bars _weeklyBars;
        private Bars _dailyBars;

        private readonly Dictionary<long, Monday> _mondaysMap = new Dictionary<long, Monday>();
        private readonly List<long> _mondaysOrder = new List<long>();

        private readonly List<RangeEvent> _highBreakouts = new List<RangeEvent>();
        private readonly List<RangeEvent> _lowBreakouts = new List<RangeEvent>();
        private readonly List<RangeEvent> _highReclaims = new List<RangeEvent>();
        private readonly List<RangeEvent> _lowReclaims = new List<RangeEvent>();

        private readonly Dictionary<long, bool> _highTouched = new Dictionary<long, bool>();
        private readonly Dictionary<long, bool> _lowTouched = new Dictionary<long, bool>();

        private int _lastProcessed = -1;

        protected override void Initialize()
        {
            _weeklyBars = MarketData.GetBars(TimeFrame.Weekly);
            _dailyBars = MarketData.GetBars(TimeFrame.Daily);
        }

        public override void Calculate(int index)
        {
            if (index <= _lastProcessed)
                return;

            _lastProcessed = index;

            if (IsWeeklyOrMonthly() || Bars.Count < 2)
                return;

            BuildMondayStorage();
            DetectBreakoutsAndReclaims(index);

            if (index == Bars.Count - 1)
                DrawAll();
        }

        private bool IsWeeklyOrMonthly()
        {
            return TimeFrame == TimeFrame.Weekly || TimeFrame == TimeFrame.Monthly;
        }

        private void BuildMondayStorage()
        {
            if (_weeklyBars == null || _dailyBars == null || _weeklyBars.Count < 1 || _dailyBars.Count < 1)
                return;

            for (var w = 0; w < _weeklyBars.Count; w++)
            {
                var wkStart = _weeklyBars.OpenTimes[w];
                var wkEnd = w < _weeklyBars.Count - 1 ? _weeklyBars.OpenTimes[w + 1] : wkStart.AddDays(7);
                var wkKey = wkStart.Ticks;

                if (_mondaysMap.ContainsKey(wkKey))
                    continue;

                var dIndex = FirstDailyBarInWeek(wkStart, wkEnd);
                if (dIndex < 0)
                    continue;

                var monday = new Monday
                {
                    WeekStart = wkStart,
                    WeekEnd = wkEnd,
                    Open = _dailyBars.OpenPrices[dIndex],
                    High = _dailyBars.HighPrices[dIndex],
                    Low = _dailyBars.LowPrices[dIndex],
                    Close = _dailyBars.ClosePrices[dIndex]
                };

                _mondaysMap[wkKey] = monday;
                _mondaysOrder.Insert(0, wkKey);

                while (_mondaysOrder.Count > MaxMondaysArraySize)
                {
                    var oldest = _mondaysOrder[_mondaysOrder.Count - 1];
                    _mondaysOrder.RemoveAt(_mondaysOrder.Count - 1);
                    _mondaysMap.Remove(oldest);
                    _highTouched.Remove(oldest);
                    _lowTouched.Remove(oldest);
                }
            }

            TrimOldEvents();
        }

        private int FirstDailyBarInWeek(DateTime wkStart, DateTime wkEnd)
        {
            for (var i = 0; i < _dailyBars.Count; i++)
            {
                var t = _dailyBars.OpenTimes[i];
                if (t >= wkStart && t < wkEnd)
                    return i;
            }

            return -1;
        }

        private void DetectBreakoutsAndReclaims(int index)
        {
            if (!EnableAlerts || index < 1 || _mondaysOrder.Count == 0)
                return;

            var t = Bars.OpenTimes[index];
            if (!TryGetWeekMonday(t, out var wkKey, out var monday))
                return;

            if (t < monday.WeekStart || t > monday.WeekEnd)
                return;

            // Pine uses opening_day_end = monday.wkStart + one_day_ms
            var openingDayEnd = monday.WeekStart.AddDays(1);
            if (t < openingDayEnd)
                return;

            var close = Bars.ClosePrices[index];
            var prevClose = Bars.ClosePrices[index - 1];
            var high = Bars.HighPrices[index];
            var low = Bars.LowPrices[index];

            if (high > monday.High || close > monday.High)
                _highTouched[wkKey] = true;

            if (low < monday.Low || close < monday.Low)
                _lowTouched[wkKey] = true;

            var highBreak = prevClose <= monday.High && close > monday.High;
            var lowBreak = prevClose >= monday.Low && close < monday.Low;
            var highReclaim = false;
            var lowReclaim = false;

            if (highBreak)
            {
                _highBreakouts.Add(new RangeEvent { BarTime = t, Price = high, WeekKey = wkKey });
                Print("Monday High Break | Price broke above Monday High: {0}", close);
            }

            if (lowBreak)
            {
                _lowBreakouts.Add(new RangeEvent { BarTime = t, Price = low, WeekKey = wkKey });
                Print("Monday Low Break | Price broke below Monday Low: {0}", close);
            }

            var highWasTouched = _highTouched.TryGetValue(wkKey, out var hTouched) && hTouched;
            if (highWasTouched && close < monday.High)
            {
                _highReclaims.Add(new RangeEvent { BarTime = t, Price = high, WeekKey = wkKey });
                _highTouched[wkKey] = false;
                highReclaim = true;
                Print("Monday High Reclaim | Price closed back below Monday High: {0}", close);
            }

            var lowWasTouched = _lowTouched.TryGetValue(wkKey, out var lTouched) && lTouched;
            if (lowWasTouched && close > monday.Low)
            {
                _lowReclaims.Add(new RangeEvent { BarTime = t, Price = low, WeekKey = wkKey });
                _lowTouched[wkKey] = false;
                lowReclaim = true;
                Print("Monday Low Reclaim | Price closed back above Monday Low: {0}", close);
            }

            if (highBreak || lowBreak)
                Print("Monday Range Break (High or Low) | Price broke Monday Range");

            if (highReclaim || lowReclaim)
                Print("Monday Range Reclaim (High or Low) | Price closed back into Monday Range");

            if (highBreak || lowBreak || highReclaim || lowReclaim)
                Print("Monday Range Setup (Any) | Monday Range event triggered at {0}", close);
        }

        private bool TryGetWeekMonday(DateTime time, out long wkKey, out Monday monday)
        {
            wkKey = 0;
            monday = null;

            for (var i = 0; i < _mondaysOrder.Count; i++)
            {
                var key = _mondaysOrder[i];
                if (!_mondaysMap.TryGetValue(key, out var m))
                    continue;

                if (time >= m.WeekStart && time <= m.WeekEnd)
                {
                    wkKey = key;
                    monday = m;
                    return true;
                }
            }

            return false;
        }

        private void DrawAll()
        {
            RemoveObjectsWithPrefix("mr_");

            var levelConfigs = BuildCustomConfigs();
            var endIndex = Math.Min(MaxMondays, _mondaysOrder.Count);

            for (var i = 0; i < endIndex; i++)
            {
                var wkKey = _mondaysOrder[i];
                if (!_mondaysMap.TryGetValue(wkKey, out var monday))
                    continue;

                var extEnd = GetExtensionEnd(monday);
                var prefix = $"mr_{wkKey}_";

                DrawLevel(prefix + "mh", monday.WeekStart, extEnd, UseMh, text: MhText, price: monday.High, color: MhColor, width: MhWidth, style: MhStyle, labelSize: LabelSize);
                DrawLevel(prefix + "ml", monday.WeekStart, extEnd, UseMl, text: MlText, price: monday.Low, color: MlColor, width: MlWidth, style: MlStyle, labelSize: LabelSize);
                DrawLevel(prefix + "mo", monday.WeekStart, extEnd, UseMo, text: MoText, price: monday.Open, color: MoColor, width: MoWidth, style: MoStyle, labelSize: LabelSize);
                DrawLevel(prefix + "mc", monday.WeekStart, extEnd, UseMc, text: McText, price: monday.Close, color: McColor, width: McWidth, style: McStyle, labelSize: LabelSize);

                for (var j = 0; j < levelConfigs.Count; j++)
                {
                    var cfg = levelConfigs[j];
                    if (!cfg.Enabled)
                        continue;

                    var levelPrice = monday.Low + monday.Range * cfg.Value;
                    DrawLevel(prefix + $"hz{j + 1}", monday.WeekStart, extEnd, true, text: cfg.Text, price: levelPrice, color: cfg.Color, width: cfg.Width, style: cfg.LineStyle, labelSize: LabelSize);
                }

                if (ShowBreakoutLabels)
                {
                    var offset = monday.Range * 0.05;
                    DrawEvents(_highBreakouts, wkKey, offset, HighBreakColor, HighBreakStyle, HighBreakLabelSize, prefix + "hb_");
                    DrawEvents(_lowBreakouts, wkKey, -offset, LowBreakColor, LowBreakStyle, LowBreakLabelSize, prefix + "lb_");
                }

                if (ShowReclaimLabels)
                {
                    var offset = monday.Range * 0.05;
                    DrawEvents(_highReclaims, wkKey, offset, HighReclaimColor, HighReclaimStyle, HighReclaimLabelSize, prefix + "hr_");
                    DrawEvents(_lowReclaims, wkKey, -offset, LowReclaimColor, LowReclaimStyle, LowReclaimLabelSize, prefix + "lr_");
                }
            }
        }

        private void DrawLevel(string id, DateTime start, DateTime end, bool enabled, string text, double price, Color color, int width, LineStyleOption style, TvSize labelSize)
        {
            if (!enabled)
                return;

            Chart.DrawTrendLine(id + "_line", start, price, end, price, color, Math.Max(1, width), ToLineStyle(style));

            if (!string.IsNullOrEmpty(text))
            {
                var txt = Chart.DrawText(id + "_txt", text, end, price, TextColor);
                txt.FontSize = ToFontSize(labelSize);

                // cTrader text has no TV label.style_label_left equivalent shape/background.
                // We keep text anchored at extension end.
            }
        }

        private void DrawEvents(List<RangeEvent> events, long wkKey, double offset, Color color, MarkerStyle markerStyle, TvSize markerSize, string idPrefix)
        {
            var markerText = MarkerGlyph(markerStyle);
            var font = ToFontSize(markerSize);

            for (var i = 0; i < events.Count; i++)
            {
                var ev = events[i];
                if (ev.WeekKey != wkKey)
                    continue;

                var text = Chart.DrawText(idPrefix + i, markerText, ev.BarTime, ev.Price + offset, color);
                text.FontSize = font;
            }
        }

        private DateTime GetExtensionEnd(Monday monday)
        {
            switch (ExtensionType)
            {
                case ExtensionMode.CurrentBar:
                    return Bars.OpenTimes.LastValue;
                case ExtensionMode.FixedBars:
                    return monday.WeekStart.AddDays(FixedBarsCount);
                default:
                    return monday.WeekEnd;
            }
        }

        private List<LevelConfig> BuildCustomConfigs()
        {
            return new List<LevelConfig>
            {
                new LevelConfig { Enabled = UseHz1, Text = Hz1Text, Value = Hz1Value, Color = Hz1Color, LineStyle = Hz1Style, Width = Hz1Width },
                new LevelConfig { Enabled = UseHz2, Text = Hz2Text, Value = Hz2Value, Color = Hz2Color, LineStyle = Hz2Style, Width = Hz2Width },
                new LevelConfig { Enabled = UseHz3, Text = Hz3Text, Value = Hz3Value, Color = Hz3Color, LineStyle = Hz3Style, Width = Hz3Width },
                new LevelConfig { Enabled = UseHz4, Text = Hz4Text, Value = Hz4Value, Color = Hz4Color, LineStyle = Hz4Style, Width = Hz4Width },
                new LevelConfig { Enabled = UseHz5, Text = Hz5Text, Value = Hz5Value, Color = Hz5Color, LineStyle = Hz5Style, Width = Hz5Width },
                new LevelConfig { Enabled = UseHz6, Text = Hz6Text, Value = Hz6Value, Color = Hz6Color, LineStyle = Hz6Style, Width = Hz6Width }
            };
        }

        private static LineStyle ToLineStyle(LineStyleOption style)
        {
            switch (style)
            {
                case LineStyleOption.Dotted:
                    return LineStyle.Dots;
                case LineStyleOption.Dashed:
                    return LineStyle.Lines;
                default:
                    return LineStyle.Solid;
            }
        }

        private static string MarkerGlyph(MarkerStyle style)
        {
            switch (style)
            {
                case MarkerStyle.TriangleDown:
                    return "▼";
                case MarkerStyle.TriangleUp:
                    return "▲";
                case MarkerStyle.LabelDown:
                    return "↓";
                default:
                    return "↑";
            }
        }

        private static int ToFontSize(TvSize size)
        {
            switch (size)
            {
                case TvSize.Tiny:
                    return 8;
                case TvSize.Small:
                    return 10;
                case TvSize.Normal:
                    return 12;
                case TvSize.Large:
                    return 14;
                case TvSize.Huge:
                    return 16;
                default:
                    return 10;
            }
        }

        private void RemoveObjectsWithPrefix(string prefix)
        {
            var names = Chart.Objects.Where(o => o.Name.StartsWith(prefix)).Select(o => o.Name).ToList();
            foreach (var n in names)
                Chart.RemoveObject(n);
        }

        private void TrimOldEvents()
        {
            var valid = new HashSet<long>(_mondaysMap.Keys);
            _highBreakouts.RemoveAll(x => !valid.Contains(x.WeekKey));
            _lowBreakouts.RemoveAll(x => !valid.Contains(x.WeekKey));
            _highReclaims.RemoveAll(x => !valid.Contains(x.WeekKey));
            _lowReclaims.RemoveAll(x => !valid.Contains(x.WeekKey));
        }
    }
}
