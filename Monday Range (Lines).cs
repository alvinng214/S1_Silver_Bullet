// Monday Range (Lines).cs
// cTrader indicator — strict mirror of Pine Script v6 "Monday Range (Lines)"
//
// Platform limitations (documented):
//   - alertcondition() → Print() to Logs tab
//   - label.style_label_left (shaped background) → plain Chart.DrawText (no background shape)
//   - Pine size enums → approximated with integer font sizes
//   - label tooltips → omitted (cTrader DrawText has no tooltip)

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

        private const string GroupDisplay     = "Display";
        private const string GroupExtension   = "Line Extension";
        private const string GroupMainLevels  = "Monday High, Low, Open & Close levels";
        private const string GroupCustomLevels = "Custom levels";
        private const string GroupLabels      = "Labels";
        private const string GroupAlerts      = "Alerts";

        #region Enums

        public enum ExtensionMode  { EndOfWeek, CurrentBar, FixedBars }
        public enum LineStyleOption { Solid, Dotted, Dashed }
        public enum TvSize          { Auto, Tiny, Small, Normal, Large, Huge }
        public enum MarkerStyle     { LabelDown, LabelUp, TriangleDown, TriangleUp }

        #endregion

        #region Internal types

        private sealed class Monday
        {
            public DateTime WeekStart;
            public DateTime WeekEnd;
            public double Open, High, Low, Close;
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

        #endregion

        #region Parameters – Display / Extension

        [Parameter("Number of weeks of ranges to display", DefaultValue = 4, MinValue = 1, MaxValue = MaxMondaysArraySize, Group = GroupDisplay)]
        public int MaxMondays { get; set; }

        [Parameter("Line extension", DefaultValue = ExtensionMode.EndOfWeek, Group = GroupExtension)]
        public ExtensionMode ExtensionType { get; set; }

        [Parameter("Fixed daily bars count", DefaultValue = 5, MinValue = 1, MaxValue = 50, Group = GroupExtension)]
        public int FixedBarsCount { get; set; }

        #endregion

        #region Parameters – Monday OHLC levels

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

        #endregion

        #region Parameters – Custom levels (6)

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

        #endregion

        #region Parameters – Labels

        [Parameter("Text Colour", DefaultValue = "Gray", Group = GroupLabels)]
        public Color TextColor { get; set; }
        [Parameter("Label Size", DefaultValue = TvSize.Small, Group = GroupLabels)]
        public TvSize LabelSize { get; set; }

        #endregion

        #region Parameters – Alerts / Markers

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

        #endregion

        #region Private fields

        private Bars _weeklyBars;
        private Bars _dailyBars;

        // Pine: mondays_map + mondays_order  (newest-first ordered keys)
        private readonly Dictionary<long, Monday> _mondaysMap = new Dictionary<long, Monday>();
        private readonly List<long> _mondaysOrder = new List<long>();

        // Pine: high_breakouts / low_breakouts / high_reclaims / low_reclaims
        private readonly List<RangeEvent> _highBreakouts = new List<RangeEvent>();
        private readonly List<RangeEvent> _lowBreakouts  = new List<RangeEvent>();
        private readonly List<RangeEvent> _highReclaims  = new List<RangeEvent>();
        private readonly List<RangeEvent> _lowReclaims   = new List<RangeEvent>();

        // Pine: high_touched / low_touched
        private readonly Dictionary<long, bool> _highTouched = new Dictionary<long, bool>();
        private readonly Dictionary<long, bool> _lowTouched  = new Dictionary<long, bool>();

        // Track which chart bars have been processed for breakout detection
        private int _lastBreakoutIndex = -1;

        // Track whether Monday storage has been built at least once
        private bool _storageBuilt;

        #endregion

        #region Initialize

        protected override void Initialize()
        {
            _weeklyBars = MarketData.GetBars(TimeFrame.Weekly);
            _dailyBars  = MarketData.GetBars(TimeFrame.Daily);
        }

        #endregion

        #region Calculate — main loop

        public override void Calculate(int index)
        {
            // Pine: var can_show_monday_range = not timeframe.isweekly and not timeframe.ismonthly
            if (IsWeeklyOrMonthly() || Bars.Count < 2)
                return;

            // ── Build/update Monday storage ──
            // First call: populate everything from historical weekly + daily bars.
            // IsLastBar ticks: refresh to catch any new weekly bar that just appeared.
            if (!_storageBuilt || IsLastBar)
            {
                BuildMondayStorage();
                _storageBuilt = true;
            }

            // ── Detect breakouts/reclaims once per new bar ──
            // Pine runs this code on every bar; we gate on index to avoid
            // re-detecting on each tick of the live bar.
            if (EnableAlerts && index > _lastBreakoutIndex)
            {
                DetectBreakoutsAndReclaims(index);
                _lastBreakoutIndex = index;
            }

            // ── Draw on every tick of the last bar ──
            // Mirrors Pine's  if barstate.islast ... (redraws each tick)
            if (IsLastBar)
                DrawAll();
        }

        #endregion

        #region BuildMondayStorage — populate Monday OHLC from weekly + daily MTF bars

        /// <summary>
        /// Iterates all weekly bars, finds the first daily bar in each week,
        /// and stores its OHLC as the "Monday" data.
        /// Mirrors Pine's  new_week + request.security("D") + map.put()
        /// ContainsKey guard ensures each week is stored only once.
        /// </summary>
        private void BuildMondayStorage()
        {
            if (_weeklyBars == null || _dailyBars == null)
                return;
            if (_weeklyBars.Count < 1 || _dailyBars.Count < 1)
                return;

            for (int w = 0; w < _weeklyBars.Count; w++)
            {
                var wkStart = _weeklyBars.OpenTimes[w];
                // Pine: wk_end = time_close of weekly bar (Friday close)
                // cTrader: approximate with next week's open, or +5 days for last bar
                var wkEnd = (w < _weeklyBars.Count - 1)
                    ? _weeklyBars.OpenTimes[w + 1]
                    : wkStart.AddDays(5);

                long wkKey = wkStart.Ticks;

                if (_mondaysMap.ContainsKey(wkKey))
                    continue;

                // Find first daily bar within this week
                int dIndex = FindFirstDailyBarInWeek(wkStart, wkEnd);
                if (dIndex < 0)
                    continue;

                _mondaysMap[wkKey] = new Monday
                {
                    WeekStart = wkStart,
                    WeekEnd   = wkEnd,
                    Open      = _dailyBars.OpenPrices[dIndex],
                    High      = _dailyBars.HighPrices[dIndex],
                    Low       = _dailyBars.LowPrices[dIndex],
                    Close     = _dailyBars.ClosePrices[dIndex]
                };

                // Pine: array.unshift → newest at index 0
                _mondaysOrder.Insert(0, wkKey);

                // Pine: while array.size >= 52 → array.pop (remove oldest from back)
                while (_mondaysOrder.Count > MaxMondaysArraySize)
                {
                    long oldest = _mondaysOrder[_mondaysOrder.Count - 1];
                    _mondaysOrder.RemoveAt(_mondaysOrder.Count - 1);
                    _mondaysMap.Remove(oldest);
                    _highTouched.Remove(oldest);
                    _lowTouched.Remove(oldest);
                }
            }

            TrimOldEvents();
        }

        /// <summary>
        /// Returns the index of the first daily bar whose OpenTime falls within [wkStart, wkEnd).
        /// This bar represents the "Monday" (or first trading day of the week).
        /// </summary>
        private int FindFirstDailyBarInWeek(DateTime wkStart, DateTime wkEnd)
        {
            for (int i = 0; i < _dailyBars.Count; i++)
            {
                var t = _dailyBars.OpenTimes[i];
                if (t >= wkStart && t < wkEnd)
                    return i;
            }
            return -1;
        }

        #endregion

        #region DetectBreakoutsAndReclaims — mirrors Pine alert section

        /// <summary>
        /// Pine: the block under  if enable_alerts and can_show_monday_range and map.contains(...)
        /// Checks crossover/crossunder of close vs Monday high/low,
        /// tracks touched state, detects reclaims.
        /// </summary>
        private void DetectBreakoutsAndReclaims(int index)
        {
            if (index < 1 || _mondaysOrder.Count == 0)
                return;

            var t = Bars.OpenTimes[index];

            // Find which Monday range this bar belongs to
            if (!TryGetWeekMonday(t, out long wkKey, out Monday monday))
                return;

            // Pine: if time >= monday.wkStart and time <= monday.wkEnd
            if (t < monday.WeekStart || t > monday.WeekEnd)
                return;

            // Pine: opening_day_end = monday.wkStart + one_day_ms
            // Only trigger after Monday (opening day) has completed
            var openingDayEnd = monday.WeekStart.AddDays(1);
            if (t < openingDayEnd)
                return;

            double close     = Bars.ClosePrices[index];
            double prevClose = Bars.ClosePrices[index - 1];
            double high      = Bars.HighPrices[index];
            double low       = Bars.LowPrices[index];

            // Pine: if high > monday.h or close > monday.h → map.put(high_touched, ..., true)
            if (high > monday.High || close > monday.High)
                _highTouched[wkKey] = true;

            if (low < monday.Low || close < monday.Low)
                _lowTouched[wkKey] = true;

            // Pine: ta.crossover(close, monday.h)  →  prevClose <= high && close > high
            bool highBreak = prevClose <= monday.High && close > monday.High;
            // Pine: ta.crossunder(close, monday.l) →  prevClose >= low && close < low
            bool lowBreak  = prevClose >= monday.Low  && close < monday.Low;
            bool highReclaim = false;
            bool lowReclaim  = false;

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

            // Pine: high_was_touched = map.get(high_touched, monday.wkStart)
            //       if high_was_touched and close < monday.h → reclaim
            bool hTouched = _highTouched.TryGetValue(wkKey, out bool hv) && hv;
            if (hTouched && close < monday.High)
            {
                _highReclaims.Add(new RangeEvent { BarTime = t, Price = high, WeekKey = wkKey });
                _highTouched[wkKey] = false;
                highReclaim = true;
                Print("Monday High Reclaim | Price closed back below Monday High: {0}", close);
            }

            bool lTouched = _lowTouched.TryGetValue(wkKey, out bool lv) && lv;
            if (lTouched && close > monday.Low)
            {
                _lowReclaims.Add(new RangeEvent { BarTime = t, Price = low, WeekKey = wkKey });
                _lowTouched[wkKey] = false;
                lowReclaim = true;
                Print("Monday Low Reclaim | Price closed back above Monday Low: {0}", close);
            }

            // Combined alert messages (mirrors Pine alertcondition groupings)
            if (highBreak || lowBreak)
                Print("Monday Range Break (High or Low) | Price broke Monday Range");
            if (highReclaim || lowReclaim)
                Print("Monday Range Reclaim (High or Low) | Price closed back into Monday Range");
            if (highBreak || lowBreak || highReclaim || lowReclaim)
                Print("Monday Range Setup (Any) | Monday Range event triggered at {0}", close);
        }

        /// <summary>
        /// Finds which stored Monday range covers the given time.
        /// </summary>
        private bool TryGetWeekMonday(DateTime time, out long wkKey, out Monday monday)
        {
            wkKey  = 0;
            monday = null;

            for (int i = 0; i < _mondaysOrder.Count; i++)
            {
                long key = _mondaysOrder[i];
                if (!_mondaysMap.TryGetValue(key, out Monday m))
                    continue;

                if (time >= m.WeekStart && time <= m.WeekEnd)
                {
                    wkKey  = key;
                    monday = m;
                    return true;
                }
            }
            return false;
        }

        #endregion

        #region DrawAll — mirrors Pine's  if barstate.islast  block

        /// <summary>
        /// Clears previous drawings and redraws all levels, labels, and markers
        /// for the N most recent Mondays.  Runs on every tick of the last bar.
        /// </summary>
        private void DrawAll()
        {
            // Remove all our previous chart objects (stable-prefix approach)
            RemoveObjectsWithPrefix("mr_");

            if (_mondaysOrder.Count == 0)
                return;

            var levelConfigs = BuildCustomConfigs();

            // Pine: end_index = math.min(max_mondays, array.size(mondays_order))
            int endIndex = Math.Min(MaxMondays, _mondaysOrder.Count);

            for (int i = 0; i < endIndex; i++)
            {
                long wkKey = _mondaysOrder[i];
                if (!_mondaysMap.TryGetValue(wkKey, out Monday monday))
                    continue;

                DateTime extEnd = GetExtensionEnd(monday);
                string   prefix = "mr_" + wkKey + "_";

                // ── Monday High ──
                DrawLevel(prefix + "mh", monday.WeekStart, extEnd,
                          UseMh, MhText, monday.High, MhColor, MhWidth, MhStyle);

                // ── Monday Low ──
                DrawLevel(prefix + "ml", monday.WeekStart, extEnd,
                          UseMl, MlText, monday.Low, MlColor, MlWidth, MlStyle);

                // ── Monday Open ──
                DrawLevel(prefix + "mo", monday.WeekStart, extEnd,
                          UseMo, MoText, monday.Open, MoColor, MoWidth, MoStyle);

                // ── Monday Close ──
                DrawLevel(prefix + "mc", monday.WeekStart, extEnd,
                          UseMc, McText, monday.Close, McColor, McWidth, McStyle);

                // ── Custom levels (Pine: for level_config in level_configs → draw_level) ──
                for (int j = 0; j < levelConfigs.Count; j++)
                {
                    var cfg = levelConfigs[j];
                    if (!cfg.Enabled)
                        continue;

                    // Pine: level_price = monday.l + (monday_range * level_config.value)
                    double levelPrice = monday.Low + monday.Range * cfg.Value;
                    DrawLevel(prefix + "hz" + (j + 1), monday.WeekStart, extEnd,
                              true, cfg.Text, levelPrice, cfg.Color, cfg.Width, cfg.LineStyle);
                }

                // ── Breakout markers ──
                if (ShowBreakoutLabels)
                {
                    double offset = monday.Range * 0.05;
                    DrawEvents(_highBreakouts, wkKey,  offset,  HighBreakColor, HighBreakStyle, HighBreakLabelSize, prefix + "hb_");
                    DrawEvents(_lowBreakouts,  wkKey, -offset,  LowBreakColor,  LowBreakStyle,  LowBreakLabelSize,  prefix + "lb_");
                }

                // ── Reclaim markers ──
                if (ShowReclaimLabels)
                {
                    double offset = monday.Range * 0.05;
                    DrawEvents(_highReclaims, wkKey,  offset,  HighReclaimColor, HighReclaimStyle, HighReclaimLabelSize, prefix + "hr_");
                    DrawEvents(_lowReclaims,  wkKey, -offset,  LowReclaimColor,  LowReclaimStyle,  LowReclaimLabelSize,  prefix + "lr_");
                }
            }
        }

        #endregion

        #region Drawing helpers

        /// <summary>
        /// Draws a single horizontal level line + optional text label.
        /// Mirrors Pine's line.new + label.new for MH/ML/MO/MC and custom levels.
        /// </summary>
        private void DrawLevel(string id, DateTime start, DateTime end,
                               bool enabled, string text, double price,
                               Color color, int width, LineStyleOption style)
        {
            if (!enabled)
                return;

            // Pine: line.new(x1=wkStart, x2=extension_end, y1=price, y2=price, ...)
            Chart.DrawTrendLine(id + "_line", start, price, end, price,
                                color, Math.Max(1, width), ToLineStyle(style));

            // Pine: label.new(x=extension_end, y=price, text=..., style=label.style_label_left,
            //                 color=transparent)
            // cTrader has no label_left shape; we use DrawText at the extension end.
            if (!string.IsNullOrEmpty(text))
            {
                var txt = Chart.DrawText(id + "_txt", text, end, price, TextColor);
                txt.FontSize = ToFontSize(LabelSize);
            }
        }

        /// <summary>
        /// Draws breakout/reclaim marker glyphs for events belonging to wkKey.
        /// Pine: label.new(x=event.bar_time, y=event.price ± offset, ...)
        /// </summary>
        private void DrawEvents(List<RangeEvent> events, long wkKey, double offset,
                                Color color, MarkerStyle markerStyle, TvSize markerSize,
                                string idPrefix)
        {
            string glyph = MarkerGlyph(markerStyle);
            int    font  = ToFontSize(markerSize);

            for (int i = 0; i < events.Count; i++)
            {
                var ev = events[i];
                if (ev.WeekKey != wkKey)
                    continue;

                var txt = Chart.DrawText(idPrefix + i, glyph, ev.BarTime, ev.Price + offset, color);
                txt.FontSize = font;
            }
        }

        #endregion

        #region Extension end calculation

        /// <summary>
        /// Pine: get_extension_end(monday, extension_type, fixed_bars)
        ///   end_of_week   → monday.wkEnd
        ///   current_bar   → time  (current bar's time)
        ///   fixed_bars    → monday.wkStart + (fixed_bars * seconds_per_day * 1000)
        /// </summary>
        private DateTime GetExtensionEnd(Monday monday)
        {
            switch (ExtensionType)
            {
                case ExtensionMode.CurrentBar:
                    return Bars.OpenTimes.LastValue;

                case ExtensionMode.FixedBars:
                    // Pine: wkStart + (fixed_bars * timeframe.in_seconds("D") * 1000)
                    return monday.WeekStart.AddDays(FixedBarsCount);

                default: // EndOfWeek
                    return monday.WeekEnd;
            }
        }

        #endregion

        #region Config builders

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

        #endregion

        #region Utility / Mapping helpers

        private bool IsWeeklyOrMonthly()
        {
            return TimeFrame == TimeFrame.Weekly || TimeFrame == TimeFrame.Monthly;
        }

        private static LineStyle ToLineStyle(LineStyleOption style)
        {
            switch (style)
            {
                case LineStyleOption.Dotted: return LineStyle.Dots;
                case LineStyleOption.Dashed: return LineStyle.Lines;
                default:                     return LineStyle.Solid;
            }
        }

        private static string MarkerGlyph(MarkerStyle style)
        {
            switch (style)
            {
                case MarkerStyle.TriangleDown: return "▼";
                case MarkerStyle.TriangleUp:   return "▲";
                case MarkerStyle.LabelDown:    return "↓";
                default:                       return "↑";
            }
        }

        private static int ToFontSize(TvSize size)
        {
            switch (size)
            {
                case TvSize.Tiny:   return 8;
                case TvSize.Small:  return 10;
                case TvSize.Normal: return 12;
                case TvSize.Large:  return 14;
                case TvSize.Huge:   return 16;
                default:            return 10;   // Auto
            }
        }

        /// <summary>
        /// Removes all chart objects whose name starts with the given prefix.
        /// Used to clear previous drawings before redraw.
        /// </summary>
        private void RemoveObjectsWithPrefix(string prefix)
        {
            var names = Chart.Objects
                             .Where(o => o.Name.StartsWith(prefix))
                             .Select(o => o.Name)
                             .ToList();

            foreach (var n in names)
                Chart.RemoveObject(n);
        }

        /// <summary>
        /// Purges breakout/reclaim events belonging to weeks no longer in storage.
        /// </summary>
        private void TrimOldEvents()
        {
            var valid = new HashSet<long>(_mondaysMap.Keys);
            _highBreakouts.RemoveAll(x => !valid.Contains(x.WeekKey));
            _lowBreakouts.RemoveAll(x  => !valid.Contains(x.WeekKey));
            _highReclaims.RemoveAll(x  => !valid.Contains(x.WeekKey));
            _lowReclaims.RemoveAll(x   => !valid.Contains(x.WeekKey));
        }

        #endregion
    }
}
