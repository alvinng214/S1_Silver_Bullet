// =============================================================================
// ICT Sessions_One Setup for Life [MK].cs — Forensic-corrected version
// =============================================================================
// Fixes applied vs original C# port:
//
//  [CRITICAL-1] Session H/L lines: post-session extension + mitigation added.
//               During-session lines always visible (matches Pine).
//               Post-session lines extend right; stop when price crosses them.
//               ShowSessionLines toggle controls ONLY the post-session portion.
//
//  [CRITICAL-2] RTH Gap 4pm close: now finds bar whose CLOSE TIME = 16:15
//               (bar i+1 open == 16:15), matching Pine's ta.valuewhen(time_close==_16_15).
//               Original was finding bar whose OPEN TIME = 16:15 (wrong bar).
//
//  [CRITICAL-3] W/M/Y open detection: uses MarketData.GetBars(TimeFrame.Weekly/Monthly)
//               for actual HTF bar-change detection (handles forex Sunday opens).
//               Yearly derived from monthly bars (find most recent January bar).
//
//  [CRITICAL-4] W/M/Y open prices: uses actual HTF bar open prices from GetBars(),
//               not the current intraday bar's open price.
//
//  [MEDIUM-5]   RTH Gaps and Opening Lines not filtered by pre_range
//               (Pine has //and pre_range and disp_RTHsess commented out).
//
//  [MEDIUM-6]   HideAfterClose (08:30/09:30 lines) implemented:
//               when current NY time is 15:00–20:00, suppress those lines.
//
//  [MEDIUM-7]   Lookback background height uses actual H/L of bars in the
//               lookback window; border colour = fill colour (no visible border).
//
//  [MEDIUM-8]   RTH Gap text "RTH\nGap" now drawn with configurable colour.
//
//  [LOW-9]      00:00 opening line extends 16 hours forward to 16:00 (was 3 h).
//
//  [LOW-10]     W/M/Y lines use LineStyle.Solid (was DotsRare).
//
//  + 23 missing parameters added with correct Pine-matching defaults.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICTSessionsOneSetupForLifeMK : Indicator
    {
        private const string Prefix = "ICTS1_";

        // ── Inner types ───────────────────────────────────────────────────────

        private sealed class SessionConfig
        {
            public string Key;
            public string Label;
            public string Session;
            public bool   Enabled;
            public Color  BorderColor;
            public Color  BoxColor;
            public Color  LineColor;
            public bool   MidAllowed; // false for NY Lunch (matches Pine's uninitialised nylmid_ln)
        }

        // Holds the final computed state of one historical session occurrence.
        private sealed class SessionInstance
        {
            public SessionConfig Config;
            public DateTime Start;        // local time: first bar of session
            public DateTime End;          // local time: close of last bar in session
            public double   High;
            public double   Low;
            public bool     SessionClosed;  // session has ended; post-session scan ran
            // Post-session H/L line endpoints (extends rightward until price crosses)
            public DateTime HighLineEnd;  // equals End during session; grows after
            public DateTime LowLineEnd;
            public bool     HighMitigated;
            public bool     LowMitigated;
            public string   UniqueKey;    // for chart-object naming
        }

        public enum LineStyleChoice { Solid, Dotted, Dashed }

        // ── Parameters — General ──────────────────────────────────────────────

        [Parameter("Timezone", Group = "General", DefaultValue = "GMT+8")]
        public string TimezoneInput { get; set; }

        [Parameter("Max Timeframe (min)", Group = "General", DefaultValue = 15, MinValue = 1, MaxValue = 240)]
        public int MaxTimeframeMinutes { get; set; }

        // ── Parameters — Lookback ─────────────────────────────────────────────

        [Parameter("Previous Days to Show", Group = "Lookback", DefaultValue = 10, MinValue = 1, MaxValue = 365)]
        public int EventDays { get; set; }

        [Parameter("Show Days Background", Group = "Lookback", DefaultValue = true)]
        public bool ShowDaysBackground { get; set; }

        // Pine: color.new(color.silver, 90) = 10% opaque silver (#C0C0C0) → alpha=25=0x19
        [Parameter("Background", Group = "Lookback", DefaultValue = "#19C0C0C0")]
        public Color LookbackBgColor { get; set; }

        // ── Parameters — Sessions (shared) ────────────────────────────────────

        // Pine: 'lines' toggle — controls post-session H/L line visibility only.
        // During-session H/L lines are always drawn regardless.
        [Parameter("Post-Session High/Low Lines", Group = "Sessions", DefaultValue = true)]
        public bool ShowSessionLines { get; set; }

        [Parameter("Session 50% Line", Group = "Sessions", DefaultValue = false)]
        public bool ShowSessionMid { get; set; }

        [Parameter("Border Width", Group = "Sessions", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int SessionBorderWidth { get; set; }

        [Parameter("Show Labels", Group = "Sessions", DefaultValue = true)]
        public bool ShowSessionLabels { get; set; }

        [Parameter("Label Color", Group = "Sessions", DefaultValue = "#FFC0C0C0")]
        public Color SessionLabelColor { get; set; }

        // ── Parameters — Asia ─────────────────────────────────────────────────

        [Parameter("Asia Session", Group = "Asia", DefaultValue = true)]
        public bool AsiaEnabled { get; set; }
        [Parameter("Time", Group = "Asia", DefaultValue = "0800-1600")]
        public string AsiaSession { get; set; }
        [Parameter("Border", Group = "Asia", DefaultValue = "#00FFA500")]
        public Color AsiaBorderColor { get; set; }
        [Parameter("Box", Group = "Asia", DefaultValue = "#33FFA500")]
        public Color AsiaBoxColor { get; set; }
        [Parameter("Line", Group = "Asia", DefaultValue = "#FFFFA500")]
        public Color AsiaLineColor { get; set; }

        // ── Parameters — London ───────────────────────────────────────────────

        [Parameter("London Session", Group = "London", DefaultValue = true)]
        public bool LondonEnabled { get; set; }
        [Parameter("Time", Group = "London", DefaultValue = "1500-1800")]
        public string LondonSession { get; set; }
        [Parameter("Border", Group = "London", DefaultValue = "#00FF0000")]
        public Color LondonBorderColor { get; set; }
        [Parameter("Box", Group = "London", DefaultValue = "#33FF0000")]
        public Color LondonBoxColor { get; set; }
        [Parameter("Line", Group = "London", DefaultValue = "#FFFF0000")]
        public Color LondonLineColor { get; set; }

        // ── Parameters — NY AM ────────────────────────────────────────────────

        [Parameter("New York AM Session", Group = "NY AM", DefaultValue = true)]
        public bool NyAmEnabled { get; set; }
        [Parameter("Time", Group = "NY AM", DefaultValue = "2230-0100")]
        public string NyAmSession { get; set; }
        [Parameter("Border", Group = "NY AM", DefaultValue = "#0000FF00")]
        public Color NyAmBorderColor { get; set; }
        [Parameter("Box", Group = "NY AM", DefaultValue = "#3300FF00")]
        public Color NyAmBoxColor { get; set; }
        [Parameter("Line", Group = "NY AM", DefaultValue = "#FF00FF00")]
        public Color NyAmLineColor { get; set; }

        // ── Parameters — NY Lunch ─────────────────────────────────────────────

        [Parameter("New York Lunch Session", Group = "NY Lunch", DefaultValue = true)]
        public bool NyLunchEnabled { get; set; }
        [Parameter("Time", Group = "NY Lunch", DefaultValue = "0100-0230")]
        public string NyLunchSession { get; set; }
        [Parameter("Border", Group = "NY Lunch", DefaultValue = "#00808080")]
        public Color NyLunchBorderColor { get; set; }
        [Parameter("Box", Group = "NY Lunch", DefaultValue = "#33808080")]
        public Color NyLunchBoxColor { get; set; }
        [Parameter("Line", Group = "NY Lunch", DefaultValue = "#FF808080")]
        public Color NyLunchLineColor { get; set; }

        // ── Parameters — NY PM ────────────────────────────────────────────────

        [Parameter("New York PM Session", Group = "NY PM", DefaultValue = true)]
        public bool NyPmEnabled { get; set; }
        [Parameter("Time", Group = "NY PM", DefaultValue = "0230-0500")]
        public string NyPmSession { get; set; }
        [Parameter("Border", Group = "NY PM", DefaultValue = "#000000FF")]
        public Color NyPmBorderColor { get; set; }
        [Parameter("Box", Group = "NY PM", DefaultValue = "#330000FF")]
        public Color NyPmBoxColor { get; set; }
        [Parameter("Line", Group = "NY PM", DefaultValue = "#FF0000FF")]
        public Color NyPmLineColor { get; set; }

        // ── Parameters — RTH Gaps ─────────────────────────────────────────────

        [Parameter("Show RTH Gap", Group = "RTH Gaps", DefaultValue = true)]
        public bool ShowRthGap { get; set; }

        [Parameter("Boxes To Show", Group = "RTH Gaps", DefaultValue = 3, MinValue = 1, MaxValue = 20)]
        public int BoxesToShow { get; set; }

        [Parameter("Extend Boxes to Current Bar", Group = "RTH Gaps", DefaultValue = false)]
        public bool ExtendGapToNow { get; set; }

        // Pine default: hoursfwd = 1.0
        [Parameter("Project Hours Forward", Group = "RTH Gaps", DefaultValue = 1.0, MinValue = 0.5, MaxValue = 6.5, Step = 0.5)]
        public double GapProjectHours { get; set; }

        // Pine: color.new(color.purple, 75) = 25% opaque purple = #40800080
        [Parameter("Gap Box Color", Group = "RTH Gaps", DefaultValue = "#40800080")]
        public Color GapBoxColor { get; set; }

        // Pine: color.new(color.purple, 0) = opaque purple = #FF800080
        [Parameter("Gap Border Color", Group = "RTH Gaps", DefaultValue = "#FF800080")]
        public Color GapBorderColor { get; set; }

        [Parameter("Gap Border Width", Group = "RTH Gaps", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int GapBorderWidth { get; set; }

        [Parameter("Show Gap Text", Group = "RTH Gaps", DefaultValue = true)]
        public bool ShowGapText { get; set; }

        // Pine: color.new(color.gray, 30) = 70% opaque gray = #B3808080
        [Parameter("Gap Text Color", Group = "RTH Gaps", DefaultValue = "#B3808080")]
        public Color GapTextColor { get; set; }

        // Pine: RThmid_cl = color.new(color.white, 0)
        [Parameter("Midline Color", Group = "RTH Gaps", DefaultValue = "#FFFFFFFF")]
        public Color RthMidColor { get; set; }

        [Parameter("Midline Style", Group = "RTH Gaps", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice RthMidStyle { get; set; }

        [Parameter("Midline Width", Group = "RTH Gaps", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int RthMidWidth { get; set; }

        // Pine: RTh2575_cl = color.new(color.white, 0)
        [Parameter("25/75% Color", Group = "RTH Gaps", DefaultValue = "#FFFFFFFF")]
        public Color Rth2575Color { get; set; }

        [Parameter("25/75% Style", Group = "RTH Gaps", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Rth2575Style { get; set; }

        [Parameter("25/75% Width", Group = "RTH Gaps", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int Rth2575Width { get; set; }

        // ── Parameters — RTH Close ────────────────────────────────────────────

        [Parameter("Show 4pm Close Line", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLine { get; set; }

        [Parameter("Show 4pm Label", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLabel { get; set; }

        [Parameter("Historical Lines", Group = "RTH Close", DefaultValue = 3, MinValue = 1, MaxValue = 20)]
        public int HistoricalRthLines { get; set; }

        [Parameter("Extend Line Right", Group = "RTH Close", DefaultValue = false)]
        public bool FourPmExtendRight { get; set; }

        // Pine: color.new(color.silver, 10) = 90% opaque silver = #E6C0C0C0
        [Parameter("Line Color", Group = "RTH Close", DefaultValue = "#E6C0C0C0")]
        public Color FourPmLineColor { get; set; }

        [Parameter("Line Style", Group = "RTH Close", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice FourPmLineStyle { get; set; }

        [Parameter("Line Width", Group = "RTH Close", DefaultValue = 3, MinValue = 1, MaxValue = 5)]
        public int FourPmLineWidth { get; set; }

        // ── Parameters — Opening Lines ────────────────────────────────────────

        [Parameter("00:00 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0000 { get; set; }

        // Pine: i_linecol1 = color.new(color.blue, 0)
        [Parameter("00:00 Color", Group = "Opening Lines", DefaultValue = "#FF0000FF")]
        public Color Open0000Color { get; set; }

        [Parameter("00:00 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0000Style { get; set; }

        [Parameter("00:00 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0000Width { get; set; }

        [Parameter("08:30 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0830 { get; set; }

        // Pine: i_linecol2 = color.new(color.yellow, 0)
        [Parameter("08:30 Color", Group = "Opening Lines", DefaultValue = "#FFFFFF00")]
        public Color Open0830Color { get; set; }

        [Parameter("08:30 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0830Style { get; set; }

        [Parameter("08:30 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0830Width { get; set; }

        [Parameter("09:30 Open", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowOpen0930 { get; set; }

        // Pine: i_linecol3 = color.new(color.green, 0)
        [Parameter("09:30 Color", Group = "Opening Lines", DefaultValue = "#FF00FF00")]
        public Color Open0930Color { get; set; }

        [Parameter("09:30 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0930Style { get; set; }

        [Parameter("09:30 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0930Width { get; set; }

        [Parameter("Show Historical", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowHistoricalOpens { get; set; }

        // Pine: hidelines — removes 08:30/09:30 lines when local time is 15:00-20:00
        [Parameter("Hide After Market Close", Group = "Opening Lines", DefaultValue = true)]
        public bool HideAfterClose { get; set; }

        // ── Parameters — W/M/Y Open ───────────────────────────────────────────

        [Parameter("Weekly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowWeeklyOpen { get; set; }

        // Pine: color.fuchsia = #FFFF00FF
        [Parameter("Weekly Color", Group = "W/M/Y Open", DefaultValue = "#FFFF00FF")]
        public Color WeeklyColor { get; set; }

        [Parameter("Monthly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowMonthlyOpen { get; set; }

        // Pine: color.new(color.aqua, 0) = #FF00FFFF
        [Parameter("Monthly Color", Group = "W/M/Y Open", DefaultValue = "#FF00FFFF")]
        public Color MonthlyColor { get; set; }

        [Parameter("Yearly", Group = "W/M/Y Open", DefaultValue = false)]
        public bool ShowYearlyOpen { get; set; }

        // Pine: color.white = #FFFFFFFF
        [Parameter("Yearly Color", Group = "W/M/Y Open", DefaultValue = "#FFFFFFFF")]
        public Color YearlyColor { get; set; }

        // Pine: LINE_WIDTH = input.int(1)
        [Parameter("W/M/Y Line Width", Group = "W/M/Y Open", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int WmyLineWidth { get; set; }

        // ── Private fields ─────────────────────────────────────────────────────

        private TimeZoneInfo _tz;
        private Bars         _weeklyBars;
        private Bars         _monthlyBars;
        private double       _avgBarMins;

        // ── Initialize ─────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            _tz = ResolveTimeZone(TimezoneInput);

            // [FIX CRITICAL-3/4] Use actual HTF bars for W/M/Y open prices and detection.
            // Yearly is derived from monthly bars (cTrader has no TimeFrame.Yearly).
            if (ShowWeeklyOpen)
                _weeklyBars = MarketData.GetBars(TimeFrame.Weekly);
            if (ShowMonthlyOpen || ShowYearlyOpen)
                _monthlyBars = MarketData.GetBars(TimeFrame.Monthly);
        }

        // ── Calculate ──────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1)
                return;

            _tz = ResolveTimeZone(TimezoneInput);
            ClearObjects();

            _avgBarMins = GetAvgBarMins();
            bool isIntraday = _avgBarMins > 0 && _avgBarMins < 1440;
            bool dispRth    = isIntraday && _avgBarMins <= MaxTimeframeMinutes;
            if (!dispRth)
                return;

            // Lookback window (local time)
            DateTime lastLocal = ToLocal(Bars.OpenTimes.LastValue);
            DateTime startTime = new DateTime(lastLocal.Year, lastLocal.Month, lastLocal.Day, 23, 59, 0);
            DateTime preTs     = startTime.AddDays(-EventDays);

            if (ShowDaysBackground)
                DrawLookbackBg(preTs, startTime);

            // ── Session scan ──────────────────────────────────────────────────
            // Scan from slightly before preTs to capture sessions whose post-session
            // extension lines start inside preTs, plus sessions spanning the boundary.
            var sessions    = BuildSessions();
            var allInstances = new List<SessionInstance>();
            var currentInst  = new Dictionary<string, SessionInstance>();
            foreach (var cfg in sessions)
                currentInst[cfg.Key] = null;

            // Find bar index for scan start (preTs - 2 days for safety)
            DateTime scanFrom = preTs.AddDays(-2);
            int scanStartBar = 0;
            for (int i = 0; i < Bars.Count; i++)
            {
                if (ToLocal(Bars.OpenTimes[i]) >= scanFrom)
                { scanStartBar = i; break; }
            }

            for (int i = scanStartBar; i < Bars.Count; i++)
            {
                DateTime localOpen  = ToLocal(Bars.OpenTimes[i]);
                DateTime localClose = (i + 1 < Bars.Count)
                    ? ToLocal(Bars.OpenTimes[i + 1])
                    : localOpen.AddMinutes(_avgBarMins);

                double hi = Bars.HighPrices[i];
                double lo = Bars.LowPrices[i];

                foreach (var cfg in sessions)
                {
                    if (!cfg.Enabled) continue;

                    bool inSess   = IsInSession(localOpen, cfg.Session);
                    bool prevSess = i > scanStartBar && IsInSession(ToLocal(Bars.OpenTimes[i - 1]), cfg.Session);
                    bool isNew    = inSess && !prevSess;

                    var inst = currentInst[cfg.Key];

                    if (inSess)
                    {
                        if (isNew)
                        {
                            // Save the previous instance (if any) before starting a new one
                            if (inst != null)
                                allInstances.Add(inst);

                            inst = new SessionInstance
                            {
                                Config       = cfg,
                                Start        = localOpen,
                                End          = localClose,
                                High         = hi,
                                Low          = lo,
                                HighLineEnd  = localClose,
                                LowLineEnd   = localClose,
                                SessionClosed = false,
                                UniqueKey    = cfg.Key + "_" + localOpen.ToString("yyyyMMddHHmm", CultureInfo.InvariantCulture)
                            };
                            currentInst[cfg.Key] = inst;
                        }
                        else if (inst != null)
                        {
                            inst.End         = localClose;
                            inst.High        = Math.Max(inst.High, hi);
                            inst.Low         = Math.Min(inst.Low, lo);
                            inst.HighLineEnd = localClose;
                            inst.LowLineEnd  = localClose;
                        }
                    }
                    else if (inst != null)
                    {
                        // [FIX CRITICAL-1] Post-session: extend H/L lines rightward.
                        // Matches Pine's _highmit/_lowmit which always set x2=time first,
                        // then check for mitigation (so the mitigating bar IS included).
                        inst.SessionClosed = true;

                        if (!inst.HighMitigated)
                        {
                            inst.HighLineEnd = localClose; // extend to include this bar
                            if (hi > inst.High)
                                inst.HighMitigated = true; // stop extending next bar
                        }
                        if (!inst.LowMitigated)
                        {
                            inst.LowLineEnd = localClose;
                            if (lo < inst.Low)
                                inst.LowMitigated = true;
                        }
                    }
                }
            }

            // Flush any still-open session instances
            foreach (var cfg in sessions)
            {
                var inst = currentInst[cfg.Key];
                if (inst != null)
                    allInstances.Add(inst);
            }

            // Draw session instances whose START falls within preRange
            foreach (var inst in allInstances)
            {
                if (inst.Start >= preTs && inst.Start < startTime)
                    DrawSessionInstance(inst);
            }

            // ── [FIX MEDIUM-5] RTH Gaps — ALL history (no pre_range filter) ──
            if (ShowRthGap)
                DrawRthGaps();

            // ── [FIX MEDIUM-5] Opening Lines — ALL history ────────────────────
            DrawOpeningLines();

            // ── [FIX CRITICAL-3/4] W/M/Y Opens from real HTF bars ────────────
            DrawWmyOpens();
        }

        // ── Session drawing ────────────────────────────────────────────────────

        private void DrawSessionInstance(SessionInstance inst)
        {
            var cfg = inst.Config;
            string k = inst.UniqueKey;
            DateTime midTime = inst.Start.AddTicks((inst.End - inst.Start).Ticks / 2);
            double midPrice  = (inst.High + inst.Low) / 2.0;

            // Box: spans session duration only
            var box = Chart.DrawRectangle(Prefix + "box_" + k, inst.Start, inst.High, inst.End, inst.Low,
                cfg.BorderColor, SessionBorderWidth, LineStyle.Solid);
            box.IsFilled = true;
            box.Color    = cfg.BoxColor;

            // [FIX CRITICAL-1] During-session H/L lines: ALWAYS drawn (matches Pine).
            // Pine draws these unconditionally; the 'lines' toggle only hides the
            // post-session extension portion.
            Chart.DrawTrendLine(Prefix + "hs_" + k, inst.Start, inst.High, inst.End, inst.High,
                cfg.LineColor, 1, LineStyle.Solid);
            Chart.DrawTrendLine(Prefix + "ls_" + k, inst.Start, inst.Low,  inst.End, inst.Low,
                cfg.LineColor, 1, LineStyle.Solid);

            // [FIX CRITICAL-1] Post-session extension: controlled by ShowSessionLines.
            // Pine: if not lines → set color transparent (hide). Here we simply omit drawing.
            if (ShowSessionLines && inst.SessionClosed)
            {
                if (inst.HighLineEnd > inst.End)
                    Chart.DrawTrendLine(Prefix + "he_" + k, inst.End, inst.High, inst.HighLineEnd, inst.High,
                        cfg.LineColor, 1, LineStyle.Solid);
                if (inst.LowLineEnd > inst.End)
                    Chart.DrawTrendLine(Prefix + "le_" + k, inst.End, inst.Low, inst.LowLineEnd, inst.Low,
                        cfg.LineColor, 1, LineStyle.Solid);
            }

            // Mid line (during session only; NY Lunch excluded via MidAllowed=false)
            if (ShowSessionMid && cfg.MidAllowed)
                Chart.DrawTrendLine(Prefix + "mid_" + k, inst.Start, midPrice, inst.End, midPrice,
                    cfg.LineColor, 1, LineStyle.Solid);

            if (ShowSessionLabels)
                Chart.DrawText(Prefix + "lbl_" + k, cfg.Label, midTime, inst.High, SessionLabelColor);
        }

        // ── RTH Gap drawing ───────────────────────────────────────────────────

        private void DrawRthGaps()
        {
            // [FIX MEDIUM-5] Scan backwards, collect up to BoxesToShow instances.
            // Pine draws for all history (no lookback filter) but limits array to BoxesToShow.
            var gaps = new List<(int BarIdx, DateTime LocalOpen, double FourPmClose, double CurOpen)>();

            for (int i = Bars.Count - 1; i >= 1 && gaps.Count < BoxesToShow; i--)
            {
                DateTime localOpen = ToLocal(Bars.OpenTimes[i]);
                if (!IsTime(localOpen, 9, 30))
                    continue;

                double? fourPmClose = FindCloseAt1615(i);
                if (!fourPmClose.HasValue)
                    continue;

                gaps.Insert(0, (i, localOpen, fourPmClose.Value, Bars.OpenPrices[i]));
            }

            int closeLineIdx = 0;
            foreach (var (bi, localOpen, fourPmClose, curOpen) in gaps)
                DrawRthGapInstance(bi, localOpen, fourPmClose, curOpen, closeLineIdx++);
        }

        // [FIX CRITICAL-2] Find bar whose CLOSE TIME = 16:15 local.
        // Pine: ta.valuewhen(time_close == _16_15, close, 0)
        // In cTrader, bar i closes at time = Bars.OpenTimes[i+1].
        private double? FindCloseAt1615(int fromBar)
        {
            int maxLook = (int)(1440.0 / _avgBarMins) + 10; // ~1 trading day
            for (int i = fromBar - 1; i >= Math.Max(0, fromBar - maxLook); i--)
            {
                DateTime closeTime = (i + 1 < Bars.Count)
                    ? ToLocal(Bars.OpenTimes[i + 1])
                    : ToLocal(Bars.OpenTimes[i]).AddMinutes(_avgBarMins);

                if (closeTime.Hour == 16 && closeTime.Minute == 15)
                    return Bars.ClosePrices[i];
            }
            return null;
        }

        private void DrawRthGapInstance(int barIndex, DateTime localOpen,
            double fourPmClose, double curOpen, int closeLineIndex)
        {
            string suffix = barIndex.ToString(CultureInfo.InvariantCulture);
            DateTime right = ExtendGapToNow
                ? ToLocal(Bars.OpenTimes.LastValue)
                : localOpen.AddHours(GapProjectHours);

            double top    = Math.Max(fourPmClose, curOpen);
            double bottom = Math.Min(fourPmClose, curOpen);

            // Gap box
            var rect = Chart.DrawRectangle(Prefix + "rthgap_" + suffix,
                localOpen, top, right, bottom, GapBorderColor, GapBorderWidth, LineStyle.Solid);
            rect.IsFilled = true;
            rect.Color    = GapBoxColor;

            // [FIX MEDIUM-8] Gap text (Pine: text='RTH\nGap')
            if (ShowGapText)
                Chart.DrawText(Prefix + "rthgtxt_" + suffix, "RTH Gap", right, (top + bottom) / 2.0, GapTextColor);

            // Internal levels: mid, 75%, 25%
            double mid = (top + bottom) / 2.0;
            double q75 = (mid + top)    / 2.0;
            double q25 = (mid + bottom) / 2.0;

            Chart.DrawTrendLine(Prefix + "rthmid_" + suffix, localOpen, mid, right, mid,
                RthMidColor, RthMidWidth, ToLineStyle(RthMidStyle));
            Chart.DrawTrendLine(Prefix + "rth75_" + suffix, localOpen, q75, right, q75,
                Rth2575Color, Rth2575Width, ToLineStyle(Rth2575Style));
            Chart.DrawTrendLine(Prefix + "rth25_" + suffix, localOpen, q25, right, q25,
                Rth2575Color, Rth2575Width, ToLineStyle(Rth2575Style));

            // 4pm close line
            if (Show4PmLine && closeLineIndex < HistoricalRthLines)
            {
                DateTime lineEnd = FourPmExtendRight
                    ? ToLocal(Bars.OpenTimes.LastValue).AddHours(24)
                    : right;
                Chart.DrawTrendLine(Prefix + "rth4pm_" + suffix, localOpen, fourPmClose, lineEnd, fourPmClose,
                    FourPmLineColor, FourPmLineWidth, ToLineStyle(FourPmLineStyle));
                if (Show4PmLabel)
                    Chart.DrawText(Prefix + "rth4pml_" + suffix,
                        fourPmClose.ToString("F2", CultureInfo.InvariantCulture),
                        lineEnd, fourPmClose, FourPmLineColor);
            }
        }

        // ── Opening Lines ─────────────────────────────────────────────────────

        private void DrawOpeningLines()
        {
            // [FIX MEDIUM-6] HideAfterClose: if current local time is 15:00–20:00,
            // suppress 08:30 and 09:30 lines (matching Pine's InSession('1500-2000') delete).
            DateTime currentLocal  = ToLocal(Bars.OpenTimes.LastValue);
            bool     isAfterClose  = IsInSession(currentLocal, "1500-2000");

            // Collect all occurrences of each opening time, scanning backwards.
            // When ShowHistoricalOpens = false, only the most recent is drawn
            // (matches Pine's `if not history: line.delete(lne[1])`).
            var found0000 = new List<(int idx, DateTime localOpen)>();
            var found0830 = new List<(int idx, DateTime localOpen)>();
            var found0930 = new List<(int idx, DateTime localOpen)>();

            for (int i = Bars.Count - 1; i >= 0; i--)
            {
                DateTime lo = ToLocal(Bars.OpenTimes[i]);

                if (ShowOpen0000 && IsTime(lo, 0,  0))  found0000.Insert(0, (i, lo));
                if (ShowOpen0830 && IsTime(lo, 8, 30)) found0830.Insert(0, (i, lo));
                if (ShowOpen0930 && IsTime(lo, 9, 30)) found0930.Insert(0, (i, lo));

                // For non-historical mode, stop once we have one of each
                if (!ShowHistoricalOpens
                    && (!ShowOpen0000 || found0000.Count > 0)
                    && (!ShowOpen0830 || found0830.Count > 0)
                    && (!ShowOpen0930 || found0930.Count > 0))
                    break;
            }

            // When not showing history, keep only the most recent entry
            void TrimToOne(List<(int, DateTime)> lst)
            {
                if (!ShowHistoricalOpens && lst.Count > 1)
                    lst.RemoveRange(0, lst.Count - 1);
            }
            TrimToOne(found0000); TrimToOne(found0830); TrimToOne(found0930);

            // [FIX LOW-9] 00:00 line extends 16 hours (to ~16:00 same day).
            // Pine: htime = 57600000ms = 16 hours.
            foreach (var (idx, lo) in found0000)
                DrawSingleOpenLine(idx, lo, 16.0, Open0000Color, Open0000Width, Open0000Style, false, isAfterClose, currentLocal);

            // 08:30 line extends 3.5 hours (to 12:00). Pine: +12600000ms.
            foreach (var (idx, lo) in found0830)
                DrawSingleOpenLine(idx, lo, 3.5, Open0830Color, Open0830Width, Open0830Style, true, isAfterClose, currentLocal);

            // 09:30 line extends 2.5 hours (to 12:00). Pine: +9000000ms.
            foreach (var (idx, lo) in found0930)
                DrawSingleOpenLine(idx, lo, 2.5, Open0930Color, Open0930Width, Open0930Style, true, isAfterClose, currentLocal);
        }

        private void DrawSingleOpenLine(int barIdx, DateTime localOpen, double fwdHours,
            Color color, int width, LineStyleChoice style,
            bool applyHideAfterClose, bool isAfterClose, DateTime currentLocal)
        {
            // [FIX MEDIUM-6] Suppress same-day 08:30/09:30 lines when market is closed
            if (applyHideAfterClose && HideAfterClose && isAfterClose
                && localOpen.Date == currentLocal.Date)
                return;

            double price = Bars.OpenPrices[barIdx];
            DateTime right = localOpen.AddHours(fwdHours);
            string id = Prefix + "open_" + barIdx.ToString(CultureInfo.InvariantCulture);
            Chart.DrawTrendLine(id, localOpen, price, right, price, color, width, ToLineStyle(style));
        }

        // ── W/M/Y Opens ───────────────────────────────────────────────────────

        // [FIX CRITICAL-3/4] Uses real HTF bars from MarketData.GetBars() for both
        // open-price and new-period detection. Pine: request.security('W', open, lookahead=on).
        private void DrawWmyOpens()
        {
            // Weekly: most recent weekly bar (lookback=1 in Pine)
            if (ShowWeeklyOpen && _weeklyBars != null && _weeklyBars.Count > 0)
            {
                int w = _weeklyBars.Count - 1;
                DrawLevel("w_0", "W Open",
                    _weeklyBars.OpenPrices[w],
                    ToLocal(_weeklyBars.OpenTimes[w]),
                    WeeklyColor);
            }

            // Monthly: most recent monthly bar
            if (ShowMonthlyOpen && _monthlyBars != null && _monthlyBars.Count > 0)
            {
                int m = _monthlyBars.Count - 1;
                DrawLevel("m_0", "M Open",
                    _monthlyBars.OpenPrices[m],
                    ToLocal(_monthlyBars.OpenTimes[m]),
                    MonthlyColor);
            }

            // Yearly: most recent January (= first month of the current or previous year).
            // Derived from monthly bars since cTrader has no TimeFrame.Yearly.
            if (ShowYearlyOpen && _monthlyBars != null)
            {
                for (int m = _monthlyBars.Count - 1; m >= 0; m--)
                {
                    DateTime mLocal = ToLocal(_monthlyBars.OpenTimes[m]);
                    if (mLocal.Month == 1) // January = start of year
                    {
                        DrawLevel("y_0", "Y Open",
                            _monthlyBars.OpenPrices[m],
                            mLocal,
                            YearlyColor);
                        break;
                    }
                }
            }
        }

        // [FIX LOW-10] LineStyle.Solid (was DotsRare). Pine constant: LINE_STYLE = line.style_solid.
        private void DrawLevel(string suffix, string text, double price, DateTime start, Color color)
        {
            // Extend well past last bar (simulates Pine's extend=extend.right)
            DateTime right = ToLocal(Bars.OpenTimes.LastValue).AddHours(48);
            Chart.DrawTrendLine(Prefix + "lvl_" + suffix, start, price, right, price,
                color, WmyLineWidth, LineStyle.Solid);
            Chart.DrawText(Prefix + "lvltxt_" + suffix, text, right, price, color);
        }

        // ── Lookback background ───────────────────────────────────────────────

        // [FIX MEDIUM-7] Uses H/L of bars within the lookback window only.
        // Border colour = fill colour (no visible border, approximates Pine's bgcolor()).
        private void DrawLookbackBg(DateTime preTs, DateTime startTime)
        {
            double bgTop    = double.MinValue;
            double bgBottom = double.MaxValue;
            for (int i = 0; i < Bars.Count; i++)
            {
                DateTime lo = ToLocal(Bars.OpenTimes[i]);
                if (lo < preTs || lo >= startTime) continue;
                bgTop    = Math.Max(bgTop,    Bars.HighPrices[i]);
                bgBottom = Math.Min(bgBottom, Bars.LowPrices[i]);
            }
            if (bgTop == double.MinValue) return;

            double pad = (bgTop - bgBottom) * 0.1;
            bgTop    += pad;
            bgBottom -= pad;

            var bg = Chart.DrawRectangle(Prefix + "lookback", preTs, bgTop, startTime, bgBottom,
                LookbackBgColor, 1, LineStyle.Solid);
            bg.IsFilled = true;
            bg.Color    = LookbackBgColor; // border same as fill → effectively no visible border
        }

        // ── Session configuration ─────────────────────────────────────────────

        private List<SessionConfig> BuildSessions()
        {
            return new List<SessionConfig>
            {
                new SessionConfig { Key="asia",   Label="Asia",      Session=AsiaSession,    Enabled=AsiaEnabled,    BorderColor=AsiaBorderColor,    BoxColor=AsiaBoxColor,    LineColor=AsiaLineColor,    MidAllowed=true  },
                new SessionConfig { Key="london", Label="London",    Session=LondonSession,  Enabled=LondonEnabled,  BorderColor=LondonBorderColor,  BoxColor=LondonBoxColor,  LineColor=LondonLineColor,  MidAllowed=true  },
                new SessionConfig { Key="nyam",   Label="NY AM",     Session=NyAmSession,    Enabled=NyAmEnabled,    BorderColor=NyAmBorderColor,    BoxColor=NyAmBoxColor,    LineColor=NyAmLineColor,    MidAllowed=true  },
                // [Pine] nylmid_ln is declared but never initialised → NYLunch has no mid line
                new SessionConfig { Key="nyl",    Label="NY Lunch",  Session=NyLunchSession, Enabled=NyLunchEnabled, BorderColor=NyLunchBorderColor, BoxColor=NyLunchBoxColor, LineColor=NyLunchLineColor, MidAllowed=false },
                new SessionConfig { Key="nypm",   Label="NY PM",     Session=NyPmSession,    Enabled=NyPmEnabled,    BorderColor=NyPmBorderColor,    BoxColor=NyPmBoxColor,    LineColor=NyPmLineColor,    MidAllowed=true  },
            };
        }

        // ── Utilities ─────────────────────────────────────────────────────────

        /// <summary>
        /// True if localTime falls inside a Pine-style session string (e.g. "0930-1200").
        /// Handles cross-midnight sessions (e.g. "2000-0000") by wraparound logic.
        /// </summary>
        private static bool IsInSession(DateTime localTime, string session)
        {
            if (string.IsNullOrWhiteSpace(session) || session.Length < 9)
                return false;

            int sh = int.Parse(session.Substring(0, 2), CultureInfo.InvariantCulture);
            int sm = int.Parse(session.Substring(2, 2), CultureInfo.InvariantCulture);
            int eh = int.Parse(session.Substring(5, 2), CultureInfo.InvariantCulture);
            int em = int.Parse(session.Substring(7, 2), CultureInfo.InvariantCulture);

            int current = localTime.Hour * 60 + localTime.Minute;
            int start   = sh * 60 + sm;
            int end     = eh * 60 + em;

            if (start == end)  return true;
            if (start < end)   return current >= start && current < end;
            return current >= start || current < end; // cross-midnight
        }

        private static bool IsTime(DateTime dt, int hour, int minute)
            => dt.Hour == hour && dt.Minute == minute;

        private DateTime ToLocal(DateTime utc)
            => TimeZoneInfo.ConvertTimeFromUtc(
                   DateTime.SpecifyKind(utc, DateTimeKind.Utc), _tz);

        private double GetAvgBarMins()
        {
            if (Bars.Count < 2) return 1.0;
            int sample = Math.Min(200, Bars.Count - 1);
            double sum = 0;
            for (int i = Bars.Count - sample; i < Bars.Count; i++)
                sum += (Bars.OpenTimes[i] - Bars.OpenTimes[i - 1]).TotalMinutes;
            return sum / sample;
        }

        private static LineStyle ToLineStyle(LineStyleChoice c)
        {
            switch (c)
            {
                case LineStyleChoice.Dotted: return LineStyle.Dots;
                case LineStyleChoice.Dashed: return LineStyle.Lines;
                default:                    return LineStyle.Solid;
            }
        }

        private TimeZoneInfo ResolveTimeZone(string tz)
        {
            if (string.Equals(tz, "America/New_York", StringComparison.OrdinalIgnoreCase))
            {
                try { return TimeZoneInfo.FindSystemTimeZoneById("America/New_York"); }
                catch { return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }
            }

            if (tz.StartsWith("GMT", StringComparison.OrdinalIgnoreCase))
            {
                string offset = tz.Substring(3);
                if (offset.StartsWith("+")) offset = offset.Substring(1);
                if (double.TryParse(offset,
                    NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture, out double hours))
                    return TimeZoneInfo.CreateCustomTimeZone(
                        "Custom" + tz, TimeSpan.FromHours(hours), tz, tz);
            }

            try { return TimeZoneInfo.FindSystemTimeZoneById(tz); }
            catch { return TimeZoneInfo.Utc; }
        }

        private void ClearObjects()
        {
            var toRemove = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    toRemove.Add(obj.Name);
            foreach (var name in toRemove)
                Chart.RemoveObject(name);
        }
    }
}
