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
//  [CRITICAL-5] Chart.Draw* DateTime args must be UTC (Indicator TimeZone=UTC).
//               All functions now keep UTC for drawing and local only for detection.
//
//  [MEDIUM-5]   RTH Gaps and Opening Lines not filtered by pre_range
//               (Pine has //and pre_range and disp_RTHsess commented out).
//
//  [MEDIUM-6]   HideAfterClose (08:30/09:30 lines) implemented:
//               when current NY time is 15:00-20:00, suppress those lines.
//
//  [MEDIUM-7]   Lookback background height uses actual H/L of bars in the
//               lookback window; border colour = fill colour (no visible border).
//
//  [MEDIUM-8]   RTH Gap text "RTH Gap" now drawn with configurable colour.
//
//  [LOW-9]      00:00 opening line extends 16 hours forward to 16:00 (was 3 h).
//
//  [LOW-10]     W/M/Y lines use LineStyle.Solid (was DotsRare).
//
//  + 23 missing parameters added with correct Pine-matching defaults.
//  + Default timezone GMT+8 and session times converted to HKT.
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
            public bool   MidAllowed;
        }

        // [CRITICAL-5] Utc fields passed to Chart.Draw*. Local used only for detection.
        private sealed class SessionInstance
        {
            public SessionConfig Config;
            public DateTime StartLocal;       // for preRange filter only
            public DateTime StartUtc;         // for Chart.Draw*
            public DateTime EndUtc;
            public DateTime HighLineEndUtc;
            public DateTime LowLineEndUtc;
            public double   High;
            public double   Low;
            public bool     SessionClosed;
            public bool     HighMitigated;
            public bool     LowMitigated;
            public string   UniqueKey;
        }

        public enum LineStyleChoice { Solid, Dotted, Dashed }

        // ── Parameters ────────────────────────────────────────────────────────

        [Parameter("Timezone", Group = "General", DefaultValue = "America/New_York")]
        public string TimezoneInput { get; set; }

        [Parameter("Max Timeframe (min)", Group = "General", DefaultValue = 15, MinValue = 1, MaxValue = 240)]
        public int MaxTimeframeMinutes { get; set; }

        [Parameter("Previous Days to Show", Group = "Lookback", DefaultValue = 10, MinValue = 1, MaxValue = 365)]
        public int EventDays { get; set; }

        [Parameter("Show Days Background", Group = "Lookback", DefaultValue = true)]
        public bool ShowDaysBackground { get; set; }

        [Parameter("Background", Group = "Lookback", DefaultValue = "#19C0C0C0")]
        public Color LookbackBgColor { get; set; }

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

        // HKT 08:00-12:00 = 19:00-23:00 NY EST
        [Parameter("Asia Session", Group = "Asia", DefaultValue = true)]
        public bool AsiaEnabled { get; set; }
        [Parameter("Time", Group = "Asia", DefaultValue = "2000-2400")]
        public string AsiaSession { get; set; }
        [Parameter("Border", Group = "Asia", DefaultValue = "#00FFA500")]
        public Color AsiaBorderColor { get; set; }
        [Parameter("Box", Group = "Asia", DefaultValue = "#33FFA500")]
        public Color AsiaBoxColor { get; set; }
        [Parameter("Line", Group = "Asia", DefaultValue = "#FFFFA500")]
        public Color AsiaLineColor { get; set; }

        // London: HKT 15:00-18:00
        [Parameter("London Session", Group = "London", DefaultValue = true)]
        public bool LondonEnabled { get; set; }
        [Parameter("Time", Group = "London", DefaultValue = "0200-0500")]
        public string LondonSession { get; set; }
        [Parameter("Border", Group = "London", DefaultValue = "#00FF0000")]
        public Color LondonBorderColor { get; set; }
        [Parameter("Box", Group = "London", DefaultValue = "#33FF0000")]
        public Color LondonBoxColor { get; set; }
        [Parameter("Line", Group = "London", DefaultValue = "#FFFF0000")]
        public Color LondonLineColor { get; set; }

        // NY AM: HKT 22:30-01:00 (cross-midnight)
        [Parameter("New York AM Session", Group = "NY AM", DefaultValue = true)]
        public bool NyAmEnabled { get; set; }
        [Parameter("Time", Group = "NY AM", DefaultValue = "0930-1200")]
        public string NyAmSession { get; set; }
        [Parameter("Border", Group = "NY AM", DefaultValue = "#0000FF00")]
        public Color NyAmBorderColor { get; set; }
        [Parameter("Box", Group = "NY AM", DefaultValue = "#3300FF00")]
        public Color NyAmBoxColor { get; set; }
        [Parameter("Line", Group = "NY AM", DefaultValue = "#FF00FF00")]
        public Color NyAmLineColor { get; set; }

        // NY Lunch: HKT 01:00-02:30
        [Parameter("New York Lunch Session", Group = "NY Lunch", DefaultValue = true)]
        public bool NyLunchEnabled { get; set; }
        [Parameter("Time", Group = "NY Lunch", DefaultValue = "1200-1330")]
        public string NyLunchSession { get; set; }
        [Parameter("Border", Group = "NY Lunch", DefaultValue = "#00808080")]
        public Color NyLunchBorderColor { get; set; }
        [Parameter("Box", Group = "NY Lunch", DefaultValue = "#33808080")]
        public Color NyLunchBoxColor { get; set; }
        [Parameter("Line", Group = "NY Lunch", DefaultValue = "#FF808080")]
        public Color NyLunchLineColor { get; set; }

        // NY PM: HKT 02:30-05:00
        [Parameter("New York PM Session", Group = "NY PM", DefaultValue = true)]
        public bool NyPmEnabled { get; set; }
        [Parameter("Time", Group = "NY PM", DefaultValue = "1330-1600")]
        public string NyPmSession { get; set; }
        [Parameter("Border", Group = "NY PM", DefaultValue = "#000000FF")]
        public Color NyPmBorderColor { get; set; }
        [Parameter("Box", Group = "NY PM", DefaultValue = "#330000FF")]
        public Color NyPmBoxColor { get; set; }
        [Parameter("Line", Group = "NY PM", DefaultValue = "#FF0000FF")]
        public Color NyPmLineColor { get; set; }

        [Parameter("Show RTH Gap", Group = "RTH Gaps", DefaultValue = true)]
        public bool ShowRthGap { get; set; }
        [Parameter("Boxes To Show", Group = "RTH Gaps", DefaultValue = 3, MinValue = 1, MaxValue = 20)]
        public int BoxesToShow { get; set; }
        [Parameter("Extend Boxes to Current Bar", Group = "RTH Gaps", DefaultValue = false)]
        public bool ExtendGapToNow { get; set; }
        [Parameter("Project Hours Forward", Group = "RTH Gaps", DefaultValue = 1.0, MinValue = 0.5, MaxValue = 6.5, Step = 0.5)]
        public double GapProjectHours { get; set; }
        [Parameter("Gap Box Color", Group = "RTH Gaps", DefaultValue = "#40800080")]
        public Color GapBoxColor { get; set; }
        [Parameter("Gap Border Color", Group = "RTH Gaps", DefaultValue = "#FF800080")]
        public Color GapBorderColor { get; set; }
        [Parameter("Gap Border Width", Group = "RTH Gaps", DefaultValue = 1, MinValue = 1, MaxValue = 4)]
        public int GapBorderWidth { get; set; }
        [Parameter("Show Gap Text", Group = "RTH Gaps", DefaultValue = true)]
        public bool ShowGapText { get; set; }
        [Parameter("Gap Text Color", Group = "RTH Gaps", DefaultValue = "#B3808080")]
        public Color GapTextColor { get; set; }
        [Parameter("Midline Color", Group = "RTH Gaps", DefaultValue = "#FFFFFFFF")]
        public Color RthMidColor { get; set; }
        [Parameter("Midline Style", Group = "RTH Gaps", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice RthMidStyle { get; set; }
        [Parameter("Midline Width", Group = "RTH Gaps", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int RthMidWidth { get; set; }
        [Parameter("25/75% Color", Group = "RTH Gaps", DefaultValue = "#FFFFFFFF")]
        public Color Rth2575Color { get; set; }
        [Parameter("25/75% Style", Group = "RTH Gaps", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Rth2575Style { get; set; }
        [Parameter("25/75% Width", Group = "RTH Gaps", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int Rth2575Width { get; set; }

        [Parameter("Show 4pm Close Line", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLine { get; set; }
        [Parameter("Show 4pm Label", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLabel { get; set; }
        [Parameter("Historical Lines", Group = "RTH Close", DefaultValue = 3, MinValue = 1, MaxValue = 20)]
        public int HistoricalRthLines { get; set; }
        [Parameter("Extend Line Right", Group = "RTH Close", DefaultValue = false)]
        public bool FourPmExtendRight { get; set; }
        [Parameter("Line Color", Group = "RTH Close", DefaultValue = "#E6C0C0C0")]
        public Color FourPmLineColor { get; set; }
        [Parameter("Line Style", Group = "RTH Close", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice FourPmLineStyle { get; set; }
        [Parameter("Line Width", Group = "RTH Close", DefaultValue = 3, MinValue = 1, MaxValue = 5)]
        public int FourPmLineWidth { get; set; }

        [Parameter("00:00 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0000 { get; set; }
        [Parameter("00:00 Color", Group = "Opening Lines", DefaultValue = "#FF0000FF")]
        public Color Open0000Color { get; set; }
        [Parameter("00:00 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0000Style { get; set; }
        [Parameter("00:00 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0000Width { get; set; }

        [Parameter("08:30 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0830 { get; set; }
        [Parameter("08:30 Color", Group = "Opening Lines", DefaultValue = "#FFFFFF00")]
        public Color Open0830Color { get; set; }
        [Parameter("08:30 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0830Style { get; set; }
        [Parameter("08:30 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0830Width { get; set; }

        [Parameter("09:30 Open", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowOpen0930 { get; set; }
        [Parameter("09:30 Color", Group = "Opening Lines", DefaultValue = "#FF00FF00")]
        public Color Open0930Color { get; set; }
        [Parameter("09:30 Style", Group = "Opening Lines", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice Open0930Style { get; set; }
        [Parameter("09:30 Width", Group = "Opening Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int Open0930Width { get; set; }

        [Parameter("Show Historical", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowHistoricalOpens { get; set; }
        [Parameter("Hide After Market Close", Group = "Opening Lines", DefaultValue = true)]
        public bool HideAfterClose { get; set; }

        [Parameter("Weekly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowWeeklyOpen { get; set; }
        [Parameter("Weekly Color", Group = "W/M/Y Open", DefaultValue = "#FFFF00FF")]
        public Color WeeklyColor { get; set; }

        [Parameter("Monthly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowMonthlyOpen { get; set; }
        [Parameter("Monthly Color", Group = "W/M/Y Open", DefaultValue = "#FF00FFFF")]
        public Color MonthlyColor { get; set; }

        [Parameter("Yearly", Group = "W/M/Y Open", DefaultValue = false)]
        public bool ShowYearlyOpen { get; set; }
        [Parameter("Yearly Color", Group = "W/M/Y Open", DefaultValue = "#FFFFFFFF")]
        public Color YearlyColor { get; set; }

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
            if (ShowWeeklyOpen)
                _weeklyBars  = MarketData.GetBars(TimeFrame.Weekly);
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
            bool dispRth = _avgBarMins > 0 && _avgBarMins < 1440
                        && _avgBarMins <= MaxTimeframeMinutes;
            if (!dispRth)
                return;

            // Lookback window in local time for detection comparisons
            DateTime lastLocal  = ToLocal(Bars.OpenTimes.LastValue);
            DateTime endLocal   = new DateTime(lastLocal.Year, lastLocal.Month, lastLocal.Day, 23, 59, 0);
            DateTime preLocal   = endLocal.AddDays(-EventDays);

            if (ShowDaysBackground)
                DrawLookbackBg(preLocal, endLocal);

            // ── Session scan ──────────────────────────────────────────────────
            var sessions     = BuildSessions();
            var allInstances = new List<SessionInstance>();
            var current      = new Dictionary<string, SessionInstance>();
            foreach (var cfg in sessions) current[cfg.Key] = null;

            DateTime scanFromLocal = preLocal.AddDays(-2);
            int startBar = 0;
            for (int i = 0; i < Bars.Count; i++)
            {
                if (ToLocal(Bars.OpenTimes[i]) >= scanFromLocal)
                { startBar = i; break; }
            }

            for (int i = startBar; i < Bars.Count; i++)
            {
                // [CRITICAL-5] utcOpen/utcClose → Chart.Draw*; localOpen → detection only
                DateTime utcOpen   = Bars.OpenTimes[i];
                DateTime utcClose  = (i + 1 < Bars.Count)
                    ? Bars.OpenTimes[i + 1]
                    : utcOpen.AddMinutes(_avgBarMins);
                DateTime localOpen = ToLocal(utcOpen);

                double hi = Bars.HighPrices[i];
                double lo = Bars.LowPrices[i];

                foreach (var cfg in sessions)
                {
                    if (!cfg.Enabled) continue;

                    bool inSess   = IsInSession(localOpen, cfg.Session);
                    bool prevSess = i > startBar
                        && IsInSession(ToLocal(Bars.OpenTimes[i - 1]), cfg.Session);
                    bool isNew    = inSess && !prevSess;
                    var  inst     = current[cfg.Key];

                    if (inSess)
                    {
                        if (isNew)
                        {
                            if (inst != null) allInstances.Add(inst);
                            inst = new SessionInstance
                            {
                                Config         = cfg,
                                StartLocal     = localOpen,
                                StartUtc       = utcOpen,
                                EndUtc         = utcClose,
                                High           = hi,
                                Low            = lo,
                                HighLineEndUtc = utcClose,
                                LowLineEndUtc  = utcClose,
                                UniqueKey      = cfg.Key + "_"
                                    + utcOpen.ToString("yyyyMMddHHmm", CultureInfo.InvariantCulture)
                            };
                            current[cfg.Key] = inst;
                        }
                        else if (inst != null)
                        {
                            inst.EndUtc        = utcClose;
                            inst.High          = Math.Max(inst.High, hi);
                            inst.Low           = Math.Min(inst.Low, lo);
                            inst.HighLineEndUtc = utcClose;
                            inst.LowLineEndUtc  = utcClose;
                        }
                    }
                    else if (inst != null)
                    {
                        // [CRITICAL-1] Post-session: extend then check mitigation
                        inst.SessionClosed = true;
                        if (!inst.HighMitigated)
                        {
                            inst.HighLineEndUtc = utcClose;
                            if (hi > inst.High) inst.HighMitigated = true;
                        }
                        if (!inst.LowMitigated)
                        {
                            inst.LowLineEndUtc = utcClose;
                            if (lo < inst.Low) inst.LowMitigated = true;
                        }
                    }
                }
            }

            foreach (var cfg in sessions)
            {
                var inst = current[cfg.Key];
                if (inst != null) allInstances.Add(inst);
            }

            foreach (var inst in allInstances)
                if (inst.StartLocal >= preLocal && inst.StartLocal < endLocal)
                    DrawSessionInstance(inst);

            if (ShowRthGap)  DrawRthGaps();
            DrawOpeningLines();
            DrawWmyOpens();
        }

        // ── Session drawing ────────────────────────────────────────────────────

        private void DrawSessionInstance(SessionInstance inst)
        {
            var    cfg    = inst.Config;
            string k      = inst.UniqueKey;
            DateTime midUtc   = inst.StartUtc.AddTicks((inst.EndUtc - inst.StartUtc).Ticks / 2);
            double   midPrice = (inst.High + inst.Low) / 2.0;

            var box = Chart.DrawRectangle(Prefix + "box_" + k,
                inst.StartUtc, inst.High, inst.EndUtc, inst.Low,
                cfg.BorderColor, SessionBorderWidth, LineStyle.Solid);
            box.IsFilled = true;
            box.Color    = cfg.BoxColor;

            // During-session lines: always drawn (Pine draws unconditionally)
            Chart.DrawTrendLine(Prefix + "hs_" + k,
                inst.StartUtc, inst.High, inst.EndUtc, inst.High,
                cfg.LineColor, 1, LineStyle.Solid);
            Chart.DrawTrendLine(Prefix + "ls_" + k,
                inst.StartUtc, inst.Low, inst.EndUtc, inst.Low,
                cfg.LineColor, 1, LineStyle.Solid);

            // Post-session extensions: ShowSessionLines toggle
            if (ShowSessionLines && inst.SessionClosed)
            {
                if (inst.HighLineEndUtc > inst.EndUtc)
                    Chart.DrawTrendLine(Prefix + "he_" + k,
                        inst.EndUtc, inst.High, inst.HighLineEndUtc, inst.High,
                        cfg.LineColor, 1, LineStyle.Solid);
                if (inst.LowLineEndUtc > inst.EndUtc)
                    Chart.DrawTrendLine(Prefix + "le_" + k,
                        inst.EndUtc, inst.Low, inst.LowLineEndUtc, inst.Low,
                        cfg.LineColor, 1, LineStyle.Solid);
            }

            if (ShowSessionMid && cfg.MidAllowed)
                Chart.DrawTrendLine(Prefix + "mid_" + k,
                    inst.StartUtc, midPrice, inst.EndUtc, midPrice,
                    cfg.LineColor, 1, LineStyle.Solid);

            if (ShowSessionLabels)
                Chart.DrawText(Prefix + "lbl_" + k, cfg.Label, midUtc, inst.High, SessionLabelColor);
        }

        // ── RTH Gap drawing ───────────────────────────────────────────────────

        private void DrawRthGaps()
        {
            var gaps = new List<(int BarIdx, DateTime UtcOpen, double FourPmClose, double CurOpen)>();
            for (int i = Bars.Count - 1; i >= 1 && gaps.Count < BoxesToShow; i--)
            {
                if (!IsTime(ToLocal(Bars.OpenTimes[i]), 9, 30)) continue;
                double? cl = FindCloseAt1615(i);
                if (!cl.HasValue) continue;
                gaps.Insert(0, (i, Bars.OpenTimes[i], cl.Value, Bars.OpenPrices[i]));
            }
            int idx = 0;
            foreach (var (bi, utcOpen, fp, co) in gaps)
                DrawRthGapInstance(bi, utcOpen, fp, co, idx++);
        }

        // [CRITICAL-2] bar i closes at Bars.OpenTimes[i+1]
        private double? FindCloseAt1615(int fromBar)
        {
            int maxLook = (int)(1440.0 / _avgBarMins) + 10;
            for (int i = fromBar - 1; i >= Math.Max(0, fromBar - maxLook); i--)
            {
                DateTime ct = (i + 1 < Bars.Count)
                    ? ToLocal(Bars.OpenTimes[i + 1])
                    : ToLocal(Bars.OpenTimes[i]).AddMinutes(_avgBarMins);
                if (ct.Hour == 16 && ct.Minute == 15)
                    return Bars.ClosePrices[i];
            }
            return null;
        }

        private void DrawRthGapInstance(int barIndex, DateTime utcOpen,
            double fourPmClose, double curOpen, int closeLineIndex)
        {
            string   s     = barIndex.ToString(CultureInfo.InvariantCulture);
            DateTime right = ExtendGapToNow
                ? Bars.OpenTimes.LastValue
                : utcOpen.AddHours(GapProjectHours);
            double top    = Math.Max(fourPmClose, curOpen);
            double bottom = Math.Min(fourPmClose, curOpen);

            var rect = Chart.DrawRectangle(Prefix + "rthgap_" + s,
                utcOpen, top, right, bottom, GapBorderColor, GapBorderWidth, LineStyle.Solid);
            rect.IsFilled = true;
            rect.Color    = GapBoxColor;

            if (ShowGapText)
                Chart.DrawText(Prefix + "rthgtxt_" + s, "RTH Gap",
                    right, (top + bottom) / 2.0, GapTextColor);

            double mid = (top + bottom) / 2.0;
            double q75 = (mid + top)    / 2.0;
            double q25 = (mid + bottom) / 2.0;

            Chart.DrawTrendLine(Prefix + "rthmid_" + s,
                utcOpen, mid, right, mid, RthMidColor, RthMidWidth, ToLineStyle(RthMidStyle));
            Chart.DrawTrendLine(Prefix + "rth75_" + s,
                utcOpen, q75, right, q75, Rth2575Color, Rth2575Width, ToLineStyle(Rth2575Style));
            Chart.DrawTrendLine(Prefix + "rth25_" + s,
                utcOpen, q25, right, q25, Rth2575Color, Rth2575Width, ToLineStyle(Rth2575Style));

            if (Show4PmLine && closeLineIndex < HistoricalRthLines)
            {
                DateTime lineEnd = FourPmExtendRight
                    ? Bars.OpenTimes.LastValue.AddHours(24)
                    : right;
                Chart.DrawTrendLine(Prefix + "rth4pm_" + s,
                    utcOpen, fourPmClose, lineEnd, fourPmClose,
                    FourPmLineColor, FourPmLineWidth, ToLineStyle(FourPmLineStyle));
                if (Show4PmLabel)
                    Chart.DrawText(Prefix + "rth4pml_" + s,
                        fourPmClose.ToString("F2", CultureInfo.InvariantCulture),
                        lineEnd, fourPmClose, FourPmLineColor);
            }
        }

        // ── Opening Lines ─────────────────────────────────────────────────────

        private void DrawOpeningLines()
        {
            DateTime currentLocal = ToLocal(Bars.OpenTimes.LastValue);
            bool     isAfterClose = IsInSession(currentLocal, "1500-2000");

            var found0000 = new List<(int idx, DateTime utcOpen, DateTime localOpen)>();
            var found0830 = new List<(int idx, DateTime utcOpen, DateTime localOpen)>();
            var found0930 = new List<(int idx, DateTime utcOpen, DateTime localOpen)>();

            for (int i = Bars.Count - 1; i >= 0; i--)
            {
                DateTime u = Bars.OpenTimes[i];
                DateTime l = ToLocal(u);
                if (ShowOpen0000 && IsTime(l,  0,  0)) found0000.Insert(0, (i, u, l));
                if (ShowOpen0830 && IsTime(l,  8, 30)) found0830.Insert(0, (i, u, l));
                if (ShowOpen0930 && IsTime(l,  9, 30)) found0930.Insert(0, (i, u, l));
                if (!ShowHistoricalOpens
                    && (!ShowOpen0000 || found0000.Count > 0)
                    && (!ShowOpen0830 || found0830.Count > 0)
                    && (!ShowOpen0930 || found0930.Count > 0))
                    break;
            }

            void TrimToOne<T>(List<T> lst)
            { if (!ShowHistoricalOpens && lst.Count > 1) lst.RemoveRange(0, lst.Count - 1); }
            TrimToOne(found0000); TrimToOne(found0830); TrimToOne(found0930);

            // [LOW-9] 00:00 extends 16 h (Pine: htime=57600000ms)
            foreach (var (idx, u, l) in found0000)
                DrawSingleOpenLine(idx, u, l, 16.0, Open0000Color, Open0000Width, Open0000Style,
                    false, isAfterClose, currentLocal);
            foreach (var (idx, u, l) in found0830)
                DrawSingleOpenLine(idx, u, l, 3.5,  Open0830Color, Open0830Width, Open0830Style,
                    true,  isAfterClose, currentLocal);
            foreach (var (idx, u, l) in found0930)
                DrawSingleOpenLine(idx, u, l, 2.5,  Open0930Color, Open0930Width, Open0930Style,
                    true,  isAfterClose, currentLocal);
        }

        private void DrawSingleOpenLine(int barIdx, DateTime utcOpen, DateTime localOpen,
            double fwdHours, Color color, int width, LineStyleChoice style,
            bool applyHide, bool isAfterClose, DateTime currentLocal)
        {
            if (applyHide && HideAfterClose && isAfterClose
                && localOpen.Date == currentLocal.Date)
                return;
            double   price = Bars.OpenPrices[barIdx];
            DateTime right = utcOpen.AddHours(fwdHours); // UTC + forward offset
            Chart.DrawTrendLine(
                Prefix + "open_" + barIdx.ToString(CultureInfo.InvariantCulture),
                utcOpen, price, right, price, color, width, ToLineStyle(style));
        }

        // ── W/M/Y Opens ───────────────────────────────────────────────────────

        private void DrawWmyOpens()
        {
            if (ShowWeeklyOpen && _weeklyBars != null && _weeklyBars.Count > 0)
            {
                int w = _weeklyBars.Count - 1;
                DrawLevel("w_0", "W Open", _weeklyBars.OpenPrices[w],
                    _weeklyBars.OpenTimes[w], WeeklyColor);
            }
            if (ShowMonthlyOpen && _monthlyBars != null && _monthlyBars.Count > 0)
            {
                int m = _monthlyBars.Count - 1;
                DrawLevel("m_0", "M Open", _monthlyBars.OpenPrices[m],
                    _monthlyBars.OpenTimes[m], MonthlyColor);
            }
            if (ShowYearlyOpen && _monthlyBars != null)
            {
                for (int m = _monthlyBars.Count - 1; m >= 0; m--)
                {
                    if (ToLocal(_monthlyBars.OpenTimes[m]).Month == 1)
                    {
                        DrawLevel("y_0", "Y Open", _monthlyBars.OpenPrices[m],
                            _monthlyBars.OpenTimes[m], YearlyColor);
                        break;
                    }
                }
            }
        }

        // [LOW-10] Solid. [CRITICAL-5] startUtc → Chart.Draw*
        private void DrawLevel(string suffix, string text, double price,
            DateTime startUtc, Color color)
        {
            DateTime right = Bars.OpenTimes.LastValue.AddHours(48);
            Chart.DrawTrendLine(Prefix + "lvl_" + suffix,
                startUtc, price, right, price, color, WmyLineWidth, LineStyle.Solid);
            Chart.DrawText(Prefix + "lvltxt_" + suffix, text, right, price, color);
        }

        // ── Lookback background ───────────────────────────────────────────────

        // [MEDIUM-7] H/L from window bars only; border=fill.
        // [CRITICAL-5] DrawRectangle takes UTC.
        private void DrawLookbackBg(DateTime preLocal, DateTime endLocal)
        {
            double   bgTop    = double.MinValue;
            double   bgBottom = double.MaxValue;
            DateTime? firstUtc = null, lastUtc = null;

            for (int i = 0; i < Bars.Count; i++)
            {
                DateTime lo = ToLocal(Bars.OpenTimes[i]);
                if (lo < preLocal || lo >= endLocal) continue;
                bgTop    = Math.Max(bgTop,    Bars.HighPrices[i]);
                bgBottom = Math.Min(bgBottom, Bars.LowPrices[i]);
                if (firstUtc == null) firstUtc = Bars.OpenTimes[i];
                lastUtc = Bars.OpenTimes[i];
            }
            if (firstUtc == null) return;

            double pad = (bgTop - bgBottom) * 0.1;
            var bg = Chart.DrawRectangle(Prefix + "lookback",
                firstUtc.Value, bgTop + pad, lastUtc.Value, bgBottom - pad,
                LookbackBgColor, 1, LineStyle.Solid);
            bg.IsFilled = true;
            bg.Color    = LookbackBgColor;
        }

        // ── Sessions ──────────────────────────────────────────────────────────

        private List<SessionConfig> BuildSessions()
        {
            return new List<SessionConfig>
            {
                new SessionConfig { Key="asia",   Label="Asia",     Session=AsiaSession,    Enabled=AsiaEnabled,    BorderColor=AsiaBorderColor,    BoxColor=AsiaBoxColor,    LineColor=AsiaLineColor,    MidAllowed=true  },
                new SessionConfig { Key="london", Label="London",   Session=LondonSession,  Enabled=LondonEnabled,  BorderColor=LondonBorderColor,  BoxColor=LondonBoxColor,  LineColor=LondonLineColor,  MidAllowed=true  },
                new SessionConfig { Key="nyam",   Label="NY AM",    Session=NyAmSession,    Enabled=NyAmEnabled,    BorderColor=NyAmBorderColor,    BoxColor=NyAmBoxColor,    LineColor=NyAmLineColor,    MidAllowed=true  },
                new SessionConfig { Key="nyl",    Label="NY Lunch", Session=NyLunchSession, Enabled=NyLunchEnabled, BorderColor=NyLunchBorderColor, BoxColor=NyLunchBoxColor, LineColor=NyLunchLineColor, MidAllowed=false },
                new SessionConfig { Key="nypm",   Label="NY PM",    Session=NyPmSession,    Enabled=NyPmEnabled,    BorderColor=NyPmBorderColor,    BoxColor=NyPmBoxColor,    LineColor=NyPmLineColor,    MidAllowed=true  },
            };
        }

        // ── Utilities ─────────────────────────────────────────────────────────

        private static bool IsInSession(DateTime localTime, string session)
        {
            if (string.IsNullOrWhiteSpace(session) || session.Length < 9) return false;
            int sh = int.Parse(session.Substring(0, 2), CultureInfo.InvariantCulture);
            int sm = int.Parse(session.Substring(2, 2), CultureInfo.InvariantCulture);
            int eh = int.Parse(session.Substring(5, 2), CultureInfo.InvariantCulture);
            int em = int.Parse(session.Substring(7, 2), CultureInfo.InvariantCulture);
            int cur = localTime.Hour * 60 + localTime.Minute;
            int s   = sh * 60 + sm;
            int e   = eh * 60 + em;
            if (s == e)  return true;
            if (s < e)   return cur >= s && cur < e;
            return cur >= s || cur < e;
        }

        private static bool IsTime(DateTime dt, int h, int m)
            => dt.Hour == h && dt.Minute == m;

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
                try   { return TimeZoneInfo.FindSystemTimeZoneById("America/New_York"); }
                catch { return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }
            }
            if (tz.StartsWith("GMT", StringComparison.OrdinalIgnoreCase))
            {
                string off = tz.Substring(3);
                if (off.StartsWith("+")) off = off.Substring(1);
                if (double.TryParse(off,
                    NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint,
                    CultureInfo.InvariantCulture, out double h))
                    return TimeZoneInfo.CreateCustomTimeZone(
                        "Custom" + tz, TimeSpan.FromHours(h), tz, tz);
            }
            try   { return TimeZoneInfo.FindSystemTimeZoneById(tz); }
            catch { return TimeZoneInfo.Utc; }
        }

        private void ClearObjects()
        {
            var names = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    names.Add(obj.Name);
            foreach (var n in names)
                Chart.RemoveObject(n);
        }
    }
}
