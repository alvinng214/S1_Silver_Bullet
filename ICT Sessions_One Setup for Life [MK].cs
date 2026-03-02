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

        private sealed class SessionConfig
        {
            public string Key;
            public string Label;
            public string Session;
            public bool Enabled;
            public Color BorderColor;
            public Color BoxColor;
            public Color LineColor;
            public bool MidAllowed;
        }

        private sealed class SessionState
        {
            public bool Active;
            public DateTime Start;
            public DateTime End;
            public double High;
            public double Low;
            public string HighLineId;
            public string LowLineId;
            public string MidLineId;
            public string BoxId;
            public string LabelId;
        }

        public enum LineStyleChoice
        {
            Solid,
            Dotted,
            Dashed
        }

        [Parameter("Timezone", Group = "General", DefaultValue = "America/New_York")]
        public string TimezoneInput { get; set; }

        [Parameter("Max Timeframe (min)", Group = "General", DefaultValue = 15, MinValue = 1, MaxValue = 240)]
        public int MaxTimeframeMinutes { get; set; }

        [Parameter("Previous Days to Show", Group = "Lookback", DefaultValue = 10, MinValue = 1, MaxValue = 365)]
        public int EventDays { get; set; }

        [Parameter("Show Days Background", Group = "Lookback", DefaultValue = true)]
        public bool ShowDaysBackground { get; set; }

        [Parameter("Background", Group = "Lookback", DefaultValue = "#E6C0C0C0")]
        public Color LookbackBgColor { get; set; }

        [Parameter("Session High/Low Lines", Group = "Sessions", DefaultValue = true)]
        public bool ShowSessionLines { get; set; }

        [Parameter("Session 50% Line", Group = "Sessions", DefaultValue = false)]
        public bool ShowSessionMid { get; set; }

        [Parameter("Border Width", Group = "Sessions", DefaultValue = 1, MinValue = 1, MaxValue = 5)]
        public int SessionBorderWidth { get; set; }

        [Parameter("Show Labels", Group = "Sessions", DefaultValue = true)]
        public bool ShowSessionLabels { get; set; }

        [Parameter("Label Color", Group = "Sessions", DefaultValue = "#FFC0C0C0")]
        public Color SessionLabelColor { get; set; }

        [Parameter("Asia Session", Group = "Asia", DefaultValue = true)]
        public bool AsiaEnabled { get; set; }
        [Parameter("Time", Group = "Asia", DefaultValue = "2000-0000")]
        public string AsiaSession { get; set; }
        [Parameter("Border", Group = "Asia", DefaultValue = "#00FFA500")]
        public Color AsiaBorderColor { get; set; }
        [Parameter("Box", Group = "Asia", DefaultValue = "#33FFA500")]
        public Color AsiaBoxColor { get; set; }
        [Parameter("Line", Group = "Asia", DefaultValue = "#FFFFA500")]
        public Color AsiaLineColor { get; set; }

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

        [Parameter("Show 4pm Close Line", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLine { get; set; }
        [Parameter("Show 4pm Label", Group = "RTH Close", DefaultValue = false)]
        public bool Show4PmLabel { get; set; }
        [Parameter("Historical Lines", Group = "RTH Close", DefaultValue = 3, MinValue = 1, MaxValue = 20)]
        public int HistoricalRthLines { get; set; }
        [Parameter("Line Color", Group = "RTH Close", DefaultValue = "#FFE0E0E0")]
        public Color FourPmLineColor { get; set; }
        [Parameter("Line Width", Group = "RTH Close", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int FourPmLineWidth { get; set; }

        [Parameter("00:00 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0000 { get; set; }
        [Parameter("08:30 Open", Group = "Opening Lines", DefaultValue = true)]
        public bool ShowOpen0830 { get; set; }
        [Parameter("09:30 Open", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowOpen0930 { get; set; }
        [Parameter("Show Historical", Group = "Opening Lines", DefaultValue = false)]
        public bool ShowHistoricalOpens { get; set; }
        [Parameter("Hide 08:30/09:30 After Close", Group = "Opening Lines", DefaultValue = true)]
        public bool HideAfterClose { get; set; }

        [Parameter("Weekly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowWeeklyOpen { get; set; }
        [Parameter("Monthly", Group = "W/M/Y Open", DefaultValue = true)]
        public bool ShowMonthlyOpen { get; set; }
        [Parameter("Yearly", Group = "W/M/Y Open", DefaultValue = false)]
        public bool ShowYearlyOpen { get; set; }

        private TimeZoneInfo _selectedTimeZone;
        private readonly List<string> _drawnObjectIds = new List<string>();

        protected override void Initialize()
        {
            _selectedTimeZone = ResolveTimeZone(TimezoneInput);
        }

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1)
                return;

            _selectedTimeZone = ResolveTimeZone(TimezoneInput);

            ClearPrefixedObjects();

            var timeframeMinutes = GetAverageBarMinutes();
            bool dispRth = timeframeMinutes <= MaxTimeframeMinutes && timeframeMinutes < 1440;
            if (!dispRth)
                return;

            DateTime lastLocal = ToLocal(Bars.OpenTimes.LastValue);
            DateTime startTime = new DateTime(lastLocal.Year, lastLocal.Month, lastLocal.Day, 23, 59, 0);
            DateTime preTs = startTime.AddDays(-EventDays);

            if (ShowDaysBackground)
            {
                DrawLookbackBackground(preTs, startTime);
            }

            var sessions = BuildSessions();
            var states = new Dictionary<string, SessionState>();
            foreach (var cfg in sessions)
                states[cfg.Key] = new SessionState();

            var rthGapCount = 0;
            var closeLineCount = 0;
            var openingCountByType = new Dictionary<string, int>();

            for (int i = 0; i < Bars.Count; i++)
            {
                DateTime barOpenUtc = Bars.OpenTimes[i];
                DateTime barCloseUtc = i < Bars.Count - 1 ? Bars.OpenTimes[i + 1] : barOpenUtc.AddMinutes(timeframeMinutes);
                DateTime localOpen = ToLocal(barOpenUtc);
                DateTime localClose = ToLocal(barCloseUtc);

                bool inPreRange = localOpen >= preTs && localOpen < startTime;
                if (!inPreRange)
                    continue;

                double high = Bars.HighPrices[i];
                double low = Bars.LowPrices[i];

                foreach (var cfg in sessions)
                {
                    if (!cfg.Enabled)
                        continue;

                    bool inSession = IsInSession(localOpen, cfg.Session);
                    bool prevInSession = i > 0 && IsInSession(ToLocal(Bars.OpenTimes[i - 1]), cfg.Session);
                    bool isNew = inSession && !prevInSession;
                    var state = states[cfg.Key];

                    if (inSession)
                    {
                        if (isNew)
                        {
                            state.Active = true;
                            state.Start = localOpen;
                            state.End = localClose;
                            state.High = high;
                            state.Low = low;
                            string suffix = $"{cfg.Key}_{i}";
                            state.BoxId = Prefix + "box_" + suffix;
                            state.HighLineId = Prefix + "h_" + suffix;
                            state.LowLineId = Prefix + "l_" + suffix;
                            state.MidLineId = Prefix + "m_" + suffix;
                            state.LabelId = Prefix + "lbl_" + suffix;
                        }
                        else if (state.Active)
                        {
                            state.End = localClose;
                            state.High = Math.Max(state.High, high);
                            state.Low = Math.Min(state.Low, low);
                        }

                        DrawSession(state, cfg);
                    }
                    else if (state.Active)
                    {
                        if (high > state.High || low < state.Low)
                        {
                            state.Active = false;
                        }
                    }
                }

                if (ShowRthGap && IsTime(localOpen, 9, 30))
                {
                    double? prevClose = FindPreviousCloseAtTime(i, 16, 15);
                    if (prevClose.HasValue)
                    {
                        rthGapCount++;
                        if (rthGapCount <= BoxesToShow)
                        {
                            DrawRthGap(i, localOpen, localClose, prevClose.Value, Bars.OpenPrices[i], closeLineCount++);
                        }
                    }
                }

                DrawOpeningLines(i, localOpen, openingCountByType);
                DrawWmyOpens(i, localOpen);
            }
        }

        private void DrawSession(SessionState state, SessionConfig cfg)
        {
            DateTime midTime = state.Start.AddTicks((state.End - state.Start).Ticks / 2);
            double midPrice = (state.High + state.Low) / 2.0;
            var box = Chart.DrawRectangle(state.BoxId, state.Start, state.High, state.End, state.Low, cfg.BorderColor, SessionBorderWidth, LineStyle.Solid);
            box.IsFilled = true;
            box.Color = cfg.BoxColor;

            if (ShowSessionLines)
            {
                Chart.DrawTrendLine(state.HighLineId, state.Start, state.High, state.End, state.High, cfg.LineColor, 1, LineStyle.Solid);
                Chart.DrawTrendLine(state.LowLineId, state.Start, state.Low, state.End, state.Low, cfg.LineColor, 1, LineStyle.Solid);
                if (ShowSessionMid && cfg.MidAllowed)
                    Chart.DrawTrendLine(state.MidLineId, state.Start, midPrice, state.End, midPrice, cfg.LineColor, 1, LineStyle.Solid);
            }

            if (ShowSessionLabels)
                Chart.DrawText(state.LabelId, cfg.Label, midTime, state.High, SessionLabelColor);
        }

        private void DrawRthGap(int index, DateTime localOpen, DateTime localClose, double fourPmClose, double currentOpen, int closeLineIndex)
        {
            DateTime right = ExtendGapToNow ? ToLocal(Bars.OpenTimes.LastValue) : localOpen.AddHours(GapProjectHours);
            double top = Math.Max(fourPmClose, currentOpen);
            double bottom = Math.Min(fourPmClose, currentOpen);
            string id = Prefix + "rthgap_" + index;
            var rect = Chart.DrawRectangle(id, localOpen, top, right, bottom, Color.FromArgb(110, Color.Gray));
            rect.IsFilled = true;

            double mid = (top + bottom) / 2.0;
            double q75 = (mid + top) / 2.0;
            double q25 = (mid + bottom) / 2.0;

            Chart.DrawTrendLine(Prefix + "rthmid_" + index, localOpen, mid, right, mid, Color.White, 2, LineStyle.Solid);
            Chart.DrawTrendLine(Prefix + "rth75_" + index, localOpen, q75, right, q75, Color.White, 1, LineStyle.Solid);
            Chart.DrawTrendLine(Prefix + "rth25_" + index, localOpen, q25, right, q25, Color.White, 1, LineStyle.Solid);

            if (Show4PmLine && closeLineIndex < HistoricalRthLines)
            {
                DateTime end = ToLocal(Bars.OpenTimes.LastValue);
                Chart.DrawTrendLine(Prefix + "rth4pm_" + index, localOpen, fourPmClose, end, fourPmClose, FourPmLineColor, FourPmLineWidth, LineStyle.Solid);
                if (Show4PmLabel)
                    Chart.DrawText(Prefix + "rth4pmlbl_" + index, fourPmClose.ToString("F2", CultureInfo.InvariantCulture), end, fourPmClose, FourPmLineColor);
            }
        }

        private void DrawOpeningLines(int i, DateTime localOpen, Dictionary<string, int> openingCountByType)
        {
            DrawOpenAt(localOpen, i, "0000", ShowOpen0000, 3.0, Color.Aqua, openingCountByType);
            DrawOpenAt(localOpen, i, "0830", ShowOpen0830, 3.5, Color.Gold, openingCountByType);
            DrawOpenAt(localOpen, i, "0930", ShowOpen0930, 2.5, Color.OrangeRed, openingCountByType);

            if (HideAfterClose && localOpen.Hour >= 15)
            {
                // cTrader has no batch hide by tag; regenerated objects naturally stop being drawn after 15:00 in this implementation.
            }
        }

        private void DrawOpenAt(DateTime localOpen, int i, string hhmm, bool enabled, double fwdHours, Color color, Dictionary<string, int> openingCountByType)
        {
            if (!enabled)
                return;
            if (!IsTime(localOpen, int.Parse(hhmm.Substring(0, 2)), int.Parse(hhmm.Substring(2, 2))))
                return;

            if (!openingCountByType.ContainsKey(hhmm))
                openingCountByType[hhmm] = 0;

            openingCountByType[hhmm]++;
            if (!ShowHistoricalOpens && openingCountByType[hhmm] > 1)
                return;

            double price = Bars.OpenPrices[i];
            DateTime right = localOpen.AddHours(fwdHours);
            Chart.DrawTrendLine(Prefix + "open_" + hhmm + "_" + i, localOpen, price, right, price, color, 2, LineStyle.Solid);
            Chart.DrawText(Prefix + "open_lbl_" + hhmm + "_" + i, hhmm + " Open", right, price, color);
        }

        private void DrawWmyOpens(int i, DateTime localOpen)
        {
            if (ShowWeeklyOpen && (localOpen.DayOfWeek == DayOfWeek.Monday) && IsTime(localOpen, 0, 0))
                DrawLevel("w_" + i, "W Open", Bars.OpenPrices[i], localOpen, Color.DodgerBlue);

            if (ShowMonthlyOpen && localOpen.Day == 1 && IsTime(localOpen, 0, 0))
                DrawLevel("m_" + i, "M Open", Bars.OpenPrices[i], localOpen, Color.DeepPink);

            if (ShowYearlyOpen && localOpen.DayOfYear == 1 && IsTime(localOpen, 0, 0))
                DrawLevel("y_" + i, "Y Open", Bars.OpenPrices[i], localOpen, Color.MediumPurple);
        }

        private void DrawLevel(string suffix, string text, double price, DateTime start, Color color)
        {
            DateTime right = ToLocal(Bars.OpenTimes.LastValue).AddHours(8);
            Chart.DrawTrendLine(Prefix + "lvl_" + suffix, start, price, right, price, color, 2, LineStyle.DotsRare);
            Chart.DrawText(Prefix + "lvltxt_" + suffix, text + " " + price.ToString("F2", CultureInfo.InvariantCulture), right, price, color);
        }

        private List<SessionConfig> BuildSessions()
        {
            return new List<SessionConfig>
            {
                new SessionConfig { Key = "asia", Label = "Asia", Session = AsiaSession, Enabled = AsiaEnabled, BorderColor = AsiaBorderColor, BoxColor = AsiaBoxColor, LineColor = AsiaLineColor, MidAllowed = true },
                new SessionConfig { Key = "london", Label = "London", Session = LondonSession, Enabled = LondonEnabled, BorderColor = LondonBorderColor, BoxColor = LondonBoxColor, LineColor = LondonLineColor, MidAllowed = true },
                new SessionConfig { Key = "nyam", Label = "NY AM", Session = NyAmSession, Enabled = NyAmEnabled, BorderColor = NyAmBorderColor, BoxColor = NyAmBoxColor, LineColor = NyAmLineColor, MidAllowed = true },
                new SessionConfig { Key = "nyl", Label = "NY Lunch", Session = NyLunchSession, Enabled = NyLunchEnabled, BorderColor = NyLunchBorderColor, BoxColor = NyLunchBoxColor, LineColor = NyLunchLineColor, MidAllowed = false },
                new SessionConfig { Key = "nypm", Label = "NY PM", Session = NyPmSession, Enabled = NyPmEnabled, BorderColor = NyPmBorderColor, BoxColor = NyPmBoxColor, LineColor = NyPmLineColor, MidAllowed = true }
            };
        }

        private void DrawLookbackBackground(DateTime start, DateTime end)
        {
            var top = Bars.HighPrices.Maximum(Bars.Count);
            var bottom = Bars.LowPrices.Minimum(Bars.Count);
            var rect = Chart.DrawRectangle(Prefix + "lookback", start, top, end, bottom, LookbackBgColor);
            rect.IsFilled = true;
        }

        private double? FindPreviousCloseAtTime(int currentBar, int hour, int minute)
        {
            for (int i = currentBar - 1; i >= 0; i--)
            {
                DateTime local = ToLocal(Bars.OpenTimes[i]);
                if (local.Hour == hour && local.Minute == minute)
                    return Bars.ClosePrices[i];
            }
            return null;
        }

        private bool IsInSession(DateTime localTime, string session)
        {
            if (string.IsNullOrWhiteSpace(session) || session.Length < 9)
                return false;

            int sh = int.Parse(session.Substring(0, 2));
            int sm = int.Parse(session.Substring(2, 2));
            int eh = int.Parse(session.Substring(5, 2));
            int em = int.Parse(session.Substring(7, 2));

            int current = localTime.Hour * 60 + localTime.Minute;
            int start = sh * 60 + sm;
            int end = eh * 60 + em;

            if (start == end)
                return true;
            if (start < end)
                return current >= start && current < end;
            return current >= start || current < end;
        }

        private bool IsTime(DateTime dt, int hour, int minute)
        {
            return dt.Hour == hour && dt.Minute == minute;
        }

        private DateTime ToLocal(DateTime utc)
        {
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.SpecifyKind(utc, DateTimeKind.Utc), _selectedTimeZone);
        }

        private double GetAverageBarMinutes()
        {
            if (Bars.Count < 2)
                return 1;
            int sample = Math.Min(200, Bars.Count - 1);
            double sum = 0;
            for (int i = Bars.Count - sample; i < Bars.Count; i++)
                sum += (Bars.OpenTimes[i] - Bars.OpenTimes[i - 1]).TotalMinutes;
            return sum / sample;
        }

        private TimeZoneInfo ResolveTimeZone(string timezone)
        {
            if (string.Equals(timezone, "America/New_York", StringComparison.OrdinalIgnoreCase))
            {
                try { return TimeZoneInfo.FindSystemTimeZoneById("America/New_York"); }
                catch { return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }
            }

            if (timezone.StartsWith("GMT", StringComparison.OrdinalIgnoreCase))
            {
                string offsetPart = timezone.Substring(3);
                if (offsetPart.StartsWith("+"))
                    offsetPart = offsetPart.Substring(1);
                if (double.TryParse(offsetPart, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double hours))
                    return TimeZoneInfo.CreateCustomTimeZone("Custom" + timezone, TimeSpan.FromHours(hours), timezone, timezone);
            }

            try { return TimeZoneInfo.FindSystemTimeZoneById(timezone); }
            catch { return TimeZoneInfo.Utc; }
        }

        private void ClearPrefixedObjects()
        {
            foreach (var obj in Chart.Objects)
            {
                if (obj.Name.StartsWith(Prefix, StringComparison.Ordinal))
                    Chart.RemoveObject(obj.Name);
            }
            _drawnObjectIds.Clear();
        }
    }
}
