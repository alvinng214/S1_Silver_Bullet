// =============================================================================
// SMC Target Liquidity V.35 (Manual Distance Control).cs — C# cTrader port
// =============================================================================
// Fixes vs previous version:
//   - Renamed struct field 'LineStyle LineStyle' to 'EvtStyle' — same name as
//     its type caused a silent compiler failure (nothing shown in cTrader).
//   - Replaced string TimezoneInput with TimezoneOption enum → dropdown list.
//
// Architecture: last-bar-only + ClearObjects + redraw.
//
// Pine logic reproduced:
//   Pivot detection : strict (no ties). Pivot at bar - prd.
//   is_broken filter: right-wing bars [pivot+1 .. current] violate level → reject.
//   Session color   : checked at pivot bar (Pine's [prd] shift). Priority: NY PM > NY AM > London > Asia.
//   Trigger order   : SFP first, then MSS, then X (elif — mutually exclusive).
//   Label midpoint  : UTC midpoint between pivot origin and trigger bar.
//   Label Y         : lvl ± (lbl_offset * TickSize) with flip toggle.
//   Active extension: extUtc = lastBarUtc + lineExtension * avgBarMins.
//
// Note: Pine's Text Size has no equivalent in cTrader's Chart.DrawText API.
//   The parameter is kept for UI parity only.
// =============================================================================

using System;
using System.Collections.Generic;
using System.Globalization;
using cAlgo.API;

namespace cAlgo
{
    // ── Enums at namespace level — required for cTrader [Parameter] dropdown ──
    public enum LineStyleChoice { Solid, Dashed, Dotted }

    public enum TimezoneOption
    {
        America_New_York,
        Asia_Bangkok,
        GMT_Minus12, GMT_Minus11, GMT_Minus10, GMT_Minus9, GMT_Minus8,
        GMT_Minus7,  GMT_Minus6,  GMT_Minus5,  GMT_Minus4, GMT_Minus3,
        GMT_Minus2,  GMT_Minus1,  GMT_0,
        GMT_Plus1,   GMT_Plus2,   GMT_Plus3,   GMT_Plus4,  GMT_Plus5,
        GMT_Plus6,   GMT_Plus7,   GMT_Plus8,   GMT_Plus9,  GMT_Plus10,
        GMT_Plus11,  GMT_Plus12,
    }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMCTargetLiquidityV35 : Indicator
    {
        private const string Prefix = "SMCTL_";

        // ── Inner types ───────────────────────────────────────────────────────

        private struct ActiveLevel
        {
            public double   Price;
            public DateTime OriginUtc;
            public Color    LineColor;
        }

        // [FIX] Field renamed EvtStyle — was 'LineStyle LineStyle' which conflicts
        // with the cAlgo.API.LineStyle type, causing a silent compiler failure.
        private struct CompletedEvent
        {
            public DateTime  OriginUtc;
            public DateTime  TriggerUtc;
            public DateTime  MidUtc;
            public double    Price;
            public Color     LineColor;
            public LineStyle EvtStyle;     // ← renamed from LineStyle
            public bool      ShowLabel;
            public string    LabelText;
            public Color     LabelTextColor;
            public double    LabelY;
        }

        // ── Parameters — Main Logic ───────────────────────────────────────────

        [Parameter("Pivot Period", Group = "Main Logic", DefaultValue = 10, MinValue = 2)]
        public int PivotPeriod { get; set; }

        [Parameter("Max Active Lines", Group = "Main Logic", DefaultValue = 5, MinValue = 1)]
        public int MaxActiveLines { get; set; }

        // [FIX] Enum → renders as dropdown in cTrader UI
        [Parameter("Timezone", Group = "Main Logic", DefaultValue = TimezoneOption.America_New_York)]
        public TimezoneOption Timezone { get; set; }

        // ── Parameters — Active Session Lines ─────────────────────────────────

        [Parameter("Active Line Width", Group = "Active Session Lines", DefaultValue = 2, MinValue = 1, MaxValue = 5)]
        public int ActiveWidth { get; set; }

        [Parameter("Asian Session", Group = "Active Session Lines", DefaultValue = "2000-0000")]
        public string AsiaSess { get; set; }

        [Parameter("Asian Color", Group = "Active Session Lines", DefaultValue = "#FFFFFF00")]
        public Color ColAsia { get; set; }

        [Parameter("London Session", Group = "Active Session Lines", DefaultValue = "0200-0500")]
        public string LondonSess { get; set; }

        [Parameter("London Color", Group = "Active Session Lines", DefaultValue = "#FF008080")]
        public Color ColLondon { get; set; }

        [Parameter("NY AM Session", Group = "Active Session Lines", DefaultValue = "0830-1100")]
        public string NyAmSess { get; set; }

        [Parameter("NY AM Color", Group = "Active Session Lines", DefaultValue = "#FFFFA500")]
        public Color ColNyAm { get; set; }

        [Parameter("NY PM Session", Group = "Active Session Lines", DefaultValue = "1330-1600")]
        public string NyPmSess { get; set; }

        [Parameter("NY PM Color", Group = "Active Session Lines", DefaultValue = "#FFFF00FF")]
        public Color ColNyPm { get; set; }

        [Parameter("Outside Session Color", Group = "Active Session Lines", DefaultValue = "#FF808080")]
        public Color ColOther { get; set; }

        // ── Parameters — Display & Size Settings ─────────────────────────────

        // Note: no font size API in cTrader Chart.DrawText — kept for UI parity only.
        [Parameter("Text Size", Group = "Display & Size Settings", DefaultValue = "Large")]
        public string LabelSizeInput { get; set; }

        [Parameter("Text Distance (Ticks)", Group = "Display & Size Settings", DefaultValue = 30.0, MinValue = 0)]
        public double LblOffset { get; set; }

        [Parameter("Flip Text Position", Group = "Display & Size Settings", DefaultValue = false)]
        public bool FlipText { get; set; }

        [Parameter("Show Price Labels (Active)", Group = "Display & Size Settings", DefaultValue = true)]
        public bool ShowPrice { get; set; }

        [Parameter("Line Extension Bars", Group = "Display & Size Settings", DefaultValue = 15, MinValue = 0)]
        public int LineExtension { get; set; }

        // ── Parameters — Style: X (Sweep) ────────────────────────────────────

        [Parameter("Show X", Group = "Style: X (Sweep)", DefaultValue = true)]
        public bool ShowX { get; set; }

        [Parameter("Line Color", Group = "Style: X (Sweep)", DefaultValue = "#FF808080")]
        public Color ColLineX { get; set; }

        [Parameter("Text Color", Group = "Style: X (Sweep)", DefaultValue = "#FF000000")]
        public Color ColTxtX { get; set; }

        [Parameter("Line Style", Group = "Style: X (Sweep)", DefaultValue = LineStyleChoice.Dotted)]
        public LineStyleChoice StyleX { get; set; }

        // ── Parameters — Style: SFP (Fakeout) ────────────────────────────────

        [Parameter("Show SFP", Group = "Style: SFP (Fakeout)", DefaultValue = true)]
        public bool ShowSfp { get; set; }

        [Parameter("Line Color", Group = "Style: SFP (Fakeout)", DefaultValue = "#FFFF0000")]
        public Color ColLineSfp { get; set; }

        [Parameter("Text Color", Group = "Style: SFP (Fakeout)", DefaultValue = "#FF000000")]
        public Color ColTxtSfp { get; set; }

        [Parameter("Line Style", Group = "Style: SFP (Fakeout)", DefaultValue = LineStyleChoice.Dashed)]
        public LineStyleChoice StyleSfp { get; set; }

        // ── Parameters — Style: MSS (Breakout) ───────────────────────────────

        [Parameter("Show MSS", Group = "Style: MSS (Breakout)", DefaultValue = true)]
        public bool ShowMss { get; set; }

        [Parameter("Line Color", Group = "Style: MSS (Breakout)", DefaultValue = "#FF000000")]
        public Color ColLineMss { get; set; }

        [Parameter("Text Color", Group = "Style: MSS (Breakout)", DefaultValue = "#FF000000")]
        public Color ColTxtMss { get; set; }

        [Parameter("Line Style", Group = "Style: MSS (Breakout)", DefaultValue = LineStyleChoice.Solid)]
        public LineStyleChoice StyleMss { get; set; }

        // ── Private fields ────────────────────────────────────────────────────

        private TimeZoneInfo _tz;
        private double       _avgBarMins;
        private int          _objId;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            _tz = ResolveTimezone(Timezone);
        }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index != Bars.Count - 1) return;

            _tz         = ResolveTimezone(Timezone);
            _avgBarMins = GetAvgBarMins();
            _objId      = 0;
            ClearObjects();

            int minStart = PivotPeriod * 2;
            if (index < minStart) return;

            double offsetVal = LblOffset * Symbol.TickSize;
            double multHigh  = FlipText ? -1.0 :  1.0; // sell labels: above normally
            double multLow   = FlipText ?  1.0 : -1.0; // buy  labels: below normally

            var activeBuy  = new List<ActiveLevel>();
            var activeSell = new List<ActiveLevel>();
            var events     = new List<CompletedEvent>();

            for (int i = minStart; i <= index; i++)
            {
                DateTime utcBar = Bars.OpenTimes[i];
                double   hi     = Bars.HighPrices[i];
                double   lo     = Bars.LowPrices[i];
                double   cl     = Bars.ClosePrices[i];
                double   prevCl = Bars.ClosePrices[i - 1];
                bool     closed = i < Bars.Count - 1;

                // ── New pivot high → sell (resistance) level ──────────────────
                if (closed && IsPivotHigh(i))
                {
                    int    pb = i - PivotPeriod;
                    double pp = Bars.HighPrices[pb];

                    // Pine: is_broken — any right-wing bar with high > ph?
                    bool broken = false;
                    for (int j = pb + 1; j <= i; j++)
                        if (Bars.HighPrices[j] > pp) { broken = true; break; }

                    if (!broken)
                    {
                        activeSell.Add(new ActiveLevel
                        {
                            Price     = pp,
                            OriginUtc = Bars.OpenTimes[pb],
                            LineColor = GetSessionColor(pb),
                        });
                        while (activeSell.Count > MaxActiveLines)
                            activeSell.RemoveAt(0);
                    }
                }

                // ── New pivot low → buy (support) level ───────────────────────
                if (closed && IsPivotLow(i))
                {
                    int    pb = i - PivotPeriod;
                    double pp = Bars.LowPrices[pb];

                    bool broken = false;
                    for (int j = pb + 1; j <= i; j++)
                        if (Bars.LowPrices[j] < pp) { broken = true; break; }

                    if (!broken)
                    {
                        activeBuy.Add(new ActiveLevel
                        {
                            Price     = pp,
                            OriginUtc = Bars.OpenTimes[pb],
                            LineColor = GetSessionColor(pb),
                        });
                        while (activeBuy.Count > MaxActiveLines)
                            activeBuy.RemoveAt(0);
                    }
                }

                if (!closed) continue;

                // ── Check sell (resistance) levels: SFP > MSS > X ────────────
                for (int s = activeSell.Count - 1; s >= 0; s--)
                {
                    ActiveLevel lvl    = activeSell[s];
                    double      labelY = lvl.Price + offsetVal * multHigh;
                    DateTime    midUtc = lvl.OriginUtc
                        + TimeSpan.FromTicks((utcBar - lvl.OriginUtc).Ticks / 2);

                    if (cl < lvl.Price && prevCl > lvl.Price)
                    {
                        // SFP: close < lvl AND close[1] > lvl
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineSfp, ToLineStyle(StyleSfp), ShowSfp, "SFP", ColTxtSfp);
                        activeSell.RemoveAt(s);
                    }
                    else if (cl > lvl.Price && prevCl > lvl.Price && cl > prevCl)
                    {
                        // MSS: close > lvl AND close[1] > lvl AND close > close[1]
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineMss, ToLineStyle(StyleMss), ShowMss, "MSS", ColTxtMss);
                        activeSell.RemoveAt(s);
                    }
                    else if (hi >= lvl.Price && cl < lvl.Price)
                    {
                        // X: high >= lvl AND close < lvl
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineX, ToLineStyle(StyleX), ShowX, "X", ColTxtX);
                        activeSell.RemoveAt(s);
                    }
                }

                // ── Check buy (support) levels: SFP > MSS > X ────────────────
                for (int b = activeBuy.Count - 1; b >= 0; b--)
                {
                    ActiveLevel lvl    = activeBuy[b];
                    double      labelY = lvl.Price + offsetVal * multLow;
                    DateTime    midUtc = lvl.OriginUtc
                        + TimeSpan.FromTicks((utcBar - lvl.OriginUtc).Ticks / 2);

                    if (cl > lvl.Price && prevCl < lvl.Price)
                    {
                        // SFP: close > lvl AND close[1] < lvl
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineSfp, ToLineStyle(StyleSfp), ShowSfp, "SFP", ColTxtSfp);
                        activeBuy.RemoveAt(b);
                    }
                    else if (cl < lvl.Price && prevCl < lvl.Price && cl < prevCl)
                    {
                        // MSS: close < lvl AND close[1] < lvl AND close < close[1]
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineMss, ToLineStyle(StyleMss), ShowMss, "MSS", ColTxtMss);
                        activeBuy.RemoveAt(b);
                    }
                    else if (lo <= lvl.Price && cl > lvl.Price)
                    {
                        // X: low <= lvl AND close > lvl
                        AppendEvent(events, lvl, utcBar, midUtc, labelY,
                            ColLineX, ToLineStyle(StyleX), ShowX, "X", ColTxtX);
                        activeBuy.RemoveAt(b);
                    }
                }
            }

            // ── Draw completed events ─────────────────────────────────────────
            foreach (CompletedEvent ev in events)
            {
                Chart.DrawTrendLine(Prefix + "el_" + NextId(),
                    ev.OriginUtc, ev.Price, ev.TriggerUtc, ev.Price,
                    ev.LineColor, ActiveWidth, ev.EvtStyle);  // ← EvtStyle (not LineStyle)

                if (ev.ShowLabel)
                    Chart.DrawText(Prefix + "etl_" + NextId(),
                        ev.LabelText, ev.MidUtc, ev.LabelY, ev.LabelTextColor);
            }

            // ── Draw active (waiting) levels ──────────────────────────────────
            DateTime extUtc = Bars.OpenTimes[index]
                + TimeSpan.FromMinutes(LineExtension * _avgBarMins);

            string fmt = "F" + Symbol.Digits;

            foreach (ActiveLevel lvl in activeSell)
            {
                Chart.DrawTrendLine(Prefix + "asl_" + NextId(),
                    lvl.OriginUtc, lvl.Price, extUtc, lvl.Price,
                    lvl.LineColor, ActiveWidth, LineStyle.Solid);
                if (ShowPrice)
                    Chart.DrawText(Prefix + "aspl_" + NextId(),
                        lvl.Price.ToString(fmt, CultureInfo.InvariantCulture),
                        extUtc, lvl.Price, lvl.LineColor);
            }

            foreach (ActiveLevel lvl in activeBuy)
            {
                Chart.DrawTrendLine(Prefix + "abl_" + NextId(),
                    lvl.OriginUtc, lvl.Price, extUtc, lvl.Price,
                    lvl.LineColor, ActiveWidth, LineStyle.Solid);
                if (ShowPrice)
                    Chart.DrawText(Prefix + "abpl_" + NextId(),
                        lvl.Price.ToString(fmt, CultureInfo.InvariantCulture),
                        extUtc, lvl.Price, lvl.LineColor);
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────

        private static void AppendEvent(List<CompletedEvent> events,
            ActiveLevel lvl, DateTime triggerUtc, DateTime midUtc, double labelY,
            Color lineColor, LineStyle evtStyle, bool show, string text, Color textColor)
        {
            events.Add(new CompletedEvent
            {
                OriginUtc      = lvl.OriginUtc,
                TriggerUtc     = triggerUtc,
                MidUtc         = midUtc,
                Price          = lvl.Price,
                LineColor      = lineColor,
                EvtStyle       = evtStyle,
                ShowLabel      = show,
                LabelText      = text,
                LabelTextColor = textColor,
                LabelY         = labelY,
            });
        }

        // ── Pivot detection ───────────────────────────────────────────────────

        private bool IsPivotHigh(int index)
        {
            int    c  = index - PivotPeriod;
            int    L  = c - PivotPeriod;
            if (L < 0) return false;
            double ph = Bars.HighPrices[c];
            // Pine: LEFT side strictly greater (no ties allowed)
            for (int j = L; j < c; j++)
                if (Bars.HighPrices[j] >= ph) return false;
            // Pine: RIGHT side allows ties (>=pivot is OK; only strictly higher fails)
            for (int j = c + 1; j <= index; j++)
                if (Bars.HighPrices[j] > ph) return false;
            return true;
        }

        private bool IsPivotLow(int index)
        {
            int    c  = index - PivotPeriod;
            int    L  = c - PivotPeriod;
            if (L < 0) return false;
            double pl = Bars.LowPrices[c];
            // Pine: LEFT side strictly lower (no ties allowed)
            for (int j = L; j < c; j++)
                if (Bars.LowPrices[j] <= pl) return false;
            // Pine: RIGHT side allows ties (<=pivot is OK; only strictly lower fails)
            for (int j = c + 1; j <= index; j++)
                if (Bars.LowPrices[j] < pl) return false;
            return true;
        }

        // ── Session color — checked at the pivot bar ──────────────────────────

        private Color GetSessionColor(int pivotBarIndex)
        {
            DateTime local = ToLocal(Bars.OpenTimes[pivotBarIndex]);
            if (IsInSession(local, NyPmSess))   return ColNyPm;
            if (IsInSession(local, NyAmSess))   return ColNyAm;
            if (IsInSession(local, LondonSess)) return ColLondon;
            if (IsInSession(local, AsiaSess))   return ColAsia;
            return ColOther;
        }

        // ── Session utilities (same pattern as ICT Sessions) ──────────────────

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

        private DateTime ToLocal(DateTime utc)
            => TimeZoneInfo.ConvertTimeFromUtc(
                   DateTime.SpecifyKind(utc, DateTimeKind.Utc), _tz);

        // ── Timezone resolution from enum ─────────────────────────────────────

        private static TimeZoneInfo ResolveTimezone(TimezoneOption opt)
        {
            switch (opt)
            {
                case TimezoneOption.America_New_York:
                    try   { return TimeZoneInfo.FindSystemTimeZoneById("America/New_York"); }
                    catch { return TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"); }

                case TimezoneOption.Asia_Bangkok:
                    try   { return TimeZoneInfo.FindSystemTimeZoneById("Asia/Bangkok"); }
                    catch { return TimeZoneInfo.FindSystemTimeZoneById("SE Asia Standard Time"); }

                case TimezoneOption.GMT_Minus12: return MakeGmt(-12);
                case TimezoneOption.GMT_Minus11: return MakeGmt(-11);
                case TimezoneOption.GMT_Minus10: return MakeGmt(-10);
                case TimezoneOption.GMT_Minus9:  return MakeGmt(-9);
                case TimezoneOption.GMT_Minus8:  return MakeGmt(-8);
                case TimezoneOption.GMT_Minus7:  return MakeGmt(-7);
                case TimezoneOption.GMT_Minus6:  return MakeGmt(-6);
                case TimezoneOption.GMT_Minus5:  return MakeGmt(-5);
                case TimezoneOption.GMT_Minus4:  return MakeGmt(-4);
                case TimezoneOption.GMT_Minus3:  return MakeGmt(-3);
                case TimezoneOption.GMT_Minus2:  return MakeGmt(-2);
                case TimezoneOption.GMT_Minus1:  return MakeGmt(-1);
                case TimezoneOption.GMT_0:        return MakeGmt(0);
                case TimezoneOption.GMT_Plus1:   return MakeGmt(1);
                case TimezoneOption.GMT_Plus2:   return MakeGmt(2);
                case TimezoneOption.GMT_Plus3:   return MakeGmt(3);
                case TimezoneOption.GMT_Plus4:   return MakeGmt(4);
                case TimezoneOption.GMT_Plus5:   return MakeGmt(5);
                case TimezoneOption.GMT_Plus6:   return MakeGmt(6);
                case TimezoneOption.GMT_Plus7:   return MakeGmt(7);
                case TimezoneOption.GMT_Plus8:   return MakeGmt(8);  // HKT
                case TimezoneOption.GMT_Plus9:   return MakeGmt(9);
                case TimezoneOption.GMT_Plus10:  return MakeGmt(10);
                case TimezoneOption.GMT_Plus11:  return MakeGmt(11);
                case TimezoneOption.GMT_Plus12:  return MakeGmt(12);
                default:                          return TimeZoneInfo.Utc;
            }
        }

        private static TimeZoneInfo MakeGmt(int hours)
            => TimeZoneInfo.CreateCustomTimeZone(
                "GMT" + (hours >= 0 ? "+" : "") + hours,
                TimeSpan.FromHours(hours),
                "GMT" + (hours >= 0 ? "+" : "") + hours,
                "GMT" + (hours >= 0 ? "+" : "") + hours);

        // ── GetAvgBarMins (weekend-gap-filtered) ──────────────────────────────

        private double GetAvgBarMins()
        {
            if (Bars.Count < 2) return 1.0;
            int sample = Math.Min(200, Bars.Count - 1);

            double minGap = double.MaxValue;
            for (int i = Bars.Count - sample; i < Bars.Count; i++)
            {
                double g = (Bars.OpenTimes[i] - Bars.OpenTimes[i - 1]).TotalMinutes;
                if (g > 0 && g < minGap) minGap = g;
            }
            if (minGap == double.MaxValue) return 1.0;

            double sum = 0; int count = 0;
            double threshold = minGap * 4.0;
            for (int i = Bars.Count - sample; i < Bars.Count; i++)
            {
                double g = (Bars.OpenTimes[i] - Bars.OpenTimes[i - 1]).TotalMinutes;
                if (g <= threshold) { sum += g; count++; }
            }
            return count > 0 ? sum / count : minGap;
        }

        // ── Misc ──────────────────────────────────────────────────────────────

        private static LineStyle ToLineStyle(LineStyleChoice c)
        {
            switch (c)
            {
                case LineStyleChoice.Dashed: return LineStyle.Lines;
                case LineStyleChoice.Dotted: return LineStyle.Dots;
                default:                    return LineStyle.Solid;
            }
        }

        private string NextId()
            => (_objId++).ToString(CultureInfo.InvariantCulture);

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
