using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    internal static class TimeFrameCompatExtensions
    {
        public static TimeSpan ToTimeSpan(this TimeFrame tf)
        {
            if (tf == TimeFrame.Minute)  return TimeSpan.FromMinutes(1);
            if (tf == TimeFrame.Minute5) return TimeSpan.FromMinutes(5);
            if (tf == TimeFrame.Minute10) return TimeSpan.FromMinutes(10);
            if (tf == TimeFrame.Minute15) return TimeSpan.FromMinutes(15);
            if (tf == TimeFrame.Minute30) return TimeSpan.FromMinutes(30);
            if (tf == TimeFrame.Hour)    return TimeSpan.FromHours(1);
            if (tf == TimeFrame.Hour4)   return TimeSpan.FromHours(4);
            if (tf == TimeFrame.Hour8)   return TimeSpan.FromHours(8);
            if (tf == TimeFrame.Hour12)  return TimeSpan.FromHours(12);
            if (tf == TimeFrame.Daily)   return TimeSpan.FromDays(1);
            if (tf == TimeFrame.Weekly)  return TimeSpan.FromDays(7);
            if (tf == TimeFrame.Monthly) return TimeSpan.FromDays(30);
            return TimeSpan.FromMinutes(1);
        }
    }

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICTMTFOrderBlockWicksMK : Indicator
    {
        public enum MitigationMode
        {
            Normal  = 1,
            Dynamic = 2,
            None    = 3,
            Half    = 4
        }

        public enum MitigationType
        {
            Wicks = 1,
            Body  = 2
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Inner types
        // ─────────────────────────────────────────────────────────────────────

        private sealed class ObZone
        {
            public string         Id;
            public string         LabelId;
            public string         TfLabel;
            public bool           IsBull;
            public double         Top;
            public double         Bottom;
            // NEW: tracking fields
            public bool           IsMitigated;        // marked true when mitigation condition fires
            public int            CreationChartIndex; // chart bar index when zone was first detected
            public ChartRectangle Box;
            public ChartText      Label;
        }

        private sealed class TfConfig
        {
            public string         Key;
            public string         Label;
            public Bars           SourceBars;
            public int            MaxCount;
            public List<ObZone>   Bulls   = new List<ObZone>();
            public List<ObZone>   Bears   = new List<ObZone>();
            public bool           NewBull;
            public bool           NewBear;
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Parameters — General
        // ─────────────────────────────────────────────────────────────────────

        [Parameter("Only Market Hours",      DefaultValue = false,                Group = "General")]
        public bool OnlyMktHrs { get; set; }

        [Parameter("Detection Method",       DefaultValue = MitigationType.Body,  Group = "General")]
        public MitigationType DetectionMethodInput { get; set; }

        [Parameter("Show Labels",            DefaultValue = true,                 Group = "General")]
        public bool ShowLabels { get; set; }

        [Parameter("Show Time On Labels",    DefaultValue = false,                Group = "General")]
        public bool ShowTimeOnLabels { get; set; }

        [Parameter("Timezone Offset",        DefaultValue = -5.0, MinValue = -12, MaxValue = 14, Step = 0.5, Group = "General")]
        public double HoursOffsetInput { get; set; }

        [Parameter("OB Offset to Right",     DefaultValue = 10, MinValue = 1, MaxValue = 100, Group = "General")]
        public int LabelShift { get; set; }

        [Parameter("Label Colour",           DefaultValue = "#FFFFA500",          Group = "General")]
        public Color LabelColor { get; set; }

        [Parameter("Incursion Alerts",       DefaultValue = true,                 Group = "General")]
        public bool IncursionAlerts { get; set; }

        [Parameter("Incursion %",            DefaultValue = 20.0, MinValue = 0, MaxValue = 100, Group = "General")]
        public double IncursionPct { get; set; }

        [Parameter("Mitigation Action",      DefaultValue = MitigationMode.Normal, Group = "General")]
        public MitigationMode MitigationActionInput { get; set; }

        [Parameter("Mitigation Type",        DefaultValue = MitigationType.Wicks, Group = "General")]
        public MitigationType MitigationTypeInput { get; set; }

        [Parameter("Change OB Color On Entry", DefaultValue = true,              Group = "General")]
        public bool EntryChangeColor { get; set; }

        [Parameter("Entry Bull",             DefaultValue = "#E6FFFFFF",          Group = "General")]
        public Color EntryBullColor { get; set; }

        [Parameter("Entry Bear",             DefaultValue = "#E6FFFFFF",          Group = "General")]
        public Color EntryBearColor { get; set; }

        [Parameter("Show Mitigated Text",    DefaultValue = false,                Group = "General")]
        public bool ShowMitigatedText { get; set; }

        // ── NEW toggles ───────────────────────────────────────────────────────

        /// <summary>
        /// When true (default), active (not yet mitigated) OB zones are drawn
        /// on the chart.  Turn off to hide all active zones while still seeing
        /// mitigated ones (if ShowMitigatedOBs is on).
        /// </summary>
        [Parameter("Show Historical OBs",    DefaultValue = true,                 Group = "General")]
        public bool ShowHistoricalOBs { get; set; }

        /// <summary>
        /// When true, a zone that reaches its mitigation condition is kept on
        /// the chart with the No-Mitigation colour instead of being deleted.
        /// Works for Normal, Dynamic, and Half mitigation modes.
        /// (None mode already keeps mitigated zones by design.)
        /// </summary>
        [Parameter("Show Mitigated OBs",     DefaultValue = false,                Group = "General")]
        public bool ShowMitigatedOBs { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        //  Parameters — Colors
        // ─────────────────────────────────────────────────────────────────────

        [Parameter("Bull OB Fill Color",     DefaultValue = "#CCFFFF00", Group = "Colors")]
        public Color BullObColor { get; set; }

        [Parameter("Bear OB Fill Color",     DefaultValue = "#CC0000FF", Group = "Colors")]
        public Color BearObColor { get; set; }

        [Parameter("Mitigated Bull OB Color", DefaultValue = "#D9FFFF00", Group = "Colors")]
        public Color MitigatedBullColor { get; set; }

        [Parameter("Mitigated Bear OB Color", DefaultValue = "#D9FF6600", Group = "Colors")]
        public Color MitigatedBearColor { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        //  Parameters — Enabled Timeframes
        // ─────────────────────────────────────────────────────────────────────

        [Parameter("Enable Current Timeframe", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableChartTf { get; set; }

        [Parameter("Enable 5 Minute",  DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable5m { get; set; }

        [Parameter("Enable 10 Minute", DefaultValue = true,  Group = "Enabled Timeframes")]
        public bool Enable10m { get; set; }

        [Parameter("Enable 15 Minute", DefaultValue = true,  Group = "Enabled Timeframes")]
        public bool Enable15m { get; set; }

        [Parameter("Enable 30 Minute", DefaultValue = true,  Group = "Enabled Timeframes")]
        public bool Enable30m { get; set; }

        [Parameter("Enable 1 Hour",    DefaultValue = true,  Group = "Enabled Timeframes")]
        public bool Enable1h { get; set; }

        [Parameter("Enable 4 Hour",    DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable4h { get; set; }

        [Parameter("Enable 8 Hour",    DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable8h { get; set; }

        [Parameter("Enable 12 Hour",   DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable12h { get; set; }

        [Parameter("Enable Daily",     DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableDaily { get; set; }

        [Parameter("Enable Weekly",    DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableWeekly { get; set; }

        [Parameter("Enable Monthly",   DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableMonthly { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        //  Parameters — Max OBs
        // ─────────────────────────────────────────────────────────────────────

        [Parameter("Max Current TF", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxChart { get; set; }

        [Parameter("Max 5 Min",  DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max5m { get; set; }

        [Parameter("Max 10 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max10m { get; set; }

        [Parameter("Max 15 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max15m { get; set; }

        [Parameter("Max 30 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max30m { get; set; }

        [Parameter("Max 1 Hr",   DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max1h { get; set; }

        [Parameter("Max 4 Hr",   DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max4h { get; set; }

        [Parameter("Max 8 Hr",   DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max8h { get; set; }

        [Parameter("Max 12 Hr",  DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max12h { get; set; }

        [Parameter("Max Daily",  DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxDaily { get; set; }

        [Parameter("Max Weekly", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxWeekly { get; set; }

        [Parameter("Max Monthly",DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxMonthly { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        //  Output series (cBot-readable signals)
        // ─────────────────────────────────────────────────────────────────────

        [Output("Bull OB Creation",      LineColor = "Lime",   PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BullCreationAlert { get; set; }

        [Output("Bear OB Creation",      LineColor = "Red",    PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BearCreationAlert { get; set; }

        [Output("BullOrBear OB Creation",LineColor = "Yellow", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BothCreationAlert { get; set; }

        // ─────────────────────────────────────────────────────────────────────
        //  State
        // ─────────────────────────────────────────────────────────────────────

        private readonly List<TfConfig> _tfs = new List<TfConfig>();
        private int _idCounter;

        // ─────────────────────────────────────────────────────────────────────
        //  Initialization
        // ─────────────────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            RegisterTf("chart", GetChartTimeframeLabel(), Bars.TimeFrame, EnableChartTf, MaxChart,  true);
            RegisterTf("5",     "5 Min",   TimeFrame.Minute5,  Enable5m,     Max5m,    false);
            RegisterTf("10",    "10 Min",  TimeFrame.Minute10, Enable10m,    Max10m,   false);
            RegisterTf("15",    "15 Min",  TimeFrame.Minute15, Enable15m,    Max15m,   false);
            RegisterTf("30",    "30 Min",  TimeFrame.Minute30, Enable30m,    Max30m,   false);
            RegisterTf("60",    "1 Hr",    TimeFrame.Hour,     Enable1h,     Max1h,    false);
            RegisterTf("240",   "4 Hr",    TimeFrame.Hour4,    Enable4h,     Max4h,    false);
            RegisterTf("480",   "8 Hr",    TimeFrame.Hour8,    Enable8h,     Max8h,    false);
            RegisterTf("720",   "12Hr",    TimeFrame.Hour12,   Enable12h,    Max12h,   false);
            RegisterTf("D",     "Daily",   TimeFrame.Daily,    EnableDaily,  MaxDaily, false);
            RegisterTf("W",     "Weekly",  TimeFrame.Weekly,   EnableWeekly, MaxWeekly,false);
            RegisterTf("M",     "Monthly", TimeFrame.Monthly,  EnableMonthly,MaxMonthly,false);

            var totalMax = _tfs.Where(tf => tf.Key != "30" && tf.Key != "720").Sum(tf => tf.MaxCount);
            if (totalMax > 500)
                Chart.DrawStaticText("mk_ob_error",
                    "MTF OB INDICATOR ERROR\n\nMax Number of OBs exceeded, please change settings.",
                    VerticalAlignment.Bottom, HorizontalAlignment.Right, Color.Red);
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Main calculate loop
        // ─────────────────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            if (index < 2) return;

            var now      = Bars.OpenTimes[index];
            var inSession = IsInSession(now);
            var low      = Bars.LowPrices[index];
            var high     = Bars.HighPrices[index];
            var close    = Bars.ClosePrices[index];
            var lastLow  = Bars.LowPrices[index - 1];
            var lastHigh = Bars.HighPrices[index - 1];

            bool newBullChart=false,newBull5=false,newBull10=false,newBull15=false,
                 newBull30=false,newBull1h=false,newBull4h=false,newBull8h=false,
                 newBull12h=false,newBullD=false,newBullW=false,newBullM=false;
            bool newBearChart=false,newBear5=false,newBear10=false,newBear15=false,
                 newBear30=false,newBear1h=false,newBear4h=false,newBear8h=false,
                 newBear12h=false,newBearD=false,newBearW=false,newBearM=false;

            for (int t = 0; t < _tfs.Count; t++)
            {
                var tf = _tfs[t];
                tf.NewBull = false;
                tf.NewBear = false;

                var srcIdx         = FindBarIndexAtOrBefore(tf.SourceBars, now);
                var currClosedIdx  = srcIdx - 1;
                var prevIdx        = currClosedIdx - 1;
                var prev2Idx       = currClosedIdx - 2;

                if (prev2Idx < 0) continue;

                var open1  = tf.SourceBars.OpenPrices[prevIdx];
                var close1 = tf.SourceBars.ClosePrices[prevIdx];
                var op     = tf.SourceBars.OpenPrices[currClosedIdx];
                var cl     = tf.SourceBars.ClosePrices[currClosedIdx];
                var high1  = tf.SourceBars.HighPrices[prevIdx];
                var low1   = tf.SourceBars.LowPrices[prevIdx];
                var high2  = tf.SourceBars.HighPrices[prev2Idx];

                bool canDetect = (OnlyMktHrs && inSession) || !OnlyMktHrs;
                var isNewBull  = canDetect && IsBullDetected(open1, close1, op, cl, high1);
                var isNewBear  = canDetect && IsBearDetected(open1, close1, op, cl, low1);

                if (isNewBull)
                {
                    if (tf.Bulls.Count > tf.MaxCount)
                        DeleteZone(tf.Bulls, 0);
                    if (!HasDuplicateByTop(tf.Bulls, high1))
                    {
                        CreateZone(tf, true, high1, open1, low1, high2, index, now);
                        tf.NewBull = true;
                    }
                }

                if (isNewBear)
                {
                    if (tf.Bears.Count > tf.MaxCount)
                        DeleteZone(tf.Bears, 0);
                    if (!HasDuplicateByTop(tf.Bears, open1))
                    {
                        CreateZone(tf, false, open1, low1, low1, high2, index, now);
                        tf.NewBear = true;
                    }
                }

                if (tf.Bulls.Count > 0)
                    UpdateBullZones(tf, index, low, close, lastLow, now);
                if (tf.Bears.Count > 0)
                    UpdateBearZones(tf, index, high, close, lastHigh, now);

                switch (tf.Key)
                {
                    case "chart": newBullChart=tf.NewBull; newBearChart=tf.NewBear; break;
                    case "5":     newBull5=tf.NewBull;     newBear5=tf.NewBear;     break;
                    case "10":    newBull10=tf.NewBull;    newBear10=tf.NewBear;    break;
                    case "15":    newBull15=tf.NewBull;    newBear15=tf.NewBear;    break;
                    case "30":    newBull30=tf.NewBull;    newBear30=tf.NewBear;    break;
                    case "60":    newBull1h=tf.NewBull;    newBear1h=tf.NewBear;    break;
                    case "240":   newBull4h=tf.NewBull;    newBear4h=tf.NewBear;    break;
                    case "480":   newBull8h=tf.NewBull;    newBear8h=tf.NewBear;    break;
                    case "720":   newBull12h=tf.NewBull;   newBear12h=tf.NewBear;   break;
                    case "D":     newBullD=tf.NewBull;     newBearD=tf.NewBear;     break;
                    case "W":     newBullW=tf.NewBull;     newBearW=tf.NewBear;     break;
                    case "M":     newBullM=tf.NewBull;     newBearM=tf.NewBear;     break;
                }
            }

            var anyBull = newBullChart||newBull5||newBull10||newBull15||newBull1h||
                          newBull4h||newBull8h||newBullD||newBullW||newBullM;
            var anyBear = newBearChart||newBear5||newBear10||newBear15||newBear1h||
                          newBear4h||newBear8h||newBearD||newBearW||newBearM;
            var markerPrice = (high + low) / 2.0;

            BullCreationAlert[index]  = anyBull            ? markerPrice : double.NaN;
            BearCreationAlert[index]  = anyBear            ? markerPrice : double.NaN;
            BothCreationAlert[index]  = (anyBull||anyBear) ? markerPrice : double.NaN;
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Registration & detection helpers
        // ─────────────────────────────────────────────────────────────────────

        private void RegisterTf(string key, string label, TimeFrame timeframe,
            bool enabled, int maxCount, bool isChartTf)
        {
            if (!enabled) return;
            if (isChartTf && !NotCurrentTimeframeEqualEnabledTfs()) return;

            var cfg = new TfConfig
            {
                Key        = key,
                Label      = label,
                SourceBars = timeframe == Bars.TimeFrame ? Bars : MarketData.GetBars(timeframe),
                MaxCount   = maxCount
            };
            EnsureSourceHistory(cfg.SourceBars);
            _tfs.Add(cfg);
        }

        private void EnsureSourceHistory(Bars sourceBars)
        {
            if (sourceBars == null || sourceBars == Bars) return;
            const int maxHistoryLoadIterations = 40;
            var lastCount = sourceBars.Count;
            for (int i = 0; i < maxHistoryLoadIterations; i++)
            {
                var loaded = sourceBars.LoadMoreHistory();
                if (loaded <= 0) break;
                if (sourceBars.Count <= lastCount) break;
                lastCount = sourceBars.Count;
            }
        }

        private bool NotCurrentTimeframeEqualEnabledTfs()
        {
            if (Bars.TimeFrame == TimeFrame.Minute5)  return !Enable5m;
            if (Bars.TimeFrame == TimeFrame.Minute10) return !Enable10m;
            if (Bars.TimeFrame == TimeFrame.Minute15) return !Enable15m;
            if (Bars.TimeFrame == TimeFrame.Minute30) return !Enable30m;
            if (Bars.TimeFrame == TimeFrame.Hour)     return !Enable1h;
            if (Bars.TimeFrame == TimeFrame.Hour4)    return !Enable4h;
            if (Bars.TimeFrame == TimeFrame.Hour8)    return !Enable8h;
            if (Bars.TimeFrame == TimeFrame.Hour12)   return !Enable12h;
            if (Bars.TimeFrame == TimeFrame.Daily)    return !EnableDaily;
            if (Bars.TimeFrame == TimeFrame.Weekly)   return !EnableWeekly;
            if (Bars.TimeFrame == TimeFrame.Monthly)  return !EnableMonthly;
            return true;
        }

        private bool IsBullDetected(double open1, double close1, double op, double cl, double high1)
        {
            if (DetectionMethodInput == MitigationType.Body)
                return open1 > close1 && op < cl && cl > high1;
            return op < high1;
        }

        private bool IsBearDetected(double open1, double close1, double op, double cl, double low1)
        {
            if (DetectionMethodInput == MitigationType.Body)
                return open1 < close1 && op > cl && cl < low1;
            return op < low1;
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Zone creation
        //  CHANGE: rectangle/label are only drawn when ShowHistoricalOBs is on.
        //  CreationChartIndex is always stored so EnsureMitigatedBox can use it.
        // ─────────────────────────────────────────────────────────────────────

        private void CreateZone(TfConfig tf, bool isBull, double top, double bottom,
            double low1, double high2, int index, DateTime now)
        {
            var id       = $"ob_{tf.Key}_{(isBull?"bull":"bear")}_{_idCounter++}";
            var labelId  = id + "_lbl";
            var zoneTop    = Math.Max(top, bottom);
            var zoneBottom = Math.Min(top, bottom);

            if (double.IsNaN(zoneTop) || double.IsNaN(zoneBottom) ||
                double.IsInfinity(zoneTop) || double.IsInfinity(zoneBottom)) return;

            ChartRectangle rect  = null;
            ChartText      label = null;

            // Only draw if ShowHistoricalOBs is on
            if (ShowHistoricalOBs)
            {
                var borderColor = isBull
                    ? Color.FromArgb(0, Color.Yellow)
                    : Color.FromArgb(0, Color.Blue);
                var fillColor = isBull ? BullObColor : BearObColor;

                // Original positioning: box floats ahead of current bar (unchanged from original)
                var leftTime  = ShiftTime(index, 20);
                var rightTime = ShiftTime(index, 200);

                rect = Chart.DrawRectangle(id, leftTime, zoneTop, rightTime, zoneBottom,
                    borderColor, 1, LineStyle.DotsRare);
                rect.IsFilled = true;
                rect.Color    = fillColor;

                if (ShowLabels)
                {
                    var text = tf.Label + (isBull ? " OB BULL" : " OB BEAR");
                    if (ShowTimeOnLabels)
                        text += " " + now.AddHours(HoursOffsetInput).ToString("HH:mm MM/dd/yy");
                    var y = (high2 + low1) / 2.0;
                    label = Chart.DrawText(labelId, text, ShiftTime(index, LabelShift), y, LabelColor);
                }
            }

            var zone = new ObZone
            {
                Id                 = id,
                LabelId            = labelId,
                TfLabel            = tf.Label,
                IsBull             = isBull,
                Top                = zoneTop,
                Bottom             = zoneBottom,
                IsMitigated        = false,
                CreationChartIndex = index,   // NEW: remembered for mitigated-box creation
                Box                = rect,
                Label              = label
            };

            if (isBull) tf.Bulls.Add(zone);
            else        tf.Bears.Add(zone);
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Bull-zone update
        //  CHANGES:
        //    1. Already-mitigated zones: only extend their right edge, then skip.
        //    2. On delete: if ShowMitigatedOBs, preserve with NoMitColor instead
        //       of deleting (call EnsureMitigatedBox so zones without a rect get one).
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateBullZones(TfConfig tf, int index, double low, double close,
            double lastLow, DateTime now)
        {
            var bullIncursionPrinted = false;

            for (int i = tf.Bulls.Count - 1; i >= 0; i--)
            {
                var z   = tf.Bulls[i];

                // ── Already-mitigated zones: box is frozen at break bar, don't touch ──
                if (z.IsMitigated)
                    continue;

                var mid            = (z.Top + z.Bottom) / 2.0;
                var threshold      = z.Top - (IncursionPct / 100.0) * (z.Top - z.Bottom);
                var lowUnderTop    = low   < z.Top;
                var lowUnderBottom = low   < z.Bottom;
                var lowUnderMid    = low   < mid;
                var closeUnderTop  = close < z.Top;
                var closeUnderBottom= close< z.Bottom;
                var closeUnderMid  = low   < mid;   // exact source parity
                var intrusion      = low < threshold && lastLow > threshold;

                if ((MitigationActionInput == MitigationMode.Normal ||
                     MitigationActionInput == MitigationMode.None) &&
                    intrusion && IncursionAlerts && !bullIncursionPrinted)
                {
                    Print("Bull OB Wick Incursion {0}", tf.Label);
                    bullIncursionPrinted = true;
                }

                if (EntryChangeColor && lowUnderTop && z.Box != null)
                    z.Box.Color = EntryBullColor;

                // Box extension must always happen when ShowLabels is on (original behaviour).
                // The null-check inside SetLabelAndBoxPosition handles zones without a rect.
                if (ShowLabels)
                    SetLabelAndBoxPosition(z, index);

                // Dynamic: shrink top boundary
                if (MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeUnderTop)
                    {
                        z.Top = close;
                        if (ShowLabels) SetLabelY(z, index, close, z.Bottom);
                    }
                    else if (lowUnderTop)
                    {
                        z.Top = low;
                        if (ShowLabels) SetLabelY(z, index, low, z.Bottom);
                    }
                    if (z.Box != null) RedrawZone(z, index);
                }

                // None mode: colour zone when fully penetrated (existing behaviour)
                if (MitigationActionInput == MitigationMode.None)
                {
                    bool penetratedNone = MitigationTypeInput == MitigationType.Body
                        ? closeUnderBottom : lowUnderBottom;
                    if (penetratedNone && !z.IsMitigated)
                    {
                        z.IsMitigated = true;
                        if (z.Box != null)
                        {
                            var leftIdx  = Math.Max(0, Math.Min(z.CreationChartIndex, Bars.Count - 1));
                            var rightIdx = Math.Min(index + 1, Bars.Count - 1);
                            z.Box.Time1  = Bars.OpenTimes[leftIdx];
                            z.Box.Time2  = Bars.OpenTimes[rightIdx];
                            z.Box.Color  = MitigatedBullColor;
                        }
                        if (ShowLabels && ShowMitigatedText && z.Label != null &&
                            !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                    continue; // None mode never removes zones
                }

                // Deletion condition for Normal / Dynamic / Half
                bool delete = false;
                if (MitigationActionInput == MitigationMode.Normal ||
                    MitigationActionInput == MitigationMode.Dynamic)
                {
                    delete = MitigationTypeInput == MitigationType.Body
                        ? closeUnderBottom : lowUnderBottom;
                }
                else if (MitigationActionInput == MitigationMode.Half)
                {
                    delete = MitigationTypeInput == MitigationType.Body
                        ? closeUnderMid : lowUnderMid;
                }

                if (delete)
                {
                    if (ShowMitigatedOBs)
                    {
                        z.IsMitigated = true;

                        // If the zone already has a box (ShowHistoricalOBs was on), it is
                        // currently floating ahead of the current bar (ShiftTime pattern).
                        // Reposition it to span creation bar → mitigation bar so it appears
                        // as a fixed historical rectangle, matching the FVG indicator's style.
                        if (z.Box != null)
                        {
                            var leftIdx  = Math.Max(0, Math.Min(z.CreationChartIndex, Bars.Count - 1));
                            var rightIdx = Math.Min(index + 1, Bars.Count - 1);
                            z.Box.Time1  = Bars.OpenTimes[leftIdx];
                            z.Box.Time2  = Bars.OpenTimes[rightIdx];
                            z.Box.Color  = MitigatedBullColor;
                        }

                        // If no box exists yet (ShowHistoricalOBs was off), create one now.
                        EnsureMitigatedBox(z, index);

                        if (ShowLabels && ShowMitigatedText && z.Label != null &&
                            !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                    else
                    {
                        DeleteZone(tf.Bulls, i);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Bear-zone update  (symmetric to UpdateBullZones)
        // ─────────────────────────────────────────────────────────────────────

        private void UpdateBearZones(TfConfig tf, int index, double high, double close,
            double lastHigh, DateTime now)
        {
            var bearIncursionPrinted = false;

            for (int i = tf.Bears.Count - 1; i >= 0; i--)
            {
                var z = tf.Bears[i];

                // ── Already-mitigated zones: box is frozen at break bar, don't touch ──
                if (z.IsMitigated)
                    continue;

                var mid             = (z.Top + z.Bottom) / 2.0;
                var threshold       = z.Bottom + (IncursionPct / 100.0) * (z.Top - z.Bottom);
                var highOverTop     = high  > z.Top;
                var highOverBottom  = high  > z.Bottom;
                var highOverMid     = high  > mid;
                var closeOverTop    = close > z.Top;
                var closeOverBottom = close > z.Bottom;
                var closeOverMid    = close > mid;
                var intrusion       = high > threshold && lastHigh < threshold;

                if ((MitigationActionInput == MitigationMode.Normal ||
                     MitigationActionInput == MitigationMode.None) &&
                    intrusion && IncursionAlerts && !bearIncursionPrinted)
                {
                    Print("Bear OB Wick Incursion {0}", tf.Label);
                    bearIncursionPrinted = true;
                }

                if (EntryChangeColor && z.Box != null)
                {
                    if (highOverBottom) z.Box.Color = EntryBearColor;
                    else                z.Box.Color = BearObColor;
                }

                // Box extension must always happen when ShowLabels is on (original behaviour).
                if (ShowLabels)
                    SetLabelAndBoxPosition(z, index);

                // Dynamic: shrink bottom boundary
                if (MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeOverBottom)
                    {
                        var oldBottom = z.Bottom;
                        z.Bottom = close;
                        if (ShowLabels) SetLabelY(z, index, close, oldBottom);
                    }
                    else if (highOverBottom)
                    {
                        z.Bottom = high;
                        if (ShowLabels) SetLabelY(z, index, z.Top, high);
                    }
                    if (z.Box != null) RedrawZone(z, index);
                }

                // None mode: colour zone when fully penetrated (existing behaviour)
                if (MitigationActionInput == MitigationMode.None)
                {
                    bool penetratedNone = MitigationTypeInput == MitigationType.Body
                        ? closeOverTop : highOverTop;
                    if (penetratedNone && !z.IsMitigated)
                    {
                        z.IsMitigated = true;
                        if (z.Box != null)
                        {
                            var leftIdx  = Math.Max(0, Math.Min(z.CreationChartIndex, Bars.Count - 1));
                            var rightIdx = Math.Min(index + 1, Bars.Count - 1);
                            z.Box.Time1  = Bars.OpenTimes[leftIdx];
                            z.Box.Time2  = Bars.OpenTimes[rightIdx];
                            z.Box.Color  = MitigatedBearColor;
                        }
                        if (ShowLabels && ShowMitigatedText && z.Label != null &&
                            !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                    continue; // None mode never removes zones
                }

                // Deletion condition for Normal / Dynamic / Half
                bool delete = false;
                if (MitigationActionInput == MitigationMode.Normal ||
                    MitigationActionInput == MitigationMode.Dynamic)
                {
                    delete = MitigationTypeInput == MitigationType.Body
                        ? closeOverTop : highOverTop;
                }
                else if (MitigationActionInput == MitigationMode.Half)
                {
                    delete = MitigationTypeInput == MitigationType.Body
                        ? closeOverMid : highOverMid;
                }

                if (delete)
                {
                    if (ShowMitigatedOBs)
                    {
                        z.IsMitigated = true;

                        // Same repositioning as bull: float→historical fixed range.
                        if (z.Box != null)
                        {
                            var leftIdx  = Math.Max(0, Math.Min(z.CreationChartIndex, Bars.Count - 1));
                            var rightIdx = Math.Min(index + 1, Bars.Count - 1);
                            z.Box.Time1  = Bars.OpenTimes[leftIdx];
                            z.Box.Time2  = Bars.OpenTimes[rightIdx];
                            z.Box.Color  = MitigatedBearColor;
                        }

                        EnsureMitigatedBox(z, index);

                        if (ShowLabels && ShowMitigatedText && z.Label != null &&
                            !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                    else
                    {
                        DeleteZone(tf.Bears, i);
                    }
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Box / label helpers
        // ─────────────────────────────────────────────────────────────────────

        /// <summary>
        /// CHANGE: null-checks Box before accessing it.
        /// When ShowHistoricalOBs is off a zone has no rect until mitigation.
        /// </summary>
        private void SetLabelAndBoxPosition(ObZone zone, int index)
        {
            if (zone.Box == null) return;

            // Original behaviour: box floats ahead of current bar (unchanged)
            var left  = ShiftTime(index, LabelShift);
            zone.Box.Time1 = left;
            zone.Box.Time2 = ShiftTime(index, 200);

            SetLabelY(zone, index, zone.Top, zone.Bottom);
        }

        private void SetLabelY(ObZone zone, int index, double top, double bottom)
        {
            if (zone.Label == null) return;
            zone.Label.Time = ShiftTime(index, LabelShift);
            zone.Label.Y    = (top + bottom) / 2.0;
        }

        private void RedrawZone(ObZone zone, int index)
        {
            if (zone.Box == null) return;
            zone.Box.Y1    = zone.Top;
            zone.Box.Y2    = zone.Bottom;
            zone.Box.Time1 = ShiftTime(index, LabelShift);
            zone.Box.Time2 = ShiftTime(index, 200);
        }

        /// <summary>
        /// NEW: Creates the chart rectangle (and optionally label) for a zone
        /// that was originally created without drawing (ShowHistoricalOBs was off).
        /// Uses the zone's stored CreationChartIndex as the left edge so the box
        /// spans from when the OB was actually formed.
        /// No-op if the zone already has a Box.
        /// </summary>
        private void EnsureMitigatedBox(ObZone zone, int currentIndex)
        {
            if (zone.Box != null) return; // already has a rect

            var leftIdx   = Math.Max(0, Math.Min(zone.CreationChartIndex, Bars.Count - 1));
            var leftTime  = Bars.OpenTimes[leftIdx];

            // RIGHT EDGE: frozen at the bar AFTER the break bar — same as LuxAlgo's
            // Bars.OpenTimes[Math.Min(index + 1, Bars.Count - 1)] at deletion time.
            // This means the box visually ends at the break candle and never extends further.
            var rightIdx  = Math.Min(currentIndex + 1, Bars.Count - 1);
            var rightTime = Bars.OpenTimes[rightIdx];

            var borderColor = zone.IsBull
                ? Color.FromArgb(0, Color.Yellow)
                : Color.FromArgb(0, Color.Blue);

            var rect = Chart.DrawRectangle(zone.Id, leftTime, zone.Top, rightTime, zone.Bottom,
                borderColor, 1, LineStyle.DotsRare);
            rect.IsFilled = true;
            rect.Color    = zone.IsBull ? MitigatedBullColor : MitigatedBearColor;
            zone.Box      = rect;

            if (ShowLabels && zone.Label == null)
            {
                var text = zone.TfLabel + (zone.IsBull ? " OB BULL" : " OB BEAR") + " Mitigated";
                var y    = (zone.Top + zone.Bottom) / 2.0;
                zone.Label = Chart.DrawText(zone.LabelId, text, leftTime, y, LabelColor);
            }
        }

        private bool HasDuplicateByTop(List<ObZone> zones, double top)
        {
            for (int i = zones.Count - 1; i >= 0; i--)
                if (zones[i].Top == top) return true;
            return false;
        }

        private void DeleteZone(List<ObZone> list, int index)
        {
            var zone = list[index];
            Chart.RemoveObject(zone.Id);
            if (zone.Label != null) Chart.RemoveObject(zone.LabelId);
            list.RemoveAt(index);
        }

        // ─────────────────────────────────────────────────────────────────────
        //  Bar-lookup + time utilities
        // ─────────────────────────────────────────────────────────────────────

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var i = bars.OpenTimes.GetIndexByTime(time);
            if (i >= 0) return i;
            for (int j = bars.Count - 1; j >= 0; j--)
                if (bars.OpenTimes[j] <= time) return j;
            return -1;
        }

        private DateTime ShiftTime(int chartIndex, int barsForward)
        {
            var baseTime    = Bars.OpenTimes[chartIndex];
            var stepSeconds = GetChartSeconds();
            return baseTime.AddSeconds(stepSeconds * barsForward);
        }

        private bool IsInSession(DateTime dt)
        {
            var minutes = dt.Hour * 60 + dt.Minute;
            return minutes >= 570 && minutes <= 960;
        }

        private string GetChartTimeframeLabel()
        {
            if (Bars.TimeFrame == TimeFrame.Daily)   return "Daily";
            if (Bars.TimeFrame == TimeFrame.Weekly)  return "Weekly";
            if (Bars.TimeFrame == TimeFrame.Monthly) return "Monthly";
            var seconds = GetChartSeconds();
            var minutes = seconds / 60.0;
            if (minutes > 59) return (minutes / 60.0).ToString("0.#") + " Hr";
            return minutes.ToString("0") + " Min";
        }

        private double GetChartSeconds()
        {
            if (Bars.Count > 1)
            {
                var delta = (Bars.OpenTimes[Bars.Count - 1] -
                             Bars.OpenTimes[Bars.Count - 2]).TotalSeconds;
                if (delta > 0) return delta;
            }
            return Bars.TimeFrame.ToTimeSpan().TotalSeconds;
        }
    }
}
