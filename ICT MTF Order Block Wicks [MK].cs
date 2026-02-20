using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class ICTMTFOrderBlockWicksMK : Indicator
    {
        public enum MitigationMode
        {
            Normal = 1,
            Dynamic = 2,
            None = 3,
            Half = 4
        }

        public enum MitigationType
        {
            Wicks = 1,
            Body = 2
        }

        private sealed class ObZone
        {
            public string Id;
            public string LabelId;
            public string TfLabel;
            public bool IsBull;
            public double Top;
            public double Bottom;
            public ChartRectangle Box;
            public ChartText Label;
        }

        private sealed class TfConfig
        {
            public string Key;
            public string Label;
            public Bars SourceBars;
            public int MaxCount;
            public List<ObZone> Bulls = new List<ObZone>();
            public List<ObZone> Bears = new List<ObZone>();
            public bool NewBull;
            public bool NewBear;
        }

        [Parameter("Only Market Hours", DefaultValue = false, Group = "General")]
        public bool OnlyMktHrs { get; set; }

        [Parameter("Detection Method", DefaultValue = MitigationType.Body, Group = "General")]
        public MitigationType DetectionMethodInput { get; set; }

        [Parameter("Show Labels", DefaultValue = true, Group = "General")]
        public bool ShowLabels { get; set; }

        [Parameter("Show Time On Labels", DefaultValue = false, Group = "General")]
        public bool ShowTimeOnLabels { get; set; }

        [Parameter("Timezone Offset", DefaultValue = -5.0, MinValue = -12, MaxValue = 14, Step = 0.5, Group = "General")]
        public double HoursOffsetInput { get; set; }

        [Parameter("OB Offset to Right", DefaultValue = 10, MinValue = 1, MaxValue = 100, Group = "General")]
        public int LabelShift { get; set; }

        [Parameter("Label Colour", DefaultValue = "#FFFFA500", Group = "General")]
        public Color LabelColor { get; set; }

        [Parameter("Incursion Alerts", DefaultValue = true, Group = "General")]
        public bool IncursionAlerts { get; set; }

        [Parameter("Incursion %", DefaultValue = 20.0, MinValue = 0, MaxValue = 100, Group = "General")]
        public double IncursionPct { get; set; }

        [Parameter("Mitigation Action", DefaultValue = MitigationMode.Normal, Group = "General")]
        public MitigationMode MitigationActionInput { get; set; }

        [Parameter("Mitigation Type", DefaultValue = MitigationType.Wicks, Group = "General")]
        public MitigationType MitigationTypeInput { get; set; }

        [Parameter("Change OB Color On Entry", DefaultValue = true, Group = "General")]
        public bool EntryChangeColor { get; set; }

        [Parameter("Entry Bull", DefaultValue = "#E6FFFFFF", Group = "General")]
        public Color EntryBullColor { get; set; }

        [Parameter("Entry Bear", DefaultValue = "#E6FFFFFF", Group = "General")]
        public Color EntryBearColor { get; set; }

        [Parameter("Show Mitigated Text", DefaultValue = false, Group = "General")]
        public bool ShowMitigatedText { get; set; }

        [Parameter("Bull OB Fill Color", DefaultValue = "#CCFFFF00", Group = "Colors")]
        public Color BullObColor { get; set; }

        [Parameter("Bear OB Fill Color", DefaultValue = "#CC0000FF", Group = "Colors")]
        public Color BearObColor { get; set; }

        [Parameter("No-Mitigation Fill Color", DefaultValue = "#D9FFFF00", Group = "Colors")]
        public Color NoMitColor { get; set; }

        [Parameter("Enable Current Timeframe", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableChartTf { get; set; }

        [Parameter("Enable 5 Minute", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable5m { get; set; }

        [Parameter("Enable 10 Minute", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable10m { get; set; }

        [Parameter("Enable 15 Minute", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable15m { get; set; }

        [Parameter("Enable 30 Minute", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable30m { get; set; }

        [Parameter("Enable 1 Hour", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable1h { get; set; }

        [Parameter("Enable 4 Hour", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable4h { get; set; }

        [Parameter("Enable 8 Hour", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable8h { get; set; }

        [Parameter("Enable 12 Hour", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable12h { get; set; }

        [Parameter("Enable Daily", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableDaily { get; set; }

        [Parameter("Enable Weekly", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableWeekly { get; set; }

        [Parameter("Enable Monthly", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableMonthly { get; set; }

        [Parameter("Max Current TF", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxChart { get; set; }

        [Parameter("Max 5 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max5m { get; set; }

        [Parameter("Max 10 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max10m { get; set; }

        [Parameter("Max 15 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max15m { get; set; }

        [Parameter("Max 30 Min", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max30m { get; set; }

        [Parameter("Max 1 Hr", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max1h { get; set; }

        [Parameter("Max 4 Hr", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max4h { get; set; }

        [Parameter("Max 8 Hr", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max8h { get; set; }

        [Parameter("Max 12 Hr", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int Max12h { get; set; }

        [Parameter("Max Daily", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxDaily { get; set; }

        [Parameter("Max Weekly", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxWeekly { get; set; }

        [Parameter("Max Monthly", DefaultValue = 8, MinValue = 1, Group = "Max OBs")]
        public int MaxMonthly { get; set; }

        [Output("Bull OB Creation", LineColor = "Lime", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BullCreationAlert { get; set; }

        [Output("Bear OB Creation", LineColor = "Red", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BearCreationAlert { get; set; }

        [Output("BullOrBear OB Creation", LineColor = "Yellow", PlotType = PlotType.DiscontinuousLine, Thickness = 1)]
        public IndicatorDataSeries BothCreationAlert { get; set; }

        private readonly List<TfConfig> _tfs = new List<TfConfig>();
        private int _idCounter;

        protected override void Initialize()
        {
            RegisterTf("chart", GetChartTimeframeLabel(), Bars.TimeFrame, EnableChartTf, MaxChart, true);
            RegisterTf("5", "5 Min", TimeFrame.Minute5, Enable5m, Max5m, false);
            RegisterTf("10", "10 Min", TimeFrame.Minute10, Enable10m, Max10m, false);
            RegisterTf("15", "15 Min", TimeFrame.Minute15, Enable15m, Max15m, false);
            RegisterTf("30", "30 Min", TimeFrame.Minute30, Enable30m, Max30m, false);
            RegisterTf("60", "1 Hr", TimeFrame.Hour, Enable1h, Max1h, false);
            RegisterTf("240", "4 Hr", TimeFrame.Hour4, Enable4h, Max4h, false);
            RegisterTf("480", "8 Hr", TimeFrame.Hour8, Enable8h, Max8h, false);
            RegisterTf("720", "12Hr", TimeFrame.Hour12, Enable12h, Max12h, false);
            RegisterTf("D", "Daily", TimeFrame.Daily, EnableDaily, MaxDaily, false);
            RegisterTf("W", "Weekly", TimeFrame.Weekly, EnableWeekly, MaxWeekly, false);
            RegisterTf("M", "Monthly", TimeFrame.Monthly, EnableMonthly, MaxMonthly, false);

            // Keep parity with source's total calculation (it omits 30m and 12h).
            var totalMax = MaxChart + Max5m + Max10m + Max15m + Max1h + Max4h + Max8h + MaxDaily + MaxWeekly + MaxMonthly;
            if (totalMax > 500)
                Chart.DrawStaticText("mk_ob_error", "MTF OB INDICATOR ERROR\n\nMax Number of OBs exceeded, please change settings.", VerticalAlignment.Bottom, HorizontalAlignment.Right, Color.Red);
        }

        public override void Calculate(int index)
        {
            if (index < 2)
                return;

            var now = Bars.OpenTimes[index];
            var inSession = IsInSession(now);
            var low = Bars.LowPrices[index];
            var high = Bars.HighPrices[index];
            var close = Bars.ClosePrices[index];
            var lastLow = Bars.LowPrices[index - 1];
            var lastHigh = Bars.HighPrices[index - 1];

            bool newBullChart = false, newBull5 = false, newBull10 = false, newBull15 = false, newBull30 = false, newBull1h = false, newBull4h = false, newBull8h = false, newBull12h = false, newBullD = false, newBullW = false, newBullM = false;
            bool newBearChart = false, newBear5 = false, newBear10 = false, newBear15 = false, newBear30 = false, newBear1h = false, newBear4h = false, newBear8h = false, newBear12h = false, newBearD = false, newBearW = false, newBearM = false;

            for (int t = 0; t < _tfs.Count; t++)
            {
                var tf = _tfs[t];
                tf.NewBull = false;
                tf.NewBear = false;

                var srcIdx = FindBarIndexAtOrBefore(tf.SourceBars, now);
                if (srcIdx < 2)
                    continue;

                var open1 = tf.SourceBars.OpenPrices[srcIdx - 1];
                var close1 = tf.SourceBars.ClosePrices[srcIdx - 1];
                var op = tf.SourceBars.OpenPrices[srcIdx];
                var cl = tf.SourceBars.ClosePrices[srcIdx];
                var high1 = tf.SourceBars.HighPrices[srcIdx - 1];
                var low1 = tf.SourceBars.LowPrices[srcIdx - 1];
                var high2 = tf.SourceBars.HighPrices[srcIdx - 2];

                bool canDetect = (OnlyMktHrs && inSession) || !OnlyMktHrs;
                var isNewBull = canDetect && IsBullDetected(open1, close1, op, cl, high1);
                var isNewBear = canDetect && IsBearDetected(open1, close1, op, cl, low1);

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
                    case "chart": newBullChart = tf.NewBull; newBearChart = tf.NewBear; break;
                    case "5": newBull5 = tf.NewBull; newBear5 = tf.NewBear; break;
                    case "10": newBull10 = tf.NewBull; newBear10 = tf.NewBear; break;
                    case "15": newBull15 = tf.NewBull; newBear15 = tf.NewBear; break;
                    case "30": newBull30 = tf.NewBull; newBear30 = tf.NewBear; break;
                    case "60": newBull1h = tf.NewBull; newBear1h = tf.NewBear; break;
                    case "240": newBull4h = tf.NewBull; newBear4h = tf.NewBear; break;
                    case "480": newBull8h = tf.NewBull; newBear8h = tf.NewBear; break;
                    case "720": newBull12h = tf.NewBull; newBear12h = tf.NewBear; break;
                    case "D": newBullD = tf.NewBull; newBearD = tf.NewBear; break;
                    case "W": newBullW = tf.NewBull; newBearW = tf.NewBear; break;
                    case "M": newBullM = tf.NewBull; newBearM = tf.NewBear; break;
                }
            }

            // Parity with Pine alertcondition section: 30m and 12h are not included there.
            var anyBull = newBullChart || newBull5 || newBull10 || newBull15 || newBull1h || newBull4h || newBull8h || newBullD || newBullW || newBullM;
            var anyBear = newBearChart || newBear5 || newBear10 || newBear15 || newBear1h || newBear4h || newBear8h || newBearD || newBearW || newBearM;
            BullCreationAlert[index] = anyBull ? 1.0 : double.NaN;
            BearCreationAlert[index] = anyBear ? 1.0 : double.NaN;
            BothCreationAlert[index] = (anyBull || anyBear) ? 1.0 : double.NaN;
        }

        private void RegisterTf(string key, string label, TimeFrame timeframe, bool enabled, int maxCount, bool isChartTf)
        {
            if (!enabled)
                return;

            // Pine parity: when current timeframe processing is enabled, it is skipped
            // if the same timeframe is also explicitly enabled in the MTF list.
            if (isChartTf && !NotCurrentTimeframeEqualEnabledTfs())
                return;

            _tfs.Add(new TfConfig
            {
                Key = key,
                Label = label,
                SourceBars = timeframe == Bars.TimeFrame ? Bars : MarketData.GetBars(timeframe),
                MaxCount = maxCount
            });
        }

        private bool NotCurrentTimeframeEqualEnabledTfs()
        {
            if (Bars.TimeFrame == TimeFrame.Minute5)
                return !Enable5m;
            if (Bars.TimeFrame == TimeFrame.Minute10)
                return !Enable10m;
            if (Bars.TimeFrame == TimeFrame.Minute15)
                return !Enable15m;
            if (Bars.TimeFrame == TimeFrame.Minute30)
                return !Enable30m;
            if (Bars.TimeFrame == TimeFrame.Hour)
                return !Enable1h;
            if (Bars.TimeFrame == TimeFrame.Hour4)
                return !Enable4h;
            if (Bars.TimeFrame == TimeFrame.Hour8)
                return !Enable8h;
            if (Bars.TimeFrame == TimeFrame.Hour12)
                return !Enable12h;
            if (Bars.TimeFrame == TimeFrame.Daily)
                return !EnableDaily;
            if (Bars.TimeFrame == TimeFrame.Weekly)
                return !EnableWeekly;
            if (Bars.TimeFrame == TimeFrame.Monthly)
                return !EnableMonthly;
            return true;
        }

        private bool IsBullDetected(double open1, double close1, double op, double cl, double high1)
        {
            var fvgMethodBody = DetectionMethodInput == MitigationType.Body;
            if (fvgMethodBody)
                return open1 > close1 && op < cl && cl > high1;
            return op < high1;
        }

        private bool IsBearDetected(double open1, double close1, double op, double cl, double low1)
        {
            var fvgMethodBody = DetectionMethodInput == MitigationType.Body;
            if (fvgMethodBody)
                return open1 < close1 && op > cl && cl < low1;
            return op < low1;
        }

        private void CreateZone(TfConfig tf, bool isBull, double top, double bottom, double low1, double high2, int index, DateTime now)
        {
            var id = $"ob_{tf.Key}_{(isBull ? "bull" : "bear")}_{_idCounter++}";
            var labelId = id + "_lbl";

            var leftTime = ShiftTime(index, 20);
            var rightTime = ShiftTime(index, 200);
            var borderColor = isBull ? Color.FromArgb(0, Color.Yellow) : Color.FromArgb(0, Color.Blue);
            var fillColor = isBull ? BullObColor : BearObColor;

            var rect = Chart.DrawRectangle(id, leftTime, top, rightTime, bottom, borderColor, 1, LineStyle.DotsRare);
            rect.IsFilled = true;
            rect.Color = fillColor;

            ChartText label = null;
            if (ShowLabels)
            {
                var text = tf.Label + (isBull ? " OB BULL" : " OB BEAR");
                if (ShowTimeOnLabels)
                    text += " " + now.AddHours(HoursOffsetInput).ToString("HH:mm MM/dd/yy");

                // parity with Pine call: (_high1[1] + _low1) / 2
                var y = (high2 + low1) / 2.0;
                label = Chart.DrawText(labelId, text, ShiftTime(index, LabelShift), y, LabelColor);
            }

            var zone = new ObZone
            {
                Id = id,
                LabelId = labelId,
                TfLabel = tf.Label,
                IsBull = isBull,
                Top = top,
                Bottom = bottom,
                Box = rect,
                Label = label
            };

            if (isBull)
                tf.Bulls.Add(zone);
            else
                tf.Bears.Add(zone);
        }

        private void UpdateBullZones(TfConfig tf, int index, double low, double close, double lastLow, DateTime now)
        {
            var bullIncursionPrinted = false;
            for (int i = tf.Bulls.Count - 1; i >= 0; i--)
            {
                var z = tf.Bulls[i];
                var mid = (z.Top + z.Bottom) / 2.0;
                var threshold = z.Top - (IncursionPct / 100.0) * (z.Top - z.Bottom);

                var lowUnderTop = low < z.Top;
                var lowUnderBottom = low < z.Bottom;
                var lowUnderMid = low < mid;
                var closeUnderTop = close < z.Top;
                var closeUnderBottom = close < z.Bottom;
                var closeUnderMid = low < mid; // exact source parity
                var intrusion = low < threshold && lastLow > threshold;

                if ((MitigationActionInput == MitigationMode.Normal || MitigationActionInput == MitigationMode.None) && intrusion && IncursionAlerts && !bullIncursionPrinted)
                {
                    Print("Bull OB Wick Incursion {0}", tf.Label);
                    bullIncursionPrinted = true;
                }

                if (EntryChangeColor && lowUnderTop)
                    z.Box.Color = EntryBullColor;

                if (ShowLabels)
                    SetLabelAndBoxPosition(z, index);

                if (MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeUnderTop)
                    {
                        z.Top = close;
                        if (ShowLabels)
                            SetLabelY(z, index, close, z.Bottom);
                    }
                    else if (lowUnderTop)
                    {
                        z.Top = low;
                        if (ShowLabels)
                            SetLabelY(z, index, low, z.Bottom);
                    }
                    RedrawZone(z, index);
                }

                if (MitigationActionInput == MitigationMode.None)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeUnderBottom)
                        z.Box.Color = NoMitColor;
                    else if (lowUnderBottom)
                        z.Box.Color = NoMitColor;

                    if ((MitigationTypeInput == MitigationType.Body && closeUnderBottom) || lowUnderBottom)
                    {
                        if (ShowLabels && ShowMitigatedText && z.Label != null && !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                }

                var delete = false;
                if (MitigationActionInput == MitigationMode.Normal || MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeUnderBottom)
                        delete = true;
                    else if (lowUnderBottom)
                        delete = true;
                }
                else if (MitigationActionInput == MitigationMode.Half)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeUnderMid)
                        delete = true;
                    else if (lowUnderMid)
                        delete = true;
                }

                if (delete)
                    DeleteZone(tf.Bulls, i);
            }
        }

        private void UpdateBearZones(TfConfig tf, int index, double high, double close, double lastHigh, DateTime now)
        {
            var bearIncursionPrinted = false;
            for (int i = tf.Bears.Count - 1; i >= 0; i--)
            {
                var z = tf.Bears[i];
                var mid = (z.Top + z.Bottom) / 2.0;
                var threshold = z.Bottom + (IncursionPct / 100.0) * (z.Top - z.Bottom);

                var highOverTop = high > z.Top;
                var highOverBottom = high > z.Bottom;
                var highOverMid = high > mid;
                var closeOverTop = close > z.Top;
                var closeOverBottom = close > z.Bottom;
                var closeOverMid = close > mid;
                var intrusion = high > threshold && lastHigh < threshold;

                if ((MitigationActionInput == MitigationMode.Normal || MitigationActionInput == MitigationMode.None) && intrusion && IncursionAlerts && !bearIncursionPrinted)
                {
                    Print("Bear OB Wick Incursion {0}", tf.Label);
                    bearIncursionPrinted = true;
                }

                if (EntryChangeColor)
                {
                    if (highOverBottom)
                        z.Box.Color = EntryBearColor;
                    else
                        z.Box.Color = BearObColor;
                }

                if (ShowLabels)
                    SetLabelAndBoxPosition(z, index);

                if (MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeOverBottom)
                    {
                        var oldBottom = z.Bottom;
                        z.Bottom = close;
                        if (ShowLabels)
                            SetLabelY(z, index, close, oldBottom); // source parity (close,bottom_pre_update)
                    }
                    else if (highOverBottom)
                    {
                        z.Bottom = high;
                        if (ShowLabels)
                            SetLabelY(z, index, z.Top, high);
                    }
                    RedrawZone(z, index);
                }

                if (MitigationActionInput == MitigationMode.None)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeOverTop)
                        z.Box.Color = NoMitColor;
                    else if (highOverTop)
                        z.Box.Color = NoMitColor;

                    if ((MitigationTypeInput == MitigationType.Body && closeOverTop) || highOverTop)
                    {
                        if (ShowLabels && ShowMitigatedText && z.Label != null && !z.Label.Text.Contains("Mitigated"))
                            z.Label.Text += " Mitigated";
                    }
                }

                var delete = false;
                if (MitigationActionInput == MitigationMode.Normal || MitigationActionInput == MitigationMode.Dynamic)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeOverTop)
                        delete = true;
                    else if (highOverTop)
                        delete = true;
                }
                else if (MitigationActionInput == MitigationMode.Half)
                {
                    if (MitigationTypeInput == MitigationType.Body && closeOverMid)
                        delete = true;
                    else if (highOverMid)
                        delete = true;
                }

                if (delete)
                    DeleteZone(tf.Bears, i);
            }
        }

        private void SetLabelAndBoxPosition(ObZone zone, int index)
        {
            var left = ShiftTime(index, LabelShift);
            zone.Box.Time1 = left;
            // cTrader has no extend.right on rectangles; keep right in the future each bar
            // to emulate Pine's right-extended OB boxes.
            zone.Box.Time2 = ShiftTime(index, 200);
            SetLabelY(zone, index, zone.Top, zone.Bottom);
        }

        private void SetLabelY(ObZone zone, int index, double top, double bottom)
        {
            if (zone.Label == null)
                return;

            zone.Label.Time = ShiftTime(index, LabelShift);
            zone.Label.Y = (top + bottom) / 2.0;
        }

        private void RedrawZone(ObZone zone, int index)
        {
            zone.Box.Y1 = zone.Top;
            zone.Box.Y2 = zone.Bottom;
            zone.Box.Time1 = ShiftTime(index, LabelShift);
            zone.Box.Time2 = ShiftTime(index, 200);
        }

        private bool HasDuplicateByTop(List<ObZone> zones, double top)
        {
            for (int i = zones.Count - 1; i >= 0; i--)
            {
                if (zones[i].Top == top)
                    return true;
            }
            return false;
        }

        private void DeleteZone(List<ObZone> list, int index)
        {
            var zone = list[index];
            Chart.RemoveObject(zone.Id);
            if (zone.Label != null)
                Chart.RemoveObject(zone.LabelId);
            list.RemoveAt(index);
        }

        private int FindBarIndexAtOrBefore(Bars bars, DateTime time)
        {
            var i = bars.OpenTimes.GetIndexByTime(time);
            if (i >= 0)
                return i;

            for (int j = bars.Count - 1; j >= 0; j--)
            {
                if (bars.OpenTimes[j] <= time)
                    return j;
            }
            return -1;
        }

        private DateTime ShiftTime(int chartIndex, int barsForward)
        {
            var baseTime = Bars.OpenTimes[chartIndex];
            var step = Bars.TimeFrame.ToTimeSpan();
            return baseTime.Add(TimeSpan.FromTicks(step.Ticks * barsForward));
        }

        private bool IsInSession(DateTime dt)
        {
            var minutes = dt.Hour * 60 + dt.Minute;
            return minutes >= 570 && minutes <= 960;
        }

        private string GetChartTimeframeLabel()
        {
            if (Bars.TimeFrame == TimeFrame.Daily)
                return "Daily";
            if (Bars.TimeFrame == TimeFrame.Weekly)
                return "Weekly";
            if (Bars.TimeFrame == TimeFrame.Monthly)
                return "Monthly";

            var minutes = Bars.TimeFrame.ToTimeSpan().TotalMinutes;
            if (minutes > 59)
                return (minutes / 60.0).ToString("0.#") + " Hr";
            return minutes.ToString("0") + " Min";
        }
    }
}
