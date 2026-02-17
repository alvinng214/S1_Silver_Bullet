using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;
using cAlgo.API.Indicators;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MtfFvgX2MkCt : Indicator
    {
        private enum MitigationMode { Normal = 1, Dynamic = 2, None = 3, Half = 4 }

        private sealed class FvgZone
        {
            public string Id;
            public bool IsBull;
            public double Top;
            public double Bottom;
            public ChartRectangle Rect;
            public ChartText Label;
            public string TimeframeLabel;
        }

        private sealed class OverlayZone
        {
            public string Id;
            public ChartRectangle Box;
            public ChartTrendLine TopLine;
            public ChartTrendLine MidLine;
            public ChartTrendLine BottomLine;
        }

        [Parameter("MTF FVGs (display to right)", DefaultValue = true, Group = "Enable/Disable")]
        public bool MtfImb { get; set; }

        [Parameter("MTF FVG (price overlay)", DefaultValue = true, Group = "Enable/Disable")]
        public bool MtfPo { get; set; }

        [Parameter("Timeframe", DefaultValue = "Hour", Group = "Enable/Disable")]
        public TimeFrame TfInput { get; set; }

        [Parameter("Only Market Hours", DefaultValue = false, Group = "MTF Fair Value Gaps")]
        public bool OnlyMktHrs { get; set; }

        [Parameter("Left O/S", DefaultValue = 5, MinValue = 1, MaxValue = 100, Group = "MTF Fair Value Gaps")]
        public int LabelShift { get; set; }

        [Parameter("Right O/S", DefaultValue = 15, MinValue = 1, MaxValue = 100, Group = "MTF Fair Value Gaps")]
        public int LabelShiftRight { get; set; }

        [Parameter("Incursion Alerts", DefaultValue = true, Group = "MTF Fair Value Gaps")]
        public bool IncursionAlerts { get; set; }

        [Parameter("Incursion %", DefaultValue = 20.0, MinValue = 0, MaxValue = 100, Group = "MTF Fair Value Gaps")]
        public double IncursionPct { get; set; }

        [Parameter("Mitigation Action", DefaultValue = "Normal", Group = "MTF Fair Value Gaps")]
        public string MitigationActionInput { get; set; }

        [Parameter("Mitigation Type", DefaultValue = "Wicks", Group = "MTF Fair Value Gaps")]
        public string MitigationTypeInput { get; set; }

        [Parameter("Change FVG Color On Entry", DefaultValue = true, Group = "MTF Fair Value Gaps")]
        public bool EntryChangeColor { get; set; }

        [Parameter("Entry Bull", DefaultValue = "#00FF00", Group = "MTF Fair Value Gaps")]
        public Color EntryBullColor { get; set; }

        [Parameter("Entry Bear", DefaultValue = "#FF0000", Group = "MTF Fair Value Gaps")]
        public Color EntryBearColor { get; set; }

        [Parameter("Show Labels", DefaultValue = true, Group = "MTF Fair Value Gaps")]
        public bool ShowLabels { get; set; }

        [Parameter("Bull FVG Color", DefaultValue = "#B3FFFF00", Group = "FVG Box Border")]
        public Color BullFvgColor { get; set; }

        [Parameter("Bear FVG Color", DefaultValue = "#B3FFFF00", Group = "FVG Box Border")]
        public Color BearFvgColor { get; set; }

        [Parameter("Label Color", DefaultValue = "#000000", Group = "FVG Box Border")]
        public Color LabelColor { get; set; }

        [Parameter("Enable Chart TF", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool EnableChartTf { get; set; }
        [Parameter("Enable 5m", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable5m { get; set; }
        [Parameter("Enable 10m", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable10m { get; set; }
        [Parameter("Enable 15m", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable15m { get; set; }
        [Parameter("Enable 30m", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable30m { get; set; }
        [Parameter("Enable 1h", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable1h { get; set; }
        [Parameter("Enable 4h", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool Enable4h { get; set; }
        [Parameter("Enable 8h", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable8h { get; set; }
        [Parameter("Enable 12h", DefaultValue = false, Group = "Enabled Timeframes")]
        public bool Enable12h { get; set; }
        [Parameter("Enable Daily", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool EnableDaily { get; set; }
        [Parameter("Enable Weekly", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool EnableWeekly { get; set; }
        [Parameter("Enable Monthly", DefaultValue = true, Group = "Enabled Timeframes")]
        public bool EnableMonthly { get; set; }

        [Parameter("Max Chart", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int MaxChart { get; set; }
        [Parameter("Max 5m", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max5m { get; set; }
        [Parameter("Max 10m", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max10m { get; set; }
        [Parameter("Max 15m", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max15m { get; set; }
        [Parameter("Max 30m", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max30m { get; set; }
        [Parameter("Max 1h", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max1h { get; set; }
        [Parameter("Max 4h", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max4h { get; set; }
        [Parameter("Max 8h", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max8h { get; set; }
        [Parameter("Max 12h", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int Max12h { get; set; }
        [Parameter("Max Daily", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int MaxDaily { get; set; }
        [Parameter("Max Weekly", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int MaxWeekly { get; set; }
        [Parameter("Max Monthly", DefaultValue = 8, MinValue = 1, Group = "Max FVG Settings")]
        public int MaxMonthly { get; set; }

        [Parameter("Boxtype", DefaultValue = "Imbalance", Group = "FVG Price Overlay")]
        public string BoxType { get; set; }
        [Parameter("Show Up", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowUp { get; set; }
        [Parameter("Show Down", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowDown { get; set; }
        [Parameter("Up Color", DefaultValue = "#B3FFFF00", Group = "FVG Price Overlay")]
        public Color UpColor { get; set; }
        [Parameter("Down Color", DefaultValue = "#B3FFFF00", Group = "FVG Price Overlay")]
        public Color DownColor { get; set; }
        [Parameter("Up Border", DefaultValue = "#644CAF4F", Group = "FVG Price Overlay")]
        public Color UpBorderColor { get; set; }
        [Parameter("Down Border", DefaultValue = "#64FF5252", Group = "FVG Price Overlay")]
        public Color DownBorderColor { get; set; }
        [Parameter("Extend Till Filled", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ExtendTillFilled { get; set; }
        [Parameter("Fill Condition", DefaultValue = "Full Fill", Group = "FVG Price Overlay")]
        public string FilledType { get; set; }
        [Parameter("Lookback", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool Lookback { get; set; }
        [Parameter("Lookback Days", DefaultValue = 5.0, Group = "FVG Price Overlay")]
        public double DaysBack { get; set; }
        [Parameter("Hide Filled", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool HideFilled { get; set; }
        [Parameter("Show Boxes", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowBoxes { get; set; }
        [Parameter("Condition Type", DefaultValue = "None", Group = "FVG Price Overlay")]
        public string ConditionType { get; set; }
        [Parameter("ATR Length", DefaultValue = 30, Group = "FVG Price Overlay")]
        public int AtrLength { get; set; }
        [Parameter("ATR Mult", DefaultValue = 1.0, Group = "FVG Price Overlay")]
        public double AtrMult { get; set; }
        [Parameter("Pct Cond", DefaultValue = 0.30, Group = "FVG Price Overlay")]
        public double PctCond { get; set; }
        [Parameter("Pct Mult", DefaultValue = 1.0, Group = "FVG Price Overlay")]
        public double PctMult { get; set; }
        [Parameter("Max Overlay Boxes", DefaultValue = 499, MinValue = 1, Group = "FVG Price Overlay")]
        public int MaxOverlayBoxes { get; set; }

        private readonly Dictionary<string, Bars> _barsByTf = new Dictionary<string, Bars>();
        private readonly Dictionary<string, List<FvgZone>> _bullByTf = new Dictionary<string, List<FvgZone>>();
        private readonly Dictionary<string, List<FvgZone>> _bearByTf = new Dictionary<string, List<FvgZone>>();
        private readonly Dictionary<string, int> _maxByTf = new Dictionary<string, int>();

        private readonly List<OverlayZone> _overlayZones = new List<OverlayZone>();
        private AverageTrueRange _overlayAtr;
        private int _id;

        protected override void Initialize()
        {
            _overlayAtr = Indicators.AverageTrueRange(AtrLength, MovingAverageType.Exponential);

            RegisterTf("Chart", Bars.TimeFrame, EnableChartTf, MaxChart);
            RegisterTf("5m", TimeFrame.Minute5, Enable5m, Max5m);
            RegisterTf("10m", TimeFrame.Minute10, Enable10m, Max10m);
            RegisterTf("15m", TimeFrame.Minute15, Enable15m, Max15m);
            RegisterTf("30m", TimeFrame.Minute30, Enable30m, Max30m);
            RegisterTf("1hr", TimeFrame.Hour, Enable1h, Max1h);
            RegisterTf("4hr", TimeFrame.Hour4, Enable4h, Max4h);
            RegisterTf("8hr", TimeFrame.Hour8, Enable8h, Max8h);
            RegisterTf("12hr", TimeFrame.Hour12, Enable12h, Max12h);
            RegisterTf("Daily", TimeFrame.Daily, EnableDaily, MaxDaily);
            RegisterTf("Weekly", TimeFrame.Weekly, EnableWeekly, MaxWeekly);
            RegisterTf("Monthly", TimeFrame.Monthly, EnableMonthly, MaxMonthly);

            ValidateMaxFvgCount();
        }

        public override void Calculate(int index)
        {
            if (MtfImb)
                ProcessMtfSubsystem(index);
            if (MtfPo)
                ProcessOverlaySubsystem(index);
        }

        private void ValidateMaxFvgCount()
        {
            var total = MaxChart + Max5m + Max10m + Max15m + Max30m + Max1h + Max4h + Max8h + Max12h + MaxDaily + MaxWeekly + MaxMonthly;
            if (total > 500)
                Chart.DrawStaticText("mk_error", "MTF FVG INDICATOR ERROR\n\nMax Number of FVGs exceeded, please change settings.", VerticalAlignment.Bottom, HorizontalAlignment.Right, Color.Red);
        }

        private void RegisterTf(string key, TimeFrame tf, bool enabled, int max)
        {
            if (!enabled)
                return;
            _barsByTf[key] = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _bullByTf[key] = new List<FvgZone>();
            _bearByTf[key] = new List<FvgZone>();
            _maxByTf[key] = max;
        }

        private MitigationMode GetMitigationMode()
        {
            if (string.Equals(MitigationActionInput, "Dynamic", StringComparison.OrdinalIgnoreCase)) return MitigationMode.Dynamic;
            if (string.Equals(MitigationActionInput, "None", StringComparison.OrdinalIgnoreCase)) return MitigationMode.None;
            if (string.Equals(MitigationActionInput, "Half", StringComparison.OrdinalIgnoreCase)) return MitigationMode.Half;
            return MitigationMode.Normal;
        }

        private bool UseBodyMitigation => string.Equals(MitigationTypeInput, "Body", StringComparison.OrdinalIgnoreCase);

        private void ProcessMtfSubsystem(int chartIndex)
        {
            foreach (var kv in _barsByTf)
            {
                var tfKey = kv.Key;
                var tfBars = kv.Value;
                var i = FindBarIndexAtOrBefore(tfBars, Bars.OpenTimes[chartIndex]);
                if (i < 3)
                    continue;

                if (OnlyMktHrs)
                {
                    var mins = Bars.OpenTimes[chartIndex].Hour * 60 + Bars.OpenTimes[chartIndex].Minute;
                    if (mins < 570 || mins > 960)
                        continue;
                }

                var h = tfBars.HighPrices[i - 1];
                var h2 = tfBars.HighPrices[i - 3];
                var l = tfBars.LowPrices[i - 1];
                var l2 = tfBars.LowPrices[i - 3];
                var close1 = tfBars.ClosePrices[i - 2];
                var open1 = tfBars.OpenPrices[i - 2];

                var newBull = IsFvgBull(l, h2, close1, open1);
                var newBear = IsFvgBear(l2, h, close1, open1);

                if (newBull)
                {
                    if (_bullByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bullByTf[tfKey], 0);

                    if (h2 != tfBars.HighPrices[Math.Max(0, i - 4)] && l != tfBars.LowPrices[Math.Max(0, i - 2)])
                        AddFvg(tfKey, true, tfBars.OpenTimes[i - 3], l, h2);
                }

                if (newBear)
                {
                    if (_bearByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bearByTf[tfKey], 0);

                    if (l2 != tfBars.LowPrices[Math.Max(0, i - 4)] && h != tfBars.HighPrices[Math.Max(0, i - 2)])
                        AddFvg(tfKey, false, tfBars.OpenTimes[i - 3], l2, h);
                }

                UpdateExistingFvgs(_bullByTf[tfKey], true, tfKey, chartIndex);
                UpdateExistingFvgs(_bearByTf[tfKey], false, tfKey, chartIndex);
            }
        }

        private bool IsFvgBull(double low, double high2, double close1, double open1) => high2 < low;
        private bool IsFvgBear(double low2, double high, double close1, double open1) => low2 > high;

        private void AddFvg(string tfKey, bool bull, DateTime left, double top, double bottom)
        {
            var id = $"mk_fvg_{tfKey}_{(bull ? 'b' : 's')}_{_id++}";
            var c = bull ? BullFvgColor : BearFvgColor;
            var rect = Chart.DrawRectangle(id, left, top, Server.Time.AddMinutes(LabelShiftRight), bottom, c);
            rect.IsFilled = true;
            rect.IsInteractive = false;

            ChartText label = null;
            if (ShowLabels)
                label = Chart.DrawText(id + "_lbl", tfKey, Server.Time.AddMinutes(LabelShift), (top + bottom) / 2.0, LabelColor);

            var z = new FvgZone { Id = id, IsBull = bull, Top = top, Bottom = bottom, Rect = rect, Label = label, TimeframeLabel = tfKey };
            (bull ? _bullByTf[tfKey] : _bearByTf[tfKey]).Add(z);
        }

        private void UpdateExistingFvgs(List<FvgZone> zones, bool bull, string tfKey, int idx)
        {
            var mode = GetMitigationMode();
            var intrPct = IncursionPct / 100.0;
            for (var i = zones.Count - 1; i >= 0; i--)
            {
                var z = zones[i];
                var mid = (z.Top + z.Bottom) / 2.0;
                var threshold = bull ? z.Top - intrPct * (z.Top - z.Bottom) : z.Bottom + intrPct * (z.Top - z.Bottom);
                var intrusion = bull
                    ? Bars.LowPrices[idx] < threshold && Bars.LowPrices[idx - 1] > threshold
                    : Bars.HighPrices[idx] > threshold && Bars.HighPrices[idx - 1] < threshold;

                if ((mode == MitigationMode.Normal || mode == MitigationMode.None) && intrusion && IncursionAlerts)
                    Print("{0} FVG Incursion {1}", bull ? "Bull" : "Bear", tfKey);

                if (EntryChangeColor)
                {
                    var entered = bull ? Bars.LowPrices[idx] < z.Top : Bars.HighPrices[idx] > z.Bottom;
                    z.Rect.Color = entered ? (bull ? EntryBullColor : EntryBearColor) : (bull ? BullFvgColor : BearFvgColor);
                }

                if (mode == MitigationMode.Dynamic)
                {
                    if (bull)
                    {
                        if (UseBodyMitigation && Bars.ClosePrices[idx] < z.Top)
                            z.Top = Bars.ClosePrices[idx];
                        else if (!UseBodyMitigation && Bars.LowPrices[idx] < z.Top)
                            z.Top = Bars.LowPrices[idx];
                    }
                    else
                    {
                        if (UseBodyMitigation && Bars.ClosePrices[idx] > z.Bottom)
                            z.Bottom = Bars.ClosePrices[idx];
                        else if (!UseBodyMitigation && Bars.HighPrices[idx] > z.Bottom)
                            z.Bottom = Bars.HighPrices[idx];
                    }
                }

                var removeFull = bull
                    ? (UseBodyMitigation ? Bars.ClosePrices[idx] < z.Bottom : Bars.LowPrices[idx] < z.Bottom)
                    : (UseBodyMitigation ? Bars.ClosePrices[idx] > z.Top : Bars.HighPrices[idx] > z.Top);
                var removeHalf = bull
                    ? (UseBodyMitigation ? Bars.ClosePrices[idx] < mid : Bars.LowPrices[idx] < mid)
                    : (UseBodyMitigation ? Bars.ClosePrices[idx] > mid : Bars.HighPrices[idx] > mid);

                if (((mode == MitigationMode.Normal || mode == MitigationMode.Dynamic) && removeFull) ||
                    (mode == MitigationMode.Half && removeHalf))
                {
                    RemoveFvgAt(zones, i);
                    continue;
                }

                z.Rect.Time2 = Server.Time.AddMinutes(LabelShiftRight);
                z.Rect.Y1 = z.Top;
                z.Rect.Y2 = z.Bottom;
                if (z.Label != null)
                {
                    z.Label.Time = Server.Time.AddMinutes(LabelShift);
                    z.Label.Y = (z.Top + z.Bottom) / 2.0;
                }
            }
        }

        private void RemoveFvgAt(List<FvgZone> zones, int i)
        {
            if (i < 0 || i >= zones.Count) return;
            var z = zones[i];
            Chart.RemoveObject(z.Id);
            Chart.RemoveObject(z.Id + "_lbl");
            zones.RemoveAt(i);
        }

        private void ProcessOverlaySubsystem(int idx)
        {
            var tfBars = MarketData.GetBars(TfInput);
            var i = FindBarIndexAtOrBefore(tfBars, Bars.OpenTimes[idx]);
            if (i < 2)
                return;

            var h0 = tfBars.HighPrices[i];
            var l0 = tfBars.LowPrices[i];
            var o1 = tfBars.OpenPrices[i - 1];
            var h1 = tfBars.HighPrices[i - 1];
            var l1 = tfBars.LowPrices[i - 1];
            var c1 = tfBars.ClosePrices[i - 1];
            var h2 = tfBars.HighPrices[i - 2];
            var l2 = tfBars.LowPrices[i - 2];
            var t = tfBars.OpenTimes;

            var inRange = true;
            if (Lookback)
                inRange = (Server.Time.Date - Bars.OpenTimes[idx].Date).TotalDays < DaysBack;
            if (!inRange) return;

            var upImbDist = (l0 - h2) / Math.Max(Math.Abs(h2), Symbol.TickSize) * 100.0;
            var downImbDist = (l2 - h0) / Math.Max(Math.Abs(h0), Symbol.TickSize) * 100.0;
            var upGapDist = (l0 - h1) / Math.Max(Math.Abs(h1), Symbol.TickSize) * 100.0;
            var downGapDist = (l1 - h0) / Math.Max(Math.Abs(h0), Symbol.TickSize) * 100.0;
            var upWickDist = (h1 - Math.Max(o1, c1)) / Math.Max(Math.Abs(Math.Max(o1, c1)), Symbol.TickSize) * 100.0;
            var downWickDist = (Math.Min(o1, c1) - l1) / Math.Max(Math.Abs(l1), Symbol.TickSize) * 100.0;
            var bodySize = Math.Abs(o1 - c1);
            var upperWick = h1 - Math.Max(o1, c1);
            var lowerWick = Math.Min(o1, c1) - l1;

            var atr = _overlayAtr.Result[Math.Max(0, idx)];
            bool c1ok, c2ok, c3ok, c4ok, c5ok, c6ok;
            if (string.Equals(ConditionType, "Percentage", StringComparison.OrdinalIgnoreCase))
            {
                c1ok = upImbDist > (PctCond * PctMult);
                c2ok = downImbDist > (PctCond * PctMult);
                c3ok = upGapDist > (PctCond * PctMult);
                c4ok = downGapDist > (PctCond * PctMult);
                c5ok = upWickDist > (PctCond * PctMult);
                c6ok = downWickDist > (PctCond * PctMult);
            }
            else if (string.Equals(ConditionType, "ATR", StringComparison.OrdinalIgnoreCase))
            {
                c1ok = (l0 - h2) > (AtrMult * atr);
                c2ok = (l2 - h0) > (AtrMult * atr);
                c3ok = (l0 - h1) > (AtrMult * atr);
                c4ok = (l1 - h0) > (AtrMult * atr);
                c5ok = upperWick > (AtrMult * atr);
                c6ok = lowerWick > (AtrMult * atr);
            }
            else
            {
                c1ok = c2ok = c3ok = c4ok = c5ok = c6ok = true;
            }

            var upColor = ShowBoxes ? UpColor : Color.FromArgb(0, 0, 0, 0);
            var downColor = ShowBoxes ? DownColor : Color.FromArgb(0, 0, 0, 0);

            if (string.Equals(BoxType, "Imbalance", StringComparison.OrdinalIgnoreCase) && ShowUp && l0 > h2 && c1ok)
                AddOverlay(t[i - 2], l0, Server.Time, h2, upColor, UpBorderColor);
            if (string.Equals(BoxType, "Imbalance", StringComparison.OrdinalIgnoreCase) && ShowDown && h0 < l2 && c2ok)
                AddOverlay(t[i - 2], l2, Server.Time, h0, downColor, DownBorderColor);

            if (string.Equals(BoxType, "Gap", StringComparison.OrdinalIgnoreCase) && ShowUp && l0 > h1 && c3ok)
                AddOverlay(t[i - 1], l0, Server.Time, h1, upColor, UpBorderColor);
            if (string.Equals(BoxType, "Gap", StringComparison.OrdinalIgnoreCase) && ShowDown && h0 < l1 && c4ok)
                AddOverlay(t[i - 1], l1, Server.Time, h0, downColor, DownBorderColor);

            if (string.Equals(BoxType, "Wick", StringComparison.OrdinalIgnoreCase) && ShowUp && upperWick > (bodySize / 6.0) && c5ok)
                AddOverlay(t[i - 1], h1, Server.Time, Math.Max(o1, c1), upColor, UpBorderColor);
            if (string.Equals(BoxType, "Wick", StringComparison.OrdinalIgnoreCase) && ShowDown && lowerWick > (bodySize / 6.0) && c6ok)
                AddOverlay(t[i - 1], Math.Min(o1, c1), Server.Time, l1, downColor, DownBorderColor);

            UpdateOverlayLifecycle(idx);
        }

        private void AddOverlay(DateTime left, double top, DateTime right, double bottom, Color fillColor, Color border)
        {
            var id = $"mk_overlay_{_id++}";
            var b = Chart.DrawRectangle(id, left, top, right, bottom, fillColor);
            b.IsFilled = true;
            b.IsInteractive = false;
            b.Color = fillColor;

            var lineColor = Color.FromArgb(31, 178, 181, 190);
            var t = Chart.DrawTrendLine(id + "_t", left, top, right, top, lineColor);
            var m = Chart.DrawTrendLine(id + "_m", left, (top + bottom) / 2.0, right, (top + bottom) / 2.0, lineColor);
            var bt = Chart.DrawTrendLine(id + "_b", left, bottom, right, bottom, lineColor);
            _overlayZones.Add(new OverlayZone { Id = id, Box = b, TopLine = t, MidLine = m, BottomLine = bt });
        }

        private void UpdateOverlayLifecycle(int idx)
        {
            for (var i = _overlayZones.Count - 1; i >= 0; i--)
            {
                var z = _overlayZones[i];
                var top = z.Box.Y1;
                var bottom = z.Box.Y2;
                var mid = (top + bottom) / 2.0;

                var filled = string.Equals(FilledType, "Touch", StringComparison.OrdinalIgnoreCase)
                    ? (Bars.HighPrices[idx] > bottom && Bars.LowPrices[idx] < bottom) || (Bars.HighPrices[idx] > top && Bars.LowPrices[idx] < top)
                    : string.Equals(FilledType, "Half Fill", StringComparison.OrdinalIgnoreCase)
                        ? (Bars.HighPrices[idx] > mid && Bars.LowPrices[idx] < mid)
                        : (Bars.HighPrices[idx] > top && Bars.LowPrices[idx] < top) || (Bars.HighPrices[idx] > bottom && Bars.LowPrices[idx] < bottom);

                if (HideFilled && filled)
                {
                    RemoveOverlayAt(i);
                    continue;
                }

                if (filled && ExtendTillFilled)
                {
                    _overlayZones.RemoveAt(i);
                    continue;
                }

                z.Box.Time2 = Server.Time.AddSeconds(1);
                z.TopLine.Time2 = Server.Time.AddSeconds(1);
                z.MidLine.Time2 = Server.Time.AddSeconds(1);
                z.BottomLine.Time2 = Server.Time.AddSeconds(1);
            }

            while (_overlayZones.Count >= MaxOverlayBoxes)
                RemoveOverlayAt(0);
        }

        private void RemoveOverlayAt(int i)
        {
            if (i < 0 || i >= _overlayZones.Count) return;
            var z = _overlayZones[i];
            Chart.RemoveObject(z.Id);
            Chart.RemoveObject(z.Id + "_t");
            Chart.RemoveObject(z.Id + "_m");
            Chart.RemoveObject(z.Id + "_b");
            _overlayZones.RemoveAt(i);
        }

        private static int FindBarIndexAtOrBefore(Bars bars, DateTime t)
        {
            var lo = 0;
            var hi = bars.Count - 1;
            var ans = -1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                if (bars.OpenTimes[mid] <= t)
                {
                    ans = mid;
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
            return ans;
        }
    }
}
