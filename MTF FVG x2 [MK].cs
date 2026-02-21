using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class MtfFvgX2MkCt : Indicator
    {
        public enum MitigationMode { Normal = 1, Dynamic = 2, None = 3, Half = 4 }
        public enum MitigationType { Wicks = 1, Body = 2 }
        public enum OverlayBoxType { Imbalance = 1, Gap = 2, Wick = 3 }
        public enum FillCondition { FullFill = 1, HalfFill = 2, Touch = 3 }
        public enum OverlayConditionType { None = 1, Percentage = 2, Atr = 3 }

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
            public bool IsUp;
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

        [Parameter("Mitigation Action", DefaultValue = MitigationMode.Normal, Group = "MTF Fair Value Gaps")]
        public MitigationMode MitigationActionInput { get; set; }

        [Parameter("Mitigation Type", DefaultValue = MitigationType.Wicks, Group = "MTF Fair Value Gaps")]
        public MitigationType MitigationTypeInput { get; set; }

        [Parameter("Change FVG Color On Entry", DefaultValue = true, Group = "MTF Fair Value Gaps")]
        public bool EntryChangeColor { get; set; }

        [Parameter("Entry Bull", DefaultValue = "#00FF00", Group = "MTF Fair Value Gaps")]
        public Color EntryBullColor { get; set; }

        [Parameter("Entry Bear", DefaultValue = "#FF0000", Group = "MTF Fair Value Gaps")]
        public Color EntryBearColor { get; set; }

        [Parameter("Show Labels", DefaultValue = true, Group = "MTF Fair Value Gaps")]
        public bool ShowLabels { get; set; }

        [Parameter("Bull FVG Color", DefaultValue = "#4DFFFF00", Group = "FVG Box Border")]
        public Color BullFvgColor { get; set; }

        [Parameter("Bear FVG Color", DefaultValue = "#4DFFFF00", Group = "FVG Box Border")]
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

        [Parameter("15m Min", DefaultValue = 1, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTf15 { get; set; }
        [Parameter("15m Max", DefaultValue = 4, MinValue = 1, MaxValue = 240, Group = "Visibility")]
        public int MaxTf15 { get; set; }
        [Parameter("1h Min", DefaultValue = 5, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTf60 { get; set; }
        [Parameter("1h Max", DefaultValue = 5, MinValue = 1, MaxValue = 240, Group = "Visibility")]
        public int MaxTf60Visible { get; set; }
        [Parameter("4h Min", DefaultValue = 15, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTf240 { get; set; }
        [Parameter("4h Max", DefaultValue = 15, MinValue = 1, MaxValue = 240, Group = "Visibility")]
        public int MaxTf240Visible { get; set; }
        [Parameter("Daily Min", DefaultValue = 60, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTfD { get; set; }
        [Parameter("Daily Max", DefaultValue = 60, MinValue = 1, MaxValue = 240, Group = "Visibility")]
        public int MaxTfDVisible { get; set; }
        [Parameter("Weekly Min", DefaultValue = 240, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTfW { get; set; }
        [Parameter("Weekly Max", DefaultValue = 240, MinValue = 1, MaxValue = 240, Group = "Visibility")]
        public int MaxTfWVisible { get; set; }
        [Parameter("Monthly Min", DefaultValue = 1440, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MinTfM { get; set; }
        [Parameter("Monthly Max", DefaultValue = 1440, MinValue = 1, MaxValue = 1440, Group = "Visibility")]
        public int MaxTfMVisible { get; set; }

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

        [Parameter("Boxtype", DefaultValue = OverlayBoxType.Imbalance, Group = "FVG Price Overlay")]
        public OverlayBoxType BoxType { get; set; }
        [Parameter("Show Up", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowUp { get; set; }
        [Parameter("Show Down", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowDown { get; set; }
        [Parameter("Up Color", DefaultValue = "#4DFFFF00", Group = "FVG Price Overlay")]
        public Color UpColor { get; set; }
        [Parameter("Down Color", DefaultValue = "#4DFFFF00", Group = "FVG Price Overlay")]
        public Color DownColor { get; set; }
        [Parameter("Up Border", DefaultValue = "#004CAF4F", Group = "FVG Price Overlay")]
        public Color UpBorderColor { get; set; }
        [Parameter("Down Border", DefaultValue = "#00FF5252", Group = "FVG Price Overlay")]
        public Color DownBorderColor { get; set; }
        [Parameter("Show Middle Line", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowMiddleLine { get; set; }
        [Parameter("Show Bottom Line", DefaultValue = false, Group = "FVG Price Overlay")]
        public bool ShowBottomLine { get; set; }
        [Parameter("Show Top Line", DefaultValue = false, Group = "FVG Price Overlay")]
        public bool ShowTopLine { get; set; }
        [Parameter("Extend Till Filled", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ExtendTillFilled { get; set; }
        [Parameter("Fill Condition", DefaultValue = FillCondition.FullFill, Group = "FVG Price Overlay")]
        public FillCondition FilledType { get; set; }
        [Parameter("Lookback", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool Lookback { get; set; }
        [Parameter("Lookback Days", DefaultValue = 5.0, Group = "FVG Price Overlay")]
        public double DaysBack { get; set; }
        [Parameter("Hide Filled", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool HideFilled { get; set; }
        [Parameter("Show Boxes", DefaultValue = true, Group = "FVG Price Overlay")]
        public bool ShowBoxes { get; set; }
        [Parameter("Condition Type", DefaultValue = OverlayConditionType.None, Group = "FVG Price Overlay")]
        public OverlayConditionType ConditionType { get; set; }
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
        private readonly Dictionary<string, int> _lastBullTfIndex = new Dictionary<string, int>();
        private readonly Dictionary<string, int> _lastBearTfIndex = new Dictionary<string, int>();

        private readonly List<OverlayZone> _overlayZones = new List<OverlayZone>();
        private int _id;
        private int _lastOverlayTfIndex = -1;
        private string _chartTfKey;
        private static readonly Color OverlayLineColor = Color.FromArgb(176, 178, 181, 190);
        private static readonly Color HiddenFvgBorder = Color.FromArgb(0, 128, 128, 128);

        protected override void Initialize()
        {
            _chartTfKey = BuildChartTfKey();

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
            _lastBullTfIndex[key] = -1;
            _lastBearTfIndex[key] = -1;
        }

        private MitigationMode GetMitigationMode() => MitigationActionInput;

        private bool UseBodyMitigation => MitigationTypeInput == MitigationType.Body;

        private void ProcessMtfSubsystem(int chartIndex)
        {
            foreach (var kv in _barsByTf)
            {
                var tfKey = kv.Key;
                if (!IsTfVisible(tfKey) || (tfKey == "Chart" && IsChartTfDuplicatedByEnabledFixedTf()))
                    continue;

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

                if (newBull && _lastBullTfIndex[tfKey] != i)
                {
                    if (_bullByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bullByTf[tfKey], 0);

                    if (i >= 4 && h2 != tfBars.HighPrices[i - 4] && l != tfBars.LowPrices[i - 2])
                    {
                        AddFvg(tfKey, true, l, h2);
                        _lastBullTfIndex[tfKey] = i;
                    }
                }

                if (newBear && _lastBearTfIndex[tfKey] != i)
                {
                    if (_bearByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bearByTf[tfKey], 0);

                    if (i >= 4 && l2 != tfBars.LowPrices[i - 4] && h != tfBars.HighPrices[i - 2])
                    {
                        AddFvg(tfKey, false, l2, h);
                        _lastBearTfIndex[tfKey] = i;
                    }
                }

                UpdateExistingFvgs(_bullByTf[tfKey], true, tfKey, chartIndex);
                UpdateExistingFvgs(_bearByTf[tfKey], false, tfKey, chartIndex);
            }
        }

        private bool IsFvgBull(double low, double high2, double close1, double open1) => high2 < low;
        private bool IsFvgBear(double low2, double high, double close1, double open1) => low2 > high;

        private void AddFvg(string tfKey, bool bull, double top, double bottom)
        {
            var id = $"mk_fvg_{tfKey}_{(bull ? 'b' : 's')}_{_id++}";
            var c = bull ? BullFvgColor : BearFvgColor;
            var rect = Chart.DrawRectangle(id, ShiftFromCurrentBar(5), top, ShiftFromCurrentBar(15), bottom, c);
            rect.IsFilled = true;
            rect.IsInteractive = false;
            rect.LineStyle = LineStyle.Dots;

            ChartText label = null;
            if (ShowLabels)
                label = Chart.DrawText(id + "_lbl", tfKey, ShiftFromCurrentBar(LabelShift), (top + bottom) / 2.0, LabelColor);

            var z = new FvgZone { Id = id, IsBull = bull, Top = top, Bottom = bottom, Rect = rect, Label = label, TimeframeLabel = tfKey };
            (bull ? _bullByTf[tfKey] : _bearByTf[tfKey]).Add(z);
        }

        private void UpdateExistingFvgs(List<FvgZone> zones, bool bull, string tfKey, int idx)
        {
            if (idx < 1)
                return;
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
                    ? (UseBodyMitigation ? Bars.LowPrices[idx] < mid : Bars.LowPrices[idx] < mid)
                    : (UseBodyMitigation ? Bars.ClosePrices[idx] > mid : Bars.HighPrices[idx] > mid);

                if (((mode == MitigationMode.Normal || mode == MitigationMode.Dynamic) && removeFull) ||
                    (mode == MitigationMode.Half && removeHalf))
                {
                    RemoveFvgAt(zones, i);
                    continue;
                }

                z.Rect.Y1 = z.Top;
                z.Rect.Y2 = z.Bottom;
                z.Rect.Time1 = ShiftFromCurrentBar(LabelShift);
                z.Rect.Time2 = ShiftFromCurrentBar(LabelShiftRight);
                if (ShowLabels && z.Label != null)
                {
                    z.Label.Time = ShiftFromCurrentBar(LabelShift);
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
            {
                var daysLeft = Math.Abs(Math.Floor((Server.Time - Bars.OpenTimes[idx]).TotalDays));
                inRange = daysLeft < DaysBack;
            }
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

            var atr = GetAtrOnTf(tfBars, i, AtrLength);
            bool c1ok, c2ok, c3ok, c4ok, c5ok, c6ok;
            if (ConditionType == OverlayConditionType.Percentage)
            {
                c1ok = upImbDist > (PctCond * PctMult);
                c2ok = downImbDist > (PctCond * PctMult);
                c3ok = upGapDist > (PctCond * PctMult);
                c4ok = downGapDist > (PctCond * PctMult);
                c5ok = upWickDist > (PctCond * PctMult);
                c6ok = downWickDist > (PctCond * PctMult);
            }
            else if (ConditionType == OverlayConditionType.Atr)
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

            var isNewTfBar = _lastOverlayTfIndex != i;
            if (isNewTfBar)
            {
                if (BoxType == OverlayBoxType.Imbalance && ShowUp && l0 > h2 && c1ok)
                    AddOverlay(t[i - 2], l0, Bars.OpenTimes[idx], h2, upColor, true);
                if (BoxType == OverlayBoxType.Imbalance && ShowDown && h0 < l2 && c2ok)
                    AddOverlay(t[i - 2], l2, Bars.OpenTimes[idx], h0, downColor, false);

                if (BoxType == OverlayBoxType.Gap && ShowUp && l0 > h1 && c3ok)
                    AddOverlay(t[i - 1], l0, Bars.OpenTimes[idx], h1, upColor, true);
                if (BoxType == OverlayBoxType.Gap && ShowDown && h0 < l1 && c4ok)
                    AddOverlay(t[i - 1], l1, Bars.OpenTimes[idx], h0, downColor, false);

                if (BoxType == OverlayBoxType.Wick && ShowUp && upperWick > (bodySize / 6.0) && c5ok)
                    AddOverlay(t[i - 1], h1, Bars.OpenTimes[idx], Math.Max(o1, c1), upColor, true);
                if (BoxType == OverlayBoxType.Wick && ShowDown && lowerWick > (bodySize / 6.0) && c6ok)
                    AddOverlay(t[i - 1], Math.Min(o1, c1), Bars.OpenTimes[idx], l1, downColor, false);

                _lastOverlayTfIndex = i;
            }

            UpdateOverlayLifecycle(idx);
        }

        private void AddOverlay(DateTime left, double top, DateTime right, double bottom, Color fillColor, bool isUp)
        {
            var id = $"mk_overlay_{_id++}";
            var b = Chart.DrawRectangle(id, left, top, right, bottom, fillColor);
            b.IsFilled = true;
            b.IsInteractive = false;
            b.Color = fillColor;
            b.LineStyle = LineStyle.Solid;

            var t = ShowTopLine ? Chart.DrawTrendLine(id + "_t", left, top, right, top, OverlayLineColor) : null;
            var m = ShowMiddleLine ? Chart.DrawTrendLine(id + "_m", left, (top + bottom) / 2.0, right, (top + bottom) / 2.0, OverlayLineColor) : null;
            var bt = ShowBottomLine ? Chart.DrawTrendLine(id + "_b", left, bottom, right, bottom, OverlayLineColor) : null;
            if (t != null) t.LineStyle = LineStyle.Dots;
            if (m != null) m.LineStyle = LineStyle.Dots;
            if (bt != null) bt.LineStyle = LineStyle.Dots;
            _overlayZones.Add(new OverlayZone { Id = id, Box = b, TopLine = t, MidLine = m, BottomLine = bt, IsUp = isUp });
        }

        private void UpdateOverlayLifecycle(int idx)
        {
            for (var i = _overlayZones.Count - 1; i >= 0; i--)
            {
                var z = _overlayZones[i];
                var top = z.Box.Y1;
                var bottom = z.Box.Y2;
                var mid = (top + bottom) / 2.0;

                var filled = FilledType == FillCondition.Touch
                    ? (Bars.HighPrices[idx] > bottom && Bars.LowPrices[idx] < bottom) || (Bars.HighPrices[idx] > top && Bars.LowPrices[idx] < top)
                    : FilledType == FillCondition.HalfFill
                        ? (Bars.HighPrices[idx] > mid && Bars.LowPrices[idx] < mid)
                        : (Bars.HighPrices[idx] > bottom && Bars.LowPrices[idx] < bottom);

                if (HideFilled && filled)
                {
                    RemoveOverlayAt(i);
                    continue;
                }

                if (filled && ExtendTillFilled)
                {
                    RemoveOverlayAt(i);
                    continue;
                }


                z.Box.Time2 = Bars.OpenTimes[idx].AddMilliseconds(1);
                if (z.TopLine != null)
                    z.TopLine.Time2 = Bars.OpenTimes[idx].AddMilliseconds(1);
                if (z.MidLine != null)
                    z.MidLine.Time2 = Bars.OpenTimes[idx].AddMilliseconds(1);
                if (z.BottomLine != null)
                    z.BottomLine.Time2 = Bars.OpenTimes[idx].AddMilliseconds(1);

                if (!filled && !ExtendTillFilled)
                {
                    RemoveOverlayAt(i);
                    continue;
                }
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


        private bool IsChartTfDuplicatedByEnabledFixedTf()
        {
            if (string.IsNullOrEmpty(_chartTfKey))
                return false;

            return (_chartTfKey == "5m" && Enable5m)
                   || (_chartTfKey == "10m" && Enable10m)
                   || (_chartTfKey == "15m" && Enable15m)
                   || (_chartTfKey == "30m" && Enable30m)
                   || (_chartTfKey == "1hr" && Enable1h)
                   || (_chartTfKey == "4hr" && Enable4h)
                   || (_chartTfKey == "8hr" && Enable8h)
                   || (_chartTfKey == "12hr" && Enable12h)
                   || (_chartTfKey == "Daily" && EnableDaily)
                   || (_chartTfKey == "Weekly" && EnableWeekly)
                   || (_chartTfKey == "Monthly" && EnableMonthly);
        }

        private string BuildChartTfKey()
        {
            if (TryGetChartMinutes(out var mins))
            {
                if (mins < 60)
                    return mins + "m";
                if (mins % 60 == 0)
                    return (mins / 60) + "hr";
            }

            var tf = Bars.TimeFrame.ToString();
            if (tf.IndexOf("Daily", StringComparison.OrdinalIgnoreCase) >= 0)
                return "Daily";
            if (tf.IndexOf("Weekly", StringComparison.OrdinalIgnoreCase) >= 0)
                return "Weekly";
            if (tf.IndexOf("Monthly", StringComparison.OrdinalIgnoreCase) >= 0)
                return "Monthly";

            return "Chart";
        }

        private bool IsTfVisible(string tfKey)
        {
            if (tfKey == "15m")
                return IsVisibilityEnabled(MinTf15, MaxTf15, monthlyRule: false);
            if (tfKey == "1hr")
                return IsVisibilityEnabled(MinTf60, MaxTf60Visible, monthlyRule: false);
            if (tfKey == "4hr")
                return IsVisibilityEnabled(MinTf240, MaxTf240Visible, monthlyRule: false);
            if (tfKey == "Daily")
                return IsVisibilityEnabled(MinTfD, MaxTfDVisible, monthlyRule: false);
            if (tfKey == "Weekly")
                return IsVisibilityEnabled(MinTfW, MaxTfWVisible, monthlyRule: false);
            if (tfKey == "Monthly")
                return IsVisibilityEnabled(MinTfM, MaxTfMVisible, monthlyRule: true);

            return true;
        }

        private bool IsVisibilityEnabled(int minTf, int maxTf, bool monthlyRule)
        {
            var isDwm = IsDailyOrHigherChart();
            if (TryGetChartMinutes(out var mins))
            {
                var dispMin = mins >= minTf;
                var disp = mins <= maxTf;
                return dispMin && (monthlyRule ? (disp || isDwm) : disp);
            }

            if (!isDwm)
                return false;

            // Pine logic: for D/W/M charts, disp_mintf_* is true, but disp_* is false except Monthly.
            return monthlyRule;
        }

        private bool IsDailyOrHigherChart() => !TryGetChartMinutes(out _);

        private bool TryGetChartMinutes(out int minutes)
        {
            minutes = 0;
            var tf = Bars.TimeFrame.ToString();
            if (tf.Equals("Minute", StringComparison.OrdinalIgnoreCase))
            {
                minutes = 1;
                return true;
            }
            if (tf.StartsWith("Minute", StringComparison.OrdinalIgnoreCase) && int.TryParse(tf.Substring("Minute".Length), out var m))
            {
                minutes = m;
                return true;
            }
            if (tf.Equals("Hour", StringComparison.OrdinalIgnoreCase))
            {
                minutes = 60;
                return true;
            }
            if (tf.StartsWith("Hour", StringComparison.OrdinalIgnoreCase) && int.TryParse(tf.Substring("Hour".Length), out var h))
            {
                minutes = h * 60;
                return true;
            }
            return false;
        }

        private static double GetAtrOnTf(Bars bars, int endIndex, int length)
        {
            if (bars == null || bars.Count < 2 || endIndex <= 0)
                return double.NaN;

            var cappedEnd = Math.Min(endIndex, bars.Count - 1);
            var maxLen = Math.Max(1, length);
            var available = cappedEnd;

            if (available < maxLen)
                return double.NaN;

            if (available == maxLen)
            {
                var sum = 0.0;
                for (var k = 1; k <= maxLen; k++)
                    sum += TrueRange(bars, k);
                return sum / maxLen;
            }

            var seedSum = 0.0;
            for (var k = 1; k <= maxLen; k++)
                seedSum += TrueRange(bars, k);
            var atr = seedSum / maxLen;

            for (var k = maxLen + 1; k <= cappedEnd; k++)
            {
                var tr = TrueRange(bars, k);
                atr = ((atr * (maxLen - 1)) + tr) / maxLen;
            }

            return atr;
        }

        private static double TrueRange(Bars bars, int idx)
        {
            var high = bars.HighPrices[idx];
            var low = bars.LowPrices[idx];
            var prevClose = bars.ClosePrices[idx - 1];
            var a = high - low;
            var b = Math.Abs(high - prevClose);
            var c = Math.Abs(low - prevClose);
            return Math.Max(a, Math.Max(b, c));
        }

        private DateTime ShiftFromCurrentBar(int barsToRight)
        {
            var last = Bars.OpenTimes[Math.Max(0, Bars.Count - 1)];
            if (Bars.Count < 2)
                return last.AddMinutes(barsToRight);

            var span = Bars.OpenTimes[Bars.Count - 1] - Bars.OpenTimes[Bars.Count - 2];
            if (span <= TimeSpan.Zero)
                span = TimeSpan.FromMinutes(1);

            return last + TimeSpan.FromTicks(span.Ticks * barsToRight);
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
