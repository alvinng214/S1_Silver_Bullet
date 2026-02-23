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
            public ChartRectangle FillRect;
            public ChartRectangle BorderRect;
            public ChartText Text;
            public ChartTrendLine TopLine;
            public ChartTrendLine MidLine;
            public ChartTrendLine BottomLine;
            public bool IsUp;
        }

        // Enable/Disable
        [Parameter("MTF FVGs (display to right)", DefaultValue = true, Group = "Enable/Disable")]
        public bool MtfImb { get; set; }

        [Parameter("MTF FVG (price overlay)", DefaultValue = true, Group = "Enable/Disable")]
        public bool MtfPo { get; set; }

        [Parameter("Timeframe", DefaultValue = "Hour", Group = "Enable/Disable")]
        public TimeFrame TfInput { get; set; }

        // MTF Fair Value Gaps
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

        // FVG Box Border
        [Parameter("Bull FVG Color", DefaultValue = "#4DFFFF00", Group = "FVG Box Border")]
        public Color BullFvgColor { get; set; }

        [Parameter("Bear FVG Color", DefaultValue = "#4DFFFF00", Group = "FVG Box Border")]
        public Color BearFvgColor { get; set; }

        [Parameter("Label Color", DefaultValue = "#000000", Group = "FVG Box Border")]
        public Color LabelColor { get; set; }

        // Enabled Timeframes
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

        // Visibility
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

        // Max FVG Settings
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

        // FVG Price Overlay
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
        private readonly Dictionary<string, double> _prevBullHigh2ByTf = new Dictionary<string, double>();
        private readonly Dictionary<string, double> _prevBullLowByTf = new Dictionary<string, double>();
        private readonly Dictionary<string, double> _prevBearLow2ByTf = new Dictionary<string, double>();
        private readonly Dictionary<string, double> _prevBearHighByTf = new Dictionary<string, double>();

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
            if (MtfImb) ProcessMtfSubsystem(index);
            if (MtfPo) ProcessOverlaySubsystem(index);
        }

        private void ValidateMaxFvgCount()
        {
            var total = MaxChart + Max5m + Max10m + Max15m + Max30m + Max1h + Max4h + Max8h + Max12h + MaxDaily + MaxWeekly + MaxMonthly;
            if (total > 500)
                Chart.DrawStaticText("mk_error",
                    "MTF FVG INDICATOR ERROR\n\nMax Number of FVGs exceeded, please change settings.",
                    VerticalAlignment.Bottom, HorizontalAlignment.Right, Color.Red);
        }

        private void RegisterTf(string key, TimeFrame tf, bool enabled, int max)
        {
            if (!enabled)
                return;

            _barsByTf[key] = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf); // MarketData.GetBars(TimeFrame)  [oai_citation:1‡cTrader](https://help.ctrader.com/ctrader-algo/references/MarketData/MarketData/?utm_source=chatgpt.com)
            _bullByTf[key] = new List<FvgZone>();
            _bearByTf[key] = new List<FvgZone>();
            _maxByTf[key] = max;

            _lastBullTfIndex[key] = -1;
            _lastBearTfIndex[key] = -1;

            _prevBullHigh2ByTf[key] = double.NaN;
            _prevBullLowByTf[key] = double.NaN;
            _prevBearLow2ByTf[key] = double.NaN;
            _prevBearHighByTf[key] = double.NaN;
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
                var i = FindBarIndexAtOrBefore(tfBars, GetSecurityAlignmentTime(chartIndex));
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

                var prevH2 = _prevBullHigh2ByTf[tfKey];
                var prevL = _prevBullLowByTf[tfKey];
                var prevL2 = _prevBearLow2ByTf[tfKey];
                var prevH = _prevBearHighByTf[tfKey];

                // Pine: (_highback2 != _highback2[1]) and (_low != _low[1])
                // Allow first valid detection (prev NaN) and require BOTH to differ.
                var bullDistinct = (double.IsNaN(prevH2) || h2 != prevH2) && (double.IsNaN(prevL) || l != prevL);
                var bearDistinct = (double.IsNaN(prevL2) || l2 != prevL2) && (double.IsNaN(prevH) || h != prevH);

                if (newBull && _lastBullTfIndex[tfKey] != i)
                {
                    if (_bullByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bullByTf[tfKey], 0);

                    if (bullDistinct)
                    {
                        AddFvg(tfKey, true, l, h2);
                        _lastBullTfIndex[tfKey] = i;
                    }
                }

                if (newBear && _lastBearTfIndex[tfKey] != i)
                {
                    if (_bearByTf[tfKey].Count > _maxByTf[tfKey])
                        RemoveFvgAt(_bearByTf[tfKey], 0);

                    if (bearDistinct)
                    {
                        AddFvg(tfKey, false, l2, h);
                        _lastBearTfIndex[tfKey] = i;
                    }
                }

                _prevBullHigh2ByTf[tfKey] = h2;
                _prevBullLowByTf[tfKey] = l;
                _prevBearLow2ByTf[tfKey] = l2;
                _prevBearHighByTf[tfKey] = h;

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

            var z = new FvgZone
            {
                Id = id,
                IsBull = bull,
                Top = top,
                Bottom = bottom,
                Rect = rect,
                Label = label,
                TimeframeLabel = tfKey
            };

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

                var threshold = bull
                    ? z.Top - intrPct * (z.Top - z.Bottom)
                    : z.Bottom + intrPct * (z.Top - z.Bottom);

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

                // Pine: Half mitigation uses Close < mid for bulls when Body mitigation is enabled
                var removeHalf = bull
                    ? (UseBodyMitigation ? Bars.ClosePrices[idx] < mid : Bars.LowPrices[idx] < mid)
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
            if (i < 0 || i >= zones.Count)
                return;

            var z = zones[i];
            Chart.RemoveObject(z.Id);
            Chart.RemoveObject(z.Id + "_lbl");
            zones.RemoveAt(i);
        }

        private void ProcessOverlaySubsystem(int idx)
        {
            var tfBars = MarketData.GetBars(TfInput); //  [oai_citation:2‡cTrader](https://help.ctrader.com/ctrader-algo/references/MarketData/MarketData/?utm_source=chatgpt.com)
            var i = FindBarIndexAtOrBefore(tfBars, GetSecurityAlignmentTime(idx));
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

            if (!inRange)
                return;

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

        private string FormatOverlayTimeframeLabel(TimeFrame tf)
        {
            if (tf == Bars.TimeFrame)
            {
                // Closest match to Pine's timeframe.period label
                if (TryGetChartMinutes(out var mins))
                {
                    if (mins < 60)
                        return mins + "m";
                    if (mins % 60 == 0)
                        return (mins / 60) + " Hr";
                    return mins + "m";
                }
                return Bars.TimeFrame.ToString();
            }

            if (tf == TimeFrame.Minute) return "1m";
            if (tf == TimeFrame.Minute2) return "2m";
            if (tf == TimeFrame.Minute3) return "3m";
            if (tf == TimeFrame.Minute4) return "4m";
            if (tf == TimeFrame.Minute5) return "5m";
            if (tf == TimeFrame.Minute10) return "10m";
            if (tf == TimeFrame.Minute15) return "15m";
            if (tf == TimeFrame.Minute30) return "30m";
            if (tf == TimeFrame.Hour) return "1 Hr";
            if (tf == TimeFrame.Hour2) return "2 Hr";
            if (tf == TimeFrame.Hour4) return "4 Hr";
            if (tf == TimeFrame.Hour8) return "8 Hr";
            if (tf == TimeFrame.Hour12) return "12 Hr";
            if (tf == TimeFrame.Daily) return "Daily";
            if (tf == TimeFrame.Weekly) return "Weekly";
            if (tf == TimeFrame.Monthly) return "Monthly";

            return tf.ToString();
        }

        private string GetOverlayBoxText()
        {
            var tfLabel = FormatOverlayTimeframeLabel(TfInput);
            switch (BoxType)
            {
                case OverlayBoxType.Gap:
                    return "GAP • " + tfLabel;
                case OverlayBoxType.Wick:
                    return "WICK • " + tfLabel;
                default:
                    // Imbalance: Pine default imbtext is empty, so the label is essentially the timeframe
                    return tfLabel;
            }
        }

        private void AddOverlay(DateTime left, double top, DateTime right, double bottom, Color fillColor, bool isUp)
        {
            var id = $"mk_overlay_{_id++}";

            // Filled rectangle (bgcolor in Pine)
            var fill = Chart.DrawRectangle(id, left, top, right, bottom, fillColor);
            fill.IsFilled = true;
            fill.IsInteractive = false;
            fill.Thickness = 1;
            fill.LineStyle = LineStyle.Solid;

            // Border rectangle (border_color in Pine)
            var borderColor = isUp ? UpBorderColor : DownBorderColor;
            var border = Chart.DrawRectangle(id + "_bd", left, top, right, bottom, borderColor);
            border.IsFilled = false;
            border.IsInteractive = false;
            border.Thickness = 1;
            border.LineStyle = LineStyle.Solid;

            // Box text (approximation of Pine's box text)
            var txt = Chart.DrawText(id + "_txt", GetOverlayBoxText(), right, (top + bottom) / 2.0, LabelColor);
            txt.IsInteractive = false;

            var t = ShowTopLine ? Chart.DrawTrendLine(id + "_t", left, top, right, top, OverlayLineColor) : null;
            var m = ShowMiddleLine ? Chart.DrawTrendLine(id + "_m", left, (top + bottom) / 2.0, right, (top + bottom) / 2.0, OverlayLineColor) : null;
            var bt = ShowBottomLine ? Chart.DrawTrendLine(id + "_b", left, bottom, right, bottom, OverlayLineColor) : null;

            // Pine default is "Dotted"
            if (t != null) t.LineStyle = LineStyle.Dots;
            if (m != null) m.LineStyle = LineStyle.Dots;
            if (bt != null) bt.LineStyle = LineStyle.Dots;

            _overlayZones.Add(new OverlayZone
            {
                Id = id,
                FillRect = fill,
                BorderRect = border,
                Text = txt,
                TopLine = t,
                MidLine = m,
                BottomLine = bt,
                IsUp = isUp
            });
        }

        private void UpdateOverlayLifecycle(int idx)
        {
            bool Cross(double level) => Bars.HighPrices[idx] > level && Bars.LowPrices[idx] < level;
            var right = Bars.OpenTimes[idx].AddMilliseconds(1);

            for (var i = _overlayZones.Count - 1; i >= 0; i--)
            {
                var z = _overlayZones[i];

                // Normalize top/bottom (Pine uses box.get_top / box.get_bottom)
                var y1 = z.FillRect.Y1;
                var y2 = z.FillRect.Y2;
                var upper = Math.Max(y1, y2);
                var lower = Math.Min(y1, y2);
                var mid = (upper + lower) / 2.0;

                bool filled;
                if (FilledType == FillCondition.Touch)
                {
                    filled = Cross(upper) || Cross(lower);
                }
                else if (FilledType == FillCondition.HalfFill)
                {
                    filled = Cross(mid);
                }
                else
                {
                    // Full Fill: Pine script's current implementation effectively checks the LOWER boundary (box.get_bottom) for all box types
                    filled = Cross(lower);
                }

                // Pine: if hidefilled and filled => delete objects
                if (HideFilled && filled)
                {
                    RemoveOverlayAt(i);
                    continue;
                }

                // Pine: if filled and extendtilfilled => stop extending (keep objects)
                if (filled && ExtendTillFilled)
                {
                    DetachOverlayAt(i);
                    continue;
                }

                // Extend right edges while still managed
                z.FillRect.Time2 = right;
                z.BorderRect.Time2 = right;
                if (z.TopLine != null) z.TopLine.Time2 = right;
                if (z.MidLine != null) z.MidLine.Time2 = right;
                if (z.BottomLine != null) z.BottomLine.Time2 = right;

                if (z.Text != null)
                {
                    z.Text.Time = right;
                    z.Text.Y = mid;
                }

                // Pine: if not filled and not extend => stop extending (keep objects)
                if (!filled && !ExtendTillFilled)
                {
                    DetachOverlayAt(i);
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
            Chart.RemoveObject(z.Id + "_bd");
            Chart.RemoveObject(z.Id + "_txt");
            Chart.RemoveObject(z.Id + "_t");
            Chart.RemoveObject(z.Id + "_m");
            Chart.RemoveObject(z.Id + "_b");

            _overlayZones.RemoveAt(i);
        }

        private void DetachOverlayAt(int i)
        {
            if (i < 0 || i >= _overlayZones.Count) return;
            _overlayZones.RemoveAt(i);
        }

        private bool IsChartTfDuplicatedByEnabledFixedTf()
        {
            if (string.IsNullOrEmpty(_chartTfKey))
                return false;

            return (_chartTfKey == "5m" && Enable5m) ||
                   (_chartTfKey == "10m" && Enable10m) ||
                   (_chartTfKey == "15m" && Enable15m) ||
                   (_chartTfKey == "30m" && Enable30m) ||
                   (_chartTfKey == "1hr" && Enable1h) ||
                   (_chartTfKey == "4hr" && Enable4h) ||
                   (_chartTfKey == "8hr" && Enable8h) ||
                   (_chartTfKey == "12hr" && Enable12h) ||
                   (_chartTfKey == "Daily" && EnableDaily) ||
                   (_chartTfKey == "Weekly" && EnableWeekly) ||
                   (_chartTfKey == "Monthly" && EnableMonthly);
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
            if (tf.IndexOf("Daily", StringComparison.OrdinalIgnoreCase) >= 0) return "Daily";
            if (tf.IndexOf("Weekly", StringComparison.OrdinalIgnoreCase) >= 0) return "Weekly";
            if (tf.IndexOf("Monthly", StringComparison.OrdinalIgnoreCase) >= 0) return "Monthly";
            return "Chart";
        }

        private bool IsTfVisible(string tfKey)
        {
            if (tfKey == "15m") return IsVisibilityEnabled(MinTf15, MaxTf15, monthlyRule: false);
            if (tfKey == "1hr") return IsVisibilityEnabled(MinTf60, MaxTf60Visible, monthlyRule: false);
            if (tfKey == "4hr") return IsVisibilityEnabled(MinTf240, MaxTf240Visible, monthlyRule: false);
            if (tfKey == "Daily") return IsVisibilityEnabled(MinTfD, MaxTfDVisible, monthlyRule: false);
            if (tfKey == "Weekly") return IsVisibilityEnabled(MinTfW, MaxTfWVisible, monthlyRule: false);
            if (tfKey == "Monthly") return IsVisibilityEnabled(MinTfM, MaxTfMVisible, monthlyRule: true);
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

            // Pine logic for D/W/M charts: allow only monthly if monthlyRule, otherwise block.
            return monthlyRule;
        }

        private bool IsDailyOrHigherChart()
        {
            var tf = Bars.TimeFrame.ToString().ToLowerInvariant();
            return tf.Contains("daily") || tf.Contains("weekly") || tf.Contains("monthly");
        }

        private bool TryGetChartMinutes(out int mins)
        {
            mins = 0;
            var tf = Bars.TimeFrame.ToString().ToLowerInvariant();

            if (tf.Contains("minute"))
            {
                // e.g., "Minute5"
                var num = ExtractTrailingInt(tf);
                if (num > 0)
                {
                    mins = num;
                    return true;
                }
            }

            if (tf.Contains("hour"))
            {
                var num = ExtractTrailingInt(tf);
                mins = num > 0 ? num * 60 : 60;
                return true;
            }

            return false;
        }

        private static int ExtractTrailingInt(string s)
        {
            var n = 0;
            var mul = 1;
            for (var i = s.Length - 1; i >= 0; i--)
            {
                var ch = s[i];
                if (ch < '0' || ch > '9')
                    break;

                n += (ch - '0') * mul;
                mul *= 10;
            }
            return n;
        }

        private double GetAtrOnTf(Bars tfBars, int tfIndex, int length)
        {
            if (length <= 0) return 0;
            var start = Math.Max(1, tfIndex - length + 1);
            var sum = 0.0;
            var cnt = 0;

            for (var i = start; i <= tfIndex; i++)
            {
                var tr = GetTrueRange(tfBars, i);
                sum += tr;
                cnt++;
            }

            return cnt > 0 ? sum / cnt : 0.0;
        }

        private static double GetTrueRange(Bars bars, int i)
        {
            if (i <= 0)
                return bars.HighPrices[i] - bars.LowPrices[i];

            var high = bars.HighPrices[i];
            var low = bars.LowPrices[i];
            var prevClose = bars.ClosePrices[i - 1];

            var tr1 = high - low;
            var tr2 = Math.Abs(high - prevClose);
            var tr3 = Math.Abs(low - prevClose);

            return Math.Max(tr1, Math.Max(tr2, tr3));
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

        private DateTime GetSecurityAlignmentTime(int chartIndex)
        {
            return Bars.OpenTimes[chartIndex];
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
