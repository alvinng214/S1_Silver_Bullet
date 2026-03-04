using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SmartMoneyConceptsLuxAlgo : Indicator
    {
        public enum DisplayMode { Historical, Present }
        public enum ThemeStyle { Colored, Monochrome }
        public enum StructureFilter { All, BOS, CHOCH }
        public enum LabelSizeOpt { Tiny, Small, Normal }
        public enum ObFilter { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum LineStyleOpt { Solid, Dashed, Dotted }

        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }
        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }
        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }

        [Parameter("Show Internal Structure", DefaultValue = true, Group = "Real Time Internal Structure")]
        public bool ShowInternalsInput { get; set; }
        [Parameter("Bullish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Internal Structure")]
        public StructureFilter ShowInternalBullInput { get; set; }
        [Parameter("Internal Bull Color", DefaultValue = "#089981", Group = "Real Time Internal Structure")]
        public Color InternalBullColorInput { get; set; }
        [Parameter("Bearish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Internal Structure")]
        public StructureFilter ShowInternalBearInput { get; set; }
        [Parameter("Internal Bear Color", DefaultValue = "#F23645", Group = "Real Time Internal Structure")]
        public Color InternalBearColorInput { get; set; }
        [Parameter("Confluence Filter", DefaultValue = false, Group = "Real Time Internal Structure")]
        public bool InternalFilterConfluenceInput { get; set; }
        [Parameter("Internal Label Size", DefaultValue = LabelSizeOpt.Tiny, Group = "Real Time Internal Structure")]
        public LabelSizeOpt InternalStructureSize { get; set; }

        [Parameter("Show Swing Structure", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowStructureInput { get; set; }
        [Parameter("Bullish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBullInput { get; set; }
        [Parameter("Swing Bull Color", DefaultValue = "#089981", Group = "Real Time Swing Structure")]
        public Color SwingBullishColorInput { get; set; }
        [Parameter("Bearish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBearInput { get; set; }
        [Parameter("Swing Bear Color", DefaultValue = "#F23645", Group = "Real Time Swing Structure")]
        public Color SwingBearishColorInput { get; set; }
        [Parameter("Swing Label Size", DefaultValue = LabelSizeOpt.Small, Group = "Real Time Swing Structure")]
        public LabelSizeOpt SwingStructureSize { get; set; }
        [Parameter("Show Swings Points", DefaultValue = false, Group = "Real Time Swing Structure")]
        public bool ShowSwingsInput { get; set; }
        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Real Time Swing Structure")]
        public int SwingsLengthInput { get; set; }
        [Parameter("Show Strong/Weak High/Low", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowHighLowSwingsInput { get; set; }

        [Parameter("Internal Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowInternalOrderBlocksInput { get; set; }
        [Parameter("Internal OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int InternalOrderBlocksSizeInput { get; set; }
        [Parameter("Swing Order Blocks", DefaultValue = false, Group = "Order Blocks")]
        public bool ShowSwingOrderBlocksInput { get; set; }
        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int SwingOrderBlocksSizeInput { get; set; }
        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "Order Blocks")]
        public ObFilter OrderBlockFilterInput { get; set; }
        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "Order Blocks")]
        public MitigationMode OrderBlockMitigationInput { get; set; }
        [Parameter("Internal Bullish OB", DefaultValue = "#CC3179F5", Group = "Order Blocks")]
        public Color InternalBullishOrderBlockColor { get; set; }
        [Parameter("Internal Bearish OB", DefaultValue = "#CCF77C80", Group = "Order Blocks")]
        public Color InternalBearishOrderBlockColor { get; set; }
        [Parameter("Bullish OB", DefaultValue = "#CC1848CC", Group = "Order Blocks")]
        public Color SwingBullishOrderBlockColor { get; set; }
        [Parameter("Bearish OB", DefaultValue = "#CCB22833", Group = "Order Blocks")]
        public Color SwingBearishOrderBlockColor { get; set; }

        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }
        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }
        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        [Parameter("Fair Value Gaps", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool ShowFairValueGapsInput { get; set; }
        [Parameter("Auto Threshold", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool FairValueGapsThresholdInput { get; set; }
        [Parameter("Bullish FVG", DefaultValue = "#7000FF68", Group = "Fair Value Gaps")]
        public Color FairValueGapsBullColorInput { get; set; }
        [Parameter("Bearish FVG", DefaultValue = "#70FF0008", Group = "Fair Value Gaps")]
        public Color FairValueGapsBearColorInput { get; set; }
        [Parameter("Extend FVG", DefaultValue = 1, MinValue = 0, Group = "Fair Value Gaps")]
        public int FairValueGapsExtendInput { get; set; }

        [Parameter("Daily", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowDailyLevelsInput { get; set; }
        [Parameter("Daily Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt DailyLevelsStyleInput { get; set; }
        [Parameter("Daily Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color DailyLevelsColorInput { get; set; }
        [Parameter("Weekly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowWeeklyLevelsInput { get; set; }
        [Parameter("Weekly Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt WeeklyLevelsStyleInput { get; set; }
        [Parameter("Weekly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color WeeklyLevelsColorInput { get; set; }
        [Parameter("Monthly", DefaultValue = false, Group = "Highs & Lows MTF")]
        public bool ShowMonthlyLevelsInput { get; set; }
        [Parameter("Monthly Style", DefaultValue = LineStyleOpt.Solid, Group = "Highs & Lows MTF")]
        public LineStyleOpt MonthlyLevelsStyleInput { get; set; }
        [Parameter("Monthly Color", DefaultValue = "#2157F3", Group = "Highs & Lows MTF")]
        public Color MonthlyLevelsColorInput { get; set; }

        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }
        [Parameter("Premium Zone", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }
        [Parameter("Equilibrium Zone", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }
        [Parameter("Discount Zone", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

        [Output("Internal Bullish BOS", LineColor = "Lime")]
        public IndicatorDataSeries InternalBullishBOS { get; set; }
        [Output("Internal Bearish BOS", LineColor = "Red")]
        public IndicatorDataSeries InternalBearishBOS { get; set; }
        [Output("Internal Bullish CHoCH", LineColor = "Lime")]
        public IndicatorDataSeries InternalBullishCHoCH { get; set; }
        [Output("Internal Bearish CHoCH", LineColor = "Red")]
        public IndicatorDataSeries InternalBearishCHoCH { get; set; }
        [Output("Swing Bullish BOS", LineColor = "Lime")]
        public IndicatorDataSeries SwingBullishBOS { get; set; }
        [Output("Swing Bearish BOS", LineColor = "Red")]
        public IndicatorDataSeries SwingBearishBOS { get; set; }
        [Output("Swing Bullish CHoCH", LineColor = "Lime")]
        public IndicatorDataSeries SwingBullishCHoCH { get; set; }
        [Output("Swing Bearish CHoCH", LineColor = "Red")]
        public IndicatorDataSeries SwingBearishCHoCH { get; set; }
        [Output("Equal High", LineColor = "DodgerBlue")]
        public IndicatorDataSeries EqualHighSignal { get; set; }
        [Output("Equal Low", LineColor = "DodgerBlue")]
        public IndicatorDataSeries EqualLowSignal { get; set; }
        [Output("Bullish FVG", LineColor = "Lime")]
        public IndicatorDataSeries BullishFvgSignal { get; set; }
        [Output("Bearish FVG", LineColor = "Red")]
        public IndicatorDataSeries BearishFvgSignal { get; set; }

        private int _internalTrend = 0;
        private int _swingTrend = 0;
        private readonly List<OrderBlock> _internalBullObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _internalBearObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBullObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBearObs = new List<OrderBlock>();
        private readonly List<FvgBox> _fvgs = new List<FvgBox>();

        private double _lastSwingHigh = double.NaN;
        private double _lastSwingLow = double.NaN;
        private int _lastSwingHighIndex = -1;
        private int _lastSwingLowIndex = -1;

        private ChartTrendLine _dailyHighLine, _dailyLowLine, _weeklyHighLine, _weeklyLowLine, _monthlyHighLine, _monthlyLowLine;
        private DateTime _lastDay = DateTime.MinValue, _lastWeek = DateTime.MinValue, _lastMonth = DateTime.MinValue;
        private double _dayHigh, _dayLow, _weekHigh, _weekLow, _monthHigh, _monthLow;

        private ChartRectangle _premiumBox, _equilibriumBox, _discountBox;

        protected override void Initialize()
        {
            _dayHigh = _weekHigh = _monthHigh = double.MinValue;
            _dayLow = _weekLow = _monthLow = double.MaxValue;
        }

        public override void Calculate(int index)
        {
            ResetSignals(index);
            if (index < Math.Max(SwingsLengthInput, EqualHighsLowsLengthInput) + 5)
                return;

            if (ShowInternalsInput || ShowStructureInput || ShowSwingsInput || ShowEqualHighsLowsInput || ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateStructure(index);

            if (ShowInternalOrderBlocksInput || ShowSwingOrderBlocksInput)
                UpdateOrderBlocks(index);

            if (ShowFairValueGapsInput)
                UpdateFvgs(index);

            if (ShowDailyLevelsInput || ShowWeeklyLevelsInput || ShowMonthlyLevelsInput)
                UpdateMtfLevels(index);

            if (ShowPremiumDiscountZonesInput)
                UpdatePremiumDiscountZones(index);

            if (ShowTrendInput)
                ColorTrendBar(index);
        }

        private void UpdateStructure(int index)
        {
            var iLen = Math.Max(2, EqualHighsLowsLengthInput);
            var sLen = Math.Max(5, SwingsLengthInput);

            var intPivotHigh = IsPivotHigh(index - iLen, iLen);
            var intPivotLow = IsPivotLow(index - iLen, iLen);
            if (intPivotHigh)
            {
                var lvl = Bars.HighPrices[index - iLen];
                EmitStructure(index, true, false, lvl, true);
            }
            if (intPivotLow)
            {
                var lvl = Bars.LowPrices[index - iLen];
                EmitStructure(index, false, false, lvl, true);
            }

            var swingPivotHigh = IsPivotHigh(index - sLen / 2, sLen / 2);
            var swingPivotLow = IsPivotLow(index - sLen / 2, sLen / 2);
            if (swingPivotHigh)
            {
                _lastSwingHigh = Bars.HighPrices[index - sLen / 2];
                _lastSwingHighIndex = index - sLen / 2;
                EmitStructure(index, true, true, _lastSwingHigh, false);
                if (ShowSwingsInput)
                    Chart.DrawText($"swH_{index}", "HH", _lastSwingHighIndex, _lastSwingHigh, SwingBearishColorInput);
            }
            if (swingPivotLow)
            {
                _lastSwingLow = Bars.LowPrices[index - sLen / 2];
                _lastSwingLowIndex = index - sLen / 2;
                EmitStructure(index, false, true, _lastSwingLow, false);
                if (ShowSwingsInput)
                    Chart.DrawText($"swL_{index}", "LL", _lastSwingLowIndex, _lastSwingLow, SwingBullishColorInput);
            }

            if (ShowEqualHighsLowsInput)
                DetectEqualHighLow(index, iLen);

            if (ShowHighLowSwingsInput)
                DrawHighLowSwings(index);
        }

        private void EmitStructure(int index, bool bullishBreak, bool swing, double level, bool internal)
        {
            var close = Bars.ClosePrices[index];
            if (bullishBreak)
            {
                if (close > level)
                {
                    var choch = (internal ? _internalTrend : _swingTrend) < 0;
                    if (internal)
                    {
                        if (choch) InternalBullishCHoCH[index] = 1; else InternalBullishBOS[index] = 1;
                        _internalTrend = 1;
                        if (ShowInternalsInput && ShouldShow(ShowInternalBullInput, choch))
                            DrawStructureLine(index, level, choch ? "iCHoCH" : "iBOS", InternalBullColorInput, InternalStructureSize);
                    }
                    else
                    {
                        if (choch) SwingBullishCHoCH[index] = 1; else SwingBullishBOS[index] = 1;
                        _swingTrend = 1;
                        if (ShowStructureInput && ShouldShow(ShowSwingBullInput, choch))
                            DrawStructureLine(index, level, choch ? "CHoCH" : "BOS", SwingBullishColorInput, SwingStructureSize);
                    }
                }
            }
            else
            {
                if (close < level)
                {
                    var choch = (internal ? _internalTrend : _swingTrend) > 0;
                    if (internal)
                    {
                        if (choch) InternalBearishCHoCH[index] = 1; else InternalBearishBOS[index] = 1;
                        _internalTrend = -1;
                        if (ShowInternalsInput && ShouldShow(ShowInternalBearInput, choch))
                            DrawStructureLine(index, level, choch ? "iCHoCH" : "iBOS", InternalBearColorInput, InternalStructureSize);
                    }
                    else
                    {
                        if (choch) SwingBearishCHoCH[index] = 1; else SwingBearishBOS[index] = 1;
                        _swingTrend = -1;
                        if (ShowStructureInput && ShouldShow(ShowSwingBearInput, choch))
                            DrawStructureLine(index, level, choch ? "CHoCH" : "BOS", SwingBearishColorInput, SwingStructureSize);
                    }
                }
            }
        }

        private void DrawStructureLine(int index, double level, string label, Color color, LabelSizeOpt size)
        {
            var x1 = Math.Max(index - 10, 0);
            var line = Chart.DrawTrendLine($"smc_{label}_{index}", x1, level, index, level, color, 1, LineStyle.Solid);
            if (ModeInput == DisplayMode.Present)
                RemoveOld("smc_", 100);
            Chart.DrawText($"smct_{index}", label, index, level, color);
        }

        private void DetectEqualHighLow(int index, int len)
        {
            var p = index - len;
            if (p <= len || p >= Bars.Count)
                return;

            if (IsPivotHigh(p, len) && !double.IsNaN(_lastSwingHigh))
            {
                var v = Bars.HighPrices[p];
                var thr = Math.Max(Symbol.TickSize, EqualHighsLowsThresholdInput * (Bars.HighPrices[index] - Bars.LowPrices[index]));
                if (Math.Abs(v - _lastSwingHigh) <= thr)
                {
                    EqualHighSignal[index] = 1;
                    var l = Chart.DrawTrendLine($"eqh_{index}", _lastSwingHighIndex, _lastSwingHigh, p, v, Color.DodgerBlue, 1, LineStyle.DotsRare);
                    Chart.DrawText($"eqht_{index}", "EQH", p, v, Color.DodgerBlue);
                }
            }
            if (IsPivotLow(p, len) && !double.IsNaN(_lastSwingLow))
            {
                var v = Bars.LowPrices[p];
                var thr = Math.Max(Symbol.TickSize, EqualHighsLowsThresholdInput * (Bars.HighPrices[index] - Bars.LowPrices[index]));
                if (Math.Abs(v - _lastSwingLow) <= thr)
                {
                    EqualLowSignal[index] = 1;
                    var l = Chart.DrawTrendLine($"eql_{index}", _lastSwingLowIndex, _lastSwingLow, p, v, Color.DodgerBlue, 1, LineStyle.DotsRare);
                    Chart.DrawText($"eqlt_{index}", "EQL", p, v, Color.DodgerBlue);
                }
            }
        }

        private void UpdateOrderBlocks(int index)
        {
            if (index < 3)
                return;

            var atrLike = AverageRange(index, 14);
            var threshold = OrderBlockFilterInput == ObFilter.Atr ? atrLike : AverageRange(index, Math.Min(index, 100));

            if (InternalBullishBOS[index] > 0 || InternalBullishCHoCH[index] > 0)
                AddOrderBlock(_internalBullObs, index - 1, true, true);
            if (InternalBearishBOS[index] > 0 || InternalBearishCHoCH[index] > 0)
                AddOrderBlock(_internalBearObs, index - 1, false, true);
            if (SwingBullishBOS[index] > 0 || SwingBullishCHoCH[index] > 0)
                AddOrderBlock(_swingBullObs, index - 1, true, false);
            if (SwingBearishBOS[index] > 0 || SwingBearishCHoCH[index] > 0)
                AddOrderBlock(_swingBearObs, index - 1, false, false);

            ManageObList(_internalBullObs, index, true, InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBullishOrderBlockColor, threshold);
            ManageObList(_internalBearObs, index, false, InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBearishOrderBlockColor, threshold);
            ManageObList(_swingBullObs, index, true, SwingOrderBlocksSizeInput, ShowSwingOrderBlocksInput, SwingBullishOrderBlockColor, threshold);
            ManageObList(_swingBearObs, index, false, SwingOrderBlocksSizeInput, ShowSwingOrderBlocksInput, SwingBearishOrderBlockColor, threshold);
        }

        private void AddOrderBlock(List<OrderBlock> list, int i, bool bullish, bool internal)
        {
            if (i < 1) return;
            var top = Math.Max(Bars.OpenPrices[i], Bars.ClosePrices[i]);
            var bottom = Math.Min(Bars.OpenPrices[i], Bars.ClosePrices[i]);
            list.Insert(0, new OrderBlock { Index = i, Top = top, Bottom = bottom, Bullish = bullish, Internal = internal });
        }

        private void ManageObList(List<OrderBlock> list, int index, bool bullish, int keep, bool show, Color color, double threshold)
        {
            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];
                if ((ob.Top - ob.Bottom) > Math.Max(Symbol.TickSize, threshold * 3))
                {
                    list.RemoveAt(i);
                    continue;
                }

                var mitigated = OrderBlockMitigationInput == MitigationMode.Close
                    ? (bullish ? Bars.ClosePrices[index] < ob.Bottom : Bars.ClosePrices[index] > ob.Top)
                    : (bullish ? Bars.LowPrices[index] < ob.Bottom : Bars.HighPrices[index] > ob.Top);

                if (mitigated)
                {
                    list.RemoveAt(i);
                    continue;
                }

                if (show)
                {
                    var name = $"ob_{(ob.Internal ? "i" : "s")}_{(bullish ? "b" : "r")}_{ob.Index}";
                    var rect = Chart.DrawRectangle(name, ob.Index, ob.Top, index + 1, ob.Bottom, color);
                    rect.IsFilled = true;
                    rect.Color = color;
                }
            }

            while (list.Count > keep)
                list.RemoveAt(list.Count - 1);
        }

        private void UpdateFvgs(int index)
        {
            if (index < 2)
                return;

            var autoTh = FairValueGapsThresholdInput ? Math.Max(Symbol.TickSize * 2, AverageRange(index, 20) * 0.15) : 0;
            var bull = Bars.LowPrices[index] > Bars.HighPrices[index - 2] && (Bars.LowPrices[index] - Bars.HighPrices[index - 2]) > autoTh;
            var bear = Bars.HighPrices[index] < Bars.LowPrices[index - 2] && (Bars.LowPrices[index - 2] - Bars.HighPrices[index]) > autoTh;

            if (bull)
            {
                BullishFvgSignal[index] = 1;
                DrawFvg(index - 2, index + FairValueGapsExtendInput, Bars.LowPrices[index], Bars.HighPrices[index - 2], FairValueGapsBullColorInput, true);
            }
            if (bear)
            {
                BearishFvgSignal[index] = 1;
                DrawFvg(index - 2, index + FairValueGapsExtendInput, Bars.LowPrices[index - 2], Bars.HighPrices[index], FairValueGapsBearColorInput, false);
            }
        }

        private void DrawFvg(int left, int right, double top, double bottom, Color color, bool bullish)
        {
            var name = $"fvg_{(bullish ? "b" : "r")}_{left}_{right}";
            var rect = Chart.DrawRectangle(name, left, top, Math.Min(right, Bars.Count - 1), bottom, color);
            rect.IsFilled = true;
            rect.Color = color;
        }

        private void UpdateMtfLevels(int index)
        {
            var t = Bars.OpenTimes[index];
            if (_lastDay == DateTime.MinValue || t.Date != _lastDay.Date)
            {
                DrawPeriodLevels(index, ref _dailyHighLine, ref _dailyLowLine, _dayHigh, _dayLow, DailyLevelsColorInput, DailyLevelsStyleInput, "D");
                _dayHigh = Bars.HighPrices[index]; _dayLow = Bars.LowPrices[index]; _lastDay = t.Date;
            }
            else
            {
                _dayHigh = Math.Max(_dayHigh, Bars.HighPrices[index]);
                _dayLow = Math.Min(_dayLow, Bars.LowPrices[index]);
            }

            var week = FirstDateOfWeek(t);
            if (_lastWeek == DateTime.MinValue || week != _lastWeek)
            {
                DrawPeriodLevels(index, ref _weeklyHighLine, ref _weeklyLowLine, _weekHigh, _weekLow, WeeklyLevelsColorInput, WeeklyLevelsStyleInput, "W");
                _weekHigh = Bars.HighPrices[index]; _weekLow = Bars.LowPrices[index]; _lastWeek = week;
            }
            else
            {
                _weekHigh = Math.Max(_weekHigh, Bars.HighPrices[index]);
                _weekLow = Math.Min(_weekLow, Bars.LowPrices[index]);
            }

            var month = new DateTime(t.Year, t.Month, 1);
            if (_lastMonth == DateTime.MinValue || month != _lastMonth)
            {
                DrawPeriodLevels(index, ref _monthlyHighLine, ref _monthlyLowLine, _monthHigh, _monthLow, MonthlyLevelsColorInput, MonthlyLevelsStyleInput, "M");
                _monthHigh = Bars.HighPrices[index]; _monthLow = Bars.LowPrices[index]; _lastMonth = month;
            }
            else
            {
                _monthHigh = Math.Max(_monthHigh, Bars.HighPrices[index]);
                _monthLow = Math.Min(_monthLow, Bars.LowPrices[index]);
            }
        }

        private void DrawPeriodLevels(int index, ref ChartTrendLine top, ref ChartTrendLine bottom, double high, double low, Color color, LineStyleOpt style, string tag)
        {
            if (double.IsInfinity(high) || double.IsInfinity(low) || high == double.MinValue || low == double.MaxValue)
                return;

            var show = (tag == "D" && ShowDailyLevelsInput) || (tag == "W" && ShowWeeklyLevelsInput) || (tag == "M" && ShowMonthlyLevelsInput);
            if (!show) return;

            top = Chart.DrawTrendLine($"lvl_{tag}_h_{index}", index - 1, high, index + 10, high, color, 1, MapLineStyle(style));
            bottom = Chart.DrawTrendLine($"lvl_{tag}_l_{index}", index - 1, low, index + 10, low, color, 1, MapLineStyle(style));
            top.ExtendToInfinity = true;
            bottom.ExtendToInfinity = true;
            Chart.DrawText($"lvl_{tag}_ht_{index}", $"P{tag}H", index, high, color);
            Chart.DrawText($"lvl_{tag}_lt_{index}", $"P{tag}L", index, low, color);
        }

        private void DrawHighLowSwings(int index)
        {
            if (!double.IsNaN(_lastSwingHigh))
            {
                var l = Chart.DrawTrendLine($"wh_{index}", _lastSwingHighIndex, _lastSwingHigh, index + 1, _lastSwingHigh, SwingBearishColorInput, 1, LineStyle.DotsRare);
                Chart.DrawText($"wht_{index}", Bars.ClosePrices[index] > _lastSwingHigh ? "Strong High" : "Weak High", index, _lastSwingHigh, SwingBearishColorInput);
            }
            if (!double.IsNaN(_lastSwingLow))
            {
                var l = Chart.DrawTrendLine($"wl_{index}", _lastSwingLowIndex, _lastSwingLow, index + 1, _lastSwingLow, SwingBullishColorInput, 1, LineStyle.DotsRare);
                Chart.DrawText($"wlt_{index}", Bars.ClosePrices[index] < _lastSwingLow ? "Strong Low" : "Weak Low", index, _lastSwingLow, SwingBullishColorInput);
            }
        }

        private void UpdatePremiumDiscountZones(int index)
        {
            if (double.IsNaN(_lastSwingHigh) || double.IsNaN(_lastSwingLow) || _lastSwingHigh <= _lastSwingLow)
                return;

            var top = _lastSwingHigh;
            var bottom = _lastSwingLow;
            var premiumTop = top;
            var premiumBottom = 0.95 * top + 0.05 * bottom;
            var eqTop = 0.525 * top + 0.475 * bottom;
            var eqBottom = 0.525 * bottom + 0.475 * top;
            var discountTop = 0.95 * bottom + 0.05 * top;
            var discountBottom = bottom;

            _premiumBox = DrawZoneRect(_premiumBox, $"premium_{index}", index - 50, premiumTop, index + 50, premiumBottom, PremiumZoneColorInput, "Premium");
            _equilibriumBox = DrawZoneRect(_equilibriumBox, $"eq_{index}", index - 50, eqTop, index + 50, eqBottom, EquilibriumZoneColorInput, "Equilibrium");
            _discountBox = DrawZoneRect(_discountBox, $"discount_{index}", index - 50, discountTop, index + 50, discountBottom, DiscountZoneColorInput, "Discount");
        }

        private ChartRectangle DrawZoneRect(ChartRectangle existing, string name, int x1, double y1, int x2, double y2, Color color, string text)
        {
            var rect = Chart.DrawRectangle(name, Math.Max(0, x1), y1, Math.Min(Bars.Count - 1, x2), y2, Color.FromArgb(80, color));
            rect.IsFilled = true;
            rect.Color = Color.FromArgb(80, color);
            Chart.DrawText($"{name}_t", text, Math.Min(Bars.Count - 1, x2), (y1 + y2) / 2.0, color);
            return rect;
        }

        private void ColorTrendBar(int index)
        {
            var color = _swingTrend >= 0
                ? (StyleInput == ThemeStyle.Colored ? SwingBullishColorInput : Color.FromHex("#B2B5BE"))
                : (StyleInput == ThemeStyle.Colored ? SwingBearishColorInput : Color.FromHex("#5D606B"));
            Chart.SetBarColor(index, color);
        }

        private bool ShouldShow(StructureFilter filter, bool choch)
        {
            if (filter == StructureFilter.All) return true;
            if (filter == StructureFilter.BOS && !choch) return true;
            if (filter == StructureFilter.CHOCH && choch) return true;
            return false;
        }

        private bool IsPivotHigh(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.HighPrices[i];
            for (var j = i - len; j <= i + len; j++)
            {
                if (j == i) continue;
                if (Bars.HighPrices[j] >= p) return false;
            }
            return true;
        }

        private bool IsPivotLow(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.LowPrices[i];
            for (var j = i - len; j <= i + len; j++)
            {
                if (j == i) continue;
                if (Bars.LowPrices[j] <= p) return false;
            }
            return true;
        }

        private double AverageRange(int index, int len)
        {
            var start = Math.Max(1, index - len + 1);
            var sum = 0.0;
            var n = 0;
            for (var i = start; i <= index; i++)
            {
                sum += Bars.HighPrices[i] - Bars.LowPrices[i];
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        private static DateTime FirstDateOfWeek(DateTime dt)
        {
            var diff = (7 + (dt.DayOfWeek - DayOfWeek.Monday)) % 7;
            return dt.Date.AddDays(-1 * diff);
        }

        private LineStyle MapLineStyle(LineStyleOpt style)
        {
            switch (style)
            {
                case LineStyleOpt.Dashed: return LineStyle.Lines;
                case LineStyleOpt.Dotted: return LineStyle.DotsRare;
                default: return LineStyle.Solid;
            }
        }

        private void RemoveOld(string prefix, int keep)
        {
            var count = 0;
            foreach (var o in Chart.Objects)
            {
                if (o.Name.StartsWith(prefix)) count++;
            }
            if (count <= keep) return;
        }

        private void ResetSignals(int index)
        {
            InternalBullishBOS[index] = InternalBearishBOS[index] = 0;
            InternalBullishCHoCH[index] = InternalBearishCHoCH[index] = 0;
            SwingBullishBOS[index] = SwingBearishBOS[index] = 0;
            SwingBullishCHoCH[index] = SwingBearishCHoCH[index] = 0;
            EqualHighSignal[index] = EqualLowSignal[index] = 0;
            BullishFvgSignal[index] = BearishFvgSignal[index] = 0;
        }

        private sealed class OrderBlock
        {
            public int Index;
            public double Top;
            public double Bottom;
            public bool Bullish;
            public bool Internal;
        }

        private sealed class FvgBox
        {
            public int Left;
            public int Right;
            public double Top;
            public double Bottom;
            public bool Bullish;
        }
    }
}
