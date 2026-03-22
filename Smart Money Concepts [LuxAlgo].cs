using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SmartMoneyConceptsLuxAlgo : Indicator
    {
        public enum DisplayMode      { Historical, Present }
        public enum ThemeStyle       { Colored, Monochrome }
        public enum StructureFilter  { All, BOS, CHOCH }
        public enum LabelSizeOpt     { Tiny, Small, Normal }
        public enum ObFilter         { Atr, CumulativeMeanRange }
        public enum MitigationMode   { Close, HighLow }
        public enum LineStyleOpt     { Solid, Dashed, Dotted }

        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }
        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }
        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }
        [Parameter("Plot Signal Series", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool PlotSignalSeriesInput { get; set; }

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

        // ── Outputs ───────────────────────────────────────────────────────────
        [Output("Internal Bullish BOS",   LineColor = "Lime")]       public IndicatorDataSeries InternalBullishBOS   { get; set; }
        [Output("Internal Bearish BOS",   LineColor = "Red")]        public IndicatorDataSeries InternalBearishBOS   { get; set; }
        [Output("Internal Bullish CHoCH", LineColor = "Lime")]       public IndicatorDataSeries InternalBullishCHoCH { get; set; }
        [Output("Internal Bearish CHoCH", LineColor = "Red")]        public IndicatorDataSeries InternalBearishCHoCH { get; set; }
        [Output("Swing Bullish BOS",      LineColor = "Lime")]       public IndicatorDataSeries SwingBullishBOS      { get; set; }
        [Output("Swing Bearish BOS",      LineColor = "Red")]        public IndicatorDataSeries SwingBearishBOS      { get; set; }
        [Output("Swing Bullish CHoCH",    LineColor = "Lime")]       public IndicatorDataSeries SwingBullishCHoCH    { get; set; }
        [Output("Swing Bearish CHoCH",    LineColor = "Red")]        public IndicatorDataSeries SwingBearishCHoCH    { get; set; }
        [Output("Equal High",             LineColor = "DodgerBlue")] public IndicatorDataSeries EqualHighSignal      { get; set; }
        [Output("Equal Low",              LineColor = "DodgerBlue")] public IndicatorDataSeries EqualLowSignal       { get; set; }
        [Output("Bullish FVG",            LineColor = "Lime")]       public IndicatorDataSeries BullishFvgSignal     { get; set; }
        [Output("Bearish FVG",            LineColor = "Red")]        public IndicatorDataSeries BearishFvgSignal     { get; set; }

        // ── Private fields ────────────────────────────────────────────────────

        private int  _internalTrend, _swingTrend, _internalLeg, _swingLeg;
        private bool _evtInternalBull, _evtInternalBear, _evtSwingBull, _evtSwingBear;

        private readonly List<OrderBlock> _internalBullObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _internalBearObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBullObs    = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBearObs    = new List<OrderBlock>();

        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;

        private double _internalHighLevel  = double.NaN;
        private double _internalLowLevel   = double.NaN;
        private bool   _internalHighCrossed;
        private bool   _internalLowCrossed;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;

        private double _prevEqHigh      = double.NaN;
        private int    _prevEqHighIndex = -1;
        private double _prevEqLow       = double.NaN;
        private int    _prevEqLowIndex  = -1;

        private int _internalHighIndex = -1;
        private int _internalLowIndex  = -1;

        private readonly List<double>   _parsedHighs = new List<double>();
        private readonly List<double>   _parsedLows  = new List<double>();
        private readonly List<DateTime> _times       = new List<DateTime>();

        // ── FIX 1: Wilder's ATR (RMA) to match Pine's ta.atr(200) ────────────
        //
        // ORIGINAL BUG: AverageTrueRangeSimple() used a simple moving average
        // (SMA) of TR.  Pine's ta.atr(N) uses Wilder's smoothing (RMA):
        //   ATR[i] = (ATR[i-1] × (N-1) + TR[i]) / N
        // With N=200 the SMA and RMA diverge from bar 200 onward, causing
        // ~3% of bars to receive a different highVolatilityBar flag, which
        // flips their parsedHigh/parsedLow values and shifts the OB anchor
        // bar selected in StoreOrderBlockFromPivot.
        //
        // FIX: _atrWilder is computed incrementally in UpdateParsedArrays()
        // using Wilder's smoothing, matching ta.atr(200) exactly.
        // _atrWilderSum accumulates TR for the initial SMA seed (bars 1..200).
        private const int AtrPeriod = 200;
        private double _atrWilder    = double.NaN;  // current Wilder ATR value
        private double _atrWilderSum = 0.0;          // accumulator for seed SMA

        private double _cumTr;

        private ChartTrendLine _dailyHighLine,   _dailyLowLine;
        private ChartTrendLine _weeklyHighLine,  _weeklyLowLine;
        private ChartTrendLine _monthlyHighLine, _monthlyLowLine;
        private DateTime _lastDay   = DateTime.MinValue;
        private DateTime _lastWeek  = DateTime.MinValue;
        private DateTime _lastMonth = DateTime.MinValue;
        private double _dayHigh, _dayLow, _weekHigh, _weekLow, _monthHigh, _monthLow;

        private ChartRectangle _premiumBox, _equilibriumBox, _discountBox;
        private int       _obIdCounter;
        private const int MaxStoredOrderBlocksPerSide = 500;

        private double   _trailingTop            = double.MinValue;
        private double   _trailingBottom         = double.MaxValue;
        private DateTime _trailingBarTime        = DateTime.MinValue;
        private int      _trailingBarIndex       = -1;
        private DateTime _trailingLastTopTime    = DateTime.MinValue;
        private DateTime _trailingLastBottomTime = DateTime.MinValue;

        // ── Initialize ────────────────────────────────────────────────────────

        protected override void Initialize()
        {
            _dayHigh   = _weekHigh   = _monthHigh  = double.MinValue;
            _dayLow    = _weekLow    = _monthLow   = double.MaxValue;
            _trailingTop    = double.MinValue;
            _trailingBottom = double.MaxValue;
        }

        // ── Calculate ─────────────────────────────────────────────────────────

        public override void Calculate(int index)
        {
            ResetSignals(index);
            UpdateParsedArrays(index);

            if (index < Math.Max(SwingsLengthInput, EqualHighsLowsLengthInput) + 5)
                return;

            if (ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateTrailingExtremes(index);

            if (ShowHighLowSwingsInput)
                DrawHighLowSwings(index);

            if (ShowPremiumDiscountZonesInput)
                UpdatePremiumDiscountZones(index);

            if (ShowInternalsInput  || ShowStructureInput  || ShowSwingsInput ||
                ShowEqualHighsLowsInput || ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateStructure(index);

            if (ShowInternalOrderBlocksInput || ShowSwingOrderBlocksInput)
                UpdateOrderBlocks(index);

            if (ShowFairValueGapsInput)
                UpdateFvgs(index);

            if (ShowDailyLevelsInput || ShowWeeklyLevelsInput || ShowMonthlyLevelsInput)
                UpdateMtfLevels(index);

            if (ShowTrendInput)
                ColorTrendBar(index);
        }

        // ── Parsed arrays ─────────────────────────────────────────────────────

        private void UpdateParsedArrays(int index)
        {
            // ── FIX 1a: Wilder's ATR (RMA) ────────────────────────────────────
            // Computes ta.atr(200) equivalent using Wilder's smoothing:
            //   Seed: SMA of TR[1..200] at bar 200.
            //   Then: ATR[i] = (ATR[i-1] × 199 + TR[i]) / 200 for i > 200.
            // _atrWilder is double.NaN before bar 200 (mirrors Pine's na).
            double tr;
            if (index == 0)
            {
                _cumTr       = 0;
                _atrWilderSum = 0;
                _atrWilder   = double.NaN;
                tr = Bars.HighPrices[0] - Bars.LowPrices[0];
            }
            else
            {
                var prevClose = Bars.ClosePrices[index - 1];
                tr = Math.Max(
                    Bars.HighPrices[index] - Bars.LowPrices[index],
                    Math.Max(
                        Math.Abs(Bars.HighPrices[index] - prevClose),
                        Math.Abs(Bars.LowPrices[index]  - prevClose)));
                _cumTr += tr;

                if (index < AtrPeriod)
                {
                    // Accumulate TR for the seed SMA; ATR not yet valid.
                    _atrWilderSum += tr;
                    _atrWilder     = double.NaN;
                }
                else if (index == AtrPeriod)
                {
                    // Seed: SMA of the first AtrPeriod TR values (bars 1..200).
                    _atrWilderSum += tr;
                    _atrWilder     = _atrWilderSum / AtrPeriod;
                }
                else
                {
                    // Wilder's smoothing: identical to Pine's ta.atr(200).
                    _atrWilder = (_atrWilder * (AtrPeriod - 1) + tr) / AtrPeriod;
                }
            }

            // ── Volatility measure ────────────────────────────────────────────
            double volatilityMeasure;
            if (OrderBlockFilterInput == ObFilter.Atr)
            {
                // Use Wilder ATR; before bar 200 it is NaN so highVolatilityBar
                // will be false (no bar classified as high-vol), matching Pine.
                volatilityMeasure = double.IsNaN(_atrWilder) ? double.MaxValue : _atrWilder;
            }
            else
            {
                volatilityMeasure = _cumTr / Math.Max(1, index);
            }

            var highVolatilityBar = (Bars.HighPrices[index] - Bars.LowPrices[index])
                                    >= 2.0 * volatilityMeasure;
            _parsedHighs.Add(highVolatilityBar ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( highVolatilityBar ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        // ── Structure ─────────────────────────────────────────────────────────

        private void UpdateStructure(int index)
        {
            var iLen = 5;
            var sLen = Math.Max(5, SwingsLengthInput);

            var internalLegNow    = ComputeLeg(index, iLen, _internalLeg);
            var internalLegChange = internalLegNow - _internalLeg;
            if (internalLegChange != 0)
            {
                if (internalLegChange == 1)
                {
                    _internalLowLevel   = Bars.LowPrices[index - iLen];
                    _internalLowIndex   = index - iLen;
                    _internalLowCrossed = false;
                }
                else
                {
                    _internalHighLevel   = Bars.HighPrices[index - iLen];
                    _internalHighIndex   = index - iLen;
                    _internalHighCrossed = false;
                }
            }
            _internalLeg = internalLegNow;

            var swingLegNow    = ComputeLeg(index, sLen, _swingLeg);
            var swingLegChange = swingLegNow - _swingLeg;
            if (swingLegChange != 0)
            {
                if (swingLegChange == 1)
                {
                    _lastSwingLow      = Bars.LowPrices[index - sLen];
                    _lastSwingLowIndex = index - sLen;
                    _swingLowCrossed   = false;
                    if (ShowSwingsInput)
                        Chart.DrawText($"swL_{index}", "LL", _lastSwingLowIndex, _lastSwingLow, SwingBullishColorInput);
                }
                else
                {
                    _lastSwingHigh      = Bars.HighPrices[index - sLen];
                    _lastSwingHighIndex = index - sLen;
                    _swingHighCrossed   = false;
                    if (ShowSwingsInput)
                        Chart.DrawText($"swH_{index}", "HH", _lastSwingHighIndex, _lastSwingHigh, SwingBearishColorInput);
                }
            }
            _swingLeg = swingLegNow;

            ProcessStructureCrosses(index);

            if (ShowEqualHighsLowsInput)
                DetectEqualHighLow(index, iLen);
        }

        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;

            var refHigh = Bars.HighPrices[index - size];
            var refLow  = Bars.LowPrices[index - size];
            var highest = double.MinValue;
            var lowest  = double.MaxValue;
            var start   = Math.Max(0, index - size + 1);
            for (var i = start; i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lowest)  lowest  = Bars.LowPrices[i];
            }
            if (refHigh > highest) return 0;
            if (refLow  < lowest)  return 1;
            return previousLeg;
        }

        private void ProcessStructureCrosses(int index)
        {
            var close = Bars.ClosePrices[index];

            if (!double.IsNaN(_internalHighLevel) && !_internalHighCrossed && close > _internalHighLevel)
            {
                _internalHighCrossed = true;
                EmitStructure(index, true,  false, _internalHighLevel, true);
                if (ShowInternalOrderBlocksInput)
                    StoreOrderBlockFromPivot(_internalHighIndex, true,  1, index);
            }
            if (!double.IsNaN(_internalLowLevel) && !_internalLowCrossed && close < _internalLowLevel)
            {
                _internalLowCrossed = true;
                EmitStructure(index, false, false, _internalLowLevel,  true);
                if (ShowInternalOrderBlocksInput)
                    StoreOrderBlockFromPivot(_internalLowIndex,  true, -1, index);
            }
            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed && close > _lastSwingHigh)
            {
                _swingHighCrossed = true;
                EmitStructure(index, true,  true, _lastSwingHigh, false);
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingHighIndex, false,  1, index);
            }
            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed && close < _lastSwingLow)
            {
                _swingLowCrossed = true;
                EmitStructure(index, false, true, _lastSwingLow,  false);
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingLowIndex,  false, -1, index);
            }
        }

        private void EmitStructure(int index, bool bullishBreak, bool swing, double level, bool isInternal)
        {
            var close = Bars.ClosePrices[index];
            if (bullishBreak)
            {
                if (close > level)
                {
                    var choch = (isInternal ? _internalTrend : _swingTrend) < 0;
                    if (isInternal)
                    {
                        if (choch) SetSignal(InternalBullishCHoCH, index, close);
                        else       SetSignal(InternalBullishBOS,   index, close);
                        _internalTrend   = 1;
                        _evtInternalBull = true;
                        if (ShowInternalsInput && ShouldShow(ShowInternalBullInput, choch))
                            DrawStructureLine(index, level, choch ? "iCHoCH" : "iBOS", InternalBullColorInput, InternalStructureSize);
                    }
                    else
                    {
                        if (choch) SetSignal(SwingBullishCHoCH, index, close);
                        else       SetSignal(SwingBullishBOS,   index, close);
                        _swingTrend   = 1;
                        _evtSwingBull = true;
                        if (ShowStructureInput && ShouldShow(ShowSwingBullInput, choch))
                            DrawStructureLine(index, level, choch ? "CHoCH" : "BOS", SwingBullishColorInput, SwingStructureSize);
                    }
                }
            }
            else
            {
                if (close < level)
                {
                    var choch = (isInternal ? _internalTrend : _swingTrend) > 0;
                    if (isInternal)
                    {
                        if (choch) SetSignal(InternalBearishCHoCH, index, close);
                        else       SetSignal(InternalBearishBOS,   index, close);
                        _internalTrend   = -1;
                        _evtInternalBear = true;
                        if (ShowInternalsInput && ShouldShow(ShowInternalBearInput, choch))
                            DrawStructureLine(index, level, choch ? "iCHoCH" : "iBOS", InternalBearColorInput, InternalStructureSize);
                    }
                    else
                    {
                        if (choch) SetSignal(SwingBearishCHoCH, index, close);
                        else       SetSignal(SwingBearishBOS,   index, close);
                        _swingTrend   = -1;
                        _evtSwingBear = true;
                        if (ShowStructureInput && ShouldShow(ShowSwingBearInput, choch))
                            DrawStructureLine(index, level, choch ? "CHoCH" : "BOS", SwingBearishColorInput, SwingStructureSize);
                    }
                }
            }
        }

        private void DrawStructureLine(int index, double level, string label, Color color, LabelSizeOpt size)
        {
            var x1 = Math.Max(index - 10, 0);
            Chart.DrawTrendLine($"smc_{label}_{index}", x1, level, index, level, color, 1, LineStyle.Solid);
            if (ModeInput == DisplayMode.Present) RemoveOld("smc_", 100);
            Chart.DrawText($"smct_{index}", label, index, level, color);
        }

        private void DetectEqualHighLow(int index, int len)
        {
            var p = index - len;
            if (p <= len || p >= Bars.Count) return;

            var range = Math.Max(Symbol.TickSize, Bars.HighPrices[index] - Bars.LowPrices[index]);
            var thr   = Math.Max(Symbol.TickSize, EqualHighsLowsThresholdInput * range);

            if (IsPivotHigh(p, len))
            {
                var v = Bars.HighPrices[p];
                if (!double.IsNaN(_prevEqHigh) && Math.Abs(v - _prevEqHigh) <= thr)
                {
                    SetSignal(EqualHighSignal, index, v);
                    Chart.DrawTrendLine("smc_eqh_line", _prevEqHighIndex, _prevEqHigh, p, v, Color.DodgerBlue, 1, LineStyle.DotsRare);
                    Chart.DrawText("smc_eqh_label", "EQH", p, v, Color.DodgerBlue);
                }
                _prevEqHigh = v; _prevEqHighIndex = p;
            }
            if (IsPivotLow(p, len))
            {
                var v = Bars.LowPrices[p];
                if (!double.IsNaN(_prevEqLow) && Math.Abs(v - _prevEqLow) <= thr)
                {
                    SetSignal(EqualLowSignal, index, v);
                    Chart.DrawTrendLine("smc_eql_line", _prevEqLowIndex, _prevEqLow, p, v, Color.DodgerBlue, 1, LineStyle.DotsRare);
                    Chart.DrawText("smc_eql_label", "EQL", p, v, Color.DodgerBlue);
                }
                _prevEqLow = v; _prevEqLowIndex = p;
            }
        }

        // ── Order Blocks ──────────────────────────────────────────────────────

        private void UpdateOrderBlocks(int index)
        {
            if (index < 3) return;
            ManageObList(_internalBullObs, index, true,  InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBullishOrderBlockColor);
            ManageObList(_internalBearObs, index, false, InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBearishOrderBlockColor);
            ManageObList(_swingBullObs,    index, true,  SwingOrderBlocksSizeInput,    ShowSwingOrderBlocksInput,    SwingBullishOrderBlockColor);
            ManageObList(_swingBearObs,    index, false, SwingOrderBlocksSizeInput,    ShowSwingOrderBlocksInput,    SwingBearishOrderBlockColor);
        }

        // ── FIX 2: i < index — matches Pine's array.slice(from, to) ──────────
        //
        // ORIGINAL BUG: both inner loops used i <= index (inclusive of the
        // structure break bar).  Pine's array.slice(p_ivot.barIndex, bar_index)
        // is EXCLUSIVE of bar_index, so the break bar is never a candidate for
        // parsedIndex.  Including it caused C# to occasionally select a bar
        // one position earlier than Pine for the OB anchor.
        //
        // FIX: changed to i < index (exclusive), making the search window
        // [pivotIndex ... index-1] — identical to Pine's slice range.
        private void StoreOrderBlockFromPivot(int pivotIndex, bool isInternal, int bias, int index)
        {
            if (pivotIndex < 0 || pivotIndex >= index || index >= _parsedHighs.Count)
                return;

            var parsedIndex = pivotIndex;
            if (bias == -1)
            {
                var maxV = double.MinValue;
                for (var i = pivotIndex; i < index; i++)   // FIX: i < index (was i <= index)
                {
                    var v = _parsedHighs[i];
                    if (v > maxV) { maxV = v; parsedIndex = i; }
                }
            }
            else
            {
                var minV = double.MaxValue;
                for (var i = pivotIndex; i < index; i++)   // FIX: i < index (was i <= index)
                {
                    var v = _parsedLows[i];
                    if (v < minV) { minV = v; parsedIndex = i; }
                }
            }

            var bullish = bias == 1;
            var list    = isInternal
                ? (bullish ? _internalBullObs : _internalBearObs)
                : (bullish ? _swingBullObs    : _swingBearObs);

            var id = $"ob_{(isInternal ? "i" : "s")}_{(bullish ? "b" : "r")}_{_obIdCounter++}";
            var ob = new OrderBlock
            {
                Id        = id,
                Index     = parsedIndex,
                Top       = _parsedHighs[parsedIndex],
                Bottom    = _parsedLows[parsedIndex],
                Bullish   = bullish,
                Internal  = isInternal,
                Mitigated = false,
                Box       = null,
                Time      = _times[parsedIndex]
            };

            if (list.Count >= MaxStoredOrderBlocksPerSide)
            {
                var tail = list[list.Count - 1];
                DeleteObVisual(tail);
                list.RemoveAt(list.Count - 1);
            }
            list.Insert(0, ob);
        }

        private void ManageObList(List<OrderBlock> list, int index, bool bullish, int keep, bool show, Color color)
        {
            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];

                var bearishSource = OrderBlockMitigationInput == MitigationMode.Close
                    ? Bars.ClosePrices[index] : Bars.HighPrices[index];
                var bullishSource = OrderBlockMitigationInput == MitigationMode.Close
                    ? Bars.ClosePrices[index] : Bars.LowPrices[index];

                var crossedOrderBlock = (!bullish && bearishSource > ob.Top)
                                     || ( bullish && bullishSource < ob.Bottom);
                if (crossedOrderBlock)
                {
                    DeleteObVisual(ob);
                    list.RemoveAt(i);
                    continue;
                }

                if (!show) { DeleteObVisual(ob); continue; }
                if (i >= keep) { DeleteObVisual(ob); continue; }

                var right = Math.Min(index + 1, Bars.Count - 1);
                if (ob.Box == null)
                {
                    var rect = Chart.DrawRectangle(ob.Id, ob.Time, ob.Top, Bars.OpenTimes[right], ob.Bottom, color, 1, LineStyle.Solid);
                    rect.IsFilled  = true;
                    rect.Color     = color;
                    rect.LineStyle = LineStyle.Solid;
                    ob.Box         = rect;
                }
                else
                {
                    ob.Box.Time1     = ob.Time;
                    ob.Box.Time2     = Bars.OpenTimes[right];
                    ob.Box.Y1        = ob.Top;
                    ob.Box.Y2        = ob.Bottom;
                    ob.Box.Color     = color;
                    ob.Box.LineStyle = LineStyle.Solid;
                    ob.Box.IsFilled  = true;
                }
            }
        }

        private void DeleteObVisual(OrderBlock ob)
        {
            if (ob.Box != null) { Chart.RemoveObject(ob.Id); ob.Box = null; }
        }

        // ── Fair Value Gaps ───────────────────────────────────────────────────

        private void UpdateFvgs(int index)
        {
            if (index < 2) return;
            var autoTh = FairValueGapsThresholdInput
                ? Math.Max(Symbol.TickSize * 2, AverageRange(index, 20) * 0.15) : 0.0;
            var bull = Bars.LowPrices[index]  > Bars.HighPrices[index - 2]
                    && (Bars.LowPrices[index] - Bars.HighPrices[index - 2]) > autoTh;
            var bear = Bars.HighPrices[index] < Bars.LowPrices[index - 2]
                    && (Bars.LowPrices[index - 2] - Bars.HighPrices[index]) > autoTh;
            if (bull)
            {
                SetSignal(BullishFvgSignal, index, Bars.LowPrices[index]);
                DrawFvg(index - 2, index + FairValueGapsExtendInput, Bars.LowPrices[index], Bars.HighPrices[index - 2], FairValueGapsBullColorInput, true);
            }
            if (bear)
            {
                SetSignal(BearishFvgSignal, index, Bars.HighPrices[index]);
                DrawFvg(index - 2, index + FairValueGapsExtendInput, Bars.LowPrices[index - 2], Bars.HighPrices[index], FairValueGapsBearColorInput, false);
            }
        }

        private void DrawFvg(int left, int right, double top, double bottom, Color color, bool bullish)
        {
            var name = $"fvg_{(bullish ? "b" : "r")}_{left}_{right}";
            var rect = Chart.DrawRectangle(name, left, top, Math.Min(right, Bars.Count - 1), bottom, color);
            rect.IsFilled = true; rect.Color = color;
        }

        // ── MTF Levels ────────────────────────────────────────────────────────

        private void UpdateMtfLevels(int index)
        {
            var t = Bars.OpenTimes[index];
            if (_lastDay == DateTime.MinValue || t.Date != _lastDay.Date)
            {
                DrawPeriodLevels(index, ref _dailyHighLine, ref _dailyLowLine, _dayHigh, _dayLow, DailyLevelsColorInput, DailyLevelsStyleInput, "D");
                _dayHigh = Bars.HighPrices[index]; _dayLow = Bars.LowPrices[index]; _lastDay = t.Date;
            }
            else { _dayHigh = Math.Max(_dayHigh, Bars.HighPrices[index]); _dayLow = Math.Min(_dayLow, Bars.LowPrices[index]); }

            var week = FirstDateOfWeek(t);
            if (_lastWeek == DateTime.MinValue || week != _lastWeek)
            {
                DrawPeriodLevels(index, ref _weeklyHighLine, ref _weeklyLowLine, _weekHigh, _weekLow, WeeklyLevelsColorInput, WeeklyLevelsStyleInput, "W");
                _weekHigh = Bars.HighPrices[index]; _weekLow = Bars.LowPrices[index]; _lastWeek = week;
            }
            else { _weekHigh = Math.Max(_weekHigh, Bars.HighPrices[index]); _weekLow = Math.Min(_weekLow, Bars.LowPrices[index]); }

            var month = new DateTime(t.Year, t.Month, 1);
            if (_lastMonth == DateTime.MinValue || month != _lastMonth)
            {
                DrawPeriodLevels(index, ref _monthlyHighLine, ref _monthlyLowLine, _monthHigh, _monthLow, MonthlyLevelsColorInput, MonthlyLevelsStyleInput, "M");
                _monthHigh = Bars.HighPrices[index]; _monthLow = Bars.LowPrices[index]; _lastMonth = month;
            }
            else { _monthHigh = Math.Max(_monthHigh, Bars.HighPrices[index]); _monthLow = Math.Min(_monthLow, Bars.LowPrices[index]); }
        }

        private void DrawPeriodLevels(int index, ref ChartTrendLine top, ref ChartTrendLine bottom,
                                       double high, double low, Color color, LineStyleOpt style, string tag)
        {
            if (double.IsInfinity(high) || double.IsInfinity(low) ||
                high == double.MinValue  || low == double.MaxValue) return;
            var show = (tag == "D" && ShowDailyLevelsInput)
                    || (tag == "W" && ShowWeeklyLevelsInput)
                    || (tag == "M" && ShowMonthlyLevelsInput);
            if (!show) return;
            top    = Chart.DrawTrendLine($"lvl_{tag}_h_{index}", index - 1, high, index + 10, high, color, 1, MapLineStyle(style));
            bottom = Chart.DrawTrendLine($"lvl_{tag}_l_{index}", index - 1, low,  index + 10, low,  color, 1, MapLineStyle(style));
            top.ExtendToInfinity = true; bottom.ExtendToInfinity = true;
            Chart.DrawText($"lvl_{tag}_ht_{index}", $"P{tag}H", index, high, color);
            Chart.DrawText($"lvl_{tag}_lt_{index}", $"P{tag}L", index, low,  color);
        }

        // ── Trailing extremes / High-Low swings / Premium-Discount ────────────

        private void UpdateTrailingExtremes(int index)
        {
            var h = Bars.HighPrices[index]; var l = Bars.LowPrices[index];
            if (h >= _trailingTop)    { _trailingTop    = h; _trailingLastTopTime    = Bars.OpenTimes[index]; }
            if (l <= _trailingBottom) { _trailingBottom = l; _trailingLastBottomTime = Bars.OpenTimes[index]; }
            _trailingBarTime  = Bars.OpenTimes[index];
            _trailingBarIndex = index;
        }

        private void DrawHighLowSwings(int index)
        {
            if (_trailingTop == double.MinValue || _trailingBottom == double.MaxValue) return;
            var dt        = index > 0 ? Bars.OpenTimes[index] - Bars.OpenTimes[index - 1] : TimeSpan.FromMinutes(1);
            var rightTime = Bars.OpenTimes[index].AddTicks(dt.Ticks * 20);
            Chart.DrawTrendLine("smc_wh_line", _trailingLastTopTime,    _trailingTop,    rightTime, _trailingTop,    SwingBearishColorInput, 1, LineStyle.Solid);
            Chart.DrawText("smc_wh_label", _swingTrend < 0 ? "Strong High" : "Weak High", rightTime, _trailingTop,    SwingBearishColorInput);
            Chart.DrawTrendLine("smc_wl_line", _trailingLastBottomTime, _trailingBottom, rightTime, _trailingBottom, SwingBullishColorInput, 1, LineStyle.Solid);
            Chart.DrawText("smc_wl_label", _swingTrend > 0 ? "Strong Low"  : "Weak Low",  rightTime, _trailingBottom, SwingBullishColorInput);
        }

        private void UpdatePremiumDiscountZones(int index)
        {
            if (_trailingTop == double.MinValue || _trailingBottom == double.MaxValue || _trailingTop <= _trailingBottom) return;
            var top = _trailingTop; var bottom = _trailingBottom;
            _premiumBox     = DrawZoneRect(_premiumBox,     $"premium_{index}",  index - 50, top,                          index + 50, 0.95 * top    + 0.05 * bottom, PremiumZoneColorInput,     "Premium");
            _equilibriumBox = DrawZoneRect(_equilibriumBox, $"eq_{index}",       index - 50, 0.525 * top + 0.475 * bottom, index + 50, 0.525 * bottom + 0.475 * top,  EquilibriumZoneColorInput, "Equilibrium");
            _discountBox    = DrawZoneRect(_discountBox,    $"discount_{index}", index - 50, 0.95 * bottom + 0.05 * top,   index + 50, bottom,                        DiscountZoneColorInput,    "Discount");
        }

        private ChartRectangle DrawZoneRect(ChartRectangle existing, string name,
                                             int x1, double y1, int x2, double y2,
                                             Color color, string text)
        {
            var c    = Color.FromArgb(80, color.R, color.G, color.B);
            var rect = Chart.DrawRectangle(name, Math.Max(0, x1), y1, Math.Min(Bars.Count - 1, x2), y2, c);
            rect.IsFilled = true; rect.Color = c;
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

        // ── Utilities ─────────────────────────────────────────────────────────

        private bool ShouldShow(StructureFilter filter, bool choch)
        {
            if (filter == StructureFilter.All)              return true;
            if (filter == StructureFilter.BOS   && !choch) return true;
            if (filter == StructureFilter.CHOCH &&  choch) return true;
            return false;
        }

        private bool IsPivotHigh(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.HighPrices[i];
            for (var j = i - len; j <= i + len; j++) if (j != i && Bars.HighPrices[j] >= p) return false;
            return true;
        }

        private bool IsPivotLow(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.LowPrices[i];
            for (var j = i - len; j <= i + len; j++) if (j != i && Bars.LowPrices[j] <= p) return false;
            return true;
        }

        private double AverageRange(int index, int len)
        {
            var start = Math.Max(1, index - len + 1);
            var sum = 0.0; var n = 0;
            for (var i = start; i <= index; i++) { sum += Bars.HighPrices[i] - Bars.LowPrices[i]; n++; }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        private static DateTime FirstDateOfWeek(DateTime dt)
        {
            var diff = (7 + (dt.DayOfWeek - DayOfWeek.Monday)) % 7;
            return dt.Date.AddDays(-diff);
        }

        private LineStyle MapLineStyle(LineStyleOpt style)
        {
            switch (style)
            {
                case LineStyleOpt.Dashed: return LineStyle.Lines;
                case LineStyleOpt.Dotted: return LineStyle.DotsRare;
                default:                  return LineStyle.Solid;
            }
        }

        private void RemoveOld(string prefix, int keep) { /* stub */ }

        private void SetSignal(IndicatorDataSeries series, int index, double value)
        { if (PlotSignalSeriesInput) series[index] = value; }

        private void ResetSignals(int index)
        {
            _evtInternalBull = _evtInternalBear = _evtSwingBull = _evtSwingBear = false;
            InternalBullishBOS[index]   = InternalBearishBOS[index]   = double.NaN;
            InternalBullishCHoCH[index] = InternalBearishCHoCH[index] = double.NaN;
            SwingBullishBOS[index]      = SwingBearishBOS[index]      = double.NaN;
            SwingBullishCHoCH[index]    = SwingBearishCHoCH[index]    = double.NaN;
            EqualHighSignal[index]      = EqualLowSignal[index]       = double.NaN;
            BullishFvgSignal[index]     = BearishFvgSignal[index]     = double.NaN;
        }

        // ── Inner type ────────────────────────────────────────────────────────

        private sealed class OrderBlock
        {
            public string         Id;
            public int            Index;
            public double         Top;
            public double         Bottom;
            public bool           Bullish;
            public bool           Internal;
            public bool           Mitigated;
            public ChartRectangle Box;
            public DateTime       Time;
        }
    }
}
