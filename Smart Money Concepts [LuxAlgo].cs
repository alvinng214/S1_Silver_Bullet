using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    /// <summary>
    /// SMC_OrderBlock_Detector
    ///
    /// Combines Smart Money Concepts [LuxAlgo] visual framework with the
    /// touch-and-bounce entry signal engine from Order-Block Detector.
    ///
    /// Order-block detection and mitigation come exclusively from the SMC engine
    /// (internal + swing OBs built on BOS/CHoCH pivot events).  The OB-Detector
    /// signal logic is then applied to those SMC order blocks:
    ///   • Bull OB touched (low ≤ ob.Top)  → wait for close above ob.Top  → Long Signal
    ///   • Bear OB touched (high ≥ ob.Bottom) → wait for close below ob.Bottom → Short Signal
    ///   • FVG touch-and-bounce signals use the same engine on SMC-detected FVGs.
    ///
    /// All UI parameters from both source indicators are preserved.
    /// </summary>
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_OrderBlock_Detector : Indicator
    {
        // ────────────────────────────────────────────────────────────────────────
        //  Enums  (from SMC)
        // ────────────────────────────────────────────────────────────────────────
        public enum DisplayMode { Historical, Present }
        public enum ThemeStyle { Colored, Monochrome }
        public enum StructureFilter { All, BOS, CHOCH }
        public enum LabelSizeOpt { Tiny, Small, Normal }
        public enum ObFilter { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum LineStyleOpt { Solid, Dashed, Dotted }

        // ────────────────────────────────────────────────────────────────────────
        //  Parameters – Smart Money Concepts
        // ────────────────────────────────────────────────────────────────────────
        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }

        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }

        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }

        [Parameter("Plot Signal Series", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool PlotSignalSeriesInput { get; set; }

        // Internal Structure
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

        // Swing Structure
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

        // Order Blocks
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

        [Parameter("Show All Historical OBs", DefaultValue = false, Group = "Order Blocks")]
        public bool ShowAllHistoricalObs { get; set; }

        [Parameter("Show Mitigated OBs", DefaultValue = false, Group = "Order Blocks")]
        public bool ShowMitigatedObs { get; set; }

        [Parameter("Mitigated OB Opacity (%)", DefaultValue = 30, MinValue = 1, MaxValue = 99, Group = "Order Blocks")]
        public int MitigatedObOpacity { get; set; }

        // EQH / EQL
        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }

        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }

        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        // Fair Value Gaps
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

        // Highs & Lows MTF
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

        // Premium & Discount Zones
        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }

        [Parameter("Premium Zone", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }

        [Parameter("Equilibrium Zone", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }

        [Parameter("Discount Zone", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

        // ────────────────────────────────────────────────────────────────────────
        //  Parameters – Signal Engine  (from Order-Block Detector)
        // ────────────────────────────────────────────────────────────────────────
        [Parameter("Line Width Liquidated", DefaultValue = 1, MinValue = 1, MaxValue = 4, Group = "Signal Display")]
        public int LineWidthLiquidated { get; set; }

        [Parameter("Show Signal Dots", DefaultValue = true, Group = "Signal Display")]
        public bool ShowSignalDots { get; set; }

        [Parameter("Show Signals OB", DefaultValue = true, Group = "Signal Display")]
        public bool ShowSignalsOb { get; set; }

        [Parameter("Show Signals FVG", DefaultValue = true, Group = "Signal Display")]
        public bool ShowSignalsFvg { get; set; }

        [Parameter("Signal Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1, Group = "Signal Display")]
        public double SignalOffsetPips { get; set; }

        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDist { get; set; }

        [Parameter("Min Dist FVG (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Signals")]
        public bool UseHeikinAshi { get; set; }

        // ────────────────────────────────────────────────────────────────────────
        //  Outputs – SMC structure series
        // ────────────────────────────────────────────────────────────────────────
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

        // ────────────────────────────────────────────────────────────────────────
        //  Outputs – Entry signal dots  (from Order-Block Detector)
        // ────────────────────────────────────────────────────────────────────────
        [Output("Long Signal", LineColor = "Lime", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries LongSignal { get; set; }

        [Output("Short Signal", LineColor = "Red", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries ShortSignal { get; set; }

        // OB price levels exposed for cBot stop-loss placement.
        // LongObBottom  = ob.Bottom of the bullish OB that triggered the long signal.
        // ShortObTop    = ob.Top    of the bearish OB that triggered the short signal.
        // Both are NaN on all other bars.
        [Output("Long OB Bottom", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries LongObBottom { get; set; }

        [Output("Short OB Top", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries ShortObTop { get; set; }

        // ────────────────────────────────────────────────────────────────────────
        //  Private inner types
        // ────────────────────────────────────────────────────────────────────────

        /// <summary>Pending entry state set the moment an OB or FVG is touched.</summary>
        private sealed class SignalState
        {
            public double Point;      // price level to watch for the confirmed bounce
            public double ObSlLevel;  // OB Bottom (bull) or OB Top (bear) for SL reference
            public bool IsBull;
            public bool Entry;        // true once signal has been fired
            public int Index;         // chart index when the state was created
        }

        /// <summary>Lightweight record for FVG zones that feeds the signal engine.</summary>
        private sealed class FvgRecord
        {
            public double Max;
            public double Min;
            public bool IsBull;
            public int DetectionIndex;
            public bool SignalFired;
        }

        /// <summary>
        /// SMC order-block zone.  SignalFired prevents the signal engine from
        /// triggering more than once per zone.
        /// </summary>
        private sealed class OrderBlock
        {
            public string Id;
            public int Index;          // chart bar index at detection
            public double Top;
            public double Bottom;
            public bool Bullish;
            public bool Internal;
            public bool Mitigated;
            public bool SignalFired;   // set true when the OB-Detector touch logic fires
            public ChartRectangle Box;
            public DateTime Time;
            public DateTime MitigatedAt;   // time the OB was broken (used to freeze the box right edge)
            public string MitigatedBoxId;  // separate chart object id for the frozen historical box
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Private fields – SMC engine
        // ────────────────────────────────────────────────────────────────────────
        private int _internalTrend, _swingTrend, _internalLeg, _swingLeg;
        private bool _evtInternalBull, _evtInternalBear, _evtSwingBull, _evtSwingBear;

        private readonly List<OrderBlock> _internalBullObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _internalBearObs = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBullObs   = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBearObs   = new List<OrderBlock>();
        // Mitigated (broken) OBs kept for historical display
        private readonly List<OrderBlock> _mitigatedObs   = new List<OrderBlock>();

        private double _lastSwingHigh = double.NaN;
        private double _lastSwingLow  = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;

        private double _internalHighLevel = double.NaN;
        private double _internalLowLevel  = double.NaN;
        private bool   _internalHighCrossed, _internalLowCrossed;
        private bool   _swingHighCrossed,    _swingLowCrossed;
        private int    _internalHighIndex = -1;
        private int    _internalLowIndex  = -1;

        private double _prevEqHigh = double.NaN;
        private int    _prevEqHighIndex = -1;
        private double _prevEqLow  = double.NaN;
        private int    _prevEqLowIndex  = -1;

        private readonly List<double>   _parsedHighs = new List<double>();
        private readonly List<double>   _parsedLows  = new List<double>();
        private readonly List<DateTime> _times       = new List<DateTime>();
        private double _cumTr;

        private ChartTrendLine _dailyHighLine, _dailyLowLine;
        private ChartTrendLine _weeklyHighLine, _weeklyLowLine;
        private ChartTrendLine _monthlyHighLine, _monthlyLowLine;
        private DateTime _lastDay   = DateTime.MinValue;
        private DateTime _lastWeek  = DateTime.MinValue;
        private DateTime _lastMonth = DateTime.MinValue;
        private double _dayHigh,   _dayLow;
        private double _weekHigh,  _weekLow;
        private double _monthHigh, _monthLow;

        private ChartRectangle _premiumBox, _equilibriumBox, _discountBox;

        private int _obIdCounter;
        private const int MaxStoredOrderBlocksPerSide = 500;

        private double   _trailingTop           = double.MinValue;
        private double   _trailingBottom        = double.MaxValue;
        private DateTime _trailingLastTopTime   = DateTime.MinValue;
        private DateTime _trailingLastBottomTime = DateTime.MinValue;

        // ────────────────────────────────────────────────────────────────────────
        //  Private fields – signal engine  (from Order-Block Detector)
        // ────────────────────────────────────────────────────────────────────────
        private SignalState _signal;
        private SignalState _signalFvg;

        private readonly List<FvgRecord> _fvgRecords = new List<FvgRecord>();
        private int _lastDetectedFvgIndex = -1;

        // Heikin-Ashi arrays built on the chart timeframe
        private readonly List<double> _haOpen  = new List<double>();
        private readonly List<double> _haClose = new List<double>();

        private int _shapeId;

        // ────────────────────────────────────────────────────────────────────────
        //  Initialize
        // ────────────────────────────────────────────────────────────────────────
        protected override void Initialize()
        {
            _dayHigh  = _weekHigh  = _monthHigh = double.MinValue;
            _dayLow   = _weekLow   = _monthLow  = double.MaxValue;
            _trailingTop    = double.MinValue;
            _trailingBottom = double.MaxValue;
            _signal    = NewEmptySignal();
            _signalFvg = NewEmptySignal();
        }

        // ────────────────────────────────────────────────────────────────────────
        //  Calculate  – main loop
        // ────────────────────────────────────────────────────────────────────────
        public override void Calculate(int index)
        {
            LongSignal[index]  = double.NaN;
            ShortSignal[index] = double.NaN;
            LongObBottom[index] = double.NaN;
            ShortObTop[index]   = double.NaN;
            ResetSignals(index);

            // Always keep HA arrays and parsed arrays in sync (needed by signal engine)
            EnsureHeikinAshi(index);
            UpdateParsedArrays(index);

            if (index < Math.Max(SwingsLengthInput, EqualHighsLowsLengthInput) + 5)
                return;

            // ── SMC visual layer ─────────────────────────────────────────────
            if (ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateTrailingExtremes(index);

            if (ShowHighLowSwingsInput)
                DrawHighLowSwings(index);

            if (ShowPremiumDiscountZonesInput)
                UpdatePremiumDiscountZones(index);

            if (ShowInternalsInput  || ShowStructureInput  || ShowSwingsInput ||
                ShowEqualHighsLowsInput || ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateStructure(index);

            // OB management with integrated OB-Detector signal touch detection
            if (ShowInternalOrderBlocksInput || ShowSwingOrderBlocksInput)
                UpdateOrderBlocksWithSignals(index);

            // FVG display + populate _fvgRecords for the signal engine
            if (ShowFairValueGapsInput)
                UpdateFvgs(index);

            // FVG touch-and-bounce signals
            HandleFvgSignals(index);

            if (ShowDailyLevelsInput || ShowWeeklyLevelsInput || ShowMonthlyLevelsInput)
                UpdateMtfLevels(index);

            if (ShowTrendInput)
                ColorTrendBar(index);

            // ── Signal engine – fire entry conditions ────────────────────────
            var candleDir    = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var signalClose  = (UseHeikinAshi && index < _haClose.Count)
                                   ? _haClose[index]
                                   : Bars.ClosePrices[index];

            var cond    = 0;
            var condFvg = 0;

            // OB signal
            if (!double.IsNaN(_signal.Point))
            {
                if (signalClose > _signal.Point && _signal.IsBull && candleDir == 1 && !_signal.Entry)
                {
                    _signal.Entry = true;
                    cond = 1;
                }
                else if (signalClose < _signal.Point && !_signal.IsBull && candleDir == -1 && !_signal.Entry)
                {
                    _signal.Entry = true;
                    cond = -1;
                }
            }

            // FVG signal
            if (!double.IsNaN(_signalFvg.Point))
            {
                if (signalClose > _signalFvg.Point && _signalFvg.IsBull && candleDir == 1 && !_signalFvg.Entry)
                {
                    _signalFvg.Entry = true;
                    condFvg = 1;
                }
                else if (signalClose < _signalFvg.Point && !_signalFvg.IsBull && candleDir == -1 && !_signalFvg.Entry)
                {
                    _signalFvg.Entry = true;
                    condFvg = -1;
                }
            }

            DrawSignals(index, cond, condFvg);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – Structure
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateStructure(int index)
        {
            var iLen = 5;
            var sLen = Math.Max(5, SwingsLengthInput);

            // ── Internal leg ─────────────────────────────────────────────────
            var internalLegNow    = ComputeLeg(index, iLen, _internalLeg);
            var internalLegChange = internalLegNow - _internalLeg;
            if (internalLegChange != 0)
            {
                if (internalLegChange == 1)
                {
                    _internalLowLevel = Bars.LowPrices[index - iLen];
                    _internalLowIndex = index - iLen;
                    _internalLowCrossed = false;
                }
                else
                {
                    _internalHighLevel = Bars.HighPrices[index - iLen];
                    _internalHighIndex = index - iLen;
                    _internalHighCrossed = false;
                }
            }
            _internalLeg = internalLegNow;

            // ── Swing leg ────────────────────────────────────────────────────
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
            if (index - size < 1)
                return previousLeg;

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
                EmitStructure(index, true, false, _internalHighLevel, true);
                if (ShowInternalOrderBlocksInput)
                    StoreOrderBlockFromPivot(_internalHighIndex, true, 1, index);
            }

            if (!double.IsNaN(_internalLowLevel) && !_internalLowCrossed && close < _internalLowLevel)
            {
                _internalLowCrossed = true;
                EmitStructure(index, false, false, _internalLowLevel, true);
                if (ShowInternalOrderBlocksInput)
                    StoreOrderBlockFromPivot(_internalLowIndex, true, -1, index);
            }

            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed && close > _lastSwingHigh)
            {
                _swingHighCrossed = true;
                EmitStructure(index, true, true, _lastSwingHigh, false);
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingHighIndex, false, 1, index);
            }

            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed && close < _lastSwingLow)
            {
                _swingLowCrossed = true;
                EmitStructure(index, false, true, _lastSwingLow, false);
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingLowIndex, false, -1, index);
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
                        _internalTrend  = 1;
                        _evtInternalBull = true;
                        if (ShowInternalsInput && ShouldShow(ShowInternalBullInput, choch))
                            DrawStructureLine(index, level, choch ? "iCHoCH" : "iBOS", InternalBullColorInput, InternalStructureSize);
                    }
                    else
                    {
                        if (choch) SetSignal(SwingBullishCHoCH, index, close);
                        else       SetSignal(SwingBullishBOS,   index, close);
                        _swingTrend  = 1;
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
            var x1   = Math.Max(index - 10, 0);
            Chart.DrawTrendLine($"smc_{label}_{index}", x1, level, index, level, color, 1, LineStyle.Solid);
            if (ModeInput == DisplayMode.Present)
                RemoveOld("smc_", 100);
            Chart.DrawText($"smct_{index}", label, index, level, color);
        }

        private void DetectEqualHighLow(int index, int len)
        {
            var p = index - len;
            if (p <= len || p >= Bars.Count)
                return;

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
                _prevEqHigh      = v;
                _prevEqHighIndex = p;
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
                _prevEqLow      = v;
                _prevEqLowIndex = p;
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC + Signal Engine – Order Blocks
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Replaces SMC's UpdateOrderBlocks.
        /// Passes each list through ManageObListWithSignals which handles both
        /// the SMC visual lifecycle and OB-Detector touch → signal detection.
        /// </summary>
        private void UpdateOrderBlocksWithSignals(int index)
        {
            if (index < 3)
                return;

            ManageObListWithSignals(_internalBullObs, index, true,  InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBullishOrderBlockColor);
            ManageObListWithSignals(_internalBearObs, index, false, InternalOrderBlocksSizeInput, ShowInternalOrderBlocksInput, InternalBearishOrderBlockColor);
            ManageObListWithSignals(_swingBullObs,    index, true,  SwingOrderBlocksSizeInput,    ShowSwingOrderBlocksInput,    SwingBullishOrderBlockColor);
            ManageObListWithSignals(_swingBearObs,    index, false, SwingOrderBlocksSizeInput,    ShowSwingOrderBlocksInput,    SwingBearishOrderBlockColor);
            ManageMitigatedObs();
        }

        /// <summary>
        /// Removes mitigated OB visuals when the ShowMitigatedObs toggle is turned off,
        /// and prunes the list.  When the toggle is on, boxes are already frozen and
        /// need no further updates.
        /// </summary>
        private void ManageMitigatedObs()
        {
            if (!ShowMitigatedObs)
            {
                // Remove any frozen boxes that are still on the chart
                for (var i = _mitigatedObs.Count - 1; i >= 0; i--)
                {
                    var ob = _mitigatedObs[i];
                    if (ob.MitigatedBoxId != null)
                    {
                        Chart.RemoveObject(ob.MitigatedBoxId);
                        ob.MitigatedBoxId = null;
                    }
                }
                _mitigatedObs.Clear();
            }
        }

        /// <summary>
        /// Combined SMC order-block lifecycle manager + OB-Detector signal touch logic.
        ///
        /// Touch detection (OB Detector):
        ///   Bull OB – when candle low ≤ ob.Top  → signal pending at ob.Top  (long)
        ///   Bear OB – when candle high ≥ ob.Bottom → signal pending at ob.Bottom (short)
        ///
        /// Full mitigation (SMC, unchanged):
        ///   Bull OB – bullishSource  &lt; ob.Bottom  → remove OB
        ///   Bear OB – bearishSource  &gt; ob.Top    → remove OB
        /// </summary>
        private void ManageObListWithSignals(
            List<OrderBlock> list,
            int              index,
            bool             bullish,
            int              keep,
            bool             show,
            Color            color)
        {
            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];

                // ── OB-Detector touch logic ──────────────────────────────────
                // Only trigger once, and only after at least MinDist bars have
                // elapsed since the OB was registered.
                if (!ob.SignalFired && ob.Index < index)
                {
                    if (bullish && Bars.LowPrices[index] <= ob.Top)
                    {
                        ob.SignalFired = true;
                        if (ob.Index + MinDist < index)
                        {
                            DrawLiquidationLine(
                                $"ob_touch_{ob.Id}",
                                ob.Time,
                                Bars.OpenTimes[index],
                                ob.Top,
                                InternalBullishOrderBlockColor);
                            _signal = NewSignal(index, ob.Top, true, ob.Bottom);
                        }
                    }
                    else if (!bullish && Bars.HighPrices[index] >= ob.Bottom)
                    {
                        ob.SignalFired = true;
                        if (ob.Index + MinDist < index)
                        {
                            DrawLiquidationLine(
                                $"ob_touch_{ob.Id}",
                                ob.Time,
                                Bars.OpenTimes[index],
                                ob.Bottom,
                                InternalBearishOrderBlockColor);
                            _signal = NewSignal(index, ob.Bottom, false, ob.Top);
                        }
                    }
                }

                // ── SMC full mitigation (unmodified from SMC source) ─────────
                var bearishSource   = OrderBlockMitigationInput == MitigationMode.Close
                                          ? Bars.ClosePrices[index] : Bars.HighPrices[index];
                var bullishSource   = OrderBlockMitigationInput == MitigationMode.Close
                                          ? Bars.ClosePrices[index] : Bars.LowPrices[index];
                var crossedOrderBlock = (!bullish && bearishSource > ob.Top)
                                     || ( bullish && bullishSource < ob.Bottom);
                if (crossedOrderBlock)
                {
                    // Mark when it was mitigated and freeze the right edge of its box
                    ob.Mitigated    = true;
                    ob.MitigatedAt  = Bars.OpenTimes[index];

                    if (ShowMitigatedObs)
                    {
                        // Freeze the live box at the mitigation bar and re-colour it dimmed
                        var alpha     = (int)Math.Round(255.0 * MitigatedObOpacity / 100.0);
                        var dimColor  = Color.FromArgb(alpha, color.R, color.G, color.B);
                        var mitId     = $"mit_{ob.Id}";

                        if (ob.Box != null)
                        {
                            // Re-use the existing rectangle: stop extending, apply dim colour
                            ob.Box.Time2    = ob.MitigatedAt;
                            ob.Box.Color    = dimColor;
                            ob.Box.IsFilled = true;
                            ob.MitigatedBoxId = ob.Id;
                            ob.Box = null; // detach so the active loop no longer updates it
                        }
                        else
                        {
                            // Box was never drawn (outside keep range) – draw it now frozen
                            var rect = Chart.DrawRectangle(mitId, ob.Time, ob.Top, ob.MitigatedAt, ob.Bottom, dimColor);
                            rect.IsFilled   = true;
                            rect.Color      = dimColor;
                            ob.MitigatedBoxId = mitId;
                        }

                        _mitigatedObs.Add(ob);
                    }
                    else
                    {
                        DeleteObVisual(ob);
                    }

                    list.RemoveAt(i);
                    continue;
                }

                if (!show)
                {
                    DeleteObVisual(ob);
                    continue;
                }

                if (!ShowAllHistoricalObs && i >= keep)
                {
                    DeleteObVisual(ob);
                    continue;
                }

                // ── Draw / update box ────────────────────────────────────────
                var right = Math.Min(index + 1, Bars.Count - 1);
                if (ob.Box == null)
                {
                    var rect = Chart.DrawRectangle(ob.Id, ob.Time, ob.Top, Bars.OpenTimes[right], ob.Bottom, color, 1, LineStyle.Solid);
                    rect.IsFilled  = true;
                    rect.Color     = color;
                    rect.LineStyle = LineStyle.Solid;
                    ob.Box = rect;
                }
                else
                {
                    ob.Box.Time1     = ob.Time;
                    ob.Box.Time2     = Bars.OpenTimes[right];
                    ob.Box.Y1        = ob.Top;
                    ob.Box.Y2        = ob.Bottom;
                    ob.Box.Color     = color;
                    ob.Box.IsFilled  = true;
                }
            }
        }

        private void StoreOrderBlockFromPivot(int pivotIndex, bool isInternal, int bias, int index)
        {
            if (pivotIndex < 0 || pivotIndex >= index || index >= _parsedHighs.Count)
                return;

            var parsedIndex = pivotIndex;
            if (bias == -1)
            {
                var maxV = double.MinValue;
                for (var i = pivotIndex; i <= index; i++)
                {
                    var v = _parsedHighs[i];
                    if (v > maxV) { maxV = v; parsedIndex = i; }
                }
            }
            else
            {
                var minV = double.MaxValue;
                for (var i = pivotIndex; i <= index; i++)
                {
                    var v = _parsedLows[i];
                    if (v < minV) { minV = v; parsedIndex = i; }
                }
            }

            var bullish = bias == 1;
            var list = isInternal
                ? (bullish ? _internalBullObs : _internalBearObs)
                : (bullish ? _swingBullObs    : _swingBearObs);

            var id = $"ob_{(isInternal ? "i" : "s")}_{(bullish ? "b" : "r")}_{_obIdCounter++}";
            var ob = new OrderBlock
            {
                Id          = id,
                Index       = parsedIndex,
                Top         = _parsedHighs[parsedIndex],
                Bottom      = _parsedLows[parsedIndex],
                Bullish     = bullish,
                Internal    = isInternal,
                Mitigated   = false,
                SignalFired = false,
                Box         = null,
                Time        = _times[parsedIndex]
            };

            if (list.Count >= MaxStoredOrderBlocksPerSide)
            {
                var tail = list[list.Count - 1];
                DeleteObVisual(tail);
                list.RemoveAt(list.Count - 1);
            }
            list.Insert(0, ob);
        }

        private void DeleteObVisual(OrderBlock ob)
        {
            if (ob.Box != null)
            {
                Chart.RemoveObject(ob.Id);
                ob.Box = null;
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – Fair Value Gaps  (display + signal record population)
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateFvgs(int index)
        {
            if (index < 2)
                return;

            var autoTh = FairValueGapsThresholdInput
                ? Math.Max(Symbol.TickSize * 2, AverageRange(index, 20) * 0.15)
                : 0.0;

            var bull = Bars.LowPrices[index]     > Bars.HighPrices[index - 2]
                    && (Bars.LowPrices[index]     - Bars.HighPrices[index - 2]) > autoTh;
            var bear = Bars.HighPrices[index]    < Bars.LowPrices[index - 2]
                    && (Bars.LowPrices[index - 2] - Bars.HighPrices[index])     > autoTh;

            if (bull)
            {
                SetSignal(BullishFvgSignal, index, Bars.LowPrices[index]);
                DrawFvg(index - 2, index + FairValueGapsExtendInput,
                        Bars.LowPrices[index], Bars.HighPrices[index - 2],
                        FairValueGapsBullColorInput, true);
            }
            if (bear)
            {
                SetSignal(BearishFvgSignal, index, Bars.HighPrices[index]);
                DrawFvg(index - 2, index + FairValueGapsExtendInput,
                        Bars.LowPrices[index - 2], Bars.HighPrices[index],
                        FairValueGapsBearColorInput, false);
            }

            // Populate FVG signal records (one entry per detected index to avoid re-entry)
            if (index != _lastDetectedFvgIndex && (bull || bear))
            {
                _lastDetectedFvgIndex = index;
                if (bull)
                {
                    _fvgRecords.Insert(0, new FvgRecord
                    {
                        Max            = Bars.LowPrices[index],
                        Min            = Bars.HighPrices[index - 2],
                        IsBull         = true,
                        DetectionIndex = index,
                        SignalFired    = false
                    });
                }
                else // bear takes precedence when both trigger (edge case)
                {
                    _fvgRecords.Insert(0, new FvgRecord
                    {
                        Max            = Bars.LowPrices[index - 2],
                        Min            = Bars.HighPrices[index],
                        IsBull         = false,
                        DetectionIndex = index,
                        SignalFired    = false
                    });
                }
            }
        }

        private void DrawFvg(int left, int right, double top, double bottom, Color color, bool bullish)
        {
            var name = $"fvg_{(bullish ? "b" : "r")}_{left}_{right}";
            var rect = Chart.DrawRectangle(name, left, top, Math.Min(right, Bars.Count - 1), bottom, color);
            rect.IsFilled = true;
            rect.Color    = color;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Signal Engine – FVG touch handler  (mirrors HandleMitigationFvg)
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Iterates stored FVG records and fires a pending FVG signal the first
        /// time price touches the zone edge (identical logic to OB Detector's
        /// HandleMitigationFvg).  Records that have been fully crossed are pruned.
        /// </summary>
        private void HandleFvgSignals(int index)
        {
            var cLow  = Bars.LowPrices[index];
            var cHigh = Bars.HighPrices[index];

            for (var i = _fvgRecords.Count - 1; i >= 0; i--)
            {
                var r = _fvgRecords[i];
                if (r.DetectionIndex >= index) continue;

                if (r.IsBull)
                {
                    // Touch: low enters the FVG (low ≤ top of gap)
                    if (!r.SignalFired && cLow <= r.Max)
                    {
                        r.SignalFired = true;
                        if (r.DetectionIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(index, r.Max, true);
                    }
                    // Fully breached below the bottom → remove record
                    if (cLow < r.Min)
                    {
                        _fvgRecords.RemoveAt(i);
                        continue;
                    }
                }
                else
                {
                    // Touch: high enters the FVG (high ≥ bottom of gap)
                    if (!r.SignalFired && cHigh >= r.Min)
                    {
                        r.SignalFired = true;
                        if (r.DetectionIndex + MinDistFvg < index)
                            _signalFvg = NewSignal(index, r.Min, false);
                    }
                    // Fully breached above the top → remove record
                    if (cHigh > r.Max)
                    {
                        _fvgRecords.RemoveAt(i);
                        continue;
                    }
                }
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – MTF Levels
        // ════════════════════════════════════════════════════════════════════════
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
                _dayLow  = Math.Min(_dayLow,  Bars.LowPrices[index]);
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
                _weekLow  = Math.Min(_weekLow,  Bars.LowPrices[index]);
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
                _monthLow  = Math.Min(_monthLow,  Bars.LowPrices[index]);
            }
        }

        private void DrawPeriodLevels(int index, ref ChartTrendLine top, ref ChartTrendLine bottom,
                                       double high, double low, Color color, LineStyleOpt style, string tag)
        {
            if (double.IsInfinity(high) || double.IsInfinity(low) ||
                high == double.MinValue  || low  == double.MaxValue) return;

            var show = (tag == "D" && ShowDailyLevelsInput)
                    || (tag == "W" && ShowWeeklyLevelsInput)
                    || (tag == "M" && ShowMonthlyLevelsInput);
            if (!show) return;

            top    = Chart.DrawTrendLine($"lvl_{tag}_h_{index}", index - 1, high, index + 10, high, color, 1, MapLineStyle(style));
            bottom = Chart.DrawTrendLine($"lvl_{tag}_l_{index}", index - 1, low,  index + 10, low,  color, 1, MapLineStyle(style));
            top.ExtendToInfinity    = true;
            bottom.ExtendToInfinity = true;
            Chart.DrawText($"lvl_{tag}_ht_{index}", $"P{tag}H", index, high, color);
            Chart.DrawText($"lvl_{tag}_lt_{index}", $"P{tag}L", index, low,  color);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – Trailing extremes / Strong-Weak swings / Premium-Discount
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateTrailingExtremes(int index)
        {
            var h = Bars.HighPrices[index];
            var l = Bars.LowPrices[index];
            if (h >= _trailingTop)    { _trailingTop    = h; _trailingLastTopTime    = Bars.OpenTimes[index]; }
            if (l <= _trailingBottom) { _trailingBottom = l; _trailingLastBottomTime = Bars.OpenTimes[index]; }
        }

        private void DrawHighLowSwings(int index)
        {
            if (_trailingTop == double.MinValue || _trailingBottom == double.MaxValue) return;

            var dt        = index > 0 ? Bars.OpenTimes[index] - Bars.OpenTimes[index - 1] : TimeSpan.FromMinutes(1);
            var rightTime = Bars.OpenTimes[index].AddTicks(dt.Ticks * 20);

            Chart.DrawTrendLine("smc_wh_line", _trailingLastTopTime, _trailingTop, rightTime, _trailingTop, SwingBearishColorInput, 1, LineStyle.Solid);
            Chart.DrawText("smc_wh_label", _swingTrend < 0 ? "Strong High" : "Weak High", rightTime, _trailingTop, SwingBearishColorInput);

            Chart.DrawTrendLine("smc_wl_line", _trailingLastBottomTime, _trailingBottom, rightTime, _trailingBottom, SwingBullishColorInput, 1, LineStyle.Solid);
            Chart.DrawText("smc_wl_label", _swingTrend > 0 ? "Strong Low" : "Weak Low", rightTime, _trailingBottom, SwingBullishColorInput);
        }

        private void UpdatePremiumDiscountZones(int index)
        {
            if (_trailingTop == double.MinValue || _trailingBottom == double.MaxValue || _trailingTop <= _trailingBottom)
                return;

            var top    = _trailingTop;
            var bottom = _trailingBottom;
            _premiumBox      = DrawZoneRect(_premiumBox,      $"premium_{index}", index - 50, top,                        index + 50, 0.95 * top + 0.05 * bottom,      PremiumZoneColorInput,      "Premium");
            _equilibriumBox  = DrawZoneRect(_equilibriumBox,  $"eq_{index}",      index - 50, 0.525 * top + 0.475 * bottom, index + 50, 0.525 * bottom + 0.475 * top, EquilibriumZoneColorInput,  "Equilibrium");
            _discountBox     = DrawZoneRect(_discountBox,     $"discount_{index}",index - 50, 0.95 * bottom + 0.05 * top, index + 50, bottom,                          DiscountZoneColorInput,     "Discount");
        }

        private ChartRectangle DrawZoneRect(ChartRectangle existing, string name, int x1, double y1, int x2, double y2, Color color, string text)
        {
            var c    = Color.FromArgb(80, color.R, color.G, color.B);
            var rect = Chart.DrawRectangle(name, Math.Max(0, x1), y1, Math.Min(Bars.Count - 1, x2), y2, c);
            rect.IsFilled = true;
            rect.Color    = c;
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

        // ════════════════════════════════════════════════════════════════════════
        //  Signal Engine helpers  (from Order-Block Detector)
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Draws entry arrow icons and sets the LongSignal / ShortSignal series.
        /// Logic mirrors DrawSignals in Order-Block Detector exactly.
        /// </summary>
        private void DrawSignals(int index, int cond, int condFvg)
        {
            var offset = SignalOffsetPips * Symbol.PipSize;

            if (ShowSignalsOb && cond == 1)
                DrawSignalIcon($"ob_buy_{index}_{_shapeId++}",  ChartIconType.UpArrow,   index, Bars.LowPrices[index],  SwingBullishColorInput, -offset);
            if (ShowSignalsOb && cond == -1)
                DrawSignalIcon($"ob_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], SwingBearishColorInput,  offset);

            if (ShowSignalsFvg && condFvg == 1)
                DrawSignalIcon($"fvg_buy_{index}_{_shapeId++}",  ChartIconType.UpArrow,   index, Bars.LowPrices[index],  FairValueGapsBullColorInput, -offset);
            if (ShowSignalsFvg && condFvg == -1)
                DrawSignalIcon($"fvg_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, index, Bars.HighPrices[index], FairValueGapsBearColorInput,  offset);

            if (ShowSignalDots)
            {
                if (cond == 1 || condFvg == 1)
                    LongSignal[index]  = Bars.LowPrices[index]  - offset;
                if (cond == -1 || condFvg == -1)
                    ShortSignal[index] = Bars.HighPrices[index] + offset;
            }

            // Always emit OB SL reference levels regardless of ShowSignalDots
            // so the cBot can read them even when dots are hidden.
            if (cond == 1)
                LongObBottom[index]  = _signal.ObSlLevel;
            if (cond == -1)
                ShortObTop[index]    = _signal.ObSlLevel;
            if (condFvg == 1)
                LongObBottom[index]  = !double.IsNaN(LongObBottom[index]) ? LongObBottom[index] : _signalFvg.ObSlLevel;
            if (condFvg == -1)
                ShortObTop[index]    = !double.IsNaN(ShortObTop[index])   ? ShortObTop[index]   : _signalFvg.ObSlLevel;
        }

        private void DrawSignalIcon(string id, ChartIconType type, int index, double y, Color color, double delta)
        {
            Chart.DrawIcon(id, type, Bars.OpenTimes[index], y + delta, color);
        }

        private void DrawLiquidationLine(string id, DateTime from, DateTime to, double price, Color color)
        {
            var line = Chart.DrawTrendLine(id, from, price, to, price, color, LineWidthLiquidated, LineStyle.LinesDots);
            line.ExtendToInfinity = false;
        }

        private SignalState NewSignal(int index, double point, bool isBull, double obSlLevel = double.NaN)
        {
            return new SignalState
            {
                Point      = point,
                ObSlLevel  = double.IsNaN(obSlLevel) ? point : obSlLevel,
                IsBull     = isBull,
                Entry      = false,
                Index      = index
            };
        }

        private static SignalState NewEmptySignal()
        {
            return new SignalState { Point = double.NaN, Entry = false };
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Heikin-Ashi builder  (operates on the chart timeframe)
        // ════════════════════════════════════════════════════════════════════════
        private void EnsureHeikinAshi(int index)
        {
            while (_haClose.Count <= index)
            {
                var i     = _haClose.Count;
                var close = (Bars.OpenPrices[i] + Bars.HighPrices[i] + Bars.LowPrices[i] + Bars.ClosePrices[i]) / 4.0;
                var open  = i == 0
                    ? (Bars.OpenPrices[i] + Bars.ClosePrices[i]) / 2.0
                    : (_haOpen[i - 1] + _haClose[i - 1]) / 2.0;
                _haOpen.Add(open);
                _haClose.Add(close);
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – Parsed arrays  (volatility-adjusted OB anchoring)
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateParsedArrays(int index)
        {
            if (index == 0)
            {
                _cumTr = 0;
            }
            else
            {
                var prevClose = Bars.ClosePrices[index - 1];
                var tr = Math.Max(
                    Bars.HighPrices[index] - Bars.LowPrices[index],
                    Math.Max(
                        Math.Abs(Bars.HighPrices[index] - prevClose),
                        Math.Abs(Bars.LowPrices[index]  - prevClose)));
                _cumTr += tr;
            }

            var atrMeasure = AverageTrueRangeSimple(index, 200);
            var volatilityMeasure = OrderBlockFilterInput == ObFilter.Atr
                ? atrMeasure
                : _cumTr / Math.Max(1, index);

            var highVolatilityBar = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * volatilityMeasure;
            _parsedHighs.Add(highVolatilityBar ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( highVolatilityBar ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC – Small utilities
        // ════════════════════════════════════════════════════════════════════════
        private double AverageTrueRangeSimple(int index, int period)
        {
            var start = Math.Max(1, index - period + 1);
            var sum   = 0.0;
            var n     = 0;
            for (var i = start; i <= index; i++)
            {
                var prevClose = Bars.ClosePrices[i - 1];
                var tr = Math.Max(
                    Bars.HighPrices[i] - Bars.LowPrices[i],
                    Math.Max(
                        Math.Abs(Bars.HighPrices[i] - prevClose),
                        Math.Abs(Bars.LowPrices[i]  - prevClose)));
                sum += tr;
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        private double AverageRange(int index, int len)
        {
            var start = Math.Max(1, index - len + 1);
            var sum   = 0.0;
            var n     = 0;
            for (var i = start; i <= index; i++)
            {
                sum += Bars.HighPrices[i] - Bars.LowPrices[i];
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

        private bool ShouldShow(StructureFilter filter, bool choch)
        {
            if (filter == StructureFilter.All)   return true;
            if (filter == StructureFilter.BOS   && !choch) return true;
            if (filter == StructureFilter.CHOCH && choch)  return true;
            return false;
        }

        private bool IsPivotHigh(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.HighPrices[i];
            for (var j = i - len; j <= i + len; j++)
                if (j != i && Bars.HighPrices[j] >= p) return false;
            return true;
        }

        private bool IsPivotLow(int i, int len)
        {
            if (i - len < 0 || i + len >= Bars.Count) return false;
            var p = Bars.LowPrices[i];
            for (var j = i - len; j <= i + len; j++)
                if (j != i && Bars.LowPrices[j] <= p) return false;
            return true;
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

        private void RemoveOld(string prefix, int keep)
        {
            var count = 0;
            foreach (var o in Chart.Objects)
                if (o.Name.StartsWith(prefix)) count++;
            // Exceeding 'keep' objects: no-op stub kept for parity with SMC source.
            // Extend with deletion logic if memory pressure is observed in Present mode.
        }

        private void SetSignal(IndicatorDataSeries series, int index, double value)
        {
            if (PlotSignalSeriesInput)
                series[index] = value;
        }

        private void ResetSignals(int index)
        {
            _evtInternalBull = _evtInternalBear = _evtSwingBull = _evtSwingBear = false;
            InternalBullishBOS[index]  = InternalBearishBOS[index]  = double.NaN;
            InternalBullishCHoCH[index]= InternalBearishCHoCH[index]= double.NaN;
            SwingBullishBOS[index]     = SwingBearishBOS[index]     = double.NaN;
            SwingBullishCHoCH[index]   = SwingBearishCHoCH[index]   = double.NaN;
            EqualHighSignal[index]     = EqualLowSignal[index]      = double.NaN;
            BullishFvgSignal[index]    = BearishFvgSignal[index]    = double.NaN;
        }
    }
}
