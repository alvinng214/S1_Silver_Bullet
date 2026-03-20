using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ════════════════════════════════════════════════════════════════════════════
    //  Swing_OB_Detector
    //
    //  Derived from SMC_OrderBlock_Detector.  All Internal Order Block logic and
    //  all Fair Value Gap logic have been completely removed.  Only Swing Order
    //  Block detection and its signal engine remain.
    //
    //  SIGNAL FLOW
    //  ───────────
    //  1. ComputeLeg() tracks the Swing pivot leg using SwingsLengthInput bars.
    //  2. ProcessSwingStructureCrosses() detects BOS / CHoCH on the swing leg.
    //     • Bullish break → StoreOrderBlockFromPivot(bias=+1) → bearish pivot bar
    //       before the break is stored as a Bullish Swing OB.
    //     • Bearish break → StoreOrderBlockFromPivot(bias=−1) → bullish pivot bar
    //       before the break is stored as a Bearish Swing OB.
    //  3. ManageObListWithSignals() runs every bar on _swingBullObs and _swingBearObs:
    //     • Touch detected  → ob.SignalFired = true; _swingObSignal updated (pending).
    //     • Mitigation      → OB removed from list; drawn dimmed if ShowMitigatedObs.
    //  4. EvaluateSignal() checks the pending _swingObSignal each bar:
    //     • Bull: close > ob.Top  AND bullish candle AND Entry=false  → condSwing = +1
    //     • Bear: close < ob.Bottom AND bearish candle AND Entry=false → condSwing = −1
    //  5. DrawSignals() writes to output series:
    //     • LongSignal / ShortSignal   – signal dot positions (requires ShowSignalDots)
    //     • LongSwingObBottom          – bullish OB bottom (SL reference for long trades)
    //     • ShortSwingObTop            – bearish OB top   (SL reference for short trades)
    //
    //  OUTPUT SERIES READABLE BY A cBOT
    //  ──────────────────────────────────
    //  LongSwingObBottom[bar]  – non-NaN when a bullish swing OB confirmed on that bar.
    //                            Value = OB bottom price (stop-loss reference for longs).
    //  ShortSwingObTop[bar]    – non-NaN when a bearish swing OB confirmed on that bar.
    //                            Value = OB top price (stop-loss reference for shorts).
    //  LongSignal[bar]         – non-NaN when condSwing=+1 AND ShowSignalDots=true.
    //  ShortSignal[bar]        – non-NaN when condSwing=−1 AND ShowSignalDots=true.
    // ════════════════════════════════════════════════════════════════════════════

    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Swing_OB_Detector : Indicator
    {
        // ════════════════════════════════════════════════════════════════════════
        //  Enums
        // ════════════════════════════════════════════════════════════════════════
        public enum DisplayMode     { Historical, Present }
        public enum ThemeStyle      { Colored, Monochrome }
        public enum StructureFilter { All, BOS, CHOCH }
        public enum LabelSizeOpt    { Tiny, Small, Normal }
        public enum ObFilter        { Atr, CumulativeMeanRange }
        public enum MitigationMode  { Close, HighLow }
        public enum LineStyleOpt    { Solid, Dashed, Dotted }

        /// <summary>
        /// Price source used to detect BOS / CHoCH crosses.
        ///   Close   – bar close must cross the swing level.
        ///   HighLow – more aggressive: bar High (bullish) or bar Low (bearish) must cross.
        /// </summary>
        public enum StructureSource { Close, HighLow }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – General
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }

        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }

        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }

        [Parameter("Plot Signal Series", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool PlotSignalSeriesInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Real Time Swing Structure
        // ════════════════════════════════════════════════════════════════════════
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

        /// <summary>
        /// Price source used to detect BOS / CHoCH.
        /// Close = bar close must cross the level.
        /// HighLow = bar High (bullish) or bar Low (bearish) must cross.
        /// </summary>
        [Parameter("BOS/CHoCH Source", DefaultValue = StructureSource.Close, Group = "Real Time Swing Structure")]
        public StructureSource SwingStructureSourceInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Swing Order Blocks
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Swing Order Blocks", DefaultValue = true, Group = "Swing Order Blocks")]
        public bool ShowSwingOrderBlocksInput { get; set; }

        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Swing Order Blocks")]
        public int SwingOrderBlocksSizeInput { get; set; }

        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "Swing Order Blocks")]
        public ObFilter OrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "Swing Order Blocks")]
        public int ObFilterAtrPeriod { get; set; }

        [Parameter("OB Filter CMR Period", DefaultValue = 0, MinValue = 0, MaxValue = 500, Group = "Swing Order Blocks")]
        public int ObFilterCmrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "Swing Order Blocks")]
        public MitigationMode OrderBlockMitigationInput { get; set; }

        [Parameter("Bullish OB Color", DefaultValue = "#CC1848CC", Group = "Swing Order Blocks")]
        public Color SwingBullishOrderBlockColor { get; set; }

        [Parameter("Bearish OB Color", DefaultValue = "#CCB22833", Group = "Swing Order Blocks")]
        public Color SwingBearishOrderBlockColor { get; set; }

        [Parameter("Show All Historical OBs", DefaultValue = true, Group = "Swing Order Blocks")]
        public bool ShowAllHistoricalObs { get; set; }

        [Parameter("Show Mitigated OBs", DefaultValue = true, Group = "Swing Order Blocks")]
        public bool ShowMitigatedObs { get; set; }

        [Parameter("Mitigated OB Opacity (%)", DefaultValue = 30, MinValue = 1, MaxValue = 99, Group = "Swing Order Blocks")]
        public int MitigatedObOpacity { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – EQH / EQL
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }

        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }

        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Highs & Lows MTF
        // ════════════════════════════════════════════════════════════════════════
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

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Premium & Discount Zones
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }

        [Parameter("Premium Zone", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }

        [Parameter("Equilibrium Zone", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }

        [Parameter("Discount Zone", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Signal Display
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Line Width Liquidated", DefaultValue = 1, MinValue = 1, MaxValue = 4, Group = "Signal Display")]
        public int LineWidthLiquidated { get; set; }

        [Parameter("Show Signal Dots", DefaultValue = false, Group = "Signal Display")]
        public bool ShowSignalDots { get; set; }

        [Parameter("Show Signals", DefaultValue = true, Group = "Signal Display")]
        public bool ShowSignalsOb { get; set; }

        [Parameter("Signal Offset (pips)", DefaultValue = 2.0, MinValue = 0.0, Step = 0.1, Group = "Signal Display")]
        public double SignalOffsetPips { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters – Signals
        // ════════════════════════════════════════════════════════════════════════
        [Parameter("Min Dist OB (bars)", DefaultValue = 1, MinValue = 1, Group = "Signals")]
        public int MinDist { get; set; }

        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Signals")]
        public bool UseHeikinAshi { get; set; }

        /// <summary>
        /// Minimum number of bars that must close after the swing BOS/CHoCH bar
        /// before the OB created by that break is allowed to fire a signal.
        /// 0 = no restriction (original behaviour).
        /// </summary>
        [Parameter("Min Bars After Structure Break", DefaultValue = 0, MinValue = 0, MaxValue = 200, Group = "Signals")]
        public int MinBarsAfterStructureBreak { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Output Series
        // ════════════════════════════════════════════════════════════════════════

        // ── Swing structure series (for PlotSignalSeriesInput) ────────────────
        [Output("Swing Bullish BOS",   LineColor = "Lime")] public IndicatorDataSeries SwingBullishBOS   { get; set; }
        [Output("Swing Bearish BOS",   LineColor = "Red")]  public IndicatorDataSeries SwingBearishBOS   { get; set; }
        [Output("Swing Bullish CHoCH", LineColor = "Lime")] public IndicatorDataSeries SwingBullishCHoCH { get; set; }
        [Output("Swing Bearish CHoCH", LineColor = "Red")]  public IndicatorDataSeries SwingBearishCHoCH { get; set; }

        // ── EQH / EQL ─────────────────────────────────────────────────────────
        [Output("Equal High", LineColor = "DodgerBlue")] public IndicatorDataSeries EqualHighSignal { get; set; }
        [Output("Equal Low",  LineColor = "DodgerBlue")] public IndicatorDataSeries EqualLowSignal  { get; set; }

        // ── Signal dots ───────────────────────────────────────────────────────
        [Output("Long Signal",  LineColor = "Lime", PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries LongSignal  { get; set; }

        [Output("Short Signal", LineColor = "Red",  PlotType = PlotType.Points, Thickness = 6)]
        public IndicatorDataSeries ShortSignal { get; set; }

        // ── OB SL reference levels (readable by a cBot) ───────────────────────
        // LongSwingObBottom[N]  – non-NaN when condSwing=+1 fired on bar N.
        //                         Value = the Bullish OB's Bottom price → use as long SL.
        [Output("Long Swing OB Bottom", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries LongSwingObBottom { get; set; }

        // ShortSwingObTop[N]    – non-NaN when condSwing=−1 fired on bar N.
        //                         Value = the Bearish OB's Top price → use as short SL.
        [Output("Short Swing OB Top", LineColor = "Transparent", PlotType = PlotType.Points, Thickness = 1)]
        public IndicatorDataSeries ShortSwingObTop { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Private inner types
        // ════════════════════════════════════════════════════════════════════════
        private sealed class SignalState
        {
            public double Point;
            public double ObSlLevel;
            public bool   IsBull;
            public bool   Entry;
            public int    Index;
        }

        private sealed class OrderBlock
        {
            public string Id;
            public int    Index;
            public double Top;
            public double Bottom;
            public bool   Bullish;
            public bool   Mitigated;
            public bool   SignalFired;

            /// <summary>
            /// Bar index of the BOS/CHoCH that created this OB.
            /// Used for the same-bar-block and MinBarsAfterStructureBreak filters.
            /// </summary>
            public int            StructureBreakIndex;

            public ChartRectangle Box;
            public DateTime       Time;
            public DateTime       MitigatedAt;
            public string         MitigatedBoxId;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields
        // ════════════════════════════════════════════════════════════════════════

        // ── Swing structure state ─────────────────────────────────────────────
        private int  _swingTrend, _swingLeg;
        private bool _evtSwingBull, _evtSwingBear;

        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;

        // ── EQH / EQL state ──────────────────────────────────────────────────
        private double _prevEqHigh      = double.NaN;
        private int    _prevEqHighIndex = -1;
        private double _prevEqLow       = double.NaN;
        private int    _prevEqLowIndex  = -1;

        // ── Swing OB pools ────────────────────────────────────────────────────
        private readonly List<OrderBlock> _swingBullObs  = new List<OrderBlock>();
        private readonly List<OrderBlock> _swingBearObs  = new List<OrderBlock>();
        private readonly List<OrderBlock> _mitigatedObs  = new List<OrderBlock>();

        // ── Parsed arrays (volatility-adjusted OB anchoring) ──────────────────
        private readonly List<double>   _parsedHighs = new List<double>();
        private readonly List<double>   _parsedLows  = new List<double>();
        private readonly List<DateTime> _times       = new List<DateTime>();
        private double _cumTr;

        // ── Strong / Weak High / Low state ────────────────────────────────────
        private double   _trailingTop            = double.MinValue;
        private double   _trailingBottom         = double.MaxValue;
        private DateTime _trailingLastTopTime    = DateTime.MinValue;
        private DateTime _trailingLastBottomTime = DateTime.MinValue;

        // ── Premium / Discount zone rectangles ───────────────────────────────
        private ChartRectangle _premiumBox, _equilibriumBox, _discountBox;

        // ── MTF level lines ───────────────────────────────────────────────────
        private ChartTrendLine _dailyHighLine,   _dailyLowLine;
        private ChartTrendLine _weeklyHighLine,  _weeklyLowLine;
        private ChartTrendLine _monthlyHighLine, _monthlyLowLine;
        private DateTime _lastDay   = DateTime.MinValue;
        private DateTime _lastWeek  = DateTime.MinValue;
        private DateTime _lastMonth = DateTime.MinValue;
        private double _dayHigh, _dayLow, _weekHigh, _weekLow, _monthHigh, _monthLow;

        // ── OB ID counter ─────────────────────────────────────────────────────
        private int _obIdCounter;
        private const int MaxStoredOrderBlocksPerSide = 500;

        // ── Signal engine ─────────────────────────────────────────────────────
        // _swingObSignal holds the pending signal state from the most recent
        // swing OB touch. It persists across bars until:
        //   (a) EvaluateSignal() fires condSwing (Entry set to true), or
        //   (b) A new OB touch overwrites it with a fresh SignalState.
        private SignalState _swingObSignal;

        // ── Heikin-Ashi arrays ────────────────────────────────────────────────
        private readonly List<double> _haOpen  = new List<double>();
        private readonly List<double> _haClose = new List<double>();

        // ── Chart drawing counter ─────────────────────────────────────────────
        private int _shapeId;

        // ════════════════════════════════════════════════════════════════════════
        //  Initialize
        // ════════════════════════════════════════════════════════════════════════
        protected override void Initialize()
        {
            _dayHigh    = _weekHigh    = _monthHigh   = double.MinValue;
            _dayLow     = _weekLow     = _monthLow    = double.MaxValue;
            _trailingTop    = double.MinValue;
            _trailingBottom = double.MaxValue;
            _swingObSignal  = NewEmptySignal();
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Calculate – main loop
        // ════════════════════════════════════════════════════════════════════════
        public override void Calculate(int index)
        {
            // Reset all output series for this bar to NaN.
            LongSignal[index]        = double.NaN;
            ShortSignal[index]       = double.NaN;
            LongSwingObBottom[index] = double.NaN;
            ShortSwingObTop[index]   = double.NaN;
            ResetSignalSeries(index);

            EnsureHeikinAshi(index);
            UpdateParsedArrays(index);

            // Warmup guard: the swing leg needs SwingsLengthInput bars before
            // it can identify valid pivots.
            if (index < SwingsLengthInput + 5)
                return;

            // ── Trailing extremes (Strong / Weak High / Low) ──────────────────
            if (ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateTrailingExtremes(index);

            if (ShowHighLowSwingsInput)
                DrawHighLowSwings(index);

            if (ShowPremiumDiscountZonesInput)
                UpdatePremiumDiscountZones(index);

            // ── Swing structure + EQH/EQL ─────────────────────────────────────
            // Gate: at least one visual/feature flag must be on so the structure
            // engine runs and swing OBs are created.  ShowSwingOrderBlocksInput
            // itself is also sufficient to keep the engine alive.
            if (ShowStructureInput || ShowSwingsInput || ShowEqualHighsLowsInput ||
                ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput || ShowSwingOrderBlocksInput)
                UpdateSwingStructure(index);

            // ── Swing OB lifecycle + signal touch ─────────────────────────────
            if (ShowSwingOrderBlocksInput)
                UpdateSwingObsWithSignals(index);

            // ── MTF Levels ────────────────────────────────────────────────────
            if (ShowDailyLevelsInput || ShowWeeklyLevelsInput || ShowMonthlyLevelsInput)
                UpdateMtfLevels(index);

            // ── Candle colouring ──────────────────────────────────────────────
            if (ShowTrendInput)
                ColorTrendBar(index);

            // ── Fire entry condition ──────────────────────────────────────────
            // candleDir: +1 = bullish bar (close > open), -1 = bearish bar.
            var candleDir   = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var signalClose = (UseHeikinAshi && index < _haClose.Count)
                                  ? _haClose[index]
                                  : Bars.ClosePrices[index];

            var condSwing = 0;
            EvaluateSignal(_swingObSignal, signalClose, candleDir, ref condSwing);

            DrawSignals(index, condSwing);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Swing Structure
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateSwingStructure(int index)
        {
            var sLen = Math.Max(5, SwingsLengthInput);

            // Track the swing leg direction.
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

            ProcessSwingStructureCrosses(index);

            if (ShowEqualHighsLowsInput)
                DetectEqualHighLow(index);
        }

        // Determines whether the current bar's leg direction has changed.
        // Returns +1 (leg turned up), -1 (leg turned down), or previousLeg.
        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;

            var highest = double.MinValue;
            var lowest  = double.MaxValue;
            var start   = Math.Max(0, index - size + 1);
            for (var i = start; i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lowest)  lowest  = Bars.LowPrices[i];
            }

            if (Bars.HighPrices[index - size] > highest) return 0;
            if (Bars.LowPrices[index  - size] < lowest)  return 1;
            return previousLeg;
        }

        private bool CrossedUp(int index, double level)
        {
            return SwingStructureSourceInput == StructureSource.HighLow
                ? Bars.HighPrices[index] > level
                : Bars.ClosePrices[index] > level;
        }

        private bool CrossedDown(int index, double level)
        {
            return SwingStructureSourceInput == StructureSource.HighLow
                ? Bars.LowPrices[index] < level
                : Bars.ClosePrices[index] < level;
        }

        private void ProcessSwingStructureCrosses(int index)
        {
            // ── Bullish swing BOS / CHoCH ─────────────────────────────────────
            if (!double.IsNaN(_lastSwingHigh) && !_swingHighCrossed && CrossedUp(index, _lastSwingHigh))
            {
                _swingHighCrossed = true;
                var choch = _swingTrend < 0;
                if (choch) SetSignalSeries(SwingBullishCHoCH, index, Bars.ClosePrices[index]);
                else       SetSignalSeries(SwingBullishBOS,   index, Bars.ClosePrices[index]);
                _swingTrend   = 1;
                _evtSwingBull = true;
                if (ShowStructureInput && ShouldShow(ShowSwingBullInput, choch))
                    DrawStructureLine(index, _lastSwingHigh, choch ? "CHoCH" : "BOS", SwingBullishColorInput, SwingStructureSize);

                // Store the bearish OB (last bearish pivot before the bullish break).
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingHighIndex, bias: 1, breakIndex: index);
            }

            // ── Bearish swing BOS / CHoCH ─────────────────────────────────────
            if (!double.IsNaN(_lastSwingLow) && !_swingLowCrossed && CrossedDown(index, _lastSwingLow))
            {
                _swingLowCrossed = true;
                var choch = _swingTrend > 0;
                if (choch) SetSignalSeries(SwingBearishCHoCH, index, Bars.ClosePrices[index]);
                else       SetSignalSeries(SwingBearishBOS,   index, Bars.ClosePrices[index]);
                _swingTrend   = -1;
                _evtSwingBear = true;
                if (ShowStructureInput && ShouldShow(ShowSwingBearInput, choch))
                    DrawStructureLine(index, _lastSwingLow, choch ? "CHoCH" : "BOS", SwingBearishColorInput, SwingStructureSize);

                // Store the bullish OB (last bullish pivot before the bearish break).
                if (ShowSwingOrderBlocksInput)
                    StoreOrderBlockFromPivot(_lastSwingLowIndex, bias: -1, breakIndex: index);
            }
        }

        private void DrawStructureLine(int index, double level, string label, Color color, LabelSizeOpt size)
        {
            var x1 = Math.Max(index - 10, 0);
            Chart.DrawTrendLine($"smc_{label}_{index}", x1, level, index, level, color, 1, LineStyle.Solid);
            if (ModeInput == DisplayMode.Present) RemoveOld("smc_", 100);
            Chart.DrawText($"smct_{index}", label, index, level, color);
        }

        // ── EQH / EQL detection ───────────────────────────────────────────────
        // Uses EqualHighsLowsLengthInput as the pivot look-back on each side.
        private void DetectEqualHighLow(int index)
        {
            var len = Math.Max(1, EqualHighsLowsLengthInput);
            var p   = index - len;
            if (p <= len || p >= Bars.Count) return;

            var range = Math.Max(Symbol.TickSize, Bars.HighPrices[index] - Bars.LowPrices[index]);
            var thr   = Math.Max(Symbol.TickSize, EqualHighsLowsThresholdInput * range);

            if (IsPivotHigh(p, len))
            {
                var v = Bars.HighPrices[p];
                if (!double.IsNaN(_prevEqHigh) && Math.Abs(v - _prevEqHigh) <= thr)
                {
                    SetSignalSeries(EqualHighSignal, index, v);
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
                    SetSignalSeries(EqualLowSignal, index, v);
                    Chart.DrawTrendLine("smc_eql_line", _prevEqLowIndex, _prevEqLow, p, v, Color.DodgerBlue, 1, LineStyle.DotsRare);
                    Chart.DrawText("smc_eql_label", "EQL", p, v, Color.DodgerBlue);
                }
                _prevEqLow      = v;
                _prevEqLowIndex = p;
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Swing OB Lifecycle + Signal Touch
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateSwingObsWithSignals(int index)
        {
            if (index < 3) return;
            ManageObList(_swingBullObs, index, bullish: true,  SwingBullishOrderBlockColor);
            ManageObList(_swingBearObs, index, bullish: false, SwingBearishOrderBlockColor);
            ManageMitigatedObs();
        }

        private void ManageMitigatedObs()
        {
            if (!ShowMitigatedObs)
            {
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
        /// Runs the OB lifecycle for one list per bar:
        ///
        ///   1. Touch check  → if price enters the OB zone and filters pass,
        ///                      _swingObSignal is updated (pending confirmation).
        ///                      ob.SignalFired is set true permanently (first-touch only).
        ///   2. Mitigation   → if price closes fully through the OB, remove it
        ///                      from the list and optionally draw it dimmed.
        ///   3. Draw / resize OB box on chart.
        ///
        /// Filter 1 – Same-bar block:
        ///   A signal on the same bar as the structure break that created the OB
        ///   is suppressed (avoids a re-entry on the break bar itself).
        ///
        /// Filter 2 – MinDist:
        ///   ob.Index + MinDist must be strictly less than index (minimum bar
        ///   distance between the OB pivot bar and the touch bar).
        ///
        /// Filter 3 – MinBarsAfterStructureBreak cooldown:
        ///   index must be > ob.StructureBreakIndex + MinBarsAfterStructureBreak.
        /// </summary>
        private void ManageObList(List<OrderBlock> list, int index, bool bullish, Color color)
        {
            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];

                // ── Touch detection ───────────────────────────────────────────
                if (!ob.SignalFired && ob.Index < index)
                {
                    if (bullish && Bars.LowPrices[index] <= ob.Top)
                    {
                        ob.SignalFired = true;  // permanently consumed (first-touch rule)

                        // Filter 1: block if this is the same bullish structure-break bar
                        var isSameBreakBar  = (index == ob.StructureBreakIndex)
                                           && Bars.ClosePrices[index] > Bars.OpenPrices[index];
                        // Filter 3: cooldown
                        var cooldownOk      = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;

                        if (ob.Index + MinDist < index && !isSameBreakBar && cooldownOk)
                        {
                            DrawLiquidationLine($"ob_touch_{ob.Id}", ob.Time, Bars.OpenTimes[index], ob.Top, color);
                            // Point = ob.Top  (confirmation: close must exceed OB top)
                            // ObSlLevel = ob.Bottom  (stop-loss reference for cBot)
                            _swingObSignal = NewSignal(index, ob.Top, isBull: true, obSlLevel: ob.Bottom);
                        }
                    }
                    else if (!bullish && Bars.HighPrices[index] >= ob.Bottom)
                    {
                        ob.SignalFired = true;

                        // Filter 1: block if this is the same bearish structure-break bar
                        var isSameBreakBar  = (index == ob.StructureBreakIndex)
                                           && Bars.ClosePrices[index] < Bars.OpenPrices[index];
                        // Filter 3: cooldown
                        var cooldownOk      = index > ob.StructureBreakIndex + MinBarsAfterStructureBreak;

                        if (ob.Index + MinDist < index && !isSameBreakBar && cooldownOk)
                        {
                            DrawLiquidationLine($"ob_touch_{ob.Id}", ob.Time, Bars.OpenTimes[index], ob.Bottom, color);
                            // Point = ob.Bottom  (confirmation: close must fall below OB bottom)
                            // ObSlLevel = ob.Top  (stop-loss reference for cBot)
                            _swingObSignal = NewSignal(index, ob.Bottom, isBull: false, obSlLevel: ob.Top);
                        }
                    }
                }

                // ── Mitigation ────────────────────────────────────────────────
                // With MitigationMode.HighLow: mitigated when High > ob.Top (bear OB)
                //                              or Low < ob.Bottom (bull OB).
                // With MitigationMode.Close:   mitigated when Close > ob.Top (bear OB)
                //                              or Close < ob.Bottom (bull OB).
                var bearishSrc = OrderBlockMitigationInput == MitigationMode.Close
                    ? Bars.ClosePrices[index] : Bars.HighPrices[index];
                var bullishSrc = OrderBlockMitigationInput == MitigationMode.Close
                    ? Bars.ClosePrices[index] : Bars.LowPrices[index];
                var mitigated  = (!bullish && bearishSrc > ob.Top)
                              || ( bullish && bullishSrc < ob.Bottom);

                if (mitigated)
                {
                    ob.Mitigated   = true;
                    ob.MitigatedAt = Bars.OpenTimes[index];

                    if (ShowMitigatedObs)
                    {
                        var alpha    = (int)Math.Round(255.0 * MitigatedObOpacity / 100.0);
                        var dimColor = Color.FromArgb(alpha, color.R, color.G, color.B);
                        var mitId    = $"mit_{ob.Id}";

                        if (ob.Box != null)
                        {
                            ob.Box.Time2      = ob.MitigatedAt;
                            ob.Box.Color      = dimColor;
                            ob.Box.IsFilled   = true;
                            ob.MitigatedBoxId = ob.Id;
                            ob.Box            = null;
                        }
                        else
                        {
                            var rect = Chart.DrawRectangle(mitId, ob.Time, ob.Top, ob.MitigatedAt, ob.Bottom, dimColor);
                            rect.IsFilled     = true;
                            rect.Color        = dimColor;
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

                // ── Prune if outside the size window ──────────────────────────
                if (!ShowAllHistoricalObs && i >= SwingOrderBlocksSizeInput)
                {
                    DeleteObVisual(ob);
                    continue;
                }

                // ── Draw / resize box ─────────────────────────────────────────
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
                    ob.Box.Time1    = ob.Time;
                    ob.Box.Time2    = Bars.OpenTimes[right];
                    ob.Box.Y1       = ob.Top;
                    ob.Box.Y2       = ob.Bottom;
                    ob.Box.Color    = color;
                    ob.Box.IsFilled = true;
                }
            }
        }

        /// <summary>
        /// Creates and stores a Swing OB from the pivot bar that preceded the
        /// swing structure break.
        ///
        ///   bias = +1  → bullish swing break → store the bearish pivot (bull OB)
        ///               located at the minimum parsed-low bar from pivotIndex to breakIndex.
        ///   bias = −1  → bearish swing break → store the bullish pivot (bear OB)
        ///               located at the maximum parsed-high bar from pivotIndex to breakIndex.
        ///
        /// The parsed arrays (_parsedHighs / _parsedLows) apply a volatility
        /// adjustment: on high-volatility bars the High/Low are swapped so the
        /// OB anchors to the body of the candle rather than the wick.
        /// </summary>
        private void StoreOrderBlockFromPivot(int pivotIndex, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex || breakIndex >= _parsedHighs.Count)
                return;

            var parsedIndex = pivotIndex;
            if (bias == -1)
            {
                // Bearish break: OB is the highest parsed-high bar in the range
                var maxV = double.MinValue;
                for (var i = pivotIndex; i <= breakIndex; i++)
                {
                    var v = _parsedHighs[i];
                    if (v > maxV) { maxV = v; parsedIndex = i; }
                }
            }
            else
            {
                // Bullish break: OB is the lowest parsed-low bar in the range
                var minV = double.MaxValue;
                for (var i = pivotIndex; i <= breakIndex; i++)
                {
                    var v = _parsedLows[i];
                    if (v < minV) { minV = v; parsedIndex = i; }
                }
            }

            var bullish = bias == 1;
            var list    = bullish ? _swingBullObs : _swingBearObs;
            var id      = $"ob_s_{(bullish ? "b" : "r")}_{_obIdCounter++}";

            var ob = new OrderBlock
            {
                Id                  = id,
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                Mitigated           = false,
                SignalFired         = false,
                Box                 = null,
                Time                = _times[parsedIndex],
                StructureBreakIndex = breakIndex
            };

            // Trim oldest OB if we have hit the hard cap
            if (list.Count >= MaxStoredOrderBlocksPerSide)
            {
                DeleteObVisual(list[list.Count - 1]);
                list.RemoveAt(list.Count - 1);
            }
            list.Insert(0, ob);
        }

        private void DeleteObVisual(OrderBlock ob)
        {
            if (ob.Box != null) { Chart.RemoveObject(ob.Id); ob.Box = null; }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Signal Engine – EvaluateSignal + DrawSignals
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Checks the pending _swingObSignal against the current bar's close
        /// and candle direction.
        ///
        /// Bull confirmation (condSwing = +1):
        ///   signalClose > _swingObSignal.Point  (close above OB top)
        ///   AND bar is bullish  (candleDir == +1)
        ///   AND signal not yet consumed (Entry == false)
        ///
        /// Bear confirmation (condSwing = −1):
        ///   signalClose < _swingObSignal.Point  (close below OB bottom)
        ///   AND bar is bearish  (candleDir == −1)
        ///   AND signal not yet consumed (Entry == false)
        ///
        /// Once Entry is set true, this signal cannot fire again.
        /// A new OB touch will overwrite _swingObSignal with Entry=false.
        /// </summary>
        private void EvaluateSignal(SignalState signal, double signalClose, int candleDir, ref int condition)
        {
            if (double.IsNaN(signal.Point))
                return;

            if (signalClose > signal.Point && signal.IsBull && candleDir == 1 && !signal.Entry)
            {
                signal.Entry = true;
                condition    = 1;
            }
            else if (signalClose < signal.Point && !signal.IsBull && candleDir == -1 && !signal.Entry)
            {
                signal.Entry = true;
                condition    = -1;
            }
        }

        private void DrawSignals(int index, int condSwing)
        {
            var offset = SignalOffsetPips * Symbol.PipSize;

            // ── Chart arrow icons ─────────────────────────────────────────────
            if (ShowSignalsOb && condSwing == 1)
                Chart.DrawIcon($"ob_buy_{index}_{_shapeId++}",  ChartIconType.UpArrow,   Bars.OpenTimes[index], Bars.LowPrices[index]  - offset, SwingBullishColorInput);
            if (ShowSignalsOb && condSwing == -1)
                Chart.DrawIcon($"ob_sell_{index}_{_shapeId++}", ChartIconType.DownArrow, Bars.OpenTimes[index], Bars.HighPrices[index] + offset, SwingBearishColorInput);

            // ── Signal dot series ─────────────────────────────────────────────
            if (ShowSignalDots)
            {
                if (condSwing ==  1) LongSignal[index]  = Bars.LowPrices[index]  - offset;
                if (condSwing == -1) ShortSignal[index] = Bars.HighPrices[index] + offset;
            }

            // ── OB SL-reference output series ─────────────────────────────────
            // These are the values a cBot reads to determine stop-loss placement.
            if (condSwing ==  1) LongSwingObBottom[index]  = _swingObSignal.ObSlLevel;
            if (condSwing == -1) ShortSwingObTop[index]    = _swingObSignal.ObSlLevel;
        }

        private void DrawLiquidationLine(string id, DateTime from, DateTime to, double price, Color color)
        {
            var line = Chart.DrawTrendLine(id, from, price, to, price, color, LineWidthLiquidated, LineStyle.LinesDots);
            line.ExtendToInfinity = false;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Heikin-Ashi
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
        //  Parsed arrays — volatility-adjusted OB anchoring
        // ════════════════════════════════════════════════════════════════════════
        // On a high-volatility bar (range >= 2 × ATR) the High and Low are swapped
        // in the parsed arrays so the OB anchors to the body, not the wick.
        private void UpdateParsedArrays(int index)
        {
            if (index == 0) { _cumTr = 0; }
            else
            {
                var prevClose = Bars.ClosePrices[index - 1];
                _cumTr += Math.Max(
                    Bars.HighPrices[index] - Bars.LowPrices[index],
                    Math.Max(Math.Abs(Bars.HighPrices[index] - prevClose),
                             Math.Abs(Bars.LowPrices[index]  - prevClose)));
            }

            var atrMeasure = AverageTrueRangeSimple(index, ObFilterAtrPeriod);

            double volatilityMeasure;
            if (OrderBlockFilterInput == ObFilter.Atr)
            {
                volatilityMeasure = atrMeasure;
            }
            else
            {
                var cmrBars = ObFilterCmrPeriod == 0
                    ? Math.Max(1, index)
                    : Math.Min(ObFilterCmrPeriod, Math.Max(1, index));
                volatilityMeasure = _cumTr / cmrBars;
            }

            var highVol = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * volatilityMeasure;
            _parsedHighs.Add(highVol ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( highVol ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Trailing extremes / Strong-Weak swings / Premium-Discount zones
        // ════════════════════════════════════════════════════════════════════════
        private void UpdateTrailingExtremes(int index)
        {
            var h = Bars.HighPrices[index]; var l = Bars.LowPrices[index];
            if (h >= _trailingTop)    { _trailingTop    = h; _trailingLastTopTime    = Bars.OpenTimes[index]; }
            if (l <= _trailingBottom) { _trailingBottom = l; _trailingLastBottomTime = Bars.OpenTimes[index]; }
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
            var top = _trailingTop; var bot = _trailingBottom;
            _premiumBox     = DrawZoneRect(_premiumBox,     $"premium_{index}", index - 50, top,                        index + 50, 0.95 * top + 0.05 * bot,       PremiumZoneColorInput,     "Premium");
            _equilibriumBox = DrawZoneRect(_equilibriumBox, $"eq_{index}",      index - 50, 0.525 * top + 0.475 * bot, index + 50, 0.525 * bot + 0.475 * top,     EquilibriumZoneColorInput, "Equilibrium");
            _discountBox    = DrawZoneRect(_discountBox,    $"discount_{index}", index - 50, 0.95 * bot + 0.05 * top,  index + 50, bot,                            DiscountZoneColorInput,    "Discount");
        }

        private ChartRectangle DrawZoneRect(ChartRectangle existing, string name, int x1, double y1, int x2, double y2, Color color, string text)
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

        // ════════════════════════════════════════════════════════════════════════
        //  MTF Levels
        // ════════════════════════════════════════════════════════════════════════
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
            if (double.IsInfinity(high) || double.IsInfinity(low) || high == double.MinValue || low == double.MaxValue) return;
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
        //  Utilities
        // ════════════════════════════════════════════════════════════════════════
        private double AverageTrueRangeSimple(int index, int period)
        {
            var start = Math.Max(1, index - period + 1);
            var sum   = 0.0; var n = 0;
            for (var i = start; i <= index; i++)
            {
                var prevClose = Bars.ClosePrices[i - 1];
                sum += Math.Max(Bars.HighPrices[i] - Bars.LowPrices[i],
                       Math.Max(Math.Abs(Bars.HighPrices[i] - prevClose),
                                Math.Abs(Bars.LowPrices[i]  - prevClose)));
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }

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

        private void RemoveOld(string prefix, int keep) { /* stub for Present mode */ }

        private void SetSignalSeries(IndicatorDataSeries series, int index, double value)
        { if (PlotSignalSeriesInput) series[index] = value; }

        private void ResetSignalSeries(int index)
        {
            _evtSwingBull = _evtSwingBear = false;
            SwingBullishBOS[index]    = SwingBearishBOS[index]    = double.NaN;
            SwingBullishCHoCH[index]  = SwingBearishCHoCH[index]  = double.NaN;
            EqualHighSignal[index]    = EqualLowSignal[index]     = double.NaN;
        }

        private SignalState NewSignal(int index, double point, bool isBull, double obSlLevel = double.NaN)
            => new SignalState { Point = point, ObSlLevel = double.IsNaN(obSlLevel) ? point : obSlLevel, IsBull = isBull, Entry = false, Index = index };

        private static SignalState NewEmptySignal()
            => new SignalState { Point = double.NaN, Entry = false };
    }
}
