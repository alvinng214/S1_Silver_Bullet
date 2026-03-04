// This work is licensed under Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
// https://creativecommons.org/licenses/by-nc-sa/4.0/
// Original Pine Script © LuxAlgo — translated to cTrader C# indicator

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Indicator(IsOverlay = true, TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class SmartMoneyConceptsLuxAlgo : Indicator
    {
        // ─── Enums ───────────────────────────────────────────────────────────────
        public enum DisplayMode    { Historical, Present }
        public enum ThemeStyle     { Colored, Monochrome }
        public enum StructureFilter{ All, BOS, CHoCH }
        public enum LabelSizeOpt   { Tiny, Small, Normal }
        public enum ObFilterOpt    { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum LineStyleOpt   { Solid, Dashed, Dotted }

        // ─── Parameters: Smart Money Concepts ────────────────────────────────────
        [Parameter("Mode", DefaultValue = DisplayMode.Historical, Group = "Smart Money Concepts")]
        public DisplayMode ModeInput { get; set; }

        [Parameter("Style", DefaultValue = ThemeStyle.Colored, Group = "Smart Money Concepts")]
        public ThemeStyle StyleInput { get; set; }

        [Parameter("Color Candles", DefaultValue = false, Group = "Smart Money Concepts")]
        public bool ShowTrendInput { get; set; }

        // ─── Parameters: Real Time Internal Structure ─────────────────────────────
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

        // ─── Parameters: Real Time Swing Structure ────────────────────────────────
        [Parameter("Show Swing Structure", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowStructureInput { get; set; }

        [Parameter("Bullish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBullInput { get; set; }

        [Parameter("Swing Bull Color", DefaultValue = "#089981", Group = "Real Time Swing Structure")]
        public Color SwingBullColorInput { get; set; }

        [Parameter("Bearish Structure", DefaultValue = StructureFilter.All, Group = "Real Time Swing Structure")]
        public StructureFilter ShowSwingBearInput { get; set; }

        [Parameter("Swing Bear Color", DefaultValue = "#F23645", Group = "Real Time Swing Structure")]
        public Color SwingBearColorInput { get; set; }

        [Parameter("Swing Label Size", DefaultValue = LabelSizeOpt.Small, Group = "Real Time Swing Structure")]
        public LabelSizeOpt SwingStructureSize { get; set; }

        [Parameter("Show Swings Points", DefaultValue = false, Group = "Real Time Swing Structure")]
        public bool ShowSwingsInput { get; set; }

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "Real Time Swing Structure")]
        public int SwingsLengthInput { get; set; }

        [Parameter("Show Strong/Weak High/Low", DefaultValue = true, Group = "Real Time Swing Structure")]
        public bool ShowHighLowSwingsInput { get; set; }

        // ─── Parameters: Order Blocks ─────────────────────────────────────────────
        [Parameter("Internal Order Blocks", DefaultValue = true, Group = "Order Blocks")]
        public bool ShowInternalOrderBlocksInput { get; set; }

        [Parameter("Internal OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int InternalOrderBlocksSizeInput { get; set; }

        [Parameter("Swing Order Blocks", DefaultValue = false, Group = "Order Blocks")]
        public bool ShowSwingOrderBlocksInput { get; set; }

        [Parameter("Swing OB Size", DefaultValue = 5, MinValue = 1, MaxValue = 20, Group = "Order Blocks")]
        public int SwingOrderBlocksSizeInput { get; set; }

        [Parameter("Order Block Filter", DefaultValue = ObFilterOpt.Atr, Group = "Order Blocks")]
        public ObFilterOpt OrderBlockFilterInput { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "Order Blocks")]
        public MitigationMode OrderBlockMitigationInput { get; set; }

        [Parameter("Internal Bullish OB", DefaultValue = "#CC3179F5", Group = "Order Blocks")]
        public Color InternalBullishObColor { get; set; }

        [Parameter("Internal Bearish OB", DefaultValue = "#CCF77C80", Group = "Order Blocks")]
        public Color InternalBearishObColor { get; set; }

        [Parameter("Bullish OB", DefaultValue = "#CC1848CC", Group = "Order Blocks")]
        public Color SwingBullishObColor { get; set; }

        [Parameter("Bearish OB", DefaultValue = "#CCB22833", Group = "Order Blocks")]
        public Color SwingBearishObColor { get; set; }

        // ─── Parameters: EQH/EQL ─────────────────────────────────────────────────
        [Parameter("Equal High/Low", DefaultValue = true, Group = "EQH/EQL")]
        public bool ShowEqualHighsLowsInput { get; set; }

        [Parameter("Bars Confirmation", DefaultValue = 3, MinValue = 1, Group = "EQH/EQL")]
        public int EqualHighsLowsLengthInput { get; set; }

        [Parameter("Threshold", DefaultValue = 0.1, MinValue = 0, MaxValue = 0.5, Group = "EQH/EQL")]
        public double EqualHighsLowsThresholdInput { get; set; }

        [Parameter("Label Size", DefaultValue = LabelSizeOpt.Tiny, Group = "EQH/EQL")]
        public LabelSizeOpt EqualHighsLowsSizeInput { get; set; }

        // ─── Parameters: Fair Value Gaps ──────────────────────────────────────────
        [Parameter("Fair Value Gaps", DefaultValue = false, Group = "Fair Value Gaps")]
        public bool ShowFairValueGapsInput { get; set; }

        [Parameter("Auto Threshold", DefaultValue = true, Group = "Fair Value Gaps")]
        public bool FairValueGapsThresholdInput { get; set; }

        // Leave at chart timeframe to mirror Pine's default empty-string (no HTF)
        [Parameter("FVG Timeframe", Group = "Fair Value Gaps")]
        public TimeFrame FairValueGapsTimeFrame { get; set; }

        [Parameter("Bullish FVG", DefaultValue = "#7000FF68", Group = "Fair Value Gaps")]
        public Color FairValueGapsBullColorInput { get; set; }

        [Parameter("Bearish FVG", DefaultValue = "#70FF0008", Group = "Fair Value Gaps")]
        public Color FairValueGapsBearColorInput { get; set; }

        [Parameter("Extend FVG (bars)", DefaultValue = 1, MinValue = 0, Group = "Fair Value Gaps")]
        public int FairValueGapsExtendInput { get; set; }

        // ─── Parameters: Highs & Lows MTF ────────────────────────────────────────
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

        // ─── Parameters: Premium & Discount Zones ────────────────────────────────
        [Parameter("Premium/Discount Zones", DefaultValue = false, Group = "Premium & Discount Zones")]
        public bool ShowPremiumDiscountZonesInput { get; set; }

        [Parameter("Premium Zone", DefaultValue = "#F23645", Group = "Premium & Discount Zones")]
        public Color PremiumZoneColorInput { get; set; }

        [Parameter("Equilibrium Zone", DefaultValue = "#878B94", Group = "Premium & Discount Zones")]
        public Color EquilibriumZoneColorInput { get; set; }

        [Parameter("Discount Zone", DefaultValue = "#089981", Group = "Premium & Discount Zones")]
        public Color DiscountZoneColorInput { get; set; }

        // ─── Outputs ──────────────────────────────────────────────────────────────
        [Output("Internal Bullish BOS",  LineColor = "#089981", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries InternalBullishBOS { get; set; }

        [Output("Internal Bearish BOS",  LineColor = "#F23645", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries InternalBearishBOS { get; set; }

        [Output("Internal Bullish CHoCH", LineColor = "#089981", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries InternalBullishCHoCH { get; set; }

        [Output("Internal Bearish CHoCH", LineColor = "#F23645", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries InternalBearishCHoCH { get; set; }

        [Output("Swing Bullish BOS",  LineColor = "#089981", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries SwingBullishBOS { get; set; }

        [Output("Swing Bearish BOS",  LineColor = "#F23645", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries SwingBearishBOS { get; set; }

        [Output("Swing Bullish CHoCH", LineColor = "#089981", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries SwingBullishCHoCH { get; set; }

        [Output("Swing Bearish CHoCH", LineColor = "#F23645", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries SwingBearishCHoCH { get; set; }

        [Output("Equal High", LineColor = "#2157F3", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries EqualHighSignal { get; set; }

        [Output("Equal Low",  LineColor = "#2157F3", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries EqualLowSignal { get; set; }

        [Output("Bullish FVG", LineColor = "#089981", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries BullishFvgSignal { get; set; }

        [Output("Bearish FVG", LineColor = "#F23645", PlotType = PlotType.Points, Thickness = 3)]
        public IndicatorDataSeries BearishFvgSignal { get; set; }

        // ─── Pivot state ─────────────────────────────────────────────────────────
        private class PivotState
        {
            public double CurrentLevel = double.NaN;
            public double LastLevel    = double.NaN;
            public bool   Crossed      = false;
            public DateTime BarTime    = DateTime.MinValue;
            public int    BarIndex     = -1;
        }

        // ─── Order block record ───────────────────────────────────────────────────
        private class ObRecord
        {
            public double  BarHigh;
            public double  BarLow;
            public DateTime BarTime;
            public int     BarIndex;
            public int     Bias;      // +1 bullish, -1 bearish
            public bool    Internal;
            public string  RectId1;
            public string  RectId2;   // border rect (swing only)
        }

        // ─── FVG record ───────────────────────────────────────────────────────────
        private class FvgRecord
        {
            public double  TopPrice;     // higher bound of gap
            public double  BottomPrice;  // lower bound of gap
            public int     Bias;         // +1 bullish, -1 bearish
            public string  RectIdTop;
            public string  RectIdBottom;
            // For mitigation check:
            // Bullish mitigated when low < BottomPrice
            // Bearish mitigated when high > TopPrice
        }

        // ─── State fields ─────────────────────────────────────────────────────────
        // Leg states (persist across bars, like Pine's `var leg`)
        private int _swingLeg    = 0;
        private int _intLeg      = 0;
        private int _equalLeg    = 0;
        private int _prevSwingLeg   = -99;
        private int _prevIntLeg     = -99;
        private int _prevEqualLeg   = -99;

        // Pivots
        private readonly PivotState _swingHigh    = new PivotState();
        private readonly PivotState _swingLow     = new PivotState();
        private readonly PivotState _intHigh      = new PivotState();
        private readonly PivotState _intLow       = new PivotState();
        private readonly PivotState _equalHigh    = new PivotState();
        private readonly PivotState _equalLow     = new PivotState();

        // Trend biases: +1 bullish, -1 bearish, 0 unknown
        private int _swingTrend  = 0;
        private int _intTrend    = 0;

        // Previous close for crossover/crossunder detection
        private double _prevClose = double.NaN;

        // Trailing extremes (for Strong/Weak H/L and premium zones)
        private double   _trailTop          = double.NaN;
        private double   _trailBottom       = double.NaN;
        private DateTime _trailBarTime      = DateTime.MinValue;
        private int      _trailBarIndex     = -1;
        private DateTime _trailLastTopTime  = DateTime.MinValue;
        private DateTime _trailLastBotTime  = DateTime.MinValue;

        // Cumulative data arrays for OB selection and MTF level lookup
        private readonly List<double>   _parsedHighs = new List<double>();
        private readonly List<double>   _parsedLows  = new List<double>();
        private readonly List<double>   _rawHighs    = new List<double>();
        private readonly List<double>   _rawLows     = new List<double>();
        private readonly List<DateTime> _times       = new List<DateTime>();

        // Running cumulative TR sum for CumulativeMeanRange filter (mirrors ta.cum(ta.tr)/bar_index)
        private double _cumTrSum = 0;
        // Running short ATR sum (200-bar window) for volatile bar detection
        private readonly Queue<double> _atrWindow = new Queue<double>(200);

        // Order block lists
        private readonly List<ObRecord> _intObs   = new List<ObRecord>();
        private readonly List<ObRecord> _swingObs = new List<ObRecord>();

        // FVG list
        private readonly List<FvgRecord> _fvgs = new List<FvgRecord>();

        // Cumulative FVG threshold state
        private double _cumFvgDeltaAbs = 0;

        // MTF bars (loaded once)
        private Bars _dailyBars, _weeklyBars, _monthlyBars;
        private Bars _fvgBars;

        // Prev structure line names for Present mode
        private string _prevIntBullLine = null, _prevIntBearLine = null;
        private string _prevSwgBullLine = null, _prevSwgBearLine = null;
        private string _prevEqhLine = null, _prevEqlLine = null;
        // Trailing extreme line names (reused every bar)
        private const string TrailTopLineId   = "smc_trail_top_line";
        private const string TrailBotLineId   = "smc_trail_bot_line";
        private const string TrailTopLabelId  = "smc_trail_top_lbl";
        private const string TrailBotLabelId  = "smc_trail_bot_lbl";
        // Zone rect ids
        private const string PremiumRectId    = "smc_zone_premium";
        private const string EqRectId         = "smc_zone_eq";
        private const string DiscountRectId   = "smc_zone_discount";

        // ─── Initialize ──────────────────────────────────────────────────────────
        protected override void Initialize()
        {
            if (ShowDailyLevelsInput)
                _dailyBars = MarketData.GetBars(TimeFrame.Daily);
            if (ShowWeeklyLevelsInput)
                _weeklyBars = MarketData.GetBars(TimeFrame.Weekly);
            if (ShowMonthlyLevelsInput)
                _monthlyBars = MarketData.GetBars(TimeFrame.Monthly);

            // FVG HTF bars — if user selected a different TF from the chart, load it.
            // If same as chart (or unset/null), _fvgBars stays null → use current chart bars.
            if (ShowFairValueGapsInput && FairValueGapsTimeFrame != null &&
                FairValueGapsTimeFrame != TimeFrame)
            {
                try { _fvgBars = MarketData.GetBars(FairValueGapsTimeFrame); }
                catch { _fvgBars = null; }
            }
        }

        // ─── Main Calculate ───────────────────────────────────────────────────────
        public override void Calculate(int index)
        {
            ResetSignals(index);

            // Build cumulative arrays (mirrors Pine's per-bar push)
            BuildArrays(index);

            // Update trailing extremes every bar (needed for zones & H/L swings)
            if (ShowHighLowSwingsInput || ShowPremiumDiscountZonesInput)
                UpdateTrailingExtremes(index);

            // Process structure for all three sizes
            // Swing: user-defined SwingsLengthInput
            ProcessStructure(index, SwingsLengthInput, false, false);
            // Internal: HARDCODED size = 5 (mirrors Pine: getCurrentStructure(5, false, true))
            ProcessStructure(index, 5, false, true);
            // Equal H/L: equalHighsLowsLengthInput
            if (ShowEqualHighsLowsInput)
                ProcessStructure(index, EqualHighsLowsLengthInput, true, false);

            // Detect BOS/CHoCH and manage OBs
            if (ShowInternalsInput || ShowInternalOrderBlocksInput || ShowTrendInput)
                DisplayStructure(index, true);
            if (ShowStructureInput || ShowSwingOrderBlocksInput || ShowHighLowSwingsInput)
                DisplayStructure(index, false);

            // Mitigate (delete) crossed order blocks
            if (ShowInternalOrderBlocksInput)
                MitigateOrderBlocks(index, true);
            if (ShowSwingOrderBlocksInput)
                MitigateOrderBlocks(index, false);

            // Redraw active OBs (extend right boundary to current bar)
            if (ShowInternalOrderBlocksInput)
                RedrawOrderBlocks(index, true);
            if (ShowSwingOrderBlocksInput)
                RedrawOrderBlocks(index, false);

            // FVGs
            if (ShowFairValueGapsInput)
            {
                MitigateFvgs(index);
                DrawFairValueGaps(index);
            }

            // Trailing High/Low swing lines
            if (ShowHighLowSwingsInput)
                DrawHighLowSwings(index);

            // Premium / Discount zones
            if (ShowPremiumDiscountZonesInput)
                DrawPremiumDiscountZones(index);

            // MTF levels (drawn at last confirmed history bar and on new bar in realtime)
            if (IsAtLastBar(index))
            {
                if (ShowDailyLevelsInput && _dailyBars != null)
                    DrawMtfLevels(index, _dailyBars, DailyLevelsColorInput, DailyLevelsStyleInput, "D");
                if (ShowWeeklyLevelsInput && _weeklyBars != null)
                    DrawMtfLevels(index, _weeklyBars, WeeklyLevelsColorInput, WeeklyLevelsStyleInput, "W");
                if (ShowMonthlyLevelsInput && _monthlyBars != null)
                    DrawMtfLevels(index, _monthlyBars, MonthlyLevelsColorInput, MonthlyLevelsStyleInput, "M");
            }

            // Candle coloring (based on internalTrend, mirrors Pine)
            if (ShowTrendInput)
                ColorBar(index);

            // Store previous close for next bar's crossover detection
            _prevClose = Bars.ClosePrices[index];
        }

        // ─── Build per-bar arrays ─────────────────────────────────────────────────
        // Mirrors Pine's per-bar push of parsedHighs, parsedLows, highs, lows, times.
        // Called once per bar; adds exactly ONE entry per Calculate(index) call.
        private void BuildArrays(int index)
        {
            if (_times.Count > index) return; // already built for this bar

            int i = index; // always just one new bar per Calculate call
            double h = Bars.HighPrices[i];
            double l = Bars.LowPrices[i];
            double c = Bars.ClosePrices[i];

            // Incremental true range
            double prevClose = i > 0 ? Bars.ClosePrices[i - 1] : c;
            double tr = Math.Max(h - l,
                         Math.Max(Math.Abs(h - prevClose), Math.Abs(l - prevClose)));

            // Incremental running ATR-200 (sliding window of last 200 TRs)
            _atrWindow.Enqueue(tr);
            if (_atrWindow.Count > 200) _atrWindow.Dequeue();
            double atr200 = _atrWindow.Count > 0 ? AtrWindowSum() / _atrWindow.Count : tr;

            // Incremental ta.cum(ta.tr)/bar_index
            _cumTrSum += tr;
            double cumTrMeasure = i > 0 ? _cumTrSum / i : tr;

            double measure = OrderBlockFilterInput == ObFilterOpt.Atr ? atr200 : cumTrMeasure;
            bool   hiVol   = (h - l) >= (2.0 * measure);

            _parsedHighs.Add(hiVol ? l : h);
            _parsedLows.Add(hiVol ? h : l);
            _rawHighs.Add(h);
            _rawLows.Add(l);
            _times.Add(Bars.OpenTimes[i]);
        }

        private double AtrWindowSum()
        {
            double sum = 0;
            foreach (var v in _atrWindow) sum += v;
            return sum;
        }

        // ─── Leg computation ─────────────────────────────────────────────────────
        // Mirrors Pine's leg(size) — detects pivot high/low via rolling window
        //   leg=0 (BEARISH_LEG) when high[size] > highest(high, size)  [pivot high formed]
        //   leg=1 (BULLISH_LEG) when low[size]  < lowest(low,  size)   [pivot low formed]
        private int ComputeLeg(int index, int size, ref int legState)
        {
            int pivotIdx = index - size;
            if (pivotIdx < 0) return legState;

            double pivotHigh = Bars.HighPrices[pivotIdx];
            double pivotLow  = Bars.LowPrices[pivotIdx];

            // ta.highest(size) = max of high[0..size-1] = Bars.HighPrices[pivotIdx+1 .. index]
            double maxRecent = double.MinValue;
            double minRecent = double.MaxValue;
            for (int j = pivotIdx + 1; j <= index; j++)
            {
                if (Bars.HighPrices[j] > maxRecent) maxRecent = Bars.HighPrices[j];
                if (Bars.LowPrices[j]  < minRecent) minRecent = Bars.LowPrices[j];
            }

            if (pivotHigh > maxRecent)      legState = 0; // BEARISH_LEG → pivot high at pivotIdx
            else if (pivotLow < minRecent)  legState = 1; // BULLISH_LEG → pivot low at pivotIdx

            return legState;
        }

        // ─── Process Structure ────────────────────────────────────────────────────
        // Mirrors Pine's getCurrentStructure(size, equalHighLow, internal)
        private void ProcessStructure(int index, int size, bool equalHighLow, bool internalMode)
        {
            int prevLeg = equalHighLow ? _prevEqualLeg
                        : internalMode ? _prevIntLeg : _prevSwingLeg;
            int curLeg;
            if (equalHighLow)
            {
                curLeg = ComputeLeg(index, size, ref _equalLeg);
                _prevEqualLeg = curLeg;
            }
            else if (internalMode)
            {
                curLeg = ComputeLeg(index, size, ref _intLeg);
                _prevIntLeg = curLeg;
            }
            else
            {
                curLeg = ComputeLeg(index, size, ref _swingLeg);
                _prevSwingLeg = curLeg;
            }

            bool newPivot  = (prevLeg != -99) && (curLeg != prevLeg);
            if (!newPivot) return;

            bool pivotLow  = (curLeg == 1); // startOfBullishLeg → pivot low formed
            bool pivotHigh = (curLeg == 0); // startOfBearishLeg → pivot high formed
            int  pivotBarIdx = index - size;
            if (pivotBarIdx < 0 || pivotBarIdx >= Bars.Count) return;

            double atr = ComputeAtr(index, 200);

            if (pivotLow)
            {
                double lvl = Bars.LowPrices[pivotBarIdx];
                PivotState ps = equalHighLow ? _equalLow
                              : internalMode ? _intLow : _swingLow;

                // EQH/EQL detection — ATR-based threshold (mirrors Pine exactly)
                if (equalHighLow && !double.IsNaN(ps.CurrentLevel))
                {
                    if (Math.Abs(ps.CurrentLevel - lvl) < EqualHighsLowsThresholdInput * atr)
                    {
                        EqualLowSignal[index] = 1;
                        DrawEqualLine(index, ps, lvl, pivotBarIdx, false);
                    }
                }

                // Update pivot
                ps.LastLevel    = ps.CurrentLevel;
                ps.CurrentLevel = lvl;
                ps.Crossed      = false;
                ps.BarTime      = Bars.OpenTimes[pivotBarIdx];
                ps.BarIndex     = pivotBarIdx;

                // Update trailing swing bottom (swing only)
                if (!equalHighLow && !internalMode)
                {
                    _trailBottom    = lvl;
                    _trailBarTime   = ps.BarTime;
                    _trailBarIndex  = pivotBarIdx;
                    _trailLastBotTime = ps.BarTime;
                }

                // Show swing point label: LL or HL
                if (ShowSwingsInput && !internalMode && !equalHighLow)
                {
                    string lbl = (!double.IsNaN(ps.LastLevel) && lvl < ps.LastLevel) ? "LL" : "HL";
                    string lblId = $"smc_swpt_low_{index}";
                    if (ModeInput == DisplayMode.Present) RemovePrefixed("smc_swpt_low_", lblId);
                    Color c = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#B2B5BE") : SwingBullColorInput;
                    Chart.DrawText(lblId, lbl, pivotBarIdx, lvl, c);
                }
            }
            else if (pivotHigh)
            {
                double lvl = Bars.HighPrices[pivotBarIdx];
                PivotState ps = equalHighLow ? _equalHigh
                              : internalMode ? _intHigh : _swingHigh;

                if (equalHighLow && !double.IsNaN(ps.CurrentLevel))
                {
                    if (Math.Abs(ps.CurrentLevel - lvl) < EqualHighsLowsThresholdInput * atr)
                    {
                        EqualHighSignal[index] = 1;
                        DrawEqualLine(index, ps, lvl, pivotBarIdx, true);
                    }
                }

                ps.LastLevel    = ps.CurrentLevel;
                ps.CurrentLevel = lvl;
                ps.Crossed      = false;
                ps.BarTime      = Bars.OpenTimes[pivotBarIdx];
                ps.BarIndex     = pivotBarIdx;

                if (!equalHighLow && !internalMode)
                {
                    _trailTop       = lvl;
                    _trailBarTime   = ps.BarTime;
                    _trailBarIndex  = pivotBarIdx;
                    _trailLastTopTime = ps.BarTime;
                }

                if (ShowSwingsInput && !internalMode && !equalHighLow)
                {
                    string lbl = (!double.IsNaN(ps.LastLevel) && lvl > ps.LastLevel) ? "HH" : "LH";
                    string lblId = $"smc_swpt_high_{index}";
                    if (ModeInput == DisplayMode.Present) RemovePrefixed("smc_swpt_high_", lblId);
                    Color c = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#5D606B") : SwingBearColorInput;
                    Chart.DrawText(lblId, lbl, pivotBarIdx, lvl, c);
                }
            }
        }

        // ─── Display Structure (BOS / CHoCH) ─────────────────────────────────────
        // Mirrors Pine's displayStructure(internal)
        private void DisplayStructure(int index, bool internalMode)
        {
            if (index < 1 || double.IsNaN(_prevClose)) return;

            // Confluence filter (mirrors Pine logic exactly)
            bool bullishBar = true, bearishBar = true;
            if (InternalFilterConfluenceInput)
            {
                double h = Bars.HighPrices[index];
                double l = Bars.LowPrices[index];
                double c = Bars.ClosePrices[index];
                double o = Bars.OpenPrices[index];
                double upperWick = h - Math.Max(c, o);
                double lower     = Math.Min(c, o - l); // mirrors Pine: math.min(close, open - low)
                bullishBar = upperWick > lower;
                bearishBar = upperWick < lower;
            }

            PivotState ph = internalMode ? _intHigh  : _swingHigh;
            PivotState pl = internalMode ? _intLow   : _swingLow;

            // Extra condition for internal — mirrors Pine exactly:
            // internalHigh.currentLevel != swingHigh.currentLevel and bullishBar
            bool extraBull = internalMode ? (!double.IsNaN(_intHigh.CurrentLevel) &&
                                             !double.IsNaN(_swingHigh.CurrentLevel) &&
                                             _intHigh.CurrentLevel != _swingHigh.CurrentLevel &&
                                             bullishBar)
                                          : true;
            bool extraBear = internalMode ? (!double.IsNaN(_intLow.CurrentLevel) &&
                                             !double.IsNaN(_swingLow.CurrentLevel) &&
                                             _intLow.CurrentLevel != _swingLow.CurrentLevel &&
                                             bearishBar)
                                          : true;

            // Resolved colors with monochrome support
            Color bullColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#B2B5BE")
                              : (internalMode ? InternalBullColorInput : SwingBullColorInput);
            Color bearColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#5D606B")
                              : (internalMode ? InternalBearColorInput : SwingBearColorInput);

            // Line style: internal = dashed, swing = solid (mirrors Pine)
            LineStyle lineStyle = internalMode ? LineStyle.Lines : LineStyle.Solid;

            double curClose  = Bars.ClosePrices[index];

            // ── Bullish break (ta.crossover) ──
            if (!double.IsNaN(ph.CurrentLevel) && !ph.Crossed && extraBull)
            {
                bool crossover = _prevClose <= ph.CurrentLevel && curClose > ph.CurrentLevel;
                if (crossover)
                {
                    bool  choch = (_swingTrend == -1 && !internalMode) || (_intTrend == -1 && internalMode);
                    // Note: for swing trend, CHoCH = previous swing trend was BEARISH
                    // For internal trend, same
                    if (!internalMode) choch = _swingTrend == -1;
                    else               choch = _intTrend == -1;

                    string tag = choch ? "CHoCH" : "BOS";

                    // Emit signals
                    if (internalMode)
                    {
                        if (choch) InternalBullishCHoCH[index] = 1; else InternalBullishBOS[index] = 1;
                        _intTrend = 1;
                    }
                    else
                    {
                        if (choch) SwingBullishCHoCH[index] = 1; else SwingBullishBOS[index] = 1;
                        _swingTrend = 1;
                    }
                    ph.Crossed = true;

                    // Display condition
                    StructureFilter filter = internalMode ? ShowInternalBullInput : ShowSwingBullInput;
                    bool showIt = internalMode ? ShowInternalsInput : ShowStructureInput;
                    if (showIt && ShouldShow(filter, choch))
                    {
                        ref string prevRef = ref (internalMode ? ref _prevIntBullLine : ref _prevSwgBullLine);
                        DrawStructureLine(index, ph, tag, bullColor, lineStyle, internalMode,
                                          internalMode ? InternalStructureSize : SwingStructureSize,
                                          true, ref prevRef);
                    }

                    // Store order block
                    bool storeOb = internalMode ? ShowInternalOrderBlocksInput : ShowSwingOrderBlocksInput;
                    if (storeOb)
                        StoreOrderBlock(index, ph, internalMode, 1);
                }
            }

            // ── Bearish break (ta.crossunder) ──
            if (!double.IsNaN(pl.CurrentLevel) && !pl.Crossed && extraBear)
            {
                bool crossunder = _prevClose >= pl.CurrentLevel && curClose < pl.CurrentLevel;
                if (crossunder)
                {
                    bool choch = (!internalMode && _swingTrend == 1) || (internalMode && _intTrend == 1);

                    string tag = choch ? "CHoCH" : "BOS";

                    if (internalMode)
                    {
                        if (choch) InternalBearishCHoCH[index] = 1; else InternalBearishBOS[index] = 1;
                        _intTrend = -1;
                    }
                    else
                    {
                        if (choch) SwingBearishCHoCH[index] = 1; else SwingBearishBOS[index] = 1;
                        _swingTrend = -1;
                    }
                    pl.Crossed = true;

                    StructureFilter filter = internalMode ? ShowInternalBearInput : ShowSwingBearInput;
                    bool showIt = internalMode ? ShowInternalsInput : ShowStructureInput;
                    if (showIt && ShouldShow(filter, choch))
                    {
                        ref string prevRef = ref (internalMode ? ref _prevIntBearLine : ref _prevSwgBearLine);
                        DrawStructureLine(index, pl, tag, bearColor, lineStyle, internalMode,
                                          internalMode ? InternalStructureSize : SwingStructureSize,
                                          false, ref prevRef);
                    }

                    bool storeOb = internalMode ? ShowInternalOrderBlocksInput : ShowSwingOrderBlocksInput;
                    if (storeOb)
                        StoreOrderBlock(index, pl, internalMode, -1);
                }
            }
        }

        // ─── Draw Structure Line ──────────────────────────────────────────────────
        // Mirrors Pine's drawStructure: line from pivot to current bar,
        // label positioned at midpoint of bar range.
        // prevLineId is passed by ref so the caller's field can be updated for Present mode.
        private void DrawStructureLine(int index, PivotState pivot, string tag,
                                       Color color, LineStyle lineStyle,
                                       bool internalMode, LabelSizeOpt labelSize,
                                       bool bullish, ref string prevLineId)
        {
            if (pivot.BarIndex < 0) return;

            string lineId = $"smc_struct_{(internalMode ? "i" : "s")}_{(bullish ? "b" : "r")}_{index}";
            string lblId  = lineId + "_lbl";
            int    midBar = pivot.BarIndex + (index - pivot.BarIndex) / 2;

            if (ModeInput == DisplayMode.Present && prevLineId != null)
            {
                Chart.RemoveObject(prevLineId);
                Chart.RemoveObject(prevLineId + "_lbl");
            }
            prevLineId = lineId;

            Chart.DrawTrendLine(lineId, pivot.BarIndex, pivot.CurrentLevel, index, pivot.CurrentLevel,
                                color, 1, lineStyle);
            Chart.DrawText(lblId, tag, midBar, pivot.CurrentLevel, color);
        }

        // ─── Draw Equal High/Low ──────────────────────────────────────────────────
        // Mirrors Pine's drawEqualHighLow: dotted line + label at midpoint
        private void DrawEqualLine(int index, PivotState prevPivot, double level,
                                   int pivotBarIdx, bool isHigh)
        {
            Color eqColor = StyleInput == ThemeStyle.Monochrome
                            ? (isHigh ? Color.FromHex("#5D606B") : Color.FromHex("#B2B5BE"))
                            : (isHigh ? SwingBearColorInput : SwingBullColorInput);
            string tag     = isHigh ? "EQH" : "EQL";
            string lineId  = $"smc_eq_{(isHigh ? "h" : "l")}_{index}";
            string lblId   = lineId + "_lbl";
            int    midBar  = prevPivot.BarIndex + (pivotBarIdx - prevPivot.BarIndex) / 2;

            if (ModeInput == DisplayMode.Present)
            {
                ref string prev = ref (isHigh ? ref _prevEqhLine : ref _prevEqlLine);
                if (prev != null) { Chart.RemoveObject(prev); Chart.RemoveObject(prev + "_lbl"); }
                prev = lineId;
            }

            Chart.DrawTrendLine(lineId, prevPivot.BarIndex, prevPivot.CurrentLevel,
                                pivotBarIdx, level, eqColor, 1, LineStyle.DotsRare);
            Chart.DrawText(lblId, tag, midBar, level, eqColor);
        }

        // ─── Store Order Block ────────────────────────────────────────────────────
        // Mirrors Pine's storeOrderBlock:
        //   BEARISH OB = bar with highest parsedHigh between pivot and current bar
        //   BULLISH OB = bar with lowest  parsedLow  between pivot and current bar
        private void StoreOrderBlock(int index, PivotState pivot, bool internalMode, int bias)
        {
            if (pivot.BarIndex < 0 || pivot.BarIndex >= index) return;

            int obBarIdx = pivot.BarIndex;

            if (bias == -1) // BEARISH: find highest parsedHigh
            {
                double maxPH = double.MinValue;
                for (int j = pivot.BarIndex; j < index && j < _parsedHighs.Count; j++)
                    if (_parsedHighs[j] > maxPH) { maxPH = _parsedHighs[j]; obBarIdx = j; }
            }
            else // BULLISH: find lowest parsedLow
            {
                double minPL = double.MaxValue;
                for (int j = pivot.BarIndex; j < index && j < _parsedLows.Count; j++)
                    if (_parsedLows[j] < minPL) { minPL = _parsedLows[j]; obBarIdx = j; }
            }

            if (obBarIdx < 0 || obBarIdx >= _parsedHighs.Count) return;

            string rectId = $"smc_ob_{(internalMode ? "i" : "s")}_{(bias > 0 ? "b" : "r")}_{obBarIdx}_{index}";

            var ob = new ObRecord
            {
                BarHigh  = _parsedHighs[obBarIdx],
                BarLow   = _parsedLows[obBarIdx],
                BarTime  = _times[obBarIdx],
                BarIndex = obBarIdx,
                Bias     = bias,
                Internal = internalMode,
                RectId1  = rectId
            };

            var list = internalMode ? _intObs : _swingObs;
            if (list.Count >= 100) list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        // ─── Mitigate Order Blocks ────────────────────────────────────────────────
        // Mirrors Pine's deleteOrderBlocks (triggered by mitigation source crossing OB)
        private void MitigateOrderBlocks(int index, bool internalMode)
        {
            var list = internalMode ? _intObs : _swingObs;
            double bearMit = OrderBlockMitigationInput == MitigationMode.Close
                             ? Bars.ClosePrices[index] : Bars.HighPrices[index];
            double bullMit = OrderBlockMitigationInput == MitigationMode.Close
                             ? Bars.ClosePrices[index] : Bars.LowPrices[index];

            for (int i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];
                bool mitigated = false;
                if (ob.Bias == -1 && bearMit > ob.BarHigh) mitigated = true;
                if (ob.Bias ==  1 && bullMit < ob.BarLow)  mitigated = true;
                if (mitigated)
                {
                    Chart.RemoveObject(ob.RectId1);
                    list.RemoveAt(i);
                }
            }
        }

        // ─── Redraw Order Blocks ──────────────────────────────────────────────────
        // Mirrors Pine's drawOrderBlocks — extends right boundary to current bar
        // Shows only the most recent N blocks per the size setting
        private void RedrawOrderBlocks(int index, bool internalMode)
        {
            var list    = internalMode ? _intObs : _swingObs;
            int maxShow = internalMode ? InternalOrderBlocksSizeInput : SwingOrderBlocksSizeInput;
            int shown   = 0;

            for (int i = 0; i < list.Count; i++)
            {
                var ob  = list[i];
                bool show = shown < maxShow;

                Color rawColor = StyleInput == ThemeStyle.Monochrome
                    ? (ob.Bias == -1 ? Color.FromArgb(50, 93, 96, 107)
                                     : Color.FromArgb(50, 178, 181, 190))
                    : (internalMode
                        ? (ob.Bias == -1 ? InternalBearishObColor : InternalBullishObColor)
                        : (ob.Bias == -1 ? SwingBearishObColor    : SwingBullishObColor));

                if (show)
                {
                    var rect = Chart.DrawRectangle(ob.RectId1, ob.BarIndex, ob.BarHigh,
                                                   index + 1, ob.BarLow, rawColor);
                    rect.IsFilled  = true;
                    // Swing OBs have a border (same color); internal OBs have no visible border
                    // ChartRectangle.Color controls the outline; set transparent for internal OBs
                    rect.Color = internalMode ? Color.Transparent : rawColor;
                    shown++;
                }
                else
                {
                    Chart.RemoveObject(ob.RectId1);
                }
            }
        }

        // ─── FVG Mitigation ───────────────────────────────────────────────────────
        // Mirrors Pine's deleteFairValueGaps
        private void MitigateFvgs(int index)
        {
            for (int i = _fvgs.Count - 1; i >= 0; i--)
            {
                var f = _fvgs[i];
                bool mitigated = false;
                // Bullish FVG: top=currentLow at creation, bottom=last2High
                //   mitigated when low < fvg.bottom (price falls below gap bottom)
                if (f.Bias == 1 && Bars.LowPrices[index] < f.BottomPrice)   mitigated = true;
                // Bearish FVG: top=currentHigh at creation (lower bound of gap), bottom=last2Low (upper bound)
                //   mitigated when high > fvg.top (price rises above gap lower bound = enters gap)
                if (f.Bias == -1 && Bars.HighPrices[index] > f.TopPrice)     mitigated = true;
                if (mitigated)
                {
                    Chart.RemoveObject(f.RectIdTop);
                    Chart.RemoveObject(f.RectIdBottom);
                    _fvgs.RemoveAt(i);
                }
            }
        }

        // ─── Draw Fair Value Gaps ─────────────────────────────────────────────────
        // Mirrors Pine's drawFairValueGaps with two-box split and cumulative threshold
        private void DrawFairValueGaps(int index)
        {
            // Determine source bars (HTF or current chart)
            double curHigh, curLow, lastClose, lastOpen, last2High, last2Low;
            bool   newTf;

            if (_fvgBars != null)
            {
                // HTF mode: find corresponding HTF bar
                int htfIdx = FindHtfBarIndex(_fvgBars, Bars.OpenTimes[index]);
                if (htfIdx < 2) return;
                bool newTfBar = htfIdx > 0 &&
                    (index == 0 || FindHtfBarIndex(_fvgBars, Bars.OpenTimes[index - 1]) < htfIdx);
                if (!newTfBar) return; // only process at start of new HTF bar
                curHigh   = _fvgBars.HighPrices[htfIdx];
                curLow    = _fvgBars.LowPrices[htfIdx];
                lastClose = _fvgBars.ClosePrices[htfIdx - 1];
                lastOpen  = _fvgBars.OpenPrices[htfIdx - 1];
                last2High = _fvgBars.HighPrices[htfIdx - 2];
                last2Low  = _fvgBars.LowPrices[htfIdx - 2];
                newTf     = true;
            }
            else
            {
                // Current chart TF — process on every bar
                if (index < 2) return;
                curHigh   = Bars.HighPrices[index];
                curLow    = Bars.LowPrices[index];
                lastClose = Bars.ClosePrices[index - 1];
                lastOpen  = Bars.OpenPrices[index - 1];
                last2High = Bars.HighPrices[index - 2];
                last2Low  = Bars.LowPrices[index - 2];
                newTf     = true;
            }

            if (!newTf) return;

            // Cumulative threshold — mirrors Pine's ta.cum(abs(barDeltaPercent)) / bar_index * 2
            double barDeltaPct = lastOpen > 0
                                 ? Math.Abs((lastClose - lastOpen) / lastOpen * 100.0)
                                 : 0;
            _cumFvgDeltaAbs += barDeltaPct;
            double threshold = FairValueGapsThresholdInput && index > 0
                               ? _cumFvgDeltaAbs / index * 2.0 : 0;

            double barDeltaSignedPct = lastOpen > 0
                                       ? (lastClose - lastOpen) / lastOpen * 100.0 : 0;

            // Bullish FVG: currentLow > last2High and lastClose > last2High
            bool bullFvg = curLow > last2High && lastClose > last2High
                           && barDeltaSignedPct > threshold;
            // Bearish FVG: currentHigh < last2Low and lastClose < last2Low
            bool bearFvg = curHigh < last2Low && lastClose < last2Low
                           && -barDeltaSignedPct > threshold;

            Color bullColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromArgb(70, 178, 181, 190) : FairValueGapsBullColorInput;
            Color bearColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromArgb(70, 93, 96, 107) : FairValueGapsBearColorInput;

            // FVG boxes extend rightward by FairValueGapsExtendInput bars from current bar
            int rightBar = Math.Min(index + FairValueGapsExtendInput, Bars.Count - 1);

            if (bullFvg)
            {
                BullishFvgSignal[index] = 1;
                // Two boxes: upper half [avg, curLow] and lower half [last2High, avg]
                double avg   = (curLow + last2High) / 2.0;
                string topId = $"smc_fvg_bt_{index}";
                string botId = $"smc_fvg_bb_{index}";
                DrawFvgRect(topId, index - 2, rightBar, curLow,   avg,      bullColor);
                DrawFvgRect(botId, index - 2, rightBar, avg,       last2High, bullColor);
                _fvgs.Add(new FvgRecord
                {
                    TopPrice    = curLow,    // higher bound of gap
                    BottomPrice = last2High, // lower bound of gap
                    Bias        = 1,
                    RectIdTop   = topId,
                    RectIdBottom = botId
                });
            }

            if (bearFvg)
            {
                BearishFvgSignal[index] = 1;
                // Two boxes: upper half [last2Low, avg] and lower half [avg, curHigh]
                double avg   = (curHigh + last2Low) / 2.0;
                string topId = $"smc_fvg_rt_{index}";
                string botId = $"smc_fvg_rb_{index}";
                DrawFvgRect(topId, index - 2, rightBar, last2Low, avg,     bearColor);
                DrawFvgRect(botId, index - 2, rightBar, avg,      curHigh, bearColor);
                _fvgs.Add(new FvgRecord
                {
                    TopPrice     = curHigh, // lower bound of gap (bears mitigated when high > this)
                    BottomPrice  = last2Low,
                    Bias         = -1,
                    RectIdTop    = topId,
                    RectIdBottom = botId
                });
            }
        }

        private void DrawFvgRect(string id, int leftBar, int rightBar,
                                  double topPrice, double bottomPrice, Color color)
        {
            var rect = Chart.DrawRectangle(id, leftBar, topPrice,
                                           Math.Min(rightBar, Bars.Count - 1), bottomPrice, color);
            rect.IsFilled = true;
            rect.Color    = color;
        }

        // ─── Update Trailing Extremes ─────────────────────────────────────────────
        // Mirrors Pine's updateTrailingExtremes: tracks absolute max/min
        private void UpdateTrailingExtremes(int index)
        {
            double h = Bars.HighPrices[index];
            double l = Bars.LowPrices[index];
            if (double.IsNaN(_trailTop) || h > _trailTop)
            {
                _trailTop         = h;
                _trailLastTopTime = Bars.OpenTimes[index];
            }
            if (double.IsNaN(_trailBottom) || l < _trailBottom)
            {
                _trailBottom      = l;
                _trailLastBotTime = Bars.OpenTimes[index];
            }
        }

        // ─── Draw Strong/Weak High/Low ────────────────────────────────────────────
        // Mirrors Pine's drawHighLowSwings:
        //   "Strong High" when swingTrend == BEARISH (last structure break was bearish)
        //   "Weak High"   when swingTrend != BEARISH
        //   "Strong Low"  when swingTrend == BULLISH
        //   "Weak Low"    when swingTrend != BULLISH
        private void DrawHighLowSwings(int index)
        {
            Color topColor = StyleInput == ThemeStyle.Monochrome
                             ? Color.FromHex("#5D606B") : SwingBearColorInput;
            Color botColor = StyleInput == ThemeStyle.Monochrome
                             ? Color.FromHex("#B2B5BE") : SwingBullColorInput;

            if (!double.IsNaN(_trailTop) && _trailLastTopTime != DateTime.MinValue)
            {
                int topStartBar = FindBarByTime(_trailLastTopTime);
                if (topStartBar >= 0)
                {
                    string topText = _swingTrend == -1 ? "Strong High" : "Weak High";
                    Chart.DrawTrendLine(TrailTopLineId, topStartBar, _trailTop,
                                        index + 1, _trailTop, topColor, 1, LineStyle.DotsRare);
                    Chart.DrawText(TrailTopLabelId, topText, index + 1, _trailTop, topColor);
                }
            }

            if (!double.IsNaN(_trailBottom) && _trailLastBotTime != DateTime.MinValue)
            {
                int botStartBar = FindBarByTime(_trailLastBotTime);
                if (botStartBar >= 0)
                {
                    string botText = _swingTrend == 1 ? "Strong Low" : "Weak Low";
                    Chart.DrawTrendLine(TrailBotLineId, botStartBar, _trailBottom,
                                        index + 1, _trailBottom, botColor, 1, LineStyle.DotsRare);
                    Chart.DrawText(TrailBotLabelId, botText, index + 1, _trailBottom, botColor);
                }
            }
        }

        // ─── Draw Premium / Discount / Equilibrium Zones ─────────────────────────
        // Mirrors Pine's drawPremiumDiscountZones using trailing extremes
        private void DrawPremiumDiscountZones(int index)
        {
            if (double.IsNaN(_trailTop) || double.IsNaN(_trailBottom)) return;
            if (_trailTop <= _trailBottom) return;
            if (_trailBarIndex < 0) return;

            double top    = _trailTop;
            double bottom = _trailBottom;
            int    leftBar = _trailBarIndex;
            int    rightBar = index;

            Color premColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#5D606B") : PremiumZoneColorInput;
            Color discColor = StyleInput == ThemeStyle.Monochrome
                              ? Color.FromHex("#B2B5BE") : DiscountZoneColorInput;
            Color eqColor   = EquilibriumZoneColorInput;

            // Premium zone: top to 0.95*top + 0.05*bottom
            DrawZoneRect(PremiumRectId, leftBar, top,
                         0.95 * top + 0.05 * bottom, rightBar,
                         premColor, "Premium");
            // Equilibrium zone: 0.525*top + 0.475*bottom to 0.525*bottom + 0.475*top
            double eqTop    = 0.525 * top + 0.475 * bottom;
            double eqBottom = 0.525 * bottom + 0.475 * top;
            DrawZoneRect(EqRectId, leftBar, eqTop, eqBottom, rightBar, eqColor, "Equilibrium");
            // Discount zone: 0.95*bottom + 0.05*top to bottom
            DrawZoneRect(DiscountRectId, leftBar,
                         0.95 * bottom + 0.05 * top, bottom, rightBar,
                         discColor, "Discount");
        }

        private void DrawZoneRect(string id, int leftBar, double topPrice, double bottomPrice,
                                   int rightBar, Color color, string label)
        {
            Color fillColor = WithAlpha(color, 80);
            var rect = Chart.DrawRectangle(id, Math.Max(0, leftBar), topPrice,
                                           Math.Min(Bars.Count - 1, rightBar), bottomPrice, fillColor);
            rect.IsFilled = true;
            rect.Color    = Color.Transparent; // no visible border on zone boxes
            Chart.DrawText(id + "_lbl", label, Math.Min(Bars.Count - 1, rightBar),
                           (topPrice + bottomPrice) / 2.0, color);
        }

        // ─── MTF Levels ───────────────────────────────────────────────────────────
        // Mirrors Pine's drawLevels — finds the specific bar with the period H/L
        // and draws extending lines with period labels
        private void DrawMtfLevels(int index, Bars htfBars, Color color,
                                    LineStyleOpt style, string tag)
        {
            int htfIdx = FindHtfBarIndex(htfBars, Bars.OpenTimes[index]);
            if (htfIdx < 1) return;

            // Previous complete HTF period
            int prevHtfIdx = htfIdx - 1;
            DateTime periodStart = htfBars.OpenTimes[prevHtfIdx];
            DateTime periodEnd   = htfBars.OpenTimes[htfIdx];

            // Find the chart bar with the max high and min low within the HTF period
            int highBar = -1, lowBar = -1;
            double maxH = double.MinValue, minL = double.MaxValue;

            for (int j = 0; j < Bars.Count; j++)
            {
                DateTime t = Bars.OpenTimes[j];
                if (t < periodStart || t >= periodEnd) continue;
                if (Bars.HighPrices[j] > maxH) { maxH = Bars.HighPrices[j]; highBar = j; }
                if (Bars.LowPrices[j]  < minL) { minL = Bars.LowPrices[j];  lowBar  = j; }
            }

            if (highBar < 0 || lowBar < 0) return;
            if (double.IsInfinity(maxH) || double.IsInfinity(minL)) return;

            LineStyle ls = MapLineStyle(style);
            int rightBar = Bars.Count - 1 + 20; // extend 20 bars to the right

            // High level
            Chart.DrawTrendLine($"lvl_{tag}_h_{index}", highBar, maxH, rightBar, maxH, color, 1, ls);
            Chart.DrawText($"lvl_{tag}_hl_{index}", $"P{tag}H", rightBar, maxH, color);

            // Low level
            Chart.DrawTrendLine($"lvl_{tag}_l_{index}", lowBar, minL, rightBar, minL, color, 1, ls);
            Chart.DrawText($"lvl_{tag}_ll_{index}", $"P{tag}L", rightBar, minL, color);
        }

        // ─── Bar Colouring ────────────────────────────────────────────────────────
        // Mirrors Pine: candleColor = internalTrend.bias == BULLISH ? bull : bear
        private void ColorBar(int index)
        {
            Color c = _intTrend >= 0
                ? (StyleInput == ThemeStyle.Monochrome ? Color.FromHex("#B2B5BE") : SwingBullColorInput)
                : (StyleInput == ThemeStyle.Monochrome ? Color.FromHex("#5D606B") : SwingBearColorInput);
            Chart.SetBarColor(index, c);
        }

        // ─── Helper: ShouldShow ───────────────────────────────────────────────────
        private static bool ShouldShow(StructureFilter filter, bool choch)
        {
            switch (filter)
            {
                case StructureFilter.All:   return true;
                case StructureFilter.BOS:   return !choch;
                case StructureFilter.CHoCH: return choch;
                default:                    return true;
            }
        }

        // ─── Helper: MapLineStyle ─────────────────────────────────────────────────
        private static LineStyle MapLineStyle(LineStyleOpt opt)
        {
            switch (opt)
            {
                case LineStyleOpt.Dashed: return LineStyle.Lines;
                case LineStyleOpt.Dotted: return LineStyle.DotsRare;
                default:                  return LineStyle.Solid;
            }
        }

        // ─── Helper: ComputeAtr ───────────────────────────────────────────────────
        // Approximates ta.atr(period) — simple average true range
        private double ComputeAtr(int index, int period)
        {
            int start = Math.Max(1, index - period + 1);
            double sum = 0;
            int    n   = 0;
            for (int i = start; i <= index && i < Bars.Count; i++)
            {
                double prevClose = Bars.ClosePrices[i - 1];
                double tr = Math.Max(Bars.HighPrices[i] - Bars.LowPrices[i],
                             Math.Max(Math.Abs(Bars.HighPrices[i] - prevClose),
                                      Math.Abs(Bars.LowPrices[i]  - prevClose)));
                sum += tr;
                n++;
            }
            return n > 0 ? sum / n : Symbol.TickSize;
        }


        // ─── Helper: FindBarByTime ────────────────────────────────────────────────
        private int FindBarByTime(DateTime t)
        {
            for (int i = _times.Count - 1; i >= 0; i--)
                if (_times[i] <= t) return i;
            return -1;
        }

        // ─── Helper: FindHtfBarIndex ──────────────────────────────────────────────
        private static int FindHtfBarIndex(Bars htfBars, DateTime chartTime)
        {
            for (int i = htfBars.Count - 1; i >= 0; i--)
                if (htfBars.OpenTimes[i] <= chartTime) return i;
            return -1;
        }

        // ─── Helper: IsAtLastBar ──────────────────────────────────────────────────
        // Renamed from IsLastBar to avoid hiding Indicator.IsLastBar property
        private bool IsAtLastBar(int index) => index == Bars.Count - 1;

        // ─── Helper: WithAlpha ────────────────────────────────────────────────────
        // Returns the same color with a new alpha (0-255).
        private static Color WithAlpha(Color c, int alpha)
            => Color.FromArgb(alpha, c.R, c.G, c.B);

        // ─── Helper: RemovePrefixed ───────────────────────────────────────────────
        // Removes all chart objects that start with a given prefix except one
        private void RemovePrefixed(string prefix, string keepId)
        {
            var toRemove = new List<string>();
            foreach (var obj in Chart.Objects)
                if (obj.Name.StartsWith(prefix) && obj.Name != keepId)
                    toRemove.Add(obj.Name);
            foreach (var name in toRemove)
                Chart.RemoveObject(name);
        }

        // ─── Reset Signals ────────────────────────────────────────────────────────
        private void ResetSignals(int index)
        {
            InternalBullishBOS[index]   = double.NaN;
            InternalBearishBOS[index]   = double.NaN;
            InternalBullishCHoCH[index] = double.NaN;
            InternalBearishCHoCH[index] = double.NaN;
            SwingBullishBOS[index]      = double.NaN;
            SwingBearishBOS[index]      = double.NaN;
            SwingBullishCHoCH[index]    = double.NaN;
            SwingBearishCHoCH[index]    = double.NaN;
            EqualHighSignal[index]      = double.NaN;
            EqualLowSignal[index]       = double.NaN;
            BullishFvgSignal[index]     = double.NaN;
            BearishFvgSignal[index]     = double.NaN;
        }
    }
}
