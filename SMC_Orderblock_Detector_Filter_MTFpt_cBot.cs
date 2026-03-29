using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo
{
    // ════════════════════════════════════════════════════════════════════════════
    //  SMC_Orderblock_Detector_Filter_MTFpt_cBot  (self-contained)
    //
    //  Base cBot : SMC_Orderblock_Detector_Filter_cBot — verbatim, unchanged.
    //  Addition  : Optional MTF Trend filter whose detection engine is embedded
    //              directly from Market Structure MTF Trend [Pt].
    //              No indicator reference.  No indicator output series.
    //              No threshold comparisons.
    //
    //  ── HOW THE MTF FILTER WORKS ─────────────────────────────────────────────
    //  EnableMtfFilter = false  →  bot is identical to base cBot.
    //
    //  EnableMtfFilter = true:
    //    For each enabled TF slot (TF1–TF4) the bot maintains a lightweight
    //    market-structure state (MtfTfState).  On every OnBar the state is
    //    advanced by processing new TF bars — exactly the same logic as
    //    ProcessTfCalcBar in the indicator — and the bot reads CurrentTrend
    //    directly from the state object.
    //
    //    Long  allowed only when enabled TF(s) are Bullish (CurrentTrend = true).
    //    Short allowed only when enabled TF(s) are Bearish (CurrentTrend = false).
    //
    //    Warmup (HasTrend = false) always passes — a TF that has not yet
    //    produced a structure break never blocks trades.
    //
    //  ── LOGIC TOGGLE ─────────────────────────────────────────────────────────
    //  OR  → trade allowed if AT LEAST ONE enabled TF agrees with direction.
    //  AND → trade allowed only if ALL  enabled TFs agree with direction.
    // ════════════════════════════════════════════════════════════════════════════

    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_Orderblock_Detector_Filter_MTFpt_cBot : Robot
    {
        // ════════════════════════════════════════════════════════════════════════
        //  Enums
        // ════════════════════════════════════════════════════════════════════════

        public enum ObFilter       { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum MtfLogicMode   { OR, AND }

        // ════════════════════════════════════════════════════════════════════════
        //  Inner types — OB / FVG entry trigger (verbatim from base cBot)
        // ════════════════════════════════════════════════════════════════════════

        private sealed class EntryObRecord
        {
            public double   Max;
            public double   Min;
            public bool     IsBull;
            public DateTime DetectionTime;
            public int      DetectionChartIndex;
        }

        private sealed class EntryFvgRecord
        {
            public double   Max;
            public double   Min;
            public bool     IsBull;
            public DateTime DetectionTime;
            public int      DetectionChartIndex;
        }

        private struct EntrySignalState
        {
            public double Point;
            public bool   IsBull;
            public bool   Entry;
            public int    Index;
        }

        private sealed class SmcObRecord
        {
            public int      Index;
            public double   Top;
            public double   Bottom;
            public bool     Bullish;
            public bool     Internal;
            public int      StructureBreakIndex;
            public DateTime Time;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Inner types — MTF market structure engine
        //
        //  Mirrors TfState from Market Structure MTF Trend [Pt].
        //  Chart-drawing, colour, alert and CalcByTfBar cache fields removed.
        //  CalcByTfBar is NOT needed: s.CurrentTrend and s.HasTrend are updated
        //  directly by MtfProcessCalcBar and read directly by CheckMtfFilter.
        // ════════════════════════════════════════════════════════════════════════

        private sealed class MtfTfState
        {
            // configuration
            public Bars  TfBars;
            public int   PivotLen;
            public bool  IsLowerTf;
            public int   TfMinutes;   // duration of one bar in minutes; used to find last closed bar

            // incremental processing cursor
            public int   LastProcessedTfBar = -1;

            // ── live trend state (read directly by CheckMtfFilter) ────────────
            public bool  CurrentTrend;   // true = bullish, false = bearish
            public bool  HasTrend;       // false during warmup; never resets to false

            // pivot tracking — mirrors TfState
            public double   LastPivotHigh  = double.NaN;
            public double   LastPivotLow   = double.NaN;
            public double   LastBrokenHigh = double.NaN;
            public double   LastBrokenLow  = double.NaN;
            public DateTime PivotHighTime  = DateTime.MinValue;
            public DateTime PivotLowTime   = DateTime.MinValue;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Entry Trigger (verbatim from base cBot)
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable OB Trigger",  DefaultValue = true,  Group = "Entry Trigger")]
        public bool EnableObTrigger  { get; set; }

        [Parameter("Enable FVG Trigger", DefaultValue = false, Group = "Entry Trigger")]
        public bool EnableFvgTrigger { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Entry Trigger Timeframe
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Use Chart Timeframe", DefaultValue = true,   Group = "Entry Trigger — Timeframe")]
        public bool UseChartTimeframe { get; set; }

        [Parameter("OB/FVG Timeframe",    DefaultValue = "Hour", Group = "Entry Trigger — Timeframe")]
        public TimeFrame InputTimeFrame { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Entry Trigger Signals
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Min Dist OB (bars)",  DefaultValue = 1, MinValue = 1, Group = "Entry Trigger — Signals")]
        public int MinDist { get; set; }

        [Parameter("Min Dist FVG (bars)", DefaultValue = 1, MinValue = 1, Group = "Entry Trigger — Signals")]
        public int MinDistFvg { get; set; }

        [Parameter("Use Heikin-Ashi", DefaultValue = false, Group = "Entry Trigger — Signals")]
        public bool UseHeikinAshi { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Risk
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("SL Lookback Bars", DefaultValue = 5, MinValue = 1, Group = "Risk")]
        public int StopLossLookbackBars { get; set; }

        [Parameter("Risk %", DefaultValue = 1.0, MinValue = 0.01, Step = 0.01, Group = "Risk")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1, Group = "Risk")]
        public double RiskRewardRatio { get; set; }

        [Parameter("SL Buffer (pips)", DefaultValue = 0.0, MinValue = 0.0, Step = 0.1, Group = "Risk")]
        public double StopLossBufferPips { get; set; }

        [Parameter("Max Open Positions", DefaultValue = 1, MinValue = 1, MaxValue = 100, Group = "Risk")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Instance Name", DefaultValue = "SMC_OB_MTFpt_cBot", Group = "Risk")]
        public string InstanceName { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — SMC Filter — Swing Structure
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "SMC Filter — Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — SMC Filter — Order Blocks
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "SMC Filter — Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter — Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter — Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Filter 1 (Internal OB)
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable Filter 1 (Internal OB)", DefaultValue = false, Group = "Filter 1 — Internal OB")]
        public bool EnableFilter1 { get; set; }

        [Parameter("Filter 1 Lookback (bars)", DefaultValue = 10, MinValue = 1, Group = "Filter 1 — Internal OB")]
        public int Filter1Lookback { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — Filter 2 (Swing OB)
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable Filter 2 (Swing OB)", DefaultValue = false, Group = "Filter 2 — Swing OB")]
        public bool EnableFilter2 { get; set; }

        [Parameter("Filter 2 Lookback (bars)", DefaultValue = 10, MinValue = 1, Group = "Filter 2 — Swing OB")]
        public int Filter2Lookback { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — OB Quality Filters
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable Min Bars From OB Origin", DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableMinBarsFromOrigin { get; set; }

        [Parameter("Min Bars — Internal OB", DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginInternal { get; set; }

        [Parameter("Min Bars — Swing OB", DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginSwing { get; set; }

        [Parameter("Enable ATR Distance Filter", DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableAtrDistanceFilter { get; set; }

        [Parameter("ATR Distance Multiplier", DefaultValue = 1.0, MinValue = 0.1, Step = 0.1, Group = "OB Quality Filters")]
        public double AtrDistanceMultiplier { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — MTF Trend Filter — General
        //
        //  Master switch.  Off = bot is identical to base cBot, no MTF code runs.
        //  MtfFilterLogic:
        //    OR  = at least one enabled TF must agree with trade direction.
        //    AND = all  enabled TFs must agree with trade direction.
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable MTF Trend Filter", DefaultValue = false, Group = "MTF Trend Filter — General")]
        public bool EnableMtfFilter { get; set; }

        [Parameter("Multi-TF Logic (OR / AND)", DefaultValue = MtfLogicMode.OR, Group = "MTF Trend Filter — General")]
        public MtfLogicMode MtfFilterLogic { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Parameters — MTF Trend Filter — TF1 … TF4
        //
        //  Timeframe strings: 1 2 3 4 5 10 15 30 45 60 120 240 480 720 D W M
        //  "Lower than chart?" = enable intrabar lookahead for sub-chart TFs.
        // ════════════════════════════════════════════════════════════════════════

        [Parameter("Enable TF1 Filter",    DefaultValue = true,  Group = "MTF Trend Filter — TF1")]
        public bool EnableMtfTf1 { get; set; }
        [Parameter("TF1 Timeframe",         DefaultValue = "15",  Group = "MTF Trend Filter — TF1")]
        public string MtfTimeframe1 { get; set; }
        [Parameter("TF1 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF1")]
        public int MtfPivotStrength1 { get; set; }
        [Parameter("TF1 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF1")]
        public bool MtfIsLowerTf1 { get; set; }

        [Parameter("Enable TF2 Filter",    DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool EnableMtfTf2 { get; set; }
        [Parameter("TF2 Timeframe",         DefaultValue = "30",  Group = "MTF Trend Filter — TF2")]
        public string MtfTimeframe2 { get; set; }
        [Parameter("TF2 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF2")]
        public int MtfPivotStrength2 { get; set; }
        [Parameter("TF2 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool MtfIsLowerTf2 { get; set; }

        [Parameter("Enable TF3 Filter",    DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool EnableMtfTf3 { get; set; }
        [Parameter("TF3 Timeframe",         DefaultValue = "60",  Group = "MTF Trend Filter — TF3")]
        public string MtfTimeframe3 { get; set; }
        [Parameter("TF3 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF3")]
        public int MtfPivotStrength3 { get; set; }
        [Parameter("TF3 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool MtfIsLowerTf3 { get; set; }

        [Parameter("Enable TF4 Filter",    DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool EnableMtfTf4 { get; set; }
        [Parameter("TF4 Timeframe",         DefaultValue = "240", Group = "MTF Trend Filter — TF4")]
        public string MtfTimeframe4 { get; set; }
        [Parameter("TF4 Pivot Strength",    DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF4")]
        public int MtfPivotStrength4 { get; set; }
        [Parameter("TF4 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool MtfIsLowerTf4 { get; set; }

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields — Entry trigger (verbatim from base cBot)
        // ════════════════════════════════════════════════════════════════════════

        private Bars                 _sourceBars;
        private List<EntryObRecord>  _obRecords  = new List<EntryObRecord>();
        private List<EntryFvgRecord> _fvgRecords = new List<EntryFvgRecord>();
        private EntrySignalState     _signal;
        private EntrySignalState     _signalFvg;
        private int _lastDetectedObSourceIndex  = -1;
        private int _lastDetectedFvgSourceIndex = -1;
        private readonly List<double> _haSourceOpen  = new List<double>();
        private readonly List<double> _haSourceClose = new List<double>();

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields — SMC filter engine (verbatim from base cBot)
        // ════════════════════════════════════════════════════════════════════════

        private readonly List<SmcObRecord> _smcInternalBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcInternalBearObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBullObs    = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBearObs    = new List<SmcObRecord>();
        private readonly List<double>   _parsedHighs = new List<double>();
        private readonly List<double>   _parsedLows  = new List<double>();
        private readonly List<DateTime> _times       = new List<DateTime>();
        private const int SmcAtrPeriod    = 200;
        private double    _smcAtrWilder    = double.NaN;
        private double    _smcAtrWilderSum = 0.0;
        private double    _smcCumTr        = 0.0;
        private int    _swingLeg;           private int    _swingTrend;
        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;   private bool   _swingLowCrossed;
        private int    _internalLeg;        private int    _internalTrend;
        private double _internalHighLevel   = double.NaN;
        private double _internalLowLevel    = double.NaN;
        private int    _internalHighIndex   = -1;
        private int    _internalLowIndex    = -1;
        private bool   _internalHighCrossed; private bool  _internalLowCrossed;
        private int       _smcObIdCounter;
        private const int MaxSmcObs       = 500;
        private int       _lastParsedIndex = -1;
        private int       _smcWarmup;

        // ════════════════════════════════════════════════════════════════════════
        //  Private fields — MTF engine
        //  null  = that slot is disabled or EnableMtfFilter = false
        // ════════════════════════════════════════════════════════════════════════

        private MtfTfState _mtfState1;
        private MtfTfState _mtfState2;
        private MtfTfState _mtfState3;
        private MtfTfState _mtfState4;

        // ════════════════════════════════════════════════════════════════════════
        //  OnStart
        // ════════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            var tf = UseChartTimeframe ? Bars.TimeFrame : InputTimeFrame;
            _sourceBars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            _smcWarmup  = Math.Max(SmcSwingsLengthInput, 5) + 5;
            _signal     = NewEmptyEntrySignal();
            _signalFvg  = NewEmptyEntrySignal();

            if (EnableMtfFilter)
            {
                _mtfState1 = EnableMtfTf1 ? MtfCreateState(MtfTimeframe1, MtfPivotStrength1, MtfIsLowerTf1) : null;
                _mtfState2 = EnableMtfTf2 ? MtfCreateState(MtfTimeframe2, MtfPivotStrength2, MtfIsLowerTf2) : null;
                _mtfState3 = EnableMtfTf3 ? MtfCreateState(MtfTimeframe3, MtfPivotStrength3, MtfIsLowerTf3) : null;
                _mtfState4 = EnableMtfTf4 ? MtfCreateState(MtfTimeframe4, MtfPivotStrength4, MtfIsLowerTf4) : null;

                Print("MTF filter ON. Logic={0}. TF1={1}({2}) TF2={3}({4}) TF3={5}({6}) TF4={7}({8})",
                    MtfFilterLogic,
                    EnableMtfTf1, MtfTimeframe1, EnableMtfTf2, MtfTimeframe2,
                    EnableMtfTf3, MtfTimeframe3, EnableMtfTf4, MtfTimeframe4);
            }
        }

        private MtfTfState MtfCreateState(string tfInput, int pivotStrength, bool isLowerTf)
        {
            var tf   = MtfParseTimeFrame(tfInput);
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            return new MtfTfState
            {
                TfBars    = bars,
                PivotLen  = Math.Max(1, pivotStrength),
                IsLowerTf = isLowerTf,
                TfMinutes = MtfTfMinutes(tf)
                // CurrentTrend = false, HasTrend = false  (C# default for bool)
                // LastProcessedTfBar = -1
            };
        }

        // ════════════════════════════════════════════════════════════════════════
        //  OnBar
        // ════════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            var index = Bars.Count - 2;

            // ── Step 1: SMC OB filter ────────────────────────────────────────
            RunSmcFilter(index);

            // ── Step 1b: MTF market structure engine ─────────────────────────
            // Only runs when EnableMtfFilter = true.
            // Processes all new TF bars since the last OnBar call.
            if (EnableMtfFilter)
            {
                var chartTime = Bars.OpenTimes[index];
                MtfAdvanceState(_mtfState1, chartTime);
                MtfAdvanceState(_mtfState2, chartTime);
                MtfAdvanceState(_mtfState3, chartTime);
                MtfAdvanceState(_mtfState4, chartTime);
            }

            // ── Step 2: Entry trigger ────────────────────────────────────────
            if (index < 2) return;
            var sourceIndex = FindBarIndexAtOrBefore(_sourceBars, Bars.OpenTimes[index]);
            if (sourceIndex < 2) return;

            EnsureHeikinAshiSource(sourceIndex);
            if (EnableObTrigger)  DetectOrderBlock(index, sourceIndex);

            var sHigh  = _sourceBars.HighPrices[sourceIndex];
            var sLow   = _sourceBars.LowPrices[sourceIndex];
            var sClose = _sourceBars.ClosePrices[sourceIndex];

            if (EnableObTrigger)  HandleMitigationOb(index, sLow, sHigh);
            if (EnableFvgTrigger) { DetectFvg(index, sourceIndex); HandleMitigationFvg(index, sLow, sHigh); }

            // ── Step 3: Signal evaluation ────────────────────────────────────
            var candleDir   = Bars.ClosePrices[index] > Bars.OpenPrices[index] ? 1 : -1;
            var signalClose = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            var fvgClose    = UseHeikinAshi ? _haSourceClose[sourceIndex] : sClose;
            var cond = 0; var condFvg = 0;

            if (EnableObTrigger && !double.IsNaN(_signal.Point))
            {
                if (signalClose > _signal.Point && _signal.IsBull    && candleDir ==  1 && !_signal.Entry)
                { cond = 1;  _signal = NewEmptyEntrySignal(); }
                if (signalClose < _signal.Point && !_signal.IsBull   && candleDir == -1 && !_signal.Entry)
                { cond = -1; _signal = NewEmptyEntrySignal(); }
            }
            if (EnableFvgTrigger && !double.IsNaN(_signalFvg.Point))
            {
                if (fvgClose > _signalFvg.Point && _signalFvg.IsBull  && candleDir ==  1 && !_signalFvg.Entry)
                { condFvg = 1;  _signalFvg = NewEmptyEntrySignal(); }
                if (fvgClose < _signalFvg.Point && !_signalFvg.IsBull && candleDir == -1 && !_signalFvg.Entry)
                { condFvg = -1; _signalFvg = NewEmptyEntrySignal(); }
            }

            var finalCond = cond != 0 ? cond : condFvg;
            if (finalCond == 0) return;

            // ── Step 4: SMC OB filter ────────────────────────────────────────
            if (!CheckFilters(index, finalCond)) return;

            // ── Step 4b: MTF trend filter ────────────────────────────────────
            // Skipped entirely when EnableMtfFilter = false.
            if (!CheckMtfFilter(finalCond > 0, index)) return;

            // ── Step 5: Execute trade ────────────────────────────────────────
            ExecuteSignalTrade(index, finalCond);
        }

        // ════════════════════════════════════════════════════════════════════════
        //  MTF engine — self-contained
        //
        //  MtfAdvanceState  : finds which TF bar corresponds to the current chart
        //                     bar and processes all new TF bars since last call.
        //  MtfProcessCalcBar: verbatim port of ProcessTfCalcBar from the indicator
        //                     (chart drawing, alerts, CalcByTfBar cache removed).
        //  MtfResolveTfBar  : verbatim port of ResolveTfBarForChartBar.
        //
        //  KEY DESIGN — no CalcByTfBar cache:
        //    The indicator uses CalcByTfBar to look up historical values for
        //    chart rendering.  The cBot only needs the CURRENT trend state, which
        //    is already held directly in s.CurrentTrend and s.HasTrend after
        //    MtfProcessCalcBar runs.  Reading back from a cache would only risk
        //    overwriting correct live state with a stale snapshot, so the cache
        //    is omitted entirely.
        // ════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Advances one MTF state slot to the TF bar that corresponds to
        /// chartTime.  Processes every new TF bar since the last call so that
        /// s.CurrentTrend and s.HasTrend are always up to date.
        /// </summary>
        private void MtfAdvanceState(MtfTfState s, DateTime chartTime)
        {
            if (s == null) return;

            var tfBarIndex = MtfResolveTfBar(s, chartTime);
            if (tfBarIndex < 0) return;

            // Process every TF bar not yet seen.
            // On the first call this catches up from bar 0.
            // On subsequent calls this typically processes 0 or 1 new TF bars.
            for (var i = s.LastProcessedTfBar + 1; i <= tfBarIndex; i++)
                MtfProcessCalcBar(s, i);

            // Advance the cursor.  Never move it backward.
            if (tfBarIndex > s.LastProcessedTfBar)
                s.LastProcessedTfBar = tfBarIndex;

            // s.CurrentTrend and s.HasTrend are now authoritative for this bar.
            // No cache read — the direct mutation in MtfProcessCalcBar is the
            // single source of truth.
        }

        /// <summary>
        /// Verbatim port of ProcessTfCalcBar from Market Structure MTF Trend [Pt].
        /// Detects pivot highs / lows and BOS / CHoCH structure breaks.
        /// Directly updates s.CurrentTrend and s.HasTrend — no cache.
        /// </summary>
        private void MtfProcessCalcBar(MtfTfState s, int tfBarIndex)
        {
            var bars              = s.TfBars;
            var prevLastPivotHigh = s.LastPivotHigh;  // snapshot before any update this bar
            var prevLastPivotLow  = s.LastPivotLow;

            // ── Pivot detection (mirrors IsPivotHigh / IsPivotLow checks) ────
            if (tfBarIndex >= s.PivotLen * 2)
            {
                var pivotIdx = tfBarIndex - s.PivotLen;

                if (MtfIsPivotHigh(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.HighPrices[pivotIdx];
                    // In bullish trend: keep the highest pivot; otherwise replace
                    s.LastPivotHigh = s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotHigh) ? pp : Math.Max(pp, s.LastPivotHigh))
                        : pp;
                    if (s.LastPivotHigh != prevLastPivotHigh)
                        s.PivotHighTime = bars.OpenTimes[pivotIdx];
                }

                if (MtfIsPivotLow(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.LowPrices[pivotIdx];
                    // In bearish trend: keep the lowest pivot; otherwise replace
                    s.LastPivotLow = !s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotLow) ? pp : Math.Min(pp, s.LastPivotLow))
                        : pp;
                    if (s.LastPivotLow != prevLastPivotLow)
                        s.PivotLowTime = bars.OpenTimes[pivotIdx];
                }
            }

            var close     = bars.ClosePrices[tfBarIndex];
            var prevClose = tfBarIndex > 0 ? bars.ClosePrices[tfBarIndex - 1] : close;

            // ── Bullish structure break (BOS or CHoCH bullish) ───────────────
            // Condition: prevClose was at or below the previous pivot high,
            // and close has now crossed above the updated pivot high.
            if (!double.IsNaN(s.LastPivotHigh) && !double.IsNaN(prevLastPivotHigh))
            {
                if (prevClose <= prevLastPivotHigh && close > s.LastPivotHigh)
                {
                    s.CurrentTrend   = true;     // now bullish
                    s.HasTrend       = true;     // warmup complete
                    s.LastBrokenHigh = s.LastPivotHigh;
                    s.LastBrokenLow  = double.NaN;
                }
            }

            // ── Bearish structure break (BOS or CHoCH bearish) ───────────────
            if (!double.IsNaN(s.LastPivotLow) && !double.IsNaN(prevLastPivotLow))
            {
                if (prevClose >= prevLastPivotLow && close < s.LastPivotLow)
                {
                    s.CurrentTrend   = false;    // now bearish
                    s.HasTrend       = true;
                    s.LastBrokenLow  = s.LastPivotLow;
                    s.LastBrokenHigh = double.NaN;
                }
            }

            // s.CurrentTrend and s.HasTrend now reflect this TF bar's state.
            // No cache write needed — CheckMtfFilter reads these fields directly.
        }

        /// <summary>
        /// Maps a chart bar open time to the TF bar index to process up to.
        ///
        /// Non-lower-TF (e.g. 15m filter on a 5m chart):
        ///   A 15m bar is only FULLY CLOSED once the next 15m bar has started,
        ///   i.e. when chartTime >= bar.openTime + tfMinutes.
        ///   Using the current (partially formed) bar would read a partial close,
        ///   which means structure breaks fire at the wrong time (or not at all).
        ///   Fix: subtract one TF period so we always land on the last completed bar.
        ///   E.g. at 5m signal bar 10:05, adjusted = 09:50 → finds 15m bar 09:45
        ///   (fully closed), NOT the still-open 15m bar at 10:00.
        ///
        /// Lower-TF (lookahead_on): unchanged — verbatim from indicator.
        /// </summary>
        private int MtfResolveTfBar(MtfTfState s, DateTime chartTime)
        {
            if (!s.IsLowerTf)
            {
                // Subtract one full TF period to ensure we only land on bars
                // whose close is final (all sub-bars within them have closed).
                var adjusted = chartTime.AddMinutes(-(s.TfMinutes > 0 ? s.TfMinutes : 1));
                return MtfFindAtOrBefore(s.TfBars, adjusted);
            }

            // Lower-TF (lookahead_on): project the first intrabar TF bar
            // that falls within the current chart bar's window.
            var chartBarIndex = MtfFindAtOrBefore(Bars, chartTime);
            if (chartBarIndex < 0) return -1;

            var chartOpen = Bars.OpenTimes[chartBarIndex];
            DateTime chartNextOpen;
            if (chartBarIndex + 1 < Bars.Count)
                chartNextOpen = Bars.OpenTimes[chartBarIndex + 1];
            else
            {
                var m = MtfTfMinutes(Bars.TimeFrame);
                chartNextOpen = chartOpen.AddMinutes(m > 0 ? m : 1);
            }

            var first = MtfFindAtOrAfter(s.TfBars, chartOpen);
            if (first < 0) return MtfFindAtOrBefore(s.TfBars, chartTime);
            if (s.TfBars.OpenTimes[first] >= chartNextOpen)
                return MtfFindAtOrBefore(s.TfBars, chartTime);
            return first;
        }

        /// <summary>
        /// Returns true when the MTF filter passes for the given trade direction.
        /// EnableMtfFilter = false → always returns true (base cBot behaviour).
        /// HasTrend = false (warmup) → that slot always passes.
        /// Uses explicit if-blocks instead of local functions for max compatibility.
        /// </summary>
        private bool CheckMtfFilter(bool isLong, int signalBar)
        {
            if (!EnableMtfFilter) return true;

            var isAnd   = MtfFilterLogic == MtfLogicMode.AND;
            var enabled = 0;
            var passing = 0;

            // Count enabled slots and how many agree with trade direction.
            // null  → slot is disabled → skip.
            // HasTrend = false → warmup, not enough data yet → counts as passing.
            if (_mtfState1 != null)
            {
                enabled++;
                if (!_mtfState1.HasTrend || _mtfState1.CurrentTrend == isLong) passing++;
            }
            if (_mtfState2 != null)
            {
                enabled++;
                if (!_mtfState2.HasTrend || _mtfState2.CurrentTrend == isLong) passing++;
            }
            if (_mtfState3 != null)
            {
                enabled++;
                if (!_mtfState3.HasTrend || _mtfState3.CurrentTrend == isLong) passing++;
            }
            if (_mtfState4 != null)
            {
                enabled++;
                if (!_mtfState4.HasTrend || _mtfState4.CurrentTrend == isLong) passing++;
            }

            // No TF slots enabled → filter passes trivially.
            if (enabled == 0) return true;

            var result = isAnd ? (passing == enabled) : (passing > 0);

            if (!result)
            {
                Print("[MTF BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | TF1:{5}({6}) TF2:{7}({8}) TF3:{9}({10}) TF4:{11}({12})",
                    isLong ? "Long" : "Short", signalBar, passing, enabled,
                    isAnd ? "AND" : "OR",
                    _mtfState1 != null ? (_mtfState1.HasTrend ? (_mtfState1.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState1 != null ? _mtfState1.LastProcessedTfBar : -1,
                    _mtfState2 != null ? (_mtfState2.HasTrend ? (_mtfState2.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState2 != null ? _mtfState2.LastProcessedTfBar : -1,
                    _mtfState3 != null ? (_mtfState3.HasTrend ? (_mtfState3.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState3 != null ? _mtfState3.LastProcessedTfBar : -1,
                    _mtfState4 != null ? (_mtfState4.HasTrend ? (_mtfState4.CurrentTrend ? "Bull" : "Bear") : "Warmup") : "Off",
                    _mtfState4 != null ? _mtfState4.LastProcessedTfBar : -1);
            }

            return result;
        }

        // ── MTF pivot helpers — verbatim from indicator ───────────────────────

        private static bool MtfIsPivotHigh(Bars bars, int idx, int len)
        {
            var left  = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.HighPrices[idx];
            for (var i = left; i <= right; i++)
                if (i != idx && bars.HighPrices[i] >= pivot) return false;
            return true;
        }

        private static bool MtfIsPivotLow(Bars bars, int idx, int len)
        {
            var left  = idx - len;
            var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.LowPrices[idx];
            for (var i = left; i <= right; i++)
                if (i != idx && bars.LowPrices[i] <= pivot) return false;
            return true;
        }

        private static int MtfFindAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1;
            while (lo <= hi)
            {
                var mid    = (lo + hi) / 2;
                var midTime = bars.OpenTimes[mid];
                if      (midTime == time) return mid;
                else if (midTime <  time) lo = mid + 1;
                else                      hi = mid - 1;
            }
            return hi;
        }

        private static int MtfFindAtOrAfter(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1; var ans = -1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                if (bars.OpenTimes[mid] >= time) { ans = mid; hi = mid - 1; }
                else                               lo = mid + 1;
            }
            return ans;
        }

        private static TimeFrame MtfParseTimeFrame(string text)
        {
            switch ((text ?? string.Empty).Trim().ToUpperInvariant())
            {
                case "1":              return TimeFrame.Minute;
                case "2":              return TimeFrame.Minute2;
                case "3":              return TimeFrame.Minute3;
                case "4":              return TimeFrame.Minute4;
                case "5":              return TimeFrame.Minute5;
                case "10":             return TimeFrame.Minute10;
                case "15":             return TimeFrame.Minute15;
                case "30":             return TimeFrame.Minute30;
                case "45":             return TimeFrame.Minute45;
                case "60":  case "1H": return TimeFrame.Hour;
                case "120": case "2H": return TimeFrame.Hour2;
                case "240": case "4H": return TimeFrame.Hour4;
                case "480": case "8H": return TimeFrame.Hour8;
                case "720": case "12H":return TimeFrame.Hour12;
                case "D":   case "1D": return TimeFrame.Daily;
                case "W":   case "1W": return TimeFrame.Weekly;
                case "M":   case "1M": return TimeFrame.Monthly;
                default:               return TimeFrame.Minute15;
            }
        }

        private static int MtfTfMinutes(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute)   return 1;
            if (tf == TimeFrame.Minute2)  return 2;
            if (tf == TimeFrame.Minute3)  return 3;
            if (tf == TimeFrame.Minute4)  return 4;
            if (tf == TimeFrame.Minute5)  return 5;
            if (tf == TimeFrame.Minute10) return 10;
            if (tf == TimeFrame.Minute15) return 15;
            if (tf == TimeFrame.Minute30) return 30;
            if (tf == TimeFrame.Minute45) return 45;
            if (tf == TimeFrame.Hour)     return 60;
            if (tf == TimeFrame.Hour2)    return 120;
            if (tf == TimeFrame.Hour4)    return 240;
            if (tf == TimeFrame.Hour8)    return 480;
            if (tf == TimeFrame.Hour12)   return 720;
            if (tf == TimeFrame.Daily)    return 1440;
            if (tf == TimeFrame.Weekly)   return 10080;
            if (tf == TimeFrame.Monthly)  return 43200;
            return 0;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC Filter check — verbatim from base cBot
        // ════════════════════════════════════════════════════════════════════════

        private bool CheckFilters(int index, int cond)
        {
            var isBull = cond > 0;
            bool? f1 = null;
            bool? f2 = null;

            if (EnableFilter1)
                f1 = HasActiveTouchInLookback(isBull ? _smcInternalBullObs : _smcInternalBearObs,
                                              index, Filter1Lookback, isBull);
            if (EnableFilter2)
                f2 = HasActiveTouchInLookback(isBull ? _smcSwingBullObs : _smcSwingBearObs,
                                              index, Filter2Lookback, isBull);

            if (f1.HasValue && f2.HasValue)
            {
                if (!f1.Value && !f2.Value)
                {
                    Print("[Filter BLOCKED] {0} at bar {1}: no OB touched in last {2}/{3} bars.",
                          isBull ? "Long" : "Short", index, Filter1Lookback, Filter2Lookback);
                    return false;
                }
                return true;
            }
            if (f1.HasValue && !f1.Value)
            {
                Print("[Filter1 BLOCKED] {0} at bar {1}: no internal OB in last {2} bars.",
                      isBull ? "Long" : "Short", index, Filter1Lookback);
                return false;
            }
            if (f2.HasValue && !f2.Value)
            {
                Print("[Filter2 BLOCKED] {0} at bar {1}: no swing OB in last {2} bars.",
                      isBull ? "Long" : "Short", index, Filter2Lookback);
                return false;
            }
            return true;
        }

        private bool HasActiveTouchInLookback(List<SmcObRecord> pool, int signalBar, int lookback, bool bullish)
        {
            if (pool.Count == 0) return false;
            var atr      = double.IsNaN(_smcAtrWilder) ? 0.0 : _smcAtrWilder;
            var closeNow = Bars.ClosePrices[signalBar];

            foreach (var ob in pool)
            {
                if (EnableMinBarsFromOrigin)
                {
                    var minB = ob.Internal ? MinBarsFromOriginInternal : MinBarsFromOriginSwing;
                    if (signalBar - ob.Index < minB) continue;
                }
                if (EnableAtrDistanceFilter && AtrDistanceMultiplier > 0 && atr > 0)
                {
                    var adv = bullish ? closeNow - ob.Top : ob.Bottom - closeNow;
                    if (adv < AtrDistanceMultiplier * atr) continue;
                }
                var lastTouchBar = -1;
                for (var b = ob.StructureBreakIndex + 1; b <= signalBar; b++)
                {
                    if ( bullish && Bars.LowPrices[b]  <= ob.Top)    lastTouchBar = b;
                    if (!bullish && Bars.HighPrices[b] >= ob.Bottom) lastTouchBar = b;
                }
                if (lastTouchBar < 0) continue;
                if (signalBar - lastTouchBar <= lookback) return true;
            }
            return false;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  SMC detection engine — verbatim from base cBot
        // ════════════════════════════════════════════════════════════════════════

        private void RunSmcFilter(int index)
        {
            for (var i = _lastParsedIndex + 1; i <= index; i++) UpdateSmcParsedArrays(i);
            _lastParsedIndex = index;
            if (index < _smcWarmup) return;

            const int iLen = 5;
            var       sLen = Math.Max(5, SmcSwingsLengthInput);

            var iLegNow = ComputeLeg(index, iLen, _internalLeg);
            var iDc     = iLegNow - _internalLeg;
            if (iDc != 0)
            {
                if (iDc == 1)
                { _internalLowLevel  = Bars.LowPrices[index - iLen];  _internalLowIndex  = index - iLen; _internalLowCrossed  = false; }
                else
                { _internalHighLevel = Bars.HighPrices[index - iLen]; _internalHighIndex = index - iLen; _internalHighCrossed = false; }
            }
            _internalLeg = iLegNow;

            var sLegNow = ComputeLeg(index, sLen, _swingLeg);
            var sDc     = sLegNow - _swingLeg;
            if (sDc != 0)
            {
                if (sDc == 1)
                { _lastSwingLow  = Bars.LowPrices[index - sLen];  _lastSwingLowIndex  = index - sLen; _swingLowCrossed  = false; }
                else
                { _lastSwingHigh = Bars.HighPrices[index - sLen]; _lastSwingHighIndex = index - sLen; _swingHighCrossed = false; }
            }
            _swingLeg = sLegNow;

            var close = Bars.ClosePrices[index];
            if (!double.IsNaN(_internalHighLevel) && !_internalHighCrossed && close > _internalHighLevel)
            { _internalHighCrossed = true; _internalTrend =  1; StoreSmcOrderBlock(_internalHighIndex, true,   1, index); }
            if (!double.IsNaN(_internalLowLevel)  && !_internalLowCrossed  && close < _internalLowLevel)
            { _internalLowCrossed  = true; _internalTrend = -1; StoreSmcOrderBlock(_internalLowIndex,  true,  -1, index); }
            if (!double.IsNaN(_lastSwingHigh)      && !_swingHighCrossed    && close > _lastSwingHigh)
            { _swingHighCrossed    = true; _swingTrend    =  1; StoreSmcOrderBlock(_lastSwingHighIndex, false,  1, index); }
            if (!double.IsNaN(_lastSwingLow)       && !_swingLowCrossed     && close < _lastSwingLow)
            { _swingLowCrossed     = true; _swingTrend    = -1; StoreSmcOrderBlock(_lastSwingLowIndex,  false, -1, index); }

            ManageSmcObList(_smcInternalBullObs, index, true);
            ManageSmcObList(_smcInternalBearObs, index, false);
            ManageSmcObList(_smcSwingBullObs,    index, true);
            ManageSmcObList(_smcSwingBearObs,    index, false);
        }

        private void UpdateSmcParsedArrays(int index)
        {
            double tr;
            if (index == 0)
            {
                _smcCumTr = 0; _smcAtrWilderSum = 0; _smcAtrWilder = double.NaN;
                tr = Bars.HighPrices[0] - Bars.LowPrices[0];
            }
            else
            {
                var pc = Bars.ClosePrices[index - 1];
                tr = Math.Max(Bars.HighPrices[index] - Bars.LowPrices[index],
                     Math.Max(Math.Abs(Bars.HighPrices[index] - pc),
                              Math.Abs(Bars.LowPrices[index]  - pc)));
                _smcCumTr += tr;
                if      (index <  SmcAtrPeriod) { _smcAtrWilderSum += tr; _smcAtrWilder = double.NaN; }
                else if (index == SmcAtrPeriod) { _smcAtrWilderSum += tr; _smcAtrWilder = _smcAtrWilderSum / SmcAtrPeriod; }
                else    _smcAtrWilder = (_smcAtrWilder * (SmcAtrPeriod - 1) + tr) / SmcAtrPeriod;
            }
            var vm = SmcOrderBlockFilterInput == ObFilter.Atr
                ? (double.IsNaN(_smcAtrWilder) ? double.MaxValue : _smcAtrWilder)
                : (_smcCumTr / Math.Max(1, index));
            var hv = (Bars.HighPrices[index] - Bars.LowPrices[index]) >= 2.0 * vm;
            _parsedHighs.Add(hv ? Bars.LowPrices[index]  : Bars.HighPrices[index]);
            _parsedLows.Add( hv ? Bars.HighPrices[index] : Bars.LowPrices[index]);
            _times.Add(Bars.OpenTimes[index]);
        }

        private void StoreSmcOrderBlock(int pivotIndex, bool isInternal, int bias, int breakIndex)
        {
            if (pivotIndex < 0 || pivotIndex >= breakIndex || breakIndex >= _parsedHighs.Count) return;
            var parsedIndex = pivotIndex;
            if (bias == -1)
            {
                var maxV = double.MinValue;
                for (var i = pivotIndex; i < breakIndex; i++)
                    if (_parsedHighs[i] > maxV) { maxV = _parsedHighs[i]; parsedIndex = i; }
            }
            else
            {
                var minV = double.MaxValue;
                for (var i = pivotIndex; i < breakIndex; i++)
                    if (_parsedLows[i] < minV) { minV = _parsedLows[i]; parsedIndex = i; }
            }
            var bullish = (bias == 1);
            var ob = new SmcObRecord
            {
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                Internal            = isInternal,
                StructureBreakIndex = breakIndex,
                Time                = _times[parsedIndex]
            };
            _smcObIdCounter++;
            var list = isInternal
                ? (bullish ? _smcInternalBullObs : _smcInternalBearObs)
                : (bullish ? _smcSwingBullObs    : _smcSwingBearObs);
            if (list.Count >= MaxSmcObs) list.RemoveAt(list.Count - 1);
            list.Insert(0, ob);
        }

        private void ManageSmcObList(List<SmcObRecord> list, int index, bool bullish)
        {
            var bearSrc = SmcOrderBlockMitigationInput == MitigationMode.Close
                ? Bars.ClosePrices[index] : Bars.HighPrices[index];
            var bullSrc = SmcOrderBlockMitigationInput == MitigationMode.Close
                ? Bars.ClosePrices[index] : Bars.LowPrices[index];
            for (var i = list.Count - 1; i >= 0; i--)
            {
                var ob = list[i];
                if (( bullish && bullSrc < ob.Bottom)
                ||  (!bullish && bearSrc > ob.Top))
                    list.RemoveAt(i);
            }
        }

        private int ComputeLeg(int index, int size, int previousLeg)
        {
            if (index - size < 1) return previousLeg;
            var refHigh = Bars.HighPrices[index - size];
            var refLow  = Bars.LowPrices[index - size];
            var highest = double.MinValue; var lowest = double.MaxValue;
            for (var i = Math.Max(0, index - size + 1); i <= index; i++)
            {
                if (Bars.HighPrices[i] > highest) highest = Bars.HighPrices[i];
                if (Bars.LowPrices[i]  < lowest)  lowest  = Bars.LowPrices[i];
            }
            if (refHigh > highest) return 0;
            if (refLow  < lowest)  return 1;
            return previousLeg;
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Entry trigger — OB / FVG detection (verbatim from base cBot)
        // ════════════════════════════════════════════════════════════════════════

        private void DetectOrderBlock(int chartIndex, int sourceIndex)
        {
            if (sourceIndex == _lastDetectedObSourceIndex) return;
            var cd  = _sourceBars.ClosePrices[sourceIndex]     > _sourceBars.OpenPrices[sourceIndex]     ? 1 : -1;
            var cdp = _sourceBars.ClosePrices[sourceIndex - 1] > _sourceBars.OpenPrices[sourceIndex - 1] ? 1 : -1;
            bool det = false; bool bull = false; double mx = 0; double mn = 0;
            if (cd ==  1 && cdp == -1 && _sourceBars.HighPrices[sourceIndex] > _sourceBars.HighPrices[sourceIndex - 1])
            { det = true; bull = true;  mx = _sourceBars.HighPrices[sourceIndex - 1]; mn = _sourceBars.LowPrices[sourceIndex - 1]; }
            if (cd == -1 && cdp ==  1 && _sourceBars.LowPrices[sourceIndex]  < _sourceBars.LowPrices[sourceIndex - 1])
            { det = true; bull = false; mx = _sourceBars.HighPrices[sourceIndex - 1]; mn = _sourceBars.LowPrices[sourceIndex - 1]; }
            if (!det) return;
            _obRecords.Insert(0, new EntryObRecord { Max = mx, Min = mn, IsBull = bull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex], DetectionChartIndex = chartIndex });
            _lastDetectedObSourceIndex = sourceIndex;
        }

        private void HandleMitigationOb(int index, double sLow, double sHigh)
        {
            var bullSet = false; var bearSet = false;
            var bullMdf = false; var bearMdf = false;
            var pb = NewEmptyEntrySignal(); var pr = NewEmptyEntrySignal();
            for (var i = _obRecords.Count - 1; i >= 0; i--)
            {
                var r = _obRecords[i]; var now = Bars.OpenTimes[index];
                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        _obRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDist < index && EntryAtrDistancePasses(index, r.Max, true))
                        { pb = NewEntrySignal(index, r.Max, true); bullSet = true; }
                        else bullMdf = true;
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        _obRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDist < index && EntryAtrDistancePasses(index, r.Min, false))
                        { pr = NewEntrySignal(index, r.Min, false); bearSet = true; }
                        else bearMdf = true;
                    }
                }
            }
            if      (bullSet) _signal = pb;
            else if (bearSet) _signal = pr;
            else if (bullMdf &&  _signal.IsBull) _signal = NewEmptyEntrySignal();
            else if (bearMdf && !_signal.IsBull && !double.IsNaN(_signal.Point)) _signal = NewEmptyEntrySignal();
        }

        private void HandleMitigationFvg(int index, double sLow, double sHigh)
        {
            var bullSet = false; var bearSet = false;
            var bullMdf = false; var bearMdf = false;
            var pb = NewEmptyEntrySignal(); var pr = NewEmptyEntrySignal();
            for (var i = _fvgRecords.Count - 1; i >= 0; i--)
            {
                var r = _fvgRecords[i]; var now = Bars.OpenTimes[index];
                if (r.IsBull)
                {
                    if ((sLow <= r.Max || Bars.LowPrices[index] <= r.Max) && r.DetectionTime < now)
                    {
                        _fvgRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDistFvg < index && EntryAtrDistancePasses(index, r.Max, true))
                        { pb = NewEntrySignal(index, r.Max, true); bullSet = true; }
                        else bullMdf = true;
                    }
                }
                else
                {
                    if ((sHigh >= r.Min || Bars.HighPrices[index] >= r.Min) && r.DetectionTime < now)
                    {
                        _fvgRecords.RemoveAt(i);
                        if (r.DetectionChartIndex + MinDistFvg < index && EntryAtrDistancePasses(index, r.Min, false))
                        { pr = NewEntrySignal(index, r.Min, false); bearSet = true; }
                        else bearMdf = true;
                    }
                }
            }
            if      (bullSet) _signalFvg = pb;
            else if (bearSet) _signalFvg = pr;
            else if (bullMdf &&  _signalFvg.IsBull) _signalFvg = NewEmptyEntrySignal();
            else if (bearMdf && !_signalFvg.IsBull && !double.IsNaN(_signalFvg.Point)) _signalFvg = NewEmptyEntrySignal();
        }

        private void DetectFvg(int chartIndex, int sourceIndex)
        {
            if (sourceIndex < 2 || sourceIndex == _lastDetectedFvgSourceIndex) return;
            bool det = false; bool bull = false; double mx = 0; double mn = 0;
            if (_sourceBars.LowPrices[sourceIndex]     > _sourceBars.HighPrices[sourceIndex - 2])
            { det = true; bull = true;  mx = _sourceBars.LowPrices[sourceIndex];     mn = _sourceBars.HighPrices[sourceIndex - 2]; }
            if (_sourceBars.LowPrices[sourceIndex - 2] > _sourceBars.HighPrices[sourceIndex])
            { det = true; bull = false; mx = _sourceBars.LowPrices[sourceIndex - 2]; mn = _sourceBars.HighPrices[sourceIndex]; }
            if (!det) return;
            _fvgRecords.Insert(0, new EntryFvgRecord { Max = mx, Min = mn, IsBull = bull,
                DetectionTime = _sourceBars.OpenTimes[sourceIndex], DetectionChartIndex = chartIndex });
            _lastDetectedFvgSourceIndex = sourceIndex;
        }

        private EntrySignalState NewEntrySignal(int index, double point, bool isBull)
            => new EntrySignalState { Point = point, IsBull = isBull, Entry = false, Index = index };
        private static EntrySignalState NewEmptyEntrySignal()
            => new EntrySignalState { Point = double.NaN, IsBull = false, Entry = false, Index = 0 };

        private bool EntryAtrDistancePasses(int index, double level, bool isBull)
        {
            if (!EnableAtrDistanceFilter || AtrDistanceMultiplier <= 0 || double.IsNaN(_smcAtrWilder)) return true;
            var adv = isBull ? Bars.ClosePrices[index] - level : level - Bars.ClosePrices[index];
            return adv >= AtrDistanceMultiplier * _smcAtrWilder;
        }

        private void EnsureHeikinAshiSource(int sourceIndex)
        {
            while (_haSourceClose.Count <= sourceIndex)
            {
                var i = _haSourceClose.Count;
                var c = (_sourceBars.OpenPrices[i] + _sourceBars.HighPrices[i]
                       + _sourceBars.LowPrices[i]  + _sourceBars.ClosePrices[i]) / 4.0;
                var o = i == 0
                    ? (_sourceBars.OpenPrices[i] + _sourceBars.ClosePrices[i]) / 2.0
                    : (_haSourceOpen[i - 1] + _haSourceClose[i - 1]) / 2.0;
                _haSourceOpen.Add(o); _haSourceClose.Add(c);
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        //  Trade execution — verbatim from base cBot
        // ════════════════════════════════════════════════════════════════════════

        private void ExecuteSignalTrade(int signalBarIndex, int direction)
        {
            var openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            { Print("Skipped: {0} open at cap={1}.", openCount, MaxOpenPositions); return; }

            var tradeType = direction > 0 ? TradeType.Buy : TradeType.Sell;
            var slPrice   = direction > 0
                ? GetLowestLowBeforeSignal(signalBarIndex, StopLossLookbackBars)
                : GetHighestHighBeforeSignal(signalBarIndex, StopLossLookbackBars);
            if (double.IsNaN(slPrice)) return;

            var entry  = direction > 0 ? Symbol.Ask : Symbol.Bid;
            var slPips = direction > 0
                ? (entry - slPrice) / Symbol.PipSize
                : (slPrice - entry) / Symbol.PipSize;
            if (slPips <= 0)
            { Print("Skipped: invalid SL. dir={0} entry={1} sl={2}", direction, entry, slPrice); return; }

            slPips += StopLossBufferPips;

            var raw = Account.Balance * (RiskPercent / 100.0) / (slPips * Symbol.PipValue);
            var vol = Symbol.NormalizeVolumeInUnits(raw, RoundingMode.Down);
            if (vol < Symbol.VolumeInUnitsMin) vol = Symbol.VolumeInUnitsMin;
            if (vol > Symbol.VolumeInUnitsMax) vol = Symbol.VolumeInUnitsMax;

            var tpPips = slPips * RiskRewardRatio;
            if (tpPips <= 0) { Print("Skipped: invalid TP."); return; }

            ExecuteMarketOrder(tradeType, SymbolName, vol, InstanceName, slPips, tpPips);
        }

        private double GetLowestLowBeforeSignal(int bar, int lb)
        {
            var from = Math.Max(0, bar - lb + 1);
            if (bar < from) return double.NaN;
            var v = double.MaxValue;
            for (var i = from; i <= bar; i++) v = Math.Min(v, Bars.LowPrices[i]);
            return v;
        }

        private double GetHighestHighBeforeSignal(int bar, int lb)
        {
            var from = Math.Max(0, bar - lb + 1);
            if (bar < from) return double.NaN;
            var v = double.MinValue;
            for (var i = from; i <= bar; i++) v = Math.Max(v, Bars.HighPrices[i]);
            return v;
        }

        private static int FindBarIndexAtOrBefore(Bars bars, DateTime t)
        {
            var lo = 0; var hi = bars.OpenTimes.Count - 1; var ans = -1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
                if (bars.OpenTimes[mid] <= t) { ans = mid; lo = mid + 1; }
                else                           hi = mid - 1;
            }
            return ans;
        }
    }
}
