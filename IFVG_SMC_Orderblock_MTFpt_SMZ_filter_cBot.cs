// =============================================================================
// IFVG_SMC_Orderblock_MTFpt_SMZ_filter_cBot
// =============================================================================
// Base    : IFVG_SMC_Orderblock_MTFpt_filter_cBot (all logic preserved verbatim)
// Added   : SMZ Trend Filter — replicates the MTF Trend Panel from the
//           "Smart Money Zones (FVG + OB) + MTF Trend Panel" indicator.
//
// Filter logic (per enabled timeframe):
//   Bullish = latest close > SMA(SmzMaPeriod) on that timeframe  → 🟢
//   Bearish = latest close < SMA(SmzMaPeriod) on that timeframe  → 🔴
//
//   Long  allowed only when enabled TFs satisfy the SMZ trend condition (bullish).
//   Short allowed only when enabled TFs satisfy the SMZ trend condition (bearish).
//
//   OR  logic → at least one enabled TF must agree with signal direction.
//   AND logic → all enabled TFs must agree with signal direction.
//
//   During SMA warmup (fewer bars than SmzMaPeriod) a TF is treated as passing
//   so early-history bars are not silently blocked.
//
// When EnableSmzFilter = false the cBot behaves identically to
// IFVG_SMC_Orderblock_MTFpt_filter_cBot.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;
using cAlgo.API.Indicators;
using cAlgo.API.Internals;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class IFVG_SMC_Orderblock_MTFpt_SMZ_filter_cBot : Robot
    {
        // ─────────────────────────────────────────────────────────────────────
        //  Enums  (base cBot enums + new SmzLogicMode)
        // ─────────────────────────────────────────────────────────────────────

        public enum ObFilter       { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }
        public enum MtfLogicMode   { OR, AND }
        public enum SmzLogicMode   { OR, AND }   // ← NEW: SMZ filter vote aggregation

        // ─────────────────────────────────────────────────────────────────────
        //  Inner types  (verbatim from base cBot)
        // ─────────────────────────────────────────────────────────────────────

        private sealed class BslPivot
        {
            public double Price;
            public int    BarIndex;
            public int    Type;
        }

        private sealed class BslPool
        {
            public double Price;
            public int    PivotIndex;
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

        private sealed class MtfTfState
        {
            public Bars     TfBars;
            public int      PivotLen;
            public bool     IsLowerTf;
            public int      TfMinutes;
            public int      LastProcessedTfBar = -1;
            public bool     CurrentTrend;
            public bool     HasTrend;
            public double   LastPivotHigh    = double.NaN;
            public double   LastPivotLow     = double.NaN;
            public double   LastBrokenHigh   = double.NaN;
            public double   LastBrokenLow    = double.NaN;
            public DateTime PivotHighTime    = DateTime.MinValue;
            public DateTime PivotLowTime     = DateTime.MinValue;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — IFVG  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("FVG Search Lookback (bars)", DefaultValue = 15, MinValue = 1, Group = "IFVG")]
        public int FvgGapBars { get; set; }

        [Parameter("Min FVG Size (pips)", DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG")]
        public double MinFvgPips { get; set; }

        [Parameter("FVG Epsilon (price units)", DefaultValue = 0.0, MinValue = 0.0, Group = "IFVG")]
        public double FvgEpsilonPoints { get; set; }

        [Parameter("MA Period", DefaultValue = 21, MinValue = 1, Group = "IFVG")]
        public int MaPeriod { get; set; }

        [Parameter("MA Type (SMA / EMA)", DefaultValue = "EMA", Group = "IFVG")]
        public string MaType { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — BSL & SSL  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left",  DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Risk Management  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk % per trade",         DefaultValue = 1.0,   MinValue = 0.1,  MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio",        DefaultValue = 2.0,   MinValue = 0.1,  Step = 0.1,       Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Open Positions",        DefaultValue = 3,     MinValue = 1,    MaxValue = 100,   Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)",    DefaultValue = 3.0,   MinValue = 0.1,                   Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)",    DefaultValue = 500.0, MinValue = 1.0,                   Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        // Buffer added ON TOP of the raw SSL/BSL-derived SL distance.
        // Set to 0 to disable (original behaviour).
        [Parameter("SL Buffer (pips)",          DefaultValue = 0.0,   MinValue = 0.0, Step = 0.1,       Group = "Risk Management")]
        public double SlBufferPips { get; set; }

        [Parameter("Instance Name",             DefaultValue = "IFVG_SMC_MTFpt_SMZ_cBot",               Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — SMC Filter  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "SMC Filter — Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }

        [Parameter("Order Block Filter",   DefaultValue = ObFilter.Atr,        Group = "SMC Filter — Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter — Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter — Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }

        [Parameter("Enable Filter 1 (Internal OB)",          DefaultValue = true,  Group = "Filter 1 — Internal OB")]
        public bool EnableFilter1 { get; set; }

        [Parameter("OB Touch Window — Internal (bars)",      DefaultValue = 10, MinValue = 0, Group = "Filter 1 — Internal OB")]
        public int Filter1Lookback { get; set; }

        [Parameter("Enable Filter 2 (Swing OB)",             DefaultValue = false, Group = "Filter 2 — Swing OB")]
        public bool EnableFilter2 { get; set; }

        [Parameter("OB Touch Window — Swing (bars)",         DefaultValue = 10, MinValue = 0, Group = "Filter 2 — Swing OB")]
        public int Filter2Lookback { get; set; }

        [Parameter("Enable Min Bars From OB Origin",         DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableMinBarsFromOrigin { get; set; }

        [Parameter("Min Bars — Internal OB",                 DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginInternal { get; set; }

        [Parameter("Min Bars — Swing OB",                    DefaultValue = 5, MinValue = 1, Group = "OB Quality Filters")]
        public int MinBarsFromOriginSwing { get; set; }

        [Parameter("Enable ATR Distance Filter",             DefaultValue = false, Group = "OB Quality Filters")]
        public bool EnableAtrDistanceFilter { get; set; }

        [Parameter("ATR Distance Multiplier",                DefaultValue = 1.0, MinValue = 0.1, Step = 0.1, Group = "OB Quality Filters")]
        public double AtrDistanceMultiplier { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — MTF Trend Filter  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable MTF Trend Filter",   DefaultValue = false,             Group = "MTF Trend Filter — General")]
        public bool EnableMtfFilter { get; set; }

        [Parameter("Multi-TF Logic (OR / AND)", DefaultValue = MtfLogicMode.OR,  Group = "MTF Trend Filter — General")]
        public MtfLogicMode MtfFilterLogic { get; set; }

        [Parameter("Enable TF1 Filter",  DefaultValue = true,  Group = "MTF Trend Filter — TF1")]
        public bool EnableMtfTf1 { get; set; }
        [Parameter("TF1 Timeframe",      DefaultValue = "15",  Group = "MTF Trend Filter — TF1")]
        public string MtfTimeframe1 { get; set; }
        [Parameter("TF1 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF1")]
        public int MtfPivotStrength1 { get; set; }
        [Parameter("TF1 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF1")]
        public bool MtfIsLowerTf1 { get; set; }

        [Parameter("Enable TF2 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool EnableMtfTf2 { get; set; }
        [Parameter("TF2 Timeframe",      DefaultValue = "30",  Group = "MTF Trend Filter — TF2")]
        public string MtfTimeframe2 { get; set; }
        [Parameter("TF2 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF2")]
        public int MtfPivotStrength2 { get; set; }
        [Parameter("TF2 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF2")]
        public bool MtfIsLowerTf2 { get; set; }

        [Parameter("Enable TF3 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool EnableMtfTf3 { get; set; }
        [Parameter("TF3 Timeframe",      DefaultValue = "60",  Group = "MTF Trend Filter — TF3")]
        public string MtfTimeframe3 { get; set; }
        [Parameter("TF3 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF3")]
        public int MtfPivotStrength3 { get; set; }
        [Parameter("TF3 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF3")]
        public bool MtfIsLowerTf3 { get; set; }

        [Parameter("Enable TF4 Filter",  DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool EnableMtfTf4 { get; set; }
        [Parameter("TF4 Timeframe",      DefaultValue = "240", Group = "MTF Trend Filter — TF4")]
        public string MtfTimeframe4 { get; set; }
        [Parameter("TF4 Pivot Strength", DefaultValue = 15, MinValue = 1, Group = "MTF Trend Filter — TF4")]
        public int MtfPivotStrength4 { get; set; }
        [Parameter("TF4 Lower than chart?", DefaultValue = false, Group = "MTF Trend Filter — TF4")]
        public bool MtfIsLowerTf4 { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — SMZ Trend Filter  (NEW)
        //
        //  Replicates the MTF Trend Panel from the
        //  "Smart Money Zones (FVG + OB) + MTF Trend Panel" indicator.
        //  Trend on each TF = latest close vs SMA(SmzMaPeriod).
        //  When EnableSmzFilter = false this entire section is bypassed.
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable SMZ Trend Filter",       DefaultValue = false,           Group = "SMZ Trend Filter — General")]
        public bool EnableSmzFilter { get; set; }

        [Parameter("SMZ MA Period",                 DefaultValue = 50, MinValue = 1, Group = "SMZ Trend Filter — General")]
        public int SmzMaPeriod { get; set; }

        [Parameter("SMZ Logic (OR / AND)",          DefaultValue = SmzLogicMode.OR, Group = "SMZ Trend Filter — General")]
        public SmzLogicMode SmzFilterLogic { get; set; }

        // Per-TF toggles — matching exactly the 7 rows shown in the SMZ panel
        [Parameter("Enable 1m  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1m  { get; set; }

        [Parameter("Enable 5m  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable5m  { get; set; }

        [Parameter("Enable 15m TF",  DefaultValue = true,  Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable15m { get; set; }

        [Parameter("Enable 30m TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable30m { get; set; }

        [Parameter("Enable 1H  TF",  DefaultValue = true,  Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1h  { get; set; }

        [Parameter("Enable 4H  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable4h  { get; set; }

        [Parameter("Enable 1D  TF",  DefaultValue = false, Group = "SMZ Trend Filter — Timeframes")]
        public bool SmzEnable1d  { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Combined TF Filter  (NEW)
        //
        //  A two-condition AND gate, one condition per timeframe:
        //
        //    Condition A (1H) : (SMZ SMA 1H bullish/bearish)
        //                    OR (MTF pivot trend 1H is Bull CHoCH/BOS)
        //
        //    Condition B (15m): (SMZ SMA 15m bullish/bearish)
        //                    OR (MTF pivot trend 15m is Bull CHoCH/BOS)
        //
        //  Both enabled conditions must pass simultaneously (AND).
        //  For a Long  trade "bullish" means close > SMA and CurrentTrend=true.
        //  For a Short trade the reverse applies automatically.
        //
        //  When EnableCombinedTfFilter = false the block is bypassed entirely.
        //  Disabling individual conditions (Enable1hCondition / Enable15mCondition)
        //  removes that leg from the AND gate.
        //  Disabling a component toggle (UseCmbSmz1h / UseCmbMtf1h etc.) removes
        //  that OR branch from its condition; if BOTH branches for a condition are
        //  disabled, that condition always passes.
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable Combined TF Filter",  DefaultValue = false, Group = "Combined TF Filter — General")]
        public bool EnableCombinedTfFilter { get; set; }

        // ── 1H condition ──────────────────────────────────────────────────────
        [Parameter("Enable 1H Condition",        DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool CmbEnable1h { get; set; }

        [Parameter("Use SMZ SMA (1H)",           DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool UseCmbSmz1h { get; set; }

        [Parameter("Use MTF Pivot Trend (1H)",   DefaultValue = true,  Group = "Combined TF Filter — 1H")]
        public bool UseCmbMtf1h { get; set; }

        [Parameter("1H MTF Pivot Strength",      DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter — 1H")]
        public int CmbMtfPivotStrength1h { get; set; }

        [Parameter("1H MTF Lower than chart?",   DefaultValue = false, Group = "Combined TF Filter — 1H")]
        public bool CmbMtfIsLowerTf1h { get; set; }

        // ── 15m condition ─────────────────────────────────────────────────────
        [Parameter("Enable 15m Condition",       DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool CmbEnable15m { get; set; }

        [Parameter("Use SMZ SMA (15m)",          DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool UseCmbSmz15m { get; set; }

        [Parameter("Use MTF Pivot Trend (15m)",  DefaultValue = true,  Group = "Combined TF Filter — 15m")]
        public bool UseCmbMtf15m { get; set; }

        [Parameter("15m MTF Pivot Strength",     DefaultValue = 15, MinValue = 1, Group = "Combined TF Filter — 15m")]
        public int CmbMtfPivotStrength15m { get; set; }

        [Parameter("15m MTF Lower than chart?",  DefaultValue = false, Group = "Combined TF Filter — 15m")]
        public bool CmbMtfIsLowerTf15m { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  FIELDS — base cBot  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        private const int MaxBslPivots = 10;

        private IndicatorDataSeries      _maSeries;
        private SimpleMovingAverage      _sma;
        private ExponentialMovingAverage _ema;

        private readonly LinkedList<BslPivot> _bslPivots         = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools   = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools  = new LinkedList<BslPool>();
        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        private readonly List<SmcObRecord> _smcInternalBullObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcInternalBearObs = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBullObs    = new List<SmcObRecord>();
        private readonly List<SmcObRecord> _smcSwingBearObs    = new List<SmcObRecord>();
        private readonly List<double>      _parsedHighs        = new List<double>();
        private readonly List<double>      _parsedLows         = new List<double>();
        private readonly List<DateTime>    _times              = new List<DateTime>();

        private const int SmcAtrPeriod    = 200;
        private double    _smcAtrWilder    = double.NaN;
        private double    _smcAtrWilderSum = 0.0;
        private double    _smcCumTr        = 0.0;

        private int    _swingLeg;
        private int    _swingTrend;
        private double _lastSwingHigh      = double.NaN;
        private double _lastSwingLow       = double.NaN;
        private int    _lastSwingHighIndex = -1;
        private int    _lastSwingLowIndex  = -1;
        private bool   _swingHighCrossed;
        private bool   _swingLowCrossed;

        private int    _internalLeg;
        private int    _internalTrend;
        private double _internalHighLevel   = double.NaN;
        private double _internalLowLevel    = double.NaN;
        private int    _internalHighIndex   = -1;
        private int    _internalLowIndex    = -1;
        private bool   _internalHighCrossed;
        private bool   _internalLowCrossed;

        private int       _smcObIdCounter;
        private const int MaxSmcObs       = 500;
        private int       _lastParsedIndex = -1;
        private int       _smcWarmup;

        private MtfTfState _mtfState1;
        private MtfTfState _mtfState2;
        private MtfTfState _mtfState3;
        private MtfTfState _mtfState4;

        private int _lastProcessed = -1;

        // ─────────────────────────────────────────────────────────────────────
        //  FIELDS — SMZ Trend Filter  (NEW)
        //  One Bars + SMA pair per SMZ panel timeframe.
        //  Null when the TF is disabled or the filter is off.
        // ─────────────────────────────────────────────────────────────────────

        private Bars _smz1mBars,  _smz5mBars,  _smz15mBars, _smz30mBars;
        private Bars _smz1hBars,  _smz4hBars,  _smz1dBars;

        private SimpleMovingAverage _smz1mSma,  _smz5mSma,  _smz15mSma, _smz30mSma;
        private SimpleMovingAverage _smz1hSma,  _smz4hSma,  _smz1dSma;

        // ─────────────────────────────────────────────────────────────────────
        //  FIELDS — Combined TF Filter  (NEW)
        //  Dedicated Bars+SMA for the 1H and 15m SMZ components, independent of
        //  the general SMZ filter so the combined filter works even when
        //  EnableSmzFilter is off.
        // ─────────────────────────────────────────────────────────────────────

        private Bars _cmbSmz1hBars,  _cmbSmz15mBars;
        private SimpleMovingAverage  _cmbSmz1hSma,  _cmbSmz15mSma;
        private MtfTfState           _cmbMtf1h,     _cmbMtf15m;

        // ═════════════════════════════════════════════════════════════════════
        //  LIFECYCLE
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            _smcWarmup = Math.Max(SmcSwingsLengthInput, 5) + 5;

            _maSeries = CreateDataSeries();
            _sma = Indicators.SimpleMovingAverage(Bars.ClosePrices, MaPeriod);
            _ema = Indicators.ExponentialMovingAverage(Bars.ClosePrices, MaPeriod);

            // ── MTF pivot-trend filter (verbatim) ────────────────────────────
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

            // ── SMZ SMA-trend filter (NEW) ───────────────────────────────────
            if (EnableSmzFilter)
            {
                SmzInitTf(SmzEnable1m,  TimeFrame.Minute,   ref _smz1mBars,  ref _smz1mSma);
                SmzInitTf(SmzEnable5m,  TimeFrame.Minute5,  ref _smz5mBars,  ref _smz5mSma);
                SmzInitTf(SmzEnable15m, TimeFrame.Minute15, ref _smz15mBars, ref _smz15mSma);
                SmzInitTf(SmzEnable30m, TimeFrame.Minute30, ref _smz30mBars, ref _smz30mSma);
                SmzInitTf(SmzEnable1h,  TimeFrame.Hour,     ref _smz1hBars,  ref _smz1hSma);
                SmzInitTf(SmzEnable4h,  TimeFrame.Hour4,    ref _smz4hBars,  ref _smz4hSma);
                SmzInitTf(SmzEnable1d,  TimeFrame.Daily,    ref _smz1dBars,  ref _smz1dSma);

                Print("SMZ filter ON. Logic={0}. MA={1}. TFs: 1m={2} 5m={3} 15m={4} 30m={5} 1h={6} 4h={7} 1d={8}",
                    SmzFilterLogic, SmzMaPeriod,
                    SmzEnable1m, SmzEnable5m, SmzEnable15m, SmzEnable30m,
                    SmzEnable1h, SmzEnable4h, SmzEnable1d);
            }

            Print("IFVG_SMC_Orderblock_MTFpt_SMZ_filter_cBot started. MaxPositions={0}, Risk={1}%, RR={2}",
                  MaxOpenPositions, RiskPercent, RiskRewardRatio);

            // ── Combined TF Filter (NEW) ─────────────────────────────────────
            if (EnableCombinedTfFilter)
            {
                // 1H SMZ SMA component
                if (CmbEnable1h && UseCmbSmz1h)
                {
                    _cmbSmz1hBars = MarketData.GetBars(TimeFrame.Hour);
                    _cmbSmz1hSma  = Indicators.SimpleMovingAverage(_cmbSmz1hBars.ClosePrices, SmzMaPeriod);
                }
                // 1H MTF pivot component
                if (CmbEnable1h && UseCmbMtf1h)
                    _cmbMtf1h = MtfCreateStateFixed(TimeFrame.Hour, CmbMtfPivotStrength1h, CmbMtfIsLowerTf1h);

                // 15m SMZ SMA component
                if (CmbEnable15m && UseCmbSmz15m)
                {
                    _cmbSmz15mBars = MarketData.GetBars(TimeFrame.Minute15);
                    _cmbSmz15mSma  = Indicators.SimpleMovingAverage(_cmbSmz15mBars.ClosePrices, SmzMaPeriod);
                }
                // 15m MTF pivot component
                if (CmbEnable15m && UseCmbMtf15m)
                    _cmbMtf15m = MtfCreateStateFixed(TimeFrame.Minute15, CmbMtfPivotStrength15m, CmbMtfIsLowerTf15m);

                Print("Combined TF filter ON. 1H={0}(SmzSMA={1},MtfPivot={2}) 15m={3}(SmzSMA={4},MtfPivot={5})",
                    CmbEnable1h, UseCmbSmz1h, UseCmbMtf1h,
                    CmbEnable15m, UseCmbSmz15m, UseCmbMtf15m);
            }

            if (SlBufferPips > 0)
                Print("SL Buffer = {0} pips", SlBufferPips);
        }

        protected override void OnStop()
        {
            Print("IFVG_SMC_Orderblock_MTFpt_SMZ_filter_cBot stopped.");
        }

        // ─────────────────────────────────────────────────────────────────────

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;

            // Step 1 — build BSL/SSL and SMC pools bar-by-bar (verbatim)
            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);
                RunSmcFilter(i);
            }
            _lastProcessed = signalBar;

            // Step 2 — advance MTF pivot-trend states (verbatim)
            if (EnableMtfFilter)
            {
                var chartTime = Bars.OpenTimes[signalBar];
                MtfAdvanceState(_mtfState1, chartTime);
                MtfAdvanceState(_mtfState2, chartTime);
                MtfAdvanceState(_mtfState3, chartTime);
                MtfAdvanceState(_mtfState4, chartTime);
            }

            // Step 2b — advance combined-filter MTF pivot states (NEW)
            if (EnableCombinedTfFilter)
            {
                var chartTime = Bars.OpenTimes[signalBar];
                MtfAdvanceState(_cmbMtf1h,  chartTime);
                MtfAdvanceState(_cmbMtf15m, chartTime);
            }

            if (signalBar < Math.Max(PivotLeft + PivotRight + 1, 3)) return;

            // Step 3 — IFVG signal detection (verbatim)
            var maValue   = CalculateMa(signalBar);
            var signalDir = DetectIfvgSignal(signalBar, maValue);
            if (signalDir == 0) return;

            // Step 4 — position cap (verbatim)
            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached — IFVG signal skipped.", signalBar, MaxOpenPositions);
                return;
            }

            // Step 5 — SMC OB filter (verbatim)
            if (!CheckFilters(signalBar, signalDir)) return;

            // Step 6 — MTF pivot-trend filter (verbatim)
            if (!CheckMtfFilter(signalDir > 0, signalBar)) return;

            // Step 7 — SMZ SMA-trend filter (NEW)
            if (!CheckSmzFilter(signalDir > 0, signalBar)) return;

            // Step 8 — Combined TF filter: (SMZ-1H OR MTF-1H) AND (SMZ-15m OR MTF-15m)
            if (!CheckCombinedTfFilter(signalDir > 0, signalBar)) return;

            // Step 9 — execute trade
            if (signalDir == 1) TryEnterLong(signalBar);
            else                TryEnterShort(signalBar);
        }

        // ═════════════════════════════════════════════════════════════════════
        //  SMZ TREND FILTER  (NEW)
        // ═════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Initialise one SMZ timeframe: subscribe to its bar series and
        /// attach a SimpleMovingAverage indicator with SmzMaPeriod.
        /// No-op when the TF toggle is off.
        /// </summary>
        private void SmzInitTf(bool enabled, TimeFrame tf,
                                ref Bars barsField, ref SimpleMovingAverage smaField)
        {
            if (!enabled) return;
            barsField = MarketData.GetBars(tf);
            smaField  = Indicators.SimpleMovingAverage(barsField.ClosePrices, SmzMaPeriod);
        }

        /// <summary>
        /// Returns true when the SMZ trend conditions allow a trade in
        /// the given direction, or when the filter is disabled.
        ///
        /// Logic mirrors GetTrend() from "Smart Money Zones (FVG + OB) + MTF
        /// Trend Panel.cs":
        ///   bullish = bars.ClosePrices[last] > sma.Result[last]
        ///   bearish = bars.ClosePrices[last] &lt; sma.Result[last]
        ///
        /// During SMA warmup the TF vote is counted as passing so early bars
        /// are not silently blocked.
        /// </summary>
        private bool CheckSmzFilter(bool isLong, int signalBar)
        {
            if (!EnableSmzFilter) return true;

            var isAnd   = SmzFilterLogic == SmzLogicMode.AND;
            var enabled = 0;
            var passing = 0;

            SmzCheckTf("1m",  _smz1mBars,  _smz1mSma,  SmzEnable1m,  isLong, ref enabled, ref passing);
            SmzCheckTf("5m",  _smz5mBars,  _smz5mSma,  SmzEnable5m,  isLong, ref enabled, ref passing);
            SmzCheckTf("15m", _smz15mBars, _smz15mSma, SmzEnable15m, isLong, ref enabled, ref passing);
            SmzCheckTf("30m", _smz30mBars, _smz30mSma, SmzEnable30m, isLong, ref enabled, ref passing);
            SmzCheckTf("1H",  _smz1hBars,  _smz1hSma,  SmzEnable1h,  isLong, ref enabled, ref passing);
            SmzCheckTf("4H",  _smz4hBars,  _smz4hSma,  SmzEnable4h,  isLong, ref enabled, ref passing);
            SmzCheckTf("1D",  _smz1dBars,  _smz1dSma,  SmzEnable1d,  isLong, ref enabled, ref passing);

            if (enabled == 0) return true;   // no TFs enabled → pass through

            var result = isAnd ? (passing == enabled) : (passing > 0);

            if (!result)
            {
                // Build per-TF status string for the log
                var sb = new System.Text.StringBuilder();
                SmzAppendStatus(sb, "1m",  _smz1mBars,  _smz1mSma,  SmzEnable1m);
                SmzAppendStatus(sb, "5m",  _smz5mBars,  _smz5mSma,  SmzEnable5m);
                SmzAppendStatus(sb, "15m", _smz15mBars, _smz15mSma, SmzEnable15m);
                SmzAppendStatus(sb, "30m", _smz30mBars, _smz30mSma, SmzEnable30m);
                SmzAppendStatus(sb, "1H",  _smz1hBars,  _smz1hSma,  SmzEnable1h);
                SmzAppendStatus(sb, "4H",  _smz4hBars,  _smz4hSma,  SmzEnable4h);
                SmzAppendStatus(sb, "1D",  _smz1dBars,  _smz1dSma,  SmzEnable1d);

                Print("[SMZ BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | {5}",
                    isLong ? "Long" : "Short", signalBar,
                    passing, enabled, isAnd ? "AND" : "OR", sb);
            }

            return result;
        }

        /// <summary>
        /// Casts one SMZ TF vote.
        /// </summary>
        private void SmzCheckTf(string label, Bars bars, SimpleMovingAverage sma,
                                 bool tfEnabled, bool isLong,
                                 ref int enabledCount, ref int passingCount)
        {
            if (!tfEnabled || bars == null || sma == null) return;
            enabledCount++;

            var idx = bars.Count - 1;
            if (idx < SmzMaPeriod - 1)
            {
                // Warmup — treat as passing so early history is not blocked
                passingCount++;
                return;
            }

            // Mirror GetTrend() from the SMZ indicator
            var close    = bars.ClosePrices[idx];
            var smaValue = sma.Result[idx];

            if (double.IsNaN(smaValue))
            {
                passingCount++;   // SMA not yet ready → pass
                return;
            }

            var isBullish = close > smaValue;
            if (isLong == isBullish) passingCount++;
        }

        /// <summary>Helper to append TF status to the blocked-trade log line.</summary>
        private void SmzAppendStatus(System.Text.StringBuilder sb,
                                     string label, Bars bars, SimpleMovingAverage sma,
                                     bool tfEnabled)
        {
            if (!tfEnabled) return;
            if (bars == null || sma == null) { sb.Append(label).Append(":Off "); return; }

            var idx = bars.Count - 1;
            if (idx < SmzMaPeriod - 1) { sb.Append(label).Append(":Warm "); return; }

            var smaVal = sma.Result[idx];
            if (double.IsNaN(smaVal)) { sb.Append(label).Append(":NaN "); return; }

            sb.Append(label)
              .Append(bars.ClosePrices[idx] > smaVal ? ":Bull " : ":Bear ");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  COMBINED TF FILTER  (NEW)
        //  (SMZ SMA 1H bullish  OR  MTF pivot 1H Bull)
        //  AND
        //  (SMZ SMA 15m bullish OR  MTF pivot 15m Bull)
        //  Inverted automatically for Short signals.
        // ═════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Evaluates the combined TF filter.
        /// Each enabled condition (1H, 15m) is an OR of its active components.
        /// Enabled conditions are combined with AND.
        /// Returns true (pass) when the filter is off or all conditions pass.
        /// </summary>
        private bool CheckCombinedTfFilter(bool isLong, int signalBar)
        {
            if (!EnableCombinedTfFilter) return true;

            var pass1h  = !CmbEnable1h  || CheckCombinedTfCondition(isLong,
                              "1H",
                              UseCmbSmz1h,  _cmbSmz1hBars,  _cmbSmz1hSma,
                              UseCmbMtf1h,  _cmbMtf1h);

            var pass15m = !CmbEnable15m || CheckCombinedTfCondition(isLong,
                              "15m",
                              UseCmbSmz15m, _cmbSmz15mBars, _cmbSmz15mSma,
                              UseCmbMtf15m, _cmbMtf15m);

            var result = pass1h && pass15m;

            if (!result)
            {
                var sb = new System.Text.StringBuilder();
                if (CmbEnable1h)
                {
                    sb.Append("1H:");
                    CmbAppendConditionStatus(sb, isLong, UseCmbSmz1h, _cmbSmz1hBars, _cmbSmz1hSma,
                                                          UseCmbMtf1h, _cmbMtf1h);
                    sb.Append(' ');
                }
                if (CmbEnable15m)
                {
                    sb.Append("15m:");
                    CmbAppendConditionStatus(sb, isLong, UseCmbSmz15m, _cmbSmz15mBars, _cmbSmz15mSma,
                                                          UseCmbMtf15m, _cmbMtf15m);
                }
                Print("[CMB BLOCKED] {0} bar={1} 1H={2} 15m={3} | {4}",
                    isLong ? "Long" : "Short", signalBar,
                    pass1h ? "PASS" : "FAIL", pass15m ? "PASS" : "FAIL", sb);
            }

            return result;
        }

        /// <summary>
        /// Evaluates one condition leg: (useSmz && smzBullish) OR (useMtf && mtfBullish).
        /// If neither component is active the condition always passes.
        /// </summary>
        private bool CheckCombinedTfCondition(
            bool isLong, string label,
            bool useSmz, Bars smzBars, SimpleMovingAverage smzSma,
            bool useMtf, MtfTfState mtfState)
        {
            // No active components → condition vacuously passes
            if (!useSmz && !useMtf) return true;

            // SMZ SMA branch
            bool smzPass = false;
            if (useSmz && smzBars != null && smzSma != null)
            {
                var idx = smzBars.Count - 1;
                if (idx < SmzMaPeriod - 1)
                {
                    smzPass = true;   // warmup → pass
                }
                else
                {
                    var smaVal = smzSma.Result[idx];
                    if (double.IsNaN(smaVal))
                        smzPass = true;
                    else
                    {
                        var isBull = smzBars.ClosePrices[idx] > smaVal;
                        smzPass = (isLong == isBull);
                    }
                }
            }

            // MTF pivot branch — CurrentTrend=true means Bull CHoCH/BOS fired
            bool mtfPass = false;
            if (useMtf && mtfState != null)
            {
                // Warmup (no trend detected yet) → pass through
                if (!mtfState.HasTrend)
                    mtfPass = true;
                else
                    mtfPass = (mtfState.CurrentTrend == isLong);
            }

            return smzPass || mtfPass;
        }

        /// <summary>Appends a short diagnostic string for one combined-filter condition.</summary>
        private void CmbAppendConditionStatus(
            System.Text.StringBuilder sb,
            bool isLong,
            bool useSmz, Bars smzBars, SimpleMovingAverage smzSma,
            bool useMtf, MtfTfState mtfState)
        {
            if (useSmz)
            {
                sb.Append("SMZ=");
                if (smzBars == null || smzSma == null) { sb.Append("Off"); }
                else
                {
                    var idx = smzBars.Count - 1;
                    if (idx < SmzMaPeriod - 1) sb.Append("Warm");
                    else
                    {
                        var v = smzSma.Result[idx];
                        sb.Append(double.IsNaN(v) ? "NaN" : (smzBars.ClosePrices[idx] > v ? "Bull" : "Bear"));
                    }
                }
                sb.Append(' ');
            }
            if (useMtf)
            {
                sb.Append("MTF=");
                if (mtfState == null)    sb.Append("Off");
                else if (!mtfState.HasTrend) sb.Append("Warm");
                else sb.Append(mtfState.CurrentTrend ? "Bull" : "Bear");
            }
        }

        /// <summary>
        /// Like MtfCreateState but accepts a concrete TimeFrame directly
        /// (for the combined filter's fixed 1H / 15m states).
        /// </summary>
        private MtfTfState MtfCreateStateFixed(TimeFrame tf, int pivotStrength, bool isLowerTf)
        {
            var bars = tf == Bars.TimeFrame ? Bars : MarketData.GetBars(tf);
            return new MtfTfState
            {
                TfBars    = bars,
                PivotLen  = Math.Max(1, pivotStrength),
                IsLowerTf = isLowerTf,
                TfMinutes = MtfTfMinutes(tf)
            };
        }

        // ═════════════════════════════════════════════════════════════════════
        //  IFVG ENGINE  (verbatim from base cBot)
        // ═════════════════════════════════════════════════════════════════════

        private double CalculateMa(int index)
        {
            _maSeries[index] = string.Equals(MaType, "SMA", StringComparison.OrdinalIgnoreCase)
                ? _sma.Result[index]
                : _ema.Result[index];
            return _maSeries[index];
        }

        private int DetectIfvgSignal(int index, double maValue)
        {
            var minSizeValue = MinFvgPips * Symbol.PipSize;
            for (var i = 1; i <= FvgGapBars; i++)
            {
                var fvgType = DetectFvg(index, i, FvgEpsilonPoints);
                if (fvgType == 0) continue;

                int signalDir;
                if (TryProcessFvgCandidate(index, i, fvgType, minSizeValue, maValue, out signalDir))
                    return signalDir;
            }
            return 0;
        }

        private int DetectFvg(int currentIndex, int idx, double epsVal)
        {
            if (idx + 2 > currentIndex) return 0;

            var h2 = Bars.HighPrices[currentIndex - (idx + 2)];
            var l2 = Bars.LowPrices[currentIndex  - (idx + 2)];
            var lt = Bars.LowPrices[currentIndex  - idx];
            var ht = Bars.HighPrices[currentIndex - idx];

            if (lt > h2 - epsVal) return  1;
            if (ht < l2 + epsVal) return -1;
            return 0;
        }

        private bool TryProcessFvgCandidate(int index, int i, int fvgType, double minSizeValue,
                                             double maValue, out int signalDir)
        {
            signalDir = 0;
            var isBearishGap = fvgType == 1;
            var gapLow  = isBearishGap ? Bars.HighPrices[index - (i + 2)] : Bars.HighPrices[index - i];
            var gapHigh = isBearishGap ? Bars.LowPrices[index  - i]        : Bars.LowPrices[index  - (i + 2)];

            if ((gapHigh - gapLow) < minSizeValue) return false;

            if (i > 1)
            {
                for (var k = i - 1; k >= 1; k--)
                {
                    var close = Bars.ClosePrices[index - k];
                    if ((isBearishGap && close < gapLow) || (!isBearishGap && close > gapHigh))
                        return false;
                }
            }

            var breakout = isBearishGap
                ? Bars.ClosePrices[index] < gapLow
                : Bars.ClosePrices[index] > gapHigh;
            if (!breakout) return false;

            var maReady     = !double.IsNaN(maValue) && !double.IsNaN(_maSeries[index - 1]);
            var maCondition = isBearishGap
                ? maReady && maValue < _maSeries[index - 1] && Bars.ClosePrices[index] < maValue
                : maReady && maValue > _maSeries[index - 1] && Bars.ClosePrices[index] > maValue;
            if (!maCondition) return false;

            signalDir = isBearishGap ? -1 : 1;
            return true;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  SMC OB FILTER  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        private bool CheckFilters(int index, int cond)
        {
            var isBull = cond > 0;
            bool? f1 = null;
            bool? f2 = null;

            if (EnableFilter1)
            {
                var pool = isBull ? _smcInternalBullObs : _smcInternalBearObs;
                f1 = HasActiveTouchInLookback(pool, index, Filter1Lookback, isBull);
            }
            if (EnableFilter2)
            {
                var pool = isBull ? _smcSwingBullObs : _smcSwingBearObs;
                f2 = HasActiveTouchInLookback(pool, index, Filter2Lookback, isBull);
            }

            if (f1.HasValue && f2.HasValue)
            {
                if (!f1.Value && !f2.Value)
                {
                    Print("[Filter BLOCKED] {0} at bar {1}: no OB touched within {2}/{3} bars.",
                          isBull ? "Long" : "Short", index, Filter1Lookback, Filter2Lookback);
                    return false;
                }
                return true;
            }
            if (f1.HasValue && !f1.Value)
            {
                Print("[Filter1 BLOCKED] {0} at bar {1}: no internal OB touched within {2} bars.",
                      isBull ? "Long" : "Short", index, Filter1Lookback);
                return false;
            }
            if (f2.HasValue && !f2.Value)
            {
                Print("[Filter2 BLOCKED] {0} at bar {1}: no swing OB touched within {2} bars.",
                      isBull ? "Long" : "Short", index, Filter2Lookback);
                return false;
            }
            return true;
        }

        private bool HasActiveTouchInLookback(List<SmcObRecord> pool, int signalBar,
                                               int lookback, bool bullish)
        {
            if (pool.Count == 0) return false;

            var currentAtr = double.IsNaN(_smcAtrWilder) ? 0.0 : _smcAtrWilder;
            var closeNow   = Bars.ClosePrices[signalBar];

            foreach (var ob in pool)
            {
                if (EnableMinBarsFromOrigin)
                {
                    var minBars = ob.Internal ? MinBarsFromOriginInternal : MinBarsFromOriginSwing;
                    if (signalBar - ob.Index < minBars) continue;
                }

                if (EnableAtrDistanceFilter && AtrDistanceMultiplier > 0 && currentAtr > 0)
                {
                    var advance = bullish ? closeNow - ob.Top : ob.Bottom - closeNow;
                    if (advance < AtrDistanceMultiplier * currentAtr) continue;
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

        // ─────────────────────────────────────────────────────────────────────

        private void RunSmcFilter(int index)
        {
            for (var i = _lastParsedIndex + 1; i <= index; i++)
                UpdateSmcParsedArrays(i);
            _lastParsedIndex = index;

            if (index < _smcWarmup) return;

            const int iLen = 5;
            var       sLen = Math.Max(5, SmcSwingsLengthInput);

            var internalLegNow = ComputeLeg(index, iLen, _internalLeg);
            var internalDc     = internalLegNow - _internalLeg;
            if (internalDc != 0)
            {
                if (internalDc == 1)
                { _internalLowLevel = Bars.LowPrices[index - iLen]; _internalLowIndex = index - iLen; _internalLowCrossed = false; }
                else
                { _internalHighLevel = Bars.HighPrices[index - iLen]; _internalHighIndex = index - iLen; _internalHighCrossed = false; }
            }
            _internalLeg = internalLegNow;

            var swingLegNow = ComputeLeg(index, sLen, _swingLeg);
            var swingDc     = swingLegNow - _swingLeg;
            if (swingDc != 0)
            {
                if (swingDc == 1)
                { _lastSwingLow = Bars.LowPrices[index - sLen]; _lastSwingLowIndex = index - sLen; _swingLowCrossed = false; }
                else
                { _lastSwingHigh = Bars.HighPrices[index - sLen]; _lastSwingHighIndex = index - sLen; _swingHighCrossed = false; }
            }
            _swingLeg = swingLegNow;

            var close = Bars.ClosePrices[index];

            if (!double.IsNaN(_internalHighLevel) && !_internalHighCrossed && close > _internalHighLevel)
            { _internalHighCrossed = true; _internalTrend = 1;  StoreSmcOrderBlock(_internalHighIndex, true,  1,  index); }
            if (!double.IsNaN(_internalLowLevel)  && !_internalLowCrossed  && close < _internalLowLevel)
            { _internalLowCrossed  = true; _internalTrend = -1; StoreSmcOrderBlock(_internalLowIndex,  true,  -1, index); }
            if (!double.IsNaN(_lastSwingHigh)      && !_swingHighCrossed    && close > _lastSwingHigh)
            { _swingHighCrossed    = true; _swingTrend    = 1;  StoreSmcOrderBlock(_lastSwingHighIndex, false, 1,  index); }
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
                else                            { _smcAtrWilder = (_smcAtrWilder * (SmcAtrPeriod - 1) + tr) / SmcAtrPeriod; }
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
            if (pivotIndex < 0 || pivotIndex >= breakIndex || breakIndex >= _parsedHighs.Count)
                return;

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

            var bullish = bias == 1;
            var ob = new SmcObRecord
            {
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                Internal            = isInternal,
                Time                = _times[parsedIndex],
                StructureBreakIndex = breakIndex
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
                if ((bullish && bullSrc < ob.Bottom) || (!bullish && bearSrc > ob.Top))
                    list.RemoveAt(i);
            }
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

        // ═════════════════════════════════════════════════════════════════════
        //  BSL / SSL ENGINE  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);
            _bslCurrentBsl = _bslBuysidePools.First  != null ? _bslBuysidePools.First.Value.Price  : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;
            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;
            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];
            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type =  1 });
            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow,  BarIndex = pivotIndex, Type = -1 });
        }

        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++) if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++) if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type &&
                    Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1) return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxBslPivots) _bslPivots.RemoveLast();
        }

        private void BslAddPoolFromNewPivot(int currentIndex)
        {
            int confirmedIdx = currentIndex - PivotRight;
            foreach (var pivot in _bslPivots)
            {
                if (pivot.BarIndex != confirmedIdx) continue;
                var pool = new BslPool { Price = pivot.Price, PivotIndex = pivot.BarIndex };
                if (pivot.Type ==  1) _bslBuysidePools.AddFirst(pool);
                if (pivot.Type == -1) _bslSellsidePools.AddFirst(pool);
            }
        }

        private void BslClearMitigated(int index)
        {
            var node = _bslSellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price) _bslSellsidePools.Remove(node);
                node = next;
            }
            node = _bslBuysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node);
                node = next;
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  TRADE ENTRY  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            { Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar); return; }
            if (sslLevel >= entry)
            { Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.", signalBar, sslLevel, entry); return; }
            double slPips = (entry - sslLevel) / Symbol.PipSize + SlBufferPips;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} | SL={3:F1}p (buf={4:F1}) | TP={5:F1}p | Vol={6}",
                  signalBar, entry, sslLevel, slPips, SlBufferPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry    = Symbol.Bid;
            double bslLevel = _bslCurrentBsl;
            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            { Print("Bar {0}: SHORT skipped – BSL unavailable.", signalBar); return; }
            if (bslLevel <= entry)
            { Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.", signalBar, bslLevel, entry); return; }
            double slPips = (bslLevel - entry) / Symbol.PipSize + SlBufferPips;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} | SL={3:F1}p (buf={4:F1}) | TP={5:F1}p | Vol={6}",
                  signalBar, entry, bslLevel, slPips, SlBufferPips, slPips * RiskRewardRatio, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, slPips, slPips * RiskRewardRatio);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} < min {3:F1} pips.", signalBar, direction, slPips, MinSlPips); return false; }
            if (slPips > MaxSlPips) { Print("Bar {0}: {1} skipped – SL {2:F1} > max {3:F1} pips.", signalBar, direction, slPips, MaxSlPips); return false; }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double volume = Symbol.NormalizeVolumeInUnits(Symbol.VolumeForFixedRisk(riskAmount, slPips), RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  MTF PIVOT-TREND FILTER ENGINE  (verbatim)
        // ═════════════════════════════════════════════════════════════════════

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
            };
        }

        private void MtfAdvanceState(MtfTfState s, DateTime chartTime)
        {
            if (s == null) return;
            var tfBarIndex = MtfResolveTfBar(s, chartTime);
            if (tfBarIndex < 0) return;
            for (var i = s.LastProcessedTfBar + 1; i <= tfBarIndex; i++)
                MtfProcessCalcBar(s, i);
            if (tfBarIndex > s.LastProcessedTfBar)
                s.LastProcessedTfBar = tfBarIndex;
        }

        private void MtfProcessCalcBar(MtfTfState s, int tfBarIndex)
        {
            var bars             = s.TfBars;
            var prevLastPivotHigh = s.LastPivotHigh;
            var prevLastPivotLow  = s.LastPivotLow;

            if (tfBarIndex >= s.PivotLen * 2)
            {
                var pivotIdx = tfBarIndex - s.PivotLen;

                if (MtfIsPivotHigh(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.HighPrices[pivotIdx];
                    s.LastPivotHigh = s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotHigh) ? pp : Math.Max(pp, s.LastPivotHigh))
                        : pp;
                    if (s.LastPivotHigh != prevLastPivotHigh)
                        s.PivotHighTime = bars.OpenTimes[pivotIdx];
                }

                if (MtfIsPivotLow(bars, pivotIdx, s.PivotLen))
                {
                    var pp = bars.LowPrices[pivotIdx];
                    s.LastPivotLow = !s.CurrentTrend
                        ? (double.IsNaN(s.LastPivotLow) ? pp : Math.Min(pp, s.LastPivotLow))
                        : pp;
                    if (s.LastPivotLow != prevLastPivotLow)
                        s.PivotLowTime = bars.OpenTimes[pivotIdx];
                }
            }

            var close     = bars.ClosePrices[tfBarIndex];
            var prevClose = tfBarIndex > 0 ? bars.ClosePrices[tfBarIndex - 1] : close;

            if (!double.IsNaN(s.LastPivotHigh) && !double.IsNaN(prevLastPivotHigh))
            {
                if (prevClose <= prevLastPivotHigh && close > s.LastPivotHigh)
                {
                    s.CurrentTrend   = true;
                    s.HasTrend       = true;
                    s.LastBrokenHigh = s.LastPivotHigh;
                    s.LastBrokenLow  = double.NaN;
                }
            }

            if (!double.IsNaN(s.LastPivotLow) && !double.IsNaN(prevLastPivotLow))
            {
                if (prevClose >= prevLastPivotLow && close < s.LastPivotLow)
                {
                    s.CurrentTrend   = false;
                    s.HasTrend       = true;
                    s.LastBrokenLow  = s.LastPivotLow;
                    s.LastBrokenHigh = double.NaN;
                }
            }
        }

        private int MtfResolveTfBar(MtfTfState s, DateTime chartTime)
        {
            if (!s.IsLowerTf)
            {
                var adjusted = chartTime.AddMinutes(-(s.TfMinutes > 0 ? s.TfMinutes : 1));
                return MtfFindAtOrBefore(s.TfBars, adjusted);
            }

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

        private bool CheckMtfFilter(bool isLong, int signalBar)
        {
            if (!EnableMtfFilter) return true;

            var isAnd   = MtfFilterLogic == MtfLogicMode.AND;
            var enabled = 0;
            var passing = 0;

            if (_mtfState1 != null) { enabled++; if (!_mtfState1.HasTrend || _mtfState1.CurrentTrend == isLong) passing++; }
            if (_mtfState2 != null) { enabled++; if (!_mtfState2.HasTrend || _mtfState2.CurrentTrend == isLong) passing++; }
            if (_mtfState3 != null) { enabled++; if (!_mtfState3.HasTrend || _mtfState3.CurrentTrend == isLong) passing++; }
            if (_mtfState4 != null) { enabled++; if (!_mtfState4.HasTrend || _mtfState4.CurrentTrend == isLong) passing++; }

            if (enabled == 0) return true;

            var result = isAnd ? (passing == enabled) : (passing > 0);

            if (!result)
                Print("[MTF BLOCKED] {0} bar={1} passing={2}/{3} logic={4} | TF1:{5}({6}) TF2:{7}({8}) TF3:{9}({10}) TF4:{11}({12})",
                    isLong ? "Long" : "Short", signalBar, passing, enabled,
                    isAnd ? "AND" : "OR",
                    _mtfState1 != null ? (_mtfState1.HasTrend ? (_mtfState1.CurrentTrend ? "Bull" : "Bear") : "Warm") : "Off", _mtfState1 != null ? _mtfState1.LastProcessedTfBar : -1,
                    _mtfState2 != null ? (_mtfState2.HasTrend ? (_mtfState2.CurrentTrend ? "Bull" : "Bear") : "Warm") : "Off", _mtfState2 != null ? _mtfState2.LastProcessedTfBar : -1,
                    _mtfState3 != null ? (_mtfState3.HasTrend ? (_mtfState3.CurrentTrend ? "Bull" : "Bear") : "Warm") : "Off", _mtfState3 != null ? _mtfState3.LastProcessedTfBar : -1,
                    _mtfState4 != null ? (_mtfState4.HasTrend ? (_mtfState4.CurrentTrend ? "Bull" : "Bear") : "Warm") : "Off", _mtfState4 != null ? _mtfState4.LastProcessedTfBar : -1);

            return result;
        }

        // ─────────────────────────────────────────────────────────────────────
        //  MTF helpers  (verbatim)
        // ─────────────────────────────────────────────────────────────────────

        private static bool MtfIsPivotHigh(Bars bars, int idx, int len)
        {
            var left = idx - len; var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.HighPrices[idx];
            for (var i = left; i <= right; i++) if (i != idx && bars.HighPrices[i] >= pivot) return false;
            return true;
        }

        private static bool MtfIsPivotLow(Bars bars, int idx, int len)
        {
            var left = idx - len; var right = idx + len;
            if (left < 0 || right >= bars.Count) return false;
            var pivot = bars.LowPrices[idx];
            for (var i = left; i <= right; i++) if (i != idx && bars.LowPrices[i] <= pivot) return false;
            return true;
        }

        private static int MtfFindAtOrBefore(Bars bars, DateTime time)
        {
            var lo = 0; var hi = bars.Count - 1;
            while (lo <= hi)
            {
                var mid = (lo + hi) / 2;
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
                else lo = mid + 1;
            }
            return ans;
        }

        private static TimeFrame MtfParseTimeFrame(string text)
        {
            switch ((text ?? string.Empty).Trim().ToUpperInvariant())
            {
                case "1":           return TimeFrame.Minute;
                case "2":           return TimeFrame.Minute2;
                case "3":           return TimeFrame.Minute3;
                case "4":           return TimeFrame.Minute4;
                case "5":           return TimeFrame.Minute5;
                case "10":          return TimeFrame.Minute10;
                case "15":          return TimeFrame.Minute15;
                case "30":          return TimeFrame.Minute30;
                case "45":          return TimeFrame.Minute45;
                case "60":
                case "1H":          return TimeFrame.Hour;
                case "120":
                case "2H":          return TimeFrame.Hour2;
                case "240":
                case "4H":          return TimeFrame.Hour4;
                case "480":
                case "8H":          return TimeFrame.Hour8;
                case "720":
                case "12H":         return TimeFrame.Hour12;
                case "D":
                case "1D":          return TimeFrame.Daily;
                case "W":
                case "1W":          return TimeFrame.Weekly;
                case "M":
                case "1M":          return TimeFrame.Monthly;
                default:            return TimeFrame.Minute15;
            }
        }

        private static int MtfTfMinutes(TimeFrame tf)
        {
            if (tf == TimeFrame.Minute)    return 1;
            if (tf == TimeFrame.Minute2)   return 2;
            if (tf == TimeFrame.Minute3)   return 3;
            if (tf == TimeFrame.Minute4)   return 4;
            if (tf == TimeFrame.Minute5)   return 5;
            if (tf == TimeFrame.Minute10)  return 10;
            if (tf == TimeFrame.Minute15)  return 15;
            if (tf == TimeFrame.Minute30)  return 30;
            if (tf == TimeFrame.Minute45)  return 45;
            if (tf == TimeFrame.Hour)      return 60;
            if (tf == TimeFrame.Hour2)     return 120;
            if (tf == TimeFrame.Hour4)     return 240;
            if (tf == TimeFrame.Hour8)     return 480;
            if (tf == TimeFrame.Hour12)    return 720;
            if (tf == TimeFrame.Daily)     return 1440;
            if (tf == TimeFrame.Weekly)    return 10080;
            if (tf == TimeFrame.Monthly)   return 43200;
            return 0;
        }
    }
}
