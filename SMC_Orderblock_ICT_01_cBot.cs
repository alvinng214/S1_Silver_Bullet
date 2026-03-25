// =============================================================================
// SMC_Orderblock_ICT_01_cBot
// =============================================================================
// Trade triggers : ICT_01 engine (self-contained, from ICT_01_cBot_Single.cs)
//                  FVG hunts/sweeps generate long/short signals.
//                  Stop loss anchored to BSL/SSL pivot levels (self-contained,
//                  from BSL and SSL.cs).
//
// SMC filter     : Smart Money Concepts [LuxAlgo] OB detection engine
//                  (self-contained, from SMC_Orderblock_Detector_Filter_cBot).
//                  Maintains live pools of unmitigated Internal and Swing OBs.
//                  Filter 1 / Filter 2 / combined OR logic — identical to
//                  SMC_Orderblock_Detector_Filter_cBot.
//
// No external indicator references — compiles as a single file.
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class SMC_Orderblock_ICT_01_cBot : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  Enums
        // ═════════════════════════════════════════════════════════════════════

        public enum ObFilter       { Atr, CumulativeMeanRange }
        public enum MitigationMode { Close, HighLow }

        // ═════════════════════════════════════════════════════════════════════
        //  Inner types — ICT BSL/SSL
        // ═════════════════════════════════════════════════════════════════════

        private sealed class BslPivot
        {
            public double Price;
            public int    BarIndex;
            public int    Type;    // 1 = pivot high (BSL), -1 = pivot low (SSL)
        }

        private sealed class BslPool
        {
            public double Price;
            public int    PivotIndex;
        }

        // ── SMC filter OB record (mirrors LuxAlgo OrderBlock, filter-only fields)

        private sealed class SmcObRecord
        {
            public int      Index;                // pivot bar (parsedIndex)
            public double   Top;
            public double   Bottom;
            public bool     Bullish;
            public bool     Internal;
            public int      StructureBreakIndex;  // bar where BOS/iBOS/CHoCH/iCHoCH fired
            public DateTime Time;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — ICT Setup 01
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("FVG Detector Multiplier", DefaultValue = 1.0, MinValue = 1.0, Group = "ICT Setup 01")]
        public double FvgDetectorMultiplier { get; set; }

        [Parameter("FVG Validity Period (bars)", DefaultValue = 15, MinValue = 2, Group = "ICT Setup 01")]
        public int FvgValidityPeriod { get; set; }

        [Parameter("Use Discount & Premium Zone", DefaultValue = false, Group = "ICT Setup 01")]
        public bool UseDiscountAndPremium { get; set; }

        [Parameter("Signal Method (Hunt / Sweeps)", DefaultValue = "Hunt", Group = "ICT Setup 01")]
        public string SignalMethod { get; set; }

        [Parameter("Signals Allowed Per Zone", DefaultValue = 3, MinValue = 1, Group = "ICT Setup 01")]
        public int SignalsAllowedPerZone { get; set; }

        [Parameter("Signal After Hunts", DefaultValue = false, Group = "ICT Setup 01")]
        public bool SignalAfterHunts { get; set; }

        [Parameter("Required Hunts Count", DefaultValue = 2, MinValue = 1, Group = "ICT Setup 01")]
        public int RequiredHunts { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — BSL & SSL
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Risk Management
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk Per Trade (%)", DefaultValue = 1.0, MinValue = 0.1, MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio", DefaultValue = 2.0, MinValue = 0.1, Step = 0.1, Group = "Risk Management")]
        public double RiskRewardRatio { get; set; }

        [Parameter("Max Simultaneous Positions", DefaultValue = 3, MinValue = 1, MaxValue = 100, Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)", DefaultValue = 3.0, MinValue = 0.1, Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)", DefaultValue = 500.0, MinValue = 10.0, Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Instance Name", DefaultValue = "SMC_ICT01_cBot", Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — SMC Filter — Swing Structure
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Swings Length", DefaultValue = 50, MinValue = 10, Group = "SMC Filter — Swing Structure")]
        public int SmcSwingsLengthInput { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — SMC Filter — Order Blocks
        //  Internal pivot length is hardcoded to 5 (matches LuxAlgo source).
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Order Block Filter", DefaultValue = ObFilter.Atr, Group = "SMC Filter — Order Blocks")]
        public ObFilter SmcOrderBlockFilterInput { get; set; }

        [Parameter("OB Filter ATR Period", DefaultValue = 200, MinValue = 1, MaxValue = 500, Group = "SMC Filter — Order Blocks")]
        public int SmcObFilterAtrPeriod { get; set; }

        [Parameter("Order Block Mitigation", DefaultValue = MitigationMode.HighLow, Group = "SMC Filter — Order Blocks")]
        public MitigationMode SmcOrderBlockMitigationInput { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Filter 1 (Internal OB)
        //
        //  Defines how long after price touches a bullish/bearish Internal OB
        //  an ICT signal is still accepted.
        //
        //  Flow per trade:
        //    1. BOS / iBOS / CHoCH / iCHoCH breaks structure → Internal OB created.
        //    2. Price returns and touches that OB (Low ≤ ob.Top for long,
        //       High ≥ ob.Bottom for short).
        //    3. An ICT_01 signal must appear within OB Touch Window bars of
        //       that touch → trade triggered.
        //
        //  OB Touch Window = 0 → ICT signal must be on the exact touch bar.
        //  OB Touch Window = 10 → ICT signal allowed up to 10 bars after touch.
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable Filter 1 (Internal OB)", DefaultValue = true, Group = "Filter 1 — Internal OB")]
        public bool EnableFilter1 { get; set; }

        [Parameter("OB Touch Window — Internal (bars)", DefaultValue = 10, MinValue = 0, Group = "Filter 1 — Internal OB")]
        public int Filter1Lookback { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — Filter 2 (Swing OB)
        //
        //  Same flow as Filter 1 but applied to the Swing OB pool.
        //  When both filters are enabled, either pool passing is sufficient (OR).
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Enable Filter 2 (Swing OB)", DefaultValue = false, Group = "Filter 2 — Swing OB")]
        public bool EnableFilter2 { get; set; }

        [Parameter("OB Touch Window — Swing (bars)", DefaultValue = 10, MinValue = 0, Group = "Filter 2 — Swing OB")]
        public int Filter2Lookback { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  Parameters — OB Quality Filters
        // ═════════════════════════════════════════════════════════════════════

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

        // ═════════════════════════════════════════════════════════════════════
        //  Constants
        // ═════════════════════════════════════════════════════════════════════

        private const int IctAtrLength = 55;
        private const int MaxBslPivots = 10;

        // ═════════════════════════════════════════════════════════════════════
        //  ICT_01 embedded fields
        // ═════════════════════════════════════════════════════════════════════

        private double _ictTrueRange;
        private double _ictAtr    = double.NaN;
        private double _ictAtrSum = 0.0;

        private double _ictBullDistal;
        private double _ictBullProximal;
        private double _ictBearDistal;
        private double _ictBearProximal;

        private int _ictBullFvgPointSaved;
        private int _ictBearFvgPointSaved;

        private double _ictBullFvgDistal;
        private double _ictBullFvgProximal;
        private int    _ictBullFvgPoint;
        private double _ictBullPremium;
        private double _ictBullDiscount;
        private double _ictBullEquilibrium;

        private double _ictBearFvgDistal;
        private double _ictBearFvgProximal;
        private int    _ictBearFvgPoint;
        private double _ictBearPremium;
        private double _ictBearDiscount;
        private double _ictBearEquilibrium;

        private bool   _ictBullFvgValid = true;
        private bool   _ictBearFvgValid = true;
        private bool   _ictIsBullFvg;
        private bool   _ictIsBearFvg;
        private double _ictLowTracker;
        private double _ictHighTracker;
        private int    _ictLongSignalCount;
        private int    _ictShortSignalCount;

        private bool _ictIsLongSignal;
        private bool _ictIsShortSignal;

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL embedded fields
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();

        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        // ═════════════════════════════════════════════════════════════════════
        //  SMC filter engine fields  (mirrors LuxAlgo + SMC_OB_Filter_cBot)
        // ═════════════════════════════════════════════════════════════════════

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

        // ═════════════════════════════════════════════════════════════════════
        //  cBot state
        // ═════════════════════════════════════════════════════════════════════

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;
        private int _lastProcessed      = -1;

        // ═════════════════════════════════════════════════════════════════════
        //  Lifecycle
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            _smcWarmup = Math.Max(SmcSwingsLengthInput, 5) + 5;
            Print("SMC_Orderblock_ICT_01_cBot started. MaxPositions={0}, Risk={1}%",
                  MaxOpenPositions, RiskPercent);
        }

        protected override void OnStop()
        {
            Print("SMC_Orderblock_ICT_01_cBot stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  OnBar
        //
        //  Step 1 — Incremental fill: BSL/SSL → ICT → SMC filter (all three
        //           must run bar-by-bar so their state is fully built before
        //           any trade decision is made).
        //  Step 2 — Read ICT signal outputs.
        //  Step 3 — SMC filter check.
        //  Step 4 — Execute trade.
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;

            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);       // SL anchor levels
                RunIct(i);          // ICT FVG signal detection
                RunSmcFilter(i);    // SMC OB pool maintenance
            }
            _lastProcessed = signalBar;

            if (signalBar < 2) return;

            bool isLong  = _ictIsLongSignal;
            bool isShort = _ictIsShortSignal;
            if (!isLong && !isShort) return;

            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            if (isLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                if (CheckFilters(signalBar, 1))
                    TryEnterLong(signalBar);
            }

            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;

            if (isShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                if (CheckFilters(signalBar, -1))
                    TryEnterShort(signalBar);
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  SMC filter — verbatim from SMC_Orderblock_Detector_Filter_cBot
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

            // Both enabled → OR (either pool passing is sufficient)
            if (f1.HasValue && f2.HasValue)
            {
                if (!f1.Value && !f2.Value)
                {
                    Print("[Filter BLOCKED] {0} at bar {1}: no unmitigated internal or swing OB touched within {2}/{3} bars of ICT signal.",
                          isBull ? "Long" : "Short", index, Filter1Lookback, Filter2Lookback);
                    return false;
                }
                return true;
            }
            if (f1.HasValue && !f1.Value)
            {
                Print("[Filter1 BLOCKED] {0} at bar {1}: no unmitigated internal OB touched within {2} bars of ICT signal.",
                      isBull ? "Long" : "Short", index, Filter1Lookback);
                return false;
            }
            if (f2.HasValue && !f2.Value)
            {
                Print("[Filter2 BLOCKED] {0} at bar {1}: no unmitigated swing OB touched within {2} bars of ICT signal.",
                      isBull ? "Long" : "Short", index, Filter2Lookback);
                return false;
            }
            return true;
        }

        private bool HasActiveTouchInLookback(
            List<SmcObRecord> pool, int signalBar, int lookback, bool bullish)
        {
            if (pool.Count == 0) return false;

            var currentAtr = double.IsNaN(_smcAtrWilder) ? 0.0 : _smcAtrWilder;
            var closeNow   = Bars.ClosePrices[signalBar];

            foreach (var ob in pool)
            {
                // ── Quality Filter A: min bars from OB pivot origin ───────────
                if (EnableMinBarsFromOrigin)
                {
                    var minBars = ob.Internal ? MinBarsFromOriginInternal : MinBarsFromOriginSwing;
                    if (signalBar - ob.Index < minBars) continue;
                }

                // ── Quality Filter B: ATR distance from OB level ──────────────
                if (EnableAtrDistanceFilter && AtrDistanceMultiplier > 0 && currentAtr > 0)
                {
                    var advance = bullish ? closeNow - ob.Top : ob.Bottom - closeNow;
                    if (advance < AtrDistanceMultiplier * currentAtr) continue;
                }

                // ── Find the most recent bar (after the structure break) that
                //    touched this OB.  No restriction on how far back the touch
                //    can be from the structure break — the OB can be left alone
                //    for any number of bars before price returns to it.
                var lastTouchBar = -1;
                for (var b = ob.StructureBreakIndex + 1; b <= signalBar; b++)
                {
                    if ( bullish && Bars.LowPrices[b]  <= ob.Top)    lastTouchBar = b;
                    if (!bullish && Bars.HighPrices[b] >= ob.Bottom) lastTouchBar = b;
                }

                if (lastTouchBar < 0) continue;   // OB never touched — skip

                // ── The ICT signal must appear within `lookback` bars of the
                //    most recent OB touch for the trade to be allowed.
                //    lookback = 0  → ICT signal must be on the same bar as the touch
                //    lookback = 10 → ICT signal allowed up to 10 bars after the touch
                if (signalBar - lastTouchBar <= lookback) return true;
            }
            return false;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  SMC filter detection engine — verbatim from SMC_OB_Filter_cBot
        // ═════════════════════════════════════════════════════════════════════

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

                if (index < SmcAtrPeriod)
                { _smcAtrWilderSum += tr; _smcAtrWilder = double.NaN; }
                else if (index == SmcAtrPeriod)
                { _smcAtrWilderSum += tr; _smcAtrWilder = _smcAtrWilderSum / SmcAtrPeriod; }
                else
                { _smcAtrWilder = (_smcAtrWilder * (SmcAtrPeriod - 1) + tr) / SmcAtrPeriod; }
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

            var bullish = (bias == 1);
            var ob = new SmcObRecord
            {
                Index               = parsedIndex,
                Top                 = _parsedHighs[parsedIndex],
                Bottom              = _parsedLows[parsedIndex],
                Bullish             = bullish,
                Internal            = isInternal,
                Time                = _times[parsedIndex],
                StructureBreakIndex = breakIndex   // touch scan must start AFTER this bar
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
        //  ICT_01 embedded engine — verbatim from ICT_01_cBot_Single
        // ═════════════════════════════════════════════════════════════════════

        private void RunIct(int index)
        {
            IctCalculateAtr(index);

            if (index == 0)
            {
                _ictBullDistal = _ictBullProximal = _ictBearDistal = _ictBearProximal = 0.0;
                _ictBullFvgPointSaved = _ictBearFvgPointSaved = 0;
                _ictIsLongSignal = _ictIsShortSignal = false;
                return;
            }

            var prevBullDistal   = _ictBullDistal;
            var prevBullProximal = _ictBullProximal;
            var prevBearDistal   = _ictBearDistal;
            var prevBearProximal = _ictBearProximal;

            var high  = Bars.HighPrices[index];
            var low   = Bars.LowPrices[index];
            var close = Bars.ClosePrices[index];

            _ictIsBullFvg = false;
            _ictIsBearFvg = false;

            if (index >= 2)
            {
                var high2    = Bars.HighPrices[index - 2];
                var low2     = Bars.LowPrices[index - 2];
                var high1    = Bars.HighPrices[index - 1];
                var low1     = Bars.LowPrices[index - 1];
                var atrValue = double.IsNaN(_ictAtr) ? 0.0 : _ictAtr;

                if ((high - low2) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low > high2 && low2 < low1 && high1 < high && (high + low2) / 2.0 >= high2)
                    {
                        _ictBullFvgDistal   = high2;   _ictBullFvgProximal = low;
                        _ictBullFvgPoint    = index;   _ictBullDiscount    = low2;
                        _ictBullPremium     = high;    _ictBullEquilibrium = (high + low2) / 2.0;
                        _ictIsBullFvg       = true;
                    }
                }
                if ((high2 - low) > (FvgDetectorMultiplier * atrValue))
                {
                    if (low2 > high && high2 > high1 && low1 > low && (low + high2) / 2.0 <= low2)
                    {
                        _ictBearFvgDistal   = low2;    _ictBearFvgProximal = high;
                        _ictBearFvgPoint    = index;   _ictBearDiscount    = low;
                        _ictBearPremium     = high2;   _ictBearEquilibrium = (low + high2) / 2.0;
                        _ictIsBearFvg       = true;
                    }
                }
            }

            if (UseDiscountAndPremium)
            {
                if (_ictIsBullFvg)
                {
                    _ictBullDistal   = _ictBullFvgDistal;
                    _ictBullProximal = _ictBullEquilibrium >= _ictBullFvgProximal ? _ictBullFvgProximal : _ictBullEquilibrium;
                }
                if (_ictIsBearFvg)
                {
                    _ictBearDistal   = _ictBearFvgDistal;
                    _ictBearProximal = _ictBearEquilibrium <= _ictBearFvgProximal ? _ictBearFvgProximal : _ictBearEquilibrium;
                }
            }
            else
            {
                if (_ictIsBullFvg) { _ictBullDistal = _ictBullFvgDistal; _ictBullProximal = _ictBullFvgProximal; }
                if (_ictIsBearFvg) { _ictBearDistal = _ictBearFvgDistal; _ictBearProximal = _ictBearFvgProximal; }
            }

            var body1 = Bars.ClosePrices[index - 1] - Bars.OpenPrices[index - 1];

            if (_ictBullFvgValid)
            {
                var bp = _ictBullProximal;
                _ictBullFvgValid = IctUpdateZoneValidity(index, body1, true,  _ictBullFvgPoint, prevBullDistal, prevBullProximal, _ictLongSignalCount,  ref bp);
                _ictBullProximal = bp;
            }
            if (_ictBearFvgValid)
            {
                var bp = _ictBearProximal;
                _ictBearFvgValid = IctUpdateZoneValidity(index, body1, false, _ictBearFvgPoint, prevBearDistal, prevBearProximal, _ictShortSignalCount, ref bp);
                _ictBearProximal = bp;
            }

            if (_ictBullFvgPointSaved != _ictBullFvgPoint)
            { _ictBullFvgValid = true; _ictLowTracker = 0.0; _ictLongSignalCount = 0; _ictIsLongSignal = false; }
            if (_ictBearFvgPointSaved != _ictBearFvgPoint)
            { _ictBearFvgValid = true; _ictHighTracker = 0.0; _ictShortSignalCount = 0; _ictIsShortSignal = false; }

            if (_ictBullFvgValid)
            {
                if (_ictLowTracker == 0.0 && low < _ictBullProximal) _ictLowTracker = low;
                if (low < _ictLowTracker && _ictLowTracker > 0.0)
                {
                    _ictLowTracker = low;
                    if (close >= _ictBullProximal)
                    { _ictLongSignalCount++; _ictIsLongSignal = SignalAfterHunts ? _ictLongSignalCount == RequiredHunts : true; }
                    else { _ictIsLongSignal = false; }
                }
                else { _ictIsLongSignal = false; }
            }
            else { _ictLowTracker = 0.0; _ictLongSignalCount = 0; _ictIsLongSignal = false; }

            if (_ictBearFvgValid)
            {
                if (_ictHighTracker == 0.0 && high > _ictBearProximal) _ictHighTracker = high;
                if (high > _ictHighTracker && _ictHighTracker > 0.0)
                {
                    _ictHighTracker = high;
                    if (close <= _ictBearProximal)
                    { _ictShortSignalCount++; _ictIsShortSignal = SignalAfterHunts ? _ictShortSignalCount == RequiredHunts : true; }
                    else { _ictIsShortSignal = false; }
                }
                else { _ictIsShortSignal = false; }
            }
            else { _ictHighTracker = 0.0; _ictShortSignalCount = 0; _ictIsShortSignal = false; }

            _ictBullFvgPointSaved = _ictBullFvgPoint;
            _ictBearFvgPointSaved = _ictBearFvgPoint;
        }

        private void IctCalculateAtr(int index)
        {
            var high = Bars.HighPrices[index]; var low = Bars.LowPrices[index];
            if (index == 0)
            { _ictTrueRange = high - low; _ictAtrSum += _ictTrueRange; _ictAtr = double.NaN; return; }
            var pc = Bars.ClosePrices[index - 1];
            _ictTrueRange = Math.Max(high - low, Math.Max(Math.Abs(high - pc), Math.Abs(low - pc)));
            _ictAtrSum   += _ictTrueRange;
            if      (index < IctAtrLength - 1)  _ictAtr = double.NaN;
            else if (index == IctAtrLength - 1) _ictAtr = _ictAtrSum / IctAtrLength;
            else                                _ictAtr = ((_ictAtr * (IctAtrLength - 1)) + _ictTrueRange) / IctAtrLength;
        }

        private bool IctUpdateZoneValidity(
            int index, double body1, bool isBull, int zonePoint,
            double prevDistal, double prevProximal, int signalCount,
            ref double updatedProximal)
        {
            var useOpen       = isBull ? body1 > 0 : body1 <= 0;
            var selectedPrice = useOpen ? Bars.OpenPrices[index - 1] : Bars.ClosePrices[index - 1];
            var sweepPrice    = isBull
                ? (SignalMethod == "Sweeps" ? selectedPrice : Bars.LowPrices[index - 1])
                : (SignalMethod == "Sweeps" ? selectedPrice : Bars.HighPrices[index - 1]);

            if (isBull ? sweepPrice < prevDistal : sweepPrice > prevDistal) return false;
            if (index > zonePoint + FvgValidityPeriod)                      return false;
            if (!SignalAfterHunts && signalCount > SignalsAllowedPerZone - 1) return false;

            var movedInside = isBull
                ? selectedPrice < prevProximal && selectedPrice > prevDistal
                : selectedPrice > prevProximal && selectedPrice < prevDistal;
            if (movedInside) updatedProximal = selectedPrice;
            return true;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL embedded engine — verbatim from ICT_01_cBot_Single
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
            while (node != null) { var next = node.Next; if (Bars.LowPrices[index]  <= node.Value.Price) _bslSellsidePools.Remove(node); node = next; }
            node = _bslBuysidePools.First;
            while (node != null) { var next = node.Next; if (Bars.HighPrices[index] >= node.Value.Price) _bslBuysidePools.Remove(node); node = next; }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  Trade entry helpers — verbatim from ICT_01_cBot_Single
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;
            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            { Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar); return; }
            if (sslLevel >= entry)
            { Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.", signalBar, sslLevel, entry); return; }
            double slPips = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: LONG  | Entry={1:F5} | SSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, sslLevel, slPips, slPips * RiskRewardRatio, volume);
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
            double slPips = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;
            double volume = GetRiskVolume(Account.Equity * (RiskPercent / 100.0), slPips);
            if (volume <= 0) { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }
            Print("Bar {0}: SHORT | Entry={1:F5} | BSL={2:F5} ({3:F1} pips) | TP={4:F1} pips | Vol={5}",
                  signalBar, entry, bslLevel, slPips, slPips * RiskRewardRatio, volume);
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
    }
}
