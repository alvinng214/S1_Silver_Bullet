// =============================================================================
// Autotrendline_BSLSSL_cBot
// =============================================================================
// Self-contained cBot.  No external indicator references.
//
// Signal engine  — Auto TrendLines [TradingFinder] logic embedded.
//   Upward   triangle = Long  signal:
//     • React on any Up TrendLine   (price bounced off rising support)
//     • Break of any Down TrendLine (price broke above falling resistance)
//   Downward triangle = Short signal:
//     • Break of any Up TrendLine   (price broke below rising support)
//     • React on any Down TrendLine (price rejected off falling resistance)
//   All chart drawing removed; only signal detection retained.
//   16 individual toggles let you enable/disable each signal source.
//
// SL anchor      — BSL & SSL logic embedded (mirrors ICT_01_cBot_single.cs).
//   Long  SL = most recent unmitigated Sellside Liquidity (SSL) pivot low.
//   Short SL = most recent unmitigated Buyside  Liquidity (BSL) pivot high.
//
// Trade sizing   — risk-based (fixed % of equity per trade).
//
// ARCHITECTURE (RunAtl per bar):
//   UpdateZigZag → SyncAdvArray → UpdateMajorMinor
//   → update x0/y0/t0 → UpdatePointers
//   → AtlProcessAllTrendLines (computes alertBreak/alertReact per TL,
//     maps to _atlIsLongSignal / _atlIsShortSignal)
//   → save end-of-bar carry-forward state
// =============================================================================

using System;
using System.Collections.Generic;
using cAlgo.API;

namespace cAlgo
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.None)]
    public class Autotrendline_BSLSSL_cBot : Robot
    {
        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — ATL Zig Zag Logic
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Period", DefaultValue = 5, MinValue = 1, Group = "ATL Zig Zag Logic")]
        public int PP { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Long Signal Enables
        //  Upward triangle → Long trade entry
        //  React Up TL  = price wicked into the TL but closed above it (bounce)
        //  Break Down TL = price closed above falling resistance (bullish breakout)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("React Major External Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MjExUp { get; set; }

        [Parameter("React Major Internal Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MjInUp { get; set; }

        [Parameter("React Minor External Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MnExUp { get; set; }

        [Parameter("React Minor Internal Up TL   → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongReact_MnInUp { get; set; }

        [Parameter("Break Major External Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MjExDown { get; set; }

        [Parameter("Break Major Internal Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MjInDown { get; set; }

        [Parameter("Break Minor External Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MnExDown { get; set; }

        [Parameter("Break Minor Internal Down TL → Long", DefaultValue = true, Group = "Long Signal Enables")]
        public bool LongBreak_MnInDown { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Short Signal Enables
        //  Downward triangle → Short trade entry
        //  Break Up TL    = price closed below rising support (bearish break)
        //  React Down TL  = price wicked into falling resistance but closed below (rejection)
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Break Major External Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MjExUp { get; set; }

        [Parameter("Break Major Internal Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MjInUp { get; set; }

        [Parameter("Break Minor External Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MnExUp { get; set; }

        [Parameter("Break Minor Internal Up TL   → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortBreak_MnInUp { get; set; }

        [Parameter("React Major External Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MjExDown { get; set; }

        [Parameter("React Major Internal Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MjInDown { get; set; }

        [Parameter("React Minor External Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MnExDown { get; set; }

        [Parameter("React Minor Internal Down TL → Short", DefaultValue = true, Group = "Short Signal Enables")]
        public bool ShortReact_MnInDown { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — BSL & SSL
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Pivot Left",  DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotLeft { get; set; }

        [Parameter("Pivot Right", DefaultValue = 5, MinValue = 1, Group = "BSL & SSL")]
        public int PivotRight { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  PARAMETERS — Risk Management
        // ═════════════════════════════════════════════════════════════════════

        [Parameter("Risk Per Trade (%)",         DefaultValue = 1.0,   MinValue = 0.1, MaxValue = 100.0, Group = "Risk Management")]
        public double RiskPercent { get; set; }

        [Parameter("Risk:Reward Ratio",           DefaultValue = 2.0,   MinValue = 0.1, Step = 0.1,      Group = "Risk Management")]
        public double RrRatio { get; set; }

        [Parameter("Max Simultaneous Positions",  DefaultValue = 3,     MinValue = 1,   MaxValue = 100,   Group = "Risk Management")]
        public int MaxOpenPositions { get; set; }

        [Parameter("Min SL Distance (pips)",      DefaultValue = 3.0,   MinValue = 0.1,                  Group = "Risk Management")]
        public double MinSlPips { get; set; }

        [Parameter("Max SL Distance (pips)",      DefaultValue = 500.0, MinValue = 1.0,                  Group = "Risk Management")]
        public double MaxSlPips { get; set; }

        [Parameter("Instance Name",               DefaultValue = "ATL_BSL_SSL_cBot",                     Group = "Risk Management")]
        public string InstanceName { get; set; }

        // ═════════════════════════════════════════════════════════════════════
        //  CONSTANTS
        // ═════════════════════════════════════════════════════════════════════

        private const int MaxBslPivots = 10;

        // ═════════════════════════════════════════════════════════════════════
        //  INNER TYPES — ATL TL state (chart objects removed)
        // ═════════════════════════════════════════════════════════════════════

        private sealed class AtlTlState
        {
            public bool   PermitSet     = false;
            public bool   PermitSetPrev = false;
            public int    LastAnchorX0  = 0;
            // Coordinates of the last ACCEPTED line; used for LinePrice in Block 2 & React check.
            // Set when Block 1's validity scan passes, regardless of Show.
            public int    ActiveX0      = 0;
            public double ActiveY0      = 0.0;
            public int    ActiveX1      = 0;
            public double ActiveY1      = 0.0;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  INNER TYPES — BSL/SSL (verbatim from ICT_01_cBot_single.cs)
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

        // ═════════════════════════════════════════════════════════════════════
        //  ATL EMBEDDED STATE FIELDS
        //  Prefixed _atl* to avoid collision.  Mirrors all carry-forward fields
        //  of AutoTrendLinesTFlab.Calculate(), stripped of chart objects.
        // ═════════════════════════════════════════════════════════════════════

        // ZZ arrays — oldest [0], newest [Count-1]
        private readonly List<string> _atlZzType  = new List<string>();
        private readonly List<double> _atlZzValue = new List<double>();
        private readonly List<int>    _atlZzIndex = new List<int>();

        // ADV (advanced) arrays
        private readonly List<string> _atlAdvType  = new List<string>();
        private readonly List<double> _atlAdvValue = new List<double>();
        private readonly List<int>    _atlAdvIndex = new List<int>();

        // Major level tracking
        private double _atlMajorHighLevel         = double.NaN;
        private double _atlMajorLowLevel          = double.NaN;
        private bool   _atlMajorLevelsInitialized = false;

        // ADV seeding locks
        private bool _atlLock0 = true;
        private bool _atlLock1 = true;

        // Last confirmed pivot values (ta.valuewhen equivalents)
        private double _atlLastHighPivotValue = double.NaN;
        private int    _atlLastHighPivotIndex = -1;
        private double _atlLastLowPivotValue  = double.NaN;
        private int    _atlLastLowPivotIndex  = -1;

        // x_0 / y_0 / t_0 — last ADV entry (Pine lines 552-556)
        private int    _atlX0     = 0;
        private double _atlY0     = 0.0;
        private string _atlT0     = string.Empty;
        private string _atlT0Prev = string.Empty;   // t_0[1]

        // ZZ snapshot for SyncAdvArray change detection
        private double _atlPrevZzLastValue      = double.NaN;
        private char   _atlPrevZzLastTypeSuffix = '\0';

        // TL type name strings (ordered by index)
        private static readonly string[] AtlTlTypeNames =
            { "MLL", "MHH", "MHL", "MLH", "mLL", "mHH", "mHL", "mLH" };

        // Pointer rolling windows (8 TLs × 2 anchors each)
        private readonly int[]    _atlPtrX0 = new int[8];
        private readonly double[] _atlPtrY0 = new double[8];
        private readonly int[]    _atlPtrX1 = new int[8];
        private readonly double[] _atlPtrY1 = new double[8];

        // Per-TL state (no chart object fields)
        private readonly AtlTlState[] _atlTlStates = new AtlTlState[8];

        // Signal outputs — set by AtlProcessAllTrendLines each bar
        private bool _atlIsLongSignal;
        private bool _atlIsShortSignal;

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL EMBEDDED FIELDS (verbatim from ICT_01_cBot_single.cs)
        // ═════════════════════════════════════════════════════════════════════

        private readonly LinkedList<BslPivot> _bslPivots        = new LinkedList<BslPivot>();
        private readonly LinkedList<BslPool>  _bslBuysidePools  = new LinkedList<BslPool>();
        private readonly LinkedList<BslPool>  _bslSellsidePools = new LinkedList<BslPool>();

        private double _bslCurrentBsl = double.NaN;
        private double _bslCurrentSsl = double.NaN;

        // ═════════════════════════════════════════════════════════════════════
        //  CBOT STATE
        // ═════════════════════════════════════════════════════════════════════

        private int _lastLongSignalBar  = -1;
        private int _lastShortSignalBar = -1;
        private int _lastProcessed      = -1;

        // ═════════════════════════════════════════════════════════════════════
        //  LIFECYCLE
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnStart()
        {
            for (int i = 0; i < 8; i++)
                _atlTlStates[i] = new AtlTlState();

            Print("Autotrendline_BSLSSL_cBot started. PP={0}, PivotL={1}, PivotR={2}, " +
                  "MaxPos={3}, Risk={4}%, RR={5}",
                  PP, PivotLeft, PivotRight, MaxOpenPositions, RiskPercent, RrRatio);
        }

        protected override void OnStop()
        {
            Print("Autotrendline_BSLSSL_cBot stopped.");
        }

        // ═════════════════════════════════════════════════════════════════════
        //  ONBAR
        // ═════════════════════════════════════════════════════════════════════

        protected override void OnBar()
        {
            int signalBar = Bars.Count - 2;   // last confirmed closed bar

            // Incrementally process every bar not yet seen.
            for (int i = _lastProcessed + 1; i <= signalBar; i++)
            {
                RunBslSsl(i);   // BSL/SSL levels first (SL anchors)
                RunAtl(i);      // ATL signal detection
            }
            _lastProcessed = signalBar;

            if (signalBar < 2 * PP) return;   // need enough bars for first pivot

            bool isLong  = _atlIsLongSignal;
            bool isShort = _atlIsShortSignal;
            if (!isLong && !isShort) return;

            // ── Global capacity guard ─────────────────────────────────────────
            int openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions)
            {
                Print("Bar {0}: max positions ({1}) reached.", signalBar, MaxOpenPositions);
                return;
            }

            // ── Long ──────────────────────────────────────────────────────────
            if (isLong && _lastLongSignalBar != signalBar)
            {
                _lastLongSignalBar = signalBar;
                TryEnterLong(signalBar);
            }

            // Re-check capacity before short
            openCount = Positions.FindAll(InstanceName, SymbolName).Length;
            if (openCount >= MaxOpenPositions) return;

            // ── Short ─────────────────────────────────────────────────────────
            if (isShort && _lastShortSignalBar != signalBar)
            {
                _lastShortSignalBar = signalBar;
                TryEnterShort(signalBar);
            }
        }

        // ═════════════════════════════════════════════════════════════════════
        //  TRADE ENTRY  (mirrors ICT_01_cBot_single.cs)
        // ═════════════════════════════════════════════════════════════════════

        private void TryEnterLong(int signalBar)
        {
            double entry    = Symbol.Ask;
            double sslLevel = _bslCurrentSsl;

            if (double.IsNaN(sslLevel) || sslLevel <= 0)
            {
                Print("Bar {0}: LONG skipped – SSL unavailable.", signalBar);
                return;
            }
            if (sslLevel >= entry)
            {
                Print("Bar {0}: LONG skipped – SSL {1:F5} not below entry {2:F5}.",
                      signalBar, sslLevel, entry);
                return;
            }

            double slPips   = (entry - sslLevel) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "LONG", slPips)) return;

            double tpPips    = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume    = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0)
            { Print("Bar {0}: LONG skipped – volume is 0.", signalBar); return; }

            Print("Bar {0}: LONG  | Entry={1:F5} | SSL SL={2:F5} ({3:F1}p) | TP={4:F1}p | Vol={5}",
                  signalBar, entry, sslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Buy, SymbolName, volume, InstanceName, slPips, tpPips);
        }

        private void TryEnterShort(int signalBar)
        {
            double entry    = Symbol.Bid;
            double bslLevel = _bslCurrentBsl;

            if (double.IsNaN(bslLevel) || bslLevel <= 0)
            {
                Print("Bar {0}: SHORT skipped – BSL unavailable.", signalBar);
                return;
            }
            if (bslLevel <= entry)
            {
                Print("Bar {0}: SHORT skipped – BSL {1:F5} not above entry {2:F5}.",
                      signalBar, bslLevel, entry);
                return;
            }

            double slPips    = (bslLevel - entry) / Symbol.PipSize;
            if (!ValidateSlPips(signalBar, "SHORT", slPips)) return;

            double tpPips    = slPips * RrRatio;
            double riskAmount = Account.Equity * (RiskPercent / 100.0);
            double volume    = GetRiskVolume(riskAmount, slPips);
            if (volume <= 0)
            { Print("Bar {0}: SHORT skipped – volume is 0.", signalBar); return; }

            Print("Bar {0}: SHORT | Entry={1:F5} | BSL SL={2:F5} ({3:F1}p) | TP={4:F1}p | Vol={5}",
                  signalBar, entry, bslLevel, slPips, tpPips, volume);
            ExecuteMarketOrder(TradeType.Sell, SymbolName, volume, InstanceName, slPips, tpPips);
        }

        private bool ValidateSlPips(int signalBar, string direction, double slPips)
        {
            if (slPips < MinSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1}p < min {3:F1}p.",
                      signalBar, direction, slPips, MinSlPips);
                return false;
            }
            if (slPips > MaxSlPips)
            {
                Print("Bar {0}: {1} skipped – SL {2:F1}p > max {3:F1}p.",
                      signalBar, direction, slPips, MaxSlPips);
                return false;
            }
            return true;
        }

        private double GetRiskVolume(double riskAmount, double slPips)
        {
            if (slPips <= 0) return 0;
            double raw    = Symbol.VolumeForFixedRisk(riskAmount, slPips);
            double volume = Symbol.NormalizeVolumeInUnits(raw, RoundingMode.Down);
            if (volume < Symbol.VolumeInUnitsMin) return 0;
            if (volume > Symbol.VolumeInUnitsMax) volume = Symbol.VolumeInUnitsMax;
            return volume;
        }

        // ═════════════════════════════════════════════════════════════════════
        //  ATL EMBEDDED ENGINE
        //  Ported from AutoTrendLinesTFlab (AutoTrendLines indicator).
        //  All chart drawing (DrawTrendLine, DrawIcon, RemoveObject) removed.
        //  All alert emission (Print, frequency logic) removed.
        //  Carry-forward fields (_atl*) replace the indicator's instance fields.
        //  Called once per closed bar from the OnBar incremental loop.
        // ═════════════════════════════════════════════════════════════════════

        private void RunAtl(int index)
        {
            // 1. ZZ state machine + ADV seeding
            AtlUpdateZigZag(index);

            // 2. Sync ADV minor entries (Pine lines 354-364)
            AtlSyncAdvArray();

            // 3. Major/minor promotion (Pine lines 366-492)
            AtlUpdateMajorMinor(index);

            // 4. Update x0/y0/t0 from last ADV entry (Pine lines 552-556)
            if (_atlAdvType.Count > 2)
            {
                int last  = _atlAdvType.Count - 1;
                _atlX0    = _atlAdvIndex[last];
                _atlY0    = _atlAdvValue[last];
                _atlT0    = _atlAdvType[last];
            }

            // 5. Update Pointer rolling windows (Pine lines 495-512)
            AtlUpdatePointers();

            // 6. Process all 8 TLs → accumulate into _atlIsLongSignal / _atlIsShortSignal
            AtlProcessAllTrendLines(index);

            // 7. Save end-of-bar state for next call's [1] comparisons
            _atlT0Prev = _atlT0;
            if (_atlZzType.Count > 0)
            {
                int n = _atlZzType.Count - 1;
                _atlPrevZzLastValue      = _atlZzValue[n];
                _atlPrevZzLastTypeSuffix = _atlZzType[n][_atlZzType[n].Length - 1];
            }
        }

        // ── Pivot detection ────────────────────────────────────────────────────
        // Mirrors DetectPivotHigh/Low in AutoTrendLines (strict rightmost rule).

        private bool AtlDetectPivotHigh(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;

            int    pivotBar    = index - PP;
            int    windowStart = index - 2 * PP;
            double candidate   = Bars.HighPrices[pivotBar];

            double max = double.MinValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            if (candidate != max) return false;

            int lastMaxBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.HighPrices[i] == max) lastMaxBar = i;
            if (lastMaxBar != pivotBar) return false;

            pivotValue = candidate;
            return true;
        }

        private bool AtlDetectPivotLow(int index, out double pivotValue)
        {
            pivotValue = double.NaN;
            if (index < 2 * PP) return false;

            int    pivotBar    = index - PP;
            int    windowStart = index - 2 * PP;
            double candidate   = Bars.LowPrices[pivotBar];

            double min = double.MaxValue;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            if (candidate != min) return false;

            int lastMinBar = windowStart;
            for (int i = windowStart; i <= index; i++)
                if (Bars.LowPrices[i] == min) lastMinBar = i;
            if (lastMinBar != pivotBar) return false;

            pivotValue = candidate;
            return true;
        }

        // ── ZZ state machine (Pine ZZ() lines 114-352) ────────────────────────
        // Verbatim from AutoTrendLines.UpdateZigZag — no chart objects.

        private void AtlUpdateZigZag(int index)
        {
            bool hasHigh = AtlDetectPivotHigh(index, out double highValue);
            bool hasLow  = AtlDetectPivotLow(index,  out double lowValue);
            if (!hasHigh && !hasLow) return;

            int    pivotBar = index - PP;
            double barClose = Bars.ClosePrices[index];

            if (hasHigh) { _atlLastHighPivotValue = highValue; _atlLastHighPivotIndex = pivotBar; }
            if (hasLow)  { _atlLastLowPivotValue  = lowValue;  _atlLastLowPivotIndex  = pivotBar; }

            string LabelHigh(double v)
            {
                int n = _atlZzType.Count;
                return n > 2 ? (_atlZzValue[n - 2] < v ? "HH" : "LH") : "H";
            }
            string LabelLow(double v)
            {
                int n = _atlZzType.Count;
                return n > 2 ? (_atlZzValue[n - 2] < v ? "HL" : "LL") : "L";
            }

            void RemoveLast()
            {
                int n = _atlZzType.Count - 1;
                _atlZzType.RemoveAt(n); _atlZzValue.RemoveAt(n); _atlZzIndex.RemoveAt(n);
            }
            void PushHigh(double v, int bar)
            {
                _atlZzType.Add(LabelHigh(v)); _atlZzValue.Add(v); _atlZzIndex.Add(bar);
            }
            void PushLow(double v, int bar)
            {
                _atlZzType.Add(LabelLow(v)); _atlZzValue.Add(v); _atlZzIndex.Add(bar);
            }

            int cnt = _atlZzType.Count;

            // CASE A: Both high AND low confirm simultaneously
            if (hasHigh && hasLow)
            {
                if (cnt == 0)
                {
                    _atlZzType.Add("H"); _atlZzValue.Add(highValue); _atlZzIndex.Add(pivotBar);
                }
                else
                {
                    string last    = _atlZzType[cnt - 1];
                    double lastVal = _atlZzValue[cnt - 1];

                    if (last == "L" || last == "LL")
                    {
                        if (lowValue < lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); }
                        else                    { PushHigh(highValue, pivotBar); }
                    }
                    else if (last == "H" || last == "HH")
                    {
                        if (highValue > lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); }
                        else                     { PushLow(lowValue, pivotBar); }
                    }
                    else if (last == "LH")
                    {
                        if (highValue < lastVal)
                        {
                            PushLow(lowValue, pivotBar);
                        }
                        else if (highValue > lastVal)
                        {
                            if      (barClose < lastVal) { RemoveLast(); PushHigh(highValue, pivotBar); }
                            else if (barClose > lastVal) { PushLow(lowValue, pivotBar); }
                        }
                    }
                    else if (last == "HL")
                    {
                        if (lowValue > lastVal)
                        {
                            PushHigh(highValue, pivotBar);
                        }
                        else if (lowValue < lastVal)
                        {
                            if      (barClose > lastVal) { RemoveLast(); PushLow(lowValue, pivotBar); }
                            else if (barClose < lastVal) { PushHigh(highValue, pivotBar); }
                        }
                    }
                }
            }
            // CASE B: Only High pivot
            else if (hasHigh)
            {
                cnt = _atlZzType.Count;
                if (cnt == 0)
                {
                    _atlZzType.Insert(0, "H"); _atlZzValue.Insert(0, highValue); _atlZzIndex.Insert(0, pivotBar);
                }
                else
                {
                    string last    = _atlZzType[cnt - 1];
                    double lastVal = _atlZzValue[cnt - 1];

                    if (last == "L" || last == "HL" || last == "LL")
                    {
                        if (highValue > lastVal)
                            PushHigh(highValue, pivotBar);
                        else if (highValue < lastVal)
                        {
                            RemoveLast();
                            if (!double.IsNaN(_atlLastLowPivotValue) && _atlLastLowPivotIndex >= 0)
                                PushLow(_atlLastLowPivotValue, _atlLastLowPivotIndex);
                        }
                    }
                    else if (last == "H" || last == "HH" || last == "LH")
                    {
                        if (lastVal < highValue) { RemoveLast(); PushHigh(highValue, pivotBar); }
                    }
                }
            }
            // CASE C: Only Low pivot
            else
            {
                cnt = _atlZzType.Count;
                if (cnt == 0)
                {
                    _atlZzType.Insert(0, "L"); _atlZzValue.Insert(0, lowValue); _atlZzIndex.Insert(0, pivotBar);
                }
                else
                {
                    string last    = _atlZzType[cnt - 1];
                    double lastVal = _atlZzValue[cnt - 1];

                    if (last == "H" || last == "HH" || last == "LH")
                    {
                        if (lowValue < lastVal)
                            PushLow(lowValue, pivotBar);
                        else if (lowValue > lastVal)
                        {
                            RemoveLast();
                            if (!double.IsNaN(_atlLastHighPivotValue) && _atlLastHighPivotIndex >= 0)
                                PushHigh(_atlLastHighPivotValue, _atlLastHighPivotIndex);
                        }
                    }
                    else if (last == "L" || last == "HL" || last == "LL")
                    {
                        if (lastVal > lowValue) { RemoveLast(); PushLow(lowValue, pivotBar); }
                    }
                }
            }

            // Major levels init: fires ONCE when ZZ count first == 2 (Pine 316-332)
            if (!_atlMajorLevelsInitialized && _atlZzType.Count == 2)
            {
                if (_atlZzType[0] == "H")
                { _atlMajorHighLevel = _atlZzValue[0]; _atlMajorLowLevel = _atlZzValue[1]; }
                else
                { _atlMajorHighLevel = _atlZzValue[1]; _atlMajorLowLevel = _atlZzValue[0]; }
                _atlMajorLevelsInitialized = true;
            }

            // ADV seeding: Lock0 fires when ZZ count first >= 1 (Pine 338-344)
            if (_atlLock0 && _atlZzType.Count >= 1)
            {
                _atlAdvType.Insert(0, "M" + _atlZzType[0]);
                _atlAdvValue.Insert(0, _atlZzValue[0]);
                _atlAdvIndex.Insert(0, _atlZzIndex[0]);
                _atlLock0 = false;
            }

            // ADV seeding: Lock1 fires when ZZ count first >= 2 (Pine 346-352)
            if (_atlLock1 && _atlZzType.Count >= 2)
            {
                _atlAdvType.Insert(1, "M" + _atlZzType[1]);
                _atlAdvValue.Insert(1, _atlZzValue[1]);
                _atlAdvIndex.Insert(1, _atlZzIndex[1]);
                _atlLock1 = false;
                // NOTE: Pine does NOT suppress SyncAdvArray here.
            }
        }

        // ── ADV Sync (Pine lines 354-364) ─────────────────────────────────────

        private void AtlSyncAdvArray()
        {
            if (_atlZzType.Count <= 1 || _atlAdvType.Count == 0) return;

            int    zzLast        = _atlZzType.Count - 1;
            double currentZzVal  = _atlZzValue[zzLast];
            string currentZzType = _atlZzType[zzLast];
            char   currentSuffix = currentZzType[currentZzType.Length - 1];

            if (double.IsNaN(_atlPrevZzLastValue) || currentZzVal == _atlPrevZzLastValue) return;

            if (currentSuffix != _atlPrevZzLastTypeSuffix)
            {
                // New type suffix → push fresh minor entry
                _atlAdvType.Add("m" + currentZzType);
                _atlAdvValue.Add(currentZzVal);
                _atlAdvIndex.Add(_atlZzIndex[zzLast]);
            }
            else
            {
                // Same type suffix, value updated → update last ADV value/index
                int advLast = _atlAdvType.Count - 1;
                _atlAdvValue[advLast] = currentZzVal;
                _atlAdvIndex[advLast] = _atlZzIndex[zzLast];
            }
        }

        // ── Major/Minor promotion (Pine lines 366-492) ────────────────────────

        private void AtlUpdateMajorMinor(int index)
        {
            if (!_atlMajorLevelsInitialized || _atlAdvType.Count <= 1) return;

            double cls = Bars.ClosePrices[index];

            string ZzType(int offset = 0)
            {
                int n = _atlZzType.Count - 1 - offset;
                return n >= 0 ? _atlZzType[n] : string.Empty;
            }

            // ---- A) close > MajorHighLevel ----
            if (cls > _atlMajorHighLevel)
            {
                int    last = _atlAdvType.Count - 1;
                string t    = _atlAdvType[last];

                if (t == "mL")
                {
                    _atlAdvType[last]   = "ML";
                    _atlMajorLowLevel   = _atlAdvValue[last];
                }
                else if (t == "mHL" || t == "mLL")
                {
                    string p = "M" + ZzType();
                    if (p.Length > 1) _atlAdvType[last] = p;
                    _atlMajorLowLevel = _atlAdvValue[last];
                }
                else if (t == "mLH" || t == "mHH" || t == "MLH" || t == "MHH")
                {
                    if (last >= 1)
                    {
                        string t2 = _atlAdvType[last - 1];
                        if (t2 == "mHL" || t2 == "mLL")
                        {
                            string p = "M" + ZzType(1);
                            if (p.Length > 1) _atlAdvType[last - 1] = p;
                            _atlMajorLowLevel = _atlAdvValue[last - 1];
                        }
                    }
                }
            }

            // ---- B) lastAdvVal > MajorHighLevel ----
            // Exactly 3 branches as in Pine: mH | mLH | (mHH | MHH)
            {
                int    last = _atlAdvType.Count - 1;
                string t    = _atlAdvType[last];

                if (_atlAdvValue[last] > _atlMajorHighLevel)
                {
                    if (t == "mH")
                    {
                        _atlAdvType[last]   = "MH";
                        _atlMajorHighLevel  = _atlAdvValue[last];
                    }
                    else if (t == "mLH")
                    {
                        string p = "M" + ZzType();
                        if (p.Length > 1) _atlAdvType[last] = p;
                        _atlMajorHighLevel = _atlAdvValue[last];
                    }
                    else if (t == "mHH" || t == "MHH")
                    {
                        string p = "M" + ZzType();
                        if (p.Length > 1) _atlAdvType[last] = p;
                        _atlMajorHighLevel = _atlAdvValue[last];
                    }
                }
            }

            // ---- C) close < MajorLowLevel ----
            if (cls < _atlMajorLowLevel)
            {
                int    last = _atlAdvType.Count - 1;
                string t    = _atlAdvType[last];

                if (t == "mH")
                {
                    _atlAdvType[last]   = "MH";
                    _atlMajorHighLevel  = _atlAdvValue[last];
                }
                else if (t == "mLH" || t == "mHH")
                {
                    string p = "M" + ZzType();
                    if (p.Length > 1) _atlAdvType[last] = p;
                    _atlMajorHighLevel = _atlAdvValue[last];
                }
                else if (t == "mHL" || t == "mLL" || t == "MHL" || t == "MLL")
                {
                    if (last >= 1)
                    {
                        string t2 = _atlAdvType[last - 1];
                        if (t2 == "mLH" || t2 == "mHH")
                        {
                            string p = "M" + ZzType(1);
                            if (p.Length > 1) _atlAdvType[last - 1] = p;
                            _atlMajorHighLevel = _atlAdvValue[last - 1];
                        }
                    }
                }
            }

            // ---- D) lastAdvVal < MajorLowLevel ----
            {
                int    last = _atlAdvType.Count - 1;
                string t    = _atlAdvType[last];

                if (_atlAdvValue[last] < _atlMajorLowLevel)
                {
                    if (t == "mL")
                    {
                        _atlAdvType[last]   = "ML";
                        _atlMajorLowLevel   = _atlAdvValue[last];
                    }
                    else if (t == "mHL" || t == "mLL" || t == "MLL")
                    {
                        string p = "M" + ZzType();
                        if (p.Length > 1) _atlAdvType[last] = p;
                        _atlMajorLowLevel = _atlAdvValue[last];
                    }
                }
            }
        }

        // ── Pointer update (Pine lines 495-512) ───────────────────────────────

        private void AtlUpdatePointers()
        {
            if (_atlT0 == _atlT0Prev) return;

            for (int i = 0; i < 8; i++)
            {
                if (_atlT0 != AtlTlTypeNames[i]) continue;

                if (_atlPtrX0[i] == 0)
                {
                    _atlPtrX0[i] = _atlX0; _atlPtrY0[i] = _atlY0;
                }
                else if (_atlPtrX1[i] == 0)
                {
                    _atlPtrX1[i] = _atlX0; _atlPtrY1[i] = _atlY0;
                }
                else
                {
                    _atlPtrX0[i] = _atlPtrX1[i]; _atlPtrY0[i] = _atlPtrY1[i];
                    _atlPtrX1[i] = _atlX0;       _atlPtrY1[i] = _atlY0;
                }
            }
        }

        // ── Signal dispatch for all 8 TLs ────────────────────────────────────

        private void AtlProcessAllTrendLines(int index)
        {
            // Reset each bar before accumulating across TLs
            _atlIsLongSignal  = false;
            _atlIsShortSignal = false;

            // TL 0: MjExUp   isUp=true   Break→Short  React→Long
            AtlProcessTrendLine(index, 0, isUp: true,
                enableBreakShort: ShortBreak_MjExUp,
                enableReactLong:  LongReact_MjExUp);

            // TL 1: MjExDown isUp=false  Break→Long   React→Short
            AtlProcessTrendLine(index, 1, isUp: false,
                enableBreakShort: ShortReact_MjExDown,
                enableReactLong:  LongBreak_MjExDown);

            // TL 2: MjInUp   isUp=true   Break→Short  React→Long
            AtlProcessTrendLine(index, 2, isUp: true,
                enableBreakShort: ShortBreak_MjInUp,
                enableReactLong:  LongReact_MjInUp);

            // TL 3: MjInDown isUp=false  Break→Long   React→Short
            AtlProcessTrendLine(index, 3, isUp: false,
                enableBreakShort: ShortReact_MjInDown,
                enableReactLong:  LongBreak_MjInDown);

            // TL 4: MnExUp   isUp=true   Break→Short  React→Long
            AtlProcessTrendLine(index, 4, isUp: true,
                enableBreakShort: ShortBreak_MnExUp,
                enableReactLong:  LongReact_MnExUp);

            // TL 5: MnExDown isUp=false  Break→Long   React→Short
            AtlProcessTrendLine(index, 5, isUp: false,
                enableBreakShort: ShortReact_MnExDown,
                enableReactLong:  LongBreak_MnExDown);

            // TL 6: MnInUp   isUp=true   Break→Short  React→Long
            AtlProcessTrendLine(index, 6, isUp: true,
                enableBreakShort: ShortBreak_MnInUp,
                enableReactLong:  LongReact_MnInUp);

            // TL 7: MnInDown isUp=false  Break→Long   React→Short
            AtlProcessTrendLine(index, 7, isUp: false,
                enableBreakShort: ShortReact_MnInDown,
                enableReactLong:  LongBreak_MnInDown);
        }

        // ── Per-TL signal computation (Correction_Checker stripped of drawing) ─
        //
        // enableBreakShort : if true, a Break event on this TL fires a Short signal
        // enableReactLong  : if true, a React event on this TL fires a Long  signal
        //
        // For isUp=true  TLs: Break fires Short, React fires Long.
        // For isUp=false TLs: Break fires Long,  React fires Short.
        // The caller already inverts the parameter mapping for isUp=false TLs
        // (enableBreakShort receives the "Short React" toggle, enableReactLong
        //  receives the "Long Break" toggle) so the internal logic here is uniform.

        private void AtlProcessTrendLine(int index, int tlIdx, bool isUp,
            bool enableBreakShort, bool enableReactLong)
        {
            AtlTlState state = _atlTlStates[tlIdx];

            int    x0 = _atlPtrX0[tlIdx];
            double y0 = _atlPtrY0[tlIdx];
            int    x1 = _atlPtrX1[tlIdx];
            double y1 = _atlPtrY1[tlIdx];

            // Snapshot PermitSet before this bar's logic (= Permit_set[1] in Pine)
            state.PermitSetPrev = state.PermitSet;

            // ---- BLOCK 1: Anchor-change event (Pine lines 514-540) ----
            if (x0 != 0 && x1 != 0 && x0 != state.LastAnchorX0)
            {
                state.LastAnchorX0 = x0;

                bool correctSlope = isUp ? (y1 > y0) : (y1 < y0);
                bool permit       = false;

                if (correctSlope)
                {
                    // Scan all closes from x0+1 to current bar (Pine: for i=1 to bar_index-X_0)
                    permit = true;
                    for (int barI = x0 + 1; barI <= index; barI++)
                    {
                        double lp  = AtlLinePrice(x0, y0, x1, y1, barI);
                        double cls = Bars.ClosePrices[barI];
                        if (isUp ? cls <= lp : cls >= lp) { permit = false; break; }
                    }
                }

                if (permit)
                {
                    // Store accepted coordinates (used by Block 2 and React detection)
                    state.ActiveX0  = x0;  state.ActiveY0 = y0;
                    state.ActiveX1  = x1;  state.ActiveY1 = y1;
                    state.PermitSet = true;
                    // No chart drawing in cBot.
                }
                // When scan fails, PermitSet is left unchanged; Block 2 handles the old state.
            }

            // ---- BLOCK 2: Per-bar validity (Pine lines 540-548) ----
            if (state.PermitSet)
            {
                if (state.ActiveX0 == 0)
                {
                    state.PermitSet = false;
                }
                else
                {
                    double lp    = AtlLinePrice(state.ActiveX0, state.ActiveY0,
                                               state.ActiveX1, state.ActiveY1, index);
                    double cls   = Bars.ClosePrices[index];
                    bool   onSide = isUp ? cls > lp : cls < lp;
                    if (!onSide) state.PermitSet = false;
                    // No line freeze in cBot (no chart object).
                }
            }

            // ---- Signal detection (Pine lines 578-625) ----
            // Break: PermitSet was true last bar, now false → price crossed the line.
            bool alertBreak = state.PermitSetPrev && !state.PermitSet;

            // React: PermitSet still true, close on correct side, but wick touched the line.
            bool alertReact = false;
            if (state.PermitSet && state.ActiveX0 != 0)
            {
                double lp   = AtlLinePrice(state.ActiveX0, state.ActiveY0,
                                           state.ActiveX1, state.ActiveY1, index);
                double cls  = Bars.ClosePrices[index];
                double high = Bars.HighPrices[index];
                double low  = Bars.LowPrices[index];
                alertReact  = isUp
                    ? (cls > lp && low  < lp)   // wick below rising support, close above
                    : (cls < lp && high > lp);   // wick above falling resistance, close below
            }

            // ---- Accumulate into Long/Short output signals ----
            //
            // isUp=true  TL: Break → downward triangle → Short
            //                React → upward   triangle → Long
            // isUp=false TL: Break → upward   triangle → Long
            //                React → downward triangle → Short
            // (For isUp=false TLs the caller has already swapped enableBreakShort/enableReactLong
            //  so the logic below is uniformly isUp-agnostic.)
            if (isUp)
            {
                if (alertBreak && enableBreakShort) _atlIsShortSignal = true;
                if (alertReact && enableReactLong)  _atlIsLongSignal  = true;
            }
            else
            {
                // enableReactLong  here carries "LongBreak_*"  toggle (Break→Long for Down TL)
                // enableBreakShort here carries "ShortReact_*" toggle (React→Short for Down TL)
                if (alertBreak && enableReactLong)  _atlIsLongSignal  = true;
                if (alertReact && enableBreakShort) _atlIsShortSignal = true;
            }
        }

        // ── Line price — linear extrapolation through (x0,y0),(x1,y1) ─────────
        // Mirrors Pine line.get_price().

        private static double AtlLinePrice(int x0, double y0, int x1, double y1, int atBar)
        {
            if (x1 == x0) return y0;
            return y0 + (y1 - y0) * (double)(atBar - x0) / (x1 - x0);
        }

        // ═════════════════════════════════════════════════════════════════════
        //  BSL/SSL EMBEDDED ENGINE
        //  Verbatim from ICT_01_cBot_single.cs (which mirrors BSL and SSL.cs).
        // ═════════════════════════════════════════════════════════════════════

        private void RunBslSsl(int index)
        {
            BslDetectAndStoreConfirmedPivots(index);
            BslAddPoolFromNewPivot(index);
            BslClearMitigated(index);

            // Most-recent unmitigated level (mirrors BSL_SSL.UpdateOutputLevels)
            _bslCurrentBsl = _bslBuysidePools.First  != null
                ? _bslBuysidePools.First.Value.Price  : double.NaN;
            _bslCurrentSsl = _bslSellsidePools.First != null
                ? _bslSellsidePools.First.Value.Price : double.NaN;
        }

        private void BslDetectAndStoreConfirmedPivots(int currentIndex)
        {
            int pivotIndex = currentIndex - PivotRight;
            if (pivotIndex <= 0) return;

            int leftStart = pivotIndex - PivotLeft;
            int rightEnd  = pivotIndex + PivotRight;   // equals currentIndex

            if (leftStart < 0 || rightEnd >= Bars.Count) return;

            double candidateHigh = Bars.HighPrices[pivotIndex];
            double candidateLow  = Bars.LowPrices[pivotIndex];

            if (BslIsPivotHigh(candidateHigh, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateHigh, BarIndex = pivotIndex, Type =  1 });

            if (BslIsPivotLow(candidateLow, leftStart, rightEnd))
                BslUnshiftPivot(new BslPivot { Price = candidateLow,  BarIndex = pivotIndex, Type = -1 });
        }

        // Non-strict (ties allowed) — mirrors ta.pivothigh/low in BSL_SSL
        private bool BslIsPivotHigh(double candidate, int start, int end)
        {
            double max = double.MinValue;
            for (int i = start; i <= end; i++)
                if (Bars.HighPrices[i] > max) max = Bars.HighPrices[i];
            return candidate == max;
        }

        private bool BslIsPivotLow(double candidate, int start, int end)
        {
            double min = double.MaxValue;
            for (int i = start; i <= end; i++)
                if (Bars.LowPrices[i] < min) min = Bars.LowPrices[i];
            return candidate == min;
        }

        // Prepend pivot; skip exact duplicates; cap list at MaxBslPivots
        private void BslUnshiftPivot(BslPivot p)
        {
            if (_bslPivots.First != null)
            {
                var f = _bslPivots.First.Value;
                if (f.BarIndex == p.BarIndex && f.Type == p.Type &&
                    Math.Abs(f.Price - p.Price) < Symbol.PipSize * 0.1)
                    return;
            }
            _bslPivots.AddFirst(p);
            while (_bslPivots.Count > MaxBslPivots)
                _bslPivots.RemoveLast();
        }

        // Mirrors AddExternalLiquidityFromNewPivot (pool management only)
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

        // Mirrors ClearMitigated in BSL_SSL:
        //   Sellside (SSL): mitigated when Low  ≤ its price
        //   Buyside  (BSL): mitigated when High ≥ its price
        private void BslClearMitigated(int index)
        {
            var node = _bslSellsidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.LowPrices[index] <= node.Value.Price)
                    _bslSellsidePools.Remove(node);
                node = next;
            }

            node = _bslBuysidePools.First;
            while (node != null)
            {
                var next = node.Next;
                if (Bars.HighPrices[index] >= node.Value.Price)
                    _bslBuysidePools.Remove(node);
                node = next;
            }
        }
    }
}
